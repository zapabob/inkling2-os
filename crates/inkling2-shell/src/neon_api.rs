//! Neon Phase 7 HTTP API Handlers for Hermes Agent & Companion Integration (`neon_api`).
//!
//! Exposes typed, Permission-Broker-gated endpoints for perception, voice, Neural Memory45,
//! autonomous tasks, and verified learning without breaking existing `/v1/chat/completions`.

use std::path::Path;
use std::sync::Mutex;

use serde_json::{Value, json};

use inkling2_agent::audit::AgentAuditLog;
use inkling2_agent::executor::{ExecutionLimits, run_task_plan};
use inkling2_agent::permissions::{ActionRequest, ActionRisk, PermissionBroker, PolicyDecision};
use inkling2_agent::planner::build_plan;
use inkling2_agent::reinforcement::{LearningLevel, RewardSignals, evaluate_reinforcement};
use inkling2_agent::scheduler::TaskScheduler;
use inkling2_agent::skill_registry::SkillRegistry;
use inkling2_agent::verifier::VerificationEvidence;
use inkling2_companion::camera_call::CameraConfig;
use inkling2_companion::dynamic_lora::DynamicLoraAdapter;
use inkling2_companion::neural_memory45::{NeuralMemory45, RecordVerification};
use inkling2_companion::voice_call::{
    SpeechController, VoicePersona, check_voicevox_health, speak_queued_sentences,
};
use inkling2_perception::camera::{camera_status, capture_once};
use inkling2_perception::event_bus::PerceptionEventBus;
use inkling2_perception::privacy::PrivacyGate;

/// Shared Neon runtime state attached to `BridgeContext`.
pub struct NeonRuntime {
    pub privacy: PrivacyGate,
    pub perception_bus: Mutex<PerceptionEventBus>,
    pub speech_controller: SpeechController,
    pub neural_memory: Mutex<NeuralMemory45>,
    pub broker: Mutex<PermissionBroker>,
    pub scheduler: Mutex<TaskScheduler>,
    pub audit: Mutex<AgentAuditLog>,
    pub skills: SkillRegistry,
}

impl Default for NeonRuntime {
    fn default() -> Self {
        Self::new()
    }
}

impl NeonRuntime {
    pub fn new() -> Self {
        let privacy = PrivacyGate::default();
        privacy.set_camera_opt_in(std::env::var("INKLING2_CAMERA_ENABLED").is_ok_and(|v| v == "1"));
        Self {
            privacy,
            perception_bus: Mutex::new(PerceptionEventBus::default()),
            speech_controller: SpeechController::default(),
            neural_memory: Mutex::new(NeuralMemory45::default()),
            broker: Mutex::new(PermissionBroker::default()),
            scheduler: Mutex::new(TaskScheduler::default()),
            audit: Mutex::new(AgentAuditLog::default()),
            skills: SkillRegistry::with_defaults(),
        }
    }

    pub fn load(state_dir: &Path) -> Result<Self, String> {
        let mut runtime = Self::new();
        let path = state_dir.join("neural-memory.ikmem2");
        if path
            .try_exists()
            .map_err(|e| format!("inspect neural memory: {e}"))?
        {
            runtime.neural_memory = Mutex::new(NeuralMemory45::load_ikmem2(&path)?);
        }
        Ok(runtime)
    }
}

/// Dispatch Neon Phase 7 endpoints. Returns `Some((http_status, json_body))` if handled.
pub fn handle_neon_endpoint(
    method: &str,
    path: &str,
    body: &Value,
    neon: &NeonRuntime,
    voice: &VoicePersona,
    lora: &Mutex<DynamicLoraAdapter>,
    state_dir: &Path,
) -> Option<(&'static str, String)> {
    match (method, path) {
        ("GET", "/v1/perception/status") => {
            let st = camera_status(&neon.privacy);
            let devices: Vec<Value> = st
                .devices
                .iter()
                .map(|d| json!({"id": d.id, "name": d.name, "backend": d.backend}))
                .collect();
            let recent_count = neon.perception_bus.lock().map(|b| b.len()).unwrap_or(0);
            Some((
                "200 OK",
                json!({
                    "active": st.active,
                    "operator_enabled": st.operator_enabled,
                    "stopped": st.stopped,
                    "raw_frame_retention": st.raw_frame_retention,
                    "native_sc117_vision_supported": st.native_sc117_vision_supported,
                    "visual_pipeline_mode": st.visual_pipeline_mode,
                    "devices": devices,
                    "buffered_events": recent_count
                })
                .to_string(),
            ))
        }
        ("POST", "/v1/perception/capture") => {
            if !neon.privacy.is_camera_allowed() {
                return Some(("403 Forbidden", json!({"error":"camera requires operator opt-in through INKLING2_CAMERA_ENABLED=1"}).to_string()));
            }
            let task_id = body["task_id"]
                .as_str()
                .unwrap_or("perception-capture")
                .to_string();
            let allowed = neon
                .broker
                .lock()
                .map(|mut b| {
                    b.evaluate(
                        &ActionRequest {
                            task_id,
                            action_name: "perception.capture".to_string(),
                            risk: ActionRisk::ReversibleLocal,
                            is_raw_shell_command: false,
                        },
                        None,
                    )
                })
                .unwrap_or(PolicyDecision::Denied {
                    reason: "permission broker lock poisoned",
                });
            if allowed != PolicyDecision::Allowed {
                return Some((
                    "403 Forbidden",
                    json!({"error": format!("{allowed:?}")}).to_string(),
                ));
            }
            let cfg = CameraConfig {
                device_name: body["device"].as_str().map(str::to_owned),
                ..CameraConfig::default()
            };
            let out_jpg = state_dir.join("perception_capture.jpg");
            let Ok(mut bus) = neon.perception_bus.lock() else {
                return Some((
                    "500 Internal Server Error",
                    r#"{"error":"perception bus lock poisoned"}"#.to_string(),
                ));
            };
            match capture_once(&cfg, &out_jpg, &neon.privacy, &mut bus) {
                Ok(ev) => Some((
                    "200 OK",
                    json!({
                        "event_id": ev.event_id,
                        "timestamp_sec": ev.timestamp_sec,
                        "modality": ev.modality.as_str(),
                        "observation": ev.observation,
                        "untrusted_content": ev.untrusted_content,
                        "provenance": ev.provenance
                    })
                    .to_string(),
                )),
                Err(e) => Some(("503 Service Unavailable", json!({"error": e}).to_string())),
            }
        }
        ("GET", "/v1/voice/status") => {
            let vv = check_voicevox_health(&voice.voicevox_url);
            Some((
                "200 OK",
                json!({
                    "voicevox_online": vv.is_ok(),
                    "voicevox_version": vv.ok(),
                    "speaker_id": voice.voicevox_speaker_id,
                    "speed_scale": voice.speed_scale,
                    "stopped": neon.speech_controller.is_stopped(),
                    "true_token_streaming_tts": false
                })
                .to_string(),
            ))
        }
        ("POST", "/v1/voice/speak") => {
            let text = body["text"].as_str().unwrap_or_default().trim();
            if text.is_empty() || text.chars().count() > 800 {
                return Some((
                    "400 Bad Request",
                    r#"{"error":"enter between 1 and 800 characters"}"#.to_string(),
                ));
            }
            neon.speech_controller.reset();
            match speak_queued_sentences(voice, text, &neon.speech_controller) {
                Ok(spoken) => Some((
                    "200 OK",
                    json!({"status": "ok", "sentences_spoken": spoken}).to_string(),
                )),
                Err(e) => Some(("503 Service Unavailable", json!({"error": e}).to_string())),
            }
        }
        ("POST", "/v1/voice/stop") => {
            neon.speech_controller.request_stop();
            Some(("200 OK", r#"{"status":"stopped"}"#.to_string()))
        }
        ("POST", "/v1/memory/search") => {
            let query = body["query"].as_str().unwrap_or_default();
            let top_k = body["top_k"]
                .as_u64()
                .map(|v| (v as usize).clamp(1, 20))
                .unwrap_or(5);
            let Ok(mem) = neon.neural_memory.lock() else {
                return Some((
                    "500 Internal Server Error",
                    r#"{"error":"neural memory lock poisoned"}"#.to_string(),
                ));
            };
            let hits: Vec<Value> = mem
                .recall(query, top_k)
                .into_iter()
                .map(|h| {
                    json!({
                        "source_record_id": h.source_record_id,
                        "text": h.text,
                        "confidence": h.confidence,
                        "timestamp_sec": h.timestamp_sec,
                        "provenance": h.provenance,
                        "verification_status": h.verification_status,
                        "associative_surprise": h.associative_surprise
                    })
                })
                .collect();
            Some(("200 OK", json!({"hits": hits}).to_string()))
        }
        ("POST", "/v1/memory/feedback") => {
            let text = body["text"].as_str().unwrap_or_default().trim();
            if text.is_empty() || text.len() > 4096 {
                return Some((
                    "400 Bad Request",
                    r#"{"error":"text must be 1..=4096 bytes"}"#.to_string(),
                ));
            }
            let importance = body["importance"].as_f64().unwrap_or(0.8) as f32;
            let Ok(mut mem) = neon.neural_memory.lock() else {
                return Some((
                    "500 Internal Server Error",
                    r#"{"error":"neural memory lock poisoned"}"#.to_string(),
                ));
            };
            // API clients provide observations, not independent verification or
            // operator approval. Those claims cannot authorise fast-weight updates.
            let mut candidate = mem.clone();
            let (id, surprise) = match candidate.ingest(
                text,
                importance,
                body["provenance"].as_str().unwrap_or("hermes-feedback"),
                RecordVerification::UnverifiedObservation,
                false,
                false,
                false,
            ) {
                Ok(record) => record,
                Err(error) => return Some(("400 Bad Request", json!({"error":error}).to_string())),
            };
            if let Err(error) = candidate.save_ikmem2(&state_dir.join("neural-memory.ikmem2")) {
                return Some((
                    "500 Internal Server Error",
                    json!({"error":error}).to_string(),
                ));
            }
            *mem = candidate;
            let st = mem.stats();
            Some((
                "200 OK",
                json!({
                    "record_id": id,
                    "associative_surprise": surprise,
                    "active_generation": st.active_generation,
                    "total_fast_weight_updates": st.total_fast_weight_updates,
                    "budget_bytes": st.total_budget_bytes,
                    "verification_status":"unverified_observation",
                    "persisted":true
                })
                .to_string(),
            ))
        }
        ("GET", "/v1/agent/status") => {
            let Ok(sched) = neon.scheduler.lock() else {
                return Some((
                    "500 Internal Server Error",
                    r#"{"error":"scheduler lock poisoned"}"#.to_string(),
                ));
            };
            let tasks: Vec<Value> = sched
                .list()
                .into_iter()
                .map(|t| {
                    json!({
                        "task_id": t.task_id,
                        "goal": t.goal,
                        "owner": t.owner,
                        "status": format!("{:?}", t.status)
                    })
                })
                .collect();
            Some((
                "200 OK",
                json!({
                    "stopped": sched.is_stopped(),
                    "registered_skills": neon.skills.list().len(),
                    "executable_skills": ["sys.info", "perception.status"],
                    "tasks": tasks
                })
                .to_string(),
            ))
        }
        ("POST", "/v1/agent/tasks") => {
            let task_id = body["task_id"].as_str().unwrap_or_default().trim();
            let goal = body["goal"].as_str().unwrap_or("typed-agent-task");
            let owner = body["owner"].as_str().unwrap_or("hermes");
            let skill = body["skill"].as_str().unwrap_or("sys.info");
            let arg = body["argument"].as_str().unwrap_or_default();
            if task_id.is_empty() {
                return Some((
                    "400 Bad Request",
                    r#"{"error":"task_id is required for deduplication"}"#.to_string(),
                ));
            }
            if !matches!(skill, "sys.info" | "perception.status") {
                return Some(("501 Not Implemented", json!({"error":"this task skill has no installed executor", "skill":skill, "verified":false}).to_string()));
            }
            let plan = match build_plan(
                task_id,
                goal,
                &[(skill.to_string(), arg.to_string())],
                &neon.skills,
                8,
            ) {
                Ok(p) => p,
                Err(e) => return Some(("400 Bad Request", json!({"error": e}).to_string())),
            };
            let (Ok(mut broker), Ok(mut sched), Ok(mut audit)) =
                (neon.broker.lock(), neon.scheduler.lock(), neon.audit.lock())
            else {
                return Some((
                    "500 Internal Server Error",
                    r#"{"error":"agent state lock poisoned"}"#.to_string(),
                ));
            };
            let mut results = Vec::new();
            match run_task_plan(
                &plan,
                owner,
                ExecutionLimits::default(),
                &neon.skills,
                &mut broker,
                &mut sched,
                &mut audit,
                None,
                |tool_name, _| {
                    let result = match tool_name {
                        "sys.info" => json!({
                            "os":std::env::consts::OS,
                            "architecture":std::env::consts::ARCH,
                            "logical_cpus":std::thread::available_parallelism().map_err(|e| e.to_string())?.get()
                        }),
                        "perception.status" => json!({
                            "operator_enabled":neon.privacy.is_camera_allowed(),
                            "active":neon.privacy.is_camera_active(),
                            "stopped":neon.privacy.is_stopped()
                        }),
                        _ => return Err("unsupported task executor".to_string()),
                    };
                    let detail = result.to_string();
                    results.push(result);
                    Ok(VerificationEvidence::ToolExitZero {
                        tool: tool_name.to_string(),
                        detail,
                    })
                },
            ) {
                Ok(summary) => Some((
                    "200 OK",
                    json!({
                        "task_id": summary.task_id,
                        "status": format!("{:?}", summary.status),
                        "verified": summary.verified,
                        "steps_executed": summary.steps_executed,
                        "results": results,
                        "reward": summary.reinforcement.as_ref().map(|r| r.reward),
                        "reinforcement_accepted": summary.reinforcement.as_ref().map(|r| r.accepted)
                    })
                    .to_string(),
                )),
                Err(e) => Some(("409 Conflict", json!({"error": e}).to_string())),
            }
        }
        ("POST", "/v1/agent/tasks/cancel") => {
            let Ok(mut sched) = neon.scheduler.lock() else {
                return Some((
                    "500 Internal Server Error",
                    r#"{"error":"scheduler lock poisoned"}"#.to_string(),
                ));
            };
            if let Some(tid) = body["task_id"].as_str().filter(|s| !s.is_empty()) {
                let cancelled = sched.cancel_task(tid);
                Some((
                    "200 OK",
                    json!({"task_id": tid, "cancelled": cancelled}).to_string(),
                ))
            } else {
                sched.stop_all();
                Some(("200 OK", r#"{"status":"all_stopped"}"#.to_string()))
            }
        }
        ("GET", "/v1/learning/status") => {
            let lora_stats = lora.lock().ok().map(|l| l.stats());
            let mem_stats = neon.neural_memory.lock().ok().map(|m| m.stats());
            Some((
                "200 OK",
                json!({
                    "lora_connected_to_cuda_forward": false,
                    "lora_evolution_steps": lora_stats.map(|s| s.evolution_steps).unwrap_or(0),
                    "lora_param_count": lora_stats.map(|s| s.param_count).unwrap_or(0),
                    "neural_memory_generation": mem_stats.map(|s| s.active_generation).unwrap_or(0),
                    "neural_memory_updates": mem_stats.map(|s| s.total_fast_weight_updates).unwrap_or(0),
                    "neural_memory_budget_bytes": mem_stats.map(|s| s.total_budget_bytes).unwrap_or(0)
                })
                .to_string(),
            ))
        }
        ("POST", "/v1/learning/evaluate") => {
            let verdict = evaluate_reinforcement(
                RewardSignals {
                    verified_task_success: body["verified_task_success"].as_f64().unwrap_or(0.0)
                        as f32,
                    user_feedback: body["user_feedback"].as_f64().unwrap_or(0.0) as f32,
                    output_quality: body["output_quality"].as_f64().unwrap_or(0.0) as f32,
                    execution_errors: body["execution_errors"].as_f64().unwrap_or(0.0) as f32,
                    resource_cost: body["resource_cost"].as_f64().unwrap_or(0.0) as f32,
                    unnecessary_actions: body["unnecessary_actions"].as_f64().unwrap_or(0.0) as f32,
                    policy_violations: body["policy_violations"]
                        .as_u64()
                        .unwrap_or(0)
                        .min(u64::from(u32::MAX)) as u32,
                },
                LearningLevel::Level3NeuralMemory,
                false,
            );
            Some((
                "200 OK",
                json!({
                    "accepted": verdict.accepted,
                    "reward": verdict.reward,
                    "reason": verdict.reason,
                    "signals_verified": false,
                    "anti_forgetting_gate_ok": null,
                    "evaluation_mode":"advisory_only"
                })
                .to_string(),
            ))
        }
        ("POST", "/v1/learning/rollback") => {
            let Ok(mut l) = lora.lock() else {
                return Some((
                    "500 Internal Server Error",
                    r#"{"error":"lora lock poisoned"}"#.to_string(),
                ));
            };
            let rolled = l.rollback();
            Some((
                "200 OK",
                json!({"rolled_back": rolled, "evolution_steps": l.stats().evolution_steps})
                    .to_string(),
            ))
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use inkling2_companion::dynamic_lora::LoraConfig;

    #[test]
    fn neon_phase7_endpoints_enforce_dedup_memory_gating_and_learning_status() {
        let neon = NeonRuntime::new();
        let voice = VoicePersona::default();
        let lora = Mutex::new(DynamicLoraAdapter::new(LoraConfig {
            hidden_dim: 32,
            rank: 4,
            ..LoraConfig::default()
        }));
        let state_dir = std::env::temp_dir().join(format!(
            "neon-api-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        assert!(std::fs::create_dir_all(&state_dir).is_ok());

        // 1. Perception & voice status
        let p_st = handle_neon_endpoint(
            "GET",
            "/v1/perception/status",
            &Value::Null,
            &neon,
            &voice,
            &lora,
            &state_dir,
        );
        assert!(
            p_st.as_ref().is_some_and(
                |(s, b)| *s == "200 OK" && b.contains("auxiliary-structured-observation")
            )
        );
        let v_stop = handle_neon_endpoint(
            "POST",
            "/v1/voice/stop",
            &json!({}),
            &neon,
            &voice,
            &lora,
            &state_dir,
        );
        assert!(v_stop.as_ref().is_some_and(|(s, _)| *s == "200 OK"));
        assert!(neon.speech_controller.is_stopped());

        // 2. Neural Memory45 feedback + grounded search
        let fb = handle_neon_endpoint(
            "POST",
            "/v1/memory/feedback",
            &json!({"text":"SC117 runs on RTX 5060 Ti","importance":0.9,"trusted":true,"operator_permitted":true,"verified":true}),
            &neon,
            &voice,
            &lora,
            &state_dir,
        );
        assert!(
            fb.as_ref()
                .is_some_and(|(s, b)| *s == "200 OK" && b.contains("record_id"))
        );
        let search = handle_neon_endpoint(
            "POST",
            "/v1/memory/search",
            &json!({"query":"RTX 5060 Ti","top_k":3}),
            &neon,
            &voice,
            &lora,
            &state_dir,
        );
        assert!(
            search
                .as_ref()
                .is_some_and(|(s, b)| *s == "200 OK" && b.contains("unverified_observation"))
        );
        assert!(
            neon.neural_memory
                .lock()
                .is_ok_and(|m| m.stats().total_fast_weight_updates == 0)
        );
        assert!(NeonRuntime::load(&state_dir).is_ok_and(|n| {
            n.neural_memory
                .lock()
                .is_ok_and(|m| m.stats().record_count == 1)
        }));

        // 3. Agent task execution + duplicate task_id rejection
        let t1 = handle_neon_endpoint(
            "POST",
            "/v1/agent/tasks",
            &json!({"task_id":"hermes-task-1","goal":"check sys","owner":"hermes","skill":"sys.info"}),
            &neon,
            &voice,
            &lora,
            &state_dir,
        );
        assert!(
            t1.as_ref()
                .is_some_and(|(s, b)| *s == "200 OK" && b.contains("Completed"))
        );
        assert!(t1.as_ref().is_some_and(|(_, b)| {
            serde_json::from_str::<Value>(b)
                .is_ok_and(|j| j["results"][0]["os"] == std::env::consts::OS)
        }));
        let t_dup = handle_neon_endpoint(
            "POST",
            "/v1/agent/tasks",
            &json!({"task_id":"hermes-task-1","goal":"check sys","owner":"inkling2","skill":"sys.info"}),
            &neon,
            &voice,
            &lora,
            &state_dir,
        );
        assert!(t_dup.as_ref().is_some_and(|(s, _)| *s == "409 Conflict"));

        // 4. Learning status & policy-violation rejection
        let l_st = handle_neon_endpoint(
            "GET",
            "/v1/learning/status",
            &Value::Null,
            &neon,
            &voice,
            &lora,
            &state_dir,
        );
        assert!(l_st.as_ref().is_some_and(
            |(s, b)| *s == "200 OK" && b.contains("\"lora_connected_to_cuda_forward\":false")
        ));
        let l_eval = handle_neon_endpoint(
            "POST",
            "/v1/learning/evaluate",
            &json!({"verified_task_success":1.0,"policy_violations":1,"source_trusted":true}),
            &neon,
            &voice,
            &lora,
            &state_dir,
        );
        assert!(
            l_eval
                .as_ref()
                .is_some_and(|(s, b)| *s == "200 OK" && b.contains("\"accepted\":false"))
        );
        let unsupported = handle_neon_endpoint(
            "POST",
            "/v1/agent/tasks",
            &json!({"task_id":"no-executor","skill":"fs.list_allowed"}),
            &neon,
            &voice,
            &lora,
            &state_dir,
        );
        assert!(
            unsupported.is_some_and(
                |(s, b)| s == "501 Not Implemented" && b.contains("\"verified\":false")
            )
        );
        let self_claim = handle_neon_endpoint(
            "POST",
            "/v1/learning/evaluate",
            &json!({"verified_task_success":1.0,"source_trusted":true}),
            &neon,
            &voice,
            &lora,
            &state_dir,
        );
        assert!(self_claim.is_some_and(|(_, b)| {
            serde_json::from_str::<Value>(&b)
                .is_ok_and(|j| j["accepted"] == false && j["anti_forgetting_gate_ok"].is_null())
        }));
        assert!(std::fs::remove_dir_all(state_dir).is_ok());
    }
}
