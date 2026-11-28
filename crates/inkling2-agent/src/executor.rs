//! Bounded Agent State Machine (`executor`):
//! `Observe -> Plan -> PolicyCheck -> Execute -> Verify -> Record -> Learn -> Idle`.

use std::time::{Duration, Instant};

use crate::audit::AgentAuditLog;
use crate::permissions::{ActionRequest, ApprovalToken, PermissionBroker, PolicyDecision};
use crate::planner::TaskPlan;
use crate::reinforcement::{
    LearningLevel, ReinforcementVerdict, RewardSignals, evaluate_reinforcement,
};
use crate::scheduler::{TaskScheduler, TaskStatus};
use crate::skill_registry::SkillRegistry;
use crate::verifier::{VerificationEvidence, verify_evidence};

/// Explicit phase of the autonomous agent loop (Phase 3.1).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AgentPhase {
    Observe,
    Plan,
    PolicyCheck,
    Execute,
    Verify,
    Record,
    Learn,
    Idle,
}

/// Hard safety limits applied to every task execution.
#[derive(Debug, Clone, Copy)]
pub struct ExecutionLimits {
    pub max_duration: Duration,
    pub max_tool_calls: usize,
    pub max_retries: usize,
}

impl Default for ExecutionLimits {
    fn default() -> Self {
        Self {
            max_duration: Duration::from_secs(60),
            max_tool_calls: 8,
            max_retries: 1,
        }
    }
}

/// Outcome of executing a bounded task plan.
#[derive(Debug, Clone)]
pub struct ExecutionSummary {
    pub task_id: String,
    pub final_phase: AgentPhase,
    pub status: TaskStatus,
    pub steps_executed: usize,
    pub verified: bool,
    pub reinforcement: Option<ReinforcementVerdict>,
}

/// Execute a validated `TaskPlan` through the state machine with immutable Permission Broker gating.
#[allow(clippy::too_many_arguments)]
pub fn run_task_plan<F>(
    plan: &TaskPlan,
    owner: &str,
    limits: ExecutionLimits,
    registry: &SkillRegistry,
    broker: &mut PermissionBroker,
    scheduler: &mut TaskScheduler,
    audit: &mut AgentAuditLog,
    approval: Option<&ApprovalToken>,
    mut runner: F,
) -> Result<ExecutionSummary, String>
where
    F: FnMut(&str, &str) -> Result<VerificationEvidence, String>,
{
    if scheduler.is_stopped() {
        return Err("agent execution stopped; refusing to start new task".to_string());
    }
    scheduler.submit(&plan.task_id, &plan.goal, owner)?;
    scheduler.set_status(&plan.task_id, TaskStatus::Running);
    audit.record(&plan.task_id, "Observe", "start", "task accepted");

    let started = Instant::now();
    let mut executed = 0usize;
    let mut errors = 0u32;
    let mut policy_violations = 0u32;
    let mut all_verified = true;

    for step in &plan.steps {
        if scheduler.is_stopped()
            || matches!(
                scheduler.get(&plan.task_id).map(|t| &t.status),
                Some(TaskStatus::Cancelled)
            )
        {
            scheduler.set_status(&plan.task_id, TaskStatus::Cancelled);
            audit.record(&plan.task_id, "Execute", &step.skill_name, "cancelled");
            return Ok(ExecutionSummary {
                task_id: plan.task_id.clone(),
                final_phase: AgentPhase::Idle,
                status: TaskStatus::Cancelled,
                steps_executed: executed,
                verified: false,
                reinforcement: None,
            });
        }
        if started.elapsed() > limits.max_duration || executed >= limits.max_tool_calls {
            scheduler.set_status(
                &plan.task_id,
                TaskStatus::Failed("execution limit reached".to_string()),
            );
            audit.record(&plan.task_id, "Execute", &step.skill_name, "limit-reached");
            return Ok(ExecutionSummary {
                task_id: plan.task_id.clone(),
                final_phase: AgentPhase::Idle,
                status: TaskStatus::Failed("execution limit reached".to_string()),
                steps_executed: executed,
                verified: false,
                reinforcement: None,
            });
        }

        let Some(spec) = registry.get(&step.skill_name) else {
            policy_violations += 1;
            all_verified = false;
            scheduler.set_status(
                &plan.task_id,
                TaskStatus::Failed(format!("unknown skill {}", step.skill_name)),
            );
            break;
        };

        let req = ActionRequest {
            task_id: plan.task_id.clone(),
            action_name: spec.name.clone(),
            risk: spec.risk,
            is_raw_shell_command: false,
        };
        let decision = broker.evaluate(&req, approval);
        audit.record(
            &plan.task_id,
            "PolicyCheck",
            &spec.name,
            format!("{decision:?}"),
        );

        match decision {
            PolicyDecision::Allowed => {}
            PolicyDecision::RequiresApproval { reason: _ } => {
                scheduler.set_status(&plan.task_id, TaskStatus::AwaitingApproval);
                return Ok(ExecutionSummary {
                    task_id: plan.task_id.clone(),
                    final_phase: AgentPhase::PolicyCheck,
                    status: TaskStatus::AwaitingApproval,
                    steps_executed: executed,
                    verified: false,
                    reinforcement: Some(evaluate_reinforcement(
                        RewardSignals {
                            verified_task_success: 0.0,
                            user_feedback: 0.0,
                            output_quality: 0.0,
                            execution_errors: 0.0,
                            resource_cost: 0.0,
                            unnecessary_actions: 0.0,
                            policy_violations: 0,
                        },
                        LearningLevel::Level1Episodic,
                        true,
                    )),
                });
            }
            PolicyDecision::Denied { reason } => {
                policy_violations += 1;
                let msg = format!("denied: {reason}");
                scheduler.set_status(&plan.task_id, TaskStatus::Failed(msg.clone()));
                let verdict = evaluate_reinforcement(
                    RewardSignals {
                        verified_task_success: 0.0,
                        user_feedback: 0.0,
                        output_quality: 0.0,
                        execution_errors: 1.0,
                        resource_cost: 0.0,
                        unnecessary_actions: 0.0,
                        policy_violations,
                    },
                    LearningLevel::Level1Episodic,
                    true,
                );
                return Ok(ExecutionSummary {
                    task_id: plan.task_id.clone(),
                    final_phase: AgentPhase::Idle,
                    status: TaskStatus::Failed(msg),
                    steps_executed: executed,
                    verified: false,
                    reinforcement: Some(verdict),
                });
            }
        }

        let mut attempt = 0usize;
        let mut step_ok = false;
        while attempt <= limits.max_retries {
            if started.elapsed() >= limits.max_duration || executed >= limits.max_tool_calls {
                break;
            }
            attempt += 1;
            executed += 1;
            match runner(&spec.name, &step.argument) {
                Ok(evidence) => {
                    let check = verify_evidence(&evidence);
                    audit.record(&plan.task_id, "Verify", &spec.name, &check.reason);
                    if check.verified {
                        step_ok = true;
                        break;
                    }
                    errors += 1;
                }
                Err(err) => {
                    errors += 1;
                    audit.record(&plan.task_id, "Execute", &spec.name, &err);
                }
            }
        }
        if !step_ok {
            all_verified = false;
            break;
        }
    }

    let verdict = evaluate_reinforcement(
        RewardSignals {
            verified_task_success: if all_verified && errors == 0 {
                1.0
            } else {
                0.0
            },
            user_feedback: 0.0,
            output_quality: if all_verified { 0.8 } else { 0.2 },
            execution_errors: (errors as f32).min(1.0),
            resource_cost: 0.05,
            unnecessary_actions: 0.0,
            policy_violations,
        },
        LearningLevel::Level1Episodic,
        true,
    );
    audit.record(
        &plan.task_id,
        "Learn",
        "reinforce",
        format!("accepted={} reward={:.3}", verdict.accepted, verdict.reward),
    );

    let final_status = if all_verified {
        TaskStatus::Completed
    } else {
        TaskStatus::Failed("verification failed".to_string())
    };
    scheduler.set_status(&plan.task_id, final_status.clone());
    Ok(ExecutionSummary {
        task_id: plan.task_id.clone(),
        final_phase: AgentPhase::Idle,
        status: final_status,
        steps_executed: executed,
        verified: all_verified,
        reinforcement: Some(verdict),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::planner::build_plan;

    #[test]
    fn agent_executes_verified_readonly_plan_and_blocks_high_risk_without_approval() {
        let reg = SkillRegistry::with_defaults();
        let mut broker = PermissionBroker::default();
        let mut sched = TaskScheduler::default();
        let mut audit = AgentAuditLog::default();

        let plan = build_plan(
            "task-ok-1",
            "Inspect system",
            &[("sys.info".to_string(), "".to_string())],
            &reg,
            4,
        );
        assert!(plan.is_ok());
        if let Ok(p) = plan {
            let summary = run_task_plan(
                &p,
                "inkling2",
                ExecutionLimits::default(),
                &reg,
                &mut broker,
                &mut sched,
                &mut audit,
                None,
                |tool, _| {
                    Ok(VerificationEvidence::ToolExitZero {
                        tool: tool.to_string(),
                        detail: "ok".to_string(),
                    })
                },
            );
            assert!(summary.is_ok());
            if let Ok(s) = summary {
                assert_eq!(s.status, TaskStatus::Completed);
                assert!(s.verified);
                assert!(
                    s.reinforcement
                        .as_ref()
                        .is_some_and(|r| r.accepted && r.reward > 0.0)
                );
            }
        }

        // High-risk delete without approval must pause at AwaitingApproval (0 unapproved executions).
        let risky = build_plan(
            "task-del-1",
            "Delete file",
            &[("fs.delete_path".to_string(), "important.txt".to_string())],
            &reg,
            4,
        );
        assert!(risky.is_ok());
        if let Ok(p) = risky {
            let mut ran = false;
            let res = run_task_plan(
                &p,
                "hermes",
                ExecutionLimits::default(),
                &reg,
                &mut broker,
                &mut sched,
                &mut audit,
                None,
                |_, _| {
                    ran = true;
                    Ok(VerificationEvidence::UserConfirmed)
                },
            );
            assert!(!ran);
            assert!(res.is_ok());
            if let Ok(s) = res {
                assert_eq!(s.status, TaskStatus::AwaitingApproval);
            }
        }
    }

    #[test]
    fn self_claim_verification_and_policy_violation_are_rejected() {
        let v = verify_evidence(&VerificationEvidence::SelfClaimOnly);
        assert!(!v.verified);

        let r = evaluate_reinforcement(
            RewardSignals {
                verified_task_success: 1.0,
                user_feedback: 1.0,
                output_quality: 1.0,
                execution_errors: 0.0,
                resource_cost: 0.0,
                unnecessary_actions: 0.0,
                policy_violations: 1,
            },
            LearningLevel::Level3NeuralMemory,
            true,
        );
        assert!(!r.accepted);
        assert!((r.reward + 1.0).abs() < 1e-6);
    }

    #[test]
    fn duplicate_task_id_and_global_stop_are_enforced() {
        let mut sched = TaskScheduler::default();
        assert!(sched.submit("dup-1", "goal", "hermes").is_ok());
        assert!(sched.submit("dup-1", "goal", "inkling2").is_err());
        sched.stop_all();
        assert!(sched.submit("after-stop", "goal", "inkling2").is_err());
    }
}
