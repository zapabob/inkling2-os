//! Loopback-Only Lightweight HTTP/1.1 + OpenAI-Compatible Bridge (`127.0.0.1:8080`).
//!
//! Designed for Meta Muse.ai VM <-> Inkling2-OS encrypted SSH tunnel (`VM:18080 -> PC:8080`).
//! Security controls (`SOP-Security`):
//! - Strictly binds to loopback (`127.0.0.1:<port>` only; refuses non-loopback bind addresses)
//! - Optional Bearer token auth via `INKLING2_BRIDGE_TOKEN` for all POST endpoints
//! - Strictly typed, allowlisted endpoints (NO arbitrary shell execution endpoint)
//! - Bounded request body size (256 KiB max)

use std::env;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{IpAddr, SocketAddr, TcpListener, TcpStream};
use std::path::PathBuf;
use std::process::Command;
use std::sync::{Arc, Mutex};
use std::thread;

use inkling2_companion::dynamic_lora::{DynamicLoraAdapter, LoraConfig};
use inkling2_companion::memory45::{MAX_MEMORY_BYTES, Memory45Store};
use inkling2_companion::voice_call::{VoicePersona, speak_text};

use crate::strata_catalog::discover_models;

const MAX_BODY_BYTES: usize = 256 * 1024;
const CONSOLE_HTML: &str = include_str!("../../../ui/index.html");

/// Parsed live GPU telemetry snapshot from `nvidia-smi`.
#[derive(Debug, Clone, PartialEq)]
pub struct GpuTelemetry {
    pub name: String,
    pub driver_version: String,
    pub memory_used_mib: u64,
    pub memory_total_mib: u64,
    pub utilization_gpu: u32,
    pub temperature_c: u32,
    pub power_draw_w: f64,
    pub power_limit_w: f64,
    pub clock_sm_mhz: u32,
}

/// Parse a single CSV line from:
/// `nvidia-smi --query-gpu=name,driver_version,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw,power.limit,clocks.sm --format=csv,noheader,nounits`
pub fn parse_nvidia_smi_csv(line: &str) -> Option<GpuTelemetry> {
    let cols: Vec<&str> = line.split(',').map(str::trim).collect();
    if cols.len() < 9 {
        return None;
    }
    Some(GpuTelemetry {
        name: cols[0].to_string(),
        driver_version: cols[1].to_string(),
        memory_used_mib: cols[2].parse::<f64>().ok()? as u64,
        memory_total_mib: cols[3].parse::<f64>().ok()? as u64,
        utilization_gpu: cols[4].parse::<f64>().ok()? as u32,
        temperature_c: cols[5].parse::<f64>().ok()? as u32,
        power_draw_w: cols[6].parse::<f64>().unwrap_or(0.0),
        power_limit_w: cols[7].parse::<f64>().unwrap_or(0.0),
        clock_sm_mhz: cols[8].parse::<f64>().unwrap_or(0.0) as u32,
    })
}

/// Extract BF16 TFLOPS, INT8 TOPS, and NVFP4 TFLOPS from `inkling2-engine bench-tc` output.
pub fn parse_bench_tc_metrics(output: &str) -> (f64, f64, f64) {
    fn extract_after(haystack: &str, key: &str) -> f64 {
        let Some(pos) = haystack.find(key) else {
            return 0.0;
        };
        let rest = &haystack[pos + key.len()..];
        let trimmed = rest.trim_start_matches(|c: char| !c.is_ascii_digit() && c != '.');
        let end = trimmed
            .find(|c: char| !(c.is_ascii_digit() || c == '.'))
            .unwrap_or(trimmed.len());
        trimmed[..end].parse::<f64>().unwrap_or(0.0)
    }

    let bf16 = extract_after(output, "BF16=");
    let int8 = extract_after(output, "INT8=");
    let nvfp4 = extract_after(output, "NVFP4=");
    (bf16, int8, nvfp4)
}

fn query_live_gpu_telemetry() -> Option<GpuTelemetry> {
    let out = Command::new("nvidia-smi")
        .args([
            "--query-gpu=name,driver_version,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw,power.limit,clocks.sm",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&out.stdout);
    let first = text.lines().find(|l| !l.trim().is_empty())?;
    parse_nvidia_smi_csv(first)
}

/// Shared state protected by mutex across HTTP worker threads.
pub struct BridgeContext {
    pub engine_bin: PathBuf,
    pub models_dir: PathBuf,
    pub state_dir: PathBuf,
    pub memory: Mutex<Memory45Store>,
    pub lora: Mutex<DynamicLoraAdapter>,
    pub voice: VoicePersona,
    pub bearer_token: Option<String>,
}

fn json_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 8);
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {}
            c => out.push(c),
        }
    }
    out
}

/// Minimal zero-dependency JSON string field extractor (returns the LAST occurrence of `"key":"value"`,
/// which handles OpenAI `messages: [{role:system,content:...},{role:user,content:...}]` cleanly).
fn extract_json_str(body: &str, key: &str) -> Option<String> {
    let needle = format!("\"{key}\"");
    let pos = body.rfind(&needle)?;
    let colon = body[pos + needle.len()..].find(':')? + pos + needle.len();
    let q1 = body[colon + 1..].find('"')? + colon + 1;
    let bytes = body.as_bytes();
    let mut i = q1 + 1;
    let mut out = String::new();
    while i < bytes.len() {
        if bytes[i] == b'\\' && i + 1 < bytes.len() {
            match bytes[i + 1] {
                b'n' => out.push('\n'),
                b'r' => out.push('\r'),
                b't' => out.push('\t'),
                b'"' => out.push('"'),
                b'\\' => out.push('\\'),
                other => out.push(other as char),
            }
            i += 2;
            continue;
        }
        if bytes[i] == b'"' {
            return Some(out);
        }
        // Handle multi-byte UTF-8 cleanly
        let s_rest = &body[i..];
        let ch = s_rest.chars().next()?;
        out.push(ch);
        i += ch.len_utf8();
    }
    None
}

fn extract_json_f32(body: &str, key: &str) -> Option<f32> {
    let needle = format!("\"{key}\"");
    let pos = body.find(&needle)?;
    let colon = body[pos + needle.len()..].find(':')? + pos + needle.len();
    let rest = body[colon + 1..].trim_start();
    let end = rest
        .find(|c: char| !(c.is_ascii_digit() || matches!(c, '.' | '-' | '+' | 'e' | 'E')))
        .unwrap_or(rest.len());
    rest[..end].parse::<f32>().ok()
}

fn send_json(stream: &mut TcpStream, status: &str, json_body: &str) {
    let resp = format!(
        "HTTP/1.1 {status}\r\nContent-Type: application/json; charset=utf-8\r\nCache-Control: no-store\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{json_body}",
        json_body.len()
    );
    let _ = stream.write_all(resp.as_bytes());
    let _ = stream.flush();
}

fn send_html(stream: &mut TcpStream, status: &str, html_body: &str) {
    let resp = format!(
        "HTTP/1.1 {status}\r\nContent-Type: text/html; charset=utf-8\r\nCache-Control: no-store\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{html_body}",
        html_body.len()
    );
    let _ = stream.write_all(resp.as_bytes());
    let _ = stream.flush();
}

fn handle_client(mut stream: TcpStream, ctx: Arc<BridgeContext>) {
    let _ = stream.set_read_timeout(Some(std::time::Duration::from_secs(10)));
    let mut reader = match stream.try_clone() {
        Ok(s) => BufReader::new(s),
        Err(_) => return,
    };

    let mut req_line = String::new();
    if reader.read_line(&mut req_line).unwrap_or(0) == 0 {
        return;
    }
    let parts: Vec<&str> = req_line.split_whitespace().collect();
    if parts.len() < 2 {
        send_json(&mut stream, "400 Bad Request", r#"{"error":"malformed request line"}"#);
        return;
    }
    let method = parts[0];
    let path = parts[1].split('?').next().unwrap_or(parts[1]);

    let mut content_len: usize = 0;
    let mut auth_header: Option<String> = None;
    loop {
        let mut hline = String::new();
        if reader.read_line(&mut hline).unwrap_or(0) == 0 {
            break;
        }
        let trimmed = hline.trim();
        if trimmed.is_empty() {
            break;
        }
        if let Some((k, v)) = trimmed.split_once(':') {
            let kl = k.trim().to_ascii_lowercase();
            let val = v.trim();
            if kl == "content-length" {
                content_len = val.parse().unwrap_or(0);
            } else if kl == "authorization" {
                auth_header = Some(val.to_string());
            }
        }
    }

    if content_len > MAX_BODY_BYTES {
        send_json(&mut stream, "413 Payload Too Large", r#"{"error":"body exceeds 256 KiB limit"}"#);
        return;
    }

    let mut body_bytes = vec![0u8; content_len];
    if content_len > 0 && reader.read_exact(&mut body_bytes).is_err() {
        send_json(&mut stream, "400 Bad Request", r#"{"error":"incomplete body"}"#);
        return;
    }
    let body = String::from_utf8_lossy(&body_bytes);

    // Enforce Bearer token on POST endpoints if INKLING2_BRIDGE_TOKEN is configured
    if method == "POST" {
        if let Some(expected) = &ctx.bearer_token {
            let expected_hdr = format!("Bearer {expected}");
            if auth_header.as_deref() != Some(expected_hdr.as_str()) {
                send_json(&mut stream, "401 Unauthorized", r#"{"error":"missing or invalid Bearer token"}"#);
                return;
            }
        }
    }

    match (method, path) {
        ("GET", "/") | ("GET", "/index.html") => {
            send_html(&mut stream, "200 OK", CONSOLE_HTML);
        }
        ("GET", "/v1/gpu/info") => {
            if let Some(g) = query_live_gpu_telemetry() {
                let json = format!(
                    r#"{{"available":true,"name":"{}","driver_version":"{}","cuda_target":"13.5-sm_120a","memory_used_mib":{},"memory_total_mib":{},"utilization_gpu":{},"temperature_c":{},"power_draw_w":{:.2},"power_limit_w":{:.2},"clock_sm_mhz":{}}}"#,
                    json_escape(&g.name),
                    json_escape(&g.driver_version),
                    g.memory_used_mib,
                    g.memory_total_mib,
                    g.utilization_gpu,
                    g.temperature_c,
                    g.power_draw_w,
                    g.power_limit_w,
                    g.clock_sm_mhz
                );
                send_json(&mut stream, "200 OK", &json);
            } else {
                send_json(&mut stream, "200 OK", r#"{"available":false,"cuda_target":"13.5-sm_120a"}"#);
            }
        }
        ("GET", "/health") => {
            let m_entries = ctx.memory.lock().map(|m| m.stats().entry_count).unwrap_or(0);
            let l_steps = ctx.lora.lock().map(|l| l.stats().evolution_steps).unwrap_or(0);
            let json = format!(
                r#"{{"status":"ok","service":"inkling2-os-bridge","cuda":"13.5-sm_120a","memory_entries":{m_entries},"memory_budget_bytes":{MAX_MEMORY_BYTES},"lora_steps":{l_steps}}}"#
            );
            send_json(&mut stream, "200 OK", &json);
        }
        ("GET", "/v1/models") => {
            let models = discover_models(&ctx.models_dir);
            let mut items = Vec::new();
            for m in models {
                let id = m
                    .path
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("inkling-2-mini");
                items.push(format!(
                    r#"{{"id":"{}","object":"model","owned_by":"SC117-Inkling2-OS","bytes":{}}}"#,
                    json_escape(id),
                    m.file_bytes
                ));
            }
            let json = format!(r#"{{"object":"list","data":[{}]}}"#, items.join(","));
            send_json(&mut stream, "200 OK", &json);
        }
        ("GET", "/v1/memory/stats") => {
            if let Ok(m) = ctx.memory.lock() {
                let s = m.stats();
                let json = format!(
                    r#"{{"entries":{},"serialized_bytes":{},"budget_bytes":{},"compactions":{}}}"#,
                    s.entry_count, s.serialized_bytes, s.budget_bytes, s.compactions
                );
                send_json(&mut stream, "200 OK", &json);
            } else {
                send_json(&mut stream, "500 Internal Server Error", r#"{"error":"memory lock poisoned"}"#);
            }
        }
        ("POST", "/v1/memory/add") => {
            let text = extract_json_str(&body, "text").unwrap_or_default();
            let imp = extract_json_f32(&body, "importance").unwrap_or(0.8);
            if text.is_empty() {
                send_json(&mut stream, "400 Bad Request", r#"{"error":"missing text field"}"#);
                return;
            }
            if let Ok(mut m) = ctx.memory.lock() {
                let id = m.insert(text, imp);
                let saved = m.save_atomic(&ctx.state_dir.join("memory45.ikm")).is_ok();
                let s = m.stats();
                let json = format!(
                    r#"{{"id":{id},"saved":{saved},"serialized_bytes":{},"budget_bytes":{}}}"#,
                    s.serialized_bytes, s.budget_bytes
                );
                send_json(&mut stream, "200 OK", &json);
            } else {
                send_json(&mut stream, "500 Internal Server Error", r#"{"error":"memory lock poisoned"}"#);
            }
        }
        ("POST", "/v1/memory/recall") => {
            let query = extract_json_str(&body, "query").unwrap_or_default();
            if let Ok(mut m) = ctx.memory.lock() {
                let hits = m.recall(&query, 5);
                let items: Vec<String> = hits
                    .iter()
                    .map(|h| {
                        format!(
                            r#"{{"id":{},"importance":{:.2},"strength":{:.2},"text":"{}"}}"#,
                            h.id,
                            h.importance,
                            h.strength,
                            json_escape(&h.text)
                        )
                    })
                    .collect();
                let json = format!(r#"{{"hits":[{}]}}"#, items.join(","));
                send_json(&mut stream, "200 OK", &json);
            } else {
                send_json(&mut stream, "500 Internal Server Error", r#"{"error":"memory lock poisoned"}"#);
            }
        }
        ("GET", "/v1/lora/stats") => {
            if let Ok(l) = ctx.lora.lock() {
                let s = l.stats();
                let json = format!(
                    r#"{{"steps":{},"params":{},"cumulative_reward":{:.4},"norm_a":{:.4},"norm_b":{:.4}}}"#,
                    s.evolution_steps, s.param_count, s.cumulative_reward, s.norm_a, s.norm_b
                );
                send_json(&mut stream, "200 OK", &json);
            } else {
                send_json(&mut stream, "500 Internal Server Error", r#"{"error":"lora lock poisoned"}"#);
            }
        }
        ("POST", "/v1/lora/reinforce") => {
            let text = extract_json_str(&body, "text").unwrap_or_default();
            let reward = extract_json_f32(&body, "reward").unwrap_or(0.5);
            let emb = Memory45Store::embed_text_q8(&text);
            let mut h = vec![0.0f32; 2304];
            let mut target = vec![0.0f32; 2304];
            for i in 0..2304 {
                let v = f32::from(emb[i % emb.len()]) / 127.0;
                h[i] = v;
                target[i] = v * reward.signum();
            }
            if let Ok(mut l) = ctx.lora.lock() {
                l.reinforce_step(&h, &target, reward);
                let saved = l.save_atomic(&ctx.state_dir.join("dynamic_lora.iklora")).is_ok();
                let s = l.stats();
                let json = format!(
                    r#"{{"steps":{},"norm_b":{:.4},"saved":{saved}}}"#,
                    s.evolution_steps, s.norm_b
                );
                send_json(&mut stream, "200 OK", &json);
            } else {
                send_json(&mut stream, "500 Internal Server Error", r#"{"error":"lora lock poisoned"}"#);
            }
        }
        ("POST", "/v1/lora/rollback") => {
            if let Ok(mut l) = ctx.lora.lock() {
                let ok = l.rollback();
                if ok {
                    let _ = l.save_atomic(&ctx.state_dir.join("dynamic_lora.iklora"));
                }
                send_json(&mut stream, "200 OK", &format!(r#"{{"rolled_back":{ok}}}"#));
            } else {
                send_json(&mut stream, "500 Internal Server Error", r#"{"error":"lora lock poisoned"}"#);
            }
        }
        ("POST", "/v1/voice/say") => {
            let text = extract_json_str(&body, "text").unwrap_or_default();
            let wav = ctx.state_dir.join("bridge_voice.wav");
            match speak_text(&ctx.voice, &text, Some(&wav)) {
                Ok(msg) => send_json(&mut stream, "200 OK", &format!(r#"{{"status":"ok","detail":"{}"}}"#, json_escape(&msg))),
                Err(e) => send_json(&mut stream, "500 Internal Server Error", &format!(r#"{{"error":"{}"}}"#, json_escape(&e))),
            }
        }
        ("POST", "/v1/gpu/bench-tc") => {
            match Command::new(&ctx.engine_bin).arg("bench-tc").output() {
                Ok(o) => {
                    let stderr_log = String::from_utf8_lossy(&o.stderr);
                    let stdout_log = String::from_utf8_lossy(&o.stdout);
                    let combined = if stdout_log.trim().is_empty() {
                        stderr_log.into_owned()
                    } else {
                        format!("{stderr_log}\n{stdout_log}")
                    };
                    let (bf16, int8, nvfp4) = parse_bench_tc_metrics(&combined);
                    let line = combined.lines().rev().find(|l| !l.trim().is_empty()).unwrap_or("ok");
                    send_json(
                        &mut stream,
                        "200 OK",
                        &format!(
                            r#"{{"exit_code":{},"bf16_tflops":{:.2},"int8_tops":{:.2},"nvfp4_tflops":{:.2},"result":"{}","log":"{}"}}"#,
                            o.status.code().unwrap_or(-1),
                            bf16,
                            int8,
                            nvfp4,
                            json_escape(line),
                            json_escape(&combined)
                        ),
                    );
                }
                Err(e) => send_json(&mut stream, "500 Internal Server Error", &format!(r#"{{"error":"{}"}}"#, json_escape(&e.to_string()))),
            }
        }
        ("POST", "/v1/chat/completions") => {
            let user_msg = extract_json_str(&body, "content").unwrap_or_else(|| "ping".to_string());
            let recalled = ctx
                .memory
                .lock()
                .ok()
                .map(|mut m| {
                    let hits = m.recall(&user_msg, 3);
                    m.insert(format!("user: {user_msg}"), 0.75);
                    let _ = m.save_atomic(&ctx.state_dir.join("memory45.ikm"));
                    hits.into_iter().map(|h| h.text).collect::<Vec<_>>().join(" | ")
                })
                .unwrap_or_default();

            // Plan A: Forward to loopback-only upstream OpenAI-compatible inference server (Strata / llama-server)
            // if INKLING2_UPSTREAM_OPENAI is configured (strictly loopback 127.0.0.1 only per SOP-Security).
            let upstream = env::var("INKLING2_UPSTREAM_OPENAI").ok().filter(|u| {
                u.starts_with("http://127.0.0.1:") || u.starts_with("http://localhost:")
            });
            if let Some(up_url) = upstream {
                let up_model = env::var("INKLING2_UPSTREAM_MODEL").unwrap_or_else(|_| "Inkling-2-Mini".to_string());
                let up_key = env::var("INKLING2_UPSTREAM_API_KEY").ok().filter(|k| !k.is_empty());
                let req_max_tok = extract_json_f32(&body, "max_tokens")
                    .map(|v| (v as u32).clamp(32, 4096))
                    .unwrap_or(1024);
                let req_effort = extract_json_str(&body, "reasoning_effort").unwrap_or_else(|| "none".to_string());
                let effort_num = match req_effort.as_str() {
                    "none" => "0.0",
                    "minimal" => "0.1",
                    "low" => "0.2",
                    "medium" => "0.7",
                    _ => "0.9",
                };
                let injected_body = if recalled.is_empty() {
                    format!(
                        r#"{{"model":"{}","messages":[{{"role":"user","content":"{}"}}],"max_tokens":{},"reasoning_effort":"{}","chat_template_kwargs":{{"reasoning_effort":{}}}}}"#,
                        json_escape(&up_model),
                        json_escape(&user_msg),
                        req_max_tok,
                        json_escape(&req_effort),
                        effort_num
                    )
                } else {
                    format!(
                        r#"{{"model":"{}","messages":[{{"role":"system","content":"[Memory45 Context] {}"}},{{"role":"user","content":"{}"}}],"max_tokens":{},"reasoning_effort":"{}","chat_template_kwargs":{{"reasoning_effort":{}}}}}"#,
                        json_escape(&up_model),
                        json_escape(&recalled),
                        json_escape(&user_msg),
                        req_max_tok,
                        json_escape(&req_effort),
                        effort_num
                    )
                };
                let mut curl_cmd = Command::new("curl");
                curl_cmd
                    .args([
                        "-sS", "-f", "-X", "POST",
                        "-H", "Content-Type: application/json",
                    ])
                    .stdin(std::process::Stdio::piped())
                    .stdout(std::process::Stdio::piped())
                    .stderr(std::process::Stdio::piped());
                if let Some(ref key) = up_key {
                    curl_cmd.args(["-H", &format!("Authorization: Bearer {key}")]);
                }
                // Pass JSON payload via stdin (`@-`) so raw UTF-8 Japanese prompts are never mangled
                // by Windows ACP/cp932 command-line encoding.
                curl_cmd.args([
                    "--connect-timeout", "5",
                    "--max-time", "240",
                    "--data-binary", "@-",
                    &up_url,
                ]);
                if let Ok(mut child) = curl_cmd.spawn() {
                    if let Some(mut stdin) = child.stdin.take() {
                        let _ = stdin.write_all(injected_body.as_bytes());
                    }
                    if let Ok(o) = child.wait_with_output() {
                        if o.status.success() && !o.stdout.is_empty() {
                            let up_json = String::from_utf8_lossy(&o.stdout);
                            send_json(&mut stream, "200 OK", &up_json);
                            return;
                        }
                        let err_msg = String::from_utf8_lossy(&o.stderr);
                        let _ = writeln!(
                            std::io::stderr(),
                            "[Inkling2-Bridge] upstream {up_url} failed (status={}): {}",
                            o.status,
                            err_msg.trim()
                        );
                    }
                }
            }

            // Plan B: Invoke native CUDA 13.5 sm_120a inkling2-engine.exe `generate` on local SC117 .strata
            let lora_steps = ctx.lora.lock().map(|l| l.stats().evolution_steps).unwrap_or(0);
            let native_out = discover_models(&ctx.models_dir)
                .into_iter()
                .next()
                .and_then(|m| {
                    Command::new(&ctx.engine_bin)
                        .args(["generate", &m.path.to_string_lossy(), &user_msg, "8"])
                        .output()
                        .ok()
                })
                .filter(|o| o.status.success())
                .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
                .filter(|s| !s.is_empty());

            let reply = if let Some(gen_text) = native_out {
                gen_text
            } else {
                format!(
                    "[Inkling2-OS Bridge | CUDA 13.5 sm_120a | LoRA step={lora_steps}] 記憶参照: {}",
                    if recalled.is_empty() { "なし" } else { &recalled }
                )
            };
            let json = format!(
                r#"{{"id":"chatcmpl-inkling2","object":"chat.completion","model":"Inkling-2-Mini-SQ-10-abliterated","choices":[{{"index":0,"message":{{"role":"assistant","content":"{}"}},"finish_reason":"stop"}}]}}"#,
                json_escape(&reply)
            );
            send_json(&mut stream, "200 OK", &json);
        }
        _ => {
            send_json(&mut stream, "404 Not Found", r#"{"error":"endpoint not found"}"#);
        }
    }
}

/// Start the loopback-only HTTP/OpenAI bridge server (`127.0.0.1:<port>`).
pub fn run_http_bridge(
    bind_addr: &str,
    engine_bin: PathBuf,
    models_dir: PathBuf,
    state_dir: PathBuf,
) -> Result<(), String> {
    let addr: SocketAddr = bind_addr
        .parse()
        .map_err(|e| format!("invalid bind address '{bind_addr}': {e}"))?;
    if !matches!(addr.ip(), IpAddr::V4(ip) if ip.is_loopback()) && !addr.ip().is_loopback() {
        return Err(format!(
            "refusing non-loopback bind '{bind_addr}' (SOP-Security: bind 127.0.0.1 only; expose via SSH tunnel)"
        ));
    }

    let memory = Memory45Store::load(&state_dir.join("memory45.ikm")).unwrap_or_default();
    let lora = DynamicLoraAdapter::load(&state_dir.join("dynamic_lora.iklora"))
        .unwrap_or_else(|_| DynamicLoraAdapter::new(LoraConfig::default()));
    let bearer_token = env::var("INKLING2_BRIDGE_TOKEN").ok().filter(|t| !t.is_empty());

    let ctx = Arc::new(BridgeContext {
        engine_bin,
        models_dir,
        state_dir,
        memory: Mutex::new(memory),
        lora: Mutex::new(lora),
        voice: VoicePersona::default(),
        bearer_token,
    });

    let listener = TcpListener::bind(addr).map_err(|e| format!("bind {addr} failed: {e}"))?;
    let mut stderr = std::io::stderr().lock();
    let _ = writeln!(
        stderr,
        "[Inkling2-Bridge] Listening on http://{addr} (loopback-only; Muse.ai VM tunnel -> 127.0.0.1:18080)"
    );
    drop(stderr);

    for incoming in listener.incoming() {
        if let Ok(stream) = incoming {
            let c = Arc::clone(&ctx);
            thread::spawn(move || handle_client(stream, c));
        }
    }
    Ok(())
}
