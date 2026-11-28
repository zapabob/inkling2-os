//! Loopback-Only Lightweight HTTP/1.1 + OpenAI-Compatible Bridge (`127.0.0.1:8080`).
//!
//! Designed for Meta Muse.ai VM <-> Inkling2-OS encrypted SSH tunnel (`VM:18080 -> PC:8080`).
//! Security controls (`SOP-Security`):
//! - Strictly binds to loopback (`127.0.0.1:<port>` only; refuses non-loopback bind addresses)
//! - Local console mode or scoped Bearer authentication for tunneled deployment
//! - Strictly typed, allowlisted endpoints (NO arbitrary shell execution endpoint)
//! - Bounded request body size (256 KiB max)

use std::env;
use std::io::{BufReader, Write};
use std::net::{IpAddr, SocketAddr, TcpListener, TcpStream};
use std::path::PathBuf;
use std::process::Command;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

use cap_std::{ambient_authority, fs::Dir};
use serde_json::{Value, json};

use inkling2_companion::dynamic_lora::{DynamicLoraAdapter, LoraConfig};
use inkling2_companion::memory45::Memory45Store;
use inkling2_companion::voice_call::{VoicePersona, speak_text};

use crate::hardened_fs::{list_jailed_dir, read_jailed_file};
use crate::http_request::{bearer_matches, read_request};
use crate::native_inference;
use crate::neon_api::{NeonRuntime, handle_neon_endpoint};
use crate::public_browser::{fetch_public_https, render_document};
use crate::resident_engine::ResidentEngine;
use crate::strata_catalog::StrataModelInfo;
use crate::strata_catalog::discover_models;

const MAX_CONNECTIONS: usize = 16;
const CONSOLE_HTML: &str = include_str!("../../../ui/index.html");
const CONSOLE_JS: &str = include_str!("../../../ui/console.js");
const THEME_JS: &str = include_str!("../../../ui/theme.js");

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

pub fn query_live_gpu_telemetry() -> Option<GpuTelemetry> {
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
    voice_slot: Mutex<()>,
    bearer_token: Option<String>,
    inference_token: Option<String>,
    local_console: bool,
    search_port: Option<u16>,
    preview_dir: Option<Dir>,
    model_listing: Option<Dir>,
    active_connections: AtomicUsize,
    inference_slot: Mutex<Option<ResidentEngine>>,
    resident: bool,
    experimental_lora: bool,
    upstream: Option<SocketAddr>,
    upstream_model: Option<String>,
    upstream_key: Option<String>,
    neon: NeonRuntime,
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

fn extract_json_str(body: &Value, key: &str) -> Option<String> {
    body.get(key)?.as_str().map(str::to_owned)
}

fn extract_json_f32(body: &Value, key: &str) -> Option<f32> {
    let value = body.get(key)?.as_f64()? as f32;
    value.is_finite().then_some(value)
}

fn send_error(stream: &mut TcpStream, status: &str, message: &str) {
    send_json(stream, status, &json!({"error": message}).to_string());
}

const SECURITY_HEADERS: &str = "X-Content-Type-Options: nosniff\r\nX-Frame-Options: DENY\r\nReferrer-Policy: no-referrer\r\nCross-Origin-Resource-Policy: same-origin\r\nContent-Security-Policy: default-src 'self'; script-src 'self'; style-src 'unsafe-inline'; img-src 'self' data:; media-src 'self' blob:; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'none'; object-src 'none'\r\nPermissions-Policy: camera=(), microphone=(), geolocation=()\r\n";

fn send_json(stream: &mut TcpStream, status: &str, json_body: &str) {
    let resp = format!(
        "HTTP/1.1 {status}\r\nContent-Type: application/json; charset=utf-8\r\nCache-Control: no-store\r\n{SECURITY_HEADERS}Content-Length: {}\r\nConnection: close\r\n\r\n{json_body}",
        json_body.len()
    );
    let _ = stream.write_all(resp.as_bytes());
    let _ = stream.flush();
}

fn send_asset(stream: &mut TcpStream, mime: &str, html_body: &str) {
    let resp = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: {mime}; charset=utf-8\r\nCache-Control: no-store\r\n{SECURITY_HEADERS}Content-Length: {}\r\nConnection: close\r\n\r\n{html_body}",
        html_body.len()
    );
    let _ = stream.write_all(resp.as_bytes());
    let _ = stream.flush();
}

fn completion_json(model: &str, reply: native_inference::NativeReply) -> Value {
    json!({
        "id":"chatcmpl-inkling2", "object":"chat.completion", "model":model,
        "choices":[{"index":0,"message":reply.message,"finish_reason":reply.finish}],
        "usage":{"prompt_tokens":reply.prompt_tokens,"completion_tokens":reply.completion_tokens,
                 "total_tokens":reply.prompt_tokens + reply.completion_tokens}
    })
}

// Close-delimited JSON accepts leading whitespace. A heartbeat keeps desktop fetch
// alive during long native prefill, without inserting bytes into the model output.
fn start_keepalive_body(
    stream: &mut TcpStream,
    streaming: bool,
) -> std::io::Result<(std::sync::mpsc::Sender<()>, thread::JoinHandle<()>)> {
    let mut socket = stream.try_clone()?;
    let mime = if streaming {
        "text/event-stream"
    } else {
        "application/json"
    };
    let heartbeat: &[u8] = if streaming { b": prefill\n\n" } else { b"\n" };
    write!(
        stream,
        "HTTP/1.1 200 OK\r\nContent-Type: {mime}\r\nCache-Control: no-store\r\n{SECURITY_HEADERS}Connection: close\r\n\r\n"
    )?;
    let (stop, rx) = std::sync::mpsc::channel();
    let writer = thread::spawn(move || {
        while matches!(
            rx.recv_timeout(Duration::from_secs(15)),
            Err(std::sync::mpsc::RecvTimeoutError::Timeout)
        ) {
            if socket.write_all(heartbeat).is_err() {
                break;
            }
        }
    });
    Ok((stop, writer))
}

fn sse_completion(model: &str, reply: &native_inference::NativeReply) -> String {
    let mut delta = reply.message.clone();
    if let Some(calls) = delta.get_mut("tool_calls").and_then(Value::as_array_mut) {
        for (i, call) in calls.iter_mut().enumerate() {
            call["index"] = i.into();
        }
    }
    let chunk = json!({"id":"chatcmpl-inkling2","object":"chat.completion.chunk","model":model,
        "choices":[{"index":0,"delta":delta,"finish_reason":null}]});
    let chunk1 = format!("data: {chunk}\n\n");
    let chunk2 = format!(
        "data: {{\"id\":\"chatcmpl-inkling2\",\"object\":\"chat.completion.chunk\",\"model\":\"{}\",\"choices\":[{{\"index\":0,\"delta\":{{}},\"finish_reason\":\"{}\"}}]}}\n\n",
        json_escape(model),
        reply.finish
    );
    format!("{chunk1}{chunk2}data: [DONE]\n\n")
}

fn send_sse_completion(stream: &mut TcpStream, model: &str, reply: &native_inference::NativeReply) {
    let payload = sse_completion(model, reply);
    let resp = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream; charset=utf-8\r\nCache-Control: no-store\r\n{SECURITY_HEADERS}Content-Length: {}\r\nConnection: close\r\n\r\n{payload}",
        payload.len()
    );
    let _ = stream.write_all(resp.as_bytes());
    let _ = stream.flush();
}

/// Strip `<script>`, `<style>`, and HTML tags to produce safe, non-executable text for the hardened OS browser.
fn sanitize_html_to_text(raw: &str) -> (String, String) {
    let lower = raw.to_ascii_lowercase();
    let title = lower
        .find("<title")
        .and_then(|s| lower[s..].find('>').map(|g| s + g + 1))
        .and_then(|start| {
            lower[start..]
                .find("</title>")
                .map(|end| raw[start..start + end].trim().to_string())
        })
        .unwrap_or_else(|| "Untitled Document".to_string());

    let mut out = String::with_capacity(raw.len().min(32768));
    let mut in_tag = false;
    let mut skip_until: Option<&str> = None;
    let bytes = raw.as_bytes();
    let mut i = 0;
    while i < bytes.len() && out.len() < 32768 {
        if let Some(close_tag) = skip_until {
            if lower[i..].starts_with(close_tag) {
                i += close_tag.len();
                skip_until = None;
                continue;
            }
            i += raw[i..].chars().next().map_or(1, char::len_utf8);
            continue;
        }
        if lower[i..].starts_with("<script") {
            skip_until = Some("</script>");
            i += 7;
            continue;
        }
        if lower[i..].starts_with("<style") {
            skip_until = Some("</style>");
            i += 6;
            continue;
        }
        let ch = raw[i..].chars().next().unwrap_or(' ');
        if ch == '<' {
            in_tag = true;
            if !out.ends_with('\n') && !out.ends_with(' ') {
                out.push(' ');
            }
        } else if ch == '>' {
            in_tag = false;
        } else if !in_tag {
            out.push(ch);
        }
        i += ch.len_utf8();
    }
    let compact: String = out
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .collect::<Vec<_>>()
        .join("\n");
    (title, compact)
}

fn resolve_fs_root<'a>(ctx: &'a BridgeContext, root_id: &str) -> Option<(&'a Dir, bool)> {
    match root_id {
        "workspace" => ctx.preview_dir.as_ref().map(|dir| (dir, true)),
        "models" => ctx.model_listing.as_ref().map(|dir| (dir, false)),
        _ => None,
    }
}

fn usable_token(name: &str) -> Result<Option<String>, String> {
    match env::var(name) {
        Err(env::VarError::NotPresent) => Ok(None),
        Ok(value)
            if (24..=256).contains(&value.len()) && value.bytes().all(|b| b.is_ascii_graphic()) =>
        {
            Ok(Some(value))
        }
        _ => Err(format!(
            "{name} must contain 24–256 printable ASCII characters without spaces"
        )),
    }
}

fn upstream_addr(raw: &str) -> Option<SocketAddr> {
    let authority = raw
        .strip_prefix("http://")?
        .strip_suffix("/v1/chat/completions")?;
    let addr: SocketAddr = if let Some(port) = authority.strip_prefix("localhost:") {
        format!("127.0.0.1:{port}").parse().ok()?
    } else {
        authority.parse().ok()?
    };
    (addr.ip().is_loopback() && addr.port() != 0).then_some(addr)
}

fn forward_chat(ctx: &BridgeContext, body: &Value) -> Option<String> {
    use std::process::Stdio;
    let url = format!("http://{}/v1/chat/completions", ctx.upstream?);
    let mut request = body.clone();
    if let Some(model) = &ctx.upstream_model {
        request["model"] = json!(model);
    }
    let escape_config = |value: &str| value.replace('\\', "\\\\").replace('"', "\\\"");
    let mut config = format!("data-raw = \"{}\"\n", escape_config(&request.to_string()));
    if let Some(key) = &ctx.upstream_key {
        config.push_str(&format!(
            "header = \"Authorization: Bearer {}\"\n",
            escape_config(key)
        ));
    }
    let mut child = Command::new("curl")
        .args([
            "--disable",
            "--silent",
            "--fail",
            "--noproxy",
            "*",
            "--proto",
            "=http",
            "--connect-timeout",
            "5",
            "--max-time",
            "240",
            "--max-filesize",
            "1048576",
            "--header",
            "Content-Type: application/json",
            "--config",
            "-",
            "--",
            &url,
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .env_remove("INKLING2_BRIDGE_TOKEN")
        .env_remove("INKLING2_INFERENCE_TOKEN")
        .env_remove("INKLING2_UPSTREAM_API_KEY")
        .spawn()
        .ok()?;
    let written = child
        .stdin
        .take()
        .is_some_and(|mut input| input.write_all(config.as_bytes()).is_ok());
    let result = child.wait_with_output().ok()?;
    if !written || !result.status.success() || result.stdout.len() > 1_048_576 {
        return None;
    }
    String::from_utf8(result.stdout).ok()
}

fn ensure_resident(
    ctx: &BridgeContext,
    slot: &mut Option<ResidentEngine>,
    model: &StrataModelInfo,
    capacity: usize,
) -> Result<(), native_inference::InferenceError> {
    if slot.as_mut().is_some_and(|worker| {
        worker.model == model.path && worker.capacity >= capacity && worker.live()
    }) {
        return Ok(());
    }
    // Release the previous model before measuring the budget for its replacement.
    slot.take();
    native_inference::admit_model(model.trunk_bytes, read_free_vram(), capacity)?;
    let worker = ResidentEngine::start(&ctx.engine_bin, &model.path, capacity)?;
    if worker
        .exchange("warmup\n".to_owned(), native_inference::LOAD_TIMEOUT)?
        .trim_end()
        != "ready"
    {
        return Err((
            "503 Service Unavailable",
            "installed engine does not support resident token requests",
        ));
    }
    *slot = Some(worker);
    Ok(())
}

fn read_free_vram() -> Option<u64> {
    let out = Command::new("nvidia-smi")
        .args([
            "--id=0",
            "--query-gpu=memory.free",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    std::str::from_utf8(&out.stdout).ok()?.trim().parse().ok()
}

fn handle_client(mut stream: TcpStream, ctx: &BridgeContext) {
    let _ = stream.set_read_timeout(Some(Duration::from_secs(5)));
    let _ = stream.set_write_timeout(Some(Duration::from_secs(5)));
    let Ok(addr) = stream.local_addr() else {
        return;
    };
    let Ok(input) = stream.try_clone() else {
        return;
    };
    let req = match read_request(&mut BufReader::new(input), addr) {
        Ok(req) => req,
        Err((status, message)) => {
            send_error(&mut stream, status, message);
            return;
        }
    };
    let method = req.method.as_str();
    let path = req.path.as_str();
    let console_auth =
        ctx.local_console || bearer_matches(req.auth.as_deref(), ctx.bearer_token.as_deref());
    let inference_auth = bearer_matches(req.auth.as_deref(), ctx.inference_token.as_deref());
    let public = matches!(
        (method, path),
        (
            "GET",
            "/" | "/index.html" | "/console.js" | "/theme.js" | "/health" | "/favicon.ico"
        )
    );
    let inference = matches!(
        (method, path),
        ("GET", "/v1/models") | ("POST", "/v1/chat/completions")
    );
    if !(public || console_auth || inference && inference_auth) {
        send_error(
            &mut stream,
            "401 Unauthorized",
            "console or inference Bearer token required",
        );
        return;
    }
    let body: Value = if method == "POST" {
        match serde_json::from_slice::<Value>(&req.body) {
            Ok(body) if body.is_object() => body,
            _ => {
                send_error(&mut stream, "400 Bad Request", "expected a JSON object");
                return;
            }
        }
    } else {
        Value::Null
    };

    match (method, path) {
        ("GET", "/" | "/index.html") => send_asset(&mut stream, "text/html", CONSOLE_HTML),
        ("GET", "/console.js") => send_asset(&mut stream, "text/javascript", CONSOLE_JS),
        ("GET", "/theme.js") => send_asset(&mut stream, "text/javascript", THEME_JS),
        ("GET", "/favicon.ico") => send_error(&mut stream, "404 Not Found", "icon unavailable"),
        ("GET", "/v1/engine/status") => {
            let mut value = if let Ok(mut slot) = ctx.inference_slot.try_lock() {
                if let Some(worker) = slot.as_mut()
                    && worker.live()
                {
                    json!({"state":"ready", "pid":worker.pid(), "context_length":worker.capacity, "model":worker.model.file_stem().and_then(|v| v.to_str())})
                } else {
                    json!({"state":if ctx.resident { "idle" } else { "on-demand" }})
                }
            } else {
                json!({"state":"busy"})
            };
            value["platform"] = json!(std::env::consts::OS);
            send_json(&mut stream, "200 OK", &value.to_string());
        }
        ("POST", "/v1/engine/unload") if ctx.resident => {
            if let Ok(mut slot) = ctx.inference_slot.try_lock() {
                slot.take();
                send_json(&mut stream, "200 OK", r#"{"state":"idle"}"#);
            } else {
                send_error(&mut stream, "409 Conflict", "inference is in progress");
            }
        }
        ("GET", "/v1/session") => {
            send_json(
                &mut stream,
                "200 OK",
                &json!({"upstream": ctx.upstream.is_some(), "experimental_lora": ctx.experimental_lora,
                    "web_search": ctx.search_port.is_some()}).to_string(),
            );
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
                send_json(
                    &mut stream,
                    "200 OK",
                    r#"{"available":false,"cuda_target":"13.5-sm_120a"}"#,
                );
            }
        }
        ("GET", "/health") => {
            let mode = if ctx.local_console {
                "local"
            } else if ctx.bearer_token.is_some() {
                "locked"
            } else {
                "preview"
            };
            let health = json!({"status": "ok", "service": "inkling2-os-bridge", "console_mode": mode, "version": env!("CARGO_PKG_VERSION")});
            send_json(&mut stream, "200 OK", &health.to_string());
        }
        ("GET", "/v1/models") => {
            let mut models = discover_models(&ctx.models_dir);
            // A new desktop chat selects the first entry until the operator chooses one.
            if ctx.resident
                && let Some(preferred) = env::var_os("INKLING2_PRELOAD_MODEL")
                && let Some(index) = models
                    .iter()
                    .position(|m| m.path.file_stem() == Some(preferred.as_os_str()))
            {
                models[..=index].rotate_right(1);
            }
            let mut items = Vec::new();
            for m in models {
                let id = m
                    .path
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("inkling-2-mini");
                let manifest = crate::native_prompt::load_manifest(&m);
                let limits = native_inference::OutputLimits::for_build(
                    manifest.as_ref().and_then(|v| v["build"].as_str()),
                );
                items.push(json!({"id":id, "object":"model", "owned_by":"SC117-Inkling2-OS",
                    "context_length": manifest.as_ref().map(crate::native_prompt::manifest_limit), "bytes":m.file_bytes,
                    "page_alignment_bytes":m.page_size,
                    "default_output_tokens":limits.default, "max_output_tokens":limits.max}).to_string());
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
                send_json(
                    &mut stream,
                    "500 Internal Server Error",
                    r#"{"error":"memory lock poisoned"}"#,
                );
            }
        }
        ("POST", "/v1/memory/add") => {
            let text = extract_json_str(&body, "text").unwrap_or_default();
            let imp = extract_json_f32(&body, "importance").unwrap_or(0.8);
            if text.is_empty() {
                send_json(
                    &mut stream,
                    "400 Bad Request",
                    r#"{"error":"missing text field"}"#,
                );
                return;
            }
            if let Ok(mut m) = ctx.memory.lock() {
                let mut next = m.clone();
                let id = next.insert(text, imp);
                if next
                    .save_atomic(&ctx.state_dir.join("memory45.ikm"))
                    .is_err()
                {
                    send_error(
                        &mut stream,
                        "500 Internal Server Error",
                        "could not save memory",
                    );
                    return;
                }
                *m = next;
                let s = m.stats();
                let json = format!(
                    r#"{{"id":{id},"saved":true,"serialized_bytes":{},"budget_bytes":{}}}"#,
                    s.serialized_bytes, s.budget_bytes
                );
                send_json(&mut stream, "200 OK", &json);
            } else {
                send_json(
                    &mut stream,
                    "500 Internal Server Error",
                    r#"{"error":"memory lock poisoned"}"#,
                );
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
                send_json(
                    &mut stream,
                    "500 Internal Server Error",
                    r#"{"error":"memory lock poisoned"}"#,
                );
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
                send_json(
                    &mut stream,
                    "500 Internal Server Error",
                    r#"{"error":"lora lock poisoned"}"#,
                );
            }
        }
        ("POST", "/v1/lora/reinforce") => {
            if !ctx.experimental_lora {
                send_error(
                    &mut stream,
                    "403 Forbidden",
                    "experimental LoRA updates are disabled",
                );
                return;
            }
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
                let saved = l
                    .save_atomic(&ctx.state_dir.join("dynamic_lora.iklora"))
                    .is_ok();
                let s = l.stats();
                let json = format!(
                    r#"{{"steps":{},"norm_b":{:.4},"saved":{saved}}}"#,
                    s.evolution_steps, s.norm_b
                );
                send_json(&mut stream, "200 OK", &json);
            } else {
                send_json(
                    &mut stream,
                    "500 Internal Server Error",
                    r#"{"error":"lora lock poisoned"}"#,
                );
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
                send_json(
                    &mut stream,
                    "500 Internal Server Error",
                    r#"{"error":"lora lock poisoned"}"#,
                );
            }
        }
        ("POST", "/v1/voice/say") => {
            let text = extract_json_str(&body, "text").unwrap_or_default();
            if text.trim().is_empty() || text.chars().count() > 800 {
                send_error(
                    &mut stream,
                    "400 Bad Request",
                    "enter between 1 and 800 characters",
                );
                return;
            }
            let Ok(_voice) = ctx.voice_slot.try_lock() else {
                send_error(&mut stream, "409 Conflict", "speech synthesis is busy");
                return;
            };
            if std::fs::create_dir_all(&ctx.state_dir).is_err() {
                send_error(
                    &mut stream,
                    "500 Internal Server Error",
                    "speech output folder is unavailable",
                );
                return;
            }
            let wav = ctx.state_dir.join("bridge_voice.wav");
            match speak_text(&ctx.voice, &text, Some(&wav)) {
                Ok(_) if body["playback"].as_bool() == Some(true) => match std::fs::read(&wav) {
                    Ok(audio)
                        if audio.starts_with(b"RIFF") && audio.get(8..12) == Some(b"WAVE") =>
                    {
                        let _ = write!(
                            stream,
                            "HTTP/1.1 200 OK\r\nContent-Type: audio/wav\r\nCache-Control: no-store\r\n{SECURITY_HEADERS}Content-Length: {}\r\nConnection: close\r\n\r\n",
                            audio.len()
                        );
                        let _ = stream.write_all(&audio);
                    }
                    _ => send_error(
                        &mut stream,
                        "500 Internal Server Error",
                        "speech backend returned no WAV audio",
                    ),
                },
                Ok(msg) => send_json(
                    &mut stream,
                    "200 OK",
                    &format!(r#"{{"status":"ok","detail":"{}"}}"#, json_escape(&msg)),
                ),
                Err(e) => send_json(
                    &mut stream,
                    "500 Internal Server Error",
                    &format!(r#"{{"error":"{}"}}"#, json_escape(&e)),
                ),
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
                    let line = combined
                        .lines()
                        .rev()
                        .find(|l| !l.trim().is_empty())
                        .unwrap_or("ok");
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
                Err(e) => send_json(
                    &mut stream,
                    "500 Internal Server Error",
                    &format!(r#"{{"error":"{}"}}"#, json_escape(&e.to_string())),
                ),
            }
        }
        ("POST", "/v1/net/search") => {
            let Some(port) = ctx.search_port else {
                send_error(
                    &mut stream,
                    "503 Service Unavailable",
                    "RustXNG is not configured",
                );
                return;
            };
            let Some(query) = body["query"]
                .as_str()
                .filter(|query| crate::web_search::valid_query(query))
            else {
                send_error(
                    &mut stream,
                    "400 Bad Request",
                    "enter a search query of up to 512 bytes",
                );
                return;
            };
            match crate::web_search::search(port, query) {
                Ok(results) => send_json(&mut stream, "200 OK", &results.to_string()),
                Err(error) => send_error(&mut stream, "502 Bad Gateway", error),
            }
        }
        ("POST", "/v1/net/fetch") => {
            let url = extract_json_str(&body, "url").unwrap_or_default();
            match fetch_public_https(&url) {
                Ok(page) => {
                    let raw = String::from_utf8_lossy(&page.bytes);
                    let (title, text, html) = if page.html {
                        let (title, text) = sanitize_html_to_text(&raw);
                        (title, text, Some(render_document(&page.url, &raw)))
                    } else {
                        (String::new(), raw.into_owned(), None)
                    };
                    let result = json!({"url": page.url, "tls": "1.2+", "bytes": page.bytes.len(), "title": title, "text": text, "html": html});
                    send_json(&mut stream, "200 OK", &result.to_string());
                }
                Err((blocked, error)) => send_error(
                    &mut stream,
                    if blocked {
                        "403 Forbidden"
                    } else {
                        "502 Bad Gateway"
                    },
                    error,
                ),
            }
        }
        ("POST", "/v1/fs/list") => {
            let root_id =
                extract_json_str(&body, "root").unwrap_or_else(|| "workspace".to_string());
            let rel = extract_json_str(&body, "path").unwrap_or_default();
            let Some((root_path, previews)) = resolve_fs_root(ctx, &root_id) else {
                send_json(
                    &mut stream,
                    "400 Bad Request",
                    r#"{"error":"requested directory was not shared"}"#,
                );
                return;
            };
            match list_jailed_dir(root_path, &rel, previews) {
                Ok(entries) => {
                    let items: Vec<String> = entries
                        .iter()
                        .map(|e| {
                            format!(
                                r#"{{"name":"{}","is_dir":{},"size":{},"blocked":{}}}"#,
                                json_escape(&e.name),
                                e.is_dir,
                                e.size,
                                e.blocked
                            )
                        })
                        .collect();
                    let json = format!(
                        r#"{{"root":"{}","path":"{}","entries":[{}]}}"#,
                        json_escape(&root_id),
                        json_escape(&rel),
                        items.join(",")
                    );
                    send_json(&mut stream, "200 OK", &json);
                }
                Err(e) => {
                    send_json(
                        &mut stream,
                        "403 Forbidden",
                        &format!(r#"{{"error":"{}"}}"#, json_escape(e)),
                    );
                }
            }
        }
        ("POST", "/v1/fs/read") => {
            let root_id =
                extract_json_str(&body, "root").unwrap_or_else(|| "workspace".to_string());
            let rel = extract_json_str(&body, "path").unwrap_or_default();
            let Some((root_path, previews)) = resolve_fs_root(ctx, &root_id) else {
                send_json(
                    &mut stream,
                    "400 Bad Request",
                    r#"{"error":"unknown root"}"#,
                );
                return;
            };
            if !previews {
                send_error(
                    &mut stream,
                    "403 Forbidden",
                    "this root allows listing only",
                );
                return;
            }
            match read_jailed_file(root_path, &rel) {
                Ok(r) => {
                    let json = format!(
                        r#"{{"root":"{}","path":"{}","size":{},"truncated":{},"fnv64":"{}","content":"{}"}}"#,
                        json_escape(&root_id),
                        json_escape(&r.rel_path),
                        r.size,
                        r.truncated,
                        r.fnv64_hex,
                        json_escape(&r.content)
                    );
                    send_json(&mut stream, "200 OK", &json);
                }
                Err(e) => {
                    send_json(
                        &mut stream,
                        "403 Forbidden",
                        &format!(r#"{{"error":"{}"}}"#, json_escape(e)),
                    );
                }
            }
        }
        ("POST", "/v1/chat/completions") => {
            let streaming = body.get("stream").and_then(Value::as_bool).unwrap_or(false);
            let Ok(mut slot) = ctx.inference_slot.try_lock() else {
                send_error(
                    &mut stream,
                    "429 Too Many Requests",
                    "one inference request is already active",
                );
                return;
            };
            // The Strata credential always routes to the dedicated native engine.
            if console_auth && !inference_auth && ctx.upstream.is_some() {
                if let Some(reply) = forward_chat(ctx, &body) {
                    let mime = if streaming {
                        "text/event-stream"
                    } else {
                        "application/json"
                    };
                    send_asset(&mut stream, mime, &reply);
                } else {
                    send_error(
                        &mut stream,
                        "502 Bad Gateway",
                        "configured upstream unavailable",
                    );
                }
                return;
            }
            let model_id = body
                .get("model")
                .and_then(Value::as_str)
                .unwrap_or_default();
            let mut models = discover_models(&ctx.models_dir)
                .into_iter()
                .filter(|model| {
                    model.path.file_stem().and_then(|name| name.to_str()) == Some(model_id)
                });
            let Some(model) = models.next().filter(|_| models.next().is_none()) else {
                send_error(&mut stream, "404 Not Found", "model absent or ambiguous");
                return;
            };
            let mut desktop_body = None;
            let result = (|| {
                let manifest = crate::native_prompt::load_manifest(&model)
                    .ok_or(("503 Service Unavailable", "model manifest unavailable"))?;
                let limits = native_inference::OutputLimits::for_build(manifest["build"].as_str());
                let (messages, max) = native_inference::prompt_and_limit(&body, limits)?;
                let tools = crate::native_tools::Tools::from_request(&body)?;
                let prompt =
                    crate::native_prompt::prepare(&model, &manifest, messages, max, tools)?;
                let mut capacity = native_inference::context_capacity(
                    &body,
                    prompt.input_ids.len() + max as usize,
                )?;
                if body.get("context_length").is_some() && capacity > prompt.model_limit {
                    return Err((
                        "400 Bad Request",
                        "requested context exceeds the model manifest limit",
                    ));
                }
                capacity = capacity.min(prompt.model_limit);
                if ctx.resident {
                    ensure_resident(ctx, &mut slot, &model, capacity)?;
                } else {
                    native_inference::admit_model(model.trunk_bytes, read_free_vram(), capacity)?;
                }
                if prompt.input_ids.len() > 512
                    && (streaming || body["desktop_keepalive"].as_bool() == Some(true))
                {
                    desktop_body =
                        Some(start_keepalive_body(&mut stream, streaming).map_err(|_| {
                            ("503 Service Unavailable", "client connection closed")
                        })?);
                }
                let result = native_inference::generate(
                    &ctx.engine_bin,
                    &model,
                    &prompt,
                    max,
                    capacity,
                    slot.as_mut(),
                );
                if result
                    .as_ref()
                    .is_err_and(|(code, _)| code.starts_with('5'))
                {
                    slot.take();
                }
                result
            })();
            if let Some((stop, writer)) = desktop_body {
                drop(stop);
                let _ = writer.join();
                if streaming {
                    let payload = match result {
                        Ok(reply) => sse_completion(model_id, &reply),
                        Err((status, message)) => {
                            let event = json!({"error":{"code":status,"message":message}});
                            format!("data: {event}\n\ndata: [DONE]\n\n")
                        }
                    };
                    let _ = stream.write_all(payload.as_bytes());
                    return;
                }
                let value = match result {
                    Ok(reply) => completion_json(model_id, reply),
                    Err((status, message)) => json!({"status":status,"error":message}),
                };
                let _ = write!(stream, "{value}");
                return;
            }
            match result {
                Ok(reply) if streaming => send_sse_completion(&mut stream, model_id, &reply),
                Ok(reply) => send_json(
                    &mut stream,
                    "200 OK",
                    &completion_json(model_id, reply).to_string(),
                ),
                Err((status, message)) => send_error(&mut stream, status, message),
            }
        }
        _ => {
            if let Some((status, json_body)) = handle_neon_endpoint(
                method,
                path,
                &body,
                &ctx.neon,
                &ctx.voice,
                &ctx.lora,
                &ctx.state_dir,
            ) {
                send_json(&mut stream, status, &json_body);
            } else {
                send_json(
                    &mut stream,
                    "404 Not Found",
                    r#"{"error":"endpoint not found"}"#,
                );
            }
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

    let memory_path = state_dir.join("memory45.ikm");
    let memory = if memory_path.try_exists().map_err(|e| e.to_string())?
        || state_dir
            .join("memory45.ikm.bak1")
            .try_exists()
            .map_err(|e| e.to_string())?
    {
        Memory45Store::load_with_recovery(
            &memory_path,
            inkling2_companion::memory45::MAX_MEMORY_BYTES,
        )?
    } else {
        Memory45Store::default()
    };
    let adapter_path = state_dir.join("dynamic_lora.iklora");
    let lora = if adapter_path.try_exists().map_err(|e| e.to_string())? {
        DynamicLoraAdapter::load(&adapter_path)?
    } else {
        DynamicLoraAdapter::new(LoraConfig::default())
    };
    let bearer_token = usable_token("INKLING2_BRIDGE_TOKEN")?;
    let inference_token = usable_token("INKLING2_INFERENCE_TOKEN")?;
    let local_console = env::var("INKLING2_LOCAL_CONSOLE").is_ok_and(|v| v == "1");
    if local_console && bearer_token.is_some() {
        return Err("local console mode cannot be combined with a console token".into());
    }
    let search_port = match env::var("INKLING2_RUSTXNG_PORT") {
        Ok(value) => Some(
            value
                .parse::<u16>()
                .ok()
                .filter(|port| *port >= 1024 && *port != addr.port())
                .ok_or("invalid RustXNG port")?,
        ),
        Err(env::VarError::NotPresent) => None,
        Err(_) => return Err("invalid RustXNG port".into()),
    };
    if bearer_token.is_some() && bearer_token == inference_token {
        return Err("console and inference tokens must be distinct".into());
    }
    let upstream = match env::var("INKLING2_UPSTREAM_OPENAI") {
        Ok(raw) => Some(
            upstream_addr(&raw)
                .filter(|up| *up != addr)
                .ok_or("upstream must be a different loopback socket with /v1/chat/completions")?,
        ),
        Err(env::VarError::NotPresent) => None,
        Err(_) => return Err("invalid upstream address".into()),
    };
    let upstream_key = env::var("INKLING2_UPSTREAM_API_KEY")
        .ok()
        .filter(|key| !key.is_empty());
    if upstream_key
        .as_ref()
        .is_some_and(|key| !key.bytes().all(|b| b.is_ascii_graphic()))
    {
        return Err("upstream API key contains invalid characters".into());
    }
    let preview_dir = env::var_os("INKLING2_PREVIEW_DIR")
        .map(|p| {
            Dir::open_ambient_dir(p, ambient_authority())
                .map_err(|_| "preview directory unavailable")
        })
        .transpose()?;
    let model_listing = Dir::open_ambient_dir(&models_dir, ambient_authority()).ok();

    let neon = NeonRuntime::load(&state_dir)?;
    let ctx = Arc::new(BridgeContext {
        engine_bin,
        models_dir,
        state_dir,
        memory: Mutex::new(memory),
        lora: Mutex::new(lora),
        voice: VoicePersona::default(),
        voice_slot: Mutex::new(()),
        bearer_token,
        inference_token,
        local_console,
        search_port,
        preview_dir,
        model_listing,
        active_connections: AtomicUsize::new(0),
        inference_slot: Mutex::new(None),
        resident: env::var("INKLING2_RESIDENT").is_ok_and(|v| v == "1"),
        experimental_lora: env::var("INKLING2_EXPERIMENTAL_LORA").is_ok_and(|v| v == "1"),
        upstream,
        upstream_model: env::var("INKLING2_UPSTREAM_MODEL").ok(),
        upstream_key,
        neon,
    });

    let listener = TcpListener::bind(addr).map_err(|e| format!("bind {addr} failed: {e}"))?;
    let mut stderr = std::io::stderr().lock();
    let _ = writeln!(
        stderr,
        "[Inkling2-Bridge] Listening on http://{addr} (loopback-only; Muse.ai VM tunnel -> 127.0.0.1:18080)"
    );
    drop(stderr);

    struct Active<'a>(&'a AtomicUsize);
    impl Drop for Active<'_> {
        fn drop(&mut self) {
            self.0.fetch_sub(1, Ordering::Relaxed);
        }
    }
    if ctx.resident
        && ctx.upstream.is_none()
        && let Ok(id) = env::var("INKLING2_PRELOAD_MODEL")
    {
        let state = Arc::clone(&ctx);
        thread::spawn(move || {
            let models = discover_models(&state.models_dir);
            let mut selected = models
                .iter()
                .filter(|m| m.path.file_stem().and_then(|v| v.to_str()) == Some(&id));
            if let Some(model) = selected.next().filter(|_| selected.next().is_none())
                && let Ok(mut slot) = state.inference_slot.lock()
                && slot.is_none()
                && let Err((_, reason)) = ensure_resident(&state, &mut slot, model, 4096)
            {
                eprintln!("Model preload: {reason}");
            }
        });
    }
    for stream in listener.incoming().flatten() {
        if ctx.active_connections.load(Ordering::Relaxed) >= MAX_CONNECTIONS {
            continue;
        }
        ctx.active_connections.fetch_add(1, Ordering::Relaxed);
        let c = Arc::clone(&ctx);
        thread::spawn(move || {
            let _active = Active(&c.active_connections);
            handle_client(stream, &c);
        });
    }
    Ok(())
}
