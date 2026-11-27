//! Adapter for the shipped Windows engine's stdin/stdout protocol.

use std::io::{BufRead, BufReader, Read, Write};
use std::path::Path;
use std::process::{Command, Stdio};
use std::sync::mpsc;
use std::thread;
use std::time::Duration;

use inkling2_resource::{VramRequest, admit_vram};
use serde_json::Value;

use crate::strata_catalog::StrataModelInfo;

pub const MAX_OUTPUT_TOKENS: u64 = 128;
const DEFAULT_OUTPUT_TOKENS: u64 = 32;
const MAX_PROMPT_BYTES: usize = 1536;
const MAX_OUTPUT_BYTES: u64 = 64 * 1024;
const MIB: u64 = 1024 * 1024;

pub type InferenceError = (&'static str, &'static str);
const UNAVAILABLE: InferenceError = (
    "503 Service Unavailable",
    "native engine did not produce a response",
);
const BAD_REQUEST: InferenceError = (
    "400 Bad Request",
    "expected text-only system, user or assistant messages",
);

pub fn prompt_and_limit(body: &Value) -> Result<(String, u64), InferenceError> {
    if body
        .get("tools")
        .is_some_and(|v| !v.is_null() && v.as_array().is_none_or(|a| !a.is_empty()))
    {
        return Err((
            "400 Bad Request",
            "tool calling is unavailable for this engine",
        ));
    }
    let messages = body
        .get("messages")
        .and_then(Value::as_array)
        .ok_or(BAD_REQUEST)?;
    if messages.is_empty() || messages.len() > 32 {
        return Err(BAD_REQUEST);
    }
    let mut prompt = String::new();
    for message in messages {
        let role = message
            .get("role")
            .and_then(Value::as_str)
            .filter(|role| matches!(*role, "system" | "user" | "assistant"))
            .ok_or(BAD_REQUEST)?;
        let content = message
            .get("content")
            .and_then(Value::as_str)
            .ok_or(BAD_REQUEST)?;
        if content
            .chars()
            .any(|c| c.is_control() && !matches!(c, '\r' | '\n' | '\t'))
        {
            return Err(BAD_REQUEST);
        }
        if !prompt.is_empty() {
            prompt.push_str(" | ");
        }
        prompt.push_str(role);
        prompt.push_str(": ");
        // The shipped daemon receives one request per line.
        prompt.extend(
            content
                .chars()
                .map(|c| if c.is_control() { ' ' } else { c }),
        );
    }
    prompt.push_str(" | assistant: ");
    if prompt.len() > MAX_PROMPT_BYTES {
        return Err((
            "400 Bad Request",
            "input exceeds the native adapter's 1536-byte limit",
        ));
    }
    let max = match body
        .get("max_completion_tokens")
        .or_else(|| body.get("max_tokens"))
    {
        None => DEFAULT_OUTPUT_TOKENS,
        Some(value) => value
            .as_u64()
            .filter(|v| (1..=MAX_OUTPUT_TOKENS).contains(v))
            .ok_or((
                "400 Bad Request",
                "native output limit must be between 1 and 128 tokens",
            ))?,
    };
    Ok((prompt, max))
}

pub fn admit_model(bytes: u64, free_mib: Option<u64>) -> Result<(), InferenceError> {
    // The native layout exposes no exact memory dry run. Admit conservatively using
    // the whole file and the upstream Windows reserve plus provisional working margins.
    admit_vram(
        free_mib.and_then(|free| free.checked_mul(MIB)),
        VramRequest {
            weights: bytes,
            experts: 0, // already included in the file
            kv_cache: 256 * MIB,
            activations: 128 * MIB,
            cuda_workspace: 256 * MIB,
            runtime: 128 * MIB,
            reserve: 1536 * MIB,
        },
    )
    .map(|_| ())
    .map_err(|_| {
        (
            "503 Service Unavailable",
            "insufficient measured GPU headroom for the selected native model",
        )
    })
}

pub fn generate(
    engine: &Path,
    model: &StrataModelInfo,
    prompt: &str,
    max: u64,
) -> Result<String, InferenceError> {
    if !engine.is_file() || !model.path.with_file_name("tokenizer.json").is_file() {
        return Err((
            "503 Service Unavailable",
            "native executable or model tokenizer is missing",
        ));
    }
    let mut child = Command::new(engine)
        .arg("daemon")
        .arg(&model.path)
        .env_remove("INKLING2_BRIDGE_TOKEN")
        .env_remove("INKLING2_INFERENCE_TOKEN")
        .env_remove("INKLING2_UPSTREAM_API_KEY")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .map_err(|_| UNAVAILABLE)?;
    let io = child.stdin.take().zip(child.stdout.take());
    let Some((mut input, output)) = io else {
        let _ = child.kill();
        let _ = child.wait();
        return Err(UNAVAILABLE);
    };
    let (tx, rx) = mpsc::sync_channel(1);
    let request = format!("{max}\t{prompt}\n");
    let reader = thread::spawn(move || {
        let result = (|| {
            let mut output = BufReader::new(output).take(MAX_OUTPUT_BYTES);
            let mut line = String::new();
            output.read_line(&mut line).ok()?;
            if line.trim_end() != "READY" {
                return None;
            }
            input.write_all(request.as_bytes()).ok()?;
            drop(input);
            line.clear();
            output.read_line(&mut line).ok()?;
            let answer = line.trim_end_matches(['\r', '\n']);
            (!answer.trim().is_empty() && line.ends_with('\n')).then(|| answer.to_owned())
        })();
        let _ = tx.send(result);
    });
    let result = rx.recv_timeout(Duration::from_secs(240));
    // Always reap the owned process, including when it stops responding.
    let _ = child.kill();
    let _ = child.wait();
    let _ = reader.join();
    result.ok().flatten().ok_or(UNAVAILABLE)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn validates_roles_and_limits_without_silent_truncation() {
        let body = json!({
            "messages": [
                {"role": "system", "content": "Be concise"},
                {"role": "user", "content": "日本語\n二行目"},
            ],
            "max_tokens": 12
        });
        let result = prompt_and_limit(&body);
        assert_eq!(
            result,
            Ok((
                "system: Be concise | user: 日本語 二行目 | assistant: ".into(),
                12
            ))
        );
        assert!(
            prompt_and_limit(&json!({"messages": body["messages"], "max_tokens": 129})).is_err()
        );
        assert!(admit_model(10 * 1024 * MIB, Some(3481)).is_err());
        assert!(admit_model(10 * 1024 * MIB, Some(15000)).is_ok());
    }
}
