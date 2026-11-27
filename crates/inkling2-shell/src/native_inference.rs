//! Adapter for the shipped Windows engine's stdin/stdout protocol.

use std::io::{BufRead, BufReader, Read, Write};
use std::path::Path;
use std::process::{Command, Stdio};
use std::sync::mpsc;
use std::thread;
use std::time::Duration;

use inkling2_resource::{VramRequest, admit_vram};
use serde_json::Value;

use crate::native_prompt::{self, NativePrompt};
use crate::strata_catalog::StrataModelInfo;

pub const MAX_OUTPUT_TOKENS: u64 = 128;
const DEFAULT_OUTPUT_TOKENS: u64 = 32;
const MAX_PROMPT_BYTES: usize = 1536;
const MAX_OUTPUT_BYTES: u64 = 64 * 1024;
const MIB: u64 = 1024 * 1024;
// Each request loads the model. The reported v0.1.0 cold load alone took about 390 seconds.
const NATIVE_REQUEST_TIMEOUT: Duration = Duration::from_secs(600);

pub type InferenceError = (&'static str, &'static str);
const UNAVAILABLE: InferenceError = (
    "503 Service Unavailable",
    "native engine did not produce a response",
);
const BAD_REQUEST: InferenceError = (
    "400 Bad Request",
    "expected text-only system, user or assistant messages",
);

pub fn prompt_and_limit(body: &Value) -> Result<(&[Value], u64), InferenceError> {
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
    if body
        .get("reasoning_effort")
        .is_some_and(|effort| !effort.is_null() && effort.as_str() != Some("none"))
    {
        return Err((
            "400 Bad Request",
            "native inference currently supports reasoning_effort none",
        ));
    }
    let mut content_bytes = 0;
    for message in messages {
        message
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
        content_bytes += content.len();
    }
    if content_bytes > MAX_PROMPT_BYTES {
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
    Ok((messages, max))
}

pub fn admit_model(trunk_bytes: u64, free_mib: Option<u64>) -> Result<(), InferenceError> {
    // Expert weights spill to host RAM and storage. Only the trunk and working memory must fit.
    admit_vram(
        free_mib.and_then(|free| free.checked_mul(MIB)),
        VramRequest {
            weights: trunk_bytes,
            experts: 128 * MIB, // Two top-8 device staging buffers of up to 8 MiB per expert.
            kv_cache: 396 * MIB, // 48 slots, 4096 positions, eight 128-wide K8V8 heads plus scales.
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

pub struct NativeReply {
    pub text: String,
    pub finish: &'static str,
}

fn read_reply(line: &str, prompt: &NativePrompt, max: u64) -> Option<NativeReply> {
    let (finish, ids) = line
        .strip_suffix('\n')?
        .trim_end_matches('\r')
        .split_once('\t')?;
    let finish = match finish {
        "stop" => "stop",
        "length" => "length",
        _ => return None,
    };
    let ids = if ids.is_empty() {
        Vec::new()
    } else {
        ids.split(',')
            .map(str::parse::<u32>)
            .collect::<Result<Vec<_>, _>>()
            .ok()?
    };
    if ids.len() as u64 > max || finish == "length" && ids.len() as u64 != max {
        return None;
    }
    let added = prompt.tokenizer.get_added_tokens_decoder();
    if ids.iter().any(|id| {
        prompt.tokenizer.id_to_token(*id).is_none()
            || added.get(id).is_some_and(|token| token.special)
    }) {
        return None;
    }
    let text = prompt.tokenizer.decode(&ids, false).ok()?;
    Some(NativeReply { text, finish })
}

pub fn generate(
    engine: &Path,
    model: &StrataModelInfo,
    messages: &[Value],
    max: u64,
) -> Result<NativeReply, InferenceError> {
    if !engine.is_file() {
        return Err(("503 Service Unavailable", "native executable is missing"));
    }
    let prompt = native_prompt::prepare(model, messages, max)?;
    let csv = |ids: &[u32]| ids.iter().map(u32::to_string).collect::<Vec<_>>().join(",");
    let request = format!(
        "{max}\t{}\t{}\n",
        csv(&prompt.stop_ids),
        csv(&prompt.input_ids)
    );
    let mut child = Command::new(engine)
        .arg("generate-token-ids")
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
    let reader = thread::spawn(move || {
        let result = (|| {
            input.write_all(request.as_bytes()).ok()?;
            drop(input);
            let mut output = BufReader::new(output).take(MAX_OUTPUT_BYTES);
            let mut line = String::new();
            output.read_line(&mut line).ok()?;
            Some(line)
        })();
        let _ = tx.send(result);
    });
    let result = rx.recv_timeout(NATIVE_REQUEST_TIMEOUT);
    // Always reap the owned process, including when it stops responding.
    let _ = child.kill();
    let _ = child.wait();
    let _ = reader.join();
    result
        .map_err(|error| match error {
            mpsc::RecvTimeoutError::Timeout => (
                "504 Gateway Timeout",
                "native model loading and generation exceeded the 600-second deadline",
            ),
            mpsc::RecvTimeoutError::Disconnected => UNAVAILABLE,
        })?
        .and_then(|line| read_reply(&line, &prompt, max))
        .ok_or(UNAVAILABLE)
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
            body["messages"]
                .as_array()
                .map(|messages| (messages.as_slice(), 12))
                .ok_or(BAD_REQUEST)
        );
        assert!(
            prompt_and_limit(&json!({"messages": body["messages"], "max_tokens": 129})).is_err()
        );
        assert!(admit_model(10 * 1024 * MIB, Some(3481)).is_err());
        assert!(admit_model(10 * 1024 * MIB, Some(15000)).is_ok());
    }
}
