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
const MAX_PROMPT_BYTES: usize = 16 * 1024 * 1024;
const MAX_OUTPUT_BYTES: u64 = 64 * 1024;
const MIB: u64 = 1024 * 1024;
// Each request loads the model. The reported v0.1.0 cold load alone took about 390 seconds.
pub const LOAD_TIMEOUT: Duration = Duration::from_secs(600);

pub fn request_timeout(input_tokens: usize) -> Duration {
    LOAD_TIMEOUT + Duration::from_secs((input_tokens.saturating_sub(512) as u64) * 4)
}

pub fn context_capacity(body: &Value, required: usize) -> Result<usize, InferenceError> {
    let max = native_prompt::MAX_CONTEXT_IDS;
    if required > max {
        return Err(("400 Bad Request", "prompt and output exceed 131072 tokens"));
    }
    match body.get("context_length") {
        None => Ok(required.max(4096).next_power_of_two()),
        Some(value) => value
            .as_u64()
            .filter(|n| (1..=max as u64).contains(n) && *n >= required as u64)
            .map(|n| n as usize)
            .ok_or((
                "400 Bad Request",
                "context_length must fit the prompt and output and be at most 131072",
            )),
    }
}

pub type InferenceError = (&'static str, &'static str);
const UNAVAILABLE: InferenceError = (
    "503 Service Unavailable",
    "native engine did not produce a response",
);
const BAD_REQUEST: InferenceError = (
    "400 Bad Request",
    "expected a text conversation with valid system, user, assistant or tool messages",
);

pub fn prompt_and_limit(body: &Value) -> Result<(&[Value], u64), InferenceError> {
    let messages = body
        .get("messages")
        .and_then(Value::as_array)
        .ok_or(BAD_REQUEST)?;
    if messages.is_empty() || messages.len() > 1024 {
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
            .filter(|role| matches!(*role, "system" | "user" | "assistant" | "tool"))
            .ok_or(BAD_REQUEST)?;
        let content = message["content"].as_str().unwrap_or_default();
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
            "input exceeds the 16 MiB chat transport limit",
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

pub fn admit_model(
    trunk_bytes: u64,
    free_mib: Option<u64>,
    capacity: usize,
) -> Result<(), InferenceError> {
    // Expert weights spill to host RAM and storage. Only the trunk and working memory must fit.
    admit_vram(
        free_mib.and_then(|free| free.checked_mul(MIB)),
        VramRequest {
            weights: trunk_bytes,
            experts: 128 * MIB, // Two top-8 device staging buffers of up to 8 MiB per expert.
            // SC117 has eight global and forty sliding logical visits. Each head stores
            // 128 int8 values and one f32 scale per position for both K and V.
            kv_cache: (8 * capacity as u64 + 40 * capacity.min(1024) as u64) * 8 * 2 * (128 + 4),
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
    pub message: Value,
    pub finish: &'static str,
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
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
    let message = prompt.tools.decode(&prompt.tokenizer, &ids, finish)?;
    let finish = if message.get("tool_calls").is_some() {
        "tool_calls"
    } else {
        finish
    };
    Some(NativeReply {
        message,
        finish,
        prompt_tokens: prompt.input_ids.len(),
        completion_tokens: ids.len(),
    })
}

pub fn generate(
    engine: &Path,
    model: &StrataModelInfo,
    prompt: &NativePrompt,
    max: u64,
    capacity: usize,
    resident: Option<&mut crate::resident_engine::ResidentEngine>,
) -> Result<NativeReply, InferenceError> {
    if !engine.is_file() {
        return Err(("503 Service Unavailable", "native executable is missing"));
    }
    let timeout = request_timeout(prompt.input_ids.len());
    let csv = |ids: &[u32]| ids.iter().map(u32::to_string).collect::<Vec<_>>().join(",");
    let request = format!(
        "{max}\t{}\t{}\n",
        csv(&prompt.stop_ids),
        csv(&prompt.input_ids)
    );
    if let Some(worker) = resident {
        let line = worker.exchange(request, timeout)?;
        return read_reply(&line, prompt, max).ok_or((
            "502 Bad Gateway",
            "the model returned an invalid output token sequence",
        ));
    }
    let mut child = Command::new(engine)
        .arg("generate-token-ids")
        .arg(&model.path)
        .arg("--context")
        .arg(capacity.to_string())
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
    let result = rx.recv_timeout(timeout);
    // Always reap the owned process, including when it stops responding.
    let _ = child.kill();
    let _ = child.wait();
    let _ = reader.join();
    result
        .map_err(|error| match error {
            mpsc::RecvTimeoutError::Timeout => (
                "504 Gateway Timeout",
                "native model loading and generation exceeded the request deadline",
            ),
            mpsc::RecvTimeoutError::Disconnected => UNAVAILABLE,
        })?
        .and_then(|line| read_reply(&line, prompt, max))
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
        assert!(admit_model(10 * 1024 * MIB, Some(3481), 4096).is_err());
        assert!(admit_model(10 * 1024 * MIB, Some(15000), 4096).is_ok());
        assert_eq!(context_capacity(&json!({}), 131_072), Ok(131_072));
        assert!(context_capacity(&json!({}), 131_073).is_err());
        assert!(context_capacity(&json!({"context_length":4096}), 4097).is_err());
    }
}
