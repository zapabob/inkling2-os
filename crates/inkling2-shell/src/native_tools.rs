//! OpenAI function declarations and conversation history for the SC117 template.
use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use serde_json::{Value, json};
use tokenizers::Tokenizer;

use crate::native_inference::InferenceError;

const BAD_TOOLS: InferenceError = (
    "400 Bad Request",
    "invalid function declaration or tool_choice",
);
const BAD_TURN: InferenceError = (
    "400 Bad Request",
    "invalid text message or function-call history",
);
const INVOKE: &str = "<|content_invoke_tool_json|>";
const TEXT: &str = "<|content_text|>";
pub const MODEL: &str = "<|message_model|>";
pub const EMPTY_THINK: &str = "<|content_thinking|><|end_message|>";

fn valid_name(name: &str) -> bool {
    (1..=64).contains(&name.len())
        && name
            .bytes()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, b'_' | b'-'))
}

#[derive(Default)]
pub struct Tools {
    pub declarations: Vec<Value>,
    names: HashSet<String>,
    forced: Option<String>,
    required: bool,
}

impl Tools {
    pub fn from_request(body: &Value) -> Result<Self, InferenceError> {
        let mut out = Self::default();
        if !body["tools"].is_null() {
            let defs = body["tools"]
                .as_array()
                .filter(|a| a.len() <= 128)
                .ok_or(BAD_TOOLS)?;
            for def in defs {
                let f = &def["function"];
                let name = f["name"]
                    .as_str()
                    .filter(|n| valid_name(n))
                    .ok_or(BAD_TOOLS)?;
                if def["type"] != "function"
                    || !out.names.insert(name.to_owned())
                    || !(f["description"].is_null() || f["description"].is_string())
                    || !(f["parameters"].is_null() || f["parameters"].is_object())
                {
                    return Err(BAD_TOOLS);
                }
            }
            out.declarations.clone_from(defs);
        }
        match &body["tool_choice"] {
            Value::Null => (),
            Value::String(choice) if choice == "auto" => (),
            Value::String(choice) if choice == "none" => return Ok(Self::default()),
            Value::String(choice) if choice == "required" && !out.names.is_empty() => {
                out.required = true;
                if out.names.len() == 1 {
                    out.forced = out.names.iter().next().cloned();
                }
            }
            choice if choice["type"] == "function" => {
                let name = choice["function"]["name"]
                    .as_str()
                    .filter(|n| out.names.contains(*n))
                    .ok_or(BAD_TOOLS)?;
                out.forced = Some(name.to_owned());
                out.required = true;
            }
            _ => return Err(BAD_TOOLS),
        }
        Ok(out)
    }

    pub fn assistant_prefix(&self) -> String {
        if let Some(name) = &self.forced {
            return format!("{EMPTY_THINK}{MODEL}{name}{INVOKE}");
        }
        if self.names.is_empty() {
            TEXT.to_owned()
        } else {
            format!("{EMPTY_THINK}{MODEL}")
        }
    }

    /// Only the channel header is constrained. Its alternatives are scored by
    /// the CUDA model; text and JSON arguments are generated normally.
    pub fn channel_prefixes(&self, tokenizer: &Tokenizer) -> Option<Vec<Vec<u32>>> {
        let mut choices = Vec::new();
        if self.forced.is_some() || self.names.is_empty() {
            return Some(choices);
        }
        if !self.required {
            choices.push(vec![tokenizer.token_to_id(TEXT)?]);
        }
        let invoke = tokenizer.token_to_id(INVOKE)?;
        for def in &self.declarations {
            let name = def["function"]["name"].as_str()?;
            let mut prefix = tokenizer.encode(name, false).ok()?.get_ids().to_vec();
            if prefix.is_empty() || prefix.len() > 64 {
                return None;
            }
            prefix.push(invoke);
            choices.push(prefix);
        }
        Some(choices)
    }

    /// Decode a single native SC117 message. Tool arguments become callable only after a stop.
    pub fn decode(&self, tokenizer: &Tokenizer, ids: &[u32], finish: &str) -> Option<Value> {
        let added = tokenizer.get_added_tokens_decoder();
        let plain = |seq: &[u32]| {
            if seq.iter().any(|id| {
                tokenizer.id_to_token(*id).is_none() || added.get(id).is_some_and(|t| t.special)
            }) {
                return None;
            }
            tokenizer.decode(seq, false).ok()
        };
        // Accept the explicit model header for recorded SC117 responses too.
        let ids = if self.forced.is_none() && !self.names.is_empty() {
            ids.strip_prefix(&[tokenizer.token_to_id(MODEL)?])
                .unwrap_or(ids)
        } else {
            ids
        };
        let invoke_id = tokenizer.token_to_id(INVOKE)?;
        let (name, payload) = if let Some(name) = &self.forced {
            (name.clone(), ids)
        } else if let Some(pos) = ids.iter().position(|id| *id == invoke_id) {
            (plain(&ids[..pos])?, &ids[pos + 1..])
        } else {
            if self.required {
                return None;
            }
            let text = if self.names.is_empty() {
                ids
            } else {
                ids.strip_prefix(&[tokenizer.token_to_id(TEXT)?])
                    .unwrap_or(ids)
            };
            return Some(json!({"role":"assistant","content":plain(text)?}));
        };
        if finish != "stop" || !self.names.contains(&name) {
            return None;
        }
        let data: Value = serde_json::from_str(&plain(payload)?).ok()?;
        if data["name"] != name || !data["args"].is_object() {
            return None;
        }
        static SEQUENCE: AtomicU64 = AtomicU64::new(1);
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .ok()?
            .as_micros();
        let id = format!(
            "call_{now:x}_{:x}",
            SEQUENCE.fetch_add(1, Ordering::Relaxed)
        );
        Some(json!({"role":"assistant","content":null,"tool_calls":[{
            "id":id,"type":"function","function":{"name":name,"arguments":data["args"].to_string()}
        }]}))
    }
}

/// Keep tool results tied to the prior assistant call that produced their ID.
pub fn history(messages: &[Value]) -> Result<Vec<Value>, InferenceError> {
    let mut pending = HashMap::<String, String>::new();
    let mut seen = HashSet::new();
    let mut turns = Vec::with_capacity(messages.len());
    for message in messages {
        let role = message["role"].as_str().ok_or(BAD_TURN)?;
        if role != "tool" && !pending.is_empty() {
            return Err(BAD_TURN);
        }
        let content = match &message["content"] {
            Value::String(s) => Value::String(s.clone()),
            Value::Null if role == "assistant" && message["tool_calls"].is_array() => Value::Null,
            Value::Array(parts)
                if !parts.is_empty()
                    && parts.iter().all(|p| {
                        matches!(p["type"].as_str(), Some("text" | "input_text"))
                            && p["text"].is_string()
                    }) =>
            {
                Value::Array(
                    parts
                        .iter()
                        .map(|p| json!({"type":"text","text":p["text"]}))
                        .collect(),
                )
            }
            _ => return Err(BAD_TURN),
        };
        let mut turn = json!({"role":role,"content":content});
        if let Some(calls) = message["tool_calls"].as_array().filter(|c| !c.is_empty()) {
            if role != "assistant" {
                return Err(BAD_TURN);
            }
            let mut normalized = Vec::with_capacity(calls.len());
            for call in calls {
                let id = call["id"]
                    .as_str()
                    .filter(|s| !s.is_empty() && s.len() <= 256)
                    .ok_or(BAD_TURN)?;
                let name = call["function"]["name"]
                    .as_str()
                    .filter(|s| valid_name(s))
                    .ok_or(BAD_TURN)?;
                if call["type"] != "function" || !seen.insert(id.to_owned()) {
                    return Err(BAD_TURN);
                }
                let raw = &call["function"]["arguments"];
                let args = if let Some(text) = raw.as_str() {
                    serde_json::from_str(text).map_err(|_| BAD_TURN)?
                } else {
                    raw.clone()
                };
                if !args.is_object() {
                    return Err(BAD_TURN);
                }
                pending.insert(id.to_owned(), name.to_owned());
                normalized.push(
                    json!({"id":id,"type":"function","function":{"name":name,"arguments":args}}),
                );
            }
            turn["tool_calls"] = normalized.into();
        } else if !message["tool_calls"].is_null() && !message["tool_calls"].is_array() {
            return Err(BAD_TURN);
        }
        if role == "tool" {
            let id = message["tool_call_id"].as_str().ok_or(BAD_TURN)?;
            let name = pending.remove(id).ok_or(BAD_TURN)?;
            if !message["name"].is_null() && message["name"] != name {
                return Err(BAD_TURN);
            }
            turn["name"] = name.into();
        }
        turns.push(turn);
    }
    if pending.is_empty() {
        Ok(turns)
    } else {
        Err(BAD_TURN)
    }
}
