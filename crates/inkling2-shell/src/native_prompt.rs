//! Model-owned prompt framing and tokenization for the native Inkling-2 path.

use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::sync::{Arc, OnceLock};

use minijinja::{Environment, Error, ErrorKind, context};
use serde_json::Value;
use tokenizers::Tokenizer;

use crate::native_inference::InferenceError;
use crate::native_tools::{self, Tools};
use crate::strata_catalog::StrataModelInfo;

const MAX_MODEL_METADATA: u64 = 64 * 1024 * 1024;
const SC117_TOKENIZER: &[u8] = include_bytes!("../assets/sc117/tokenizer.json");
pub const MAX_CONTEXT_IDS: usize = 131_072;
const MODEL_CONFIG_ERROR: InferenceError = (
    "503 Service Unavailable",
    "model chat template or tokenizer does not match the Inkling-2 configuration",
);

pub struct NativePrompt {
    pub tokenizer: Arc<Tokenizer>,
    pub tools: Tools,
    pub input_ids: Vec<u32>,
    pub stop_ids: Vec<u32>,
    pub channel_prefixes: Vec<Vec<u32>>,
    pub model_limit: usize,
}

fn render(template: &str, messages: &[Value], tools: &Tools) -> Result<String, Error> {
    // Transformers records the ranges inside these tags without emitting text.
    let template = template
        .replace("{%- generation -%}", "{#- generation -#}")
        .replace("{%- endgeneration -%}", "{#- endgeneration -#}")
        .replace("{% generation %}", "{# generation #}")
        .replace("{% endgeneration %}", "{# endgeneration #}");
    let mut env = Environment::new();
    env.set_fuel(Some(100_000));
    env.set_trim_blocks(true);
    env.set_lstrip_blocks(true);
    // Transformers' tojson accepts the Python JSON keyword arguments used by SC117.
    env.add_filter(
        "tojson",
        |value: minijinja::Value, kwargs: minijinja::value::Kwargs| -> Result<String, Error> {
            let sort = kwargs.get::<Option<bool>>("sort_keys")?.unwrap_or(false);
            let separators = kwargs.get::<Option<Vec<String>>>("separators")?;
            kwargs.assert_all_used()?;
            if separators.is_some_and(|s| s != [",", ":"]) {
                return Err(Error::new(
                    ErrorKind::InvalidOperation,
                    "unsupported JSON separators",
                ));
            }
            let mut json = serde_json::to_value(value)
                .map_err(|e| Error::new(ErrorKind::InvalidOperation, e.to_string()))?;
            if sort {
                json.sort_all_objects();
            }
            Ok(json.to_string())
        },
    );
    env.add_function("raise_exception", |message: String| -> Result<(), Error> {
        Err(Error::new(ErrorKind::InvalidOperation, message))
    });
    // Reproduce the empty reasoning block preceding earlier tool-enabled turns.
    // The template emits a block only for a nonempty value. User data containing
    // reserved markers has already been rejected before rendering.
    let framed;
    let messages =
        if !tools.declarations.is_empty() && messages.iter().any(|m| m["role"] == "assistant") {
            framed = messages
                .iter()
                .cloned()
                .map(|mut m| {
                    if m["role"] == "assistant"
                        && (m["tool_calls"].is_array()
                            || m["content"].as_str().is_some_and(|v| !v.is_empty()))
                    {
                        m["reasoning_content"] = "<|content_thinking|>".into();
                    }
                    m
                })
                .collect::<Vec<_>>();
            framed.as_slice()
        } else {
            messages
        };
    let mut text = env.render_str(
        &template,
        context! {
            messages => messages,
            tools => &tools.declarations,
            add_generation_prompt => true,
            reasoning_effort => "none",
        },
    )?;
    if !tools.declarations.is_empty() {
        text = text.replace(
            "<|content_thinking|><|content_thinking|><|end_message|>",
            native_tools::EMPTY_THINK,
        );
    }
    // The model template ends after the last completed turn. Open the visible
    // assistant channel in the same way as Strata's non-thinking Inkling path.
    if !text.ends_with("<|content_text|>") {
        if !text.ends_with("<|message_model|>") {
            text.push_str("<|message_model|>");
        }
        text.push_str(&tools.assistant_prefix());
    }
    Ok(text)
}

fn load_manifest(model: &StrataModelInfo) -> Option<Value> {
    let mut file = File::open(&model.path).ok()?;
    file.seek(SeekFrom::Start(model.manifest_offset)).ok()?;
    let mut decoder = zstd::stream::read::Decoder::new(file.take(model.manifest_bytes)).ok()?;
    decoder.window_log_max(26).ok()?;
    let mut bytes = Vec::new();
    decoder
        .take(MAX_MODEL_METADATA + 1)
        .read_to_end(&mut bytes)
        .ok()?;
    if bytes.len() as u64 > MAX_MODEL_METADATA {
        return None;
    }
    serde_json::from_slice(&bytes).ok()
}

fn manifest_limit(manifest: &Value) -> usize {
    manifest
        .pointer("/config/text_config/model_max_length")
        .and_then(Value::as_u64)
        .unwrap_or(4096)
        .min(MAX_CONTEXT_IDS as u64) as usize
}

pub fn model_context_limit(model: &StrataModelInfo) -> Option<usize> {
    load_manifest(model).map(|m| manifest_limit(&m))
}

pub fn prepare(
    model: &StrataModelInfo,
    messages: &[Value],
    max_output: u64,
    tools: Tools,
) -> Result<NativePrompt, InferenceError> {
    let manifest = load_manifest(model).ok_or(MODEL_CONFIG_ERROR)?;
    let template = manifest["chat_template"]
        .as_str()
        .filter(|text| !text.is_empty() && text.len() <= 64 * 1024)
        .ok_or(MODEL_CONFIG_ERROR)?;
    let tokenizer = load_tokenizer(model, &manifest)?;

    let token = |name| tokenizer.token_to_id(name).ok_or(MODEL_CONFIG_ERROR);
    let model_token = token("<|message_model|>")?;
    let text_token = token("<|content_text|>")?;
    let stop_ids = vec![
        token("<|content_model_end_sampling|>")?,
        token("<|end_message|>")?,
        token("<|endoftext|>")?,
    ];
    let added = tokenizer.get_added_tokens_decoder();
    let text_messages = native_tools::history(messages)?;
    let untrusted = serde_json::json!([&text_messages, &tools.declarations]).to_string();
    if untrusted.contains("<|")
        && added
            .values()
            .any(|v| v.special && untrusted.contains(&v.content))
    {
        return Err(("400 Bad Request", "message contains a reserved model token"));
    }
    let text = render(template, &text_messages, &tools).map_err(|_| MODEL_CONFIG_ERROR)?;
    let encoding = tokenizer
        .encode(text, false)
        .map_err(|_| MODEL_CONFIG_ERROR)?;
    let input_ids = encoding.get_ids().to_vec();
    if tools.declarations.is_empty() && !input_ids.ends_with(&[model_token, text_token]) {
        return Err(MODEL_CONFIG_ERROR);
    }
    let model_limit = manifest_limit(&manifest);
    if input_ids.len() as u64 + max_output > model_limit as u64 {
        return Err((
            "400 Bad Request",
            "prompt and reserved output exceed the model context (maximum 131072 tokens)",
        ));
    }
    let channel_prefixes = tools
        .channel_prefixes(&tokenizer)
        .ok_or(MODEL_CONFIG_ERROR)?;
    Ok(NativePrompt {
        channel_prefixes,
        tokenizer,
        tools,
        input_ids,
        stop_ids,
        model_limit,
    })
}

fn load_tokenizer(
    model: &StrataModelInfo,
    manifest: &Value,
) -> Result<Arc<Tokenizer>, InferenceError> {
    if let Some(value) = manifest.get("tokenizer_json").filter(|v| !v.is_null()) {
        let parsed = match value.as_str() {
            Some(raw) => Tokenizer::from_bytes(raw),
            None => Tokenizer::from_bytes(value.to_string()),
        };
        return parsed.map(Arc::new).map_err(|_| MODEL_CONFIG_ERROR);
    }
    let sc117 = manifest["arch"] == "inkling2_loop"
        && manifest
            .pointer("/config/text_config/vocab_size")
            .and_then(Value::as_u64)
            == Some(201024)
        && manifest
            .pointer("/config/eos_token_id")
            .and_then(Value::as_u64)
            == Some(200006)
        && matches!(
            manifest["build"].as_str(),
            Some("SQ-10" | "SQ-15" | "SQ-20")
        );
    match File::open(model.path.with_file_name("tokenizer.json")) {
        Ok(file) => {
            let mut bytes = Vec::new();
            file.take(MAX_MODEL_METADATA + 1)
                .read_to_end(&mut bytes)
                .map_err(|_| MODEL_CONFIG_ERROR)?;
            if bytes.len() as u64 > MAX_MODEL_METADATA {
                return Err(MODEL_CONFIG_ERROR);
            }
            if sc117 && bytes == SC117_TOKENIZER {
                return built_in_tokenizer();
            }
            Tokenizer::from_bytes(bytes)
                .map(Arc::new)
                .map_err(|_| MODEL_CONFIG_ERROR)
        }
        Err(err) if err.kind() == std::io::ErrorKind::NotFound && sc117 => built_in_tokenizer(),
        Err(_) => Err(MODEL_CONFIG_ERROR),
    }
}

fn built_in_tokenizer() -> Result<Arc<Tokenizer>, InferenceError> {
    static BUILTIN: OnceLock<Option<Arc<Tokenizer>>> = OnceLock::new();
    BUILTIN
        .get_or_init(|| Tokenizer::from_bytes(SC117_TOKENIZER).ok().map(Arc::new))
        .clone()
        .ok_or(MODEL_CONFIG_ERROR)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::strata_catalog::inspect_strata_header;
    use serde_json::json;
    use std::path::Path;

    fn prepare(
        model: &StrataModelInfo,
        messages: &[Value],
        max: u64,
    ) -> Result<NativePrompt, InferenceError> {
        super::prepare(model, messages, max, Tools::default())
    }

    #[test]
    fn matches_sc117_function_round_trip() -> Result<(), String> {
        let request = json!({"tools":[{"type":"function","function":{
            "name":"get_hour","description":"Read the hour","parameters":{"type":"object"}
        }}]});
        let tools = Tools::from_request(&request).map_err(|e| e.1.to_owned())?;
        let tokenizer = built_in_tokenizer().map_err(|e| e.1.to_owned())?;
        let encode = |s: &str| {
            tokenizer
                .encode(s, false)
                .map(|e| e.get_ids().to_vec())
                .map_err(|e| e.to_string())
        };
        let ids = encode(
            r#"<|message_model|>get_hour<|content_invoke_tool_json|>{"name":"get_hour","args":{}}"#,
        )?;
        let call = tools
            .decode(&tokenizer, &ids, "stop")
            .ok_or("call was rejected")?;
        assert_eq!(
            call["tool_calls"][0]["function"],
            json!({"name":"get_hour","arguments":"{}"})
        );
        assert!(tools.decode(&tokenizer, &ids, "length").is_none());
        let unknown = encode(r#"other<|content_invoke_tool_json|>{"name":"other","args":{}}"#)?;
        assert!(tools.decode(&tokenizer, &unknown, "stop").is_none());
        let text = tools.decode(
            &tokenizer,
            &encode("<|message_model|><|content_text|>09")?,
            "stop",
        );
        assert_eq!(text, Some(json!({"role":"assistant","content":"09"})));
        let user = json!({"role":"user","content":"Time?"});
        let result =
            json!({"role":"tool","tool_call_id":call["tool_calls"][0]["id"],"content":"09"});
        assert!(native_tools::history(&[user.clone(), call.clone()]).is_err());
        assert!(native_tools::history(std::slice::from_ref(&result)).is_err());
        let turns = native_tools::history(&[user, call, result]).map_err(|e| e.1.to_owned())?;
        let rendered = render(
            include_str!("../assets/sc117/chat_template.jinja"),
            &turns,
            &tools,
        )
        .map_err(|e| e.to_string())?;
        // Golden output from the publisher's Jinja2 template with reasoning disabled.
        assert_eq!(
            rendered,
            concat!(
                r#"<|message_system|>tool_declare<|content_xml|>[{"description":"Read the hour","name":"get_hour","parameters":{"type":"object"},"type":"function"}]<|end_message|>"#,
                "<|message_system|><|content_text|>Thinking effort level: 0<|end_message|>",
                "<|message_user|><|content_text|>Time?<|end_message|>",
                r#"<|message_model|><|content_thinking|><|end_message|><|message_model|>get_hour<|content_invoke_tool_json|>{"name":"get_hour","args":{}}<|end_message|>"#,
                "<|content_model_end_sampling|><|message_tool|>get_hour<|content_text|>09<|end_message|><|message_model|><|content_thinking|><|end_message|><|message_model|>"
            )
        );
        let prefixes = tools
            .channel_prefixes(&tokenizer)
            .ok_or("channel encoding failed")?;
        assert_eq!(
            prefixes,
            vec![
                encode("<|content_text|>")?,
                encode("get_hour<|content_invoke_tool_json|>")?
            ]
        );
        Ok(())
    }

    #[test]
    #[ignore = "requires the original SC117 model"]
    fn matches_reference_token_ids() -> Result<(), String> {
        let path = std::env::var("INKLING2_TEST_MODEL").map_err(|error| error.to_string())?;
        let model = inspect_strata_header(Path::new(&path))?;
        let messages = [
            json!({"role":"system","content":"Be concise."}),
            json!({"role":"user","content":"日本語\n二行目"}),
            json!({"role":"assistant","content":"Got it."}),
            json!({"role":"user","content":"Try again? 123!"}),
        ];
        let prompt = prepare(&model, &messages, 8).map_err(|(_, error)| error.to_owned())?;
        // Independent Python Jinja2 and HF Tokenizers reference using the same local model.
        assert_eq!(
            prompt.input_ids,
            [
                200002, 200004, 1618, 149911, 4123, 13, 200010, 200002, 200004, 34182, 279, 154554,
                150628, 25, 220, 15, 200010, 200000, 200004, 90626, 22912, 198, 19084, 21718,
                21588, 200010, 200001, 200004, 35399, 149921, 13, 200010, 200006, 200000, 200004,
                32726, 151433, 30, 220, 70904, 0, 200010, 200001, 200004,
            ]
        );
        assert_eq!(prompt.stop_ids, [200006, 200010, 199999]);
        // Each repeated " x" adds exactly one token with the shipping tokenizer.
        let first = [json!({"role":"user", "content":" x"})];
        let small = prepare(&model, &first, 1).map_err(|(_, error)| error.to_owned())?;
        let repetitions = MAX_CONTEXT_IDS - small.input_ids.len();
        let boundary = [json!({"role":"user", "content":" x".repeat(repetitions)})];
        let long = prepare(&model, &boundary, 1).map_err(|(_, error)| error.to_owned())?;
        assert_eq!(long.input_ids.len() + 1, MAX_CONTEXT_IDS);
        assert!(matches!(
            prepare(&model, &boundary, 2),
            Err(("400 Bad Request", _))
        ));
        assert!(matches!(
            prepare(
                &model,
                &[json!({"role":"user","content":"<|message_model|>"})],
                8
            ),
            Err(("400 Bad Request", _))
        ));
        Ok(())
    }
}
