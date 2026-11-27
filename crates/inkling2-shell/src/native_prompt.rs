//! Model-owned prompt framing and tokenization for the native Inkling-2 path.

use std::fs::File;
use std::io::{Read, Seek, SeekFrom};

use minijinja::{Environment, Error, ErrorKind, context};
use serde_json::Value;
use tokenizers::Tokenizer;

use crate::native_inference::InferenceError;
use crate::strata_catalog::StrataModelInfo;

const MAX_MODEL_METADATA: u64 = 64 * 1024 * 1024;
const MAX_CONTEXT_IDS: usize = 4096;
const MODEL_CONFIG_ERROR: InferenceError = (
    "503 Service Unavailable",
    "model chat template or tokenizer does not match the Inkling-2 configuration",
);

pub struct NativePrompt {
    pub tokenizer: Tokenizer,
    pub input_ids: Vec<u32>,
    pub stop_ids: Vec<u32>,
}

fn render(template: &str, messages: &[Value]) -> Result<String, Error> {
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
    env.add_function("raise_exception", |message: String| -> Result<(), Error> {
        Err(Error::new(ErrorKind::InvalidOperation, message))
    });
    let mut text = env.render_str(
        &template,
        context! {
            messages => messages,
            tools => Vec::<Value>::new(),
            add_generation_prompt => true,
            reasoning_effort => "none",
        },
    )?;
    // The model template ends after the last completed turn. Open the visible
    // assistant channel in the same way as Strata's non-thinking Inkling path.
    if !text.ends_with("<|content_text|>") {
        if !text.ends_with("<|message_model|>") {
            text.push_str("<|message_model|>");
        }
        text.push_str("<|content_text|>");
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

pub fn prepare(
    model: &StrataModelInfo,
    messages: &[Value],
    max_output: u64,
) -> Result<NativePrompt, InferenceError> {
    let manifest = load_manifest(model).ok_or(MODEL_CONFIG_ERROR)?;
    let template = manifest["chat_template"]
        .as_str()
        .filter(|text| !text.is_empty() && text.len() <= 64 * 1024)
        .ok_or(MODEL_CONFIG_ERROR)?;
    let tokenizer = if let Some(embedded) = manifest.get("tokenizer_json").filter(|v| !v.is_null())
    {
        match embedded.as_str() {
            Some(raw) => Tokenizer::from_bytes(raw),
            None => Tokenizer::from_bytes(embedded.to_string()),
        }
    } else {
        let file = File::open(model.path.with_file_name("tokenizer.json"))
            .map_err(|_| MODEL_CONFIG_ERROR)?;
        let mut bytes = Vec::new();
        file.take(MAX_MODEL_METADATA + 1)
            .read_to_end(&mut bytes)
            .map_err(|_| MODEL_CONFIG_ERROR)?;
        if bytes.len() as u64 > MAX_MODEL_METADATA {
            return Err(MODEL_CONFIG_ERROR);
        }
        Tokenizer::from_bytes(bytes)
    }
    .map_err(|_| MODEL_CONFIG_ERROR)?;

    let token = |name| tokenizer.token_to_id(name).ok_or(MODEL_CONFIG_ERROR);
    let model_token = token("<|message_model|>")?;
    let text_token = token("<|content_text|>")?;
    let stop_ids = vec![
        token("<|content_model_end_sampling|>")?,
        token("<|end_message|>")?,
        token("<|endoftext|>")?,
    ];
    let added = tokenizer.get_added_tokens_decoder();
    if messages.iter().any(|message| {
        message["content"].as_str().is_some_and(|text| {
            added
                .values()
                .any(|value| value.special && text.contains(&value.content))
        })
    }) {
        return Err(("400 Bad Request", "message contains a reserved model token"));
    }
    let text_messages: Vec<_> = messages
        .iter()
        .map(|message| serde_json::json!({"role": message["role"], "content": message["content"]}))
        .collect();
    let text = render(template, &text_messages).map_err(|_| MODEL_CONFIG_ERROR)?;
    let encoding = tokenizer
        .encode(text, false)
        .map_err(|_| MODEL_CONFIG_ERROR)?;
    let input_ids = encoding.get_ids().to_vec();
    if !input_ids.ends_with(&[model_token, text_token]) {
        return Err(MODEL_CONFIG_ERROR);
    }
    if input_ids.len() + max_output as usize > MAX_CONTEXT_IDS {
        return Err((
            "400 Bad Request",
            "request exceeds the native 4096-token context",
        ));
    }
    Ok(NativePrompt {
        tokenizer,
        input_ids,
        stop_ids,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::strata_catalog::inspect_strata_header;
    use serde_json::json;
    use std::path::Path;

    #[test]
    #[ignore = "requires the original SQ-10 model and its bundled tokenizer"]
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
