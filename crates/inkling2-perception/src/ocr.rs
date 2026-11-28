//! Auxiliary OCR observation adapter (explicitly marks extracted text as untrusted data).

use crate::event_bus::{Modality, PerceptionEvent, PerceptionEventBus};
use crate::privacy::{RetentionPolicy, Sensitivity, wrap_untrusted_observation};

/// Publish OCR text extracted by an auxiliary local OCR backend onto the perception bus.
///
/// Extracted pixel text is wrapped as untrusted observation data so it can never be
/// confused with privileged operator commands.
pub fn publish_ocr_observation(
    source: &str,
    extracted_text: &str,
    confidence: f32,
    bus: &mut PerceptionEventBus,
) -> PerceptionEvent {
    let wrapped = wrap_untrusted_observation(source, extracted_text);
    bus.publish(
        source,
        Modality::Ocr,
        wrapped,
        confidence,
        Sensitivity::Personal,
        RetentionPolicy::Ephemeral,
        "auxiliary-ocr",
        true,
    )
}
