//! Audio / STT perception boundary with opt-in microphone guard and simple energy VAD.

use crate::privacy::PrivacyGate;

/// Voice Activity Detection configuration (CPU-first to preserve GPU VRAM for SC117).
#[derive(Debug, Clone, Copy)]
pub struct VadConfig {
    /// Minimum RMS amplitude in `[0.0, 1.0]` to classify a PCM frame as speech.
    pub rms_threshold: f32,
    /// Push-to-talk mode (default: true; continuous listening requires explicit opt-in).
    pub push_to_talk: bool,
}

impl Default for VadConfig {
    fn default() -> Self {
        Self {
            rms_threshold: 0.02,
            push_to_talk: true,
        }
    }
}

/// Compute normalized RMS energy of signed 16-bit PCM samples.
pub fn pcm16_rms(samples: &[i16]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let sum_sq: f64 = samples
        .iter()
        .map(|&s| {
            let v = f64::from(s) / 32768.0;
            v * v
        })
        .sum();
    (sum_sq / (samples.len() as f64)).sqrt() as f32
}

/// Check whether speech is present and permitted under the active `PrivacyGate`.
pub fn detect_voice_activity(samples: &[i16], cfg: VadConfig, gate: &PrivacyGate) -> bool {
    if !cfg.push_to_talk && !gate.is_mic_allowed() {
        return false;
    }
    if gate.is_stopped() {
        return false;
    }
    pcm16_rms(samples) >= cfg.rms_threshold
}
