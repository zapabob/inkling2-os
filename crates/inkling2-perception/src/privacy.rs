//! Privacy boundary, trust labelling, and emergency stop controls for perception inputs.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

/// Sensitivity classification of a perception observation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Sensitivity {
    Public,
    Personal,
    Confidential,
}

/// Retention policy for a perception event (raw camera frames are disabled by default).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RetentionPolicy {
    /// Discard immediately after working-memory turn; never fold into persistent weights.
    Ephemeral,
    /// Allowed in explicit episodic memory if operator permits.
    SessionOnly,
    /// Explicitly approved by operator for persistent storage.
    OperatorApproved,
}

/// Global perception privacy switchboard (camera active indicator, mic opt-in, instant stop).
#[derive(Debug, Clone)]
pub struct PrivacyGate {
    camera_opt_in: Arc<AtomicBool>,
    camera_active: Arc<AtomicBool>,
    mic_opt_in: Arc<AtomicBool>,
    emergency_stop: Arc<AtomicBool>,
    retain_raw_frames: Arc<AtomicBool>,
}

impl Default for PrivacyGate {
    fn default() -> Self {
        Self {
            camera_opt_in: Arc::new(AtomicBool::new(false)),
            camera_active: Arc::new(AtomicBool::new(false)),
            mic_opt_in: Arc::new(AtomicBool::new(false)),
            emergency_stop: Arc::new(AtomicBool::new(false)),
            retain_raw_frames: Arc::new(AtomicBool::new(false)),
        }
    }
}

impl PrivacyGate {
    /// Set by the operator's process configuration, independently of model requests.
    pub fn set_camera_opt_in(&self, enabled: bool) {
        self.camera_opt_in.store(enabled, Ordering::SeqCst);
    }

    pub fn is_camera_allowed(&self) -> bool {
        self.camera_opt_in.load(Ordering::SeqCst) && !self.is_stopped()
    }

    pub fn set_camera_active(&self, active: bool) {
        self.camera_active.store(active, Ordering::SeqCst);
    }

    pub fn is_camera_active(&self) -> bool {
        self.camera_active.load(Ordering::SeqCst) && !self.is_stopped()
    }

    /// Always-on microphone monitoring requires explicit operator opt-in.
    pub fn set_mic_opt_in(&self, enabled: bool) {
        self.mic_opt_in.store(enabled, Ordering::SeqCst);
    }

    pub fn is_mic_allowed(&self) -> bool {
        self.mic_opt_in.load(Ordering::SeqCst) && !self.is_stopped()
    }

    /// Raw frame retention is disabled by default and requires explicit opt-in.
    pub fn raw_frame_retention_enabled(&self) -> bool {
        self.retain_raw_frames.load(Ordering::SeqCst)
    }

    /// Trigger immediate stop of all perception capture and clear active indicators.
    pub fn trigger_stop(&self) {
        self.emergency_stop.store(true, Ordering::SeqCst);
        self.camera_active.store(false, Ordering::SeqCst);
        self.mic_opt_in.store(false, Ordering::SeqCst);
    }

    pub fn reset_stop(&self) {
        self.emergency_stop.store(false, Ordering::SeqCst);
    }

    pub fn is_stopped(&self) -> bool {
        self.emergency_stop.load(Ordering::SeqCst)
    }
}

/// Sanitise untrusted visual/OCR text so it is never treated as a privileged instruction.
pub fn wrap_untrusted_observation(source: &str, raw_text: &str) -> String {
    let cleaned: String = raw_text
        .chars()
        .filter(|&c| !c.is_control() || c == '\n' || c == '\t')
        .take(2048)
        .collect();
    format!("[UNTRUSTED_OBSERVATION source={source}] {cleaned} [/UNTRUSTED_OBSERVATION]")
}
