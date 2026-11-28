//! Frame-difference motion detector for suppressing redundant visual inference.

use inkling2_companion::camera_call::{frame_motion_delta, should_trigger_inference};

/// Stateful motion gate that tracks the previous luma/RGB frame.
#[derive(Debug, Clone)]
pub struct MotionGate {
    threshold: f32,
    prev_frame: Option<Vec<u8>>,
    suppressed_frames: u64,
    triggered_frames: u64,
}

impl MotionGate {
    pub fn new(threshold: f32) -> Self {
        Self {
            threshold: if threshold.is_finite() && threshold > 0.0 {
                threshold
            } else {
                6.0
            },
            prev_frame: None,
            suppressed_frames: 0,
            triggered_frames: 0,
        }
    }

    /// Evaluate a new frame buffer. Returns `(should_infer, delta)`.
    pub fn evaluate(&mut self, frame: &[u8]) -> (bool, f32) {
        let Some(prev) = &self.prev_frame else {
            self.prev_frame = Some(frame.to_vec());
            self.triggered_frames += 1;
            return (true, 255.0);
        };
        let delta = frame_motion_delta(prev, frame);
        let trigger = should_trigger_inference(prev, frame, self.threshold);
        if trigger {
            self.prev_frame = Some(frame.to_vec());
            self.triggered_frames += 1;
        } else {
            self.suppressed_frames += 1;
        }
        (trigger, delta)
    }

    pub fn stats(&self) -> (u64, u64) {
        (self.triggered_frames, self.suppressed_frames)
    }
}
