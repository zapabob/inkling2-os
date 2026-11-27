//! Live Webcam Video-Call Frame Pipeline (`camera_call`).
//!
//! Captures 40x40-patch-aligned frames (e.g., 400x400) matching Inkling-2-Mini's
//! Perception Encoder vision tower, with motion-delta gating to avoid redundant GPU encoding.

use std::path::Path;
use std::process::Command;

/// Inkling-2-Mini Vision patch size (40x40 pixels per visual token).
pub const VISION_PATCH_PX: u32 = 40;

/// Webcam capture settings.
#[derive(Debug, Clone)]
pub struct CameraConfig {
    /// Target width (must be a multiple of 40 for Inkling-2-Mini Vision).
    pub width: u32,
    /// Target height (must be a multiple of 40 for Inkling-2-Mini Vision).
    pub height: u32,
    /// Motion threshold (mean absolute pixel diff in `0..=255`) to trigger vision re-encoding.
    pub motion_threshold: f32,
}

impl Default for CameraConfig {
    fn default() -> Self {
        Self {
            width: 400,
            height: 400,
            motion_threshold: 6.0,
        }
    }
}

/// Compute mean absolute difference between two equal-length raw RGB/luma frame buffers.
pub fn frame_motion_delta(prev: &[u8], curr: &[u8]) -> f32 {
    let n = prev.len().min(curr.len());
    if n == 0 {
        return 0.0;
    }
    let sum: u64 = prev
        .iter()
        .zip(curr.iter())
        .map(|(&a, &b)| u64::from(a.abs_diff(b)))
        .sum();
    (sum as f32) / (n as f32)
}

/// Capture a single 40x40-patch-aligned JPEG frame from the default webcam via `ffmpeg`.
pub fn capture_webcam_frame(cfg: &CameraConfig, out_jpg: &Path) -> Result<String, String> {
    let w = (cfg.width / VISION_PATCH_PX).max(1) * VISION_PATCH_PX;
    let h = (cfg.height / VISION_PATCH_PX).max(1) * VISION_PATCH_PX;
    let vf = format!("scale={w}:{h}");

    let ffmpeg_bin = if Path::new(r"C:\Users\downl\.hermes\bin\ffmpeg.exe").exists() {
        r"C:\Users\downl\.hermes\bin\ffmpeg.exe"
    } else {
        "ffmpeg"
    };

    let output = if cfg!(target_os = "linux") {
        Command::new(ffmpeg_bin)
            .args([
                "-y",
                "-f",
                "video4linux2",
                "-i",
                "/dev/video0",
                "-frames:v",
                "1",
                "-vf",
                &vf,
                &out_jpg.to_string_lossy(),
            ])
            .output()
    } else {
        Command::new(ffmpeg_bin)
            .args([
                "-y",
                "-f",
                "dshow",
                "-i",
                "video=Integrated Camera",
                "-frames:v",
                "1",
                "-vf",
                &vf,
                &out_jpg.to_string_lossy(),
            ])
            .output()
    };

    match output {
        Ok(o) if o.status.success() => Ok(format!(
            "Captured {w}x{h} vision frame -> {}",
            out_jpg.display()
        )),
        Ok(o) => Err(format!(
            "webcam capture unavailable ({}): {}",
            o.status,
            String::from_utf8_lossy(&o.stderr)
                .lines()
                .last()
                .unwrap_or("no camera device")
        )),
        Err(e) => Err(format!("ffmpeg unavailable: {e}")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn motion_delta_detects_scene_change() {
        let a = vec![10u8; 400];
        let b = vec![10u8; 400];
        let c = vec![30u8; 400];
        assert!(frame_motion_delta(&a, &b) < 1e-6);
        assert!((frame_motion_delta(&a, &c) - 20.0).abs() < 1e-4);
    }
}
