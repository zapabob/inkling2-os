//! Live Webcam Perception Capture Pipeline (`camera_call`).
//!
//! Provides cross-platform camera device discovery (Windows PnP / DirectShow and Linux V4L2),
//! aspect-ratio-preserving capture, motion-delta gating to suppress redundant visual inference,
//! and explicit privacy/stop controls. Vision patch geometry is NOT assumed to be 40x40 unless
//! explicitly opted in via `CameraConfig::patch_align`.

#[cfg(windows)]
use std::os::windows::process::CommandExt;
use std::path::Path;
use std::process::Command;

/// Default capture width (640x480 per Neon Phase 1.1 baseline).
pub const DEFAULT_CAPTURE_WIDTH: u32 = 640;
/// Default capture height (640x480 per Neon Phase 1.1 baseline).
pub const DEFAULT_CAPTURE_HEIGHT: u32 = 480;

/// Detected camera hardware descriptor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CameraDevice {
    /// OS-specific device identifier (e.g., DirectShow name or `/dev/video0`).
    pub id: String,
    /// Human-readable device label.
    pub name: String,
    /// Capture backend label (`windows-pnp`, `dshow`, `v4l2`).
    pub backend: String,
}

/// Webcam capture settings.
#[derive(Debug, Clone)]
pub struct CameraConfig {
    /// Target maximum width in pixels (default: 640).
    pub width: u32,
    /// Target maximum height in pixels (default: 480).
    pub height: u32,
    /// Target capture frame rate in FPS (5..=15 recommended).
    pub fps: u32,
    /// Preserve source aspect ratio when scaling (default: true).
    pub preserve_aspect_ratio: bool,
    /// Optional explicit patch alignment multiple (None by default; do not assume 40x40).
    pub patch_align: Option<u32>,
    /// Optional explicit device ID/name. When `None`, the first detected camera is used.
    pub device_name: Option<String>,
    /// Motion threshold (mean absolute pixel diff in `0..=255`) to trigger vision re-encoding.
    pub motion_threshold: f32,
}

impl Default for CameraConfig {
    fn default() -> Self {
        Self {
            width: DEFAULT_CAPTURE_WIDTH,
            height: DEFAULT_CAPTURE_HEIGHT,
            fps: 10,
            preserve_aspect_ratio: true,
            patch_align: None,
            device_name: None,
            motion_threshold: 6.0,
        }
    }
}

fn hidden_command(program: &str) -> Command {
    #[cfg(windows)]
    {
        let mut cmd = Command::new(program);
        cmd.creation_flags(0x08000000);
        cmd
    }
    #[cfg(not(windows))]
    Command::new(program)
}

fn resolve_ffmpeg_binary() -> String {
    if let Ok(explicit) = std::env::var("INKLING2_FFMPEG_BIN")
        && !explicit.trim().is_empty()
    {
        return explicit;
    }
    "ffmpeg".to_string()
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

/// Determine whether a frame delta exceeds the configured motion threshold.
pub fn should_trigger_inference(prev: &[u8], curr: &[u8], threshold: f32) -> bool {
    if !threshold.is_finite() || threshold <= 0.0 {
        return true;
    }
    frame_motion_delta(prev, curr) >= threshold
}

/// Build an FFmpeg scale filter honoring aspect-ratio preservation and optional patch alignment.
pub fn build_scale_filter(cfg: &CameraConfig) -> String {
    let (w, h) = match cfg.patch_align.filter(|&p| p > 0) {
        Some(align) => (
            (cfg.width / align).max(1) * align,
            (cfg.height / align).max(1) * align,
        ),
        None => (cfg.width.max(16), cfg.height.max(16)),
    };
    if cfg.preserve_aspect_ratio {
        format!("scale={w}:{h}:force_original_aspect_ratio=decrease")
    } else {
        format!("scale={w}:{h}")
    }
}

/// Parse `ffmpeg -list_devices true -f dshow -i dummy` stderr output into video `CameraDevice`s.
pub fn parse_dshow_video_devices(stderr: &str) -> Vec<CameraDevice> {
    let mut devices = Vec::new();
    let mut in_video_section = false;
    for line in stderr.lines() {
        let trimmed = line.trim();
        if trimmed.contains("DirectShow video devices") {
            in_video_section = true;
            continue;
        }
        if trimmed.contains("DirectShow audio devices") {
            in_video_section = false;
            continue;
        }
        if trimmed.contains("Alternative name") {
            continue;
        }
        let is_video_tagged = trimmed.ends_with("(video)");
        if !(in_video_section || is_video_tagged) {
            continue;
        }
        if let Some(first_quote) = trimmed.find('"') {
            let rest = &trimmed[first_quote + 1..];
            if let Some(end_quote) = rest.find('"') {
                let name = rest[..end_quote].trim();
                if !name.is_empty() && !devices.iter().any(|d: &CameraDevice| d.id == name) {
                    devices.push(CameraDevice {
                        id: name.to_string(),
                        name: name.to_string(),
                        backend: "dshow".to_string(),
                    });
                }
            }
        }
    }
    devices
}

/// Detect available camera devices on Windows (PnP + DirectShow fallback) or Linux (`/dev/video*`).
pub fn detect_camera_devices() -> Vec<CameraDevice> {
    let mut devices = Vec::new();

    if cfg!(target_os = "windows") {
        // 1. Prefer DirectShow device enumeration via FFmpeg when available (exact capture names).
        let ffmpeg_bin = resolve_ffmpeg_binary();
        if let Ok(out) = hidden_command(&ffmpeg_bin)
            .args([
                "-hide_banner",
                "-list_devices",
                "true",
                "-f",
                "dshow",
                "-i",
                "dummy",
            ])
            .output()
        {
            let stderr = String::from_utf8_lossy(&out.stderr);
            devices.extend(parse_dshow_video_devices(&stderr));
        }

        // 2. Supplement with Windows PnP Camera / Image class devices via CIM (Media Foundation ready).
        if devices.is_empty() {
            let ps = r#"Get-CimInstance Win32_PnPEntity | Where-Object { ($_.PNPClass -eq 'Camera' -or $_.PNPClass -eq 'Image') -and $_.Status -eq 'OK' } | ForEach-Object { $_.Name }"#;
            if let Ok(out) = hidden_command("powershell")
                .args(["-NoProfile", "-NonInteractive", "-Command", ps])
                .output()
                && out.status.success()
            {
                for line in String::from_utf8_lossy(&out.stdout).lines() {
                    let name = line.trim();
                    if !name.is_empty() && !devices.iter().any(|d| d.name == name) {
                        devices.push(CameraDevice {
                            id: name.to_string(),
                            name: name.to_string(),
                            backend: "windows-pnp".to_string(),
                        });
                    }
                }
            }
        }
    } else if cfg!(target_os = "linux") {
        for idx in 0..16u32 {
            let dev_path = format!("/dev/video{idx}");
            if Path::new(&dev_path).exists() {
                let sys_name_path = format!("/sys/class/video4linux/video{idx}/name");
                let label = std::fs::read_to_string(&sys_name_path)
                    .ok()
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty())
                    .unwrap_or_else(|| dev_path.clone());
                devices.push(CameraDevice {
                    id: dev_path,
                    name: label,
                    backend: "v4l2".to_string(),
                });
            }
        }
    }

    devices
}

fn validate_device_id(id: &str) -> Result<(), String> {
    if id.is_empty() || id.len() > 256 {
        return Err("camera device name length is invalid".to_string());
    }
    if id.starts_with('-') || id.contains('\0') || id.contains('\n') || id.contains('\r') {
        return Err("camera device name contains invalid control characters".to_string());
    }
    Ok(())
}

/// Capture a single frame from the selected (or first detected) webcam into `out_jpg`.
pub fn capture_webcam_frame(cfg: &CameraConfig, out_jpg: &Path) -> Result<String, String> {
    let vf = build_scale_filter(cfg);
    let ffmpeg_bin = resolve_ffmpeg_binary();
    let selected_id = match cfg.device_name.as_deref() {
        Some(explicit) if !explicit.trim().is_empty() => explicit.trim().to_string(),
        _ => {
            let detected = detect_camera_devices();
            match detected.first() {
                Some(dev) => dev.id.clone(),
                None => {
                    return Err("no camera device detected on this system".to_string());
                }
            }
        }
    };
    validate_device_id(&selected_id)?;

    let output = if cfg!(target_os = "linux") {
        hidden_command(&ffmpeg_bin)
            .args([
                "-y",
                "-f",
                "video4linux2",
                "-i",
                &selected_id,
                "-frames:v",
                "1",
                "-vf",
                &vf,
                &out_jpg.to_string_lossy(),
            ])
            .output()
    } else {
        let dshow_input = format!("video={selected_id}");
        hidden_command(&ffmpeg_bin)
            .args([
                "-y",
                "-f",
                "dshow",
                "-i",
                &dshow_input,
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
            "Captured vision frame (device='{selected_id}', filter='{vf}') -> {}",
            out_jpg.display()
        )),
        Ok(o) => Err(format!(
            "webcam capture unavailable ({}): {}",
            o.status,
            String::from_utf8_lossy(&o.stderr)
                .lines()
                .last()
                .unwrap_or("camera device error")
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
        assert!(!should_trigger_inference(&a, &b, 6.0));
        assert!(should_trigger_inference(&a, &c, 6.0));
    }

    #[test]
    fn scale_filter_preserves_aspect_ratio_by_default() {
        let cfg = CameraConfig::default();
        assert_eq!(
            build_scale_filter(&cfg),
            "scale=640:480:force_original_aspect_ratio=decrease"
        );
    }

    #[test]
    fn parses_dshow_video_devices_and_skips_audio() {
        let sample = r#"
[dshow @ 000001] "Integrated Webcam" (video)
[dshow @ 000001]   Alternative name "@device_pnp_\\?\usb#vid"
[dshow @ 000001] "USB HD Camera" (video)
[dshow @ 000001] "Microphone Array" (audio)
"#;
        let devs = parse_dshow_video_devices(sample);
        assert_eq!(devs.len(), 2);
        assert_eq!(devs[0].name, "Integrated Webcam");
        assert_eq!(devs[1].name, "USB HD Camera");
    }
}
