//! Camera perception facade wrapping `inkling2_companion::camera_call` with privacy controls.

use std::path::Path;

use inkling2_companion::camera_call::{
    CameraConfig, CameraDevice, capture_webcam_frame, detect_camera_devices,
};

use crate::event_bus::{Modality, PerceptionEvent, PerceptionEventBus};
use crate::privacy::{PrivacyGate, RetentionPolicy, Sensitivity};

/// Runtime status snapshot for the camera perception subsystem.
#[derive(Debug, Clone)]
pub struct CameraRuntimeStatus {
    pub operator_enabled: bool,
    pub active: bool,
    pub stopped: bool,
    pub raw_frame_retention: bool,
    pub native_sc117_vision_supported: bool,
    pub visual_pipeline_mode: &'static str,
    pub devices: Vec<CameraDevice>,
}

/// Query current camera hardware & privacy status.
pub fn camera_status(gate: &PrivacyGate) -> CameraRuntimeStatus {
    CameraRuntimeStatus {
        operator_enabled: gate.is_camera_allowed(),
        active: gate.is_camera_active(),
        stopped: gate.is_stopped(),
        raw_frame_retention: gate.raw_frame_retention_enabled(),
        native_sc117_vision_supported: false,
        visual_pipeline_mode: "auxiliary-structured-observation",
        devices: detect_camera_devices(),
    }
}

/// Capture one frame under privacy-gate control and publish an ephemeral `PerceptionEvent`.
pub fn capture_once(
    cfg: &CameraConfig,
    out_jpg: &Path,
    gate: &PrivacyGate,
    bus: &mut PerceptionEventBus,
) -> Result<PerceptionEvent, String> {
    if !gate.is_camera_allowed() {
        return Err("camera requires operator opt-in and a cleared privacy stop".to_string());
    }
    gate.set_camera_active(true);
    let result = capture_webcam_frame(cfg, out_jpg);
    gate.set_camera_active(false);

    if !gate.raw_frame_retention_enabled() {
        match std::fs::remove_file(out_jpg) {
            Ok(()) => {}
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => return Err(format!("could not discard captured frame: {error}")),
        }
    }

    let detail = result?;
    let retention = if gate.raw_frame_retention_enabled() {
        RetentionPolicy::SessionOnly
    } else {
        RetentionPolicy::Ephemeral
    };
    let ev = bus.publish(
        cfg.device_name.as_deref().unwrap_or("camera:auto"),
        Modality::Camera,
        detail,
        0.9,
        Sensitivity::Personal,
        retention,
        "local-camera-capture",
        true,
    );
    Ok(ev)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capture_requires_independent_operator_opt_in() {
        let gate = PrivacyGate::default();
        let mut bus = PerceptionEventBus::default();
        assert!(
            capture_once(
                &CameraConfig::default(),
                Path::new("unused.jpg"),
                &gate,
                &mut bus
            )
            .is_err()
        );
        assert!(bus.is_empty());
        gate.set_camera_opt_in(true);
        assert!(gate.is_camera_allowed());
        gate.trigger_stop();
        assert!(!gate.is_camera_allowed());
    }
}
