//! Inkling2-OS Multimodal Perception Runtime (`inkling2-perception`, Neon Phase 1).
//!
//! Converts camera, audio, desktop, and system observations into typed `PerceptionEvent`
//! records with explicit provenance, sensitivity classification, privacy stop-gates,
//! and untrusted-content isolation (preventing prompt injection from visual/OCR text).

pub mod audio;
pub mod camera;
pub mod event_bus;
pub mod motion;
pub mod ocr;
pub mod privacy;
pub mod vision;
