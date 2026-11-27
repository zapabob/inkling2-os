//! Inkling2-OS Multimodal AI Companion Subsystem:
//! - `memory45`: 45 MiB hard-budgeted compressed episodic & semantic memory (Ebbinghaus decay + int8 embeddings)
//! - `dynamic_lora`: Dynamic rank-16 LoRA weights `(A, B)` with online self-reinforcement & rollback
//! - `voice_call`: Cute Japanese girl voice TTS (VOICEVOX / Windows Haruka / Linux Piper) + mic capture
//! - `camera_call`: 40x40-patch-aligned webcam capture & motion-gated vision pipeline for Inkling-2-Mini

pub mod camera_call;
pub mod dynamic_lora;
pub mod memory45;
pub mod voice_call;
