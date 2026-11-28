//! Inkling2-OS Multimodal AI Companion Subsystem:
//! - `memory45`: 45 MiB hard-budgeted compressed episodic & semantic memory (Ebbinghaus decay + int8 embeddings)
//! - `dynamic_lora`: Dynamic rank-16 LoRA weights `(A, B)` with online self-reinforcement & rollback
//! - `voice_call`: Local TTS (VOICEVOX / Windows Haruka / Linux Piper) with playback & stop control
//! - `camera_call`: Cross-platform webcam discovery, aspect-preserving capture & motion-gated vision pipeline
//! - `permission_broker`: Immutable 4-tier OS action permission broker and audit log

pub mod camera_call;
pub mod dynamic_lora;
pub mod memory45;
pub mod neural_memory45;
pub mod permission_broker;
pub mod voice_call;
