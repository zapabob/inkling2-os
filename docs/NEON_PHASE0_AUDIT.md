# Neon Phase 0 Audit Report

**Audit Date:** 2026-11-28
**Repositories Audited:**
- `inkling2-os` (HEAD `f3c06ae7d2b68203f6e2e645560e0a6f3bbaee7d`, branch `main`)
- `inkling2-cuda-engine` (HEAD `f5b234fff399637a0643baad2f4a49227e305d7e`, branch `perf/cuda135-tc-gemm-gqa-nvme-opt`)

---

## 1. Capability Classification Matrix

Per Phase 0 of the Neon Implementation Master Plan, every subsystem is classified strictly into one of five states:

1. **Implemented and Actively Used**
2. **Standalone Prototype**
3. **Implemented but Disconnected from Inference Path**
4. **Unimplemented Design**
5. **Hardware-Verified**

| Subsystem | Source Location | Classification | Audit Findings |
| :--- | :--- | :--- | :--- |
| **SC117 CUDA Inference Engine** | `inkling2-cuda-engine` (`serve-token-ids`, `generate-token-ids`) | **1 & 5** (Active & Hardware-Verified) | Verified live on Windows 11 + RTX 5060 Ti 16 GiB (`sm_120a`, CUDA 13.5) via `inkling2-os-bridge` on `127.0.0.1:18083`. `Inkling-2-Mini-SQ-15-abliterated` answered arithmetic probe `2+2=?` with `4`. |
| **OpenAI Chat Completions & Native Tool Bridge** | `crates/inkling2-shell/src/http_bridge.rs`, `native_tools.rs` | **1 & 5** (Active & Hardware-Verified) | Loopback-only HTTP/1.1 bridge with Host/Origin/Fetch Metadata checks, optional DPAPI-stored Console/Inference Bearer tokens, and SC117 channel-header constrained tool calling. |
| **Memory45 Store** | `crates/inkling2-companion/src/memory45.rs` | **1** (Active; Hardened in P0) | **P0 Fixes Applied:** (a) single oversized record now truncated on UTF-8 boundary so `serialized_bytes <= budget_bytes` always holds; (b) legacy `.bak2`/`.bak3` pruned and `.bak1` retained only when `active + backup <= budget_bytes`; (c) FNV-1a 64-bit integrity trailer + strict bounds/NaN checks prevent OOM/corruption; (d) `load_with_recovery` falls back to `.bak1`. |
| **Dynamic LoRA** | `crates/inkling2-companion/src/dynamic_lora.rs` | **3** (Disconnected from Inference Path) | Standalone Rank-16 `(A, B)` module with unit-tested rollback. **Not connected to SC117 CUDA forward pass**; must not be conflated with verified Transformer weight adaptation. |
| **Camera Capture (`camera_call`)** | `crates/inkling2-companion/src/camera_call.rs` | **2** (Prototype; Hardened in P0) | **P0 Fixes Applied:** Replaced hardcoded `"video=Integrated Camera"` with cross-platform device discovery (Windows DirectShow/PnP and Linux `/dev/video*`), defaulted to `640x480` aspect-preserving resize, and made 40x40 patch alignment opt-in pending real SC117 vision-tower inspection. |
| **Voice TTS (`voice_call`)** | `crates/inkling2-companion/src/voice_call.rs` | **1** (Active; Hardened in P0) | **P0 Fixes Applied:** Implemented `check_voicevox_health` (loopback-only), actual `speedScale` patching in VOICEVOX `audio_query`, `speed_scale` -> Windows SAPI `Rate` mapping, synchronous WAV playback (`play_wav_file`), Japanese/English sentence splitting, and cooperative `SpeechController` stop flag. True token-level streaming remains unimplemented until resident worker streams tokens. |
| **Permission Broker** | `crates/inkling2-companion/src/permission_broker.rs` | **1** (Implemented in P0; Integration Pending) | New immutable 4-tier risk classifier (`ReadOnly`, `ReversibleLocal`, `ExternalSideEffect`, `DestructiveOrPrivileged`) with explicit `ApprovalToken` enforcement, unconditional denial of arbitrary raw shell commands, and structured audit ring. |
| **Neural Memory45 (`.ikmem2`, PQ, Fast Weights)** | Not yet implemented | **4** (Unimplemented Design) | Scheduled for Phase 4 (P1) after P0 baseline stabilisation. |
| **Multimodal Perception Event Bus / STT / OS Agent Loop** | Not yet implemented | **4** (Unimplemented Design) | Scheduled for Phases 1–3 (P1). |

This historical audit predates the Windows 1.3.0 integration. See [NEON architecture](NEON_ARCHITECTURE.md) for shipped behaviour.
