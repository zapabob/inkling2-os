# Inkling2-OS (`inkling2-os`)

> **A WDDM-free, AI-Native Appliance Operating System and Strata-aligned macOS-style Console, purpose-built for CUDA 13.5 (`sm_120a`) Tensor Core inference.**

---

## Overview

`inkling2-os` is a lightweight, security-hardened AI-native operating system and local runtime console engineered around **SC117 Inkling-2-Mini** and NVIDIA Blackwell (**CUDA 13.5 `sm_120a`**). It pairs a Rust 2024 PID 1 initialisation daemon (`inkling2-init`), a Linux `io_uring` + `O_DIRECT` + 2 MiB HugePages runtime, and an ergonomic **Strata-aligned CLI & macOS-inspired Web GUI** (`inkling2-shell`).

The entire operational console is compiled directly into a single self-contained Rust binary with zero external CDN dependencies, ensuring identical behaviour whether launched on a local Windows workstation or booted offline from the bare-metal UEFI/BIOS ISO image.

---

## Key Capabilities

### 1. Bare-Metal AI Appliance OS (`inkling2-init` & `os-image`)
- **Custom Rust 2024 PID 1 (`inkling2-init`)**: Mounts `/proc`, `/sys`, `/dev`, `hugetlbfs` (2 MiB HugePages), loads NVIDIA kernel modules (`nvidia`, `nvidia_uvm`, `nvidia_modeset`), tunes kernel sysctls, and calculates WDDM-free VRAM budgets prior to spawning the shell.
- **Reproducible Bootable ISO**: Docker-based UEFI/BIOS GRUB ISO builder (`os-image/build-iso.ps1`) producing a verified ~89.7 MiB bootable image.
- **Linux High-Throughput I/O (`engine-linux`)**: Native `io_uring` + `O_DIRECT` + 2 MiB HugePages runtime layer for low-latency cold-expert streaming from NVMe storage.

### 2. Strata-Aligned CLI & macOS-Inspired Web Console (`inkling2-shell`)
- **Ergonomic CLI**: Direct top-level commands modelled upon `zapabob/Strata`—`launch`, `status`, `ui`, `models`, `bench-tc`, `bench`, and `inspect`.
- **macOS-Style Frosted Glass Console (`GET /`)**: Unified titlebar with segmented navigation (`Chat`, `Tensor Core`, `Memory & LoRA`, `Models & CLI`), system-aware Light/Dark themes, live RTX GPU telemetry (`nvidia-smi`), and one-click CUDA 13.5 `sm_120a` Tensor Core benchmarking.
- **OpenAI-Compatible Loopback Bridge**: Strictly bound to `127.0.0.1` by default, with optional Bearer token authentication (`INKLING2_BRIDGE_TOKEN`), UTF-8 stdin-piped upstream forwarding, and zero arbitrary shell-execution endpoints.

### 3. Companion Memory, Dynamic LoRA & Multimodal Subsystems (`inkling2-companion`)
- **Memory45 (`memory45.rs`)**: Hard-capped **45 MiB** compressed episodic memory store with automatic importance-weighted compaction and atomic persistence.
- **Online Dynamic LoRA (`dynamic_lora.rs`)**: Rank-16 online adapter (`D=2304`, 73,728 parameters) with reward-driven reinforcement and single-step rollback snapshots.
- **Voice & Vision Pipelines**: Local TTS integration (`voice_call.rs`), 40×40-patch-aligned webcam capture (`camera_call.rs`), and optional `hakua-memory` Python bridge managed via `uv`.
- **Hardened Networking**: TLS 1.2+ SSRF-guarded `net fetch`, allowlisted Hugging Face Hub model pulls (`net hf-pull`), default-deny inbound `nftables` ruleset, and VM-initiated SSH tunnel tooling (`muse`).

---

## Quick Start

### Build & Test (Rust 2024 · Zero Warnings)
```powershell
cargo test --workspace
cargo build --release --workspace
```

### Launch the Strata-Aligned macOS Web Console
```powershell
# Start the loopback bridge and open http://127.0.0.1:8080/ in your default browser
.\target\release\inkling2-shell.exe launch --port 8080 --open

# Inspect live bridge health and RTX 5060 Ti CUDA 13.5 telemetry
.\target\release\inkling2-shell.exe status --port 8080

# List discovered SC117 .strata v1.1 models
.\target\release\inkling2-shell.exe models

# Execute the native CUDA 13.5 sm_120a Tensor Core benchmark
.\target\release\inkling2-shell.exe bench-tc
```

### Build the Bootable UEFI/BIOS ISO
```powershell
.\os-image\build-iso.ps1
```

---

## Security Posture

- **Loopback-Only Binding**: `inkling2-shell serve` and `launch` refuse non-loopback bind addresses (`127.0.0.1` / `::1` only). Remote access is intended solely via encrypted SSH port forwarding.
- **SSRF Protection**: `net fetch` enforces HTTPS (TLS 1.2+) and blocks loopback, link-local, RFC 1918 private ranges, and cloud metadata endpoints.
- **Strict Rust Safety**: Workspace configured with `warnings = "deny"`, `unwrap_used = "deny"`, and `expect_used = "deny"`.

---

## Repository Layout

```text
crates/
  inkling2-init/        Rust 2024 PID 1 boot supervisor & GPU VRAM planner
  inkling2-companion/   Memory45 (45 MiB), Dynamic LoRA, Voice TTS, Camera pipeline
  inkling2-shell/       Strata-aligned CLI, HTTP/OpenAI bridge, SSRF-guarded net client
ui/                     Embedded macOS-style Strata Web Console (single HTML bundle)
engine-linux/           Linux io_uring + O_DIRECT + 2 MiB HugePages CUDA runtime
os-image/               Docker UEFI/BIOS ISO builder, GRUB config, kernel fragment, nftables
python/                 Optional hakua-memory bridge (managed via uv)
scripts/                VM-initiated SSH tunnel & state synchronisation utilities
DESIGN.md               Formal macOS-inspired design token specification
```

---

## Dedicated Inference Engine

For the standalone C++20 / CUDA 13.5 (`sm_120a`) Blackwell Tensor Core inference engine, kindly refer to **[`inkling2-cuda-engine`](https://github.com/zapabob/inkling2-cuda-engine)**.

## Licence

Distributed under the MIT Licence.
