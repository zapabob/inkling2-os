# Inkling2-OS

Inkling2-OS is a Linux appliance and Windows console for local SC117 Inkling-2-Mini inference. The v0.1.1 development branch concentrates on the live-image boot path, a supervised console, and a small authenticated API for the separate Inkling2 CUDA Engine.

| Status | Capability |
| --- | --- |
| Implemented | Loopback console, separate console and inference credentials, read-only document previews, public HTTPS text view and measured-GPU admission. |
| Verified | Windows console, scoped API and Strata 0.8.6 provider registration. QEMU BIOS and UEFI boot, PID 1, service restart and shutdown. |
| Experimental | Short text inference using the standalone Windows engine's stdin protocol. The OpenAI-compatible event stream emits the answer once generation finishes. |
| Pending qualification | Physical Linux CUDA inference, 65,536 and 131,072 token contexts, and matched hardware benchmarks. |
| Future work | Integrated Linux CUDA executable, phase-aware scheduling, measured expert-tier policy and validated online LoRA. |

## Windows console

Build with `cargo build --release --locked --workspace -j 1`. From a Windows bundle, the launcher selects the packaged executable automatically. In PowerShell 7, run:

```powershell
./scripts/start-windows-console.ps1 -EnginePath 'C:/Models/bin/inkling2-engine.exe' -ModelsDir 'H:/Models/SC117-SQ-10' -PreviewDir ./docs -Open
```

The default address is `http://127.0.0.1:18081/`. Double-click `scripts/copy-console-token.cmd`, or run `./scripts/start-windows-console.ps1 -CopyToken Console` in PowerShell. Paste the token into **Unlock**. The tab holds the token in memory. Close or lock the tab when finished.

In Strata, add a custom OpenAI Chat Completions connection with base URL `http://127.0.0.1:18081/v1`. Use the **Inference** token and the exact model ID returned by `GET /v1/models`. Choose the added model for a new chat. Model files require the matching `tokenizer.json` beside them. GGUF remains available through Strata’s own engines.

## Linux ISO

Run `./os-image/build-iso.ps1` on an amd64 Docker host. Use the supplied QEMU verification target before booting physical hardware. The image is a live appliance and does not include NVIDIA binaries, model weights, or an installer.

Read the [architecture](docs/ARCHITECTURE.md), [boot process](docs/BOOT_PROCESS.md), [security model](docs/SECURITY.md), [hardware support](docs/HARDWARE_COMPATIBILITY.md), [limitations](docs/KNOWN_LIMITATIONS.md) and [validation record](docs/V0.1.1_VALIDATION.md).

## Licence

Distributed under the MIT Licence. The [standalone Inkling2 CUDA Engine](https://github.com/zapabob/inkling2-cuda-engine) is a separate project.
