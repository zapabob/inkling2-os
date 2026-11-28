# Changelog

## 1.1.0 for Windows

Inkling2 Desktop and the CUDA engine now share a release version. Both executables report it with `--version`. The Windows bridge also reports it at `/health`. This release includes the SC117 tokenizer, up to 131,072 tokens of context and native function selection for local Hermes Agent conversations.

## v0.1.1 development candidate

The Windows console has scoped authentication, protected file previews and a public HTTPS text browser. The standalone Windows Inkling2 CUDA Engine has a dedicated, authenticated local provider for Strata. Admission rejects requests when the GPU measurement is absent or its free memory falls below the conservative model budget. Chat no longer stores prompts implicitly or returns synthetic assistant replies when inference fails.

The Linux image now builds a Casper initramfs and a userspace root with its runtime libraries. PID 1 runs the loopback bridge under an unprivileged account and checks its health. The build recipe and QEMU verifier are subject to [the validation record](docs/V0.1.1_VALIDATION.md).

## v0.1.0

Initial release with a Rust console, companion utilities, a Linux runtime library, and ISO packaging. See the [baseline audit](docs/V0.1.0_AUDIT.md).
