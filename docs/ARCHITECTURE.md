# Architecture

Linux boots via GRUB, the Ubuntu kernel and a Casper initramfs. Casper mounts SquashFS with a writable overlay and hands off to the Rust PID 1. PID 1 inspects hardware, sets up the loopback interface and firewall, and supervises the least-privilege bridge. The live overlay and the service state under `/run` are volatile.

On Windows, the same bridge hosts the local console. An inference-only bearer grants access to model discovery and chat completions. Strata connects to that interface as a custom chat provider. The bridge validates the selected model, measures free VRAM, and admits one native request at a time. It passes only the configured executable path and resolved model path to the command line. The prompt and output travel through pipes.

The platform-specific Linux I/O library, the standalone Windows inference engine and the browser console have separate lifecycle and qualification requirements.
