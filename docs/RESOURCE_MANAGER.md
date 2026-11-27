# Resource manager

The Rust resource crate sums the seven budget terms using checked integer arithmetic. The caller supplies a fresh free-VRAM measurement from device 0. Measurements that are missing or below the configured requirement cause admission to fail before the engine starts.

The Windows adapter counts the entire file as the weight bound. It allows 768 MiB for estimated KV cache, activations, CUDA workspace and runtime, plus the standalone engine’s 1,536 MiB reserve. The allocation model has yet to be calibrated against the executable’s measured peak. The Linux boot code reports the configured 2 MiB HugePages from sysfs and leaves their reservation to the platform operator.

No admission path changes model precision or the requested output limit. Dynamic tier assignment, expert telemetry and context-aware placement depend on a real engine contract and remain open.
