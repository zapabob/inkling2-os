# NEON architecture

The Windows 1.3 desktop runs a loopback Rust bridge and a separate resident CUDA worker. Hermes uses Chat Completions for inference. Console-authorised requests can inspect the system, submit explicit observations or request speech. In the default key-free mode, local processes can use those console routes. Optional token mode distinguishes console from inference authority.

| Component | Behaviour in this release |
|---|---|
| Perception | The Rust perception crate provides a bounded event bus, camera discovery, motion gating, audio energy detection and helpers for describing observations. A single capture needs operator opt-in and a local capture backend. The bridge discards the temporary JPEG and returns a capture event. SC117 receives text tokens. |
| Voice | The bridge queues sentences through loopback VOICEVOX or Windows SAPI. Stop takes effect between sentences. Streaming chat sends keepalives and delivers the completed response. |
| Agent | The registry describes typed skills. HTTP currently executes `sys.info` and `perception.status`. Tasks are deduplicated by ID. Results and verification come from the executing handler. An unavailable handler returns HTTP 501. |
| Neural Memory45 | Explicit feedback stores text with a source ID and provenance. HTTP feedback is unverified and cannot authorise a fast-weight update. Snapshots support integrity checks and reconstruction on startup. The 45 MiB budget governs the snapshot layout. |
| Learning | Reinforcement scoring is advisory. Anti-forgetting evaluation requires real baseline and candidate scores. The HTTP learning endpoint reports unevaluated comparisons as null. |
| Dynamic LoRA | The companion library supports low-rank consolidation. A separate native manager in the CUDA worker validates adapters and can add a bounded residual before final RMS normalisation. Activation uses the resident engine's own protocol. HTTP learning and rollback affect companion state. |

The original SC117 model file remains read-only. Model-specific evaluation is required to measure what a trained adapter changes. The [Hermes integration guide](HERMES_INTEGRATION.md) lists routes and permission boundaries. See the [memory format](NEURAL_MEMORY45.md) for admission, replay and storage limits.
