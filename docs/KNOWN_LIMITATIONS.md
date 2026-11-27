# Known limitations

The Windows adapter runs the native process per admitted request. It buffers the result before emitting OpenAI-compatible events and supports text only. A real SQ-10 request returned repeated punctuation. Source inspection found an approximate tokenizer and generic prompt framing in the current stack. Their contributions to the failure require isolation against reference token IDs and logits.

The Linux runtime library lacks a packaged Linux inference executable. GPU-backed boot, runtime telemetry, expert placement, scheduler profiles and comparative benchmarks require further work. The ISO’s companion state is volatile. Dynamic LoRA remains an adapter-state experiment, without a confirmed model-weight update path.

The built-in Browser is restricted to public IPv4 HTTPS text retrieval. It cannot follow redirects or display interactive pages. Finder requires an explicitly configured preview directory and exposes no file mutation operation.
