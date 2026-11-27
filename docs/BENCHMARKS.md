# Benchmarks

Windows reported roughly 3.0–3.4 GiB free of 16,311 MiB during the initial Strata workload. Admission rejected the request before the standalone engine started. A later run, with enough free VRAM, returned an eight-token response after 91.46 seconds including model load. The repeated punctuation failed the correctness check. Throughput, per-token latency, power and long-context results require further measurement.

Run `scripts/benchmark-native-api.ps1` with the actual model ID, a UTF-8 prompt file and a new output path. It records per-run wall time, the pre-run GPU observation and HTTP errors. The adapter currently returns complete results, so this measures round-trip time. TTFT, output token rate and per-token latency need an engine that exposes those measurements.

Use the same weights, quantisation, prompt, token limit, cache state and engine revision for Windows Strata, conventional Linux and the appliance. Save each run’s GPU and driver identifiers, free and peak memory, exact launch options, and output token count. Compare warm-up and measured runs independently. Record TTFT, prefill and decode rates, p50/p95 token time, peak RSS/VRAM, cache hits, NVMe bytes, power, errors and correctness by context tier. Preserve the raw responses and timing logs before aggregating.
