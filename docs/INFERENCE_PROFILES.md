# Inference profiles

The candidate has one admission policy and one in-flight native request. It accepts text prompts and explicit output budgets from 1 to 128 tokens. Native prompt length is limited to 1,536 UTF-8 bytes, including framing, until the engine provides reliable context accounting.

Eco, Balanced, Performance and Deep Reasoning remain planned profiles. Each profile requires measured bounds for its KV format, expert residency, prefill batch, CPU affinity and I/O queue.
