# Inkling2 Desktop 1.2.0 for Windows

Download the desktop and matching CUDA engine from the two release pages. The desktop supports SC117 SQ-10, SQ-15 and SQ-20, up to 131,072 tokens of context, up to 1,024 tokens per reply and local Hermes function calls. Long streaming requests receive keepalives during prefill.

The SQ-15 resident profile uses larger expert caches and Windows cached reads. The engine includes new Tensor Core matrix kernels and file-offset ordering for cold reads. See the [setup and qualification guide](README.md) and the [matching engine](https://github.com/zapabob/inkling2-cuda-engine/releases/tag/v1.2.0).
