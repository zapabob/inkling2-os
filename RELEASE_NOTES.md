# Inkling2 Desktop 1.4.0 for Windows

The Windows desktop discovers the 16 KiB SQv2 CHiPS SQ-20 pack alongside classic SC117 SQ-10, SQ-15 and SQ-20. The configured resident model is selected when a new desktop opens. Existing local weights can be used in place.

Pair the console with the [1.4.0 CUDA engine](https://github.com/zapabob/inkling2-cuda-engine/releases/tag/v1.4.0). The CHiPS resident profile uses GPU and host memory for routed weights, CUDA projection graphs and compensated BF16 Tensor Core kernels. Expert placement and startup time depend on available memory and storage speed. The console archive includes the tokenizer. Supply your own model weights. See the [Windows installation guide](https://github.com/zapabob/inkling2-os/blob/v1.4.0/README.md#windows-console).
