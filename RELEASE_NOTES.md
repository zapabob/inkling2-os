# Inkling2-OS 1.1.0 for Windows

Pair the Windows console with the [1.1.0 CUDA engine](https://github.com/zapabob/inkling2-cuda-engine/releases/tag/v1.1.0). A desktop shortcut opens local chat, files and web access. Hermes can send conversations and function definitions to the same key-free loopback endpoint. The engine chooses a declared function, Hermes executes it, and the model can answer with the returned value.

The desktop embeds the SC117 tokenizer and reads the conversation template from the original SQ-10, SQ-15 or SQ-20 weights. The context can hold 131,072 tokens, including up to 128 output tokens. Each tool turn supports one function. Streaming clients receive buffered events when generation completes.

See the [Windows setup and upgrade instructions](README.md#windows-console) and the [published Windows files and checksums](https://github.com/zapabob/inkling2-os/releases/tag/v1.1.0). The public Linux image is distributed under its own tag.
