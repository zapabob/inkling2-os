# Hardware compatibility

The target is an x86-64 Linux machine with an AMD Ryzen 6-core/12-thread processor, 32 GiB RAM, a Blackwell RTX 5060 Ti with 16 GiB nominal VRAM, and NVMe storage. The C++ engine target uses CUDA 13.5 and `sm_120a`. The device, usable VRAM and actual driver must be probed at runtime.

The Rust console runs on Windows x64 independently of the CUDA executable. An existing Strata engine can remain loaded while the provider is registered. Testing generation requires enough additional headroom or an orderly unload of the existing model. The stock Linux image does not bundle the Windows CUDA executable.
