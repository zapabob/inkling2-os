# Project Baseline: inkling2-os

This project adheres strictly to `C:\Users\downl\AGENTS.md`, `C:\Users\downl\_docs\sop\SOP-Rust.md`, and `C:\Users\downl\_docs\sop\SOP-CPP.md`.

## Architecture
- **PID 1 & AI-OS Shell**: Rust 2024 (`inkling2-init`, `inkling2-shell`), zero `unwrap()` in production, zero `clippy` warnings.
- **Inference Substrate**: WDDM-free Linux CUDA 13.5 (`sm_120a`) native engine with `io_uring` + `O_DIRECT` + `MAP_HUGETLB` (2MB HugePages) + `SCHED_FIFO` + 6C/12T Ryzen CPU isolation.
- **Target Hardware**: AMD Ryzen 6-core/12-thread + 32 GiB RAM + NVIDIA GeForce RTX 5060 Ti 16GB (Blackwell `sm_120a`) + NVMe SSD.
- **Target Models**: SC117 Inkling-2-Mini `.strata` v1.1 (`SQ-10`, `SQ-15`, `SQ-20`).
