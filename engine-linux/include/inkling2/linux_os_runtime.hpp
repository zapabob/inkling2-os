#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace inkling2::linux_os {

struct LinuxHardwareProfile {
    std::string gpu_name;
    int cuda_driver_version = 0;
    int cuda_runtime_version = 0;
    int sm_major = 0;
    int sm_minor = 0;
    int sm_count = 0;
    size_t vram_total_bytes = 0;
    size_t vram_free_bytes = 0;
    size_t ram_total_bytes = 0;
    size_t hugepages_free_bytes = 0;
    uint32_t logical_cpus = 0;
    bool rt_fifo_enabled = false;
    bool mlockall_enabled = false;
};

/// WDDM-Free Linux OS-level hardware runtime for Inkling2-OS on RTX 5060 Ti 16GB + 6C/12T Ryzen:
/// - Exclusive 16,311 MiB GDDR7 VRAM (zero WDDM fragmentation / zero desktop compositor tax)
/// - 2 MiB HugePages (`MAP_HUGETLB`) + `cudaHostRegister` for zero-TLB-miss Pinned RAM expert slabs
/// - `mlockall(MCL_CURRENT | MCL_FUTURE)` + `SCHED_FIFO` + `pthread_setaffinity_np` on isolated cores 2..11
/// - Kernel `io_uring` + `O_DIRECT` async NVMe submission ring for SQ-20 cold expert streaming
class LinuxOsRuntime {
public:
    LinuxOsRuntime() = default;
    ~LinuxOsRuntime() = default;

    bool initialize(int rt_priority = 50);
    static bool pin_current_thread(uint32_t logical_core_id);
    LinuxHardwareProfile probe_hardware() const;

    /// Under WDDM-free Linux, up to ~15.1 GiB of RTX 5060 Ti's 16.3 GiB VRAM is directly allocatable.
    size_t recommended_vram_budget_bytes(size_t reserve_mib = 1024) const;

    /// Allocate 2 MiB HugePage-backed pinned host memory (registered with CUDA for DMA).
    static void* alloc_hugepage_pinned(size_t bytes);
    static void free_hugepage_pinned(void* ptr, size_t bytes) noexcept;
};

/// Linux `io_uring` + `O_DIRECT` Direct NVMe Reader for 4096-byte aligned `.strata` v1.1 blobs.
class IoUringNvmeReader {
public:
    IoUringNvmeReader() = default;
    ~IoUringNvmeReader();

    IoUringNvmeReader(const IoUringNvmeReader&) = delete;
    IoUringNvmeReader& operator=(const IoUringNvmeReader&) = delete;

    bool open(const std::string& path, uint32_t queue_depth = 64);
    void close() noexcept;
    uint64_t file_size() const noexcept { return file_size_; }

    struct ReadReq {
        uint64_t offset;
        void* dst_aligned;
        size_t bytes_aligned;
    };

    /// Submit all NVMe reads in a single `io_uring_submit_and_wait` syscall!
    bool read_batch_uring(const std::vector<ReadReq>& reqs) const;

private:
    int fd_ = -1;
    uint64_t file_size_ = 0;
    void* ring_ = nullptr;  // struct io_uring*
};

}  // namespace inkling2::linux_os
