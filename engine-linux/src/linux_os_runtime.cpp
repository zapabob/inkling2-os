#include "inkling2/linux_os_runtime.hpp"

#ifdef __linux__

#include <cuda_runtime.h>
#include <fcntl.h>
#include <liburing.h>
#include <pthread.h>
#include <sched.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cstdio>
#include <cstring>

namespace inkling2::linux_os {

namespace {
constexpr size_t kHugePage2M = 2ULL * 1024ULL * 1024ULL;
}  // namespace

bool LinuxOsRuntime::initialize(int rt_priority) {
    // 1. Lock process memory into RAM (prevents any pageout during SQ-20 streaming).
    const bool locked = (mlockall(MCL_CURRENT | MCL_FUTURE) == 0);
    (void)locked;

    // 2. Request SCHED_FIFO real-time scheduling on isolated inference cores (2..11).
    sched_param sp{};
    sp.sched_priority = rt_priority;
    (void)sched_setscheduler(0, SCHED_FIFO, &sp);

    // 3. Configure CUDA 13.5 device for lowest-latency sync without WDDM overhead.
    cudaError_t ce = cudaSetDeviceFlags(cudaDeviceScheduleYield | cudaDeviceMapHost);
    if (ce != cudaSuccess && ce != cudaErrorSetOnActiveProcess) {
        return false;
    }
    return cudaSetDevice(0) == cudaSuccess;
}

bool LinuxOsRuntime::pin_current_thread(uint32_t logical_core_id) {
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(logical_core_id, &set);
    return pthread_setaffinity_np(pthread_self(), sizeof(set), &set) == 0;
}

LinuxHardwareProfile LinuxOsRuntime::probe_hardware() const {
    LinuxHardwareProfile hp{};
    hp.logical_cpus = static_cast<uint32_t>(sysconf(_SC_NPROCESSORS_ONLN));
    hp.ram_total_bytes = static_cast<size_t>(sysconf(_SC_PHYS_PAGES)) *
                         static_cast<size_t>(sysconf(_SC_PAGE_SIZE));

    cudaDriverGetVersion(&hp.cuda_driver_version);
    cudaRuntimeGetVersion(&hp.cuda_runtime_version);
    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, 0) == cudaSuccess) {
        hp.gpu_name = prop.name;
        hp.sm_major = prop.major;
        hp.sm_minor = prop.minor;
        hp.sm_count = prop.multiProcessorCount;
    }
    cudaMemGetInfo(&hp.vram_free_bytes, &hp.vram_total_bytes);
    return hp;
}

size_t LinuxOsRuntime::recommended_vram_budget_bytes(size_t reserve_mib) const {
    size_t free_b = 0, total_b = 0;
    if (cudaMemGetInfo(&free_b, &total_b) != cudaSuccess) {
        return 0;
    }
    const size_t reserve_b = reserve_mib * 1024ULL * 1024ULL;
    // WDDM-free Linux allows clean single/multi-slab allocation up to ~95% of physical GDDR7!
    return (free_b > reserve_b) ? (free_b - reserve_b) : 0;
}

void* LinuxOsRuntime::alloc_hugepage_pinned(size_t bytes) {
    const size_t aligned = (bytes + kHugePage2M - 1) & ~(kHugePage2M - 1);
    void* ptr = mmap(nullptr, aligned, PROT_READ | PROT_WRITE,
                     MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB | MAP_POPULATE, -1, 0);
    if (ptr == MAP_FAILED) {
        // Fallback to standard 4KB-aligned pinned CUDA host memory if HugePages are exhausted.
        if (cudaHostAlloc(&ptr, aligned, cudaHostAllocDefault) != cudaSuccess) {
            return nullptr;
        }
        return ptr;
    }
    if (cudaHostRegister(ptr, aligned, cudaHostRegisterDefault) != cudaSuccess) {
        munmap(ptr, aligned);
        return nullptr;
    }
    return ptr;
}

void LinuxOsRuntime::free_hugepage_pinned(void* ptr, size_t bytes) noexcept {
    if (ptr == nullptr) return;
    const size_t aligned = (bytes + kHugePage2M - 1) & ~(kHugePage2M - 1);
    if (cudaHostUnregister(ptr) == cudaSuccess) {
        munmap(ptr, aligned);
    } else {
        cudaFreeHost(ptr);
    }
}

// ---------------------------------------------------------------------------
// IoUringNvmeReader
// ---------------------------------------------------------------------------

IoUringNvmeReader::~IoUringNvmeReader() {
    close();
}

bool IoUringNvmeReader::open(const std::string& path, uint32_t queue_depth) {
    close();
    fd_ = ::open(path.c_str(), O_RDONLY | O_DIRECT | O_CLOEXEC);
    if (fd_ < 0) {
        return false;
    }
    struct stat st{};
    if (fstat(fd_, &st) != 0) {
        close();
        return false;
    }
    file_size_ = static_cast<uint64_t>(st.st_size);

    auto* r = new io_uring{};
    if (io_uring_queue_init(queue_depth, r, IORING_SETUP_SQPOLL) != 0) {
        // Fallback without SQPOLL if unprivileged
        if (io_uring_queue_init(queue_depth, r, 0) != 0) {
            delete r;
            close();
            return false;
        }
    }
    ring_ = r;
    return true;
}

void IoUringNvmeReader::close() noexcept {
    if (ring_ != nullptr) {
        auto* r = static_cast<io_uring*>(ring_);
        io_uring_queue_exit(r);
        delete r;
        ring_ = nullptr;
    }
    if (fd_ >= 0) {
        ::close(fd_);
        fd_ = -1;
    }
    file_size_ = 0;
}

bool IoUringNvmeReader::read_batch_uring(const std::vector<ReadReq>& reqs) const {
    if (fd_ < 0 || ring_ == nullptr || reqs.empty()) {
        return reqs.empty();
    }
    auto* r = static_cast<io_uring*>(ring_);
    for (size_t i = 0; i < reqs.size(); ++i) {
        io_uring_sqe* sqe = io_uring_get_sqe(r);
        if (sqe == nullptr) return false;
        io_uring_prep_read(sqe, fd_, reqs[i].dst_aligned,
                           static_cast<unsigned>(reqs[i].bytes_aligned), reqs[i].offset);
        io_uring_sqe_set_data64(sqe, i);
    }

    const int submitted = io_uring_submit_and_wait(r, static_cast<unsigned>(reqs.size()));
    if (submitted < static_cast<int>(reqs.size())) {
        return false;
    }

    bool ok = true;
    for (size_t i = 0; i < reqs.size(); ++i) {
        io_uring_cqe* cqe = nullptr;
        if (io_uring_wait_cqe(r, &cqe) != 0 || cqe == nullptr) {
            ok = false;
            break;
        }
        const size_t idx = static_cast<size_t>(io_uring_cqe_get_data64(cqe));
        if (cqe->res < 0 || static_cast<size_t>(cqe->res) != reqs[idx].bytes_aligned) {
            ok = false;
        }
        io_uring_cqe_seen(r, cqe);
    }
    return ok;
}

}  // namespace inkling2::linux_os

#endif  // __linux__
