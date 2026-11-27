//! GPU Hardware Supervisor for NVIDIA GeForce RTX 5060 Ti 16GB (Blackwell sm_120a) under WDDM-free Linux.
//!
//! Because Inkling2-OS runs with zero X11/Wayland compositor and zero WDDM overhead,
//! the entire 16,311 MiB GDDR7 VRAM is available exclusively to the Inkling-2-Mini CUDA 13.5 engine.

use std::process::Command;

use crate::boot::BootMode;
use crate::{os_info, os_warn};

/// Target GPU policy for Inkling2-OS on RTX 5060 Ti 16GB.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GpuPolicy {
    /// Target CUDA device index.
    pub device_index: u32,
    /// Power limit in watts (RTX 5060 Ti TDP = 180W).
    pub power_limit_watts: u32,
    /// Enable exclusive-process compute mode (blocks accidental multi-tenant VRAM fragmentation).
    pub exclusive_compute: bool,
    /// Reserve MiB for CUDA driver context + K8V8 KV cache headroom.
    pub driver_kv_reserve_mib: u64,
    /// Total physical GDDR7 VRAM in MiB (RTX 5060 Ti 16GB = 16311 MiB).
    pub total_vram_mib: u64,
}

impl Default for GpuPolicy {
    fn default() -> Self {
        Self {
            device_index: 0,
            power_limit_watts: 180,
            exclusive_compute: true,
            driver_kv_reserve_mib: 1200,
            total_vram_mib: 16_311,
        }
    }
}

/// Planned 3-tier memory budget for SC117 `.strata` builds under WDDM-free Linux.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TierBudgetPlan {
    /// VRAM available for trunk + hot routed experts (MiB).
    pub usable_vram_mib: u64,
    /// 2 MiB HugePage Pinned RAM available for warm experts (MiB).
    pub hugepage_ram_mib: u64,
    /// Estimated VRAM residency percentage (0..=100) for a given `.strata` file size.
    pub estimated_vram_residency_pct: u8,
}

/// Compute the WDDM-free Linux VRAM + HugePage RAM budget plan for a `.strata` model.
pub fn compute_tier_plan(
    policy: GpuPolicy,
    model_size_mib: u64,
    hugepages_2m: u32,
) -> TierBudgetPlan {
    let usable_vram_mib = policy
        .total_vram_mib
        .saturating_sub(policy.driver_kv_reserve_mib);
    let hugepage_ram_mib = u64::from(hugepages_2m) * 2;
    let pct = usable_vram_mib
        .saturating_mul(100)
        .checked_div(model_size_mib)
        .unwrap_or(100)
        .min(100) as u8;
    TierBudgetPlan {
        usable_vram_mib,
        hugepage_ram_mib,
        estimated_vram_residency_pct: pct,
    }
}

/// Apply GPU persistence mode, exclusive compute mode, and power limit via `nvidia-smi`.
pub fn apply_gpu_policy(policy: GpuPolicy, mode: BootMode) {
    let dev = policy.device_index.to_string();
    let pl = policy.power_limit_watts.to_string();

    if mode == BootMode::DryRunSimulation {
        os_info!(
            "GpuGuard",
            "[sim] nvidia-smi -i {dev} -pm 1 && -c EXCLUSIVE_PROCESS && -pl {pl}W (usable_vram={} MiB)",
            policy
                .total_vram_mib
                .saturating_sub(policy.driver_kv_reserve_mib)
        );
        return;
    }

    let commands: &[&[&str]] = &[
        &["-i", &dev, "-pm", "1"],
        &[
            "-i",
            &dev,
            "-c",
            if policy.exclusive_compute {
                "EXCLUSIVE_PROCESS"
            } else {
                "DEFAULT"
            },
        ],
        &["-i", &dev, "-pl", &pl],
    ];

    for args in commands {
        match Command::new("/usr/bin/nvidia-smi").args(*args).status() {
            Ok(s) if s.success() => os_info!("GpuGuard", "nvidia-smi {:?} OK", args),
            Ok(s) => os_warn!("GpuGuard", "nvidia-smi {:?} exited with {s}", args),
            Err(e) => os_warn!("GpuGuard", "nvidia-smi {:?} failed: {e}", args),
        }
    }
}
