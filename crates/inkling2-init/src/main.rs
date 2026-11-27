//! Inkling2-OS PID 1 (`inkling2-init`)
//!
//! Custom minimalist Linux PID 1 replacing systemd, X11/Wayland, and desktop daemons.
//! Boots directly into:
//!   1. Kernel mounts + 2 MiB HugePages + NVIDIA Open Kernel Modules + 6C/12T Ryzen CPU isolation
//!   2. Exclusive RTX 5060 Ti 16GB GPU policy (no WDDM overhead!)
//!   3. `inkling2-engine` CUDA 13.5 sm_120a inference daemon + `inkling2-shell` AI-Native console

mod boot;
mod gpu_guard;
mod logger;
mod net;
#[cfg(target_os = "linux")]
mod supervisor;

use std::env;
use std::process::{self, ExitCode};

use boot::{BootMode, run_early_boot};
use gpu_guard::{GpuPolicy, apply_gpu_policy, compute_tier_plan};
use net::{NetPolicy, init_network};

fn main() -> ExitCode {
    let args: Vec<String> = env::args().collect();
    let dry_run = args.iter().any(|a| a == "--dry-run" || a == "--simulate")
        || cfg!(not(target_os = "linux"));

    if !dry_run && process::id() != 1 {
        os_error!(
            "PID1",
            "Live boot requires Linux PID 1; use --dry-run to inspect the boot plan"
        );
        return ExitCode::FAILURE;
    }
    let mode = if dry_run {
        BootMode::DryRunSimulation
    } else {
        BootMode::LiveLinuxPid1
    };

    os_info!(
        "PID1",
        "=== Inkling2-OS v0.1.1 (CUDA 13.5 sm_120a AI-Native Linux Appliance) ==="
    );

    let report = match run_early_boot(mode) {
        Ok(r) => r,
        Err(e) => {
            os_error!("PID1", "Fatal early boot error: {e}");
            return ExitCode::FAILURE;
        }
    };

    let policy = GpuPolicy::default();
    if dry_run {
        apply_gpu_policy(policy, mode);
    } else {
        let root_type = std::fs::read_to_string("/proc/self/mountinfo")
            .ok()
            .and_then(|mounts| {
                mounts.lines().find_map(|line| {
                    let (left, right) = line.split_once(" - ")?;
                    (left.split_whitespace().nth(4) == Some("/"))
                        .then(|| right.split_whitespace().next().map(str::to_owned))
                        .flatten()
                })
            })
            .unwrap_or_else(|| "unknown".into());
        os_info!(
            "PID1",
            "Root filesystem active: {root_type}; PID={}",
            process::id()
        );
        let gpu = process::Command::new("/usr/bin/nvidia-smi")
            .args([
                "--query-gpu=name,memory.free,memory.total",
                "--format=csv,noheader",
            ])
            .output();
        if let Some(gpu) = gpu.ok().filter(|output| output.status.success()) {
            os_info!(
                "PID1",
                "GPU revealed by the driver: {}",
                String::from_utf8_lossy(&gpu.stdout).trim()
            );
        } else {
            os_warn!("PID1", "GPU unavailable; inference is in diagnostic mode");
        }
    }

    let net_rep = init_network(NetPolicy::default(), mode);
    os_info!(
        "PID1",
        "Network ready: firewall={} lo={} nics={:?} ntp_tls={}",
        net_rep.firewall_loaded,
        net_rep.loopback_up,
        net_rep.interfaces,
        net_rep.ntp_synced
    );

    if dry_run {
        // Log deterministic VRAM + 2 MiB HugePages tier plans for the SC117 SQ-10 / SQ-15 / SQ-20 series
        for (tier, size_mib) in [
            ("SQ-10", 9_986u64),
            ("SQ-15", 13_685u64),
            ("SQ-20", 18_453u64),
        ] {
            let plan = compute_tier_plan(policy, size_mib, report.hugepages_2m);
            os_info!(
                "PID1",
                "Tier Plan [{tier} ~{size_mib} MiB]: UsableVRAM={} MiB ({}% resident), HugePageRAM={} MiB",
                plan.usable_vram_mib,
                plan.estimated_vram_residency_pct,
                plan.hugepage_ram_mib
            );
        }

        os_info!("PID1", "Dry-run boot planning finished");
        return ExitCode::SUCCESS;
    }

    #[cfg(target_os = "linux")]
    supervisor::supervise();
    #[cfg(not(target_os = "linux"))]
    ExitCode::FAILURE
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dry_run_early_boot_succeeds() {
        let rep = run_early_boot(BootMode::DryRunSimulation);
        assert!(rep.is_ok());
        if let Ok(r) = rep {
            assert_eq!(r.mounts_applied, boot::EARLY_MOUNTS.len());
            assert_eq!(r.modules_loaded, boot::KERNEL_MODULES.len());
            assert_eq!(r.hugepages_2m, 8192);
        }
    }

    #[test]
    fn wddm_free_vram_plans_for_sc117_series() {
        let pol = GpuPolicy::default();
        // SQ-10 (~10 GB): 100% resident in 16 GB RTX 5060 Ti without WDDM!
        let sq10 = compute_tier_plan(pol, 9_986, 8192);
        assert_eq!(sq10.estimated_vram_residency_pct, 100);
        // SQ-15 (~13.7 GB): 100% resident in 16 GB RTX 5060 Ti without WDDM!
        let sq15 = compute_tier_plan(pol, 13_685, 8192);
        assert_eq!(sq15.estimated_vram_residency_pct, 100);
        // SQ-20 (~18.45 GB): ~81% resident in 16 GB VRAM + remaining ~3.3 GB in 16 GB 2 MiB HugePages!
        let sq20 = compute_tier_plan(pol, 18_453, 8192);
        assert!(sq20.estimated_vram_residency_pct >= 80);
        assert_eq!(sq20.hugepage_ram_mib, 16_384);
    }
}
