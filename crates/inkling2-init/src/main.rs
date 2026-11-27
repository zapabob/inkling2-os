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

use std::env;
use std::process::{Command, ExitCode};

use boot::{BootMode, run_early_boot};
use gpu_guard::{GpuPolicy, apply_gpu_policy, compute_tier_plan};
use net::{NetPolicy, init_network};

fn main() -> ExitCode {
    let args: Vec<String> = env::args().collect();
    let dry_run = args.iter().any(|a| a == "--dry-run" || a == "--simulate") || cfg!(not(target_os = "linux"));

    let mode = if dry_run {
        BootMode::DryRunSimulation
    } else {
        BootMode::LiveLinuxPid1
    };

    os_info!("PID1", "=== Inkling2-OS v0.1.0 (CUDA 13.5 sm_120a AI-Native Linux Appliance) ===");

    let report = match run_early_boot(mode) {
        Ok(r) => r,
        Err(e) => {
            os_error!("PID1", "Fatal early boot error: {e}");
            return ExitCode::FAILURE;
        }
    };

    let policy = GpuPolicy::default();
    apply_gpu_policy(policy, mode);

    let net_rep = init_network(NetPolicy::default(), mode);
    os_info!(
        "PID1",
        "Network ready: firewall={} lo={} nics={:?} ntp_tls={}",
        net_rep.firewall_loaded,
        net_rep.loopback_up,
        net_rep.interfaces,
        net_rep.ntp_synced
    );

    // Log deterministic VRAM + 2 MiB HugePages tier plans for the SC117 SQ-10 / SQ-15 / SQ-20 series
    for (tier, size_mib) in [("SQ-10", 9_986u64), ("SQ-15", 13_685u64), ("SQ-20", 18_453u64)] {
        let plan = compute_tier_plan(policy, size_mib, report.hugepages_2m);
        os_info!(
            "PID1",
            "Tier Plan [{tier} ~{size_mib} MiB]: UsableVRAM={} MiB ({}% resident), HugePageRAM={} MiB",
            plan.usable_vram_mib,
            plan.estimated_vram_residency_pct,
            plan.hugepage_ram_mib
        );
    }

    if mode == BootMode::DryRunSimulation {
        os_info!("PID1", "Dry-run boot verification complete. Ready for live Linux ISO execution.");
        return ExitCode::SUCCESS;
    }

    // Live Linux PID 1 supervision loop: spawn inkling2-shell on tty1 and restart on exit.
    os_info!("PID1", "Launching /usr/bin/inkling2-shell on primary console...");
    loop {
        match Command::new("/usr/bin/inkling2-shell").status() {
            Ok(status) => {
                os_warn!("PID1", "inkling2-shell exited with {status}; respawning in 1s...");
                std::thread::sleep(std::time::Duration::from_secs(1));
            }
            Err(e) => {
                os_error!("PID1", "Failed to spawn /usr/bin/inkling2-shell: {e}; falling back to /bin/sh");
                let _ = Command::new("/bin/sh").status();
                std::thread::sleep(std::time::Duration::from_secs(2));
            }
        }
    }
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
