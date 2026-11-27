//! Network & Firewall Bring-Up for Inkling2-OS PID 1.
//!
//! Security & Reliability controls (`SOP-Internet-Connectivity` & `SOP-Security`):
//! 1. Load default-deny `nftables` ruleset BEFORE bringing up any external NIC.
//! 2. Bring up loopback (`lo`) and discover Ethernet (`eth*`/`en*`) or Wi-Fi (`wl*`) interfaces.
//! 3. Acquire IP/DNS via DHCP with bounded timeout (never blocks boot indefinitely).
//! 4. Sync system clock via NTP (`chronyd -q`) so TLS certificate verification succeeds.
//! 5. Degraded-mode fallback: if offline, log actionable warning and proceed into local inference mode.

use std::fs;
use std::path::Path;
use std::process::Command;

use crate::boot::BootMode;
use crate::{os_info, os_warn};

/// Network bring-up policy for PID 1.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NetPolicy {
    /// Load default-deny inbound `nftables` ruleset before enabling NICs.
    pub enforce_firewall: bool,
    /// DHCP acquisition timeout in seconds.
    pub dhcp_timeout_secs: u32,
    /// Attempt one-shot NTP sync after acquiring link so TLS cert validation works.
    pub sync_ntp_for_tls: bool,
}

impl Default for NetPolicy {
    fn default() -> Self {
        Self {
            enforce_firewall: true,
            dhcp_timeout_secs: 10,
            sync_ntp_for_tls: true,
        }
    }
}

/// Summary of PID 1 network initialization.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NetReport {
    /// Whether the default-deny inbound firewall was loaded.
    pub firewall_loaded: bool,
    /// Whether loopback `lo` is up.
    pub loopback_up: bool,
    /// Detected or simulated external NIC names.
    pub interfaces: Vec<String>,
    /// Whether NTP clock sync was triggered for TLS.
    pub ntp_synced: bool,
}

/// Discover non-loopback network interfaces from `/sys/class/net` on Linux.
fn discover_linux_nics() -> Vec<String> {
    let mut nics = Vec::new();
    let Ok(entries) = fs::read_dir("/sys/class/net") else {
        return nics;
    };
    for entry in entries.flatten() {
        let name = entry.file_name().to_string_lossy().into_owned();
        if name != "lo"
            && (name.starts_with("en") || name.starts_with("eth") || name.starts_with("wl"))
        {
            nics.push(name);
        }
    }
    nics.sort();
    nics
}

/// Execute Inkling2-OS PID 1 network & firewall bring-up.
pub fn init_network(policy: NetPolicy, mode: BootMode) -> NetReport {
    if mode == BootMode::DryRunSimulation {
        os_info!(
            "NetInit",
            "[sim] nft -f /etc/inkling2/nftables.conf (default-deny ingress and egress)"
        );
        os_info!(
            "NetInit",
            "[sim] ip link set lo up && ip addr add 127.0.0.1/8 dev lo"
        );
        os_info!(
            "NetInit",
            "[sim] dhcpcd --timeout {} (Ethernet/Wi-Fi) + /etc/ssl/certs/ca-certificates.crt",
            policy.dhcp_timeout_secs
        );
        if policy.sync_ntp_for_tls {
            os_info!(
                "NetInit",
                "[sim] chronyd -q (NTP sync for TLS cert validation)"
            );
        }
        return NetReport {
            firewall_loaded: policy.enforce_firewall,
            loopback_up: true,
            interfaces: vec!["eth0".to_string()],
            ntp_synced: policy.sync_ntp_for_tls,
        };
    }

    // 1. Load default-deny inbound firewall BEFORE exposing any external NIC.
    let mut firewall_loaded = false;
    if policy.enforce_firewall && Path::new("/etc/inkling2/nftables.conf").exists() {
        match Command::new("/usr/sbin/nft")
            .args(["-f", "/etc/inkling2/nftables.conf"])
            .status()
        {
            Ok(s) if s.success() => {
                firewall_loaded = true;
                os_info!(
                    "NetInit",
                    "Loaded default-deny ingress and egress nftables firewall"
                );
            }
            Ok(s) => os_warn!("NetInit", "nftables returned {s}"),
            Err(e) => os_warn!("NetInit", "nftables unavailable: {e}"),
        }
    }

    // 2. Bring up loopback
    let loopback_up = Command::new("/sbin/ip")
        .args(["link", "set", "lo", "up"])
        .status()
        .map(|s| s.success())
        .unwrap_or(false);

    if policy.enforce_firewall && !firewall_loaded {
        os_warn!(
            "NetInit",
            "Firewall unavailable; external interface setup skipped"
        );
        return NetReport {
            firewall_loaded,
            loopback_up,
            interfaces: Vec::new(),
            ntp_synced: false,
        };
    }

    // The installed appliance has no running udev daemon at this point.
    if Path::new("/etc/inkling2/model-volume").is_file()
        && !Command::new("/sbin/modprobe")
            .arg("r8169")
            .status()
            .is_ok_and(|status| status.success())
    {
        os_warn!("NetInit", "Realtek RTL8168 Ethernet driver could not load");
    }
    // 3. Discover NICs and configure DHCP / Wi-Fi
    let interfaces = discover_linux_nics();
    if interfaces.is_empty() {
        os_warn!(
            "NetInit",
            "No external NICs found; continuing in offline local-inference mode"
        );
        return NetReport {
            firewall_loaded,
            loopback_up,
            interfaces,
            ntp_synced: false,
        };
    }

    let timeout_str = policy.dhcp_timeout_secs.to_string();
    for nic in &interfaces {
        let _ = Command::new("/sbin/ip")
            .args(["link", "set", nic, "up"])
            .status();
        if nic.starts_with("wl") && Path::new("/etc/inkling2/wifi.conf").exists() {
            let _ = Command::new("/sbin/wpa_supplicant")
                .args(["-B", "-i", nic, "-c", "/etc/inkling2/wifi.conf"])
                .status();
            os_info!("NetInit", "Started wpa_supplicant on {nic}");
        }
        match Command::new("/sbin/dhcpcd")
            .args([
                "--timeout",
                &timeout_str,
                "--background",
                "--nohook",
                "resolv.conf",
                nic,
            ])
            .status()
        {
            Ok(s) if s.success() => os_info!("NetInit", "DHCP started on {nic}"),
            Ok(s) => os_warn!("NetInit", "dhcpcd on {nic} exited with {s}"),
            Err(e) => os_warn!("NetInit", "dhcpcd unavailable for {nic}: {e}"),
        }
    }

    // 4. Sync clock via NTP for TLS certificate verification
    let mut ntp_synced = false;
    if policy.sync_ntp_for_tls
        && let Ok(s) = Command::new("/usr/sbin/chronyd")
            .args(["-q", "-u", "root", "-t", "10"])
            .status()
    {
        ntp_synced = s.success();
    }

    NetReport {
        firewall_loaded,
        loopback_up,
        interfaces,
        ntp_synced,
    }
}
