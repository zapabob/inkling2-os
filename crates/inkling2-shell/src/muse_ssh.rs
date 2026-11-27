//! Hardened OpenSSH Bridge for Meta Muse.ai Cloud VM Integration (`muse_ssh`).
//!
//! Security controls (`SOP-Security` / `SOP-Internet-Connectivity`):
//! - Key-only authentication (`BatchMode=yes`, `PasswordAuthentication=no`, `PubkeyAuthentication=yes`)
//! - Strict target validation (rejects shell metacharacters & malformed host strings)
//! - Encrypted loopback-only reverse tunnels (`-R 127.0.0.1:<remote>:127.0.0.1:<local>`)
//! - Explicit connect timeouts & keepalives (`ConnectTimeout=10`, `ServerAliveInterval=30`)
//! - Never prints or logs private key material

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;

/// Validate an SSH target of the form `user@host` or `host` (rejects shell metacharacters).
pub fn validate_ssh_target(target: &str) -> Result<(), String> {
    if target.is_empty() || target.len() > 253 {
        return Err("SSH target length invalid".to_string());
    }
    if target.starts_with('-') {
        return Err("SSH target must not start with '-'".to_string());
    }
    let valid = target
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || matches!(c, '@' | '.' | '-' | '_' | ':'));
    if !valid {
        return Err(format!(
            "SSH target '{target}' contains forbidden characters"
        ));
    }
    Ok(())
}

/// Default Ed25519 identity path for Inkling2-OS <-> Muse.ai SSH auth.
pub fn muse_key_path(state_dir: &Path) -> PathBuf {
    state_dir.join("ssh").join("inkling2_muse_ed25519")
}

/// Generate an Ed25519 keypair for Muse.ai VM authorization and output ONLY the public key.
pub fn cmd_muse_keygen(state_dir: &Path, out: &mut impl Write) {
    let key_path = muse_key_path(state_dir);
    if let Some(parent) = key_path.parent()
        && let Err(e) = fs::create_dir_all(parent)
    {
        let _ = writeln!(out, "muse keygen failed (mkdir): {e}");
        return;
    }
    if !key_path.exists() {
        let st = Command::new("ssh-keygen")
            .args([
                "-t",
                "ed25519",
                "-N",
                "",
                "-C",
                "inkling2-os@muse-bridge",
                "-f",
                &key_path.to_string_lossy(),
            ])
            .output();
        match st {
            Ok(o) if o.status.success() => {
                let _ = writeln!(out, "Generated Ed25519 keypair at {}", key_path.display());
            }
            Ok(o) => {
                let _ = writeln!(
                    out,
                    "ssh-keygen failed: {}",
                    String::from_utf8_lossy(&o.stderr).trim()
                );
                return;
            }
            Err(e) => {
                let _ = writeln!(out, "ssh-keygen unavailable: {e}");
                return;
            }
        }
    }
    let pub_path = key_path.with_extension("pub");
    match fs::read_to_string(&pub_path) {
        Ok(pk) => {
            let _ = writeln!(
                out,
                "Public key (add to Muse.ai VM ~/.ssh/authorized_keys):"
            );
            let _ = writeln!(out, "  {}", pk.trim());
        }
        Err(e) => {
            let _ = writeln!(out, "Failed to read public key {}: {e}", pub_path.display());
        }
    }
}

/// Build hardened OpenSSH base arguments (key-only, keepalive, timeout).
pub fn hardened_ssh_args(key_path: &Path, port: u16) -> Vec<String> {
    let mut args = vec![
        "-p".to_string(),
        port.to_string(),
        "-o".to_string(),
        "BatchMode=yes".to_string(),
        "-o".to_string(),
        "PasswordAuthentication=no".to_string(),
        "-o".to_string(),
        "PubkeyAuthentication=yes".to_string(),
        "-o".to_string(),
        "StrictHostKeyChecking=accept-new".to_string(),
        "-o".to_string(),
        "ConnectTimeout=10".to_string(),
        "-o".to_string(),
        "ServerAliveInterval=30".to_string(),
        "-o".to_string(),
        "ServerAliveCountMax=3".to_string(),
    ];
    if key_path.exists() {
        args.push("-i".to_string());
        args.push(key_path.to_string_lossy().into_owned());
    }
    args
}

/// Test key-authenticated SSH connectivity to a Muse.ai VM.
pub fn cmd_muse_ping(state_dir: &Path, target: &str, port: u16, out: &mut impl Write) {
    if let Err(e) = validate_ssh_target(target) {
        let _ = writeln!(out, "muse ping rejected: {e}");
        return;
    }
    let mut args = hardened_ssh_args(&muse_key_path(state_dir), port);
    args.push(target.to_string());
    args.push("echo INKLING2_MUSE_SSH_OK".to_string());

    match Command::new("ssh").args(&args).output() {
        Ok(o) if o.status.success() => {
            let _ = writeln!(
                out,
                "Muse.ai SSH OK ({target}:{port}): {}",
                String::from_utf8_lossy(&o.stdout).trim()
            );
        }
        Ok(o) => {
            let _ = writeln!(
                out,
                "Muse.ai SSH failed: {}",
                String::from_utf8_lossy(&o.stderr).trim()
            );
        }
        Err(e) => {
            let _ = writeln!(out, "ssh binary unavailable: {e}");
        }
    }
}

/// Establish a loopback-only reverse SSH tunnel so Muse.ai VM can reach local Inkling2-OS services.
pub fn cmd_muse_tunnel(
    state_dir: &Path,
    target: &str,
    remote_port: u16,
    local_port: u16,
    ssh_port: u16,
    out: &mut impl Write,
) {
    if let Err(e) = validate_ssh_target(target) {
        let _ = writeln!(out, "muse tunnel rejected: {e}");
        return;
    }
    let mut args = hardened_ssh_args(&muse_key_path(state_dir), ssh_port);
    args.push("-N".to_string());
    args.push("-o".to_string());
    args.push("ExitOnForwardFailure=yes".to_string());
    args.push("-R".to_string());
    args.push(format!("127.0.0.1:{remote_port}:127.0.0.1:{local_port}"));
    args.push(target.to_string());

    let _ = writeln!(
        out,
        "Opening reverse SSH tunnel: Muse.ai 127.0.0.1:{remote_port} -> Inkling2-OS 127.0.0.1:{local_port} via {target}:{ssh_port}..."
    );
    match Command::new("ssh").args(&args).status() {
        Ok(s) => {
            let _ = writeln!(out, "Reverse tunnel exited with {s}");
        }
        Err(e) => {
            let _ = writeln!(out, "Failed to start ssh reverse tunnel: {e}");
        }
    }
}

/// Prepare PC/Inkling2-OS inbox/outbox directories for VM-initiated Muse.ai sync over Tailscale SSH.
pub fn cmd_muse_inbox_ready(state_dir: &Path, out: &mut impl Write) {
    let inbox = state_dir.join("muse-inbox");
    if let Err(e) = fs::create_dir_all(&inbox) {
        let _ = writeln!(out, "Failed to create inbox {}: {e}", inbox.display());
        return;
    }
    let _ = writeln!(out, "VM-Initiated Muse.ai Inbox Ready: {}", inbox.display());
    let _ = writeln!(
        out,
        "  1) From Muse.ai VM -> PC pull  : scp <pc>:{}/memory45.ikm ~/inkling2-state/",
        state_dir.display()
    );
    let _ = writeln!(
        out,
        "  2) From Muse.ai VM -> PC push  : scp ~/inkling2-state/memory45.ikm <pc>:{}/",
        inbox.display()
    );
    let _ = writeln!(
        out,
        "  3) From Muse.ai VM -> PC tunnel: ssh -N -L 127.0.0.1:18080:127.0.0.1:8080 <pc>"
    );
    let _ = writeln!(
        out,
        "  4) On PC validate & activate   : inkling2-shell -c \"muse import\""
    );
}

/// Validate magic & 45 MiB budget on files pushed by Muse.ai VM into `state/muse-inbox/`, then atomically activate.
pub fn cmd_muse_import(state_dir: &Path, out: &mut impl Write) {
    let inbox = state_dir.join("muse-inbox");
    let max_mem_bytes: u64 = 45 * 1024 * 1024;

    let mem_in = inbox.join("memory45.ikm");
    if mem_in.exists() {
        match fs::read(&mem_in) {
            Ok(bytes) if bytes.len() as u64 <= max_mem_bytes && bytes.starts_with(b"IKMEM45\0") => {
                let dst = state_dir.join("memory45.ikm");
                if fs::rename(&mem_in, &dst).is_ok() {
                    let _ = writeln!(
                        out,
                        "Imported validated memory45.ikm ({} bytes <= 45 MiB)",
                        bytes.len()
                    );
                }
            }
            Ok(bytes) => {
                let _ = writeln!(
                    out,
                    "Rejected inbox memory45.ikm (invalid magic or >45 MiB: {} bytes)",
                    bytes.len()
                );
            }
            Err(e) => {
                let _ = writeln!(out, "Failed reading inbox memory45.ikm: {e}");
            }
        }
    } else {
        let _ = writeln!(out, "No pending memory45.ikm in {}", inbox.display());
    }

    let lora_in = inbox.join("dynamic_lora.iklora");
    if lora_in.exists() {
        match fs::read(&lora_in) {
            Ok(bytes) if bytes.starts_with(b"IKLORA1\0") => {
                let dst = state_dir.join("dynamic_lora.iklora");
                if fs::rename(&lora_in, &dst).is_ok() {
                    let _ = writeln!(
                        out,
                        "Imported validated dynamic_lora.iklora ({} bytes)",
                        bytes.len()
                    );
                }
            }
            Ok(bytes) => {
                let _ = writeln!(
                    out,
                    "Rejected inbox dynamic_lora.iklora (invalid IKLORA1 magic, {} bytes)",
                    bytes.len()
                );
            }
            Err(e) => {
                let _ = writeln!(out, "Failed reading inbox dynamic_lora.iklora: {e}");
            }
        }
    }
}

pub fn cmd_muse_sync(state_dir: &Path, target: &str, remote_dir: &str, out: &mut impl Write) {
    if let Err(e) = validate_ssh_target(target) {
        let _ = writeln!(out, "muse sync rejected: {e}");
        return;
    }
    if remote_dir.contains("..") || remote_dir.starts_with('-') {
        let _ = writeln!(out, "muse sync rejected: invalid remote_dir '{remote_dir}'");
        return;
    }
    let key = muse_key_path(state_dir);
    for fname in ["memory45.ikm", "dynamic_lora.iklora"] {
        let local = state_dir.join(fname);
        if !local.exists() {
            continue;
        }
        let mut args = vec![
            "-o".to_string(),
            "BatchMode=yes".to_string(),
            "-o".to_string(),
            "ConnectTimeout=10".to_string(),
        ];
        if key.exists() {
            args.push("-i".to_string());
            args.push(key.to_string_lossy().into_owned());
        }
        args.push(local.to_string_lossy().into_owned());
        args.push(format!("{target}:{remote_dir}/{fname}"));
        match Command::new("scp").args(&args).status() {
            Ok(s) if s.success() => {
                let _ = writeln!(out, "Synced {fname} -> {target}:{remote_dir}/{fname}");
            }
            Ok(s) => {
                let _ = writeln!(out, "scp {fname} exited with {s}");
            }
            Err(e) => {
                let _ = writeln!(out, "scp unavailable: {e}");
            }
        }
    }
}
