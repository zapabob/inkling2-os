//! Secure Internet & Hugging Face Hub Client for Inkling2-OS Shell.
//!
//! Enforces `SOP-Internet-Connectivity` & `SOP-Security`:
//! - HTTPS-only (`https://`) with TLS 1.2+ minimum (`--proto =https --tlsv1.2`)
//! - SSRF protection (blocks loopback, link-local metadata 169.254.169.254, and private RFC1918 hosts)
//! - Allowlisted Hugging Face Hub base URL (`https://huggingface.co`) for `hf-pull`
//! - Strict path-traversal validation on model filenames
//! - Explicit connect/read timeouts and bounded retries

use std::io::Write;
use std::path::Path;
use std::process::Command;

/// Validate an outbound HTTPS URL against SSRF and plaintext-transport rules.
pub fn validate_https_url(url: &str) -> Result<(), String> {
    let rest = url
        .strip_prefix("https://")
        .ok_or_else(|| "only https:// URLs are permitted (TLS required)".to_string())?;
    let host_port = rest.split('/').next().unwrap_or("");
    let host = host_port
        .split('@')
        .next_back()
        .unwrap_or("")
        .split(':')
        .next()
        .unwrap_or("")
        .to_ascii_lowercase();

    if host.is_empty() {
        return Err("URL host is empty".to_string());
    }
    // SSRF guard: reject loopback, link-local cloud metadata, and RFC1918 private nets
    let blocked = host == "localhost"
        || host.ends_with(".local")
        || host == "127.0.0.1"
        || host == "::1"
        || host == "[::1]"
        || host.starts_with("127.")
        || host.starts_with("10.")
        || host.starts_with("169.254.")
        || host.starts_with("192.168.")
        || host.starts_with("172.16.")
        || host.starts_with("172.17.")
        || host.starts_with("172.18.")
        || host.starts_with("172.19.")
        || host.starts_with("172.2")
        || host.starts_with("172.30.")
        || host.starts_with("172.31.");

    if blocked {
        return Err(format!("blocked private/loopback/metadata host '{host}' (SSRF guard)"));
    }
    Ok(())
}

/// Build an allowlisted Hugging Face Hub resolve URL for a `.strata` model file.
pub fn build_hf_resolve_url(repo: &str, filename: &str) -> Result<String, String> {
    if repo.is_empty()
        || repo.contains("..")
        || !repo.contains('/')
        || repo.chars().any(|c| !(c.is_ascii_alphanumeric() || matches!(c, '/' | '-' | '_' | '.')))
    {
        return Err(format!("invalid Hugging Face repo identifier '{repo}'"));
    }
    if filename.is_empty()
        || filename.contains("..")
        || filename.contains('/')
        || filename.contains('\\')
        || filename.chars().any(|c| !(c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.')))
    {
        return Err(format!("invalid model filename '{filename}'"));
    }
    Ok(format!("https://huggingface.co/{repo}/resolve/main/{filename}"))
}

/// Show network status (interfaces & HTTPS/TLS probe).
pub fn cmd_net_status(out: &mut impl Write) {
    let _ = writeln!(out, "Inkling2-OS Network Status (TLS 1.2+ / Default-Deny Inbound):");
    if cfg!(target_os = "linux") {
        if let Ok(o) = Command::new("/sbin/ip").args(["-br", "addr"]).output() {
            let _ = out.write_all(&o.stdout);
        }
    } else {
        let _ = writeln!(out, "  Host OS: {} (Linux NICs managed by inkling2-init on ISO boot)", std::env::consts::OS);
    }
    let probe = Command::new("curl")
        .args([
            "-sS", "-I", "--proto", "=https", "--tlsv1.2",
            "--connect-timeout", "5", "--max-time", "10",
            "https://huggingface.co",
        ])
        .output();
    match probe {
        Ok(o) if o.status.success() => {
            let first = String::from_utf8_lossy(&o.stdout)
                .lines()
                .next()
                .unwrap_or("HTTPS OK")
                .to_string();
            let _ = writeln!(out, "  TLS Probe (huggingface.co): REACHABLE ({first})");
        }
        Ok(o) => {
            let err = String::from_utf8_lossy(&o.stderr);
            let _ = writeln!(out, "  TLS Probe (huggingface.co): OFFLINE/BLOCKED ({})", err.trim());
        }
        Err(e) => {
            let _ = writeln!(out, "  TLS Probe: curl unavailable ({e})");
        }
    }
}

/// Perform a bounded HTTPS GET head/body fetch (enforces TLS 1.2+, SSRF guard, timeouts).
pub fn cmd_net_fetch(url: &str, out: &mut impl Write) {
    if let Err(e) = validate_https_url(url) {
        let _ = writeln!(out, "net fetch rejected: {e}");
        return;
    }
    let res = Command::new("curl")
        .args([
            "-sS", "-f", "-L",
            "--proto", "=https",
            "--proto-redir", "=https",
            "--tlsv1.2",
            "--connect-timeout", "10",
            "--max-time", "30",
            "--retry", "2",
            "--retry-delay", "2",
            url,
        ])
        .output();
    match res {
        Ok(o) if o.status.success() => {
            let preview_len = o.stdout.len().min(2048);
            let _ = writeln!(out, "Fetched {} bytes over TLS:", o.stdout.len());
            let _ = out.write_all(&o.stdout[..preview_len]);
            if o.stdout.len() > preview_len {
                let _ = writeln!(out, "\n... [truncated]");
            } else {
                let _ = writeln!(out);
            }
        }
        Ok(o) => {
            let _ = writeln!(out, "net fetch failed: {}", String::from_utf8_lossy(&o.stderr).trim());
        }
        Err(e) => {
            let _ = writeln!(out, "net fetch error: {e}");
        }
    }
}

/// Resumable HTTPS download of a `.strata` model from allowlisted `huggingface.co` into `models_dir`.
pub fn cmd_hf_pull(repo: &str, filename: &str, models_dir: &Path, out: &mut impl Write) {
    let url = match build_hf_resolve_url(repo, filename) {
        Ok(u) => u,
        Err(e) => {
            let _ = writeln!(out, "hf-pull rejected: {e}");
            return;
        }
    };
    let dest = models_dir.join(filename);
    let _ = writeln!(out, "Downloading {url} -> {} (resumable TLS 1.2+)...", dest.display());
    let status = Command::new("curl")
        .args([
            "-f", "-L", "-C", "-",
            "--proto", "=https",
            "--proto-redir", "=https",
            "--tlsv1.2",
            "--connect-timeout", "15",
            "--retry", "3",
            "--retry-delay", "3",
            "-o", &dest.to_string_lossy(),
            &url,
        ])
        .status();
    match status {
        Ok(s) if s.success() => {
            let _ = writeln!(out, "Download complete: {}", dest.display());
        }
        Ok(s) => {
            let _ = writeln!(out, "hf-pull failed with exit status {s}");
        }
        Err(e) => {
            let _ = writeln!(out, "hf-pull failed to invoke curl: {e}");
        }
    }
}
