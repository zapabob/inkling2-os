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

use crate::public_browser::fetch_public_https;

/// Build an allowlisted Hugging Face Hub resolve URL for a `.strata` model file.
pub fn build_hf_resolve_url(repo: &str, filename: &str) -> Result<String, String> {
    if repo.is_empty()
        || repo.contains("..")
        || !repo.contains('/')
        || repo
            .chars()
            .any(|c| !(c.is_ascii_alphanumeric() || matches!(c, '/' | '-' | '_' | '.')))
    {
        return Err(format!("invalid Hugging Face repo identifier '{repo}'"));
    }
    if filename.is_empty()
        || filename.contains("..")
        || filename.contains('/')
        || filename.contains('\\')
        || filename
            .chars()
            .any(|c| !(c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.')))
    {
        return Err(format!("invalid model filename '{filename}'"));
    }
    Ok(format!(
        "https://huggingface.co/{repo}/resolve/main/{filename}"
    ))
}

/// Show network status (interfaces & HTTPS/TLS probe).
pub fn cmd_net_status(out: &mut impl Write) {
    let _ = writeln!(
        out,
        "Inkling2-OS Network Status (TLS 1.2+ / Default-Deny Inbound):"
    );
    if cfg!(target_os = "linux") {
        if let Ok(o) = Command::new("/sbin/ip").args(["-br", "addr"]).output() {
            let _ = out.write_all(&o.stdout);
        }
    } else {
        let _ = writeln!(
            out,
            "  Host OS: {} (Linux NICs managed by inkling2-init on ISO boot)",
            std::env::consts::OS
        );
    }
    let probe = Command::new("curl")
        .args([
            "-sS",
            "-I",
            "--proto",
            "=https",
            "--tlsv1.2",
            "--connect-timeout",
            "5",
            "--max-time",
            "10",
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
            let _ = writeln!(
                out,
                "  TLS Probe (huggingface.co): OFFLINE/BLOCKED ({})",
                err.trim()
            );
        }
        Err(e) => {
            let _ = writeln!(out, "  TLS Probe: curl unavailable ({e})");
        }
    }
}

/// Perform a bounded HTTPS GET head/body fetch (enforces TLS 1.2+, SSRF guard, timeouts).
pub fn cmd_net_fetch(url: &str, out: &mut impl Write) {
    match fetch_public_https(url) {
        Ok(bytes) => {
            let preview_len = bytes.len().min(2048);
            let _ = writeln!(out, "Fetched {} bytes over TLS:", bytes.len());
            let preview = String::from_utf8_lossy(&bytes[..preview_len]);
            // Avoid feeding untrusted terminal control characters to the console.
            let text: String = preview
                .chars()
                .filter(|c| !c.is_control() || matches!(c, '\n' | '\t'))
                .collect();
            let _ = writeln!(out, "{text}");
            if bytes.len() > preview_len {
                let _ = writeln!(out, "[truncated]");
            }
        }
        Err((_, error)) => {
            let _ = writeln!(out, "net fetch failed: {error}");
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
    let _ = writeln!(
        out,
        "Downloading {url} -> {} (resumable TLS 1.2+)...",
        dest.display()
    );
    let status = Command::new("curl")
        .args([
            "-f",
            "-L",
            "-C",
            "-",
            "--proto",
            "=https",
            "--proto-redir",
            "=https",
            "--tlsv1.2",
            "--connect-timeout",
            "15",
            "--retry",
            "3",
            "--retry-delay",
            "3",
            "-o",
            &dest.to_string_lossy(),
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
