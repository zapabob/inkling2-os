//! Native `.strata` v1.1 header scanner for Inkling2-OS Shell.
//! Scans `/models` (or any configured model root) without external dependencies.

use std::fs::{self, File};
use std::io::Read;
use std::path::{Path, PathBuf};

/// Parsed `.strata` v1.1 file header metadata.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StrataModelInfo {
    /// Absolute path to the `.strata` file.
    pub path: PathBuf,
    /// Total file size in bytes.
    pub file_bytes: u64,
    /// Binary format version field.
    pub format_version: u16,
    /// Section table count.
    pub section_count: u16,
    /// Offset of the zstd-compressed JSON manifest.
    pub manifest_offset: u64,
    /// Compressed manifest byte length.
    pub manifest_bytes: u64,
}

/// Inspect a single `.strata` v1.1 binary header (first 64 bytes).
pub fn inspect_strata_header(path: &Path) -> Result<StrataModelInfo, String> {
    let meta = fs::metadata(path).map_err(|e| format!("stat {}: {e}", path.display()))?;
    let mut f = File::open(path).map_err(|e| format!("open {}: {e}", path.display()))?;
    let mut hdr = [0u8; 64];
    f.read_exact(&mut hdr)
        .map_err(|e| format!("read header {}: {e}", path.display()))?;

    if &hdr[0..6] != b"STRATA" {
        return Err(format!("invalid STRATA magic in {}", path.display()));
    }

    let format_version = u16::from_le_bytes([hdr[8], hdr[9]]);
    let section_count = u16::from_le_bytes([hdr[14], hdr[15]]);
    let manifest_offset = u64::from_le_bytes([
        hdr[16], hdr[17], hdr[18], hdr[19], hdr[20], hdr[21], hdr[22], hdr[23],
    ]);
    let manifest_bytes = u64::from_le_bytes([
        hdr[24], hdr[25], hdr[26], hdr[27], hdr[28], hdr[29], hdr[30], hdr[31],
    ]);

    Ok(StrataModelInfo {
        path: path.to_path_buf(),
        file_bytes: meta.len(),
        format_version,
        section_count,
        manifest_offset,
        manifest_bytes,
    })
}

/// Discover all valid `.strata` v1.1 files inside `root` (depth <= 2).
pub fn discover_models(root: &Path) -> Vec<StrataModelInfo> {
    let mut found = Vec::new();
    let Ok(entries) = fs::read_dir(root) else {
        return found;
    };
    for entry in entries.flatten() {
        let p = entry.path();
        if p.is_file() && p.extension().and_then(|e| e.to_str()) == Some("strata") {
            if let Ok(info) = inspect_strata_header(&p) {
                found.push(info);
            }
        } else if p.is_dir() {
            if let Ok(sub) = fs::read_dir(&p) {
                for se in sub.flatten() {
                    let sp = se.path();
                    if sp.is_file() && sp.extension().and_then(|e| e.to_str()) == Some("strata") {
                        if let Ok(info) = inspect_strata_header(&sp) {
                            found.push(info);
                        }
                    }
                }
            }
        }
    }
    found
}
