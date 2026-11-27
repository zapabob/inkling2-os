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
    pub trunk_bytes: u64,
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
    let meta = fs::symlink_metadata(path).map_err(|e| format!("stat {}: {e}", path.display()))?;
    if !meta.is_file() || meta.len() < 4096 || is_reparse_point(&meta) {
        return Err("model must be a regular local file with a complete header".into());
    }
    let mut f = File::open(path).map_err(|e| format!("open {}: {e}", path.display()))?;
    let mut hdr = [0u8; 144];
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
    let le64 = |start| {
        hdr.get(start..start + 8)
            .and_then(|bytes| <[u8; 8]>::try_from(bytes).ok())
            .map(u64::from_le_bytes)
            .ok_or("invalid section descriptor")
    };
    let trunk_offset = le64(80)?;
    let trunk_bytes = le64(88)?;
    if hdr[128..132] != 1_u32.to_le_bytes()
        || trunk_bytes == 0
        || trunk_offset < 4096
        || trunk_offset % 4096 != 0
        || trunk_offset
            .checked_add(trunk_bytes)
            .is_none_or(|end| end > meta.len())
        || !(4..=32).contains(&section_count)
        || manifest_offset < 4096
        || !(1..=64 * 1024 * 1024).contains(&manifest_bytes)
        || manifest_offset
            .checked_add(manifest_bytes)
            .is_none_or(|end| end > meta.len())
    {
        return Err("model manifest is outside the accepted header bounds".into());
    }

    Ok(StrataModelInfo {
        path: path.to_path_buf(),
        file_bytes: meta.len(),
        trunk_bytes,
        format_version,
        section_count,
        manifest_offset,
        manifest_bytes,
    })
}

fn is_reparse_point(meta: &fs::Metadata) -> bool {
    #[cfg(windows)]
    {
        use std::os::windows::fs::MetadataExt;
        const FILE_ATTRIBUTE_REPARSE_POINT: u32 = 0x400;
        meta.file_attributes() & FILE_ATTRIBUTE_REPARSE_POINT != 0
    }
    #[cfg(not(windows))]
    meta.file_type().is_symlink()
}

/// Discover all valid `.strata` v1.1 files inside `root` (depth <= 2).
pub fn discover_models(root: &Path) -> Vec<StrataModelInfo> {
    let mut found = scan_root(root);
    if let Some(extra) = std::env::var_os("INKLING2_ADDITIONAL_MODEL_DIRS") {
        for path in std::env::split_paths(&extra).filter(|path| path.is_absolute() && path != root)
        {
            found.extend(scan_root(&path));
        }
    }
    found.sort_by(|a, b| a.path.cmp(&b.path));
    found.dedup_by(|a, b| a.path == b.path);
    found
}

fn scan_root(root: &Path) -> Vec<StrataModelInfo> {
    let mut found = Vec::new();
    let Ok(entries) = fs::read_dir(root) else {
        return found;
    };
    for entry in entries.take(1024).flatten() {
        let p = entry.path();
        let Ok(meta) = fs::symlink_metadata(&p) else {
            continue;
        };
        if is_reparse_point(&meta) {
            continue;
        }
        if meta.is_file() && p.extension().and_then(|e| e.to_str()) == Some("strata") {
            if let Ok(info) = inspect_strata_header(&p) {
                found.push(info);
            }
        } else if meta.is_dir()
            && let Ok(sub) = fs::read_dir(&p)
        {
            for entry in sub.take(1024).flatten() {
                let path = entry.path();
                if path.extension().and_then(|ext| ext.to_str()) == Some("strata")
                    && let Ok(info) = inspect_strata_header(&path)
                {
                    found.push(info);
                }
            }
        }
    }
    found.sort_by(|a, b| a.path.cmp(&b.path));
    found
}
