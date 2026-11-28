//! Native `.strata` v1.1 header scanner for Inkling2-OS Shell.
//! Supports both the classic 4 KiB layout (section table at 0x10) and the SQv2
//! 16 KiB-aligned layout (section table at 0x20, e.g. `Inkling-2-Mini-SQv2-20-CHiPS.strata`).
//! Scans `/models` (or any configured model root) without external dependencies.

use std::fs::{self, File};
use std::io::Read;
use std::path::{Path, PathBuf};

const STRATA_NUM_SECTIONS: usize = 6;
const STRATA_SECTION_DESC_BYTES: usize = 64;
const STRATA_TABLE_OFFSET_4K: usize = 0x10;
const STRATA_TABLE_OFFSET_16K: usize = 0x20;
const STRATA_PAGE_4K: u64 = 4096;
const STRATA_PAGE_16K: u64 = 16384;
const STRATA_HEADER_NEED: usize =
    STRATA_TABLE_OFFSET_16K + STRATA_NUM_SECTIONS * STRATA_SECTION_DESC_BYTES + 32;

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
    /// Page alignment size (4096 for classic v1.1, 16384 for SQv2 16 KiB layout).
    pub page_size: u32,
    /// Offset of the zstd-compressed JSON manifest.
    pub manifest_offset: u64,
    /// Compressed manifest byte length.
    pub manifest_bytes: u64,
}

#[derive(Debug, Clone, Copy, Default)]
struct SectionEntry {
    offset: u64,
    size: u64,
}

struct HeaderProbe {
    sections: [SectionEntry; STRATA_NUM_SECTIONS],
    aligned: bool,
    ids: bool,
    ranges: bool,
    any_section: bool,
}

impl HeaderProbe {
    fn strict(&self) -> bool {
        self.aligned && self.ids && self.ranges && self.any_section
    }
}

fn le_u16(buf: &[u8], off: usize) -> u16 {
    u16::from_le_bytes([buf[off], buf[off + 1]])
}

fn le_u32(buf: &[u8], off: usize) -> u32 {
    u32::from_le_bytes([buf[off], buf[off + 1], buf[off + 2], buf[off + 3]])
}

fn le_u64(buf: &[u8], off: usize) -> u64 {
    u64::from_le_bytes([
        buf[off],
        buf[off + 1],
        buf[off + 2],
        buf[off + 3],
        buf[off + 4],
        buf[off + 5],
        buf[off + 6],
        buf[off + 7],
    ])
}

fn probe_section_table(hdr: &[u8], table_off: usize, page: u64, file_len: u64) -> HeaderProbe {
    let mut probe = HeaderProbe {
        sections: [SectionEntry::default(); STRATA_NUM_SECTIONS],
        aligned: true,
        ids: true,
        ranges: true,
        any_section: false,
    };

    for (i, slot) in probe.sections.iter_mut().enumerate() {
        let base = table_off + i * STRATA_SECTION_DESC_BYTES;
        let offset = le_u64(hdr, base);
        let size = le_u64(hdr, base + 8);
        let id = le_u32(hdr, base + 48);
        *slot = SectionEntry { offset, size };

        if id as usize != i {
            probe.ids = false;
        }
        if offset > file_len {
            probe.ranges = false;
        }
        if size == 0 {
            continue;
        }
        probe.any_section = true;
        if !offset.is_multiple_of(page) {
            probe.aligned = false;
        }
        if offset < STRATA_PAGE_4K || offset.checked_add(size).is_none_or(|end| end > file_len) {
            probe.ranges = false;
        }
    }

    if probe.ranges {
        let mut active: Vec<SectionEntry> = probe
            .sections
            .iter()
            .copied()
            .filter(|s| s.size > 0)
            .collect();
        active.sort_by_key(|s| s.offset);
        for pair in active.windows(2) {
            if pair[1].offset < pair[0].offset.saturating_add(pair[0].size) {
                probe.ranges = false;
                break;
            }
        }
    }

    probe
}

/// Parse a `.strata` header buffer of at least `STRATA_HEADER_NEED` bytes.
pub fn parse_strata_header_bytes(
    hdr: &[u8],
    file_len: u64,
    path: &Path,
) -> Result<StrataModelInfo, String> {
    if hdr.len() < STRATA_HEADER_NEED {
        return Err("short STRATA header frame".into());
    }
    if &hdr[0..8] != b"STRATA\x00\x01" {
        return Err(format!("invalid STRATA magic in {}", path.display()));
    }

    let format_version = le_u16(hdr, 8);
    if format_version != 1 {
        return Err("unsupported STRATA format version".into());
    }
    let section_count = le_u16(hdr, 14);
    if section_count as usize != STRATA_NUM_SECTIONS {
        return Err(format!("unexpected STRATA section count: {section_count}"));
    }

    let p4k = probe_section_table(hdr, STRATA_TABLE_OFFSET_4K, STRATA_PAGE_4K, file_len);
    let p16k = probe_section_table(hdr, STRATA_TABLE_OFFSET_16K, STRATA_PAGE_16K, file_len);
    if p4k.strict() && p16k.strict() {
        return Err("ambiguous STRATA header: matches both 4 KiB and 16 KiB layouts".into());
    }

    let (sections, page_size) = if p16k.strict() {
        if le_u16(hdr, 0x10) != 0 {
            return Err("unsupported SQv2 tile layout".into());
        }
        (p16k.sections, STRATA_PAGE_16K as u32)
    } else if p4k.strict() {
        (p4k.sections, STRATA_PAGE_4K as u32)
    } else {
        return Err("model manifest is outside the accepted header bounds".into());
    };

    let manifest = sections[0];
    let trunk = sections[1];
    if trunk.size == 0
        || manifest.offset < STRATA_PAGE_4K
        || !(1..=64 * 1024 * 1024).contains(&manifest.size)
    {
        return Err("model manifest is outside the accepted header bounds".into());
    }

    Ok(StrataModelInfo {
        path: path.to_path_buf(),
        file_bytes: file_len,
        trunk_bytes: trunk.size,
        format_version,
        section_count,
        page_size,
        manifest_offset: manifest.offset,
        manifest_bytes: manifest.size,
    })
}

/// Inspect a single `.strata` v1.1 binary header (4 KiB or 16 KiB SQv2 layout).
pub fn inspect_strata_header(path: &Path) -> Result<StrataModelInfo, String> {
    let meta = fs::symlink_metadata(path).map_err(|e| format!("stat {}: {e}", path.display()))?;
    if !meta.is_file() || meta.len() < 4096 || is_reparse_point(&meta) {
        return Err("model must be a regular local file with a complete header".into());
    }
    let mut f = File::open(path).map_err(|e| format!("open {}: {e}", path.display()))?;
    let mut hdr = [0u8; STRATA_HEADER_NEED];
    f.read_exact(&mut hdr)
        .map_err(|e| format!("read header {}: {e}", path.display()))?;
    parse_strata_header_bytes(&hdr, meta.len(), path)
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

#[cfg(test)]
mod tests {
    use super::*;

    fn write_sec(hdr: &mut [u8], table_off: usize, idx: u32, offset: u64, size: u64, flags: u32) {
        let base = table_off + idx as usize * STRATA_SECTION_DESC_BYTES;
        hdr[base..base + 8].copy_from_slice(&offset.to_le_bytes());
        hdr[base + 8..base + 16].copy_from_slice(&size.to_le_bytes());
        hdr[base + 48..base + 52].copy_from_slice(&idx.to_le_bytes());
        hdr[base + 52..base + 56].copy_from_slice(&flags.to_le_bytes());
    }

    #[test]
    fn parses_classic_4k_and_sqv2_16k_headers() -> Result<(), String> {
        let mut h4k = [0u8; STRATA_HEADER_NEED];
        h4k[0..8].copy_from_slice(b"STRATA\x00\x01");
        h4k[8..10].copy_from_slice(&1u16.to_le_bytes());
        h4k[14..16].copy_from_slice(&6u16.to_le_bytes());
        write_sec(&mut h4k, STRATA_TABLE_OFFSET_4K, 0, 4096, 8192, 0);
        write_sec(&mut h4k, STRATA_TABLE_OFFSET_4K, 1, 16384, 65536, 0);
        write_sec(&mut h4k, STRATA_TABLE_OFFSET_4K, 2, 0, 0, 0);
        write_sec(&mut h4k, STRATA_TABLE_OFFSET_4K, 3, 81920, 104448, 1);
        write_sec(&mut h4k, STRATA_TABLE_OFFSET_4K, 4, 188416, 4096, 0);
        write_sec(&mut h4k, STRATA_TABLE_OFFSET_4K, 5, 0, 0, 0);

        let info4 = parse_strata_header_bytes(&h4k, 192_512, Path::new("SQ-20.strata"))?;
        assert_eq!(info4.page_size, 4096);
        assert_eq!(info4.manifest_offset, 4096);
        assert_eq!(info4.manifest_bytes, 8192);
        assert_eq!(info4.trunk_bytes, 65536);

        // Exact geometry of Kujira/Inkling-2-Mini-Strata-CHiPS-Abliterated
        let mut h16k = [0u8; STRATA_HEADER_NEED];
        h16k[0..8].copy_from_slice(b"STRATA\x00\x01");
        h16k[8..10].copy_from_slice(&1u16.to_le_bytes());
        h16k[12..14].copy_from_slice(&1u16.to_le_bytes());
        h16k[14..16].copy_from_slice(&6u16.to_le_bytes());
        write_sec(
            &mut h16k,
            STRATA_TABLE_OFFSET_16K,
            0,
            19_374_751_744,
            10_699,
            0,
        );
        write_sec(
            &mut h16k,
            STRATA_TABLE_OFFSET_16K,
            1,
            16_384,
            3_671_130_112,
            0,
        );
        write_sec(&mut h16k, STRATA_TABLE_OFFSET_16K, 2, 0, 0, 0);
        write_sec(
            &mut h16k,
            STRATA_TABLE_OFFSET_16K,
            3,
            19_374_620_672,
            114_688,
            1,
        );
        write_sec(
            &mut h16k,
            STRATA_TABLE_OFFSET_16K,
            4,
            3_671_146_496,
            15_703_474_176,
            0,
        );
        write_sec(
            &mut h16k,
            STRATA_TABLE_OFFSET_16K,
            5,
            19_374_735_360,
            383,
            0,
        );

        let info16 = parse_strata_header_bytes(
            &h16k,
            19_374_768_128,
            Path::new("Inkling-2-Mini-SQv2-20-CHiPS.strata"),
        )?;
        assert_eq!(info16.page_size, 16384);
        assert_eq!(info16.manifest_offset, 19_374_751_744);
        assert_eq!(info16.manifest_bytes, 10_699);
        assert_eq!(info16.trunk_bytes, 3_671_130_112);
        h16k[0x10] = 1;
        assert!(parse_strata_header_bytes(&h16k, 19_374_768_128, Path::new("bad.strata")).is_err());
        h16k[0x10] = 0;

        // Overlapping 16K sections must be rejected
        write_sec(&mut h16k, STRATA_TABLE_OFFSET_16K, 1, 16_384, 32_768, 0);
        write_sec(&mut h16k, STRATA_TABLE_OFFSET_16K, 4, 32_768, 16_384, 0);
        assert!(parse_strata_header_bytes(&h16k, 19_374_768_128, Path::new("bad.strata")).is_err());
        Ok(())
    }
}
