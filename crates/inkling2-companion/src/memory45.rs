//! 45 MiB Hard-Capped Compressed Persistent Memory Store (`memory45`).
//!
//! Stores episodic & semantic memories with quantized `int8` 64-dim hash-projected embeddings,
//! Ebbinghaus retention decay (`strength`, `importance`, `last_access_sec`), strict single-record
//! truncation, on-disk total-budget accounting (active + rollback backup <= 45 MiB), and
//! FNV-1a 64-bit integrity trailer verification.

use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

/// Hard storage budget: exactly 45 MiB (47,185,920 bytes).
pub const MAX_MEMORY_BYTES: usize = 45 * 1024 * 1024;
/// Compact quantized embedding dimension for fast <1ms recall.
pub const EMBED_DIM: usize = 64;
const MAGIC: &[u8; 8] = b"IKMEM45\0";
const CHECKSUM_MAGIC: &[u8; 8] = b"IKCHK64\0";
const HEADER_BYTES: usize = 32;
const RECORD_FIXED_BYTES: usize = 92;
const TRAILER_BYTES: usize = 16;

/// Single episodic/semantic memory record.
#[derive(Debug, Clone)]
pub struct MemoryEntry {
    /// Monotonic record ID.
    pub id: u64,
    /// Creation/last-recall UNIX timestamp in seconds.
    pub timestamp_sec: u64,
    /// Importance score in `[0.0, 1.0]`.
    pub importance: f32,
    /// Ebbinghaus memory strength (grows on recall/reinforcement).
    pub strength: f32,
    /// Quantized int8 semantic signature vector (`EMBED_DIM`).
    pub embed_q8: [i8; EMBED_DIM],
    /// UTF-8 memory content (compact summary or dialogue turn).
    pub text: String,
}

/// Telemetry for the 45 MiB compressed memory store.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MemoryStats {
    /// Number of stored memories.
    pub entry_count: usize,
    /// Current serialized byte footprint (including header and integrity trailer).
    pub serialized_bytes: usize,
    /// Hard cap in bytes (45 MiB by default).
    pub budget_bytes: usize,
    /// Number of compaction/eviction passes executed.
    pub compactions: u64,
}

/// 45 MiB hard-budgeted memory bank with Ebbinghaus retention & semantic recall.
#[derive(Debug, Clone)]
pub struct Memory45Store {
    entries: Vec<MemoryEntry>,
    next_id: u64,
    compactions: u64,
    budget_bytes: usize,
}

impl Default for Memory45Store {
    fn default() -> Self {
        Self::with_budget(MAX_MEMORY_BYTES)
    }
}

fn fnv1a_64(bytes: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf29ce484222325;
    for &b in bytes {
        hash ^= u64::from(b);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

fn truncate_utf8_bytes(s: &str, max_bytes: usize) -> String {
    if s.len() <= max_bytes {
        return s.to_owned();
    }
    let mut end = max_bytes;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    s[..end].to_owned()
}

fn backup_path(path: &Path, suffix: &str) -> PathBuf {
    let mut os = path.as_os_str().to_os_string();
    os.push(suffix);
    PathBuf::from(os)
}

impl Memory45Store {
    /// Create a store with a custom byte cap (defaults to 45 MiB via `Default`).
    pub fn with_budget(budget_bytes: usize) -> Self {
        let min_budget = HEADER_BYTES + RECORD_FIXED_BYTES + TRAILER_BYTES + 64;
        Self {
            entries: Vec::new(),
            next_id: 1,
            compactions: 0,
            budget_bytes: budget_bytes.max(min_budget),
        }
    }

    /// Maximum UTF-8 byte length permitted for a single record under this store's budget.
    pub fn max_single_record_text_bytes(&self) -> usize {
        self.budget_bytes
            .saturating_sub(HEADER_BYTES + RECORD_FIXED_BYTES + TRAILER_BYTES)
    }

    /// Compute a deterministic quantized `int8` trigram-hash embedding for fast semantic recall.
    pub fn embed_text_q8(text: &str) -> [i8; EMBED_DIM] {
        let mut acc = [0.0f32; EMBED_DIM];
        let bytes = text.as_bytes();
        if bytes.is_empty() {
            return [0i8; EMBED_DIM];
        }
        for win in bytes.windows(3) {
            let mut h: u64 = 0xcbf29ce484222325;
            for &b in win {
                h ^= u64::from(b);
                h = h.wrapping_mul(0x100000001b3);
            }
            let idx = (h as usize) % EMBED_DIM;
            let sign = if (h >> 63) == 0 { 1.0f32 } else { -1.0f32 };
            acc[idx] += sign;
        }
        let max_abs = acc.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
        let scale = if max_abs > 1e-6 { 127.0 / max_abs } else { 1.0 };
        let mut out = [0i8; EMBED_DIM];
        for (o, &v) in out.iter_mut().zip(acc.iter()) {
            *o = (v * scale).round().clamp(-127.0, 127.0) as i8;
        }
        out
    }

    fn cosine_q8(a: &[i8; EMBED_DIM], b: &[i8; EMBED_DIM]) -> f32 {
        let mut dot = 0i32;
        let mut na = 0i32;
        let mut nb = 0i32;
        for i in 0..EMBED_DIM {
            let av = i32::from(a[i]);
            let bv = i32::from(b[i]);
            dot += av * bv;
            na += av * av;
            nb += bv * bv;
        }
        if na == 0 || nb == 0 {
            return 0.0;
        }
        (dot as f32) / ((na as f32).sqrt() * (nb as f32).sqrt())
    }

    fn now_sec() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0)
    }

    /// Add a memory entry and strictly enforce the hard byte budget.
    ///
    /// Oversized single records are truncated on a valid UTF-8 boundary so a single
    /// insertion can never violate `budget_bytes`.
    pub fn insert(&mut self, text: impl Into<String>, importance: f32) -> u64 {
        let raw_text = text.into();
        let max_text = self.max_single_record_text_bytes();
        let text_s = truncate_utf8_bytes(&raw_text, max_text);
        let imp = if importance.is_finite() {
            importance.clamp(0.0, 1.0)
        } else {
            0.0
        };
        let id = self.next_id;
        self.next_id = self.next_id.saturating_add(1);
        let entry = MemoryEntry {
            id,
            timestamp_sec: Self::now_sec(),
            importance: imp,
            strength: 1.0 + imp * 2.0,
            embed_q8: Self::embed_text_q8(&text_s),
            text: text_s,
        };
        self.entries.push(entry);
        self.enforce_budget();
        id
    }

    /// Recall top-k memories by hybrid semantic cosine similarity + Ebbinghaus retention score.
    pub fn recall(&mut self, query: &str, top_k: usize) -> Vec<MemoryEntry> {
        let q_emb = Self::embed_text_q8(query);
        let now = Self::now_sec();
        let mut scored: Vec<(usize, f32)> = self
            .entries
            .iter()
            .enumerate()
            .map(|(idx, e)| {
                let sim = Self::cosine_q8(&q_emb, &e.embed_q8);
                let dt_hours = (now.saturating_sub(e.timestamp_sec) as f32) / 3600.0;
                let retention = (-dt_hours / (24.0 * e.strength.max(0.1))).exp();
                let score = 0.65 * sim + 0.20 * e.importance + 0.15 * retention;
                (idx, score)
            })
            .collect();

        scored.sort_by(|a, b| b.1.total_cmp(&a.1));
        let mut results = Vec::new();
        for (idx, _) in scored.into_iter().take(top_k) {
            if let Some(e) = self.entries.get_mut(idx) {
                e.strength = (e.strength + 0.5).min(30.0);
                e.timestamp_sec = now;
                results.push(e.clone());
            }
        }
        results
    }

    /// Remove a memory entry by ID. Returns `true` if an entry was removed.
    pub fn delete(&mut self, id: u64) -> bool {
        let before = self.entries.len();
        self.entries.retain(|e| e.id != id);
        self.entries.len() < before
    }

    /// Exact serialized binary footprint in bytes (header + records + integrity trailer).
    pub fn serialized_size_bytes(&self) -> usize {
        let mut total = HEADER_BYTES + TRAILER_BYTES;
        for e in &self.entries {
            total = total.saturating_add(RECORD_FIXED_BYTES + e.text.len());
        }
        total
    }

    /// Enforce strict `<= budget_bytes` by evicting lowest Ebbinghaus retention*importance entries.
    fn enforce_budget(&mut self) {
        if self.serialized_size_bytes() <= self.budget_bytes {
            return;
        }
        let now = Self::now_sec();
        self.entries.sort_by(|a, b| {
            let ra = a.importance
                * (-(now.saturating_sub(a.timestamp_sec) as f32) / (86400.0 * a.strength.max(0.1)))
                    .exp();
            let rb = b.importance
                * (-(now.saturating_sub(b.timestamp_sec) as f32) / (86400.0 * b.strength.max(0.1)))
                    .exp();
            rb.total_cmp(&ra)
        });
        while self.serialized_size_bytes() > self.budget_bytes && !self.entries.is_empty() {
            self.entries.pop();
        }
        self.compactions = self.compactions.saturating_add(1);
    }

    /// Current memory telemetry.
    pub fn stats(&self) -> MemoryStats {
        MemoryStats {
            entry_count: self.entries.len(),
            serialized_bytes: self.serialized_size_bytes(),
            budget_bytes: self.budget_bytes,
            compactions: self.compactions,
        }
    }

    fn serialize_bytes(&self) -> Result<Vec<u8>, String> {
        let total = self.serialized_size_bytes();
        if total > self.budget_bytes {
            return Err(format!(
                "memory payload ({total} bytes) exceeds hard budget ({} bytes)",
                self.budget_bytes
            ));
        }
        let mut buf = Vec::with_capacity(total);
        buf.extend_from_slice(MAGIC);
        buf.extend_from_slice(&(self.entries.len() as u64).to_le_bytes());
        buf.extend_from_slice(&self.next_id.to_le_bytes());
        buf.extend_from_slice(&self.compactions.to_le_bytes());

        for e in &self.entries {
            buf.extend_from_slice(&e.id.to_le_bytes());
            buf.extend_from_slice(&e.timestamp_sec.to_le_bytes());
            buf.extend_from_slice(&e.importance.to_le_bytes());
            buf.extend_from_slice(&e.strength.to_le_bytes());
            let emb_u8: [u8; EMBED_DIM] = e.embed_q8.map(|v| v as u8);
            buf.extend_from_slice(&emb_u8);
            let tb = e.text.as_bytes();
            let tb_len = u32::try_from(tb.len())
                .map_err(|_| "record text length exceeds u32::MAX".to_string())?;
            buf.extend_from_slice(&tb_len.to_le_bytes());
            buf.extend_from_slice(tb);
        }

        let checksum = fnv1a_64(&buf);
        buf.extend_from_slice(CHECKSUM_MAGIC);
        buf.extend_from_slice(&checksum.to_le_bytes());
        Ok(buf)
    }

    /// Measure combined on-disk bytes for `path` and its rollback backup(s).
    pub fn total_disk_bytes(path: &Path) -> u64 {
        let mut total = fs::metadata(path).map(|m| m.len()).unwrap_or(0);
        for suffix in [".bak1", ".bak2", ".bak3"] {
            let p = backup_path(path, suffix);
            total = total.saturating_add(fs::metadata(&p).map(|m| m.len()).unwrap_or(0));
        }
        total
    }

    /// Atomically persist the memory bank with integrity checksum and bounded rollback backup.
    ///
    /// Legacy multi-generation backups (`.bak2`, `.bak3`) are pruned, and `.bak1` is only
    /// retained when `active_bytes + backup_bytes <= budget_bytes` so total on-disk personal
    /// memory never silently exceeds the hard budget.
    pub fn save_atomic(&self, path: &Path) -> Result<(), String> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).map_err(|e| format!("mkdir {}: {e}", parent.display()))?;
        }
        let payload = self.serialize_bytes()?;

        // Prune legacy unbounded multi-generation backups so disk footprint stays bounded.
        for legacy in [".bak2", ".bak3"] {
            let p = backup_path(path, legacy);
            if p.exists() {
                let _ = fs::remove_file(&p);
            }
        }
        // Also prune old `.with_extension("ikm.bak2")` / `.bak3` names if present.
        for ext in ["ikm.bak2", "ikm.bak3"] {
            let p = path.with_extension(ext);
            if p != *path && p.exists() {
                let _ = fs::remove_file(&p);
            }
        }

        let b1 = backup_path(path, ".bak1");
        if path.exists() {
            let existing_len = fs::metadata(path).map(|m| m.len() as usize).unwrap_or(0);
            if payload.len().saturating_add(existing_len) <= self.budget_bytes {
                let _ = fs::copy(path, &b1);
            } else if b1.exists() {
                let _ = fs::remove_file(&b1);
            }
        }

        let tmp = backup_path(path, ".tmp");
        let write_result = (|| -> Result<(), String> {
            let mut f = File::create(&tmp).map_err(|e| format!("create {}: {e}", tmp.display()))?;
            f.write_all(&payload).map_err(|e| e.to_string())?;
            f.sync_all().map_err(|e| e.to_string())?;
            Ok(())
        })();
        if let Err(err) = write_result {
            let _ = fs::remove_file(&tmp);
            return Err(err);
        }

        fs::rename(&tmp, path).map_err(|e| {
            let _ = fs::remove_file(&tmp);
            format!("rename {}: {e}", path.display())
        })?;
        Ok(())
    }

    /// Load a persisted memory bank from disk using the default 45 MiB hard cap.
    pub fn load(path: &Path) -> Result<Self, String> {
        Self::load_with_budget(path, MAX_MEMORY_BYTES)
    }

    /// Load a persisted memory bank with explicit budget enforcement and corruption rejection.
    pub fn load_with_budget(path: &Path, budget_bytes: usize) -> Result<Self, String> {
        let min_budget = HEADER_BYTES + RECORD_FIXED_BYTES + TRAILER_BYTES + 64;
        let effective_budget = budget_bytes.max(min_budget);
        let meta = fs::metadata(path).map_err(|e| format!("stat {}: {e}", path.display()))?;
        if meta.len() > effective_budget as u64 {
            return Err(format!(
                "memory file {} ({} bytes) exceeds hard budget ({effective_budget} bytes)",
                path.display(),
                meta.len()
            ));
        }

        let mut f = File::open(path).map_err(|e| format!("open {}: {e}", path.display()))?;
        let mut raw = Vec::with_capacity(meta.len() as usize);
        f.read_to_end(&mut raw)
            .map_err(|e| format!("read {}: {e}", path.display()))?;

        if raw.len() < HEADER_BYTES {
            return Err("truncated IKMEM45 header".to_string());
        }
        if &raw[0..8] != MAGIC {
            return Err("invalid IKMEM45 magic".to_string());
        }

        // Verify optional trailing FNV-1a 64-bit integrity trailer if present.
        let body_end = if raw.len() >= HEADER_BYTES + TRAILER_BYTES
            && &raw[raw.len() - TRAILER_BYTES..raw.len() - 8] == CHECKSUM_MAGIC
        {
            let payload_end = raw.len() - TRAILER_BYTES;
            let expected = u64::from_le_bytes([
                raw[raw.len() - 8],
                raw[raw.len() - 7],
                raw[raw.len() - 6],
                raw[raw.len() - 5],
                raw[raw.len() - 4],
                raw[raw.len() - 3],
                raw[raw.len() - 2],
                raw[raw.len() - 1],
            ]);
            let actual = fnv1a_64(&raw[..payload_end]);
            if actual != expected {
                return Err(format!(
                    "IKMEM45 checksum mismatch (expected {expected:#018x}, got {actual:#018x})"
                ));
            }
            payload_end
        } else {
            raw.len()
        };

        let count_u64 = u64::from_le_bytes([
            raw[8], raw[9], raw[10], raw[11], raw[12], raw[13], raw[14], raw[15],
        ]);
        let max_possible_entries = (body_end - HEADER_BYTES) / RECORD_FIXED_BYTES;
        if count_u64 > max_possible_entries as u64 {
            return Err(format!(
                "corrupted entry count {count_u64} exceeds maximum possible {max_possible_entries}"
            ));
        }
        let count = count_u64 as usize;

        let next_id = u64::from_le_bytes([
            raw[16], raw[17], raw[18], raw[19], raw[20], raw[21], raw[22], raw[23],
        ]);
        let compactions = u64::from_le_bytes([
            raw[24], raw[25], raw[26], raw[27], raw[28], raw[29], raw[30], raw[31],
        ]);

        let mut cursor = HEADER_BYTES;
        let mut entries = Vec::with_capacity(count);
        for _ in 0..count {
            if cursor + RECORD_FIXED_BYTES > body_end {
                return Err("truncated memory record".to_string());
            }
            let rec = &raw[cursor..cursor + RECORD_FIXED_BYTES];
            cursor += RECORD_FIXED_BYTES;

            let id = u64::from_le_bytes([
                rec[0], rec[1], rec[2], rec[3], rec[4], rec[5], rec[6], rec[7],
            ]);
            let timestamp_sec = u64::from_le_bytes([
                rec[8], rec[9], rec[10], rec[11], rec[12], rec[13], rec[14], rec[15],
            ]);
            let importance = f32::from_le_bytes([rec[16], rec[17], rec[18], rec[19]]);
            let strength = f32::from_le_bytes([rec[20], rec[21], rec[22], rec[23]]);
            if !importance.is_finite() || !strength.is_finite() || strength < 0.0 {
                return Err("corrupted non-finite importance or strength".to_string());
            }

            let mut embed_q8 = [0i8; EMBED_DIM];
            for (dst, &src) in embed_q8.iter_mut().zip(rec[24..24 + EMBED_DIM].iter()) {
                *dst = src as i8;
            }
            let text_len = u32::from_le_bytes([rec[88], rec[89], rec[90], rec[91]]) as usize;
            if text_len > body_end.saturating_sub(cursor) {
                return Err(format!(
                    "corrupted text_len {text_len} exceeds remaining payload bytes"
                ));
            }
            let text = std::str::from_utf8(&raw[cursor..cursor + text_len])
                .map_err(|e| format!("utf8: {e}"))?
                .to_owned();
            cursor += text_len;

            entries.push(MemoryEntry {
                id,
                timestamp_sec,
                importance: importance.clamp(0.0, 1.0),
                strength: strength.clamp(0.1, 30.0),
                embed_q8,
                text,
            });
        }

        if cursor != body_end {
            return Err(format!(
                "trailing unparsed bytes in memory file ({} bytes)",
                body_end - cursor
            ));
        }

        let mut store = Self {
            entries,
            next_id,
            compactions,
            budget_bytes: effective_budget,
        };
        store.enforce_budget();
        Ok(store)
    }

    /// Load from `path`, automatically falling back to `.bak1` if the primary file is corrupted.
    pub fn load_with_recovery(path: &Path, budget_bytes: usize) -> Result<Self, String> {
        match Self::load_with_budget(path, budget_bytes) {
            Ok(store) => Ok(store),
            Err(primary_err) => {
                let bak = backup_path(path, ".bak1");
                if bak.exists() {
                    Self::load_with_budget(&bak, budget_bytes).map_err(|bak_err| {
                        format!("primary failed ({primary_err}); backup failed ({bak_err})")
                    })
                } else {
                    Err(primary_err)
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn memory45_enforces_hard_budget_and_recalls_japanese() {
        let mut store = Memory45Store::with_budget(2048);
        store.insert("ユーザーはRTX 5060 TiでInkling-2-Miniを動かしている", 0.95);
        for i in 0..40 {
            store.insert(format!("一時的なログ項目 #{i} テストデータ文字列"), 0.1);
        }
        let st = store.stats();
        assert!(st.serialized_bytes <= 2048);
        assert!(st.compactions > 0);
        let hits = store.recall("RTX 5060 Ti Inkling", 1);
        assert!(!hits.is_empty());
        assert!(hits[0].text.contains("RTX 5060 Ti"));
    }

    #[test]
    fn single_oversized_record_cannot_exceed_budget() {
        let mut store = Memory45Store::with_budget(1024);
        let huge = "あ".repeat(2000);
        store.insert(huge, 1.0);
        let st = store.stats();
        assert!(
            st.serialized_bytes <= 1024,
            "single oversized record exceeded budget: {}",
            st.serialized_bytes
        );
        assert_eq!(st.entry_count, 1);
    }

    #[test]
    fn save_atomic_bounds_total_disk_bytes_and_recovers_from_corruption() {
        let dir =
            std::env::temp_dir().join(format!("inkling2_mem45_test_{}", Memory45Store::now_sec()));
        let _ = fs::remove_dir_all(&dir);
        let path = dir.join("memory45.ikm");

        let mut store = Memory45Store::with_budget(2048);
        store.insert("first verified fact", 0.9);
        assert!(store.save_atomic(&path).is_ok());

        store.insert("second verified fact", 0.95);
        assert!(store.save_atomic(&path).is_ok());
        assert!(Memory45Store::total_disk_bytes(&path) <= 2048);

        // Corrupt one byte in the primary file and verify checksum rejection + backup recovery.
        if let Ok(mut bytes) = fs::read(&path) {
            if let Some(last) = bytes.last_mut() {
                *last ^= 0xFF;
            }
            let _ = fs::write(&path, &bytes);
        }
        assert!(Memory45Store::load_with_budget(&path, 2048).is_err());
        let recovered = Memory45Store::load_with_recovery(&path, 2048);
        assert!(recovered.is_ok());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn load_rejects_corrupted_count_without_oom() {
        let dir = std::env::temp_dir().join(format!(
            "inkling2_mem45_corrupt_{}",
            Memory45Store::now_sec()
        ));
        let _ = fs::create_dir_all(&dir);
        let path = dir.join("bad.ikm");
        let mut bad = Vec::new();
        bad.extend_from_slice(MAGIC);
        bad.extend_from_slice(&u64::MAX.to_le_bytes()); // malicious count
        bad.extend_from_slice(&1u64.to_le_bytes());
        bad.extend_from_slice(&0u64.to_le_bytes());
        let _ = fs::write(&path, &bad);

        let err = Memory45Store::load_with_budget(&path, 4096);
        assert!(err.is_err());
        let _ = fs::remove_dir_all(&dir);
    }
}
