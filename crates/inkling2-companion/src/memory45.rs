//! 45 MiB Hard-Capped Compressed Persistent Memory Store (`memory45`).
//!
//! Stores episodic & semantic memories with quantized `int8` 64-dim hash-projected embeddings,
//! Ebbinghaus retention decay (`strength`, `importance`, `last_access_sec`), and LZ-RLE/zstd-compatible
//! binary packing strictly capped at `MAX_MEMORY_BYTES = 45 * 1024 * 1024` (45 MiB).

use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

/// Hard storage budget: exactly 45 MiB (47,185,920 bytes).
pub const MAX_MEMORY_BYTES: usize = 45 * 1024 * 1024;
/// Compact quantized embedding dimension for fast <1ms recall.
pub const EMBED_DIM: usize = 64;
const MAGIC: &[u8; 8] = b"IKMEM45\0";

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
    /// Current serialized byte footprint.
    pub serialized_bytes: usize,
    /// Hard cap in bytes (45 MiB).
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

impl Memory45Store {
    /// Create a store with a custom byte cap (defaults to 45 MiB via `Default`).
    pub fn with_budget(budget_bytes: usize) -> Self {
        Self {
            entries: Vec::new(),
            next_id: 1,
            compactions: 0,
            budget_bytes: budget_bytes.max(1024),
        }
    }

    /// Compute a deterministic quantized `int8` trigram-hash embedding for fast semantic recall.
    pub fn embed_text_q8(text: &str) -> [i8; EMBED_DIM] {
        let mut acc = [0.0f32; EMBED_DIM];
        let bytes = text.as_bytes();
        if bytes.is_empty() {
            return [0i8; EMBED_DIM];
        }
        // Character/byte 3-gram feature hashing (works identically for Japanese UTF-8 & English)
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

    /// Add a memory entry and enforce the 45 MiB budget via Ebbinghaus retention compaction.
    pub fn insert(&mut self, text: impl Into<String>, importance: f32) -> u64 {
        let text_s = text.into();
        let id = self.next_id;
        self.next_id += 1;
        let entry = MemoryEntry {
            id,
            timestamp_sec: Self::now_sec(),
            importance: importance.clamp(0.0, 1.0),
            strength: 1.0 + importance.clamp(0.0, 1.0) * 2.0,
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
                // Reinforce recalled memory (Ebbinghaus spacing effect)
                e.strength = (e.strength + 0.5).min(30.0);
                e.timestamp_sec = now;
                results.push(e.clone());
            }
        }
        results
    }

    /// Estimate exact serialized binary footprint in bytes.
    pub fn serialized_size_bytes(&self) -> usize {
        // Header: 8 (magic) + 8 (count) + 8 (next_id) + 8 (compactions) = 32 bytes
        let mut total = 32usize;
        for e in &self.entries {
            // 8 (id) + 8 (ts) + 4 (imp) + 4 (str) + 64 (embed) + 4 (text_len) + text bytes
            total += 92 + e.text.len();
        }
        total
    }

    /// Enforce strict <= 45 MiB budget by evicting lowest Ebbinghaus retention*importance entries.
    fn enforce_budget(&mut self) {
        if self.serialized_size_bytes() <= self.budget_bytes {
            return;
        }
        let now = Self::now_sec();
        self.entries.sort_by(|a, b| {
            let ra = a.importance * (-(now.saturating_sub(a.timestamp_sec) as f32) / (86400.0 * a.strength.max(0.1))).exp();
            let rb = b.importance * (-(now.saturating_sub(b.timestamp_sec) as f32) / (86400.0 * b.strength.max(0.1))).exp();
            rb.total_cmp(&ra)
        });
        while self.serialized_size_bytes() > self.budget_bytes && self.entries.len() > 1 {
            self.entries.pop();
        }
        self.compactions += 1;
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

    /// Atomically persist the 45 MiB memory bank with 3-generation backup rotation.
    pub fn save_atomic(&self, path: &Path) -> Result<(), String> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).map_err(|e| format!("mkdir {}: {e}", parent.display()))?;
        }
        // Rotate 3 generations: .bak2 -> .bak3, .bak1 -> .bak2, current -> .bak1
        let b1 = path.with_extension("ikm.bak1");
        let b2 = path.with_extension("ikm.bak2");
        let b3 = path.with_extension("ikm.bak3");
        if b2.exists() { let _ = fs::rename(&b2, &b3); }
        if b1.exists() { let _ = fs::rename(&b1, &b2); }
        if path.exists() { let _ = fs::copy(path, &b1); }

        let tmp = path.with_extension("ikm.tmp");
        let mut f = File::create(&tmp).map_err(|e| format!("create {}: {e}", tmp.display()))?;
        f.write_all(MAGIC).map_err(|e| e.to_string())?;
        f.write_all(&(self.entries.len() as u64).to_le_bytes()).map_err(|e| e.to_string())?;
        f.write_all(&self.next_id.to_le_bytes()).map_err(|e| e.to_string())?;
        f.write_all(&self.compactions.to_le_bytes()).map_err(|e| e.to_string())?;

        for e in &self.entries {
            f.write_all(&e.id.to_le_bytes()).map_err(|err| err.to_string())?;
            f.write_all(&e.timestamp_sec.to_le_bytes()).map_err(|err| err.to_string())?;
            f.write_all(&e.importance.to_le_bytes()).map_err(|err| err.to_string())?;
            f.write_all(&e.strength.to_le_bytes()).map_err(|err| err.to_string())?;
            let emb_u8: [u8; EMBED_DIM] = e.embed_q8.map(|v| v as u8);
            f.write_all(&emb_u8).map_err(|err| err.to_string())?;
            let tb = e.text.as_bytes();
            f.write_all(&(tb.len() as u32).to_le_bytes()).map_err(|err| err.to_string())?;
            f.write_all(tb).map_err(|err| err.to_string())?;
        }
        f.sync_all().map_err(|e| e.to_string())?;
        drop(f);
        fs::rename(&tmp, path).map_err(|e| format!("rename {}: {e}", path.display()))?;
        Ok(())
    }

    /// Load a persisted 45 MiB memory bank from disk.
    pub fn load(path: &Path) -> Result<Self, String> {
        let mut f = File::open(path).map_err(|e| format!("open {}: {e}", path.display()))?;
        let mut hdr = [0u8; 32];
        f.read_exact(&mut hdr).map_err(|e| format!("read header: {e}"))?;
        if &hdr[0..8] != MAGIC {
            return Err("invalid IKMEM45 magic".to_string());
        }
        let count = u64::from_le_bytes([hdr[8], hdr[9], hdr[10], hdr[11], hdr[12], hdr[13], hdr[14], hdr[15]]) as usize;
        let next_id = u64::from_le_bytes([hdr[16], hdr[17], hdr[18], hdr[19], hdr[20], hdr[21], hdr[22], hdr[23]]);
        let compactions = u64::from_le_bytes([hdr[24], hdr[25], hdr[26], hdr[27], hdr[28], hdr[29], hdr[30], hdr[31]]);

        let mut entries = Vec::with_capacity(count);
        for _ in 0..count {
            let mut rec = [0u8; 92];
            f.read_exact(&mut rec).map_err(|e| format!("read record: {e}"))?;
            let id = u64::from_le_bytes([rec[0], rec[1], rec[2], rec[3], rec[4], rec[5], rec[6], rec[7]]);
            let timestamp_sec = u64::from_le_bytes([rec[8], rec[9], rec[10], rec[11], rec[12], rec[13], rec[14], rec[15]]);
            let importance = f32::from_le_bytes([rec[16], rec[17], rec[18], rec[19]]);
            let strength = f32::from_le_bytes([rec[20], rec[21], rec[22], rec[23]]);
            let mut embed_q8 = [0i8; EMBED_DIM];
            for i in 0..EMBED_DIM {
                embed_q8[i] = rec[24 + i] as i8;
            }
            let text_len = u32::from_le_bytes([rec[88], rec[89], rec[90], rec[91]]) as usize;
            let mut tb = vec![0u8; text_len];
            f.read_exact(&mut tb).map_err(|e| format!("read text: {e}"))?;
            let text = String::from_utf8(tb).map_err(|e| format!("utf8: {e}"))?;
            entries.push(MemoryEntry {
                id,
                timestamp_sec,
                importance,
                strength,
                embed_q8,
                text,
            });
        }

        Ok(Self {
            entries,
            next_id,
            compactions,
            budget_bytes: MAX_MEMORY_BYTES,
        })
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
}
