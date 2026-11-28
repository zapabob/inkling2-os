//! Neural Memory45 — Bounded 45 MiB Hybrid Associative Memory & `.ikmem2` Container (Neon Phase 4).
//!
//! Enforces the exact `MAX_PERSISTENT_MEMORY_BYTES = 47,185,920` (45 MiB) hard budget across:
//! - Exact episodic & semantic memory (12 MiB)
//! - Product-Quantised embeddings & index (6 MiB; 512-dim -> 32x16 subspaces, 256 centroids, 32 B/code)
//! - Neural fast-weight associative memory dual bank (5 MiB x 2 = 10 MiB; 32 shards x 256 x 512 INT8 + FP16 scale)
//! - Dynamic LoRA dual bank reservation (4 MiB x 2 = 8 MiB)
//! - Update journal & replay buffer (5 MiB)
//! - Metadata & dual-superblock integrity (2 MiB)
//! - Emergency reserve (2 MiB)

use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

/// Hard persistent storage budget: exactly 45 MiB (`47,185,920` bytes).
pub const MAX_PERSISTENT_MEMORY_BYTES: usize = 45 * 1024 * 1024;
pub const REGION_EXACT_BYTES: usize = 12 * 1024 * 1024;
pub const REGION_PQ_BYTES: usize = 6 * 1024 * 1024;
pub const REGION_FAST_WEIGHTS_DUAL_BYTES: usize = 10 * 1024 * 1024;
pub const REGION_DYNAMIC_LORA_DUAL_BYTES: usize = 8 * 1024 * 1024;
pub const REGION_JOURNAL_REPLAY_BYTES: usize = 5 * 1024 * 1024;
pub const REGION_METADATA_BYTES: usize = 2 * 1024 * 1024;
pub const REGION_EMERGENCY_BYTES: usize = 2 * 1024 * 1024;

const _: () = assert!(
    REGION_EXACT_BYTES
        + REGION_PQ_BYTES
        + REGION_FAST_WEIGHTS_DUAL_BYTES
        + REGION_DYNAMIC_LORA_DUAL_BYTES
        + REGION_JOURNAL_REPLAY_BYTES
        + REGION_METADATA_BYTES
        + REGION_EMERGENCY_BYTES
        == MAX_PERSISTENT_MEMORY_BYTES
);

pub const VALUE_DIM: usize = 512;
pub const KEY_DIM: usize = 256;
pub const NUM_SHARDS: usize = 32;
pub const PQ_SUBSPACES: usize = 32;
pub const PQ_SUBDIM: usize = 16;
pub const PQ_CENTROIDS: usize = 256;
pub const PQ_CODEBOOK_FP16_BYTES: usize = PQ_SUBSPACES * PQ_CENTROIDS * PQ_SUBDIM * 2;
pub const FAST_WEIGHT_PARAMS: usize = NUM_SHARDS * KEY_DIM * VALUE_DIM;
pub const FAST_WEIGHT_BANK_MAX_BYTES: usize = 5 * 1024 * 1024;

const IKMEM2_MAGIC: &[u8; 8] = b"IKMEM2\0\0";
const SUPERBLOCK_BYTES: usize = 64;
const MAX_RECORDS: usize = 4096;
const MAX_TEXT_BYTES: usize = 4096;
const MAX_PROVENANCE_BYTES: usize = 256;
const RECORD_ACCOUNTING_BYTES: usize = 128;
static SNAPSHOT_SEQUENCE: AtomicU64 = AtomicU64::new(0);

fn fnv1a_64(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &b in bytes {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

fn now_sec() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Convert `f32` to IEEE 754 binary16 (`u16`) for compact row-wise quantisation scales.
pub fn f32_to_f16_bits(val: f32) -> u16 {
    let bits = val.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32 - 127 + 15;
    let mant = bits & 0x007f_ffff;
    if (bits & 0x7fff_ffff) > 0x7f80_0000 {
        return sign | 0x7e00;
    }
    if exp >= 31 {
        return sign | 0x7c00;
    }
    if exp < -10 {
        return sign;
    }
    let (significand, shift) = if exp <= 0 {
        (mant | 0x0080_0000, (14 - exp) as u32)
    } else {
        (mant, 13)
    };
    let truncated = significand >> shift;
    let remainder = significand & ((1u32 << shift) - 1);
    let halfway = 1u32 << (shift - 1);
    let rounded =
        truncated + u32::from(remainder > halfway || (remainder == halfway && truncated & 1 != 0));
    let exponent = if exp > 0 { (exp as u16) << 10 } else { 0 };
    sign | (exponent + rounded as u16)
}

/// Convert IEEE 754 binary16 (`u16`) back to `f32`.
pub fn f16_bits_to_f32(bits: u16) -> f32 {
    let sign = u32::from(bits & 0x8000) << 16;
    let exp = u32::from((bits >> 10) & 0x1f);
    let mant = u32::from(bits & 0x03ff);
    if exp == 0 {
        let value = (mant as f32) * (1.0 / 16_777_216.0);
        if sign == 0 { value } else { -value }
    } else if exp == 31 {
        f32::from_bits(sign | 0x7f80_0000 | (mant << 13))
    } else {
        let f_exp = (exp + 127 - 15) << 23;
        let f_mant = mant << 13;
        f32::from_bits(sign | f_exp | f_mant)
    }
}

/// Deterministic fixed encoder producing an L2-normalised 512-dim semantic value vector.
pub fn encode_value_512(text: &str) -> [f32; VALUE_DIM] {
    let mut vec = [0.0f32; VALUE_DIM];
    let bytes = text.as_bytes();
    if bytes.is_empty() {
        return vec;
    }
    for win_size in [2usize, 3usize, 4usize] {
        for win in bytes.windows(win_size) {
            let h = fnv1a_64(win);
            let idx = (h as usize) % VALUE_DIM;
            let sign = if (h >> 63) == 0 { 1.0f32 } else { -1.0f32 };
            vec[idx] += sign;
        }
    }
    let norm = vec.iter().map(|v| v * v).sum::<f32>().sqrt();
    if norm > 1e-8 {
        for v in &mut vec {
            *v /= norm;
        }
    }
    vec
}

/// Deterministic fixed key encoder producing an L2-normalised 256-dim key vector.
pub fn encode_key_256(text: &str) -> [f32; KEY_DIM] {
    let mut vec = [0.0f32; KEY_DIM];
    let bytes = text.as_bytes();
    if bytes.is_empty() {
        return vec;
    }
    for win in bytes.windows(3) {
        let h = fnv1a_64(win).wrapping_mul(0x9e3779b97f4a7c15);
        let idx = (h as usize) % KEY_DIM;
        let sign = if (h >> 63) == 0 { 1.0f32 } else { -1.0f32 };
        vec[idx] += sign;
    }
    let norm = vec.iter().map(|v| v * v).sum::<f32>().sqrt();
    if norm > 1e-8 {
        for v in &mut vec {
            *v /= norm;
        }
    }
    vec
}

/// Deterministic Product Quantisation codebook (`32 subspaces x 256 centroids x 16 dims`).
#[derive(Debug, Clone)]
pub struct PqCodebook {
    pub version: u32,
    pub seed: u64,
    /// Flattened centroids: `[PQ_SUBSPACES][PQ_CENTROIDS][PQ_SUBDIM]`.
    centroids: Vec<f32>,
}

impl PqCodebook {
    /// Generate a reproducible versioned codebook from a fixed seed.
    pub fn from_seed(version: u32, seed: u64) -> Self {
        let total = PQ_SUBSPACES * PQ_CENTROIDS * PQ_SUBDIM;
        let mut centroids = vec![0.0f32; total];
        let mut state = seed ^ 0x517c_c1b7_2722_0a95;
        for c in centroids.as_chunks_mut::<PQ_SUBDIM>().0 {
            let mut norm_sq = 0.0f32;
            for v in c.iter_mut() {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let u = ((state >> 33) as f32) / (u32::MAX as f32) * 2.0 - 1.0;
                *v = u;
                norm_sq += u * u;
            }
            let inv = if norm_sq > 1e-8 {
                1.0 / (norm_sq.sqrt() * (PQ_SUBSPACES as f32).sqrt())
            } else {
                0.0
            };
            for v in c.iter_mut() {
                *v *= inv;
            }
        }
        Self {
            version,
            seed,
            centroids,
        }
    }

    /// Encode a 512-dim vector into a 32-byte PQ code.
    pub fn encode(&self, vec: &[f32; VALUE_DIM]) -> [u8; PQ_SUBSPACES] {
        let mut code = [0u8; PQ_SUBSPACES];
        for (sub, out_code) in code.iter_mut().enumerate() {
            let sub_vec = &vec[sub * PQ_SUBDIM..(sub + 1) * PQ_SUBDIM];
            let mut best_idx = 0u8;
            let mut best_dist = f32::INFINITY;
            let base = sub * PQ_CENTROIDS * PQ_SUBDIM;
            for cid in 0..PQ_CENTROIDS {
                let c = &self.centroids[base + cid * PQ_SUBDIM..base + (cid + 1) * PQ_SUBDIM];
                let mut dist = 0.0f32;
                for i in 0..PQ_SUBDIM {
                    let d = sub_vec[i] - c[i];
                    dist += d * d;
                }
                if dist < best_dist {
                    best_dist = dist;
                    best_idx = cid as u8;
                }
            }
            *out_code = best_idx;
        }
        code
    }

    /// Asymmetric inner-product similarity between a full query vector and a 32-byte PQ code.
    pub fn asymmetric_dot(&self, query: &[f32; VALUE_DIM], code: &[u8; PQ_SUBSPACES]) -> f32 {
        let mut acc = 0.0f32;
        for (sub, &cid) in code.iter().enumerate() {
            let q = &query[sub * PQ_SUBDIM..(sub + 1) * PQ_SUBDIM];
            let base = (sub * PQ_CENTROIDS + usize::from(cid)) * PQ_SUBDIM;
            let c = &self.centroids[base..base + PQ_SUBDIM];
            for i in 0..PQ_SUBDIM {
                acc += q[i] * c[i];
            }
        }
        acc
    }

    /// Theoretical FP16 codebook footprint in bytes (262,144 bytes).
    pub fn fp16_footprint_bytes(&self) -> usize {
        PQ_CODEBOOK_FP16_BYTES
    }
}

/// One row-wise INT8-quantised associative memory shard (`VALUE_DIM x KEY_DIM`).
#[derive(Debug, Clone)]
pub struct AssociativeShard {
    /// Quantised INT8 weights (`VALUE_DIM * KEY_DIM`).
    weights_q8: Vec<i8>,
    /// Row-wise FP16 scales (`VALUE_DIM`).
    row_scales_f16: Vec<u16>,
    /// Deterministic routing signature (`KEY_DIM`).
    router_key: [f32; KEY_DIM],
    pub updates: u64,
}

impl AssociativeShard {
    fn new(shard_index: usize) -> Self {
        let mut router_key = [0.0f32; KEY_DIM];
        let mut state = (shard_index as u64)
            .wrapping_add(1)
            .wrapping_mul(0x9e3779b97f4a7c15);
        let mut norm_sq = 0.0f32;
        for v in &mut router_key {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let u = ((state >> 33) as f32) / (u32::MAX as f32) * 2.0 - 1.0;
            *v = u;
            norm_sq += u * u;
        }
        let inv = if norm_sq > 1e-8 {
            1.0 / norm_sq.sqrt()
        } else {
            0.0
        };
        for v in &mut router_key {
            *v *= inv;
        }
        Self {
            weights_q8: vec![0i8; VALUE_DIM * KEY_DIM],
            row_scales_f16: vec![f32_to_f16_bits(1.0 / 127.0); VALUE_DIM],
            router_key,
            updates: 0,
        }
    }

    fn router_score(&self, key: &[f32; KEY_DIM]) -> f32 {
        self.router_key
            .iter()
            .zip(key.iter())
            .map(|(a, b)| a * b)
            .sum()
    }

    /// Read `predicted_value = M @ key`.
    pub fn read(&self, key: &[f32; KEY_DIM]) -> [f32; VALUE_DIM] {
        let mut out = [0.0f32; VALUE_DIM];
        for (row_idx, out_val) in out.iter_mut().enumerate() {
            let scale = f16_bits_to_f32(self.row_scales_f16[row_idx]);
            let row = &self.weights_q8[row_idx * KEY_DIM..(row_idx + 1) * KEY_DIM];
            let mut acc = 0.0f32;
            for (&w, &k) in row.iter().zip(key.iter()) {
                acc += f32::from(w) * k;
            }
            *out_val = acc * scale;
        }
        out
    }

    /// Apply gated associative update:
    /// `M_next = (1 - decay) * M + clip(eta * gate * outer(error, key) / (eps + ||key||^2))`.
    /// Returns the pre-update surprise `||value - M @ key||_2^2`.
    pub fn gated_update(
        &mut self,
        key: &[f32; KEY_DIM],
        value: &[f32; VALUE_DIM],
        gate: f32,
        eta: f32,
        decay: f32,
    ) -> f32 {
        if !gate.is_finite()
            || !(0.0..=1.0).contains(&gate)
            || !eta.is_finite()
            || !(0.0..=1.0).contains(&eta)
            || !decay.is_finite()
            || !(0.0..=0.25).contains(&decay)
            || key
                .iter()
                .chain(value.iter())
                .any(|v| !v.is_finite() || v.abs() > 1.0)
        {
            return f32::INFINITY;
        }
        let predicted = self.read(key);
        let mut error = [0.0f32; VALUE_DIM];
        let mut surprise = 0.0f32;
        for i in 0..VALUE_DIM {
            let e = value[i] - predicted[i];
            error[i] = e;
            surprise += e * e;
        }
        if gate <= 1e-6 || eta <= 0.0 {
            return surprise;
        }

        let key_sq_norm: f32 = key.iter().map(|k| k * k).sum();
        let denom = 1e-5 + key_sq_norm;
        let step_coeff = (eta * gate) / denom;
        let retain = (1.0 - decay.clamp(0.0, 0.25)).clamp(0.75, 1.0);
        let max_delta = 0.25f32;

        let mut row_fp32 = [0.0f32; KEY_DIM];
        for (row_idx, &err_r) in error.iter().enumerate() {
            let old_scale = f16_bits_to_f32(self.row_scales_f16[row_idx]);
            let row_q = &mut self.weights_q8[row_idx * KEY_DIM..(row_idx + 1) * KEY_DIM];
            let mut max_abs = 0.0f32;
            for col in 0..KEY_DIM {
                let w_old = f32::from(row_q[col]) * old_scale;
                let delta = (step_coeff * err_r * key[col]).clamp(-max_delta, max_delta);
                let w_new = w_old * retain + delta;
                row_fp32[col] = w_new;
                max_abs = max_abs.max(w_new.abs());
            }
            let new_scale = if max_abs > 1e-7 {
                max_abs / 127.0
            } else {
                1.0 / 127.0
            };
            self.row_scales_f16[row_idx] = f32_to_f16_bits(new_scale.max(1.0 / 16_777_216.0));
            let inv_scale = 1.0 / f16_bits_to_f32(self.row_scales_f16[row_idx]);
            for col in 0..KEY_DIM {
                row_q[col] = (row_fp32[col] * inv_scale).round().clamp(-127.0, 127.0) as i8;
            }
        }
        self.updates = self.updates.saturating_add(1);
        surprise
    }

    pub fn persisted_bytes(&self) -> usize {
        self.weights_q8.len() + self.row_scales_f16.len() * 2
    }
}

/// 32-shard INT8 fast-weight associative memory bank with Top-2 sparse routing.
#[derive(Debug, Clone)]
pub struct FastWeightBank {
    pub generation: u64,
    shards: Vec<AssociativeShard>,
}

impl Default for FastWeightBank {
    fn default() -> Self {
        Self::new(0)
    }
}

impl FastWeightBank {
    pub fn new(generation: u64) -> Self {
        let shards = (0..NUM_SHARDS).map(AssociativeShard::new).collect();
        Self { generation, shards }
    }

    /// Select Top-2 shard indices for sparse routing.
    pub fn route_top2(&self, key: &[f32; KEY_DIM]) -> [usize; 2] {
        let mut best_i = 0usize;
        let mut best_s = f32::NEG_INFINITY;
        let mut second_i = 1usize;
        let mut second_s = f32::NEG_INFINITY;
        for (i, shard) in self.shards.iter().enumerate() {
            let s = shard.router_score(key);
            if s > best_s {
                second_i = best_i;
                second_s = best_s;
                best_i = i;
                best_s = s;
            } else if s > second_s {
                second_i = i;
                second_s = s;
            }
        }
        [best_i, second_i]
    }

    /// Read blended associative prediction from the Top-2 routed shards.
    pub fn read(&self, key: &[f32; KEY_DIM]) -> [f32; VALUE_DIM] {
        let [s0, s1] = self.route_top2(key);
        let v0 = self.shards[s0].read(key);
        let v1 = self.shards[s1].read(key);
        let mut out = [0.0f32; VALUE_DIM];
        for i in 0..VALUE_DIM {
            out[i] = 0.5 * (v0[i] + v1[i]);
        }
        out
    }

    /// Update only the Top-2 routed shards and return average surprise.
    pub fn update_top2(
        &mut self,
        key: &[f32; KEY_DIM],
        value: &[f32; VALUE_DIM],
        gate: f32,
        eta: f32,
        decay: f32,
    ) -> f32 {
        let [s0, s1] = self.route_top2(key);
        let sur0 = self.shards[s0].gated_update(key, value, gate, eta, decay);
        let sur1 = self.shards[s1].gated_update(key, value, gate, eta, decay);
        0.5 * (sur0 + sur1)
    }

    /// Total persisted weight + scale byte footprint for this single bank (must stay <= 5 MiB).
    pub fn persisted_bytes(&self) -> usize {
        self.shards
            .iter()
            .map(AssociativeShard::persisted_bytes)
            .sum()
    }

    pub fn total_updates(&self) -> u64 {
        self.shards.iter().map(|s| s.updates).sum()
    }
}

/// Provenance and verification status of an explicit episodic record.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecordVerification {
    VerifiedFact,
    OperatorInstruction,
    UnverifiedObservation,
}

impl RecordVerification {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::VerifiedFact => "verified_fact",
            Self::OperatorInstruction => "operator_instruction",
            Self::UnverifiedObservation => "unverified_observation",
        }
    }
}

/// Explicit L1 episodic record paired with L2 PQ code and learning eligibility flags.
#[derive(Debug, Clone)]
pub struct ExplicitRecord {
    pub id: u64,
    pub timestamp_sec: u64,
    pub text: String,
    pub importance: f32,
    pub provenance: String,
    pub verification: RecordVerification,
    /// Only operator-approved, trusted, deletable/reconstructible records may fold into L3 weights.
    pub foldable: bool,
    pub pq_code: [u8; PQ_SUBSPACES],
}

/// Grounded recall hit returned to LLM/Hermes (never presents raw NN output as bare fact).
#[derive(Debug, Clone)]
pub struct ReliableRecallHit {
    pub source_record_id: u64,
    pub text: String,
    pub confidence: f32,
    pub timestamp_sec: u64,
    pub provenance: String,
    pub verification_status: &'static str,
    pub associative_surprise: f32,
}

/// Region byte usage telemetry verifying strict <= 45 MiB enforcement.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NeuralMemoryStats {
    pub exact_bytes: usize,
    pub pq_index_bytes: usize,
    pub fast_weights_dual_bytes: usize,
    pub metadata_bytes: usize,
    pub total_budget_bytes: usize,
    pub record_count: usize,
    pub active_generation: u64,
    pub total_fast_weight_updates: u64,
}

/// Full Hybrid Neural Memory45 engine (L1 Exact + L2 PQ + L3 Dual-Bank Fast Weights).
#[derive(Debug, Clone)]
pub struct NeuralMemory45 {
    codebook: PqCodebook,
    records: Vec<ExplicitRecord>,
    active_bank: FastWeightBank,
    candidate_bank: FastWeightBank,
    next_id: u64,
    generation: u64,
}

impl Default for NeuralMemory45 {
    fn default() -> Self {
        Self::new(1, 0x2026_1128_0045_u64)
    }
}

impl NeuralMemory45 {
    pub fn new(codebook_version: u32, seed: u64) -> Self {
        Self {
            codebook: PqCodebook::from_seed(codebook_version, seed),
            records: Vec::new(),
            active_bank: FastWeightBank::new(0),
            candidate_bank: FastWeightBank::new(0),
            next_id: 1,
            generation: 0,
        }
    }

    /// Insert an explicit record; fold into candidate fast weights ONLY when verified, trusted, and permitted.
    #[allow(clippy::too_many_arguments)]
    pub fn ingest(
        &mut self,
        text: impl Into<String>,
        importance: f32,
        provenance: impl Into<String>,
        verification: RecordVerification,
        trusted: bool,
        operator_permitted: bool,
        sensitive_non_deletable: bool,
    ) -> Result<(u64, f32), String> {
        let mut text_s = text.into();
        let mut provenance = provenance.into();
        if text_s.trim().is_empty()
            || text_s.len() > MAX_TEXT_BYTES
            || provenance.len() > MAX_PROVENANCE_BYTES
            || !importance.is_finite()
            || !(0.0..=1.0).contains(&importance)
        {
            return Err("invalid neural memory text, provenance or importance".to_string());
        }
        let added_bytes = RECORD_ACCOUNTING_BYTES + text_s.len() + provenance.len();
        if self.records.len() >= MAX_RECORDS
            || self.stats().exact_bytes.saturating_add(added_bytes) > REGION_EXACT_BYTES
            || self.next_id == u64::MAX
        {
            return Err("neural memory admission budget exhausted".to_string());
        }
        text_s.shrink_to_fit();
        provenance.shrink_to_fit();
        let imp = importance;
        let value = encode_value_512(&text_s);
        let key = encode_key_256(&text_s);
        let pq_code = self.codebook.encode(&value);

        // Phase 4.4 / 4.7 gate: exclude untrusted, unverified, denied, or non-deletable sensitive data.
        let foldable = trusted
            && operator_permitted
            && !sensitive_non_deletable
            && verification != RecordVerification::UnverifiedObservation;

        let id = self.next_id;
        self.next_id += 1;
        self.records.push(ExplicitRecord {
            id,
            timestamp_sec: now_sec(),
            text: text_s,
            importance: imp,
            provenance,
            verification,
            foldable,
            pq_code,
        });

        let mut surprise = 0.0f32;
        if foldable {
            let gate = imp.max(0.2);
            self.candidate_bank = self.active_bank.clone();
            surprise = self
                .candidate_bank
                .update_top2(&key, &value, gate, 0.35, 0.01);
            // Evaluate pre vs post error on this key before promoting candidate bank.
            let after_pred = self.candidate_bank.read(&key);
            let after_err: f32 = value
                .iter()
                .zip(after_pred.iter())
                .map(|(a, b)| (a - b) * (a - b))
                .sum();
            if after_err <= surprise + 1e-6
                && self.candidate_bank.persisted_bytes() <= FAST_WEIGHT_BANK_MAX_BYTES
            {
                self.generation = self.generation.saturating_add(1);
                self.candidate_bank.generation = self.generation;
                self.active_bank = self.candidate_bank.clone();
            }
        }
        Ok((id, surprise))
    }

    /// Grounded hybrid recall: blends associative latent readout with query, retrieves via PQ + lexical check.
    pub fn recall(&self, query: &str, top_k: usize) -> Vec<ReliableRecallHit> {
        if query.len() > MAX_TEXT_BYTES {
            return Vec::new();
        }
        let q_val = encode_value_512(query);
        let q_key = encode_key_256(query);
        let assoc_val = self.active_bank.read(&q_key);
        let surprise: f32 = q_val
            .iter()
            .zip(assoc_val.iter())
            .map(|(a, b)| (a - b) * (a - b))
            .sum();

        let mut blended = [0.0f32; VALUE_DIM];
        for i in 0..VALUE_DIM {
            blended[i] = 0.75 * q_val[i] + 0.25 * assoc_val[i];
        }

        let query_lower = query.to_lowercase();
        let mut scored: Vec<(&ExplicitRecord, f32)> = self
            .records
            .iter()
            .map(|rec| {
                let pq_sim = self.codebook.asymmetric_dot(&blended, &rec.pq_code);
                let lexical_bonus =
                    if !query_lower.is_empty() && rec.text.to_lowercase().contains(&query_lower) {
                        0.35
                    } else {
                        0.0
                    };
                let score = (0.65 * pq_sim + 0.20 * rec.importance + lexical_bonus).clamp(0.0, 1.0);
                (rec, score)
            })
            .collect();

        scored.sort_by(|a, b| b.1.total_cmp(&a.1));
        scored
            .into_iter()
            .take(top_k.min(20))
            .map(|(rec, conf)| ReliableRecallHit {
                source_record_id: rec.id,
                text: rec.text.clone(),
                confidence: conf,
                timestamp_sec: rec.timestamp_sec,
                provenance: rec.provenance.clone(),
                verification_status: if conf >= 0.25 {
                    rec.verification.as_str()
                } else {
                    "unverified_estimate"
                },
                associative_surprise: surprise,
            })
            .collect()
    }

    /// Phase 4.7 Forgetting & Deletion: remove record from exact memory + PQ and rebuild fast weights
    /// strictly from remaining permitted foldable records.
    pub fn delete_and_rebuild(&mut self, record_id: u64) -> bool {
        let before = self.records.len();
        self.records.retain(|r| r.id != record_id);
        if self.records.len() == before {
            return false;
        }
        self.rebuild_fast_weights(self.generation.saturating_add(1));
        true
    }

    fn rebuild_fast_weights(&mut self, generation: u64) {
        let mut rebuilt = FastWeightBank::new(generation);
        for rec in &self.records {
            if rec.foldable {
                let key = encode_key_256(&rec.text);
                let val = encode_value_512(&rec.text);
                let mut candidate = rebuilt.clone();
                let before = candidate.update_top2(&key, &val, rec.importance.max(0.2), 0.35, 0.01);
                let predicted = candidate.read(&key);
                let after: f32 = val
                    .iter()
                    .zip(predicted)
                    .map(|(v, p)| (v - p) * (v - p))
                    .sum();
                if after <= before + 1e-6 {
                    rebuilt = candidate;
                }
            }
        }
        self.generation = generation;
        self.active_bank = rebuilt.clone();
        self.candidate_bank = rebuilt;
    }

    /// Exact byte telemetry proving compliance with the 45 MiB persistent cap.
    pub fn stats(&self) -> NeuralMemoryStats {
        let exact_bytes: usize = self
            .records
            .iter()
            .map(|r| RECORD_ACCOUNTING_BYTES + r.text.len() + r.provenance.len())
            .sum();
        let pq_index_bytes =
            self.codebook.fp16_footprint_bytes() + self.records.len() * (PQ_SUBSPACES + 8);
        let fast_weights_dual_bytes =
            self.active_bank.persisted_bytes() + self.candidate_bank.persisted_bytes();
        NeuralMemoryStats {
            exact_bytes,
            pq_index_bytes,
            fast_weights_dual_bytes,
            metadata_bytes: SUPERBLOCK_BYTES * 2,
            total_budget_bytes: MAX_PERSISTENT_MEMORY_BYTES,
            record_count: self.records.len(),
            active_generation: self.generation,
            total_fast_weight_updates: self.active_bank.total_updates(),
        }
    }

    /// Persist a synced, atomically replaced record snapshot (strictly `<= 45 MiB`).
    /// Fast weights are reconstructed from the admitted records when loading.
    pub fn save_ikmem2(&self, path: &Path) -> Result<usize, String> {
        let stats = self.stats();
        if stats.exact_bytes > REGION_EXACT_BYTES
            || stats.pq_index_bytes > REGION_PQ_BYTES
            || stats.fast_weights_dual_bytes > REGION_FAST_WEIGHTS_DUAL_BYTES
        {
            return Err("region budget exceeded before .ikmem2 write".to_string());
        }
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).map_err(|e| format!("mkdir {}: {e}", parent.display()))?;
        }

        let mut payload = Vec::new();
        payload.extend_from_slice(&(self.records.len() as u64).to_le_bytes());
        payload.extend_from_slice(&self.next_id.to_le_bytes());
        payload.extend_from_slice(&self.codebook.version.to_le_bytes());
        payload.extend_from_slice(&self.codebook.seed.to_le_bytes());
        for rec in &self.records {
            payload.extend_from_slice(&rec.id.to_le_bytes());
            payload.extend_from_slice(&rec.timestamp_sec.to_le_bytes());
            payload.extend_from_slice(&rec.importance.to_le_bytes());
            let flags: u8 = (match rec.verification {
                RecordVerification::VerifiedFact => 1,
                RecordVerification::OperatorInstruction => 2,
                RecordVerification::UnverifiedObservation => 3,
            }) | if rec.foldable { 0x80 } else { 0 };
            payload.push(flags);
            payload.extend_from_slice(&rec.pq_code);
            let tb = rec.text.as_bytes();
            let pb = rec.provenance.as_bytes();
            let tlen = u32::try_from(tb.len()).map_err(|_| "text too large".to_string())?;
            let plen = u32::try_from(pb.len()).map_err(|_| "provenance too large".to_string())?;
            payload.extend_from_slice(&tlen.to_le_bytes());
            payload.extend_from_slice(tb);
            payload.extend_from_slice(&plen.to_le_bytes());
            payload.extend_from_slice(pb);
        }

        let total_size = SUPERBLOCK_BYTES * 2 + payload.len();
        if total_size > MAX_PERSISTENT_MEMORY_BYTES {
            return Err(format!(
                ".ikmem2 container ({total_size} bytes) exceeds 45 MiB hard cap"
            ));
        }

        let payload_hash = fnv1a_64(&payload);
        let mut sb = [0u8; SUPERBLOCK_BYTES];
        sb[0..8].copy_from_slice(IKMEM2_MAGIC);
        sb[8..16].copy_from_slice(&self.generation.to_le_bytes());
        sb[16..24].copy_from_slice(&(payload.len() as u64).to_le_bytes());
        sb[24..32].copy_from_slice(&payload_hash.to_le_bytes());
        let sb_hash = fnv1a_64(&sb[0..56]);
        sb[56..64].copy_from_slice(&sb_hash.to_le_bytes());

        let sequence = SNAPSHOT_SEQUENCE.fetch_add(1, Ordering::Relaxed);
        let tmp = path.with_extension(format!("ikmem2.{}.{sequence}.tmp", std::process::id()));
        let mut f = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&tmp)
            .map_err(|e| e.to_string())?;
        let result = (|| {
            // Redundant metadata headers describe the same checksummed payload.
            f.write_all(&sb).map_err(|e| e.to_string())?;
            f.write_all(&sb).map_err(|e| e.to_string())?;
            f.write_all(&payload).map_err(|e| e.to_string())?;
            f.sync_all().map_err(|e| e.to_string())?;
            drop(f);
            fs::rename(&tmp, path).map_err(|e| e.to_string())?;
            Ok(total_size)
        })();
        if result.is_err() {
            let _ = fs::remove_file(&tmp);
        }
        result
    }

    /// Load and verify a `.ikmem2` container using dual-superblock fallback and payload checksum.
    pub fn load_ikmem2(path: &Path) -> Result<Self, String> {
        let meta = fs::symlink_metadata(path).map_err(|e| e.to_string())?;
        if !meta.is_file() || meta.len() > MAX_PERSISTENT_MEMORY_BYTES as u64 {
            return Err("ikmem2 must be a regular file within the 45 MiB limit".to_string());
        }
        let mut raw = Vec::with_capacity(meta.len() as usize);
        File::open(path)
            .map_err(|e| e.to_string())?
            .take(MAX_PERSISTENT_MEMORY_BYTES as u64 + 1)
            .read_to_end(&mut raw)
            .map_err(|e| e.to_string())?;
        if raw.len() > MAX_PERSISTENT_MEMORY_BYTES {
            return Err("ikmem2 grew beyond the 45 MiB limit while reading".to_string());
        }
        if raw.len() < SUPERBLOCK_BYTES * 2 {
            return Err("truncated ikmem2 dual superblock".to_string());
        }

        let parse_sb = |slice: &[u8]| -> Option<(u64, usize, u64)> {
            if &slice[0..8] != IKMEM2_MAGIC {
                return None;
            }
            let expected_sb = u64::from_le_bytes([
                slice[56], slice[57], slice[58], slice[59], slice[60], slice[61], slice[62],
                slice[63],
            ]);
            if fnv1a_64(&slice[0..56]) != expected_sb {
                return None;
            }
            let sb_gen = u64::from_le_bytes([
                slice[8], slice[9], slice[10], slice[11], slice[12], slice[13], slice[14],
                slice[15],
            ]);
            let plen = u64::from_le_bytes([
                slice[16], slice[17], slice[18], slice[19], slice[20], slice[21], slice[22],
                slice[23],
            ]);
            let plen = usize::try_from(plen).ok()?;
            let phash = u64::from_le_bytes([
                slice[24], slice[25], slice[26], slice[27], slice[28], slice[29], slice[30],
                slice[31],
            ]);
            Some((sb_gen, plen, phash))
        };

        let sb_a = parse_sb(&raw[0..SUPERBLOCK_BYTES]);
        let sb_b = parse_sb(&raw[SUPERBLOCK_BYTES..SUPERBLOCK_BYTES * 2]);
        let (generation, payload_len, payload_hash) = match (sb_a, sb_b) {
            (Some(a), Some(b)) => {
                if a.0 >= b.0 {
                    a
                } else {
                    b
                }
            }
            (Some(a), None) => a,
            (None, Some(b)) => b,
            (None, None) => return Err("both ikmem2 superblocks are corrupted".to_string()),
        };

        let payload = &raw[SUPERBLOCK_BYTES * 2..];
        if payload.len() != payload_len || fnv1a_64(payload) != payload_hash {
            return Err("ikmem2 payload checksum mismatch".to_string());
        }
        if payload.len() < 28 {
            return Err("truncated ikmem2 payload header".to_string());
        }
        let count = u64::from_le_bytes([
            payload[0], payload[1], payload[2], payload[3], payload[4], payload[5], payload[6],
            payload[7],
        ]);
        let count = usize::try_from(count).map_err(|_| "invalid ikmem2 record count")?;
        if count > MAX_RECORDS || count > (payload.len() - 28) / (61 + 1) {
            return Err("ikmem2 record count exceeds admission limits".to_string());
        }
        let next_id = u64::from_le_bytes([
            payload[8],
            payload[9],
            payload[10],
            payload[11],
            payload[12],
            payload[13],
            payload[14],
            payload[15],
        ]);
        let version = u32::from_le_bytes([payload[16], payload[17], payload[18], payload[19]]);
        let seed = u64::from_le_bytes([
            payload[20],
            payload[21],
            payload[22],
            payload[23],
            payload[24],
            payload[25],
            payload[26],
            payload[27],
        ]);

        let mut cursor = 28usize;
        let mut mem = Self::new(version, seed);
        mem.next_id = next_id;
        mem.generation = generation;
        let mut exact_bytes = 0usize;
        let mut last_id = 0u64;

        for _ in 0..count {
            if cursor + 21 + PQ_SUBSPACES + 4 > payload.len() {
                return Err("truncated ikmem2 record".to_string());
            }
            let id = u64::from_le_bytes([
                payload[cursor],
                payload[cursor + 1],
                payload[cursor + 2],
                payload[cursor + 3],
                payload[cursor + 4],
                payload[cursor + 5],
                payload[cursor + 6],
                payload[cursor + 7],
            ]);
            cursor += 8;
            let timestamp_sec = u64::from_le_bytes([
                payload[cursor],
                payload[cursor + 1],
                payload[cursor + 2],
                payload[cursor + 3],
                payload[cursor + 4],
                payload[cursor + 5],
                payload[cursor + 6],
                payload[cursor + 7],
            ]);
            cursor += 8;
            let importance = f32::from_le_bytes([
                payload[cursor],
                payload[cursor + 1],
                payload[cursor + 2],
                payload[cursor + 3],
            ]);
            cursor += 4;
            let flags = payload[cursor];
            cursor += 1;
            let foldable = (flags & 0x80) != 0;
            let verification = match flags & 0x0f {
                1 => RecordVerification::VerifiedFact,
                2 => RecordVerification::OperatorInstruction,
                3 => RecordVerification::UnverifiedObservation,
                _ => return Err("invalid ikmem2 verification flags".to_string()),
            };
            if id <= last_id
                || id >= next_id
                || flags & 0x70 != 0
                || !importance.is_finite()
                || !(0.0..=1.0).contains(&importance)
                || (foldable && verification == RecordVerification::UnverifiedObservation)
            {
                return Err("invalid ikmem2 record metadata".to_string());
            }
            last_id = id;
            let mut pq_code = [0u8; PQ_SUBSPACES];
            pq_code.copy_from_slice(&payload[cursor..cursor + PQ_SUBSPACES]);
            cursor += PQ_SUBSPACES;

            let tlen = u32::from_le_bytes([
                payload[cursor],
                payload[cursor + 1],
                payload[cursor + 2],
                payload[cursor + 3],
            ]) as usize;
            cursor += 4;
            if tlen == 0 || tlen > MAX_TEXT_BYTES || cursor + tlen + 4 > payload.len() {
                return Err("truncated ikmem2 text".to_string());
            }
            let text = std::str::from_utf8(&payload[cursor..cursor + tlen])
                .map_err(|e| e.to_string())?
                .to_owned();
            cursor += tlen;

            let plen = u32::from_le_bytes([
                payload[cursor],
                payload[cursor + 1],
                payload[cursor + 2],
                payload[cursor + 3],
            ]) as usize;
            cursor += 4;
            if plen > MAX_PROVENANCE_BYTES || cursor + plen > payload.len() {
                return Err("truncated ikmem2 provenance".to_string());
            }
            let provenance = std::str::from_utf8(&payload[cursor..cursor + plen])
                .map_err(|e| e.to_string())?
                .to_owned();
            cursor += plen;

            exact_bytes += RECORD_ACCOUNTING_BYTES + tlen + plen;
            if exact_bytes > REGION_EXACT_BYTES || text.trim().is_empty() {
                return Err("ikmem2 exact record budget exceeded".to_string());
            }
            mem.records.push(ExplicitRecord {
                id,
                timestamp_sec,
                text,
                importance,
                provenance,
                verification,
                foldable,
                pq_code,
            });
        }
        if cursor != payload.len() || next_id == 0 {
            return Err("ikmem2 has trailing bytes or an invalid next ID".to_string());
        }
        mem.rebuild_fast_weights(generation);
        Ok(mem)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn binary16_preserves_subnormal_scales_and_rounds_even() {
        for bits in 0u16..=u16::MAX {
            if bits & 0x7c00 != 0x7c00 {
                assert_eq!(f32_to_f16_bits(f16_bits_to_f32(bits)), bits);
            }
        }
        assert_eq!(f16_bits_to_f32(1), 1.0 / 16_777_216.0);
        assert_eq!(f32_to_f16_bits(1.0 + 1.0 / 2048.0), 0x3c00);
        assert!(f16_bits_to_f32(f32_to_f16_bits(f32::NAN)).is_nan());
    }

    #[test]
    fn normalized_first_update_has_nonzero_prediction_and_rejects_nan() {
        let mut shard = AssociativeShard::new(0);
        let key = [1.0 / 16.0; KEY_DIM];
        let value = [1.0 / (VALUE_DIM as f32).sqrt(); VALUE_DIM];
        let before = shard.gated_update(&key, &value, 1.0, 0.35, 0.01);
        let predicted = shard.read(&key);
        assert!(predicted.iter().any(|v| *v > 0.0));
        let after: f32 = value
            .iter()
            .zip(predicted)
            .map(|(v, p)| (v - p) * (v - p))
            .sum();
        assert!(after < before * 0.5);
        let snapshot = shard.weights_q8.clone();
        assert!(
            shard
                .gated_update(&key, &value, f32::NAN, 0.35, 0.01)
                .is_infinite()
        );
        assert_eq!(shard.weights_q8, snapshot);
        assert_eq!(shard.updates, 1);
    }

    #[test]
    fn admission_rejects_oversized_and_nonfinite_records_without_mutation() {
        let mut mem = NeuralMemory45::default();
        for (text, importance, provenance) in [
            ("x".repeat(MAX_TEXT_BYTES + 1), 0.5, "source".to_string()),
            ("text".to_string(), f32::NAN, "source".to_string()),
            (
                "text".to_string(),
                0.5,
                "x".repeat(MAX_PROVENANCE_BYTES + 1),
            ),
        ] {
            assert!(
                mem.ingest(
                    text,
                    importance,
                    provenance,
                    RecordVerification::VerifiedFact,
                    true,
                    true,
                    false
                )
                .is_err()
            );
        }
        assert_eq!(mem.stats().record_count, 0);
        assert_eq!(mem.stats().total_fast_weight_updates, 0);
        assert_eq!(mem.next_id, 1);
        let record = ExplicitRecord {
            id: 1,
            timestamp_sec: 0,
            text: "existing".to_string(),
            importance: 0.5,
            provenance: "fixture".to_string(),
            verification: RecordVerification::UnverifiedObservation,
            foldable: false,
            pq_code: [0; PQ_SUBSPACES],
        };
        mem.records.resize(MAX_RECORDS, record);
        assert!(
            mem.ingest(
                "one more",
                0.5,
                "fixture",
                RecordVerification::UnverifiedObservation,
                false,
                false,
                false
            )
            .is_err()
        );
        assert_eq!(mem.stats().record_count, MAX_RECORDS);
        assert_eq!(mem.next_id, 1);
    }

    #[test]
    fn checksummed_snapshot_still_rejects_nonfinite_record_metadata() -> Result<(), String> {
        let mut mem = NeuralMemory45::default();
        mem.ingest(
            "explicit record",
            0.5,
            "fixture",
            RecordVerification::UnverifiedObservation,
            false,
            false,
            false,
        )?;
        let serial = SNAPSHOT_SEQUENCE.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!(
            "inkling2_ikmem2_invalid_{}_{serial}",
            std::process::id()
        ));
        let path = dir.join("memory.ikmem2");
        mem.save_ikmem2(&path)?;
        let mut bytes = fs::read(&path).map_err(|e| e.to_string())?;
        let importance_offset = SUPERBLOCK_BYTES * 2 + 28 + 16;
        bytes[importance_offset..importance_offset + 4].copy_from_slice(&f32::NAN.to_le_bytes());
        let payload_hash = fnv1a_64(&bytes[SUPERBLOCK_BYTES * 2..]);
        for offset in [0, SUPERBLOCK_BYTES] {
            bytes[offset + 24..offset + 32].copy_from_slice(&payload_hash.to_le_bytes());
            let header_hash = fnv1a_64(&bytes[offset..offset + 56]);
            bytes[offset + 56..offset + 64].copy_from_slice(&header_hash.to_le_bytes());
        }
        fs::write(&path, bytes).map_err(|e| e.to_string())?;
        assert!(NeuralMemory45::load_ikmem2(&path).is_err());
        let _ = fs::remove_dir_all(dir);
        Ok(())
    }

    #[test]
    fn fast_weight_bank_fits_5mib_and_reduces_surprise() {
        let mut bank = FastWeightBank::new(0);
        assert_eq!(FAST_WEIGHT_PARAMS, 4_194_304);
        assert!(bank.persisted_bytes() <= FAST_WEIGHT_BANK_MAX_BYTES);

        let key = encode_key_256("RTX 5060 Ti Blackwell SC117");
        let val = encode_value_512("RTX 5060 Ti Blackwell SC117");
        let s0 = bank.update_top2(&key, &val, 0.9, 0.5, 0.01);
        let s1 = bank.update_top2(&key, &val, 0.9, 0.5, 0.01);
        assert!(
            s1 < s0,
            "gated update must reduce associative prediction error"
        );
    }

    #[test]
    fn neural_memory45_ingests_recalls_forgets_and_persists_ikmem2() -> Result<(), String> {
        let mut mem = NeuralMemory45::default();
        let (id1, _) = mem.ingest(
            "Operator prefers Japanese responses on RTX 5060 Ti",
            0.95,
            "operator-session",
            RecordVerification::OperatorInstruction,
            true,
            true,
            false,
        )?;
        // Untrusted observation must NOT fold into fast weights.
        let updates_before = mem.stats().total_fast_weight_updates;
        let (_id2, _) = mem.ingest(
            "Untrusted web banner text",
            0.9,
            "web-scrape",
            RecordVerification::UnverifiedObservation,
            false,
            false,
            false,
        )?;
        assert_eq!(mem.stats().total_fast_weight_updates, updates_before);

        let hits = mem.recall("RTX 5060 Ti", 2);
        assert!(!hits.is_empty());
        assert_eq!(hits[0].source_record_id, id1);
        assert_eq!(hits[0].verification_status, "operator_instruction");

        let stats = mem.stats();
        assert!(stats.pq_index_bytes >= PQ_CODEBOOK_FP16_BYTES);
        assert!(stats.fast_weights_dual_bytes <= REGION_FAST_WEIGHTS_DUAL_BYTES);

        let dir = std::env::temp_dir().join(format!("inkling2_ikmem2_test_{}", now_sec()));
        let path = dir.join("memory.ikmem2");
        let written = mem.save_ikmem2(&path);
        assert!(written.is_ok());

        // Corrupt Slot A superblock; Slot B must recover cleanly.
        if let Ok(mut bytes) = fs::read(&path) {
            bytes[0] ^= 0xFF;
            let _ = fs::write(&path, &bytes);
        }
        let loaded = NeuralMemory45::load_ikmem2(&path);
        assert!(loaded.is_ok());
        if let Ok(mut restored) = loaded {
            assert_eq!(restored.stats().record_count, 2);
            let key = encode_key_256("Operator prefers Japanese responses on RTX 5060 Ti");
            assert_eq!(restored.active_bank.read(&key), mem.active_bank.read(&key));
            assert!(restored.delete_and_rebuild(id1));
            assert_eq!(restored.stats().record_count, 1);
            assert_eq!(restored.active_bank.total_updates(), 0);
            restored.save_ikmem2(&path)?;
            let deleted = NeuralMemory45::load_ikmem2(&path)?;
            assert_eq!(deleted.stats().record_count, 1);
            assert!(
                deleted
                    .recall("Operator", 20)
                    .iter()
                    .all(|h| h.source_record_id != id1)
            );
        }
        let _ = fs::remove_dir_all(&dir);
        Ok(())
    }
}
