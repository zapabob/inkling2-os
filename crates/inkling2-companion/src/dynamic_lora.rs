//! Dynamic Weights & Online LoRA Self-Evolution Engine (`dynamic_lora`).
//!
//! Forms live low-rank adapter matrices `A: [rank, D]` and `B: [D, rank]` aligned with
//! Inkling-2-Mini's `hidden_size = 2304`, supporting:
//! - Residual delta forward pass: `Δh = (alpha / rank) * B * (A * h)` with L2 safety clamp
//! - Online self-reinforcement updates from conversational reward signals `r ∈ [-1.0, +1.0]`
//! - Atomic `.iklora` checkpointing + instant rollback to prevent catastrophic drift

use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::Path;

const LORA_MAGIC: &[u8; 8] = b"IKLORA1\0";
const MAX_HIDDEN_DIM: usize = 16_384;
const MAX_RANK: usize = 128;
const MAX_CONSOLIDATION_RANK: usize = 128;

fn validate_config(cfg: LoraConfig) -> Result<usize, String> {
    if !(1..=MAX_HIDDEN_DIM).contains(&cfg.hidden_dim)
        || !(1..=MAX_RANK).contains(&cfg.rank)
        || !cfg.alpha.is_finite()
        || cfg.alpha < 0.0
        || !cfg.learning_rate.is_finite()
        || cfg.learning_rate <= 0.0
        || !cfg.weight_decay.is_finite()
        || !(0.0..=1.0).contains(&cfg.weight_decay)
        || !cfg.max_delta_norm.is_finite()
        || cfg.max_delta_norm <= 0.0
    {
        return Err("invalid or unbounded LoRA configuration".to_string());
    }
    cfg.hidden_dim
        .checked_mul(cfg.rank)
        .ok_or_else(|| "adapter dimensions overflow".to_string())
}

/// Configuration for an Inkling-2-Mini dynamic LoRA adapter.
#[derive(Debug, Clone, Copy)]
pub struct LoraConfig {
    /// Hidden dimension (Inkling-2-Mini = 2304).
    pub hidden_dim: usize,
    /// Low-rank bottleneck (default = 16).
    pub rank: usize,
    /// Scaling factor `alpha` (effective scale = `alpha / rank`).
    pub alpha: f32,
    /// Base learning rate for online reinforcement updates.
    pub learning_rate: f32,
    /// L2 weight decay to prevent unbounded parameter drift.
    pub weight_decay: f32,
    /// Maximum L2 norm of injected `Δh` per step (stability guardrail).
    pub max_delta_norm: f32,
}

impl Default for LoraConfig {
    fn default() -> Self {
        Self {
            hidden_dim: 2304,
            rank: 16,
            alpha: 32.0,
            learning_rate: 1e-3,
            weight_decay: 1e-4,
            max_delta_norm: 2.0,
        }
    }
}

/// Telemetry for the dynamic LoRA adapter.
#[derive(Debug, Clone, Copy)]
pub struct LoraStats {
    /// Total online reinforcement steps applied.
    pub evolution_steps: u64,
    /// Cumulative reward signal.
    pub cumulative_reward: f64,
    /// Frobenius norm of `A`.
    pub norm_a: f32,
    /// Frobenius norm of `B`.
    pub norm_b: f32,
    /// Total parameter count (`2 * rank * hidden_dim`).
    pub param_count: usize,
}

/// Live Dynamic LoRA adapter with online self-reinforcement and rollback snapshot.
#[derive(Debug, Clone)]
pub struct DynamicLoraAdapter {
    cfg: LoraConfig,
    /// `A`: `[rank, hidden_dim]` row-major.
    a: Vec<f32>,
    /// `B`: `[hidden_dim, rank]` row-major (zero-initialized -> exact identity at step 0).
    b: Vec<f32>,
    /// Previous generation snapshot for safe rollback.
    prev_a: Option<Vec<f32>>,
    prev_b: Option<Vec<f32>>,
    evolution_steps: u64,
    cumulative_reward: f64,
    prev_steps: Option<u64>,
    prev_reward: Option<f64>,
}

impl DynamicLoraAdapter {
    /// Initialize an identity-preserving adapter. Invalid configurations use defaults.
    /// Use `try_new` to report invalid caller-supplied configurations.
    pub fn new(cfg: LoraConfig) -> Self {
        // Preserve the infallible API with a bounded default for invalid trusted
        // configuration. Use try_new for caller-supplied configuration.
        let cfg = if validate_config(cfg).is_ok() {
            cfg
        } else {
            LoraConfig::default()
        };
        Self::from_valid_config(cfg)
    }

    /// Construct an adapter after validating caller-supplied configuration.
    pub fn try_new(cfg: LoraConfig) -> Result<Self, String> {
        validate_config(cfg)?;
        Ok(Self::from_valid_config(cfg))
    }

    fn from_valid_config(cfg: LoraConfig) -> Self {
        let mut a = vec![0.0f32; cfg.rank * cfg.hidden_dim];
        // Deterministic Kaiming-uniform-like initialization without external RNG crate
        let scale = (1.0 / (cfg.hidden_dim as f32)).sqrt();
        let mut state: u64 = 0x9e3779b97f4a7c15;
        for v in &mut a {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let u = ((state >> 33) as f32) / (u32::MAX as f32) * 2.0 - 1.0;
            *v = u * scale;
        }
        let b = vec![0.0f32; cfg.hidden_dim * cfg.rank];
        Self {
            cfg,
            a,
            b,
            prev_a: None,
            prev_b: None,
            evolution_steps: 0,
            cumulative_reward: 0.0,
            prev_steps: None,
            prev_reward: None,
        }
    }

    /// Compute `Δh = clamp((alpha / rank) * B * (A * h), max_delta_norm)`.
    pub fn forward_delta(&self, h: &[f32]) -> Vec<f32> {
        let d = self.cfg.hidden_dim;
        let r = self.cfg.rank;
        if h.len() != d || h.iter().any(|v| !v.is_finite()) {
            return vec![0.0; d];
        }
        let mut z = vec![0.0f32; r];
        for (ri, projected) in z.iter_mut().enumerate() {
            let row = &self.a[ri * d..(ri + 1) * d];
            let mut acc = 0.0f32;
            for (wv, &hv) in row.iter().zip(h.iter().take(d)) {
                acc += wv * hv;
            }
            *projected = acc;
        }

        let scale = self.cfg.alpha / (r as f32);
        let mut delta = vec![0.0f32; d];
        let mut sq_norm = 0.0f64;
        for (di, update) in delta.iter_mut().enumerate() {
            let row = &self.b[di * r..(di + 1) * r];
            let mut acc = 0.0f32;
            for ri in 0..r {
                acc += row[ri] * z[ri];
            }
            let val = acc * scale;
            if !val.is_finite() {
                return vec![0.0; d];
            }
            *update = val;
            sq_norm += f64::from(val) * f64::from(val);
        }

        let norm = sq_norm.sqrt();
        if norm > f64::from(self.cfg.max_delta_norm) {
            let c = (f64::from(self.cfg.max_delta_norm) / norm) as f32;
            for v in &mut delta {
                *v *= c;
            }
        }
        delta
    }

    /// Perform one bounded online self-reinforcement LoRA update from `(h, target_dir, reward)`.
    pub fn reinforce_step(&mut self, h: &[f32], target_dir: &[f32], reward: f32) {
        if !reward.is_finite()
            || h.len() != self.cfg.hidden_dim
            || target_dir.len() != self.cfg.hidden_dim
            || h.iter().chain(target_dir).any(|v| !v.is_finite())
        {
            return;
        }
        let r_clamped = reward.clamp(-1.0, 1.0);
        if r_clamped.abs() < 1e-6 {
            return;
        }
        let mut next_a = self.a.clone();
        let mut next_b = self.b.clone();

        let d = self.cfg.hidden_dim;
        let r = self.cfg.rank;
        let lr = self.cfg.learning_rate * r_clamped;
        let wd = 1.0 - self.cfg.weight_decay;

        // z = A * h
        let mut z = vec![0.0f32; r];
        for (ri, projected) in z.iter_mut().enumerate() {
            let row = &self.a[ri * d..(ri + 1) * d];
            let mut acc = 0.0f32;
            for (wv, &hv) in row.iter().zip(h.iter().take(d)) {
                acc += wv * hv;
            }
            *projected = acc;
        }

        // g_z = B^T * target_dir
        let mut gz = vec![0.0f32; r];
        for (di, &td) in target_dir.iter().take(d).enumerate() {
            let row = &self.b[di * r..(di + 1) * r];
            for ri in 0..r {
                gz[ri] += row[ri] * td;
            }
        }

        // Update B += lr * target_dir * z^T
        for (di, &td) in target_dir.iter().take(d).enumerate() {
            let row = &mut next_b[di * r..(di + 1) * r];
            for ri in 0..r {
                row[ri] = row[ri] * wd + lr * td * z[ri];
            }
        }

        // Update A += lr * g_z * h^T
        for (ri, &g) in gz.iter().enumerate() {
            let row = &mut next_a[ri * d..(ri + 1) * d];
            for (wv, &hv) in row.iter_mut().zip(h.iter().take(d)) {
                *wv = *wv * wd + lr * g * hv;
            }
        }

        if next_a.iter().chain(&next_b).any(|v| !v.is_finite()) {
            return;
        }
        let Some(next_steps) = self.evolution_steps.checked_add(1) else {
            return;
        };
        self.prev_a = Some(std::mem::replace(&mut self.a, next_a));
        self.prev_b = Some(std::mem::replace(&mut self.b, next_b));
        self.prev_steps = Some(self.evolution_steps);
        self.prev_reward = Some(self.cumulative_reward);
        self.evolution_steps = next_steps;
        self.cumulative_reward += f64::from(r_clamped);
    }

    /// Roll back to the previous weight snapshot if a self-reinforcement step degraded quality.
    pub fn rollback(&mut self) -> bool {
        if let (Some(pa), Some(pb)) = (self.prev_a.take(), self.prev_b.take()) {
            self.a = pa;
            self.b = pb;
            self.evolution_steps = self.prev_steps.take().unwrap_or(self.evolution_steps);
            self.cumulative_reward = self.prev_reward.take().unwrap_or(self.cumulative_reward);
            true
        } else {
            false
        }
    }

    /// Adapter statistics.
    pub fn stats(&self) -> LoraStats {
        let norm_a = self.a.iter().map(|v| v * v).sum::<f32>().sqrt();
        let norm_b = self.b.iter().map(|v| v * v).sum::<f32>().sqrt();
        LoraStats {
            evolution_steps: self.evolution_steps,
            cumulative_reward: self.cumulative_reward,
            norm_a,
            norm_b,
            param_count: self.a.len() + self.b.len(),
        }
    }

    /// Save dynamic LoRA weights atomically to `.iklora`.
    pub fn save_atomic(&self, path: &Path) -> Result<(), String> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).map_err(|e| format!("mkdir {}: {e}", parent.display()))?;
        }
        let tmp = path.with_extension("iklora.tmp");
        let mut f = File::create(&tmp).map_err(|e| e.to_string())?;
        f.write_all(LORA_MAGIC).map_err(|e| e.to_string())?;
        f.write_all(&(self.cfg.hidden_dim as u32).to_le_bytes())
            .map_err(|e| e.to_string())?;
        f.write_all(&(self.cfg.rank as u32).to_le_bytes())
            .map_err(|e| e.to_string())?;
        f.write_all(&self.cfg.alpha.to_le_bytes())
            .map_err(|e| e.to_string())?;
        f.write_all(&self.evolution_steps.to_le_bytes())
            .map_err(|e| e.to_string())?;
        for &v in &self.a {
            f.write_all(&v.to_le_bytes()).map_err(|e| e.to_string())?;
        }
        for &v in &self.b {
            f.write_all(&v.to_le_bytes()).map_err(|e| e.to_string())?;
        }
        f.sync_all().map_err(|e| e.to_string())?;
        drop(f);
        fs::rename(&tmp, path).map_err(|e| e.to_string())?;
        Ok(())
    }

    /// Load dynamic LoRA weights from `.iklora`.
    pub fn load(path: &Path) -> Result<Self, String> {
        let mut f = File::open(path).map_err(|e| e.to_string())?;
        let mut hdr = [0u8; 28];
        f.read_exact(&mut hdr).map_err(|e| e.to_string())?;
        if &hdr[0..8] != LORA_MAGIC {
            return Err("invalid IKLORA1 magic".to_string());
        }
        let hidden_dim = u32::from_le_bytes([hdr[8], hdr[9], hdr[10], hdr[11]]) as usize;
        let rank = u32::from_le_bytes([hdr[12], hdr[13], hdr[14], hdr[15]]) as usize;
        let alpha = f32::from_le_bytes([hdr[16], hdr[17], hdr[18], hdr[19]]);
        let evolution_steps = u64::from_le_bytes([
            hdr[20], hdr[21], hdr[22], hdr[23], hdr[24], hdr[25], hdr[26], hdr[27],
        ]);

        let cfg = LoraConfig {
            hidden_dim,
            rank,
            alpha,
            ..LoraConfig::default()
        };
        let elems = validate_config(cfg)?;
        let expected_bytes = elems
            .checked_mul(8)
            .and_then(|n| n.checked_add(28))
            .ok_or_else(|| "adapter payload length overflow".to_string())?;
        let metadata = f.metadata().map_err(|e| e.to_string())?;
        if !metadata.is_file() || metadata.len() != expected_bytes as u64 {
            return Err("adapter file size does not match bounded dimensions".to_string());
        }
        let mut a = vec![0.0f32; elems];
        let mut b = vec![0.0f32; hidden_dim * rank];
        let mut buf = [0u8; 4];
        for v in &mut a {
            f.read_exact(&mut buf).map_err(|e| e.to_string())?;
            *v = f32::from_le_bytes(buf);
        }
        for v in &mut b {
            f.read_exact(&mut buf).map_err(|e| e.to_string())?;
            *v = f32::from_le_bytes(buf);
        }
        if a.iter().chain(&b).any(|v| !v.is_finite()) {
            return Err("adapter matrices contain non-finite values".to_string());
        }
        Ok(Self {
            cfg,
            a,
            b,
            prev_a: None,
            prev_b: None,
            evolution_steps,
            cumulative_reward: 0.0,
            prev_steps: None,
            prev_reward: None,
        })
    }

    /// Read-only access to `A: [rank, hidden_dim]`.
    pub fn matrix_a(&self) -> &[f32] {
        &self.a
    }

    /// Read-only access to `B: [hidden_dim, rank]`.
    pub fn matrix_b(&self) -> &[f32] {
        &self.b
    }

    /// Configuration of this adapter.
    pub fn config(&self) -> LoraConfig {
        self.cfg
    }
}

/// Phase 6.3 Anti-Forgetting benchmark gate comparing baseline vs candidate scores.
#[derive(Debug, Clone, Copy)]
pub struct BaselineEvalScores {
    pub math: f32,
    pub coding: f32,
    pub language_ja_en: f32,
    pub long_context: f32,
    pub tool_calling: f32,
    pub instruction_following: f32,
    pub privacy_security: f32,
}

/// Accept a candidate adapter only if no capability regresses beyond `max_drop` and security never drops.
pub fn approve_candidate_adapter(
    baseline: BaselineEvalScores,
    candidate: BaselineEvalScores,
    max_drop: f32,
) -> Result<(), &'static str> {
    let scores = |s: BaselineEvalScores| {
        [
            s.math,
            s.coding,
            s.language_ja_en,
            s.long_context,
            s.tool_calling,
            s.instruction_following,
            s.privacy_security,
        ]
    };
    if !max_drop.is_finite()
        || !(0.0..=1.0).contains(&max_drop)
        || scores(baseline)
            .into_iter()
            .chain(scores(candidate))
            .any(|v| !v.is_finite() || !(0.0..=1.0).contains(&v))
    {
        return Err("invalid benchmark scores or tolerance");
    }
    let tol = max_drop;
    if candidate.privacy_security + 1e-6 < baseline.privacy_security {
        return Err("candidate regressed on privacy/security");
    }
    if candidate.math + tol < baseline.math
        || candidate.coding + tol < baseline.coding
        || candidate.language_ja_en + tol < baseline.language_ja_en
        || candidate.long_context + tol < baseline.long_context
        || candidate.tool_calling + tol < baseline.tool_calling
        || candidate.instruction_following + tol < baseline.instruction_following
    {
        return Err("candidate regressed on baseline capability benchmark");
    }
    Ok(())
}

/// Thin modified Gram-Schmidt QR decomposition of `M` (`rows x cols`, row-major) into `Q` (`rows x cols`) and `R` (`cols x cols`).
fn thin_qr(m: &[f32], rows: usize, cols: usize) -> (Vec<f32>, Vec<f32>) {
    let mut q = vec![0.0f32; rows * cols];
    let mut r = vec![0.0f32; cols * cols];
    for j in 0..cols {
        let mut v = vec![0.0f32; rows];
        for i in 0..rows {
            v[i] = m[i * cols + j];
        }
        for k in 0..j {
            let mut dot = 0.0f32;
            for i in 0..rows {
                dot += q[i * cols + k] * v[i];
            }
            r[k * cols + j] = dot;
            for i in 0..rows {
                v[i] -= dot * q[i * cols + k];
            }
        }
        let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        r[j * cols + j] = norm;
        if norm > 1e-8 {
            for i in 0..rows {
                q[i * cols + j] = v[i] / norm;
            }
        }
    }
    (q, r)
}

/// Power-iteration deflated top-`k` SVD on a small `n x n` matrix `M` (row-major).
/// Returns `(U: [n, k], S: [k], Vt: [k, n])`.
fn small_truncated_svd(m: &[f32], n: usize, k: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let r = k.min(n);
    let mut rem = m.to_vec();
    let mut u_out = vec![0.0f32; n * r];
    let mut s_out = vec![0.0f32; r];
    let mut vt_out = vec![0.0f32; r * n];

    for comp in 0..r {
        let mut v = vec![0.0f32; n];
        for (idx, val) in v.iter_mut().enumerate() {
            *val = if idx == comp {
                1.0
            } else {
                0.01 * ((idx + 1) as f32)
            };
        }
        let mut norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 1e-8 {
            for x in &mut v {
                *x /= norm;
            }
        }

        let mut u = vec![0.0f32; n];
        let mut sigma = 0.0f32;
        for _ in 0..24 {
            // u = rem * v
            for i in 0..n {
                let mut acc = 0.0f32;
                for j in 0..n {
                    acc += rem[i * n + j] * v[j];
                }
                u[i] = acc;
            }
            let un = u.iter().map(|x| x * x).sum::<f32>().sqrt();
            if un <= 1e-8 {
                sigma = 0.0;
                break;
            }
            for x in &mut u {
                *x /= un;
            }
            // v = rem^T * u
            for j in 0..n {
                let mut acc = 0.0f32;
                for i in 0..n {
                    acc += rem[i * n + j] * u[i];
                }
                v[j] = acc;
            }
            norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            sigma = norm;
            if norm <= 1e-8 {
                break;
            }
            for x in &mut v {
                *x /= norm;
            }
        }

        s_out[comp] = sigma;
        for i in 0..n {
            u_out[i * r + comp] = u[i];
            vt_out[comp * n + i] = v[i];
        }
        if sigma > 1e-8 {
            for i in 0..n {
                for j in 0..n {
                    rem[i * n + j] -= sigma * u[i] * v[j];
                }
            }
        }
    }

    (u_out, s_out, vt_out)
}

/// Phase 6.2 Low-Rank Weight Consolidation without materialising a dense `D x D` `delta_W`.
///
/// Folds `sum_i (alpha_i / r_i) * B_i @ A_i` into a single `target_rank` adapter via
/// thin QR factorisation of concatenated low-rank factors followed by a small SVD of `R_B @ R_A^T`.
pub fn consolidate_adapters(
    adapters: &[&DynamicLoraAdapter],
    target_rank: usize,
) -> Result<DynamicLoraAdapter, String> {
    if adapters.is_empty() || target_rank == 0 {
        return Err(
            "consolidate_adapters requires at least one adapter and target_rank > 0".to_string(),
        );
    }
    let d = adapters[0].cfg.hidden_dim;
    if target_rank > MAX_RANK || adapters.len() > MAX_CONSOLIDATION_RANK {
        return Err("consolidation exceeds the bounded rank budget".to_string());
    }
    let mut total_r = 0usize;
    for adapter in adapters {
        validate_config(adapter.cfg)?;
        total_r = total_r
            .checked_add(adapter.cfg.rank)
            .filter(|r| *r <= MAX_CONSOLIDATION_RANK)
            .ok_or_else(|| "consolidation exceeds the bounded rank budget".to_string())?;
    }
    if adapters.iter().any(|a| a.cfg.hidden_dim != d) {
        return Err("all adapters must share the same hidden_dim".to_string());
    }

    let evolution_steps = adapters.iter().try_fold(0_u64, |steps, a| {
        steps
            .checked_add(a.evolution_steps)
            .ok_or_else(|| "consolidated evolution count overflows".to_string())
    })?;

    // Build concatenated B_cat: [d, total_r] and A_cat_t: [d, total_r] (scaled by alpha/rank).
    let mut b_cat = vec![0.0f32; d * total_r];
    let mut a_cat_t = vec![0.0f32; d * total_r];
    let mut col_offset = 0usize;
    for ad in adapters {
        let r = ad.cfg.rank;
        let scale = ad.cfg.alpha / (r as f32);
        for i in 0..d {
            for j in 0..r {
                b_cat[i * total_r + col_offset + j] = ad.b[i * r + j] * scale;
                a_cat_t[i * total_r + col_offset + j] = ad.a[j * d + i];
            }
        }
        col_offset += r;
    }

    let (qb, rb) = thin_qr(&b_cat, d, total_r);
    let (qa, ra) = thin_qr(&a_cat_t, d, total_r);

    // Core matrix M = R_B @ R_A^T : [total_r, total_r]
    let mut core = vec![0.0f32; total_r * total_r];
    for i in 0..total_r {
        for j in 0..total_r {
            let mut acc = 0.0f32;
            for k in 0..total_r {
                acc += rb[i * total_r + k] * ra[j * total_r + k];
            }
            core[i * total_r + j] = acc;
        }
    }

    let eff_r = target_rank.min(total_r);
    let (u_core, s_core, vt_core) = small_truncated_svd(&core, total_r, eff_r);

    // B_new = Q_B @ (U_core * sqrt(S)), A_new = (sqrt(S) * Vt_core) @ Q_A^T
    let mut b_new = vec![0.0f32; d * target_rank];
    let mut a_new = vec![0.0f32; target_rank * d];
    for c in 0..eff_r {
        let root_s = s_core[c].max(0.0).sqrt();
        for i in 0..d {
            let mut acc_b = 0.0f32;
            let mut acc_a = 0.0f32;
            for k in 0..total_r {
                acc_b += qb[i * total_r + k] * u_core[k * eff_r + c];
                acc_a += vt_core[c * total_r + k] * qa[i * total_r + k];
            }
            b_new[i * target_rank + c] = acc_b * root_s;
            a_new[c * d + i] = root_s * acc_a;
        }
    }

    if a_new.iter().chain(&b_new).any(|v| !v.is_finite()) {
        return Err("consolidation produced non-finite matrices".to_string());
    }
    // Set alpha = target_rank so (alpha / target_rank) == 1.0 preserves exact folded delta_W.
    let cfg = LoraConfig {
        hidden_dim: d,
        rank: target_rank,
        alpha: target_rank as f32,
        ..LoraConfig::default()
    };
    Ok(DynamicLoraAdapter {
        cfg,
        a: a_new,
        b: b_new,
        prev_a: None,
        prev_b: None,
        evolution_steps,
        cumulative_reward: adapters.iter().map(|a| a.cumulative_reward).sum(),
        prev_steps: None,
        prev_reward: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn invalid_configuration_and_consolidation_are_bounded() {
        for cfg in [
            LoraConfig {
                rank: 0,
                ..LoraConfig::default()
            },
            LoraConfig {
                rank: usize::MAX,
                ..LoraConfig::default()
            },
            LoraConfig {
                hidden_dim: usize::MAX,
                ..LoraConfig::default()
            },
            LoraConfig {
                alpha: f32::NAN,
                ..LoraConfig::default()
            },
        ] {
            assert!(DynamicLoraAdapter::try_new(cfg).is_err());
            let compatible = DynamicLoraAdapter::new(cfg);
            assert_eq!(compatible.config().rank, LoraConfig::default().rank);
        }
        let adapter = DynamicLoraAdapter::new(LoraConfig::default());
        assert!(consolidate_adapters(&[&adapter], usize::MAX).is_err());
        assert!(consolidate_adapters(&[&adapter; 9], 16).is_err());
    }

    #[test]
    fn invalid_reinforcement_preserves_weights_and_rollback_metadata() {
        let mut adapter = DynamicLoraAdapter::new(LoraConfig {
            hidden_dim: 8,
            rank: 2,
            ..LoraConfig::default()
        });
        let h = [0.5; 8];
        adapter.reinforce_step(&h, &[0.8; 8], 0.7);
        let a = adapter.matrix_a().to_vec();
        let b = adapter.matrix_b().to_vec();
        adapter.reinforce_step(&h, &[0.8; 8], f32::NAN);
        adapter.reinforce_step(&[f32::MAX; 8], &[f32::MAX; 8], 1.0);
        assert_eq!(adapter.matrix_a(), a);
        assert_eq!(adapter.matrix_b(), b);
        assert_eq!(adapter.stats().evolution_steps, 1);
        assert!(
            adapter
                .forward_delta(&[f32::NAN; 8])
                .iter()
                .all(|v| *v == 0.0)
        );
        assert!(adapter.rollback());
        assert_eq!(adapter.stats().evolution_steps, 0);
        assert_eq!(adapter.stats().cumulative_reward, 0.0);
    }

    #[test]
    fn file_loader_checks_dimensions_length_and_finiteness_before_use() -> Result<(), String> {
        let path = std::env::temp_dir().join(format!(
            "inkling-lora-validation-{}.iklora",
            std::process::id()
        ));
        let mut adapter = DynamicLoraAdapter::new(LoraConfig {
            hidden_dim: 8,
            rank: 2,
            ..LoraConfig::default()
        });
        adapter.reinforce_step(&[0.5; 8], &[0.8; 8], 0.7);
        adapter.save_atomic(&path)?;
        let original = fs::read(&path).map_err(|e| e.to_string())?;
        let loaded = DynamicLoraAdapter::load(&path)?;
        assert_eq!(loaded.matrix_a(), adapter.matrix_a());
        assert_eq!(loaded.matrix_b(), adapter.matrix_b());
        let mut malformed = original.clone();
        malformed[8..12].copy_from_slice(&u32::MAX.to_le_bytes());
        fs::write(&path, &malformed).map_err(|e| e.to_string())?;
        assert!(DynamicLoraAdapter::load(&path).is_err());
        fs::write(&path, &original[..original.len() - 1]).map_err(|e| e.to_string())?;
        assert!(DynamicLoraAdapter::load(&path).is_err());
        malformed = original.clone();
        malformed[28..32].copy_from_slice(&f32::NAN.to_le_bytes());
        fs::write(&path, &malformed).map_err(|e| e.to_string())?;
        assert!(DynamicLoraAdapter::load(&path).is_err());
        malformed = original;
        malformed.push(0);
        fs::write(&path, &malformed).map_err(|e| e.to_string())?;
        assert!(DynamicLoraAdapter::load(&path).is_err());
        fs::remove_file(path).map_err(|e| e.to_string())?;
        Ok(())
    }

    #[test]
    fn lora_starts_at_identity_evolves_and_rolls_back() {
        let cfg = LoraConfig {
            hidden_dim: 64,
            rank: 8,
            ..LoraConfig::default()
        };
        let mut lora = DynamicLoraAdapter::new(cfg);
        let h = vec![0.5f32; 64];
        let d0 = lora.forward_delta(&h);
        assert!(d0.iter().all(|&v| v.abs() < 1e-6));

        let target = vec![1.0f32; 64];
        lora.reinforce_step(&h, &target, 0.9);
        assert_eq!(lora.stats().evolution_steps, 1);
        let d1 = lora.forward_delta(&h);
        assert!(d1.iter().map(|v| v * v).sum::<f32>().sqrt() > 1e-4);

        assert!(lora.rollback());
        assert_eq!(lora.stats().evolution_steps, 0);
    }

    #[test]
    fn qr_svd_consolidation_preserves_combined_delta_without_dense_matrix() {
        let cfg = LoraConfig {
            hidden_dim: 32,
            rank: 4,
            alpha: 4.0,
            ..LoraConfig::default()
        };
        let mut a1 = DynamicLoraAdapter::new(cfg);
        let mut a2 = DynamicLoraAdapter::new(cfg);
        let h = vec![0.25f32; 32];
        let t1 = vec![0.8f32; 32];
        let mut t2 = vec![-0.4f32; 32];
        t2[0] = 1.2;
        a1.reinforce_step(&h, &t1, 0.9);
        a2.reinforce_step(&h, &t2, 0.8);

        let d1 = a1.forward_delta(&h);
        let d2 = a2.forward_delta(&h);
        let expected: Vec<f32> = d1.iter().zip(d2.iter()).map(|(x, y)| x + y).collect();

        let folded = consolidate_adapters(&[&a1, &a2], 4);
        assert!(folded.is_ok());
        if let Ok(f) = folded {
            let actual = f.forward_delta(&h);
            let err: f32 = expected
                .iter()
                .zip(actual.iter())
                .map(|(x, y)| (x - y).powi(2))
                .sum::<f32>()
                .sqrt();
            assert!(err < 1e-3, "folded delta diverged: err={err}");
        }
    }

    #[test]
    fn anti_forgetting_gate_rejects_security_or_capability_regression() {
        let base = BaselineEvalScores {
            math: 0.80,
            coding: 0.78,
            language_ja_en: 0.85,
            long_context: 0.75,
            tool_calling: 0.90,
            instruction_following: 0.88,
            privacy_security: 1.00,
        };
        let mut degraded = base;
        degraded.privacy_security = 0.99;
        assert!(approve_candidate_adapter(base, degraded, 0.02).is_err());

        let mut invalid = base;
        invalid.math = f32::NAN;
        assert!(approve_candidate_adapter(base, invalid, 0.02).is_err());
        invalid.math = f32::INFINITY;
        assert!(approve_candidate_adapter(base, invalid, 0.02).is_err());
        assert!(approve_candidate_adapter(base, base, f32::NAN).is_err());

        let mut improved = base;
        improved.coding = 0.82;
        assert!(approve_candidate_adapter(base, improved, 0.02).is_ok());
    }
}
