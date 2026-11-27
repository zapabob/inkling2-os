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
}

impl DynamicLoraAdapter {
    /// Initialize a new identity-preserving dynamic LoRA adapter (`B = 0`, deterministic Kaiming `A`).
    pub fn new(cfg: LoraConfig) -> Self {
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
        }
    }

    /// Compute `Δh = clamp((alpha / rank) * B * (A * h), max_delta_norm)`.
    pub fn forward_delta(&self, h: &[f32]) -> Vec<f32> {
        let d = self.cfg.hidden_dim;
        let r = self.cfg.rank;
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
        let mut sq_norm = 0.0f32;
        for (di, update) in delta.iter_mut().enumerate() {
            let row = &self.b[di * r..(di + 1) * r];
            let mut acc = 0.0f32;
            for ri in 0..r {
                acc += row[ri] * z[ri];
            }
            let val = acc * scale;
            *update = val;
            sq_norm += val * val;
        }

        let norm = sq_norm.sqrt();
        if norm > self.cfg.max_delta_norm && norm > 1e-8 {
            let c = self.cfg.max_delta_norm / norm;
            for v in &mut delta {
                *v *= c;
            }
        }
        delta
    }

    /// Perform one bounded online self-reinforcement LoRA update from `(h, target_dir, reward)`.
    pub fn reinforce_step(&mut self, h: &[f32], target_dir: &[f32], reward: f32) {
        let r_clamped = reward.clamp(-1.0, 1.0);
        if r_clamped.abs() < 1e-6 {
            return;
        }
        // Snapshot before mutation for instant rollback
        self.prev_a = Some(self.a.clone());
        self.prev_b = Some(self.b.clone());

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
            let row = &mut self.b[di * r..(di + 1) * r];
            for ri in 0..r {
                row[ri] = row[ri] * wd + lr * td * z[ri];
            }
        }

        // Update A += lr * g_z * h^T
        for (ri, &g) in gz.iter().enumerate() {
            let row = &mut self.a[ri * d..(ri + 1) * d];
            for (wv, &hv) in row.iter_mut().zip(h.iter().take(d)) {
                *wv = *wv * wd + lr * g * hv;
            }
        }

        self.evolution_steps += 1;
        self.cumulative_reward += f64::from(r_clamped);
    }

    /// Roll back to the previous weight snapshot if a self-reinforcement step degraded quality.
    pub fn rollback(&mut self) -> bool {
        if let (Some(pa), Some(pb)) = (self.prev_a.take(), self.prev_b.take()) {
            self.a = pa;
            self.b = pb;
            self.evolution_steps = self.evolution_steps.saturating_sub(1);
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

        let mut a = vec![0.0f32; rank * hidden_dim];
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
        Ok(Self {
            cfg: LoraConfig {
                hidden_dim,
                rank,
                alpha,
                ..LoraConfig::default()
            },
            a,
            b,
            prev_a: None,
            prev_b: None,
            evolution_steps,
            cumulative_reward: 0.0,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
