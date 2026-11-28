//! Verified Self-Reinforcement Reward Model & Learning Levels (Neon Phase 5).
//!
//! Computes normalised rewards in `[-1.0, 1.0]` strictly from verified external
//! outcomes and operator feedback. Any policy or security violation disqualifies
//! the candidate regardless of other positive signals.

/// Progressive learning tiers (Phase 5.2).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum LearningLevel {
    /// Level 1: Episodic memory importance and procedure recording.
    Level1Episodic = 1,
    /// Level 2: Procedural tool selection and skill ranking.
    Level2Procedural = 2,
    /// Level 3: Neural fast-weight associative memory updates.
    Level3NeuralMemory = 3,
    /// Level 4: SC117 Dynamic LoRA low-rank model adaptation (experimental).
    Level4ModelAdaptation = 4,
}

/// Raw normalised signal inputs for reward calculation (each component in `[0.0, 1.0]`).
#[derive(Debug, Clone, Copy)]
pub struct RewardSignals {
    /// Ground-truth verified task success (`0.0` or `1.0`). Self-claims must pass `0.0`.
    pub verified_task_success: f32,
    /// Explicit user feedback in `[-1.0, 1.0]`.
    pub user_feedback: f32,
    /// Measured output quality in `[0.0, 1.0]`.
    pub output_quality: f32,
    /// Execution error penalty in `[0.0, 1.0]`.
    pub execution_errors: f32,
    /// Resource cost penalty (VRAM/latency) in `[0.0, 1.0]`.
    pub resource_cost: f32,
    /// Unnecessary action penalty in `[0.0, 1.0]`.
    pub unnecessary_actions: f32,
    /// Number of policy or security violations (any value > 0 disqualifies candidate).
    pub policy_violations: u32,
}

/// Result of evaluating a candidate experience for reinforcement.
#[derive(Debug, Clone, PartialEq)]
pub struct ReinforcementVerdict {
    pub accepted: bool,
    pub reward: f32,
    pub max_allowed_level: LearningLevel,
    pub reason: &'static str,
}

fn clean_unit(v: f32) -> f32 {
    if v.is_finite() {
        v.clamp(0.0, 1.0)
    } else {
        0.0
    }
}

fn clean_signed(v: f32) -> f32 {
    if v.is_finite() {
        v.clamp(-1.0, 1.0)
    } else {
        0.0
    }
}

/// Compute bounded reward in `[-1.0, 1.0]` and decide whether learning may proceed.
pub fn evaluate_reinforcement(
    signals: RewardSignals,
    requested_level: LearningLevel,
    source_trusted: bool,
) -> ReinforcementVerdict {
    if signals.policy_violations > 0 {
        return ReinforcementVerdict {
            accepted: false,
            reward: -1.0,
            max_allowed_level: LearningLevel::Level1Episodic,
            reason: "policy or security violation unconditionally disqualifies candidate",
        };
    }
    if !source_trusted {
        return ReinforcementVerdict {
            accepted: false,
            reward: 0.0,
            max_allowed_level: LearningLevel::Level1Episodic,
            reason: "untrusted source is excluded from reinforcement learning",
        };
    }

    let pos = 0.45 * clean_unit(signals.verified_task_success)
        + 0.35 * clean_signed(signals.user_feedback)
        + 0.20 * clean_unit(signals.output_quality);
    let neg = 0.45 * clean_unit(signals.execution_errors)
        + 0.20 * clean_unit(signals.resource_cost)
        + 0.20 * clean_unit(signals.unnecessary_actions);
    let reward = (pos - neg).clamp(-1.0, 1.0);

    // Require verified success or explicit positive user feedback for Levels 3–4.
    let has_external_verification = clean_unit(signals.verified_task_success) > 0.0
        || clean_signed(signals.user_feedback) > 0.0;
    if requested_level >= LearningLevel::Level3NeuralMemory && !has_external_verification {
        return ReinforcementVerdict {
            accepted: false,
            reward,
            max_allowed_level: LearningLevel::Level2Procedural,
            reason: "neural memory and LoRA adaptation require external verification",
        };
    }

    ReinforcementVerdict {
        accepted: true,
        reward,
        max_allowed_level: requested_level,
        reason: "verified reinforcement signal accepted",
    }
}
