//! Ground-truth external verifier (rejects self-reported LLM success claims).

use std::path::Path;

/// External verification evidence source (Phase 5.1).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VerificationEvidence {
    /// Confirmed file existence on disk.
    FileExists(String),
    /// Confirmed non-error return code from a typed tool execution.
    ToolExitZero { tool: String, detail: String },
    /// Confirmed automated test suite pass.
    TestPassed { suite: String, passed: u32 },
    /// Explicit positive evaluation provided by the human operator.
    UserConfirmed,
    /// Self-reported claim by the LLM without external proof (never accepted as ground truth).
    SelfClaimOnly,
}

/// Verification outcome for a completed action or task.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerificationResult {
    pub verified: bool,
    pub reason: String,
}

/// Verify ground-truth evidence. Self-claims are unconditionally rejected.
pub fn verify_evidence(evidence: &VerificationEvidence) -> VerificationResult {
    match evidence {
        VerificationEvidence::FileExists(path) => {
            if Path::new(path).exists() {
                VerificationResult {
                    verified: true,
                    reason: format!("verified file exists: {path}"),
                }
            } else {
                VerificationResult {
                    verified: false,
                    reason: format!("expected file missing: {path}"),
                }
            }
        }
        VerificationEvidence::ToolExitZero { tool, detail } => VerificationResult {
            verified: true,
            reason: format!("typed tool '{tool}' succeeded: {detail}"),
        },
        VerificationEvidence::TestPassed { suite, passed } => {
            if *passed > 0 {
                VerificationResult {
                    verified: true,
                    reason: format!("test suite '{suite}' passed ({passed} checks)"),
                }
            } else {
                VerificationResult {
                    verified: false,
                    reason: format!("test suite '{suite}' ran 0 checks"),
                }
            }
        }
        VerificationEvidence::UserConfirmed => VerificationResult {
            verified: true,
            reason: "explicit operator confirmation received".to_string(),
        },
        VerificationEvidence::SelfClaimOnly => VerificationResult {
            verified: false,
            reason: "self-reported success is never accepted for learning rewards".to_string(),
        },
    }
}
