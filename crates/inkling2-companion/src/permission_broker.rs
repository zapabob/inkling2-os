//! Immutable OS Permission Broker (`permission_broker`).
//!
//! Enforces Neon Phase 3.3 action risk classification and user-approval gates.
//! By design, the policy table is immutable: learning algorithms, reward signals,
//! memory contents, and model outputs cannot modify or bypass permission rules.

use std::time::{SystemTime, UNIX_EPOCH};

/// Four-tier risk classification for OS and agent actions (Phase 3.3).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActionRisk {
    /// Pure inspection (file read in allowed dir, system telemetry, status check).
    ReadOnly,
    /// Low-risk local action that can be undone (open app, create note in workspace).
    ReversibleLocal,
    /// Outbound network post, email/message send, API mutation, external purchase.
    ExternalSideEffect,
    /// Deletion, disk/partition write, privilege escalation, software install, raw shell.
    DestructiveOrPrivileged,
}

/// Evaluation verdict emitted by the Permission Broker.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PolicyDecision {
    /// Action may proceed automatically under the active policy.
    Allowed,
    /// Action is blocked until an explicit user approval token is presented.
    RequiresApproval { reason: &'static str },
    /// Action is unconditionally forbidden (e.g., raw unvalidated shell execution).
    Denied { reason: &'static str },
}

/// Explicit user approval token bound to a specific action name and task ID.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ApprovalToken {
    /// Task identifier authorized by the user.
    pub task_id: String,
    /// Typed tool/action identifier authorized by the user.
    pub action_name: String,
    /// Non-empty operator confirmation string or signed nonce.
    pub operator_grant: String,
}

/// Typed action request submitted to the Permission Broker.
#[derive(Debug, Clone)]
pub struct ActionRequest {
    /// Unique task identifier for deduplication and audit correlation.
    pub task_id: String,
    /// Typed tool name (e.g. `fs.read`, `voice.speak`, `os.launch_app`).
    pub action_name: String,
    /// Risk tier of the action.
    pub risk: ActionRisk,
    /// True if the caller attempts arbitrary raw shell execution (always denied).
    pub is_raw_shell_command: bool,
}

/// Structured audit record for every permission check.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AuditEntry {
    pub timestamp_sec: u64,
    pub task_id: String,
    pub action_name: String,
    pub risk: ActionRisk,
    pub decision: PolicyDecision,
}

/// Immutable Permission Broker with bounded in-memory audit ring.
#[derive(Debug, Clone)]
pub struct PermissionBroker {
    allow_reversible_auto: bool,
    audit_log: Vec<AuditEntry>,
    max_audit_entries: usize,
}

impl Default for PermissionBroker {
    fn default() -> Self {
        Self::new(true)
    }
}

impl PermissionBroker {
    /// Construct a broker. `allow_reversible_auto` governs `ReversibleLocal` actions only;
    /// `ExternalSideEffect` and `DestructiveOrPrivileged` ALWAYS require explicit approval.
    pub fn new(allow_reversible_auto: bool) -> Self {
        Self {
            allow_reversible_auto,
            audit_log: Vec::new(),
            max_audit_entries: 1024,
        }
    }

    fn now_sec() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0)
    }

    fn valid_action_name(name: &str) -> bool {
        (1..=64).contains(&name.len())
            && name
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'.' | b'_' | b'-'))
    }

    fn token_matches(req: &ActionRequest, token: Option<&ApprovalToken>) -> bool {
        let Some(tok) = token else {
            return false;
        };
        !tok.operator_grant.trim().is_empty()
            && tok.task_id == req.task_id
            && tok.action_name == req.action_name
    }

    /// Evaluate an action request and append an immutable audit record.
    pub fn evaluate(
        &mut self,
        req: &ActionRequest,
        approval: Option<&ApprovalToken>,
    ) -> PolicyDecision {
        let decision = if req.is_raw_shell_command {
            PolicyDecision::Denied {
                reason: "arbitrary shell execution is prohibited; use typed tools",
            }
        } else if !Self::valid_action_name(&req.action_name) || req.task_id.trim().is_empty() {
            PolicyDecision::Denied {
                reason: "invalid action_name or empty task_id",
            }
        } else {
            match req.risk {
                ActionRisk::ReadOnly => PolicyDecision::Allowed,
                ActionRisk::ReversibleLocal => {
                    if self.allow_reversible_auto || Self::token_matches(req, approval) {
                        PolicyDecision::Allowed
                    } else {
                        PolicyDecision::RequiresApproval {
                            reason: "reversible local action requires operator approval",
                        }
                    }
                }
                ActionRisk::ExternalSideEffect => {
                    if Self::token_matches(req, approval) {
                        PolicyDecision::Allowed
                    } else {
                        PolicyDecision::RequiresApproval {
                            reason: "external side-effect requires explicit user approval",
                        }
                    }
                }
                ActionRisk::DestructiveOrPrivileged => {
                    if Self::token_matches(req, approval) {
                        PolicyDecision::Allowed
                    } else {
                        PolicyDecision::RequiresApproval {
                            reason: "destructive or privileged action requires explicit user approval",
                        }
                    }
                }
            }
        };

        if self.audit_log.len() >= self.max_audit_entries {
            self.audit_log.remove(0);
        }
        self.audit_log.push(AuditEntry {
            timestamp_sec: Self::now_sec(),
            task_id: req.task_id.clone(),
            action_name: req.action_name.clone(),
            risk: req.risk,
            decision: decision.clone(),
        });
        decision
    }

    /// Inspect recorded audit entries.
    pub fn audit_entries(&self) -> &[AuditEntry] {
        &self.audit_log
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn raw_shell_is_always_denied_even_with_approval_token() {
        let mut broker = PermissionBroker::default();
        let req = ActionRequest {
            task_id: "task-1".to_string(),
            action_name: "shell.exec".to_string(),
            risk: ActionRisk::ReadOnly,
            is_raw_shell_command: true,
        };
        let tok = ApprovalToken {
            task_id: "task-1".to_string(),
            action_name: "shell.exec".to_string(),
            operator_grant: "approved".to_string(),
        };
        assert!(matches!(
            broker.evaluate(&req, Some(&tok)),
            PolicyDecision::Denied { .. }
        ));
    }

    #[test]
    fn high_risk_actions_require_matching_user_approval() {
        let mut broker = PermissionBroker::default();
        let req = ActionRequest {
            task_id: "task-42".to_string(),
            action_name: "fs.delete_file".to_string(),
            risk: ActionRisk::DestructiveOrPrivileged,
            is_raw_shell_command: false,
        };
        assert!(matches!(
            broker.evaluate(&req, None),
            PolicyDecision::RequiresApproval { .. }
        ));

        let wrong_tok = ApprovalToken {
            task_id: "task-99".to_string(),
            action_name: "fs.delete_file".to_string(),
            operator_grant: "yes".to_string(),
        };
        assert!(matches!(
            broker.evaluate(&req, Some(&wrong_tok)),
            PolicyDecision::RequiresApproval { .. }
        ));

        let good_tok = ApprovalToken {
            task_id: "task-42".to_string(),
            action_name: "fs.delete_file".to_string(),
            operator_grant: "operator-confirmed".to_string(),
        };
        assert_eq!(
            broker.evaluate(&req, Some(&good_tok)),
            PolicyDecision::Allowed
        );
        assert_eq!(broker.audit_entries().len(), 3);
    }

    #[test]
    fn readonly_and_reversible_local_follow_policy() {
        let mut broker = PermissionBroker::new(true);
        let ro = ActionRequest {
            task_id: "t1".to_string(),
            action_name: "gpu.info".to_string(),
            risk: ActionRisk::ReadOnly,
            is_raw_shell_command: false,
        };
        let rev = ActionRequest {
            task_id: "t2".to_string(),
            action_name: "voice.speak".to_string(),
            risk: ActionRisk::ReversibleLocal,
            is_raw_shell_command: false,
        };
        assert_eq!(broker.evaluate(&ro, None), PolicyDecision::Allowed);
        assert_eq!(broker.evaluate(&rev, None), PolicyDecision::Allowed);
    }
}
