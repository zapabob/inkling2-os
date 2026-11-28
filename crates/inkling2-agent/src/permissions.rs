//! Immutable Permission Broker bridge for the Autonomous Agent.
//!
//! Learning algorithms, reward signals, and memory updates have NO mutable access
//! to policy rules.

pub use inkling2_companion::permission_broker::{
    ActionRequest, ActionRisk, ApprovalToken, AuditEntry, PermissionBroker, PolicyDecision,
};
