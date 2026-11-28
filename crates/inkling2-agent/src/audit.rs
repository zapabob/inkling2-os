//! Structured execution audit trail for autonomous agent tasks.

use std::time::{SystemTime, UNIX_EPOCH};

/// Single audit record for a task state transition or tool invocation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AgentAuditRecord {
    pub timestamp_sec: u64,
    pub task_id: String,
    pub phase: String,
    pub action: String,
    pub outcome: String,
}

/// Bounded in-memory audit log for agent execution.
#[derive(Debug, Clone)]
pub struct AgentAuditLog {
    records: Vec<AgentAuditRecord>,
    capacity: usize,
}

impl Default for AgentAuditLog {
    fn default() -> Self {
        Self::new(1024)
    }
}

impl AgentAuditLog {
    pub fn new(capacity: usize) -> Self {
        Self {
            records: Vec::new(),
            capacity: capacity.max(32),
        }
    }

    fn now_sec() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0)
    }

    pub fn record(
        &mut self,
        task_id: impl Into<String>,
        phase: impl Into<String>,
        action: impl Into<String>,
        outcome: impl Into<String>,
    ) {
        if self.records.len() >= self.capacity {
            self.records.remove(0);
        }
        self.records.push(AgentAuditRecord {
            timestamp_sec: Self::now_sec(),
            task_id: task_id.into(),
            phase: phase.into(),
            action: action.into(),
            outcome: outcome.into(),
        });
    }

    pub fn entries(&self) -> &[AgentAuditRecord] {
        &self.records
    }
}
