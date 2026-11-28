//! Task scheduler with task-ID deduplication (prevents duplicate Hermes/Inkling2 execution).

use std::collections::{HashMap, HashSet};

/// Lifecycle state of a scheduled autonomous task.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TaskStatus {
    Queued,
    Running,
    Completed,
    Cancelled,
    AwaitingApproval,
    Failed(String),
}

/// Summary metadata for a tracked task.
#[derive(Debug, Clone)]
pub struct TaskRecord {
    pub task_id: String,
    pub goal: String,
    pub owner: String,
    pub status: TaskStatus,
    pub tool_calls_used: usize,
}

/// Deduplicating task registry and cancellation switchboard.
#[derive(Debug, Default)]
pub struct TaskScheduler {
    tasks: HashMap<String, TaskRecord>,
    seen_ids: HashSet<String>,
    global_stop: bool,
}

impl TaskScheduler {
    /// Submit a task with explicit `task_id` and `owner` (`inkling2` or `hermes`).
    /// Rejects duplicate task IDs and rejects new tasks when global stop is active.
    pub fn submit(
        &mut self,
        task_id: impl Into<String>,
        goal: impl Into<String>,
        owner: impl Into<String>,
    ) -> Result<&TaskRecord, String> {
        if self.global_stop {
            return Err("agent scheduler is stopped; no new tasks accepted".to_string());
        }
        let id = task_id.into();
        if id.trim().is_empty() || id.len() > 128 {
            return Err("invalid task_id".to_string());
        }
        let goal = goal.into();
        let owner = owner.into();
        if goal.len() > 8192 || owner.len() > 128 {
            return Err("task goal or owner exceeds its size limit".to_string());
        }
        if self.tasks.len() >= 4096 {
            return Err(
                "task registry is full; restart after retaining required audit records".to_string(),
            );
        }
        if !self.seen_ids.insert(id.clone()) {
            return Err(format!(
                "duplicate task_id '{id}' rejected to prevent double execution"
            ));
        }
        self.tasks.insert(
            id.clone(),
            TaskRecord {
                task_id: id.clone(),
                goal,
                owner,
                status: TaskStatus::Queued,
                tool_calls_used: 0,
            },
        );
        self.tasks
            .get(&id)
            .ok_or_else(|| "internal scheduler insertion error".to_string())
    }

    pub fn set_status(&mut self, task_id: &str, status: TaskStatus) {
        if let Some(t) = self.tasks.get_mut(task_id) {
            t.status = status;
        }
    }

    pub fn cancel_task(&mut self, task_id: &str) -> bool {
        if let Some(t) = self.tasks.get_mut(task_id) {
            t.status = TaskStatus::Cancelled;
            return true;
        }
        false
    }

    /// Issue a global stop order: cancels queued/running tasks and blocks new submissions.
    pub fn stop_all(&mut self) {
        self.global_stop = true;
        for t in self.tasks.values_mut() {
            if matches!(
                t.status,
                TaskStatus::Queued | TaskStatus::Running | TaskStatus::AwaitingApproval
            ) {
                t.status = TaskStatus::Cancelled;
            }
        }
    }

    pub fn resume(&mut self) {
        self.global_stop = false;
    }

    pub fn is_stopped(&self) -> bool {
        self.global_stop
    }

    pub fn list(&self) -> Vec<&TaskRecord> {
        let mut out: Vec<&TaskRecord> = self.tasks.values().collect();
        out.sort_by(|a, b| a.task_id.cmp(&b.task_id));
        out
    }

    pub fn get(&self, task_id: &str) -> Option<&TaskRecord> {
        self.tasks.get(task_id)
    }
}
