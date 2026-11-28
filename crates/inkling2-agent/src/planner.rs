//! Deterministic task planner translating goals into typed skill steps.

use crate::skill_registry::SkillRegistry;

/// One planned typed tool invocation.
#[derive(Debug, Clone)]
pub struct PlanStep {
    pub step_index: usize,
    pub skill_name: String,
    pub argument: String,
}

/// Ordered task execution plan bounded by maximum step count.
#[derive(Debug, Clone)]
pub struct TaskPlan {
    pub task_id: String,
    pub goal: String,
    pub steps: Vec<PlanStep>,
}

/// Build a validated `TaskPlan` from explicit typed step tuples `(skill_name, argument)`.
pub fn build_plan(
    task_id: impl Into<String>,
    goal: impl Into<String>,
    requested_steps: &[(String, String)],
    registry: &SkillRegistry,
    max_steps: usize,
) -> Result<TaskPlan, String> {
    if requested_steps.is_empty() {
        return Err("plan must contain at least one typed step".to_string());
    }
    if requested_steps.len() > max_steps {
        return Err(format!(
            "plan exceeds max_steps ({} > {max_steps})",
            requested_steps.len()
        ));
    }
    let mut steps = Vec::with_capacity(requested_steps.len());
    for (idx, (skill, arg)) in requested_steps.iter().enumerate() {
        if registry.get(skill).is_none() {
            return Err(format!("unregistered skill '{skill}' rejected by planner"));
        }
        steps.push(PlanStep {
            step_index: idx,
            skill_name: skill.clone(),
            argument: arg.clone(),
        });
    }
    Ok(TaskPlan {
        task_id: task_id.into(),
        goal: goal.into(),
        steps,
    })
}
