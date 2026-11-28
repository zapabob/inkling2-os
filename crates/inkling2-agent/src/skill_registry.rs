//! Typed OS Skill Registry (prohibits arbitrary raw PowerShell/Bash execution).

use std::collections::HashMap;

use crate::permissions::ActionRisk;

/// Typed OS capability descriptor registered with the agent.
#[derive(Debug, Clone)]
pub struct SkillSpec {
    pub name: String,
    pub description: String,
    pub risk: ActionRisk,
    pub reversible: bool,
}

/// Registry of allowed typed OS skills.
#[derive(Debug, Clone, Default)]
pub struct SkillRegistry {
    skills: HashMap<String, SkillSpec>,
}

impl SkillRegistry {
    /// Construct a registry pre-populated with safe built-in typed OS tools.
    pub fn with_defaults() -> Self {
        let mut reg = Self::default();
        let defaults = [
            SkillSpec {
                name: "sys.info".to_string(),
                description: "Read OS platform and resource telemetry".to_string(),
                risk: ActionRisk::ReadOnly,
                reversible: true,
            },
            SkillSpec {
                name: "fs.list_allowed".to_string(),
                description: "List entries within an operator-approved directory".to_string(),
                risk: ActionRisk::ReadOnly,
                reversible: true,
            },
            SkillSpec {
                name: "perception.status".to_string(),
                description: "Query camera and privacy gate status".to_string(),
                risk: ActionRisk::ReadOnly,
                reversible: true,
            },
            SkillSpec {
                name: "voice.speak".to_string(),
                description: "Speak final user-facing response via local TTS".to_string(),
                risk: ActionRisk::ReversibleLocal,
                reversible: true,
            },
            SkillSpec {
                name: "os.notify".to_string(),
                description: "Emit a local desktop notification".to_string(),
                risk: ActionRisk::ReversibleLocal,
                reversible: true,
            },
            SkillSpec {
                name: "net.send_external".to_string(),
                description: "Send data to an external endpoint (requires explicit approval)"
                    .to_string(),
                risk: ActionRisk::ExternalSideEffect,
                reversible: false,
            },
            SkillSpec {
                name: "fs.delete_path".to_string(),
                description: "Delete a file path (destructive; requires explicit approval)"
                    .to_string(),
                risk: ActionRisk::DestructiveOrPrivileged,
                reversible: false,
            },
        ];
        for s in defaults {
            reg.register(s);
        }
        reg
    }

    pub fn register(&mut self, spec: SkillSpec) {
        self.skills.insert(spec.name.clone(), spec);
    }

    pub fn get(&self, name: &str) -> Option<&SkillSpec> {
        self.skills.get(name)
    }

    pub fn list(&self) -> Vec<&SkillSpec> {
        let mut out: Vec<&SkillSpec> = self.skills.values().collect();
        out.sort_by(|a, b| a.name.cmp(&b.name));
        out
    }
}
