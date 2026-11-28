//! Inkling2-OS Autonomous Agent Runtime & Verified Self-Reinforcement (`inkling2-agent`).
//!
//! Implements Neon Phase 3 (Observe -> Plan -> PolicyCheck -> Execute -> Verify -> Record -> Learn -> Idle)
//! and Phase 5 (Verified Self-Reinforcement with immutable Permission Broker enforcement).

pub mod audit;
pub mod executor;
pub mod permissions;
pub mod planner;
pub mod reinforcement;
pub mod scheduler;
pub mod skill_registry;
pub mod verifier;
