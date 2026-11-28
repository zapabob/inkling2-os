//! Unified Multimodal Perception Event Bus (`event_bus`).

use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use crate::privacy::{RetentionPolicy, Sensitivity};

/// Source modality of a perception event.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Modality {
    Camera,
    Ocr,
    Audio,
    Desktop,
    System,
}

impl Modality {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Camera => "camera",
            Self::Ocr => "ocr",
            Self::Audio => "audio",
            Self::Desktop => "desktop",
            Self::System => "system",
        }
    }
}

/// Unified perception event structure (Neon Phase 1.2).
#[derive(Debug, Clone)]
pub struct PerceptionEvent {
    pub event_id: u64,
    pub timestamp_sec: u64,
    pub source: String,
    pub modality: Modality,
    pub observation: String,
    pub confidence: f32,
    pub sensitivity: Sensitivity,
    pub retention_policy: RetentionPolicy,
    pub provenance: String,
    /// True when the observation originates from untrusted external pixels/text/web.
    pub untrusted_content: bool,
}

/// Bounded in-memory ring bus for multimodal perception events.
#[derive(Debug)]
pub struct PerceptionEventBus {
    next_id: AtomicU64,
    capacity: usize,
    events: VecDeque<PerceptionEvent>,
}

impl Default for PerceptionEventBus {
    fn default() -> Self {
        Self::new(256)
    }
}

impl PerceptionEventBus {
    pub fn new(capacity: usize) -> Self {
        Self {
            next_id: AtomicU64::new(1),
            capacity: capacity.max(16),
            events: VecDeque::with_capacity(capacity.max(16)),
        }
    }

    fn now_sec() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0)
    }

    /// Publish a new perception event onto the bounded ring bus.
    #[allow(clippy::too_many_arguments)]
    pub fn publish(
        &mut self,
        source: impl Into<String>,
        modality: Modality,
        observation: impl Into<String>,
        confidence: f32,
        sensitivity: Sensitivity,
        retention_policy: RetentionPolicy,
        provenance: impl Into<String>,
        untrusted_content: bool,
    ) -> PerceptionEvent {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let ev = PerceptionEvent {
            event_id: id,
            timestamp_sec: Self::now_sec(),
            source: source.into(),
            modality,
            observation: observation.into(),
            confidence: if confidence.is_finite() {
                confidence.clamp(0.0, 1.0)
            } else {
                0.0
            },
            sensitivity,
            retention_policy,
            provenance: provenance.into(),
            untrusted_content,
        };
        if self.events.len() >= self.capacity {
            self.events.pop_front();
        }
        self.events.push_back(ev.clone());
        ev
    }

    /// Return the most recent `limit` events in reverse chronological order.
    pub fn recent(&self, limit: usize) -> Vec<PerceptionEvent> {
        self.events.iter().rev().take(limit).cloned().collect()
    }

    pub fn len(&self) -> usize {
        self.events.len()
    }

    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }

    /// Drop all ephemeral working-memory events (e.g., on privacy stop).
    pub fn purge_ephemeral(&mut self) {
        self.events
            .retain(|e| e.retention_policy != RetentionPolicy::Ephemeral);
    }
}
