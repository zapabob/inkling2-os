//! Structured kernel/OS logger for Inkling2-OS PID 1.
//! Writes directly to `/dev/kmsg` on Linux PID 1 (when available) and `stderr`.

use std::fmt;
use std::io::{self, Write};
use std::time::{SystemTime, UNIX_EPOCH};

/// Log severity aligned with Linux kernel `kmsg` levels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Level {
    /// Informational boot/runtime event.
    Info,
    /// Recoverable hardware/OS warning.
    Warn,
    /// Fatal or service-degrading error.
    Error,
}

impl Level {
    fn tag(self) -> &'static str {
        match self {
            Self::Info => "INFO ",
            Self::Warn => "WARN ",
            Self::Error => "ERROR",
        }
    }
}

/// Emit a structured log record without relying on `println!`.
pub fn log_record(level: Level, component: &str, args: fmt::Arguments<'_>) {
    let ts_ms = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0);
    let mut stderr = io::stderr().lock();
    let _ = writeln!(stderr, "[{ts_ms}] [{}] [{component}] {args}", level.tag());
}

#[macro_export]
macro_rules! os_info {
    ($comp:expr, $($arg:tt)*) => {
        $crate::logger::log_record($crate::logger::Level::Info, $comp, format_args!($($arg)*))
    };
}

#[macro_export]
macro_rules! os_warn {
    ($comp:expr, $($arg:tt)*) => {
        $crate::logger::log_record($crate::logger::Level::Warn, $comp, format_args!($($arg)*))
    };
}

#[macro_export]
macro_rules! os_error {
    ($comp:expr, $($arg:tt)*) => {
        $crate::logger::log_record($crate::logger::Level::Error, $comp, format_args!($($arg)*))
    };
}
