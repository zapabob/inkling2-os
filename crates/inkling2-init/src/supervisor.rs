//! Linux PID 1 service supervision. The bridge and all of its children run unprivileged.

use std::fs;
use std::io::{self, Read, Write};
use std::net::{SocketAddr, TcpStream};
use std::os::unix::process::CommandExt;
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering::Relaxed};
use std::thread::sleep;
use std::time::{Duration, Instant};

use crate::{os_error, os_info, os_warn};

const SERVICE_UID: u32 = 61000;
const PORT: u16 = 8080;
const START_TIMEOUT: Duration = Duration::from_secs(15);
const SHUTDOWN_TIMEOUT: Duration = Duration::from_secs(5);
const MAX_ATTEMPTS: usize = 5;
static STOP: AtomicBool = AtomicBool::new(false);

extern "C" fn request_stop(_sig: libc::c_int) {
    STOP.store(true, Relaxed);
}

fn register_signals() -> io::Result<()> {
    // Each handler only touches a lock-free atomic. PID 1 needs explicit handlers
    // for SIGTERM and SIGINT, which the kernel otherwise ignores for it.
    unsafe {
        for signal in [libc::SIGTERM, libc::SIGINT] {
            let mut action: libc::sigaction = std::mem::zeroed();
            action.sa_sigaction = request_stop as *const () as usize;
            libc::sigemptyset(&mut action.sa_mask);
            if libc::sigaction(signal, &action, std::ptr::null_mut()) != 0 {
                return Err(io::Error::last_os_error());
            }
        }
    }
    Ok(())
}

fn reap(target: Option<libc::pid_t>) -> bool {
    let mut done = false;
    loop {
        // waitpid with WNOHANG also collects orphaned descendants adopted by PID 1.
        let pid = unsafe { libc::waitpid(-1, std::ptr::null_mut(), libc::WNOHANG) };
        if pid <= 0 {
            return done;
        }
        done |= target == Some(pid);
    }
}

fn signal_group(pid: libc::pid_t, signal: libc::c_int) {
    // Each service starts in its own process group. A negative PID addresses that group.
    let _ = unsafe { libc::kill(-pid, signal) };
}

fn stop(pid: libc::pid_t) {
    signal_group(pid, libc::SIGTERM);
    let deadline = Instant::now() + SHUTDOWN_TIMEOUT;
    while Instant::now() < deadline {
        if reap(Some(pid)) {
            break;
        }
        sleep(Duration::from_millis(50));
    }
    // Close out descendants even if the group leader has already exited.
    signal_group(pid, libc::SIGKILL);
    let deadline = Instant::now() + Duration::from_secs(1);
    while Instant::now() < deadline {
        if reap(Some(pid)) {
            break;
        }
        sleep(Duration::from_millis(50));
    }
}

fn healthy() -> bool {
    let addr = SocketAddr::from(([127, 0, 0, 1], PORT));
    let timeout = Duration::from_millis(250);
    let Ok(mut stream) = TcpStream::connect_timeout(&addr, timeout) else {
        return false;
    };
    let _ = stream.set_read_timeout(Some(timeout));
    let _ = stream.set_write_timeout(Some(timeout));
    let request = format!("GET /health HTTP/1.1\r\nHost: {addr}\r\nConnection: close\r\n\r\n");
    let mut reply = String::new();
    stream.write_all(request.as_bytes()).is_ok()
        && stream.take(1024).read_to_string(&mut reply).is_ok()
        && reply.starts_with("HTTP/1.1 200 OK")
        && reply.contains("inkling2-os-bridge")
}

fn spawn() -> io::Result<libc::pid_t> {
    let mut service = Command::new("/usr/bin/inkling2-shell");
    service
        .env_clear()
        .env("PATH", "/usr/sbin:/usr/bin:/sbin:/bin")
        .env("INKLING2_STATE_DIR", "/run/inkling2")
        .env("INKLING2_MODELS_DIR", "/models")
        .arg("serve")
        .arg(format!("127.0.0.1:{PORT}"))
        .stdin(Stdio::null())
        .uid(SERVICE_UID)
        .gid(SERVICE_UID)
        .process_group(0);
    for (file, variable) in [
        ("console.token", "INKLING2_BRIDGE_TOKEN"),
        ("inference.token", "INKLING2_INFERENCE_TOKEN"),
    ] {
        if let Ok(token) = fs::read_to_string(format!("/etc/inkling2/{file}")) {
            service.env(variable, token.trim());
        }
    }
    let child = service.spawn()?;
    i32::try_from(child.id()).map_err(io::Error::other)
}

fn state_directory() -> io::Result<()> {
    use std::os::unix::fs::{PermissionsExt, chown};
    let dir = "/run/inkling2";
    fs::create_dir(dir)?;
    fs::set_permissions(dir, fs::Permissions::from_mode(0o700))?;
    chown(dir, Some(SERVICE_UID), Some(SERVICE_UID))
}

fn requested_smoke() -> bool {
    fs::read_to_string("/proc/cmdline")
        .is_ok_and(|line| line.split_whitespace().any(|arg| arg == "inkling2.smoke=1"))
}

fn run() -> io::Result<()> {
    register_signals()?;
    state_directory()?;
    let smoke = requested_smoke();
    for attempt in 0..MAX_ATTEMPTS {
        if STOP.load(Relaxed) {
            break;
        }
        if let Ok(pid) = spawn() {
            let start = Instant::now();
            let mut ready = false;
            let mut misses = 0;
            let mut exited = false;
            let mut injected = false;
            while !STOP.load(Relaxed) {
                if reap(Some(pid)) {
                    exited = true;
                    os_warn!("Supervisor", "Bridge exited; attempt {}", attempt + 1);
                    break;
                }
                if healthy() {
                    misses = 0;
                    if !ready {
                        os_info!(
                            "Supervisor",
                            "Bridge healthy on loopback; attempt {}",
                            attempt + 1
                        );
                        ready = true;
                        if smoke && attempt == 0 {
                            os_info!(
                                "Supervisor",
                                "Smoke: terminating bridge to exercise restart"
                            );
                            signal_group(pid, libc::SIGTERM);
                            injected = true;
                        } else if smoke && attempt == 1 {
                            os_info!("Supervisor", "Smoke: restart and health check passed");
                            STOP.store(true, Relaxed);
                        }
                    }
                } else {
                    misses += 1;
                    if !ready && start.elapsed() >= START_TIMEOUT || ready && misses >= 3 {
                        os_warn!("Supervisor", "Bridge health check timed out");
                        break;
                    }
                }
                if injected && start.elapsed() >= START_TIMEOUT {
                    break;
                }
                sleep(Duration::from_millis(500));
            }
            if exited {
                signal_group(pid, libc::SIGKILL);
            } else {
                stop(pid);
            }
        } else {
            os_error!(
                "Supervisor",
                "Bridge could not start; attempt {}",
                attempt + 1
            );
        }
        let delay = Instant::now() + Duration::from_secs(1 << attempt);
        while !STOP.load(Relaxed) && Instant::now() < delay {
            reap(None);
            sleep(Duration::from_millis(100));
        }
    }
    if !STOP.load(Relaxed) {
        os_error!("Supervisor", "Restart limit reached; diagnostic mode");
        while !STOP.load(Relaxed) {
            reap(None);
            sleep(Duration::from_secs(1));
        }
    }
    Ok(())
}

pub fn supervise() -> ! {
    if let Err(error) = run() {
        os_error!("Supervisor", "Diagnostic mode: {error}");
        loop {
            reap(None);
            sleep(Duration::from_secs(1));
        }
    }
    os_info!("Supervisor", "Shutdown complete");
    // This module is reachable only from the real Linux PID 1 path.
    unsafe {
        libc::sync();
        libc::reboot(libc::RB_POWER_OFF);
    }
    loop {
        sleep(Duration::from_secs(1));
    }
}
