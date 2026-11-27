//! Linux PID 1 service supervision. The bridge and all of its children run unprivileged.

use std::fs;
use std::io::{self, Read, Write};
use std::net::{SocketAddr, TcpStream};
use std::os::unix::process::CommandExt;
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering::Relaxed};
use std::thread::sleep;
use std::time::{Duration, Instant};

use crate::{os_error, os_info, os_warn};

const SERVICE_UID: u32 = 61000;
const CONSOLE_UID: u32 = 61001;
const SEARCH_UID: u32 = 61002;
const PORT: u16 = 8080;
const SEARCH_PORT: &str = "29483";
const PREVIEW_DIR: &str = "/var/lib/inkling2/files";
const DESKTOP: &str = "/usr/bin/inkling2-desktop";
const BUNDLED_BOOT_MODEL: &str = "Inkling-2-Mini-SQ-10-abliterated";
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

fn set_identity(command: &mut Command, uid: u32) {
    command.uid(uid).gid(uid).process_group(0);
    // An exec of a setuid binary or a file-capability binary must not grant privileges.
    unsafe {
        command.pre_exec(|| {
            if libc::prctl(libc::PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0 {
                return Err(io::Error::last_os_error());
            }
            Ok(())
        });
    }
}

fn spawn_search() -> io::Result<Option<libc::pid_t>> {
    let binary = "/usr/bin/rustxng";
    if !std::path::Path::new(binary).is_file() {
        return Ok(None);
    }
    let mut command = Command::new(binary);
    command
        .env_clear()
        .args([
            "--bind",
            "127.0.0.1",
            "--port",
            SEARCH_PORT,
            "--profile",
            "full",
        ])
        .stdin(Stdio::null());
    let config = "/etc/inkling2/rustxng.toml";
    if std::path::Path::new(config).is_file() {
        command.args(["--config", config]);
    }
    set_identity(&mut command, SEARCH_UID);
    let child = command.spawn()?;
    i32::try_from(child.id())
        .map(Some)
        .map_err(io::Error::other)
}

fn spawn(search: bool, state_dir: &str) -> io::Result<libc::pid_t> {
    let mut service = Command::new("/usr/bin/inkling2-shell");
    service
        .env_clear()
        .env("PATH", "/usr/sbin:/usr/bin:/sbin:/bin")
        .env("INKLING2_STATE_DIR", state_dir)
        .env("INKLING2_MODELS_DIR", "/models")
        .env("INKLING2_PREVIEW_DIR", PREVIEW_DIR)
        .env("INKLING2_LOCAL_CONSOLE", "1")
        .arg("serve")
        .arg(format!("127.0.0.1:{PORT}"))
        .stdin(Stdio::null());
    // Dedicated administrator-mounted scratch directories for internal drives.
    service.env("INKLING2_EXPERT_NVME_DIR", "/mnt/nvme/inkling2-experts");
    service.env("INKLING2_EXPERT_SATA_DIR", "/mnt/sata/inkling2-experts");
    if state_dir == "/var/lib/inkling2/state" {
        service.env("INKLING2_EXPERT_USB_DIR", state_dir);
    }
    if std::path::Path::new("/etc/inkling2/model-volume").is_file() {
        service.env("INKLING2_RESIDENT", "1");
        if std::path::Path::new("/dev/nvidia0").exists()
            && std::path::Path::new("/models")
                .join(format!("{BUNDLED_BOOT_MODEL}.strata"))
                .is_file()
        {
            service.env("INKLING2_PRELOAD_MODEL", BUNDLED_BOOT_MODEL);
        }
    }
    set_identity(&mut service, SERVICE_UID);
    if search {
        service.env("INKLING2_RUSTXNG_PORT", SEARCH_PORT);
    }
    let child = service.spawn()?;
    i32::try_from(child.id()).map_err(io::Error::other)
}

fn state_directory() -> io::Result<&'static str> {
    use std::os::unix::fs::{PermissionsExt, chown, symlink};
    let installed = fs::read_to_string("/proc/self/mountinfo").is_ok_and(|mounts| {
        mounts.lines().any(|line| {
            line.split_once(" - ").is_some_and(|(mount, details)| {
                mount.split_whitespace().nth(4) == Some("/")
                    && details.split_whitespace().next() == Some("ext4")
            })
        })
    });
    let (state, wine) = if installed {
        ("/var/lib/inkling2/state", "/var/lib/inkling2/wine-state")
    } else {
        ("/run/inkling2", "/run/inkling2-wine")
    };
    for (dir, uid) in [(state, SERVICE_UID), (wine, CONSOLE_UID)] {
        fs::create_dir_all(dir)?;
        fs::set_permissions(dir, fs::Permissions::from_mode(0o700))?;
        chown(dir, Some(uid), Some(uid))?;
    }
    if installed {
        symlink(wine, "/run/inkling2-wine")?;
    }
    Ok(state)
}

fn requested_smoke() -> bool {
    fs::read_to_string("/proc/cmdline")
        .is_ok_and(|line| line.split_whitespace().any(|arg| arg == "inkling2.smoke=1"))
}

fn load_apparmor_profile(profile: &str) -> io::Result<()> {
    use std::path::Path;
    if !Path::new("/proc/sys/kernel/apparmor_restrict_unprivileged_userns").exists() {
        return Ok(());
    }
    if !Path::new("/sys/kernel/security/apparmor").exists()
        && !Command::new("/bin/mount")
            .args(["-t", "securityfs", "securityfs", "/sys/kernel/security"])
            .status()?
            .success()
    {
        return Err(io::Error::other("could not mount AppArmor securityfs"));
    }
    if !Command::new("/usr/sbin/apparmor_parser")
        .args(["--replace", "--skip-cache", profile])
        .status()?
        .success()
    {
        return Err(io::Error::other(format!("could not load {profile}")));
    }
    Ok(())
}

fn spawn_desktop() -> io::Result<Child> {
    load_apparmor_profile("/etc/apparmor.d/inkling2-desktop")?;
    Command::new(DESKTOP)
        .env_clear()
        .env("PATH", "/usr/sbin:/usr/bin:/sbin:/bin")
        .stdin(Stdio::null())
        .process_group(0)
        .spawn()
}

fn smoke_desktop() -> io::Result<()> {
    wait_for_desktop_window("_NET_WM_NAME", "Strata CUDA 13.5 Console", None)?;
    os_info!("Supervisor", "Smoke: local graphical console loaded");
    if std::path::Path::new("/opt/inkling2-tor-browser/Browser/start-tor-browser").is_file() {
        let mut launch = Command::new("/usr/bin/inkling2-desktop-launch");
        set_identity(launch.arg("tor"), CONSOLE_UID);
        if !launch.status()?.success() {
            return Err(io::Error::other(
                "Tor Browser launcher rejected the request",
            ));
        }
        wait_for_desktop_window("WM_CLASS", "tor browser", Some(61003))?;
        os_info!(
            "Supervisor",
            "Smoke: Tor Browser opened under its dedicated account"
        );
    }
    Ok(())
}

fn wait_for_desktop_window(property: &str, expected: &str, uid: Option<u32>) -> io::Result<()> {
    let xprop = |args: &[&str]| {
        let mut command = Command::new("/usr/bin/xprop");
        command
            .env_clear()
            .env("DISPLAY", ":0")
            .env("XAUTHORITY", "/run/inkling2-desktop/Xauthority")
            .args(args);
        set_identity(&mut command, CONSOLE_UID);
        command.output()
    };
    let deadline = Instant::now() + Duration::from_secs(90);
    while Instant::now() < deadline {
        let windows = xprop(&["-root", "_NET_CLIENT_LIST"])?;
        for window in String::from_utf8_lossy(&windows.stdout)
            .split(|character: char| character.is_ascii_whitespace() || character == ',')
            .filter(|word| word.starts_with("0x"))
        {
            let value = xprop(&["-id", window, property])?;
            if !value.status.success()
                || !String::from_utf8_lossy(&value.stdout)
                    .to_ascii_lowercase()
                    .contains(&expected.to_ascii_lowercase())
            {
                continue;
            }
            if let Some(uid) = uid {
                let info = xprop(&["-id", window, "_NET_WM_PID"])?;
                let pid = String::from_utf8_lossy(&info.stdout)
                    .split_once('=')
                    .and_then(|(_, number)| number.trim().parse::<u32>().ok());
                let actual = pid
                    .and_then(|pid| fs::read_to_string(format!("/proc/{pid}/status")).ok())
                    .and_then(|status| {
                        status.lines().find_map(|line| {
                            line.strip_prefix("Uid:")?
                                .split_whitespace()
                                .next()?
                                .parse::<u32>()
                                .ok()
                        })
                    });
                if actual != Some(uid) {
                    continue;
                }
            }
            return Ok(());
        }
        sleep(Duration::from_millis(500));
    }
    Err(io::Error::other(format!(
        "desktop window did not load: {expected}"
    )))
}

fn smoke_wine() -> io::Result<()> {
    let mut command = Command::new("/usr/bin/timeout");
    // The first sandboxed Win64 prefix took 264 seconds on the UEFI/TCG qualification guest.
    command.args(["--signal=KILL", "300", "/usr/bin/wine-inkling2", "--check"]);
    command.uid(CONSOLE_UID).gid(CONSOLE_UID).process_group(0);
    let output = command.output()?;
    if !output.status.success()
        || !String::from_utf8_lossy(&output.stdout).contains("INKLING2_WINE_OK")
    {
        os_warn!(
            "Supervisor",
            "Wine probe failed ({}): {}",
            output.status,
            String::from_utf8_lossy(&output.stderr).trim()
        );
        return Err(io::Error::other(
            "Wine could not run its Windows executable",
        ));
    }
    os_info!("Supervisor", "Smoke: offline Wine executed Windows cmd.exe");
    Ok(())
}

fn spawn_terminal() -> io::Result<libc::pid_t> {
    let child = Command::new("/sbin/agetty")
        .env_clear()
        .env("PATH", "/usr/sbin:/usr/bin:/sbin:/bin")
        .args([
            "--autologin",
            "inkling2-console",
            "--noclear",
            "tty2",
            "linux",
        ])
        .process_group(0)
        .spawn()?;
    i32::try_from(child.id()).map_err(io::Error::other)
}

fn smoke_access_policy() -> io::Result<()> {
    let under = |uid, executable, args: &[&str]| {
        let mut command = Command::new(executable);
        set_identity(
            command
                .args(args)
                .stdin(Stdio::null())
                .stdout(Stdio::null())
                .stderr(Stdio::null()),
            uid,
        );
        command.status().map(|status| status.success())
    };
    let health = [
        "--disable",
        "--noproxy",
        "*",
        "--fail",
        "--max-time",
        "2",
        "http://127.0.0.1:8080/health",
    ];
    if !under(CONSOLE_UID, "/usr/bin/curl", &health)?
        || under(65534, "/usr/bin/curl", &health)?
        || under(CONSOLE_UID, "/usr/bin/test", &["-r", PREVIEW_DIR])?
        || !under(SERVICE_UID, "/usr/bin/test", &["-r", PREVIEW_DIR])?
    {
        return Err(io::Error::other(
            "operator and broker access boundaries failed",
        ));
    }
    let files = [
        "--disable",
        "--noproxy",
        "*",
        "--fail",
        "--max-time",
        "3",
        "--header",
        "Content-Type: application/json",
        "--data",
        r#"{"root":"workspace"}"#,
        "http://127.0.0.1:8080/v1/fs/list",
    ];
    if !under(CONSOLE_UID, "/usr/bin/curl", &files)? {
        return Err(io::Error::other("key-free Finder listing failed"));
    }
    if std::path::Path::new("/etc/inkling2/model-volume").is_file()
        && (under(CONSOLE_UID, "/usr/bin/test", &["-r", "/models"])?
            || !under(SERVICE_UID, "/usr/bin/test", &["-r", "/models"])?)
    {
        return Err(io::Error::other(
            "bundled model volume access boundary failed",
        ));
    }
    os_info!(
        "Supervisor",
        "Smoke: console UID and private file directory verified"
    );
    Ok(())
}

fn run(firewall_loaded: bool) -> io::Result<()> {
    register_signals()?;
    if !firewall_loaded {
        return Err(io::Error::other(
            "kernel access policy missing; console and search disabled",
        ));
    }
    let state_dir = state_directory()?;
    let smoke = requested_smoke();
    if std::path::Path::new("/etc/inkling2/model-volume").is_file() {
        let status = Command::new("/bin/mount").arg("/models").status()?;
        if status.success() {
            os_info!("Supervisor", "Bundled model volume mounted read-only");
        } else {
            os_warn!(
                "Supervisor",
                "Bundled model volume could not mount: {status}"
            );
            if smoke {
                return Err(io::Error::other("bundled model volume mount failed"));
            }
        }
    }
    if let Err(error) = load_apparmor_profile("/etc/apparmor.d/inkling2-wine") {
        os_warn!("Supervisor", "Wine sandbox initialization failed: {error}");
        if smoke {
            return Err(error);
        }
    }
    if !smoke
        && std::path::Path::new("/dev/tty2").exists()
        && let Err(error) = spawn_terminal()
    {
        os_warn!(
            "Supervisor",
            "Local operator terminal could not start: {error}"
        );
    }
    let private_desktop = std::path::Path::new(DESKTOP).is_file();
    let mut desktop = None;
    let result = run_services(state_dir, smoke, private_desktop, &mut desktop);
    if let Some(mut child) = desktop
        && child.try_wait().is_ok_and(|status| status.is_none())
        && let Ok(pid) = i32::try_from(child.id())
    {
        stop(pid);
    }
    result
}

fn run_services(
    state_dir: &str,
    smoke: bool,
    private_desktop: bool,
    desktop: &mut Option<Child>,
) -> io::Result<()> {
    for attempt in 0..MAX_ATTEMPTS {
        if STOP.load(Relaxed) {
            break;
        }
        let search = spawn_search().unwrap_or_else(|error| {
            os_warn!("Supervisor", "RustXNG launch failed: {error}");
            None
        });
        if let Ok(pid) = spawn(search.is_some(), state_dir) {
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
                        if private_desktop && desktop.is_none() {
                            match spawn_desktop() {
                                Ok(child) => *desktop = Some(child),
                                Err(error) => {
                                    os_warn!("Supervisor", "Local display failed: {error}")
                                }
                            }
                        }
                        if smoke && attempt == 0 {
                            let verified = smoke_access_policy().and_then(|()| {
                                if private_desktop {
                                    smoke_desktop()?;
                                }
                                smoke_wine()
                            });
                            if let Err(error) = verified {
                                stop(pid);
                                if let Some(search_pid) = search {
                                    stop(search_pid);
                                }
                                return Err(error);
                            }
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
        if let Some(search_pid) = search {
            stop(search_pid);
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

pub fn supervise(firewall_loaded: bool) -> ! {
    if let Err(error) = run(firewall_loaded) {
        os_error!("Supervisor", "Diagnostic mode: {error}");
        if !requested_smoke() {
            loop {
                reap(None);
                sleep(Duration::from_secs(1));
            }
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
