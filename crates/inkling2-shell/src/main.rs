//! Inkling2-OS AI-Native Shell (`inkling2-shell`).
//!
//! Primary user interface spawned by PID 1 (`inkling2-init`) on `/dev/tty1`.
//! Integrates:
//! - CUDA 13.5 `sm_120a` inference engine (`inkling2-engine`)
//! - 45 MiB compressed episodic memory (`memory45`) + `hakua-memory` bridge
//! - Dynamic rank-16 LoRA online self-evolution & rollback (`dynamic_lora`)
//! - Cute girl voice TTS (`voice_call`) + 40x40-patch webcam pipeline (`camera_call`)
//! - TLS 1.2+ SSRF-guarded internet & Hugging Face Hub client (`net_client`)

mod hardened_fs;
mod http_bridge;
mod http_request;
mod muse_ssh;
mod native_inference;
mod native_prompt;
mod net_client;
mod public_browser;
mod resident_engine;
mod strata_catalog;
mod web_search;

use std::env;
use std::io::{self, BufRead, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode};

use inkling2_companion::camera_call::{CameraConfig, capture_webcam_frame};
use inkling2_companion::dynamic_lora::{DynamicLoraAdapter, LoraConfig};
use inkling2_companion::memory45::{MAX_MEMORY_BYTES, Memory45Store};
use inkling2_companion::voice_call::{VoicePersona, speak_text};
use muse_ssh::{
    cmd_muse_import, cmd_muse_inbox_ready, cmd_muse_keygen, cmd_muse_ping, cmd_muse_sync,
    cmd_muse_tunnel,
};
use net_client::{cmd_hf_pull, cmd_net_fetch, cmd_net_status};
use strata_catalog::{discover_models, inspect_strata_header};

struct ShellConfig {
    engine_bin: PathBuf,
    models_dir: PathBuf,
    state_dir: PathBuf,
}

impl ShellConfig {
    fn detect() -> Self {
        let engine_bin = env::var("INKLING2_ENGINE_BIN")
            .map(PathBuf::from)
            .unwrap_or_else(|_| {
                if cfg!(target_os = "windows") {
                    PathBuf::from(r"C:\Users\downl\inkling2-cuda-engine\bin\inkling2-engine.exe")
                } else {
                    PathBuf::from("/usr/bin/inkling2-engine")
                }
            });
        let models_dir = env::var("INKLING2_MODELS_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|_| {
                if cfg!(target_os = "windows") {
                    PathBuf::from(r"H:\Strata\models\Inkling-2-Mini-Heretic-ModeB-NVFP4")
                } else {
                    PathBuf::from("/models")
                }
            });
        let state_dir = env::var("INKLING2_STATE_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|_| {
                if cfg!(target_os = "windows") {
                    PathBuf::from(r"C:\Users\downl\inkling2-os\state")
                } else {
                    PathBuf::from("/var/lib/inkling2")
                }
            });
        Self {
            engine_bin,
            models_dir,
            state_dir,
        }
    }

    fn memory_path(&self) -> PathBuf {
        self.state_dir.join("memory45.ikm")
    }

    fn lora_path(&self) -> PathBuf {
        self.state_dir.join("dynamic_lora.iklora")
    }
}

struct ShellState {
    memory: Memory45Store,
    lora: DynamicLoraAdapter,
    voice: VoicePersona,
    cam: CameraConfig,
}

impl ShellState {
    fn load_or_new(cfg: &ShellConfig) -> Self {
        let memory = Memory45Store::load(&cfg.memory_path()).unwrap_or_default();
        let lora = DynamicLoraAdapter::load(&cfg.lora_path())
            .unwrap_or_else(|_| DynamicLoraAdapter::new(LoraConfig::default()));
        Self {
            memory,
            lora,
            voice: VoicePersona::default(),
            cam: CameraConfig::default(),
        }
    }
}

fn write_line(out: &mut impl Write, text: &str) {
    let _ = writeln!(out, "{text}");
    let _ = out.flush();
}

fn handle_command(
    cfg: &ShellConfig,
    st: &mut ShellState,
    line: &str,
    out: &mut impl Write,
) -> bool {
    let parts: Vec<&str> = line.split_whitespace().collect();
    let Some(cmd) = parts.first().copied() else {
        return true;
    };

    match cmd {
        "help" => {
            write_line(out, "Inkling2-OS Commands:");
            write_line(
                out,
                "  models                        Scan models directory for SC117 .strata v1.1 builds",
            );
            write_line(
                out,
                "  inspect <file.strata>         Inspect .strata v1.1 binary header",
            );
            write_line(
                out,
                "  bench-tc                      Run CUDA 13.5 sm_120a Tensor Core benchmark",
            );
            write_line(
                out,
                "  bench <file> [tokens]         Run 48-logical-layer looped MoE decode benchmark",
            );
            write_line(out, "  mem stats | add <imp> <txt> | recall <query>");
            write_line(out, "  lora stats | reinforce <r> <txt> | rollback");
            write_line(
                out,
                "  voice say <text>              Speak with cute girl voice (VOICEVOX / Haruka)",
            );
            write_line(
                out,
                "  cam snap [out.jpg]            Capture 40x40-patch-aligned webcam frame",
            );
            write_line(out, "  net status | fetch <url> | hf-pull <repo> <file>");
            write_line(
                out,
                "  muse keygen | ping <user@host> [port] | tunnel <user@host> <rport> <lport> | sync <user@host> <rdir>",
            );
            write_line(
                out,
                "  sysinfo                       Show OS runtime and paths",
            );
            write_line(
                out,
                "  exit | poweroff               Save state and exit AI-OS shell",
            );
        }
        "muse" => match parts.get(1).copied() {
            Some("keygen") => cmd_muse_keygen(&cfg.state_dir, out),
            Some("inbox") => cmd_muse_inbox_ready(&cfg.state_dir, out),
            Some("import") => {
                cmd_muse_import(&cfg.state_dir, out);
                st.memory = Memory45Store::load(&cfg.memory_path()).unwrap_or_default();
                st.lora = DynamicLoraAdapter::load(&cfg.lora_path())
                    .unwrap_or_else(|_| DynamicLoraAdapter::new(LoraConfig::default()));
            }
            Some("ping") if parts.len() >= 3 => {
                let port: u16 = parts.get(3).and_then(|s| s.parse().ok()).unwrap_or(22);
                cmd_muse_ping(&cfg.state_dir, parts[2], port, out);
            }
            Some("tunnel") if parts.len() >= 5 => {
                let rport: u16 = parts[3].parse().unwrap_or(18080);
                let lport: u16 = parts[4].parse().unwrap_or(8080);
                let sport: u16 = parts.get(5).and_then(|s| s.parse().ok()).unwrap_or(22);
                cmd_muse_tunnel(&cfg.state_dir, parts[2], rport, lport, sport, out);
            }
            Some("sync") if parts.len() >= 4 => {
                cmd_muse_sync(&cfg.state_dir, parts[2], parts[3], out);
            }
            _ => write_line(
                out,
                "usage: muse <keygen | ping <user@host> [port] | tunnel <user@host> <rport> <lport> [ssh_port] | sync <user@host> <rdir>>",
            ),
        },
        "mem" => match parts.get(1).copied() {
            Some("stats") => {
                let s = st.memory.stats();
                write_line(
                    out,
                    &format!(
                        "Memory45: entries={} size={} / {} bytes ({:.2} / {:.2} MiB) compactions={}",
                        s.entry_count,
                        s.serialized_bytes,
                        s.budget_bytes,
                        (s.serialized_bytes as f64) / (1024.0 * 1024.0),
                        (MAX_MEMORY_BYTES as f64) / (1024.0 * 1024.0),
                        s.compactions
                    ),
                );
            }
            Some("add") if parts.len() >= 4 => {
                let imp: f32 = parts[2].parse().unwrap_or(0.8);
                let text = parts[3..].join(" ");
                let id = st.memory.insert(text, imp);
                match st.memory.save_atomic(&cfg.memory_path()) {
                    Ok(()) => {
                        write_line(out, &format!("Memory45 stored id={id} (budget<=45 MiB OK)"))
                    }
                    Err(e) => write_line(out, &format!("Memory45 save error: {e}")),
                }
            }
            Some("recall") if parts.len() >= 3 => {
                let q = parts[2..].join(" ");
                let hits = st.memory.recall(&q, 5);
                if hits.is_empty() {
                    write_line(out, "Memory45: no matching entries");
                } else {
                    for h in hits {
                        write_line(
                            out,
                            &format!(
                                "  [id={} imp={:.2} str={:.1}] {}",
                                h.id, h.importance, h.strength, h.text
                            ),
                        );
                    }
                }
            }
            _ => write_line(
                out,
                "usage: mem <stats | add <0.0..1.0> <text> | recall <query>>",
            ),
        },
        "lora" => match parts.get(1).copied() {
            Some("stats") => {
                let s = st.lora.stats();
                write_line(
                    out,
                    &format!(
                        "DynamicLoRA: steps={} params={} cum_reward={:.3} ||A||={:.4} ||B||={:.4}",
                        s.evolution_steps, s.param_count, s.cumulative_reward, s.norm_a, s.norm_b
                    ),
                );
            }
            Some("reinforce") if parts.len() >= 4 => {
                let reward: f32 = parts[2].parse().unwrap_or(0.5);
                let text = parts[3..].join(" ");
                let emb = Memory45Store::embed_text_q8(&text);
                let mut h = vec![0.0f32; 2304];
                let mut target = vec![0.0f32; 2304];
                for i in 0..2304 {
                    let v = f32::from(emb[i % emb.len()]) / 127.0;
                    h[i] = v;
                    target[i] = v * reward.signum();
                }
                st.lora.reinforce_step(&h, &target, reward);
                let s = st.lora.stats();
                let save_res = st.lora.save_atomic(&cfg.lora_path());
                write_line(
                    out,
                    &format!(
                        "DynamicLoRA reinforced: step={} ||B||={:.4} persisted={}",
                        s.evolution_steps,
                        s.norm_b,
                        save_res.is_ok()
                    ),
                );
            }
            Some("rollback") => {
                if st.lora.rollback() {
                    let _ = st.lora.save_atomic(&cfg.lora_path());
                    write_line(out, "DynamicLoRA rolled back to previous generation.");
                } else {
                    write_line(out, "DynamicLoRA: no previous snapshot available.");
                }
            }
            _ => write_line(
                out,
                "usage: lora <stats | reinforce <-1..1> <text> | rollback>",
            ),
        },
        "voice" => match parts.get(1).copied() {
            Some("say") if parts.len() >= 3 => {
                let text = parts[2..].join(" ");
                let wav = cfg.state_dir.join("last_voice.wav");
                let _ = std::fs::create_dir_all(&cfg.state_dir);
                match speak_text(&st.voice, &text, Some(&wav)) {
                    Ok(msg) => write_line(out, &format!("Voice: {msg}")),
                    Err(e) => write_line(out, &format!("Voice error: {e}")),
                }
            }
            _ => write_line(out, "usage: voice say <text>"),
        },
        "cam" => match parts.get(1).copied() {
            Some("snap") => {
                let target = parts
                    .get(2)
                    .map(PathBuf::from)
                    .unwrap_or_else(|| cfg.state_dir.join("cam_frame.jpg"));
                let _ = std::fs::create_dir_all(&cfg.state_dir);
                match capture_webcam_frame(&st.cam, &target) {
                    Ok(msg) => write_line(out, &format!("Camera: {msg}")),
                    Err(e) => write_line(out, &format!("Camera: {e}")),
                }
            }
            _ => write_line(out, "usage: cam snap [out.jpg]"),
        },
        "net" => match parts.get(1).copied() {
            Some("status") => cmd_net_status(out),
            Some("fetch") => {
                if let Some(url) = parts.get(2) {
                    cmd_net_fetch(url, out);
                } else {
                    write_line(out, "usage: net fetch <https-url>");
                }
            }
            Some("hf-pull") => {
                if let (Some(repo), Some(file)) = (parts.get(2), parts.get(3)) {
                    cmd_hf_pull(repo, file, &cfg.models_dir, out);
                } else {
                    write_line(out, "usage: net hf-pull <owner/repo> <filename.strata>");
                }
            }
            _ => write_line(
                out,
                "usage: net <status | fetch <url> | hf-pull <repo> <file>>",
            ),
        },
        "sysinfo" => {
            write_line(
                out,
                &format!("OS Target   : Inkling2-OS v0.1.0 ({})", env::consts::OS),
            );
            write_line(out, &format!("Engine Bin  : {}", cfg.engine_bin.display()));
            write_line(out, &format!("Models Root : {}", cfg.models_dir.display()));
            write_line(out, &format!("State Root  : {}", cfg.state_dir.display()));
        }
        "models" => {
            let models = discover_models(&cfg.models_dir);
            if models.is_empty() {
                write_line(
                    out,
                    &format!(
                        "No .strata v1.1 models found under {}",
                        cfg.models_dir.display()
                    ),
                );
            } else {
                for m in models {
                    let gib = (m.file_bytes as f64) / (1024.0 * 1024.0 * 1024.0);
                    write_line(
                        out,
                        &format!(
                            "  [fmt={} sec={}] {:.2} GiB  {}",
                            m.format_version,
                            m.section_count,
                            gib,
                            m.path.display()
                        ),
                    );
                }
            }
        }
        "inspect" => {
            let Some(path_str) = parts.get(1) else {
                write_line(out, "usage: inspect <path.strata>");
                return true;
            };
            match inspect_strata_header(Path::new(path_str)) {
                Ok(m) => write_line(
                    out,
                    &format!(
                        "STRATA v1.{} | sections={} | size={} bytes | manifest_off={} ({} bytes)",
                        m.format_version,
                        m.section_count,
                        m.file_bytes,
                        m.manifest_offset,
                        m.manifest_bytes
                    ),
                ),
                Err(e) => write_line(out, &format!("error: {e}")),
            }
        }
        "bench-tc" => {
            let status = Command::new(&cfg.engine_bin).arg("bench-tc").status();
            match status {
                Ok(s) => write_line(out, &format!("bench-tc exited with {s}")),
                Err(e) => write_line(
                    out,
                    &format!("failed to launch {}: {e}", cfg.engine_bin.display()),
                ),
            }
        }
        "bench" => {
            let Some(path_str) = parts.get(1) else {
                write_line(out, "usage: bench <path.strata> [tokens]");
                return true;
            };
            let tokens = parts.get(2).copied().unwrap_or("16");
            let status = Command::new(&cfg.engine_bin)
                .args(["bench", path_str, tokens])
                .status();
            match status {
                Ok(s) => write_line(out, &format!("bench exited with {s}")),
                Err(e) => write_line(
                    out,
                    &format!("failed to launch {}: {e}", cfg.engine_bin.display()),
                ),
            }
        }
        "exit" | "quit" | "poweroff" => {
            let _ = st.memory.save_atomic(&cfg.memory_path());
            let _ = st.lora.save_atomic(&cfg.lora_path());
            write_line(out, "State saved. Shutting down Inkling2-OS Shell...");
            return false;
        }
        _ => {
            write_line(
                out,
                &format!("Unknown command '{cmd}'. Type 'help' for available commands."),
            );
        }
    }
    true
}

fn open_browser(url: &str) {
    if cfg!(target_os = "windows") {
        let _ = Command::new("cmd").args(["/C", "start", "", url]).spawn();
    } else {
        let _ = Command::new("xdg-open").arg(url).spawn();
    }
}

/// Parse Strata-aligned `launch` CLI flags (`--port`, `--host`, `--open`).
fn parse_launch_args(args: &[String]) -> (String, bool) {
    let mut host = "127.0.0.1".to_string();
    let mut port: u16 = 8080;
    let mut open_ui = false;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--port" | "-p" => {
                if let Some(v) = args.get(i + 1).and_then(|s| s.parse::<u16>().ok()) {
                    port = v;
                    i += 1;
                }
            }
            "--host" => {
                if let Some(v) = args.get(i + 1) {
                    host = v.clone();
                    i += 1;
                }
            }
            "--open" => {
                open_ui = true;
            }
            _ => {}
        }
        i += 1;
    }
    (format!("{host}:{port}"), open_ui)
}

fn cmd_cli_status(port: u16, out: &mut impl Write) {
    let health_url = format!("http://127.0.0.1:{port}/health");
    write_line(
        out,
        &format!("Inkling2-OS status (http://127.0.0.1:{port})"),
    );
    match Command::new("curl")
        .args([
            "--disable",
            "--silent",
            "--fail",
            "--noproxy",
            "*",
            "--connect-timeout",
            "2",
            "--max-time",
            "5",
            "--",
            &health_url,
        ])
        .output()
    {
        Ok(result) if result.status.success() => {
            write_line(
                out,
                &format!("Bridge: {}", String::from_utf8_lossy(&result.stdout).trim()),
            );
        }
        _ => write_line(out, "Bridge: offline"),
    }
    if let Some(gpu) = http_bridge::query_live_gpu_telemetry() {
        write_line(
            out,
            &format!(
                "GPU: {} · {} / {} MiB",
                gpu.name, gpu.memory_used_mib, gpu.memory_total_mib
            ),
        );
    }
}

fn main() -> ExitCode {
    let cfg = ShellConfig::detect();
    let mut st = ShellState::load_or_new(&cfg);
    let args: Vec<String> = env::args().collect();
    let mut stdout = io::stdout().lock();

    if args.len() >= 2 {
        match args[1].as_str() {
            "serve" => {
                let bind = args.get(2).map(String::as_str).unwrap_or("127.0.0.1:8080");
                if let Err(e) = http_bridge::run_http_bridge(
                    bind,
                    cfg.engine_bin.clone(),
                    cfg.models_dir.clone(),
                    cfg.state_dir.clone(),
                ) {
                    let _ = writeln!(io::stderr(), "[Inkling2-Bridge] error: {e}");
                    return ExitCode::FAILURE;
                }
                return ExitCode::SUCCESS;
            }
            "launch" => {
                let (bind, open_ui) = parse_launch_args(&args[2..]);
                if open_ui {
                    let url = format!("http://{bind}/");
                    open_browser(&url);
                }
                if let Err(e) = http_bridge::run_http_bridge(
                    &bind,
                    cfg.engine_bin.clone(),
                    cfg.models_dir.clone(),
                    cfg.state_dir.clone(),
                ) {
                    let _ = writeln!(io::stderr(), "[Inkling2-Bridge] error: {e}");
                    return ExitCode::FAILURE;
                }
                return ExitCode::SUCCESS;
            }
            "status" => {
                let mut port: u16 = 8080;
                if let Some(pos) = args.iter().position(|a| a == "--port" || a == "-p")
                    && let Some(v) = args.get(pos + 1).and_then(|s| s.parse::<u16>().ok())
                {
                    port = v;
                }
                cmd_cli_status(port, &mut stdout);
                return ExitCode::SUCCESS;
            }
            "ui" => {
                let mut port: u16 = 8080;
                if let Some(pos) = args.iter().position(|a| a == "--port" || a == "-p")
                    && let Some(v) = args.get(pos + 1).and_then(|s| s.parse::<u16>().ok())
                {
                    port = v;
                }
                let url = format!("http://127.0.0.1:{port}/");
                write_line(
                    &mut stdout,
                    &format!("Opening Inkling2-OS macOS Console: {url}"),
                );
                open_browser(&url);
                return ExitCode::SUCCESS;
            }
            "-c" if args.len() >= 3 => {
                let joined = args[2..].join(" ");
                handle_command(&cfg, &mut st, &joined, &mut stdout);
                return ExitCode::SUCCESS;
            }
            "-h" | "--help" | "help" => {
                write_line(
                    &mut stdout,
                    "Inkling2-OS Strata-Aligned CLI (CUDA 13.5 sm_120a)",
                );
                write_line(
                    &mut stdout,
                    "  inkling2-shell launch [--port 8080] [--open]  Start loopback server + macOS Web GUI",
                );
                write_line(
                    &mut stdout,
                    "  inkling2-shell status [--port 8080]           Check live bridge & RTX 5060 Ti telemetry",
                );
                write_line(
                    &mut stdout,
                    "  inkling2-shell ui [--port 8080]               Open macOS Web GUI in default browser",
                );
                write_line(
                    &mut stdout,
                    "  inkling2-shell serve [127.0.0.1:8080]         Run loopback HTTP/OpenAI bridge",
                );
                write_line(
                    &mut stdout,
                    "  inkling2-shell models | bench-tc | bench | inspect | sysinfo | mem | lora | voice | net | muse",
                );
                return ExitCode::SUCCESS;
            }
            // Direct Strata-style top-level execution for all shell subcommands
            "models" | "inspect" | "bench-tc" | "bench" | "sysinfo" | "mem" | "lora" | "voice"
            | "cam" | "net" | "muse" => {
                let joined = args[1..].join(" ");
                handle_command(&cfg, &mut st, &joined, &mut stdout);
                return ExitCode::SUCCESS;
            }
            _ => {}
        }
    }

    write_line(
        &mut stdout,
        "Inkling2-OS AI-Native Shell v0.1.0 (CUDA 13.5 sm_120a + Memory45 + DynamicLoRA)",
    );
    write_line(&mut stdout, "Type 'help' for commands.");

    let stdin = io::stdin();
    for line in stdin.lock().lines() {
        let _ = write!(stdout, "inkling2-os> ");
        let _ = stdout.flush();
        match line {
            Ok(l) => {
                if !handle_command(&cfg, &mut st, l.trim(), &mut stdout) {
                    break;
                }
            }
            Err(_) => break,
        }
    }
    ExitCode::SUCCESS
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shell_help_mem_lora_and_sysinfo_execute_cleanly() {
        let cfg = ShellConfig {
            state_dir: env::temp_dir().join("inkling2_os_test_state"),
            ..ShellConfig::detect()
        };
        let mut st = ShellState::load_or_new(&cfg);
        let mut buf: Vec<u8> = Vec::new();
        assert!(handle_command(&cfg, &mut st, "help", &mut buf));
        assert!(handle_command(
            &cfg,
            &mut st,
            "mem add 0.9 Inkling2-OSテスト記憶",
            &mut buf
        ));
        assert!(handle_command(
            &cfg,
            &mut st,
            "mem recall Inkling2-OS",
            &mut buf
        ));
        assert!(handle_command(
            &cfg,
            &mut st,
            "lora reinforce 0.8 音声対話フィードバック",
            &mut buf
        ));
        assert!(handle_command(&cfg, &mut st, "lora stats", &mut buf));
        let text = String::from_utf8_lossy(&buf);
        assert!(text.contains("Memory45 stored"));
        assert!(text.contains("DynamicLoRA reinforced"));
    }

    #[test]
    fn net_security_guards_block_http_ssrf_and_traversal() {
        assert!(public_browser::host_for_url("https://huggingface.co/SC117").is_ok());
        assert!(public_browser::host_for_url("http://huggingface.co").is_err());
        assert!(public_browser::host_for_url("https://127.0.0.1/secret").is_err());
        assert!(public_browser::host_for_url("https://169.254.169.254/latest/meta-data/").is_err());
        assert!(public_browser::host_for_url("https://192.168.1.1/admin").is_err());
        assert!(
            net_client::build_hf_resolve_url(
                "SC117/Inkling-2-Mini-abliterated-Strata",
                "Inkling-2-Mini-SQ-20-abliterated.strata"
            )
            .is_ok()
        );
        assert!(net_client::build_hf_resolve_url("SC117/repo", "../etc/passwd").is_err());
    }

    #[test]
    fn muse_ssh_validates_targets_and_rejects_injection() {
        assert!(muse_ssh::validate_ssh_target("muse@100.64.0.12").is_ok());
        assert!(muse_ssh::validate_ssh_target("user@muse-vm.ts.net").is_ok());
        assert!(muse_ssh::validate_ssh_target("-oProxyCommand=evil").is_err());
        assert!(muse_ssh::validate_ssh_target("muse@host;rm -rf /").is_err());
    }

    #[test]
    fn strata_cli_and_gpu_parsers_work() {
        let args = vec![
            "--port".to_string(),
            "8090".to_string(),
            "--open".to_string(),
        ];
        assert_eq!(
            parse_launch_args(&args),
            ("127.0.0.1:8090".to_string(), true)
        );

        let csv = "NVIDIA GeForce RTX 5060 Ti, 621.24, 6500, 16311, 42, 54, 88.50, 180.00, 2610";
        let gpu = http_bridge::parse_nvidia_smi_csv(csv);
        assert!(gpu.is_some());
        if let Some(g) = gpu {
            assert_eq!(g.memory_used_mib, 6500);
            assert_eq!(g.memory_total_mib, 16311);
            assert_eq!(g.temperature_c, 54);
            assert_eq!(g.clock_sm_mhz, 2610);
        }

        let bench_line = "[INFO] [BenchTC] BF16=8.56 TFLOPS | INT8=16.90 TOPS | NVFP4=29.99 TFLOPS";
        let (bf16, int8, nvfp4) = http_bridge::parse_bench_tc_metrics(bench_line);
        assert!((bf16 - 8.56).abs() < 0.01);
        assert!((int8 - 16.90).abs() < 0.01);
        assert!((nvfp4 - 29.99).abs() < 0.01);
    }
}
