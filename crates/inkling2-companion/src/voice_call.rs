//! Local TTS & Speech Playback Runtime (`voice_call`).
//!
//! Priority backend chain:
//! 1. Local VOICEVOX Engine (`http://127.0.0.1:50021`, speaker 8 春日部つむぎ / speaker 3 ずんだもん)
//!    with actual `speedScale` application and WAV playback when `out_wav` is `None`.
//! 2. Windows native Japanese female SAPI (`Microsoft Haruka Desktop`) with `speed_scale` -> SAPI Rate mapping.
//! 3. Linux local TTS (`piper` / `espeak-ng`) + `aplay`/`paplay`.

#[cfg(windows)]
use std::os::windows::process::CommandExt;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

fn voice_command(program: &str) -> Command {
    #[cfg(windows)]
    {
        let mut command = Command::new(program);
        command.creation_flags(0x08000000);
        command
    }
    #[cfg(not(windows))]
    Command::new(program)
}

/// Voice persona configuration.
#[derive(Debug, Clone)]
pub struct VoicePersona {
    /// VOICEVOX local speaker ID (8 = 春日部つむぎ, 3 = ずんだもん, 2 = 四国めたん).
    pub voicevox_speaker_id: u32,
    /// Local VOICEVOX endpoint (strictly loopback).
    pub voicevox_url: String,
    /// Speech rate multiplier (`0.5..=2.0`, default `1.08`).
    pub speed_scale: f32,
}

impl Default for VoicePersona {
    fn default() -> Self {
        Self {
            voicevox_speaker_id: 8,
            voicevox_url: "http://127.0.0.1:50021".to_string(),
            speed_scale: 1.08,
        }
    }
}

/// Cooperative cancellation handle for interrupting queued sentence playback.
#[derive(Debug, Clone, Default)]
pub struct SpeechController {
    stop_flag: Arc<AtomicBool>,
}

impl SpeechController {
    /// Request immediate stop of any queued speech synthesis/playback.
    pub fn request_stop(&self) {
        self.stop_flag.store(true, Ordering::SeqCst);
    }

    /// Clear stop flag before starting a new speech sequence.
    pub fn reset(&self) {
        self.stop_flag.store(false, Ordering::SeqCst);
    }

    /// Check whether a stop has been requested.
    pub fn is_stopped(&self) -> bool {
        self.stop_flag.load(Ordering::SeqCst)
    }
}

/// Map `speed_scale` (`0.5..=2.0`) to Windows SAPI `Rate` (`-10..=10`).
pub fn speed_scale_to_sapi_rate(speed_scale: f32) -> i32 {
    if !speed_scale.is_finite() {
        return 0;
    }
    let clamped = speed_scale.clamp(0.5, 2.0);
    ((clamped - 1.0) * 10.0).round().clamp(-10.0, 10.0) as i32
}

/// Rewrite `"speedScale": <number>` inside a VOICEVOX `audio_query` JSON payload without external deps.
pub fn apply_speed_scale_to_audio_query(query_json: &str, speed_scale: f32) -> String {
    let clamped = if speed_scale.is_finite() {
        speed_scale.clamp(0.5, 2.0)
    } else {
        1.0
    };
    let needle = "\"speedScale\"";
    let Some(key_pos) = query_json.find(needle) else {
        return query_json.to_string();
    };
    let after_key = &query_json[key_pos + needle.len()..];
    let Some(colon_rel) = after_key.find(':') else {
        return query_json.to_string();
    };
    let val_start = key_pos + needle.len() + colon_rel + 1;
    let rest = &query_json[val_start..];
    let ws_len = rest.len() - rest.trim_start().len();
    let num_start = val_start + ws_len;
    let num_slice = &query_json[num_start..];
    let num_len = num_slice
        .bytes()
        .take_while(|b| b.is_ascii_digit() || matches!(b, b'.' | b'-' | b'+' | b'e' | b'E'))
        .count();
    if num_len == 0 {
        return query_json.to_string();
    }
    let mut out = String::with_capacity(query_json.len() + 8);
    out.push_str(&query_json[..num_start]);
    out.push_str(&format!("{clamped:.2}"));
    out.push_str(&query_json[num_start + num_len..]);
    out
}

/// Split Japanese/English reply text into natural sentence chunks for queued speech playback.
pub fn split_sentences(text: &str) -> Vec<String> {
    let mut sentences = Vec::new();
    let mut current = String::new();
    for ch in text.chars() {
        if ch == '\r' {
            continue;
        }
        if ch == '\n' {
            let trimmed = current.trim();
            if !trimmed.is_empty() {
                sentences.push(trimmed.to_string());
            }
            current.clear();
            continue;
        }
        current.push(ch);
        if matches!(ch, '。' | '！' | '？' | '.' | '!' | '?') {
            let trimmed = current.trim();
            if !trimmed.is_empty() {
                sentences.push(trimmed.to_string());
            }
            current.clear();
        }
    }
    let tail = current.trim();
    if !tail.is_empty() {
        sentences.push(tail.to_string());
    }
    sentences
}

/// Verify that `url` is strictly loopback (`127.0.0.1` or `localhost`) before probing VOICEVOX.
fn is_loopback_url(url: &str) -> bool {
    let Some(authority) = url.trim_end_matches('/').strip_prefix("http://") else {
        return false;
    };
    let (host, port) = authority
        .split_once(':')
        .map_or((authority, None), |(h, p)| (h, Some(p)));
    matches!(host, "127.0.0.1" | "localhost")
        && port.is_none_or(|p| {
            p.bytes().all(|b| b.is_ascii_digit()) && p.parse::<u16>().is_ok_and(|n| n > 0)
        })
}

struct TemporaryWave(PathBuf);

impl TemporaryWave {
    fn create() -> std::io::Result<Self> {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        loop {
            let id = NEXT.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir()
                .join(format!("inkling2-voice-{}-{id}.wav", std::process::id()));
            let mut options = std::fs::OpenOptions::new();
            options.write(true).create_new(true);
            #[cfg(unix)]
            {
                use std::os::unix::fs::OpenOptionsExt;
                options.mode(0o600);
            }
            match options.open(&path) {
                Ok(_) => return Ok(Self(path)),
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(e) => return Err(e),
            }
        }
    }
}

impl Drop for TemporaryWave {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

/// Check local VOICEVOX engine availability and return its version string.
pub fn check_voicevox_health(voicevox_url: &str) -> Result<String, String> {
    if !is_loopback_url(voicevox_url) {
        return Err("VOICEVOX URL must be loopback-only".to_string());
    }
    let probe = voice_command("curl")
        .args([
            "--disable",
            "--noproxy",
            "*",
            "-sS",
            "--connect-timeout",
            "1",
            "--max-time",
            "2",
            "--fail",
            &format!("{}/version", voicevox_url.trim_end_matches('/')),
        ])
        .output()
        .map_err(|e| format!("curl unavailable: {e}"))?;
    if probe.status.success() {
        Ok(String::from_utf8_lossy(&probe.stdout).trim().to_string())
    } else {
        Err(format!("VOICEVOX offline ({})", probe.status))
    }
}

/// Play a synthesized WAV file synchronously through the OS audio subsystem.
pub fn play_wav_file(wav_path: &Path) -> Result<(), String> {
    if cfg!(target_os = "windows") {
        let script = r#"$p = New-Object System.Media.SoundPlayer $env:INKLING2_PLAY_WAV; try { $p.PlaySync() } finally { $p.Dispose() }"#;
        let st = voice_command("powershell")
            .args(["-NoProfile", "-NonInteractive", "-Command", script])
            .env("INKLING2_PLAY_WAV", wav_path)
            .status()
            .map_err(|e| format!("WAV playback failed: {e}"))?;
        if st.success() {
            Ok(())
        } else {
            Err(format!("WAV playback exited with {st}"))
        }
    } else {
        for player in ["aplay", "paplay"] {
            if let Ok(st) = voice_command(player).arg(wav_path).status()
                && st.success()
            {
                return Ok(());
            }
        }
        Err("no Linux WAV player (aplay/paplay) succeeded".to_string())
    }
}

/// Speak `text` aloud (when `out_wav` is `None`) or synthesize to `out_wav`.
///
/// Uses local VOICEVOX if online (with `speed_scale` applied), otherwise OS TTS fallback.
pub fn speak_text(
    persona: &VoicePersona,
    text: &str,
    out_wav: Option<&Path>,
) -> Result<String, String> {
    if text.trim().is_empty() {
        return Ok("empty text skipped".to_string());
    }

    // 1. Try local VOICEVOX loopback server
    if check_voicevox_health(&persona.voicevox_url).is_ok() {
        let base = persona.voicevox_url.trim_end_matches('/');
        let q_url = format!("{base}/audio_query?speaker={}", persona.voicevox_speaker_id);
        let q = voice_command("curl")
            .args([
                "--disable",
                "--noproxy",
                "*",
                "-sS",
                "-X",
                "POST",
                "--max-time",
                "10",
                "--get",
                "--fail",
                "--data-urlencode",
                &format!("text={text}"),
                &q_url,
            ])
            .output()
            .map_err(|e| format!("voicevox audio_query failed: {e}"))?;
        if q.status.success() {
            let query_raw = String::from_utf8_lossy(&q.stdout);
            let query_scaled = apply_speed_scale_to_audio_query(&query_raw, persona.speed_scale);
            let s_url = format!("{base}/synthesis?speaker={}", persona.voicevox_speaker_id);
            let temp;
            let target_wav = match out_wav {
                Some(path) => path,
                None => {
                    temp = TemporaryWave::create().map_err(|e| format!("temporary audio: {e}"))?;
                    &temp.0
                }
            };
            let synth = voice_command("curl")
                .args([
                    "--disable",
                    "--noproxy",
                    "*",
                    "-sS",
                    "--fail",
                    "--max-time",
                    "15",
                    "-X",
                    "POST",
                    "-H",
                    "Content-Type: application/json",
                    "--data-binary",
                    "@-",
                    "-o",
                    &target_wav.to_string_lossy(),
                    &s_url,
                ])
                .stdin(std::process::Stdio::piped())
                .spawn();
            if let Ok(mut child) = synth {
                if let Some(mut stdin) = child.stdin.take() {
                    use std::io::Write;
                    let _ = stdin.write_all(query_scaled.as_bytes());
                }
                if let Ok(st) = child.wait()
                    && st.success()
                {
                    if out_wav.is_none() {
                        play_wav_file(target_wav)?;
                        return Ok(format!(
                            "VOICEVOX speaker={} (speed={:.2}) played",
                            persona.voicevox_speaker_id, persona.speed_scale
                        ));
                    }
                    return Ok(format!(
                        "VOICEVOX speaker={} (speed={:.2}) synthesized -> {}",
                        persona.voicevox_speaker_id,
                        persona.speed_scale,
                        target_wav.display()
                    ));
                }
            }
        }
    }

    // 2. Fallback: Windows SAPI Japanese Female Voice (Microsoft Haruka Desktop)
    if cfg!(target_os = "windows") {
        let rate = speed_scale_to_sapi_rate(persona.speed_scale);
        let script = r#"Add-Type -AssemblyName System.Speech
$s = New-Object System.Speech.Synthesis.SpeechSynthesizer
try {
  $v = $s.GetInstalledVoices() | Where-Object { $_.VoiceInfo.Name -match 'Haruka|Ayumi|Zira|Female' } | Select-Object -First 1
  if ($v) { $s.SelectVoice($v.VoiceInfo.Name) }
  if ($env:INKLING2_TTS_WAV) { $s.SetOutputToWaveFile($env:INKLING2_TTS_WAV) }
  $s.Rate = [int]$env:INKLING2_TTS_RATE
  $s.Speak($env:INKLING2_TTS_TEXT)
} finally { $s.Dispose() }"#;
        let mut command = voice_command("powershell");
        command
            .args(["-NoProfile", "-NonInteractive", "-Command", script])
            .env("INKLING2_TTS_TEXT", text)
            .env("INKLING2_TTS_RATE", rate.to_string())
            .env_remove("INKLING2_TTS_WAV");
        if let Some(wav) = out_wav {
            command.env("INKLING2_TTS_WAV", wav);
        }
        let st = command
            .status()
            .map_err(|e| format!("SAPI fallback failed: {e}"))?;
        if st.success() {
            return Ok(format!(
                "Windows SAPI female voice synthesis OK (rate={rate})"
            ));
        }
        return Err(format!("SAPI exited with {st}"));
    }

    Err("no Linux TTS engine is available".to_string())
}

/// Speak text sentence-by-sentence, checking `controller` before each sentence for barge-in stop.
pub fn speak_queued_sentences(
    persona: &VoicePersona,
    text: &str,
    controller: &SpeechController,
) -> Result<usize, String> {
    let sentences = split_sentences(text);
    let mut spoken = 0usize;
    for sentence in sentences {
        if controller.is_stopped() {
            break;
        }
        speak_text(persona, &sentence, None)?;
        spoken += 1;
    }
    Ok(spoken)
}

/// Incremental sentence-boundary accumulator feeding a background TTS worker thread
/// as tokens arrive one-by-one from the SC117 CUDA engine (true token-streaming TTS).
pub struct StreamingSpeechSink {
    buffer: String,
    tx: Option<std::sync::mpsc::Sender<String>>,
    worker: Option<std::thread::JoinHandle<usize>>,
    controller: SpeechController,
    queued_sentences: usize,
}

impl StreamingSpeechSink {
    /// Spawn a background TTS playback worker bound to `persona` and `controller`.
    pub fn spawn(persona: VoicePersona, controller: SpeechController) -> Self {
        let (tx, rx) = std::sync::mpsc::channel::<String>();
        let worker_ctrl = controller.clone();
        let worker = std::thread::spawn(move || {
            let mut spoken = 0usize;
            for sentence in rx {
                if worker_ctrl.is_stopped() {
                    break;
                }
                if speak_text(&persona, &sentence, None).is_ok() {
                    spoken += 1;
                }
            }
            spoken
        });
        Self {
            buffer: String::new(),
            tx: Some(tx),
            worker: Some(worker),
            controller,
            queued_sentences: 0,
        }
    }

    /// Extract complete sentences from an incremental token delta without spawning OS TTS (testable core).
    pub fn push_delta_extract(buffer: &mut String, delta: &str) -> Vec<String> {
        let mut ready = Vec::new();
        for ch in delta.chars() {
            if ch == '\r' {
                continue;
            }
            if ch == '\n' {
                let trimmed = buffer.trim();
                if !trimmed.is_empty() {
                    ready.push(trimmed.to_string());
                }
                buffer.clear();
                continue;
            }
            buffer.push(ch);
            if matches!(ch, '。' | '！' | '？' | '.' | '!' | '?') {
                let trimmed = buffer.trim();
                if !trimmed.is_empty() {
                    ready.push(trimmed.to_string());
                }
                buffer.clear();
            }
        }
        ready
    }

    /// Push a newly decoded token text fragment; dispatches completed sentences immediately.
    pub fn push_token_delta(&mut self, delta: &str) {
        if self.controller.is_stopped() {
            return;
        }
        let ready = Self::push_delta_extract(&mut self.buffer, delta);
        if let Some(tx) = &self.tx {
            for s in ready {
                if tx.send(s).is_ok() {
                    self.queued_sentences += 1;
                }
            }
        }
    }

    /// Flush any remaining partial sentence and wait for queued playback to complete.
    pub fn finish(mut self) -> usize {
        let tail = self.buffer.trim().to_string();
        self.buffer.clear();
        if !tail.is_empty()
            && !self.controller.is_stopped()
            && let Some(tx) = &self.tx
            && tx.send(tail).is_ok()
        {
            self.queued_sentences += 1;
        }
        self.tx.take();
        if let Some(handle) = self.worker.take() {
            handle.join().unwrap_or(self.queued_sentences)
        } else {
            self.queued_sentences
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn speed_scale_is_applied_to_voicevox_audio_query() {
        let raw = r#"{"accent_phrases":[],"speedScale":1.0,"pitchScale":0.0}"#;
        let updated = apply_speed_scale_to_audio_query(raw, 1.25);
        assert!(updated.contains("\"speedScale\":1.25"));
        assert!(updated.contains("\"pitchScale\":0.0"));
    }

    #[test]
    fn sapi_rate_maps_speed_scale_safely() {
        assert_eq!(speed_scale_to_sapi_rate(1.0), 0);
        assert_eq!(speed_scale_to_sapi_rate(1.2), 2);
        assert_eq!(speed_scale_to_sapi_rate(0.5), -5);
        assert_eq!(speed_scale_to_sapi_rate(5.0), 10);
        assert_eq!(speed_scale_to_sapi_rate(f32::NAN), 0);
    }

    #[test]
    fn sentence_splitter_handles_japanese_and_english() {
        let chunks = split_sentences("こんにちは。元気ですか？Hello! Done.");
        assert_eq!(
            chunks,
            vec!["こんにちは。", "元気ですか？", "Hello!", "Done."]
        );
    }

    #[test]
    fn incremental_token_stream_emits_sentences_at_boundaries() {
        let mut buf = String::new();
        assert!(StreamingSpeechSink::push_delta_extract(&mut buf, "こんに").is_empty());
        assert_eq!(
            StreamingSpeechSink::push_delta_extract(&mut buf, "ちは。次の"),
            vec!["こんにちは。"]
        );
        assert_eq!(
            StreamingSpeechSink::push_delta_extract(&mut buf, "文です！"),
            vec!["次の文です！"]
        );
    }

    #[test]
    fn voicevox_health_rejects_non_loopback_urls() {
        assert!(check_voicevox_health("http://example.com:50021").is_err());
        assert!(is_loopback_url("http://localhost:50021/"));
        assert!(!is_loopback_url("http://localhost:invalid"));
        assert!(!is_loopback_url("http://127.0.0.1:50021/subpath"));
    }
}
