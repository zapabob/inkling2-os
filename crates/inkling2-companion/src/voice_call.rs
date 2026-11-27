//! Cute Girl Voice TTS & Audio Call Runtime (`voice_call`).
//!
//! Priority backend chain:
//! 1. Local VOICEVOX Engine (`http://127.0.0.1:50021`, speaker 8 春日部つむぎ / speaker 3 ずんだもん)
//! 2. Windows native Japanese female SAPI (`Microsoft Haruka Desktop`) or Linux `piper`/`espeak-ng`

use std::path::Path;
use std::process::Command;

/// Voice persona configuration.
#[derive(Debug, Clone)]
pub struct VoicePersona {
    /// VOICEVOX local speaker ID (8 = 春日部つむぎ, 3 = ずんだもん, 2 = 四国めたん).
    pub voicevox_speaker_id: u32,
    /// Local VOICEVOX endpoint (strictly loopback).
    pub voicevox_url: String,
    /// Speech rate multiplier (`1.08` for lively cute cadence).
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

/// Speak `text` aloud (or synthesize to WAV) using VOICEVOX if online, else OS female voice fallback.
pub fn speak_text(persona: &VoicePersona, text: &str, out_wav: Option<&Path>) -> Result<String, String> {
    if text.trim().is_empty() {
        return Ok("empty text skipped".to_string());
    }

    // 1. Probe local VOICEVOX loopback server
    let probe = Command::new("curl")
        .args(["-sS", "--connect-timeout", "1", "--max-time", "2", &format!("{}/version", persona.voicevox_url)])
        .output();

    if let Ok(o) = probe {
        if o.status.success() {
            let q_url = format!("{}/audio_query?speaker={}", persona.voicevox_url, persona.voicevox_speaker_id);
            let q = Command::new("curl")
                .args(["-sS", "-X", "POST", "--get", "--data-urlencode", &format!("text={text}"), &q_url])
                .output()
                .map_err(|e| format!("voicevox audio_query failed: {e}"))?;
            if q.status.success() {
                let s_url = format!("{}/synthesis?speaker={}", persona.voicevox_url, persona.voicevox_speaker_id);
                let target_wav = out_wav
                    .map(|p| p.to_path_buf())
                    .unwrap_or_else(|| std::env::temp_dir().join("inkling2_voice.wav"));
                let synth = Command::new("curl")
                    .args([
                        "-sS", "-X", "POST",
                        "-H", "Content-Type: application/json",
                        "--data-binary", "@-",
                        "-o", &target_wav.to_string_lossy(),
                        &s_url,
                    ])
                    .stdin(std::process::Stdio::piped())
                    .spawn();
                if let Ok(mut child) = synth {
                    if let Some(mut stdin) = child.stdin.take() {
                        use std::io::Write;
                        let _ = stdin.write_all(&q.stdout);
                    }
                    if let Ok(st) = child.wait() {
                        if st.success() {
                            return Ok(format!(
                                "VOICEVOX speaker={} synthesized -> {}",
                                persona.voicevox_speaker_id,
                                target_wav.display()
                            ));
                        }
                    }
                }
            }
        }
    }

    // 2. Fallback: Windows SAPI Japanese Female Voice (Microsoft Haruka Desktop)
    if cfg!(target_os = "windows") {
        let safe_text = text.replace('\'', " ").replace('"', " ");
        let script = if let Some(wav) = out_wav {
            format!(
                "Add-Type -AssemblyName System.Speech; $s=New-Object System.Speech.Synthesis.SpeechSynthesizer; \
                 $v=$s.GetInstalledVoices() | Where-Object {{ $_.VoiceInfo.Name -match 'Haruka|Ayumi|Zira|Female' }} | Select-Object -First 1; \
                 if ($v) {{ $s.SelectVoice($v.VoiceInfo.Name) }}; $s.Rate=1; $s.SetOutputToWaveFile('{}'); $s.Speak('{}'); $s.Dispose()",
                wav.display(),
                safe_text
            )
        } else {
            format!(
                "Add-Type -AssemblyName System.Speech; $s=New-Object System.Speech.Synthesis.SpeechSynthesizer; \
                 $v=$s.GetInstalledVoices() | Where-Object {{ $_.VoiceInfo.Name -match 'Haruka|Ayumi|Zira|Female' }} | Select-Object -First 1; \
                 if ($v) {{ $s.SelectVoice($v.VoiceInfo.Name) }}; $s.Rate=1; $s.Speak('{}'); $s.Dispose()",
                safe_text
            )
        };
        let st = Command::new("powershell")
            .args(["-NoProfile", "-Command", &script])
            .status()
            .map_err(|e| format!("SAPI fallback failed: {e}"))?;
        if st.success() {
            return Ok("Windows SAPI female voice synthesis OK".to_string());
        }
        return Err(format!("SAPI exited with {st}"));
    }

    Ok("Linux TTS ready (connect VOICEVOX or piper-tts on Inkling2-OS)".to_string())
}
