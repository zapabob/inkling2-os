# Hermes and NEON on Windows

Use `http://127.0.0.1:18083/v1` for a local client. Check `GET /health` at the server root for the installed version. Chat Completions and model discovery accept the inference credential when token mode is enabled. NEON routes use console authority. A key-free desktop accepts local connections after the Host and browser-origin checks.

| Method and path | Behaviour |
|---|---|
| `GET /v1/perception/status` | Reports hardware discovery, capture opt-in and the visual pipeline mode. |
| `POST /v1/perception/capture` | Requests one frame. Requires FFmpeg, a camera and `INKLING2_CAMERA_ENABLED=1` in the bridge launch environment. Capture is off by default. |
| `GET /v1/voice/status` | Reports local VOICEVOX availability, speed and queue stop state. |
| `POST /v1/voice/speak` | Speaks the supplied `text` (1–800 characters). |
| `POST /v1/voice/stop` | Requests a stop before the next sentence. |
| `POST /v1/memory/feedback` | Persists explicit `text` with optional `provenance` as an unverified observation. |
| `POST /v1/memory/search` | Returns matching explicit records for `query` with provenance. |
| `GET /v1/memory/stats` | Reports the separate legacy Memory45 store. Neural Memory45 counters are available through learning status. |
| `GET /v1/agent/status` | Lists executable skills and task status. |
| `POST /v1/agent/tasks` | Takes a unique `task_id` and an executable `skill`. Reusing a task ID returns HTTP 409. |
| `POST /v1/agent/tasks/cancel` | Cancels a named task. An empty object stops the scheduler from accepting further tasks. |
| `GET /v1/learning/status` | Reports companion adapter and neural-memory counters. |
| `POST /v1/learning/evaluate` | Reports advisory reward signals. Client assertions cannot approve an adapter or promote weights. |
| `POST /v1/learning/rollback` | Requests rollback of the in-process companion adapter. Native GPU adapters are controlled through the engine protocol. |

```powershell
$api = 'http://127.0.0.1:18083/v1'
Invoke-RestMethod "$api/agent/status"
$task = @{task_id=(New-Guid).ToString(); skill='sys.info'} | ConvertTo-Json
Invoke-RestMethod "$api/agent/tasks" -Method Post -ContentType application/json -Body $task
```

A successful system inspection returns the actual architecture and logical CPU count in its results. The other executable skill, `perception.status`, reports the camera privacy state. A status request does not capture a frame.
