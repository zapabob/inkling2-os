# Inkling2 Desktop 1.3.0

Inkling2 Desktop runs a resident SC117 model on your Windows GPU. Open chat, documents, web search and applications from the desktop, or connect Hermes and Strata to the same model. NEON adds console APIs for system inspection, explicit memory, camera status and queued speech.

[Windows console ZIP and checksums](https://github.com/zapabob/inkling2-os/releases/tag/v1.3.0) · [Matching CUDA engine and runtime DLLs](https://github.com/zapabob/inkling2-cuda-engine/releases/tag/v1.3.0) · [Linux diagnostic releases](https://github.com/zapabob/inkling2-os/releases?q=linux)

![Windows desktop with the model and context selectors](docs/media/inkling2-desktop-131k.png)

## Use NEON from a local client

With the desktop running, inspect the available task executors and ask for the host OS, architecture and logical CPU count. Each submitted task needs a fresh ID. Windows listens on loopback and requires no key by default.

```powershell
$api = 'http://127.0.0.1:18083/v1'
Invoke-RestMethod "$api/agent/status"
$task = @{ task_id = (New-Guid).ToString(); skill = 'sys.info' } | ConvertTo-Json
Invoke-RestMethod "$api/agent/tasks" -Method Post -ContentType 'application/json' -Body $task
```

The task service also runs `perception.status`. Camera capture is opt-in through the operator's launch environment and discards the temporary image. The observation describes the capture. SC117 image understanding is outside the supplied model path. Speech stops between sentences. Explicit memory feedback persists as an unverified observation and can be recalled with its source after restart. Learning evaluation is advisory. The CUDA engine supports manually activated native adapters, while the HTTP learning routes operate on separate companion state.

See the [NEON architecture](docs/NEON_ARCHITECTURE.md), [Hermes API and examples](docs/HERMES_INTEGRATION.md) and [Neural Memory45 storage limits](docs/NEURAL_MEMORY45.md).

## Windows console

The desktop also recognises the 16 KiB SQv2 layout used by [Kujira’s CHiPS SQ-20 pack](https://huggingface.co/Kujira/Inkling-2-Mini-Strata-CHiPS-Abliterated). Keep the downloaded `.strata` in place. Pass its directory through `-AdditionalModelsDirs` on `start-windows-console.ps1`, or add the directory to the per-user `INKLING2_ADDITIONAL_MODEL_DIRS` environment variable for desktop launches (separate Windows paths with semicolons). Restart the console and select `Inkling-2-Mini-SQv2-20-CHiPS` from the model menu. The same ID is returned by `/v1/models` and accepted by `/v1/chat/completions`. The pack supplies its chat template and can use the embedded SC117 tokenizer.

Use an RTX 5060 Ti 16 GB or a compatible Blackwell `sm_120a` system with a CUDA 13.5-capable driver, PowerShell 7.4 or newer and original SC117 SQ-10, SQ-15 or SQ-20 `.strata` weights. The desktop embeds the SC117 tokenizer and reads the chat template from the model. The two Windows archives include the shell, CUDA executable and its runtime libraries. Supply the weights separately.

1. Compare each downloaded ZIP to the SHA-256 file in its own release. Extract the desktop ZIP into an empty directory and the engine ZIP into its `bin` subdirectory.
2. From the desktop directory, check both versions. Then open the desktop using your model directory.

```powershell
./bin/inkling2-shell.exe --version
./bin/inkling2-engine.exe --version
./scripts/start-windows-desktop.ps1 -ModelsDir 'C:/Models/SC117'
```

Both commands should print `1.3.0`. To create the Start menu and sign-in shortcuts, run the same desktop command with `-RegisterAtLogon`. Add `-SkipUsbCache` to keep a removable drive available for other work. An existing qualified installation can be upgraded in place using the install stage below.

## Chat, files and web

Open **Inkling2 に話す**, select a model and send a message. **Context → Auto** selects a worker capacity for the conversation. **131,072 tokens** includes the prompt, history and reserved reply. The three SC117 tiers default to 512 output tokens and accept up to 1,024, within that context. The hardware qualification uses short prompts at the maximum capacity. Full-window recall and throughput are separate measurements.

Finder previews files inside the configured directory. The HTTPS reader fetches public pages and removes scripts. For web search, build the pinned RustXNG v0.32.0 revision from the [GitLab mirror](https://gitlab.com/sahenjp/rustxng-mirror) and run `./scripts/start-rustxng-full.ps1 -Binary 'C:/Tools/rustxng.exe'`. The launcher validates the full search profile and accepts a local configuration through `-Config`. Search queries go to enabled providers. The native Tor launcher opens the signed [Tor Browser](https://www.torproject.org/download/) installed under `%LOCALAPPDATA%/Programs/Tor Browser`.

## Hermes and Strata

Choose a custom Chat Completions provider with base URL `http://127.0.0.1:18083/v1`. Select the exact model ID from `GET /v1/models`, for example `Inkling-2-Mini-SQ-15-abliterated`. The key-free desktop listens on loopback. Set reasoning off for the native adapter. For a shared machine or forwarded port, the console launcher supports `-UseTokens` and `-CopyToken Inference`.

The model list advertises context and output limits. Each tool turn can select one declared function. Hermes validates and executes it under its own permissions and sends the result back with its call ID. During longer input processing the streaming endpoint sends keepalives, then delivers the completed response as buffered events.

The desktop preloads SQ-15 by default. One model stays resident, with sequence state reset between requests. The SQ-15 desktop profile caps expert placement at 6,144 MiB VRAM and 2,048 MiB pinned RAM and uses the Windows file cache for remaining expert reads. Actual allocations depend on available memory. Other tiers retain the 2,048/512 MiB desktop caps. The launcher discovers installed Strata weights and can also use `-AdditionalModelsDirs`. The [engine README](https://github.com/zapabob/inkling2-cuda-engine/blob/v1.3.0/README.md) lists cache controls.

```powershell
Invoke-RestMethod http://127.0.0.1:18083/health
Invoke-RestMethod http://127.0.0.1:18083/v1/engine/status
```

The standalone web console starts on port 18081 with `./scripts/start-windows-console.ps1 -ModelsDir 'C:/Models/SC117' -PreviewDir ./docs -Open`. To unload an idle desktop worker, POST an empty JSON object to `/v1/engine/unload`.

## Publishing a Windows pair

Build the Rust workspace with `cargo build --release --locked --workspace`. To qualify an upgrade, run the following in an x64 Visual Studio Developer PowerShell with the CUDA 13.5 toolkit. Set the variables to the engine checkout, three original model paths and a new qualification directory.

```powershell
./scripts/qualify-native-pair.ps1 -WindowsOnly -EngineRepo $engine -Model $sq10 -AdditionalModel @($sq15,$sq20) -CudaRoot $env:CUDA_PATH -OutputDir $qualification
$pair = @{ QualificationDir = $qualification; EngineRepo = $engine; Port = 18083 }
./scripts/promote-native-pair.ps1 @pair -Stage Install -ModelPath $sq10 -Resident -SkipUsbCache
```

Qualification tests CUDA, the tokenizer, one-shot generation and 131,072-context resident generation on all three weights. The receipt binds the sources, binaries and archives by SHA-256. Installation keeps rollback copies and preserves the original directory and desktop shortcuts. Verify the versions, installed chat reply and resident state.

Push the qualified revisions to both main branches and dispatch the OS `release.yml` workflow on main. After OS validation, OS staging and engine CI pass on those revisions, publish with `./scripts/promote-native-pair.ps1 @pair -Stage Publish -Tag v1.3.0`. Each repository publishes checksums for its own assets.

## Linux diagnostic ISO

Run `./os-image/build-iso.ps1` on an amd64 build host. The live ISO boots directly on the target machine without Docker. It uses memory-backed state and provides a terminal with the diagnostic console service. NVIDIA binaries and SC117 weights are supplied separately for private installations.

## Private Linux USB

The appliance signs in the local console account on tty1 once the bridge is ready. The Inkling2 desktop shows the engine state, network route and time in the top bar. The Dock opens Chat, Finder, Web, Tor Browser, Terminal and System Information.

Chat, Finder and Web share the local console at `http://127.0.0.1:8080/` and require no API key. The private build verifies Mozilla's signature and the Firefox ESR archive checksum before installing the console browser. Review Mozilla's terms when Firefox opens for the first time. The console account reaches the local bridge through loopback. Web uses the broker for public HTTPS documents and RustXNG Full for search. Finder previews approved files under `/var/lib/inkling2/files` through the same broker. The AI bridge keeps running when the browser window closes.

The **Tor Browser** launcher opens the native [Tor Project release](https://blog.torproject.org/new-release-tor-browser-1603/) in its own Linux account and uses the upstream default profile. The build verifies the Linux archive against the pinned [Tor Browser Developers key](https://support.torproject.org/tor-browser/getting-started/verifying-tor-browser/). Tor Browser operates independently of Inkling2 Browser and Finder.

On the build host, `os-image/Dockerfile.cuda-builder` builds the Linux engine for CUDA 13.5 `sm_120a` and the NVIDIA modules for the appliance kernel. The `private-disk-tools` target stages the driver and graphical session. The disk bundle uses the target's exact byte capacity and contains a UEFI system partition, a persistent ext4 root and a 48 GiB model partition. The Windows writer requests the matching disk identity and verified bundle hashes before replacing the partition table, and reads back its written regions.

The writer streams the original SQ-10, SQ-15, SQ-20 and their shared `tokenizer.json` directly from other physical drives, then verifies all four files on the USB. The appliance mounts the model partition read-only with the Linux NTFS3 driver and grants read access to the inference account. `-CheckOnly` performs the source checks without administrator access. Use the machine's UEFI boot menu to launch the installed USB.

The private appliance enables resident inference. When the GPU and bundled SQ-10 are present, the bridge starts loading SQ-10 in the background while the desktop opens. Selecting another model switches the resident model.

Both Linux images include Wine for 64-bit Windows command-line programs. At the physical console use Ctrl+Alt+F2 to open the unprivileged Inkling2 terminal. Stage approved executables in `/var/lib/inkling2/wine-apps`, then run `wine-inkling2 run PROGRAM.exe` from the terminal. Wine uses a separate display, an isolated network namespace and its own prefix. The installed disk persists the prefix and service state in `/var/lib/inkling2`.

See the [architecture](docs/ARCHITECTURE.md), [boot process](docs/BOOT_PROCESS.md), [security model](docs/SECURITY.md) and [hardware support](docs/HARDWARE_COMPATIBILITY.md). Distributed under the [MIT licence](LICENSE).
