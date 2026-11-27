# Inkling2-OS

Inkling2-OS is a native Linux appliance and Windows console for SC117 Inkling-2-Mini. Inkling2 brokers local inference, public HTTPS documents, RustXNG search and read-only previews of an operator-managed folder. A local Strata client can use the same inference API without an API key.

| Status | Capability |
| --- | --- |
| Implemented | Loopback console, optional scoped credentials, read-only document previews, public HTTPS document browsing, RustXNG search and measured-GPU admission. |
| Verified | Windows console, key-free browser and Finder, scoped API and Strata 0.8.6 provider registration. Windows SQ-10, SQ-15 and SQ-20 generated the expected digit and stopped on the arithmetic fixture. Earlier diagnostic media passed QEMU BIOS and UEFI boot. |
| Experimental | Short text inference using the standalone Windows engine's stdin protocol. The OpenAI-compatible event stream emits the answer once generation finishes. |
| Pending qualification | Physical Linux CUDA inference, 65,536 and 131,072 token contexts, and matched hardware benchmarks. |
| In development | USB image with the Linux CUDA executable, NVIDIA modules and a private model volume. |

## Windows console

Build with `cargo build --release --locked --workspace -j 1`. From a Windows bundle, the launcher selects the packaged executable automatically. In PowerShell 7.4 or later, run:

```powershell
./scripts/start-windows-console.ps1 -EnginePath 'C:/Models/bin/inkling2-engine.exe' -ModelsDir 'H:/Models/SC117-SQ-10' -PreviewDir ./docs -Open
```

The default address is `http://127.0.0.1:18081/`. The launcher defaults to key-free local mode. Use `-UseTokens` on shared Windows machines or when forwarding the port. In that mode, `-CopyToken Console` copies the console credential for **Unlock**.

In Strata, add a custom OpenAI Chat Completions connection with base URL `http://127.0.0.1:18081/v1` and use the exact model ID returned by `GET /v1/models`. Local mode does not require an API key. When the launcher uses `-UseTokens`, configure the **Inference** token. Keep each model’s matching `tokenizer.json` beside it.

Use the CUDA executable that supports `generate-token-ids`. The shell reads the matching model chat template and tokenizer, and the engine accepts the resulting token IDs. Select no reasoning for the native adapter.

### SQ-15 and SQ-20 on Windows

The launcher also scans the installed Strata folder under `%LOCALAPPDATA%/Strata/model-weights/SC117/Inkling-2-Mini`. You can add other model folders with `-AdditionalModelsDirs`. All three SC117 weights use the same root `tokenizer.json` from the model publisher. On this workstation the launcher detects the NVMe, SATA SSD and removable USB by their Windows storage bus. The engine keeps priority experts in VRAM and CPU RAM, then caches cold experts in temporary files of up to 256 MiB per drive. It leaves 2 GiB free per volume and uses the original weights when a cache is unavailable. The installed Start menu entry opens the local GUI.

For a Windows-only release, run `scripts/qualify-native-pair.ps1 -WindowsOnly` with the SQ-10 path in `-Model` and the SQ-15 and SQ-20 paths in `-AdditionalModel`. Use the resulting receipt with `scripts/promote-native-pair.ps1 -Stage Install`. Windows-only release tags have the form `v0.1.1-windows.1`.

### Web search

The image builds RustXNG v0.32.0 from the pinned [GitLab mirror](https://gitlab.com/sahenjp/rustxng-mirror). On Windows, build that pinned revision with `cargo build --locked --release --no-default-features --bin rustxng` and run the resulting executable through the launcher:

```powershell
./scripts/start-rustxng-full.ps1 -Binary 'C:/Tools/rustxng.exe'
./scripts/start-windows-console.ps1 -RustXngPort 29483 -PreviewDir ./docs -Open
```

The launcher validates the upstream `full` profile before starting it. Pass `-Config` for a local RustXNG TOML or YAML override. The appliance reads an optional root-managed `/etc/inkling2/rustxng.toml` and otherwise uses the bundled profile. Enabled web providers receive the search query. Providers that require credentials are available only when explicitly configured. Database engine extensions are omitted from the appliance build.

## Linux ISO

Run `./os-image/build-iso.ps1` on an amd64 build host. The resulting live image boots directly on the target machine without Docker. The live image uses memory-backed state and does not include NVIDIA binaries, model weights, or a persistent disk installer.

The `installed-artifact` build target in `os-image/Dockerfile.iso-builder` produces a separate, size-specific disk bundle. Set `INSTALL_DISK_BYTES` to the target's exact byte capacity. The bundle boots in UEFI from its EFI partition and uses a writable ext4 root. The Windows writer at `scripts/write-installed-usb.ps1` checks the target identity and bundle hashes before replacing the partition table, then reads back every written region. `-CheckOnly` performs the checks without administrator access.

For a private inference USB, `os-image/Dockerfile.cuda-builder` builds the Linux engine, CUDA 13.5 runtime and kernel-matched NVIDIA driver. The `private-disk-tools` target incorporates that runtime and adds a model partition. The writer streams the original `.strata` and `tokenizer.json` directly from their existing drive to the USB, then verifies both files there. Linux mounts that volume read-only for the inference account. This path needs disk space for the OS bundle and build caches only. The weights stay in the private installation.

Both images include Wine for 64-bit Windows command-line programs. At the physical console use Ctrl+Alt+F2 to open the unprivileged Inkling2 terminal. Stage approved executables in `/var/lib/inkling2/wine-apps`, then run `wine-inkling2 run PROGRAM.exe` from the terminal. Wine uses a private prefix and runs without network access. The installed disk persists the prefix and service state under `/var/lib/inkling2`.

## Paired qualification and release

Run `scripts/qualify-native-pair.ps1` from the x64 Visual Studio Developer PowerShell with the engine checkout, original SQ-10 model, CUDA 13.5 toolkit and an ignored output directory. The gate runs the Rust workspace and tokenizer suites, the CUDA tests and real SQ-10 reference check, then the BIOS, UEFI and installed-disk boot tests. It writes a qualification receipt only after a successful run. The receipt binds both source trees and the packaged assets to checksums.

Commit and push each qualified tree on its own branch. `scripts/promote-native-pair.ps1 -Stage Install` checks those pushed revisions and the archive hashes, replaces the local Windows console binaries, and keeps rollback copies. For the dedicated USB, `scripts/prepare-private-usb.ps1` builds and boots the private disk bundle in QEMU. Pass its verification hashes and original model path to `scripts/write-installed-usb.ps1`. `scripts/promote-native-pair.ps1 -Stage Publish` checks the installed Windows binaries and the verified USB write log before publishing the paired releases. The public ISO supports diagnostics. The private USB carries its own CUDA and model assets.

Read the [architecture](docs/ARCHITECTURE.md), [boot process](docs/BOOT_PROCESS.md), [security model](docs/SECURITY.md), [hardware support](docs/HARDWARE_COMPATIBILITY.md), [limitations](docs/KNOWN_LIMITATIONS.md) and [validation record](docs/V0.1.1_VALIDATION.md).

## Licence

Distributed under the MIT Licence. The [standalone Inkling2 CUDA Engine](https://github.com/zapabob/inkling2-cuda-engine) is a separate project.
