[CmdletBinding()]
param(
    [Parameter(Mandatory)][string]$EngineRepo,
    [Parameter(Mandatory)][string]$Model,
    [Parameter(Mandatory)][ValidateCount(2,2)][string[]]$AdditionalModel,
    [Parameter(Mandatory)][string]$CudaRoot,
    [Parameter(Mandatory)][string]$OutputDir,
    [string]$WindowsBuildDir,
    [switch]$WindowsOnly,
    [ValidateRange(8589934592, 4000000000000)][long]$UsbDiskBytes = 123942731776
)

$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $true
$osRepo = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$engineSource = (Resolve-Path -LiteralPath $EngineRepo).Path
$modelPath = (Resolve-Path -LiteralPath $Model).Path
$tokenizer = (Resolve-Path -LiteralPath (Join-Path $osRepo 'crates/inkling2-shell/assets/sc117/tokenizer.json')).Path
$additional = @($AdditionalModel | ForEach-Object { Get-Item -LiteralPath $_ } | Sort-Object Name)
$expectedNames = @('Inkling-2-Mini-SQ-15-abliterated.strata','Inkling-2-Mini-SQ-20-abliterated.strata')
if ((Split-Path $modelPath -Leaf) -ne 'Inkling-2-Mini-SQ-10-abliterated.strata' -or
    (Compare-Object $expectedNames $additional.Name)) { throw 'Expected SC117 SQ-10, SQ-15 and SQ-20.' }
$cuda = (Resolve-Path -LiteralPath $CudaRoot).Path
$compiler = (Resolve-Path -LiteralPath (Join-Path $cuda 'bin/nvcc.exe')).Path
$root = (New-Item -ItemType Directory -Path $OutputDir).FullName
$windowsBuild = if ($WindowsBuildDir) { $WindowsBuildDir } else { Join-Path $root 'engine-build' }
$osBuild = if ($env:CARGO_TARGET_DIR) { $env:CARGO_TARGET_DIR } else { Join-Path $osRepo 'target' }
$assets = (New-Item -ItemType Directory -Path (Join-Path $root 'assets')).FullName
$bootImage = "inkling2-os-qualification:$((New-Guid).ToString('N'))"

. (Join-Path $PSScriptRoot 'release-source-digest.ps1')

$engineDigest = Get-ReleaseSourceDigest $engineSource
$osScope = if ($WindowsOnly) { 'WindowsOS' } else { 'All' }
$osDigest = Get-ReleaseSourceDigest $osRepo $osScope
$toolEnv = @{
    CUDACXX = $env:CUDACXX; CUDA_PATH = $env:CUDA_PATH; Path = $env:Path
    INKLING2_TEST_MODEL = $env:INKLING2_TEST_MODEL; CARGO_TARGET_DIR = $env:CARGO_TARGET_DIR
    RUSTC_WRAPPER = $env:RUSTC_WRAPPER; RUSTC_WORKSPACE_WRAPPER = $env:RUSTC_WORKSPACE_WRAPPER
    VSLANG = $env:VSLANG
}

Push-Location -LiteralPath $osRepo
try {
    $env:CUDACXX = $compiler
    $env:CUDA_PATH = $cuda
    $env:VSLANG = '1033'
    $env:Path = (Join-Path $cuda 'bin') + [IO.Path]::PathSeparator + $env:Path
    $env:CARGO_TARGET_DIR = $osBuild
    $env:INKLING2_TEST_MODEL = $modelPath
    foreach ($name in 'RUSTC_WRAPPER','RUSTC_WORKSPACE_WRAPPER') {
        $wrapper = [Environment]::GetEnvironmentVariable($name, 'Process')
        if ($wrapper -and -not (Get-Command $wrapper -ErrorAction SilentlyContinue)) {
            [Environment]::SetEnvironmentVariable($name, '', 'Process')
        }
    }
    if (-not (Get-Command cl.exe -ErrorAction SilentlyContinue)) {
        throw 'Start the Visual Studio x64 Developer PowerShell before qualifying the CUDA engine.'
    }

    cargo fmt --all -- --check
    cargo clippy --locked --workspace --all-targets -- -D warnings
    cargo test --locked --workspace
    cargo test --locked -p inkling2-shell native_prompt::tests::matches_reference_token_ids -- --ignored --exact

    cmake -S $engineSource -B $windowsBuild -G Ninja -DCMAKE_BUILD_TYPE=Release `
        "-DCUDAToolkit_ROOT=$cuda" "-DINKLING2_RELEASE_MODEL=$modelPath"
    cmake --build $windowsBuild --parallel 2
    & (Join-Path $PSScriptRoot 'stage-zstd-runtime.ps1') -Directory $windowsBuild
    ctest --test-dir $windowsBuild --output-on-failure --no-tests=error

    foreach ($weight in $additional) {
        $env:INKLING2_TEST_MODEL = $weight.FullName
        cargo test --locked -p inkling2-shell native_prompt::tests::matches_reference_token_ids -- --ignored --exact
        $python = (Get-Content -LiteralPath (Join-Path $windowsBuild 'CMakeCache.txt') |
            Select-String '^_Python3_EXECUTABLE:INTERNAL=(.+)$').Matches[0].Groups[1].Value
        & $python (Join-Path $engineSource 'tests/sq10_release.py') (Join-Path $windowsBuild 'inkling2-engine.exe') $weight.FullName
    }

    if (-not $WindowsOnly) {
        docker build --file os-image/Dockerfile.iso-builder --target installed-verification `
            --build-arg "INSTALL_DISK_BYTES=$UsbDiskBytes" --tag $bootImage .
        docker run --rm --network none --tmpfs '/tmp:rw,nosuid,nodev,size=3500m' $bootImage --firmware bios --firmware uefi
        docker run --rm --network none --tmpfs '/tmp:rw,nosuid,nodev,size=3500m' $bootImage --installed-bundle /out/install
    }
    if ((Get-ReleaseSourceDigest $engineSource) -ne $engineDigest -or (Get-ReleaseSourceDigest $osRepo $osScope) -ne $osDigest) {
        throw 'Source files changed during qualification. Build and test the new candidate.'
    }
    cargo build --locked --workspace --release
    if (-not $WindowsOnly) {
        $container = docker create $bootImage
        try {
            docker cp "${container}:/out/inkling2-os-v0.1.1-x86_64-sm120.iso" $assets
            docker cp "${container}:/out/linux-rootfs-packages.tsv" $assets
        } finally {
            docker rm $container | Out-Null
        }
    }
    & (Join-Path $PSScriptRoot 'package-windows-console.ps1') -OutDir $assets -BinaryDir (Join-Path $osBuild 'release')
    $engineBinary = Join-Path $windowsBuild 'inkling2-engine.exe'
    $runtime = (Resolve-Path -LiteralPath (Join-Path $cuda 'bin/cudart64_13.dll')).Path
    $engineArchive = Join-Path $assets 'inkling2-cuda-engine-win64-sm120a.zip'
    Compress-Archive -LiteralPath $engineBinary,$runtime,(Join-Path $windowsBuild 'libzstd.dll'),(Join-Path $windowsBuild 'LICENSE-zstd.txt'),(Join-Path $engineSource 'LICENSE'),(Join-Path $engineSource 'README.md') -DestinationPath $engineArchive
    if ((Get-ReleaseSourceDigest $engineSource) -ne $engineDigest -or (Get-ReleaseSourceDigest $osRepo $osScope) -ne $osDigest) {
        throw 'Source files changed during packaging.'
    }
    $verifiedAssets = Get-ChildItem -LiteralPath $assets -File |
        Where-Object Extension -In '.iso','.zip','.txt','.spdx','.json','.tsv' |
        Sort-Object Name | ForEach-Object { @{ file = $_.Name; sha256 = (Get-FileHash -LiteralPath $_.FullName).Hash } }
    @{
        schema = 1; generated_utc = [DateTime]::UtcNow.ToString('o')
        sources = @{ engine = $engineDigest; os = $osDigest }
        binaries = @{ engine = (Get-FileHash -LiteralPath $engineBinary).Hash
                       shell = (Get-FileHash -LiteralPath (Join-Path $osBuild 'release/inkling2-shell.exe')).Hash }
        platform = if ($WindowsOnly) { 'windows' } else { 'paired' }
        boot_verifier = if (-not $WindowsOnly) { docker image inspect $bootImage --format '{{.Id}}' }
        model_sha256 = (Get-FileHash -LiteralPath $modelPath).Hash
        additional_models = @($additional | ForEach-Object { @{name=$_.Name; sha256=(Get-FileHash -LiteralPath $_.FullName).Hash} })
        tokenizer_sha256 = (Get-FileHash -LiteralPath $tokenizer).Hash
        assets = @($verifiedAssets)
    } | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $root 'qualification.json') -Encoding utf8
    Write-Host "Both suites passed. Verified assets and their receipt are in $root."
} finally {
    foreach ($entry in $toolEnv.GetEnumerator()) {
        [Environment]::SetEnvironmentVariable($entry.Key, $entry.Value, 'Process')
    }
    Pop-Location
}
