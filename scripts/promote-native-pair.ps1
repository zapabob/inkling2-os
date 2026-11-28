[CmdletBinding()]
param(
    [Parameter(Mandatory)][ValidateSet('Install','Publish')][string]$Stage,
    [Parameter(Mandatory)][string]$QualificationDir,
    [Parameter(Mandatory)][string]$EngineRepo,
    [string]$ModelPath,
    [ValidateRange(1024,65535)][int]$Port = 18081,
    [ValidateRange(0,65535)][int]$RustXngPort = 29483,
    [switch]$Resident,
    [string]$PreloadModel = 'Inkling-2-Mini-SQ-15-abliterated',
    [switch]$SkipUsbCache,
    [string]$UsbWriteLog,
    [string]$UsbVerification,
    [ValidatePattern('^v\d+\.\d+\.\d+(?:-[A-Za-z0-9.-]+)?$')][string]$Tag
)

$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $true
. (Join-Path $PSScriptRoot 'release-source-digest.ps1')

$qualification = (Resolve-Path -LiteralPath $QualificationDir).Path
$receipt = Get-Content -LiteralPath (Join-Path $qualification 'qualification.json') -Raw | ConvertFrom-Json
$version = if ($receipt.version) { $receipt.version } else { '0.1.1' }
if ($version -notmatch '^\d+\.\d+\.\d+$') { throw 'Invalid qualified release version.' }
if (!$Tag) { $Tag = "v$version" }
$osRepo = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$engineSource = (Resolve-Path -LiteralPath $EngineRepo).Path
# Upgrade the original installation in place so its shortcuts and per-user state still resolve.
$installation = Join-Path $env:LOCALAPPDATA 'Inkling2-OS/v0.1.1/native/bin'
$shell = Join-Path $installation 'inkling2-shell.exe'
$engine = Join-Path $installation 'inkling2-engine.exe'
$assets = Join-Path $qualification 'assets'

function Get-PushedRevision([string]$Repo, [string]$Slug, [string]$SourceDigest, [string]$Scope = 'All') {
    $scopePaths = @(Get-ReleaseSourcePaths $Scope)
    if ((git -C $Repo status --porcelain=v1 --untracked-files=all -- @scopePaths) -or
        (Get-ReleaseSourceDigest $Repo $Scope) -ne $SourceDigest) {
        throw "Commit the qualified source tree with no pending changes for $Slug."
    }
    $remote = git -C $Repo remote get-url origin
    if ($remote -notin "https://github.com/$Slug", "https://github.com/${Slug}.git", "git@github.com:${Slug}.git") {
        throw "Unexpected origin for $Slug."
    }
    $sha = git -C $Repo rev-parse HEAD
    $published = git -C $Repo ls-remote --exit-code origin refs/heads/main
    if (($published -split '\s+', 2)[0] -ne $sha) { throw "Push the qualified commit to main for $Slug." }
    $sha
}

function Assert-GreenWorkflow([string]$Slug, [string]$Sha, [string]$Workflow, [string]$Event) {
    $runs = gh run list --repo $Slug --workflow $Workflow --branch main --commit $Sha --event $Event --limit 1 --json status,conclusion,url
    $latest = $runs | ConvertFrom-Json | Select-Object -First 1
    if ($latest.status -ne 'completed' -or $latest.conclusion -ne 'success') {
        throw "A successful $Workflow run ($Event) on main at $Sha is required for $Slug. $($latest.url)"
    }
}

if ($receipt.schema -ne 1 -or @($receipt.additional_models).Count -ne 2 -or $receipt.model_sha256 -notmatch '^[0-9A-Fa-f]{64}$') {
    throw 'Invalid paired qualification receipt.'
}
$windowsOnly = $receipt.platform -eq 'windows'
$osScope = if ($windowsOnly) { 'WindowsOS' } else { 'All' }
if ($Stage -eq 'Publish') {
    $osCommit = Get-PushedRevision $osRepo 'zapabob/inkling2-os' $receipt.sources.os $osScope
    $engineCommit = Get-PushedRevision $engineSource 'zapabob/inkling2-cuda-engine' $receipt.sources.engine
    Assert-GreenWorkflow 'zapabob/inkling2-cuda-engine' $engineCommit 'release.yml' 'push'
    Assert-GreenWorkflow 'zapabob/inkling2-os' $osCommit 'validation.yml' 'push'
    Assert-GreenWorkflow 'zapabob/inkling2-os' $osCommit 'release.yml' 'workflow_dispatch'
} elseif ((Get-ReleaseSourceDigest $engineSource) -ne $receipt.sources.engine -or
          (Get-ReleaseSourceDigest $osRepo $osScope) -ne $receipt.sources.os) {
    throw 'Installation inputs differ from the tested source.'
}
$files = @{}
foreach ($asset in $receipt.assets) {
    if ($asset.file -notmatch '^[A-Za-z0-9][A-Za-z0-9._-]+$' -or $files.ContainsKey($asset.file)) {
        throw 'Invalid release asset name.'
    }
    $path = Join-Path $assets $asset.file
    if ((Get-FileHash -LiteralPath $path).Hash -ne $asset.sha256) { throw "Changed asset $($asset.file)." }
    $files[$asset.file] = $path
}
$consoleZip = $files["inkling2-os-v$version-win64-console.zip"]
$engineZip = $files['inkling2-cuda-engine-win64-sm120a.zip']
$iso = $files['inkling2-os-v1.1.0-x86_64-sm120.iso']
if (-not $consoleZip -or -not $engineZip -or (-not $windowsOnly -and -not $iso)) { throw 'The paired release assets are incomplete.' }

if ($Stage -eq 'Install') {
    $source = Get-Item -LiteralPath $ModelPath
    $tokenizer = Join-Path $osRepo 'crates/inkling2-shell/assets/sc117/tokenizer.json'
    if ((Get-FileHash -LiteralPath $source.FullName).Hash -ne $receipt.model_sha256 -or
        (Get-FileHash -LiteralPath $tokenizer).Hash -ne $receipt.tokenizer_sha256) {
        throw 'The model and tokenizer changed after qualification.'
    }
    $stageDir = (New-Item -ItemType Directory -Path (Join-Path $qualification ('install-' + (New-Guid).ToString('N')))).FullName
    Expand-Archive -LiteralPath $consoleZip -DestinationPath (Join-Path $stageDir 'console')
    Expand-Archive -LiteralPath $engineZip -DestinationPath (Join-Path $stageDir 'engine')
    $candidate = @{
        'inkling2-shell.exe' = Join-Path $stageDir 'console/bin/inkling2-shell.exe'
        'inkling2-engine.exe' = Join-Path $stageDir 'engine/inkling2-engine.exe'
        'cudart64_13.dll' = Join-Path $stageDir 'engine/cudart64_13.dll'
        'libzstd.dll' = Join-Path $stageDir 'engine/libzstd.dll'
        'LICENSE-zstd.txt' = Join-Path $stageDir 'engine/LICENSE-zstd.txt'
    }
    if ((Get-FileHash -LiteralPath $candidate['inkling2-shell.exe']).Hash -ne $receipt.binaries.shell -or
        (Get-FileHash -LiteralPath $candidate['inkling2-engine.exe']).Hash -ne $receipt.binaries.engine) {
        throw 'The extracted binaries do not match the tested executables.'
    }

    $candidate['../scripts/start-windows-console.ps1'] = Join-Path $stageDir 'console/scripts/start-windows-console.ps1'
    $candidate['../scripts/start-windows-desktop.ps1'] = Join-Path $stageDir 'console/scripts/start-windows-desktop.ps1'
    $candidate['../scripts/start-rustxng-full.ps1'] = Join-Path $stageDir 'console/scripts/start-rustxng-full.ps1'
    $listener = @(Get-NetTCPConnection -State Listen -LocalPort $Port -ErrorAction SilentlyContinue)
    if ($listener.Count) {
        $service = Get-CimInstance Win32_Process -Filter "ProcessId=$($listener[0].OwningProcess)"
        if ($listener.Count -ne 1 -or $listener[0].LocalAddress -ne '127.0.0.1' -or
            $service.ExecutablePath -ne $shell -or
            (Invoke-RestMethod -Uri "http://127.0.0.1:$Port/health" -TimeoutSec 3).console_mode -ne 'local') {
            throw 'Port is in use by a different service or authentication mode.'
        }
        if (Get-CimInstance Win32_Process -Filter "ParentProcessId=$($service.ProcessId) AND Name='inkling2-engine.exe'") {
            throw 'The installed console is currently processing a model request.'
        }
    }

    $backup = (New-Item -ItemType Directory -Path (Join-Path $stageDir 'backup')).FullName
    New-Item -ItemType Directory -Path $installation -Force | Out-Null
    New-Item -ItemType Directory -Path (Join-Path $installation '../scripts') -Force | Out-Null
    foreach ($name in $candidate.Keys) {
        $old = Join-Path $installation $name
        if (Test-Path -LiteralPath $old -PathType Leaf) { Copy-Item -LiteralPath $old -Destination $backup }
    }
    $launch = @{
        Port = $Port; ShellPath = $shell; EnginePath = $engine; ModelsDir = $source.DirectoryName
        PreviewDir = Join-Path $osRepo 'docs'; RustXngPort = $RustXngPort
        Resident = $Resident; PreloadModel = $PreloadModel; SkipUsbCache = $SkipUsbCache
    }
    if ($listener.Count) {
        $previousBridge = Get-Process -Id $service.ProcessId
        Stop-Process -InputObject $previousBridge
        if (!$previousBridge.WaitForExit(15000)) { throw 'The previous bridge is still releasing its executable.' }
    }
    try {
        foreach ($name in $candidate.Keys) { Copy-Item -LiteralPath $candidate[$name] -Destination (Join-Path $installation $name) -Force }
        & (Join-Path $PSScriptRoot 'start-windows-console.ps1') @launch
    } catch {
        $running = Get-NetTCPConnection -State Listen -LocalPort $Port -ErrorAction SilentlyContinue
        if ($running) {
            $process = Get-CimInstance Win32_Process -Filter "ProcessId=$($running.OwningProcess)"
            if ($process.ExecutablePath -eq $shell) { Stop-Process -Id $process.ProcessId }
        }
        foreach ($name in $candidate.Keys) {
            $old = Join-Path $backup (Split-Path $name -Leaf)
            $destination = Join-Path $installation $name
            if (Test-Path -LiteralPath $old -PathType Leaf) {
                Copy-Item -LiteralPath $old -Destination $destination -Force
            } elseif (Test-Path -LiteralPath $destination -PathType Leaf) {
                Remove-Item -LiteralPath $destination
            }
        }
        if ($listener.Count) { & (Join-Path $PSScriptRoot 'start-windows-console.ps1') @launch }
        throw
    }
    $menu = [Environment]::GetFolderPath('Programs')
    $shortcut = (New-Object -ComObject WScript.Shell).CreateShortcut((Join-Path $menu 'Inkling2-OS.lnk'))
    $shortcut.TargetPath = (Get-Command pwsh.exe).Source
    $launcher = (Resolve-Path (Join-Path $installation '../scripts/start-windows-console.ps1')).Path
    $shortcut.Arguments = '-NoProfile -WindowStyle Hidden -File "' + $launcher + '" -Open -Port ' + $Port + ' -ModelsDir "' + $source.DirectoryName + '"'
    $shortcut.WorkingDirectory = Split-Path $installation
    $shortcut.WindowStyle = 7
    $shortcut.Save()
    Write-Host "Qualified Windows pair installed. Rollback copies are in $backup."
    return
}

if ($windowsOnly) {
    $validTag = $Tag -eq "v$version" -or $Tag -match ('^v' + [regex]::Escape($version) + '-windows\.[0-9]+$')
    if (-not $validTag -or
        (Get-FileHash -LiteralPath $shell).Hash -ne $receipt.binaries.shell -or
        (Get-FileHash -LiteralPath $engine).Hash -ne $receipt.binaries.engine -or
        (Invoke-RestMethod -Uri "http://127.0.0.1:$Port/health" -TimeoutSec 3).service -ne 'inkling2-os-bridge') {
        throw 'The Windows release requires a Windows tag and the tested local installation.'
    }
} else {
    $usbReceipt = Get-Content -LiteralPath $UsbVerification -Raw | ConvertFrom-Json
    $usbEvidence = Get-Content -LiteralPath $UsbWriteLog -Raw
    foreach ($entry in $receipt.additional_models) {
        if ($entry.sha256 -notmatch '^[0-9a-f]{64}$' -or
            @($usbReceipt.additional_models | Where-Object { $_.name -eq $entry.name -and $_.sha256 -eq $entry.sha256 }).Count -ne 1 -or
            $usbEvidence -notmatch "(?m)^USB_EXTRA_MODEL_SHA256=$($entry.sha256)\r?$") {
            throw "USB readback is missing for $($entry.name)."
        }
    }
    if ($usbEvidence -notmatch '(?m)^INKLING2_USB_VERIFIED\r?$' -or
        $usbReceipt.model_sha256 -ne $receipt.model_sha256 -or
        $usbReceipt.layout_sha256 -notmatch '^[0-9A-Fa-f]{64}$' -or
        $usbEvidence -notmatch "(?m)^USB_LAYOUT_SHA256=$($usbReceipt.layout_sha256)\r?$" -or
        $usbEvidence -notmatch "(?m)^USB_MODEL_SHA256=$($receipt.model_sha256)\r?$" -or
        (Get-FileHash -LiteralPath $shell).Hash -ne $receipt.binaries.shell -or
        (Get-FileHash -LiteralPath $engine).Hash -ne $receipt.binaries.engine -or
        (Invoke-RestMethod -Uri "http://127.0.0.1:$Port/health" -TimeoutSec 3).service -ne 'inkling2-os-bridge') {
        throw 'Verify the paired Windows installation and the USB model readback before publishing.'
    }
}

$osAssets = @($consoleZip)
if (-not $windowsOnly) { $osAssets += $iso }
$osAssets += @($files.GetEnumerator() | Where-Object { $_.Key -like 'SBOM-*' } | ForEach-Object Value)
$engineSums = Join-Path $qualification 'SHA256SUMS-engine.txt'
"$($receipt.assets.Where({ $_.file -eq (Split-Path $engineZip -Leaf) })[0].sha256.ToLowerInvariant())  $(Split-Path $engineZip -Leaf)" |
    Set-Content -LiteralPath $engineSums -Encoding ascii
$osSums = Join-Path $qualification 'SHA256SUMS-os.txt'
$osAssets | ForEach-Object { "$((Get-FileHash -LiteralPath $_).Hash.ToLowerInvariant())  $(Split-Path $_ -Leaf)" } |
    Set-Content -LiteralPath $osSums -Encoding ascii

foreach ($repository in $engineSource, $osRepo) {
    if (git -C $repository ls-remote --tags origin "refs/tags/$Tag") {
        throw "The tag $Tag already exists at origin. Choose a new release tag."
    }
}

$osNotes = if ($windowsOnly) { 'Windows console for the SC117 SQ-10, SQ-15 and SQ-20 native CUDA engine.' } else { 'Qualified Windows console and Linux diagnostic ISO.' }
gh release create $Tag $engineZip $engineSums --repo zapabob/inkling2-cuda-engine --target $engineCommit --draft `
    --title "Inkling2 CUDA Engine $Tag" --notes 'Qualified Windows Blackwell CUDA 13.5 engine and runtime.'
gh release create $Tag @osAssets $osSums --repo zapabob/inkling2-os --target $osCommit --draft `
    --title "Inkling2-OS $Tag" --notes $osNotes
gh release edit $Tag --repo zapabob/inkling2-cuda-engine --draft=false
gh release edit $Tag --repo zapabob/inkling2-os --draft=false
