[CmdletBinding()]
param(
    [ValidateRange(1024, 65535)]
    [int]$Port = 18081,
    [string]$ShellPath,
    [string]$EnginePath = $env:INKLING2_ENGINE_BIN,
    [string]$ModelsDir = $env:INKLING2_MODELS_DIR,
    [string[]]$AdditionalModelsDirs,
    [string]$PreviewDir = $env:INKLING2_PREVIEW_DIR,
    [ValidateRange(0, 65535)]
    [int]$RustXngPort = 0,
    [switch]$Open,
    [switch]$UseTokens,
    [switch]$Resident,
    [string]$PreloadModel,
    [switch]$SkipUsbCache,
    [ValidateSet('Console', 'Inference')]
    [string]$CopyToken
)

$ErrorActionPreference = 'Stop'
$root = Join-Path $env:LOCALAPPDATA 'Inkling2-OS/v0.1.1'

function Get-LocalToken([string]$Name, [bool]$Provision = $false) {
    $path = Join-Path $root "$Name.dpapi"
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        if (-not $Provision) { throw "$Name token has not been provisioned. Start the console first." }
        $rng = [Security.Cryptography.RandomNumberGenerator]::Create()
        try {
            $bytes = [byte[]]::new(32)
            $rng.GetBytes($bytes)
            $secret = ConvertTo-SecureString ([Convert]::ToHexString($bytes)) -AsPlainText -Force
            $protected = ConvertFrom-SecureString $secret
            $temporary = Join-Path $root ([IO.Path]::GetRandomFileName())
            [IO.File]::WriteAllText($temporary, $protected)
            try { [IO.File]::Move($temporary, $path) }
            catch [IO.IOException] {
                if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { throw }
                Remove-Item -LiteralPath $temporary
            }
        } finally { $rng.Dispose() }
    }
    $loaded = ConvertTo-SecureString (Get-Content -LiteralPath $path -Raw)
    [Net.NetworkCredential]::new('', $loaded).Password
}

New-Item -ItemType Directory -Force -Path $root | Out-Null
if ($CopyToken) {
    Get-LocalToken $CopyToken | Set-Clipboard
    Write-Host "$CopyToken token copied to the clipboard."
    return
}

if (-not $ShellPath) {
    $packaged = Join-Path $PSScriptRoot '../bin/inkling2-shell.exe'
    $ShellPath = if (Test-Path -LiteralPath $packaged) { $packaged } else {
        Join-Path $PSScriptRoot '../target/release/inkling2-shell.exe'
    }
}
$executable = (Resolve-Path -LiteralPath $ShellPath).Path
$packagedEngine = Join-Path $PSScriptRoot '../bin/inkling2-engine.exe'
if (-not $EnginePath -and (Test-Path -LiteralPath $packagedEngine -PathType Leaf)) {
    $EnginePath = (Resolve-Path -LiteralPath $packagedEngine).Path
}
$url = "http://127.0.0.1:$Port/"
$listener = @(Get-NetTCPConnection -State Listen -LocalPort $Port -ErrorAction SilentlyContinue)
if ($listener.Count) {
    if ($Open -and $listener.Count -eq 1 -and $listener[0].LocalAddress -eq '127.0.0.1' -and
        (Get-Process -Id $listener[0].OwningProcess).Path -eq $executable -and
        (Invoke-RestMethod -Uri ($url + 'health') -TimeoutSec 3).service -eq 'inkling2-os-bridge') {
        Start-Process $url
        return
    }
    throw "TCP port $Port is already in use."
}
$installedSearch = Join-Path $PSScriptRoot '../rustxng/rustxng.exe'
$searchConfig = Join-Path $PSScriptRoot '../rustxng/settings.full.toml'
if (-not $RustXngPort -and (Test-Path -LiteralPath $installedSearch -PathType Leaf)) {
    $RustXngPort = 29483
    $existing = @(Get-NetTCPConnection -State Listen -LocalPort $RustXngPort -ErrorAction SilentlyContinue)
    if (-not $existing.Count) {
        & (Join-Path $PSScriptRoot 'start-rustxng-full.ps1') -Binary $installedSearch -Config $searchConfig -Port $RustXngPort
    } elseif ($existing.Count -ne 1 -or $existing[0].LocalAddress -ne '127.0.0.1' -or
        (Get-Process -Id $existing[0].OwningProcess).Path -ne (Resolve-Path $installedSearch).Path) {
        throw "The configured search port belongs to a different process."
    }
}
if (-not $PreviewDir) {
    foreach ($folder in '../documents','../docs') {
        $packagedDocs = Join-Path $PSScriptRoot $folder
        if (Test-Path -LiteralPath $packagedDocs -PathType Container) { $PreviewDir = $packagedDocs; break }
    }
}
$state = if ($UseTokens) { $root } else { Join-Path $root 'local-state' }
$settings = @{
    INKLING2_LOCAL_CONSOLE = if ($UseTokens) { '0' } else { '1' }
    INKLING2_BRIDGE_TOKEN = if ($UseTokens) { Get-LocalToken Console $true } else { $null }
    INKLING2_INFERENCE_TOKEN = if ($UseTokens) { Get-LocalToken Inference $true } else { $null }
    INKLING2_RUSTXNG_PORT = if ($RustXngPort -ne 0) { [string]$RustXngPort } else { $null }
    INKLING2_STATE_DIR = $state
    INKLING2_ENGINE_BIN = if ($EnginePath) { (Resolve-Path -LiteralPath $EnginePath).Path } else { $null }
    INKLING2_MODELS_DIR = if ($ModelsDir) { (Resolve-Path -LiteralPath $ModelsDir).Path } else { $null }
    INKLING2_PREVIEW_DIR = if ($PreviewDir) { (Resolve-Path -LiteralPath $PreviewDir).Path } else { $null }
}
if ($Resident) {
    $settings.INKLING2_RESIDENT = '1'
    if ($PreloadModel) { $settings.INKLING2_PRELOAD_MODEL = $PreloadModel }
    if (-not $env:INKLING2_EXPERT_VRAM_MIB) { $settings.INKLING2_EXPERT_VRAM_MIB = '2048' }
    if (-not $env:INKLING2_EXPERT_RAM_MIB) { $settings.INKLING2_EXPERT_RAM_MIB = '512' }
}
if ($SkipUsbCache) { $settings.INKLING2_EXPERT_USB_DIR = $null }
$strataModels = Join-Path $env:LOCALAPPDATA 'Strata/model-weights/SC117/Inkling-2-Mini'
if (-not $AdditionalModelsDirs -and (Test-Path -LiteralPath $strataModels -PathType Container)) {
    $AdditionalModelsDirs = @($strataModels)
}
if ($AdditionalModelsDirs) {
    $settings.INKLING2_ADDITIONAL_MODEL_DIRS = ($AdditionalModelsDirs | ForEach-Object {
        (Resolve-Path -LiteralPath $_).Path
    }) -join [IO.Path]::PathSeparator
}
foreach ($physical in (Get-PhysicalDisk -ErrorAction SilentlyContinue)) {
    $medium = switch ([string]$physical.BusType) {
        'NVMe' { 'NVME' }
        'SATA' { if ($physical.MediaType -eq 'SSD') { 'SATA' } }
        'USB'  { 'USB' }
    }
    if (-not $medium) { continue }
    $key = "INKLING2_EXPERT_${medium}_DIR"
    if ($settings.ContainsKey($key) -or [Environment]::GetEnvironmentVariable($key)) { continue }
    $disk = Get-Disk -Number $physical.DeviceId -ErrorAction SilentlyContinue
    if (-not $disk -or $disk.FriendlyName -ne $physical.FriendlyName -or $disk.BusType -ne $physical.BusType -or $disk.IsReadOnly -or $disk.IsOffline) { continue }
    foreach ($volume in ($disk | Get-Partition | Get-Volume -ErrorAction SilentlyContinue)) {
        if (-not $volume.DriveLetter -or $volume.SizeRemaining -le 2GB -or $volume.FileSystem -notin 'NTFS','exFAT') { continue }
        $drive = "$($volume.DriveLetter):"
        $base = if ($drive -eq $env:SystemDrive) { $env:LOCALAPPDATA } else { "$drive\" }
        $path = Join-Path $base 'Inkling2-Experts'
        try { $settings[$key] = (New-Item -ItemType Directory -Force -Path $path -ErrorAction Stop).FullName }
        catch { continue }
        break
    }
}
$bridge = Start-Process -FilePath $executable -ArgumentList 'serve', "127.0.0.1:$Port" `
    -WorkingDirectory $PSScriptRoot -WindowStyle Hidden -PassThru -Environment $settings `
    -RedirectStandardError (Join-Path $root "console-$Port.stderr.log") `
    -RedirectStandardOutput (Join-Path $root "console-$Port.stdout.log")

for ($attempt = 0; $attempt -lt 30; $attempt++) {
    if ($bridge.HasExited) { throw "Inkling2-OS exited with code $($bridge.ExitCode)." }
    try {
        $health = Invoke-RestMethod -Uri ($url + 'health') -TimeoutSec 2
        if ($health.service -eq 'inkling2-os-bridge') {
            if ($Open) { Start-Process $url }
            Write-Host "Inkling2-OS is listening on $url (PID $($bridge.Id))."
            if ($UseTokens) {
                Write-Host "To unlock, run $($MyInvocation.MyCommand.Name) -CopyToken Console and paste the token in the browser."
            }
            return
        }
    } catch {
        Start-Sleep -Milliseconds 250
    }
}
throw "Inkling2-OS did not report healthy on TCP port $Port."
