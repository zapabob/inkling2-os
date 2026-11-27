[CmdletBinding()]
param(
    [Parameter(Mandatory)]
    [string]$Binary,
    [string]$Config,
    [ValidateRange(1024, 65535)]
    [int]$Port = 29483
)
$ErrorActionPreference = 'Stop'
$binaryPath = (Resolve-Path -LiteralPath $Binary).Path
if (Get-NetTCPConnection -State Listen -LocalPort $Port -ErrorAction SilentlyContinue) {
    throw "RustXNG port $Port is already occupied."
}
$profileArgs = @('--profile', 'full')
$launchArgs = @('--profile', 'full', '--bind', '127.0.0.1', '--port', [string]$Port)
$launchOptions = @{}
if ($Config) {
    $configPath = (Resolve-Path -LiteralPath $Config).Path
    $profileArgs += @('--config', $configPath)
    $launchOptions.Environment = @{ SEARXNG_SETTINGS_PATH = $configPath }
}
& $binaryPath @profileArgs --check-config
if ($LASTEXITCODE -ne 0) {
    throw "RustXNG full profile validation exited with code $LASTEXITCODE."
}
$runDir = Join-Path $env:LOCALAPPDATA 'Inkling2-OS/v0.1.1'
New-Item -ItemType Directory -Force -Path $runDir | Out-Null
$searchProcess = Start-Process @launchOptions -FilePath $binaryPath -ArgumentList $launchArgs -WorkingDirectory $runDir `
    -WindowStyle Hidden -PassThru -RedirectStandardError (Join-Path $runDir "rustxng-$Port.stderr.log") `
    -RedirectStandardOutput (Join-Path $runDir "rustxng-$Port.stdout.log")
try {
    $listen = "http://127.0.0.1:$Port"
    for ($attempt = 0; $attempt -lt 30; $attempt++) {
        if ($searchProcess.HasExited) { throw "RustXNG exited with code $($searchProcess.ExitCode)." }
        try { $health = Invoke-RestMethod -Uri "$listen/healthz" -TimeoutSec 1 }
        catch { Start-Sleep -Milliseconds 200; continue }
        if ($health) {
            if (-not (Get-NetTCPConnection -State Listen -LocalAddress 127.0.0.1 -LocalPort $Port -OwningProcess $searchProcess.Id -ErrorAction SilentlyContinue)) {
                throw "RustXNG did not own the expected loopback listener."
            }
            Write-Host "RustXNG full profile is listening on $listen (PID $($searchProcess.Id))."
            return
        }
    }
    throw 'RustXNG did not become ready.'
} catch {
    if (-not $searchProcess.HasExited) { $searchProcess.Kill() }
    throw
}
