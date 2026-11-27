[CmdletBinding()]
param(
    [ValidateRange(1024, 65535)]
    [int]$Port = 18081,
    [string]$ShellPath,
    [string]$EnginePath = $env:INKLING2_ENGINE_BIN,
    [string]$ModelsDir = $env:INKLING2_MODELS_DIR,
    [string]$PreviewDir = $env:INKLING2_PREVIEW_DIR,
    [switch]$Open,
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
$url = "http://127.0.0.1:$Port/"
if (Get-NetTCPConnection -State Listen -LocalPort $Port -ErrorAction SilentlyContinue) {
    throw "TCP port $Port is already in use."
}
$settings = @{
    INKLING2_BRIDGE_TOKEN = Get-LocalToken Console $true
    INKLING2_INFERENCE_TOKEN = Get-LocalToken Inference $true
    INKLING2_STATE_DIR = $root
    INKLING2_ENGINE_BIN = $EnginePath
    INKLING2_MODELS_DIR = $ModelsDir
    INKLING2_PREVIEW_DIR = $PreviewDir
}
$previous = @{}
try {
    foreach ($key in $settings.Keys) {
        $previous[$key] = [Environment]::GetEnvironmentVariable($key, 'Process')
        [Environment]::SetEnvironmentVariable($key, $settings[$key], 'Process')
    }
    $bridge = Start-Process -FilePath $executable -ArgumentList 'serve', "127.0.0.1:$Port" `
        -WorkingDirectory $PSScriptRoot -WindowStyle Hidden -PassThru `
        -RedirectStandardError (Join-Path $root "console-$Port.stderr.log") `
        -RedirectStandardOutput (Join-Path $root "console-$Port.stdout.log")
} finally {
    foreach ($key in $previous.Keys) {
        [Environment]::SetEnvironmentVariable($key, $previous[$key], 'Process')
    }
}

for ($attempt = 0; $attempt -lt 30; $attempt++) {
    if ($bridge.HasExited) { throw "Inkling2-OS exited with code $($bridge.ExitCode)." }
    try {
        $health = Invoke-RestMethod -Uri ($url + 'health') -TimeoutSec 2
        if ($health.service -eq 'inkling2-os-bridge') {
            if ($Open) { Start-Process $url }
            Write-Host "Inkling2-OS is listening on $url (PID $($bridge.Id))."
            Write-Host "To unlock, run $($MyInvocation.MyCommand.Name) -CopyToken Console and paste the token in the browser."
            return
        }
    } catch {
        Start-Sleep -Milliseconds 250
    }
}
throw "Inkling2-OS did not report healthy on TCP port $Port."
