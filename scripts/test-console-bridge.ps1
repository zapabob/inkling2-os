[CmdletBinding()]
param(
    [ValidateRange(1024, 65535)]
    [int]$Port = 18081,
    [switch]$UseTokens
)

$ErrorActionPreference = 'Stop'
$url = "http://127.0.0.1:$Port"
$root = Join-Path $env:LOCALAPPDATA 'Inkling2-OS/v0.1.1'

function Read-LocalToken([string]$Name) {
    $value = ConvertTo-SecureString (Get-Content -LiteralPath (Join-Path $root "$Name.dpapi") -Raw)
    [Net.NetworkCredential]::new('', $value).Password
}

function Assert-Status([string]$Test, [int]$Status, [string]$Path, [hashtable]$Headers, $Body) {
    $args = @{ Uri="$url$Path"; Headers=$Headers; SkipHttpErrorCheck=$true; TimeoutSec=20; NoProxy=$true }
    if ($null -ne $Body) {
        $args.Method = 'Post'
        $args.ContentType = 'application/json'
        $args.Body = $Body | ConvertTo-Json -Depth 5 -Compress
    }
    $actual = (Invoke-WebRequest @args).StatusCode
    if ($actual -ne $Status) { throw "$Test returned HTTP $actual; expected $Status." }
    Write-Host "$Test passed (HTTP $actual)"
}

Assert-Status 'Public health' 200 '/health' @{} $null
$health = Invoke-RestMethod "$url/health" -NoProxy -TimeoutSec 5
$expectedMode = if ($UseTokens) { 'locked' } else { 'local' }
if ($health.service -ne 'inkling2-os-bridge' -or $health.console_mode -ne $expectedMode) {
    throw "Expected the Inkling2 console in $expectedMode mode on port $Port."
}
$console = @{}
if ($UseTokens) {
    $console = @{ Authorization = "Bearer $(Read-LocalToken Console)" }
    $inference = @{ Authorization = "Bearer $(Read-LocalToken Inference)" }
    Assert-Status 'Anonymous model catalogue denied' 401 '/v1/models' @{} $null
    Assert-Status 'Inference model catalogue' 200 '/v1/models' $inference $null
    Assert-Status 'Inference token has no file access' 401 '/v1/fs/list' $inference @{root='workspace'}
} else {
    Assert-Status 'Key-free model catalogue' 200 '/v1/models' @{} $null
}
Assert-Status 'Console session' 200 '/v1/session' $console $null
Assert-Status 'Shared directory listing' 200 '/v1/fs/list' $console @{root='workspace'}
Assert-Status 'Parent traversal denied' 403 '/v1/fs/read' $console @{root='workspace';path='../README.md'}
Assert-Status 'Unshared state denied' 400 '/v1/fs/list' $console @{root='state'}
Assert-Status 'Unexpected Origin denied' 403 '/v1/session' ($console + @{Origin='https://example.com'}) $null
Assert-Status 'Unexpected Host denied' 403 '/health' @{Host='example.com'} $null
Assert-Status 'Public browser rejects local destinations' 403 '/v1/net/fetch' $console @{url='https://localhost/'}
Assert-Status 'Public browser retrieves ordinary HTTPS' 200 '/v1/net/fetch' $console @{url='https://example.com/'}
Assert-Status 'Experimental LoRA disabled by default' 403 '/v1/lora/reinforce' $console @{text='test';reward=0}
