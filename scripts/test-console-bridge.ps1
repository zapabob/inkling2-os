[CmdletBinding()]
param(
    [ValidateRange(1024, 65535)]
    [int]$Port = 18081
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

$console = @{ Authorization = "Bearer $(Read-LocalToken Console)" }
$inference = @{ Authorization = "Bearer $(Read-LocalToken Inference)" }
Assert-Status 'Public health' 200 '/health' @{} $null
Assert-Status 'Console token' 200 '/v1/session' $console $null
Assert-Status 'Anonymous model catalogue denied' 401 '/v1/models' @{} $null
Assert-Status 'Inference model catalogue' 200 '/v1/models' $inference $null
Assert-Status 'Inference token has no file access' 401 '/v1/fs/list' $inference @{root='workspace'}
Assert-Status 'Shared directory listing' 200 '/v1/fs/list' $console @{root='workspace'}
Assert-Status 'Parent traversal denied' 403 '/v1/fs/read' $console @{root='workspace';path='../README.md'}
Assert-Status 'Unshared state denied' 400 '/v1/fs/list' $console @{root='state'}
Assert-Status 'Unexpected Origin denied' 403 '/v1/session' ($console + @{Origin='https://example.com'}) $null
Assert-Status 'Unexpected Host denied' 403 '/health' @{Host='example.com'} $null
Assert-Status 'Public browser rejects local destinations' 403 '/v1/net/fetch' $console @{url='https://localhost/'}
Assert-Status 'Public browser retrieves ordinary HTTPS' 200 '/v1/net/fetch' $console @{url='https://example.com/'}
Assert-Status 'Experimental LoRA disabled by default' 403 '/v1/lora/reinforce' $console @{text='test';reward=0}
