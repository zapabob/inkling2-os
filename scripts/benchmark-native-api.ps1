[CmdletBinding()]
param(
    [Parameter(Mandatory)]
    [string]$PromptFile,
    [Parameter(Mandatory)]
    [string]$Model,
    [Parameter(Mandatory)]
    [string]$OutFile,
    [ValidateRange(1, 128)]
    [int]$MaxTokens = 32,
    [ValidateRange(1, 5)]
    [int]$Warmups = 1,
    [ValidateRange(2, 20)]
    [int]$Repeats = 3,
    [ValidateRange(1024, 65535)]
    [int]$Port = 18081
)

$ErrorActionPreference = 'Stop'
$address = "http://127.0.0.1:$Port/v1/chat/completions"
$tokenPath = Join-Path $env:LOCALAPPDATA 'Inkling2-OS/v0.1.1/Inference.dpapi'
$secret = ConvertTo-SecureString (Get-Content -LiteralPath $tokenPath -Raw)
$key = [Net.NetworkCredential]::new('', $secret).Password
$prompt = Get-Content -LiteralPath $PromptFile -Raw -Encoding utf8
$body = @{
    model=$Model; max_tokens=$MaxTokens; stream=$false
    messages=@(@{role='user';content=$prompt})
} | ConvertTo-Json -Depth 5 -Compress
$bytes = [Text.Encoding]::UTF8.GetBytes($body)

function Get-GpuSnapshot {
    if (-not (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) { return $null }
    $output = nvidia-smi --id=0 --query-gpu=name,driver_version,memory.free,memory.total --format=csv,noheader,nounits
    if ($LASTEXITCODE -eq 0) { $output }
}

$samples = for ($i=0; $i -lt $Warmups+$Repeats; $i++) {
    $gpu = Get-GpuSnapshot
    $watch = [Diagnostics.Stopwatch]::StartNew()
    $response = Invoke-WebRequest -Uri $address -Method Post -NoProxy -SkipHttpErrorCheck -TimeoutSec 255 `
        -Headers @{Authorization="Bearer $key"} -ContentType 'application/json; charset=utf-8' -Body $bytes
    $watch.Stop()
    $parsed = $response.Content | ConvertFrom-Json
    [pscustomobject]@{
        run=$i+1
        warmup=$i -lt $Warmups
        httpStatus=$response.StatusCode
        totalMilliseconds=[math]::Round($watch.Elapsed.TotalMilliseconds, 1)
        outputTokens=$parsed.usage.completion_tokens
        resultPresent=[bool]($parsed.choices | Select-Object -First 1 | ForEach-Object { $_.message.content })
        error=$parsed.error
        gpuBefore=$gpu
    }
    if ($response.StatusCode -ne 200) { break }
}

[pscustomobject]@{
    measuredAt=(Get-Date).ToUniversalTime().ToString('o')
    model=$Model
    maxRequestedOutputTokens=$MaxTokens
    promptSha256=(Get-FileHash -LiteralPath $PromptFile -Algorithm SHA256).Hash.ToLowerInvariant()
    samples=@($samples)
} | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $OutFile -Encoding utf8
if (@($samples | Where-Object httpStatus -ne 200).Count) { exit 1 }
