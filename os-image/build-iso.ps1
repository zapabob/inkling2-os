# Inkling2-OS Bootable ISO Builder (Windows Host -> Docker -> inkling2-os-x86_64-sm120.iso)
param(
    [string]$OutDir = (Join-Path $PSScriptRoot '../dist')
)

$ErrorActionPreference = "Stop"
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

Write-Host "[Inkling2-OS] Building the UEFI/BIOS diagnostic ISO with the Rust PID 1..."
$repo = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$OutDir = (Resolve-Path -LiteralPath $OutDir).Path
docker build `
    -f (Join-Path $PSScriptRoot 'Dockerfile.iso-builder') `
    --target artifact `
    --output "type=local,dest=$OutDir" `
    $repo
if ($LASTEXITCODE -ne 0) {
    throw "[Inkling2-OS] docker build failed with exit code $LASTEXITCODE"
}
$isoPath = Join-Path $OutDir "inkling2-os-v1.2.0-x86_64-sm120.iso"
if (-not (Test-Path $isoPath)) {
    throw "[Inkling2-OS] Expected ISO not found at $isoPath"
}
$sz = (Get-Item $isoPath).Length
Write-Host "[Inkling2-OS] ISO generated at $isoPath ($([math]::Round($sz/1MB, 2)) MiB)"
