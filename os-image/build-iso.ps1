# Inkling2-OS Bootable ISO Builder (Windows Host -> Docker -> inkling2-os-x86_64-sm120.iso)
param(
    [string]$OutDir = "C:\Users\downl\inkling2-os\dist"
)

$ErrorActionPreference = "Stop"
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

Write-Host "[Inkling2-OS] Building reproducible UEFI/BIOS ISO via Docker (CUDA 13.5 sm_120a + Rust 2024 PID 1)..."
docker build `
    -f "C:\Users\downl\inkling2-os\os-image\Dockerfile.iso-builder" `
    --target artifact `
    --output "type=local,dest=$OutDir" `
    "C:\Users\downl\inkling2-os"
if ($LASTEXITCODE -ne 0) {
    throw "[Inkling2-OS] docker build failed with exit code $LASTEXITCODE"
}
$isoPath = Join-Path $OutDir "inkling2-os-x86_64-sm120.iso"
if (-not (Test-Path $isoPath)) {
    throw "[Inkling2-OS] Expected ISO not found at $isoPath"
}
$sz = (Get-Item $isoPath).Length
Write-Host "[Inkling2-OS] ISO generated at $isoPath ($([math]::Round($sz/1MB, 2)) MiB)"
