[CmdletBinding()]
param(
    [Parameter(Mandatory)][string]$QualificationDir,
    [Parameter(Mandatory)][string]$EngineRepo,
    [Parameter(Mandatory)][string]$OutputDir,
    [Parameter(Mandatory)][ValidateRange(61203283968,4000000000000)][long]$DiskBytes
)

$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $true
. (Join-Path $PSScriptRoot 'release-source-digest.ps1')
$qualification = (Resolve-Path -LiteralPath $QualificationDir).Path
$receipt = Get-Content -LiteralPath (Join-Path $qualification 'qualification.json') -Raw | ConvertFrom-Json
$osRepo = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$engineSource = (Resolve-Path -LiteralPath $EngineRepo).Path
if ($receipt.schema -ne 1 -or @($receipt.additional_models).Count -ne 2 -or $receipt.boot_verifier -notmatch '^sha256:[0-9a-f]{64}$' -or
    (Get-ReleaseSourceDigest $osRepo) -ne $receipt.sources.os -or
    (Get-ReleaseSourceDigest $engineSource) -ne $receipt.sources.engine) {
    throw 'Private USB preparation requires the same two source trees that passed qualification.'
}
docker image inspect $receipt.boot_verifier --format '{{.Id}}' | Out-Null
$destination = (New-Item -ItemType Directory -Path $OutputDir).FullName
$savedContext = $env:INKLING2_ENGINE_CONTEXT
Push-Location -LiteralPath $osRepo
try {
    $env:INKLING2_ENGINE_CONTEXT = $engineSource.Replace('\','/')
    docker buildx bake "--allow=fs.read=$env:INKLING2_ENGINE_CONTEXT" -f os-image/docker-bake.hcl --load private-disk-tools
    docker run --rm --network none --tmpfs '/tmp:rw,nosuid,nodev,size=7g' `
        --mount "type=bind,source=$destination,target=/out-usb" inkling2-private-disk:local `
        --disk-bytes $DiskBytes --output /out-usb/bundle
    docker run --rm --network none --tmpfs '/tmp:rw,nosuid,nodev,size=7g' `
        --mount "type=bind,source=$destination,target=/out-usb,readonly" $receipt.boot_verifier `
        --installed-bundle /out-usb/bundle --require-desktop --require-tor
    $bundle = Join-Path $destination 'bundle'
    @{
        schema = 1; disk_bytes = $DiskBytes
        layout_sha256 = (Get-FileHash -LiteralPath (Join-Path $bundle 'layout.json')).Hash
        payload_sha256 = (Get-FileHash -LiteralPath (Join-Path $bundle 'disk-data.gz')).Hash
        additional_models = @($receipt.additional_models)
        model_sha256 = $receipt.model_sha256; tokenizer_sha256 = $receipt.tokenizer_sha256
    } | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $destination 'usb-verification.json') -Encoding ascii
    Write-Host "Private USB system verified, ready for the approved device at $destination. The weights remain at their original path until installation."
} finally {
    [Environment]::SetEnvironmentVariable('INKLING2_ENGINE_CONTEXT', $savedContext, 'Process')
    Pop-Location
}
