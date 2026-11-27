[CmdletBinding()]
param(
    [string]$OutDir = (Join-Path $PSScriptRoot '../dist/v0.1.1'),
    [string]$BinaryDir = (Join-Path $PSScriptRoot '../target/release')
)

$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$name = 'inkling2-os-v0.1.1-win64-console'
$stage = Join-Path $OutDir ('stage-' + [IO.Path]::GetRandomFileName())
New-Item -ItemType Directory -Force -Path $OutDir, "$stage/bin", "$stage/scripts", "$stage/notices" | Out-Null
Copy-Item -LiteralPath (Join-Path $BinaryDir 'inkling2-shell.exe') -Destination "$stage/bin"
Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'start-windows-console.ps1') -Destination "$stage/scripts"
Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'copy-console-token.cmd') -Destination "$stage/scripts"
foreach ($file in 'LICENSE', 'README.md', 'CHANGELOG.md', 'RELEASE_NOTES.md') {
    Copy-Item -LiteralPath (Join-Path $repo $file) -Destination $stage
}
Copy-Item -LiteralPath (Join-Path $repo 'docs') -Destination $stage -Recurse

$raw = cargo metadata --locked --offline --format-version 1 --filter-platform x86_64-pc-windows-msvc --manifest-path (Join-Path $repo 'Cargo.toml')
if ($LASTEXITCODE -ne 0) { throw 'Cargo metadata could not be read.' }
$graph = $raw | ConvertFrom-Json
$included = @{}
foreach ($node in $graph.resolve.nodes) { $included[$node.id] = $true }
$components = foreach ($package in $graph.packages) {
    if (-not $included.ContainsKey($package.id)) { continue }
    $component = @{
        type='library'; name=$package.name; version=$package.version
        'bom-ref'=$package.id
        purl="pkg:cargo/$($package.name)@$($package.version)"
    }
    if ($package.license) { $component.licenses = @(@{expression=$package.license}) }
    if ($package.source -like 'registry+*') {
        $licences = Get-ChildItem -LiteralPath (Split-Path $package.manifest_path) -File |
            Where-Object Name -Match '^(LICEN[CS]E|COPYING|NOTICE)([-_.].*)?$'
        if ($licences) {
            $notice = Join-Path "$stage/notices" "$($package.name)-$($package.version)"
            New-Item -ItemType Directory -Force -Path $notice | Out-Null
            foreach ($licence in $licences) {
                Copy-Item -LiteralPath $licence.FullName -Destination $notice
            }
        }
    }
    $component
}
$dependencies = foreach ($node in $graph.resolve.nodes) {
    @{ref=$node.id; dependsOn=@($node.dependencies)}
}
$sbomFile = 'SBOM-win64.cdx.json'
@{
    bomFormat='CycloneDX'; specVersion='1.5'; version=1
    metadata=@{component=@{type='application'; name=$name; version='0.1.1'}}
    components=@($components)
    dependencies=@($dependencies)
} | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath (Join-Path $stage $sbomFile) -Encoding utf8
Copy-Item -LiteralPath (Join-Path $stage $sbomFile) -Destination $OutDir -Force

Compress-Archive -Path "$stage/*" -DestinationPath (Join-Path $OutDir "$name.zip") -Force
$packageList = Join-Path $OutDir 'linux-rootfs-packages.tsv'
if (Test-Path -LiteralPath $packageList -PathType Leaf) {
    $iso = Join-Path $OutDir 'inkling2-os-v0.1.1-x86_64-sm120.iso'
    $imageHash = (Get-FileHash -LiteralPath $iso -Algorithm SHA256).Hash.ToLowerInvariant()
    $spdx = @(
        'SPDXVersion: SPDX-2.3'
        'DataLicense: CC0-1.0'
        'SPDXID: SPDXRef-DOCUMENT'
        'DocumentName: Inkling2-OS-v0.1.1-Linux-rootfs'
        "DocumentNamespace: https://github.com/zapabob/inkling2-os/sbom/$imageHash"
        'Creator: Tool: Inkling2-OS package-windows-console.ps1'
        "Created: $((Get-Date).ToUniversalTime().ToString('yyyy-MM-ddTHH:mm:ssZ'))"
    )
    $index = 0
    foreach ($line in Get-Content -LiteralPath $packageList) {
        $pkg, $version = $line -split "`t", 2
        if (-not $pkg -or -not $version) { throw 'Invalid Linux package inventory.' }
        $index++
        $spdx += '', "PackageName: $pkg", "SPDXID: SPDXRef-Package-$index",
            "PackageVersion: $version", 'PackageDownloadLocation: NOASSERTION',
            'FilesAnalyzed: false', 'PackageLicenseConcluded: NOASSERTION',
            'PackageLicenseDeclared: NOASSERTION', 'PackageCopyrightText: NOASSERTION',
            "Relationship: SPDXRef-DOCUMENT DESCRIBES SPDXRef-Package-$index"
    }
    $spdx | Set-Content -LiteralPath (Join-Path $OutDir 'SBOM-linux-rootfs.spdx') -Encoding utf8
}
$files = Get-ChildItem -LiteralPath $OutDir -File |
    Where-Object Name -In "$name.zip", $sbomFile, 'inkling2-os-v0.1.1-x86_64-sm120.iso', 'SBOM-linux-rootfs.spdx'
$sums = foreach ($file in ($files | Sort-Object Name)) {
    $hash = (Get-FileHash -Algorithm SHA256 -LiteralPath $file.FullName).Hash.ToLowerInvariant()
    "$hash  $($file.Name)"
}
$sums | Set-Content -LiteralPath (Join-Path $OutDir 'SHA256SUMS.txt') -Encoding ascii
if ((Resolve-Path -LiteralPath $stage).Path.StartsWith((Resolve-Path -LiteralPath $OutDir).Path + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) {
    Remove-Item -LiteralPath $stage -Recurse
}
Write-Host "Windows console bundle and checksum manifest created in $OutDir."
