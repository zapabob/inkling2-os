function Get-ReleaseSourcePaths([string]$Scope) {
    if ($Scope -eq 'WindowsOS') {
        return @('Cargo.toml','Cargo.lock','crates/inkling2-shell','crates/inkling2-resource','crates/inkling2-companion','ui','docs','README.md','CHANGELOG.md','RELEASE_NOTES.md',
            'LICENSE','scripts/start-windows-console.ps1','scripts/start-windows-desktop.ps1','scripts/start-rustxng-full.ps1','scripts/stage-zstd-runtime.ps1',
            'scripts/copy-console-token.cmd','scripts/package-windows-console.ps1','scripts/release-source-digest.ps1',
            'scripts/qualify-native-pair.ps1','scripts/promote-native-pair.ps1')
    }
    return @('.')
}

function Get-ReleaseSourceDigest([string]$Repo, [string]$Scope = 'All') {
    $scopePaths = @(Get-ReleaseSourcePaths $Scope)
    $paths = git -C $Repo -c core.quotePath=false ls-files --cached --others --exclude-standard -- @scopePaths
    if ($LASTEXITCODE -ne 0) { throw 'Could not enumerate the repository source.' }
    $inputLines = foreach ($relativePath in ($paths | Sort-Object -CaseSensitive -Unique)) {
        $file = Join-Path $Repo $relativePath
        $digest = if (Test-Path -LiteralPath $file -PathType Leaf) {
            (Get-FileHash -LiteralPath $file -Algorithm SHA256).Hash
        } else { 'deleted' }
        "$digest $relativePath"
    }
    [Convert]::ToHexString([Security.Cryptography.SHA256]::HashData([Text.Encoding]::UTF8.GetBytes($inputLines -join "`n")))
}
