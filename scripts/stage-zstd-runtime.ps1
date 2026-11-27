[CmdletBinding()]
param([Parameter(Mandatory)][string]$Directory)

$ErrorActionPreference = 'Stop'
$directoryPath = (Resolve-Path -LiteralPath $Directory).Path
$client = [Net.Http.HttpClient]::new()
function Get-VerifiedBytes([uri]$Url, [string]$Sha256) {
    $bytes = $client.GetByteArrayAsync($Url).GetAwaiter().GetResult()
    if ([Convert]::ToHexString([Security.Cryptography.SHA256]::HashData($bytes)) -ne $Sha256) {
        throw "Zstandard download checksum mismatch for $Url."
    }
    return ,$bytes
}
try {
    $zipBytes = Get-VerifiedBytes 'https://github.com/meta/zstd/releases/download/v1.5.7/zstd-v1.5.7-win64.zip' 'ACB4E8111511749DC7A3EBEDCA9B04190E37A17AFEB73F55D4425DBF0B90FAD9'
    $licence = Get-VerifiedBytes 'https://raw.githubusercontent.com/meta/zstd/f8745da6ff1ad1e7bab384bd1f9d742439278e99/LICENSE' '7055266497633C9025B777C78EB7235AF13922117480ED5C674677ADC381C9D8'
    $archive = [IO.Compression.ZipArchive]::new([IO.MemoryStream]::new($zipBytes))
    try {
        $entry = $archive.GetEntry('zstd-v1.5.7-win64/dll/libzstd.dll')
        if (-not $entry) { throw 'The upstream win64 package lacks libzstd.dll.' }
        [IO.Compression.ZipFileExtensions]::ExtractToFile($entry, (Join-Path $directoryPath 'libzstd.dll'), $true)
        [IO.File]::WriteAllBytes((Join-Path $directoryPath 'LICENSE-zstd.txt'), $licence)
    } finally { $archive.Dispose() }
} finally { $client.Dispose() }
