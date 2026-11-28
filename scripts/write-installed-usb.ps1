[CmdletBinding()]
param(
    [Parameter(Mandatory)][ValidateRange(0, 1024)][int]$DiskNumber,
    [Parameter(Mandatory)][ValidatePattern('^[0-9A-Fa-f]{64}$')][string]$ExpectedIdentity,
    [Parameter(Mandatory)][string]$Bundle,
    [Parameter(Mandatory)][ValidatePattern('^[0-9A-Fa-f]{64}$')][string]$LayoutSha256,
    [Parameter(Mandatory)][ValidatePattern('^[0-9A-Fa-f]{64}$')][string]$PayloadSha256,
    [string]$ModelPath,
    [ValidatePattern('^[0-9A-Fa-f]{64}$')][string]$ModelSha256,
    [ValidatePattern('^[0-9A-Fa-f]{64}$')][string]$TokenizerSha256,
    [string[]]$AdditionalModelPath = @(),
    [string[]]$AdditionalModelSha256 = @(),
    [string]$LogPath,
    [switch]$CheckOnly
)

$ErrorActionPreference = 'Stop'
$sectorBytes = 512

function Assert-TargetDisk {
    $disk = Get-Disk -Number $DiskNumber
    $hardware = Get-CimInstance Win32_DiskDrive -Filter "Index=$DiskNumber"
    if ($disk.IsBoot -or $disk.IsSystem -or $disk.IsOffline -or $disk.IsReadOnly -or
        $disk.BusType -ne 'USB' -or $disk.LogicalSectorSize -ne $sectorBytes -or
        $disk.Size -ne [long]$layout.disk_bytes -or [string]::IsNullOrWhiteSpace($hardware.PNPDeviceID)) {
        throw 'The current disk is not the approved online USB target.'
    }
    $identityText = "$($disk.FriendlyName)|$($disk.Size)|$($disk.LogicalSectorSize)|$($hardware.PNPDeviceID)"
    $identity = [Convert]::ToHexString([Security.Cryptography.SHA256]::HashData([Text.Encoding]::UTF8.GetBytes($identityText)))
    if ($identity -ne $ExpectedIdentity) { throw 'The USB identity changed.' }
    $disk
}

function Assert-SeparateVolume([string]$Path) {
    $resolved = (Resolve-Path -LiteralPath $Path).ProviderPath
    $volume = Get-Volume -FilePath $resolved -ErrorAction Stop
    $partitions = @(Get-Partition -Volume $volume -ErrorAction Stop)
    if ($partitions.Count -ne 1 -or $partitions[0].DiskNumber -eq $DiskNumber) {
        throw "The installation source or log must be on a separate physical disk: $resolved"
    }
}

function Transfer-Region(
    [IO.Stream]$Source, [IO.Stream]$Destination, [long]$Length,
    [byte[]]$Buffer, [string]$Digest
) {
    $hash = [Security.Cryptography.IncrementalHash]::CreateHash([Security.Cryptography.HashAlgorithmName]::SHA256)
    try {
        $remaining = $Length
        while ($remaining -gt 0) {
            $count = [int][math]::Min($remaining, $Buffer.Length)
            $Source.ReadExactly($Buffer, 0, $count)
            $hash.AppendData($Buffer, 0, $count)
            if ($Destination) { $Destination.Write($Buffer, 0, $count) }
            $remaining -= $count
        }
        if ([Convert]::ToHexString($hash.GetHashAndReset()) -ne $Digest) {
            throw 'Disk region checksum did not match.'
        }
    } finally { $hash.Dispose() }
}

function Install-ModelFiles($Specification, $Files) {
    Update-HostStorageCache
    $null = Assert-TargetDisk
    $partition = Get-Partition -DiskNumber $DiskNumber -PartitionNumber $Specification.number
    if ([Guid]$partition.Guid -ne [Guid]$Specification.guid -or
        $partition.Offset -ne [long]$Specification.offset -or $partition.Size -ne [long]$Specification.length) {
        throw 'The written model partition did not match its verified GUID and geometry.'
    }
    Format-Volume -Partition $partition -FileSystem NTFS -NewFileSystemLabel INKLING_MODELS -Confirm:$false -Force | Out-Null
    $partition = Get-Partition -DiskNumber $DiskNumber -PartitionNumber $Specification.number
    if (-not $partition.DriveLetter -or $partition.DriveLetter -eq [char]0) {
        $partition = Add-PartitionAccessPath -InputObject $partition -AssignDriveLetter -PassThru
    }
    $letter = $partition.DriveLetter
    $mounted = Get-Partition -DriveLetter $letter
    if ($mounted.DiskNumber -ne $DiskNumber -or [Guid]$mounted.Guid -ne [Guid]$Specification.guid) {
        throw 'The temporary model drive letter does not belong to the approved USB partition.'
    }
    $volumeRoot = '{0}:\' -f $letter
    try {
        Write-Host 'Streaming original model files directly onto the USB.'
        foreach ($entry in $Files) {
            Copy-Item -LiteralPath $entry.file.FullName -Destination $volumeRoot
            if ((Get-FileHash -LiteralPath (Join-Path $volumeRoot $entry.file.Name)).Hash -ne $entry.hash) {
                throw "USB readback failed for $($entry.file.Name)."
            }
        }
        Set-Partition -InputObject $mounted -NoDefaultDriveLetter $true
    } finally {
        Remove-PartitionAccessPath -InputObject $mounted -AccessPath $volumeRoot
    }
}

$bundlePath = (Resolve-Path -LiteralPath $Bundle).Path
$layoutPath = Join-Path $bundlePath 'layout.json'
$payloadPath = Join-Path $bundlePath 'disk-data.gz'
Assert-SeparateVolume $layoutPath
Assert-SeparateVolume $payloadPath
if ($LogPath) {
    $logFile = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($LogPath)
    if (Test-Path -LiteralPath $logFile) { throw 'Choose a new path for the USB installation log.' }
    Assert-SeparateVolume ([IO.Path]::GetDirectoryName($logFile))
}
if ((Get-FileHash -LiteralPath $layoutPath -Algorithm SHA256).Hash -ne $LayoutSha256 -or
    (Get-FileHash -LiteralPath $payloadPath -Algorithm SHA256).Hash -ne $PayloadSha256) {
    throw 'The disk bundle does not match the verified artifact.'
}
$layout = Get-Content -LiteralPath $layoutPath -Raw | ConvertFrom-Json
$regions = @($layout.regions)
if ($layout.version -notin 1, 2, 3 -or $layout.sector_bytes -ne $sectorBytes -or $regions.Count -lt 4 -or
    $regions[-1].offset -ne 0 -or $regions[-1].length -ne (34 * $sectorBytes) -or
    $regions[-2].offset -ne ([long]$layout.disk_bytes - 33 * $sectorBytes) -or
    $regions[-2].length -ne (33 * $sectorBytes)) {
    throw 'Unsupported or incomplete GPT disk bundle.'
}
$model = $null
$tokenizer = $null
$modelPartition = $layout.model_partition
$files = @()
if ($layout.version -in 2, 3) {
    $expectedLength = if ($layout.version -eq 2) { 12GB } else { 48GB }
    if ($AdditionalModelPath.Count -ne $AdditionalModelSha256.Count -or
        ($layout.version -eq 3 -and $AdditionalModelPath.Count -ne 2) -or
        ($layout.version -eq 2 -and $AdditionalModelPath.Count -ne 0)) {
        throw 'The model set must match the qualified disk layout.'
    }
    $modelGuid = [Guid]::Empty
    if (-not $modelPartition -or $modelPartition.number -ne 3 -or $modelPartition.length -ne $expectedLength -or
        $modelPartition.offset -le 0 -or $modelPartition.offset % 1MB -ne 0 -or
        $modelPartition.offset + $modelPartition.length -gt $regions[-2].offset -or
        -not [Guid]::TryParse($modelPartition.guid, [ref]$modelGuid) -or $modelGuid -eq [Guid]::Empty) {
        throw 'Model partition metadata is invalid.'
    }
    if (-not $ModelPath -or -not $ModelSha256 -or -not $TokenizerSha256) {
        throw 'Provide the model and SHA-256 hashes from qualification.'
    }
    $model = Get-Item -LiteralPath $ModelPath
    $tokenizerPath = Join-Path $model.DirectoryName 'tokenizer.json'
    if (!(Test-Path -LiteralPath $tokenizerPath -PathType Leaf)) {
        $tokenizerPath = Join-Path $PSScriptRoot '../crates/inkling2-shell/assets/sc117/tokenizer.json'
    }
    $tokenizer = Get-Item -LiteralPath $tokenizerPath
    $files = @(@{file=$model; hash=$ModelSha256}, @{file=$tokenizer; hash=$TokenizerSha256})
    foreach ($i in 0..1) {
        if ($i -lt $AdditionalModelPath.Count) {
            $files += @{file=(Get-Item -LiteralPath $AdditionalModelPath[$i]); hash=$AdditionalModelSha256[$i]}
        }
    }
    $names = @{}
    $total = 0L
    foreach ($entry in $files) {
        $file = $entry.file
        Assert-SeparateVolume $file.FullName
        if ($file -isnot [IO.FileInfo] -or ($file.Name -ne 'tokenizer.json' -and $file.Extension -ne '.strata') -or
            $names.ContainsKey($file.Name) -or $entry.hash -notmatch '^[0-9a-fA-F]{64}$' -or
            (Get-FileHash -LiteralPath $file.FullName).Hash -ne $entry.hash) {
            throw 'Model or tokenizer did not match qualification.'
        }
        $names[$file.Name] = $true
        $total += $file.Length
    }
    if ($total -gt ($modelPartition.length - 256MB)) { throw 'The qualified model set exceeds the partition.' }
} elseif ($modelPartition -or $ModelPath -or $ModelSha256 -or $TokenizerSha256 -or $AdditionalModelPath -or $AdditionalModelSha256) {
    throw 'This bundle has no model partition.'
}
$previousEnd = 0L
foreach ($region in ($regions | Sort-Object {[long]$_.offset})) {
    $offset = [long]$region.offset
    $length = [long]$region.length
    if ($length -le 0 -or $offset -lt $previousEnd -or $offset % $sectorBytes -ne 0 -or
        $length % $sectorBytes -ne 0 -or $length -gt ([long]$layout.disk_bytes - $offset) -or
        ($modelPartition -and $offset -lt ($modelPartition.offset + $modelPartition.length) -and
            ($offset + $length) -gt $modelPartition.offset) -or
        $region.sha256 -notmatch '^[0-9a-f]{64}$') {
        throw 'A disk bundle region is out of bounds or overlaps another region.'
    }
    $previousEnd = $offset + $length
}
$selected = Assert-TargetDisk
Write-Host "Verified bundle and target. Disk $($selected.Number), $($selected.FriendlyName), $($selected.Size) bytes."
if ($CheckOnly) { return }

$principal = [Security.Principal.WindowsPrincipal]::new([Security.Principal.WindowsIdentity]::GetCurrent())
if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    throw 'Writing the physical USB requires an elevated Windows session.'
}
if ($LogPath) { Start-Transcript -LiteralPath $logFile -NoClobber | Out-Null }
try {
    $devicePath = "\\.\PhysicalDrive$DiskNumber"
    $buffer = [byte[]]::new(1024 * 1024)
    $test = [IO.FileStream]::new($devicePath, [IO.FileMode]::Open, [IO.FileAccess]::ReadWrite, [IO.FileShare]::ReadWrite, 0)
    try {
        if (-not $test.CanSeek) { throw 'Windows did not open a seekable disk handle.' }
        $test.ReadExactly($buffer, 0, $sectorBytes)
    } finally { $test.Dispose() }

    $selected = Assert-TargetDisk
    Write-Host "Replacing the partition table on verified USB disk $DiskNumber."
    Clear-Disk -InputObject $selected -RemoveData -RemoveOEM -Confirm:$false
    $null = Assert-TargetDisk
    Write-Host 'Writing and reading back the verified UEFI disk bundle.'

    $diskStream = [IO.FileStream]::new($devicePath, [IO.FileMode]::Open, [IO.FileAccess]::ReadWrite, [IO.FileShare]::ReadWrite, 0)
    try {
        $archive = [IO.File]::OpenRead($payloadPath)
        $payload = [IO.Compression.GZipStream]::new($archive, [IO.Compression.CompressionMode]::Decompress)
        try {
            for ($index = 0; $index -lt $regions.Count; $index++) {
                $region = $regions[$index]
                $offset = [long]$region.offset
                $length = [long]$region.length
                $diskStream.Position = $offset
                Transfer-Region $payload $diskStream $length $buffer $region.sha256
                $diskStream.Flush($true)
                $diskStream.Position = $offset
                Transfer-Region $diskStream $null $length $buffer $region.sha256
                Write-Progress -Activity 'Writing and verifying the Linux USB' -PercentComplete ((100 * ($index + 1)) / $regions.Count)
            }
            if ($payload.ReadByte() -ne -1) { throw 'Unexpected trailing bytes in disk bundle.' }
        } finally {
            $payload.Dispose()
            $archive.Dispose()
        }
    } finally { $diskStream.Dispose() }
    if ($modelPartition) { Install-ModelFiles $modelPartition $files }
    Write-Progress -Activity 'Writing and verifying the Linux USB' -Completed
    Write-Host "USB_LAYOUT_SHA256=$LayoutSha256"
    if ($modelPartition) { Write-Host "USB_MODEL_SHA256=$ModelSha256" }
    foreach ($hash in $AdditionalModelSha256) { Write-Host "USB_EXTRA_MODEL_SHA256=$hash" }
    Write-Host 'INKLING2_USB_VERIFIED'
} finally {
    if ($LogPath) { Stop-Transcript | Out-Null }
}
