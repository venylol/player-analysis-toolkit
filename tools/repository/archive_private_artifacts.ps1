[CmdletBinding(SupportsShouldProcess = $true)]
param(
    [Parameter(Mandatory = $true)]
    [string]$SourceRoot,

    [Parameter(Mandatory = $true)]
    [string]$ArchiveRoot,

    [Parameter(Mandatory = $true)]
    [string]$ManifestPath
)

$ErrorActionPreference = 'Stop'
$utf8NoBom = [System.Text.UTF8Encoding]::new($false)
$source = [System.IO.Path]::GetFullPath($SourceRoot)
$archive = [System.IO.Path]::GetFullPath($ArchiveRoot)
$manifestFile = [System.IO.Path]::GetFullPath($ManifestPath)

if (-not (Test-Path -LiteralPath $source -PathType Container)) {
    throw "Source root does not exist: $source"
}
if ($archive.StartsWith($source + [System.IO.Path]::DirectorySeparatorChar, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw 'ArchiveRoot must be outside SourceRoot.'
}

$manifest = Get-Content -LiteralPath $manifestFile -Raw -Encoding UTF8 | ConvertFrom-Json
if (-not $manifest.items) {
    throw 'The manifest contains no items.'
}

$resolvedItems = foreach ($item in $manifest.items) {
    $relative = [string]$item.path
    if ([System.IO.Path]::IsPathRooted($relative) -or $relative.Contains('..')) {
        throw "Manifest path must be a safe relative path: $relative"
    }
    $sourcePath = [System.IO.Path]::GetFullPath((Join-Path $source $relative))
    $destinationPath = [System.IO.Path]::GetFullPath((Join-Path $archive $relative))
    if (-not $sourcePath.StartsWith($source + [System.IO.Path]::DirectorySeparatorChar, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Manifest path escapes SourceRoot: $relative"
    }
    if (-not (Test-Path -LiteralPath $sourcePath)) {
        throw "Manifest source is missing: $relative"
    }
    if (Test-Path -LiteralPath $destinationPath) {
        throw "Archive destination already exists: $destinationPath"
    }
    [pscustomobject]@{
        Relative = $relative
        Source = $sourcePath
        Destination = $destinationPath
    }
}

$auditRows = foreach ($item in $resolvedItems) {
    $sourceItem = Get-Item -LiteralPath $item.Source -Force
    $enumerationErrors = @()
    $files = if ($sourceItem.PSIsContainer) {
        @(Get-ChildItem -LiteralPath $item.Source -Recurse -Force -File -ErrorAction SilentlyContinue -ErrorVariable +enumerationErrors)
    } else {
        @($sourceItem)
    }
    [pscustomobject]@{
        path = $item.Relative
        type = if ($sourceItem.PSIsContainer) { 'directory' } else { 'file' }
        files = $files.Count
        bytes = [long](($files | Measure-Object Length -Sum).Sum)
        enumerationWarnings = $enumerationErrors.Count
    }
}

if ($PSCmdlet.ShouldProcess($archive, "Archive $($resolvedItems.Count) private artifact groups")) {
    foreach ($item in $resolvedItems) {
        $destinationParent = Split-Path -Parent $item.Destination
        [System.IO.Directory]::CreateDirectory($destinationParent) | Out-Null
        Move-Item -LiteralPath $item.Source -Destination $item.Destination
    }
    [System.IO.Directory]::CreateDirectory($archive) | Out-Null
    $audit = [ordered]@{
        schema = 'player-analysis-private-archive-v1'
        archivedAtUtc = [DateTimeOffset]::UtcNow.ToString('o')
        sourceRoot = $source
        archiveRoot = $archive
        itemCount = $auditRows.Count
        fileCount = [int](($auditRows | Measure-Object files -Sum).Sum)
        totalBytes = [long](($auditRows | Measure-Object bytes -Sum).Sum)
        items = @($auditRows)
    }
    [System.IO.File]::WriteAllText(
        (Join-Path $archive 'ARCHIVE_MANIFEST.json'),
        ($audit | ConvertTo-Json -Depth 6),
        $utf8NoBom
    )
}

$auditRows | Format-Table -AutoSize
