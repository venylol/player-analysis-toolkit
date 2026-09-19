[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$SourceRoot,

    [Parameter(Mandatory = $true)]
    [string]$OutputDirectory,

    [Parameter(Mandatory = $true)]
    [string]$ManifestPath,

    [Parameter(Mandatory = $true)]
    [string]$BlockedRegex
)

$ErrorActionPreference = 'Stop'
$source = [System.IO.Path]::GetFullPath($SourceRoot)
$output = [System.IO.Path]::GetFullPath($OutputDirectory)
$manifestFile = [System.IO.Path]::GetFullPath($ManifestPath)
$utf8NoBom = [System.Text.UTF8Encoding]::new($false)
$manifest = Get-Content -LiteralPath $manifestFile -Raw -Encoding UTF8 | ConvertFrom-Json

if (-not (Test-Path -LiteralPath $source -PathType Container)) {
    throw "SourceRoot does not exist: $source"
}
if (Test-Path -LiteralPath $output) {
    throw "OutputDirectory already exists: $output"
}
if (-not (Get-Command rg -ErrorAction SilentlyContinue)) {
    throw 'ripgrep (rg) is required for the privacy scan.'
}

$resolved = foreach ($asset in $manifest.assets) {
    $relative = [string]$asset.path
    $assetSource = [System.IO.Path]::GetFullPath((Join-Path $source $relative))
    if (-not $assetSource.StartsWith($source + [System.IO.Path]::DirectorySeparatorChar, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Asset path escapes SourceRoot: $relative"
    }
    if (-not (Test-Path -LiteralPath $assetSource)) {
        throw "Asset source is missing: $relative"
    }
    [pscustomobject]@{
        Name = [string]$asset.name
        Relative = $relative
        Source = $assetSource
    }
}

$blockedRegex = $BlockedRegex
foreach ($asset in $resolved) {
    if ($asset.Relative -match $blockedRegex) {
        throw "Blocked identifier appears in asset path: $($asset.Relative)"
    }
    $matches = @(& rg -l -i --hidden --glob '!__pycache__/**' --glob '!*.pyc' -- $blockedRegex $asset.Source 2>$null)
    if ($matches.Count -gt 0) {
        throw "Blocked identifier appears in release asset $($asset.Name): $($matches[0])"
    }
}

[System.IO.Directory]::CreateDirectory($output) | Out-Null
$assetRows = foreach ($asset in $resolved) {
    $zipPath = Join-Path $output ($asset.Name + '.zip')
    Compress-Archive -LiteralPath $asset.Source -DestinationPath $zipPath -CompressionLevel Optimal
    $files = if (Test-Path -LiteralPath $asset.Source -PathType Container) {
        @(Get-ChildItem -LiteralPath $asset.Source -Recurse -Force -File -ErrorAction Stop)
    } else {
        @(Get-Item -LiteralPath $asset.Source -Force)
    }
    $zip = Get-Item -LiteralPath $zipPath
    [pscustomobject]@{
        name = $zip.Name
        source = $asset.Relative.Replace('\', '/')
        sourceFiles = $files.Count
        sourceBytes = [long](($files | Measure-Object Length -Sum).Sum)
        assetBytes = [long]$zip.Length
        sha256 = (Get-FileHash -LiteralPath $zipPath -Algorithm SHA256).Hash.ToLowerInvariant()
    }
}

$releaseManifest = [ordered]@{
    schema = 'player-analysis-public-release-assets-v1'
    createdAtUtc = [DateTimeOffset]::UtcNow.ToString('o')
    assets = @($assetRows)
}
[System.IO.File]::WriteAllText(
    (Join-Path $output 'release-assets.json'),
    ($releaseManifest | ConvertTo-Json -Depth 6),
    $utf8NoBom
)
$sums = ($assetRows | ForEach-Object { "$($_.sha256)  $($_.name)" }) -join "`n"
[System.IO.File]::WriteAllText((Join-Path $output 'SHA256SUMS.txt'), $sums + "`n", $utf8NoBom)
$assetRows | Format-Table -AutoSize
