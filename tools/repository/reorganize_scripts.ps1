[CmdletBinding(SupportsShouldProcess = $true)]
param(
    [Parameter(Mandatory = $true)]
    [string]$Root,

    [Parameter(Mandatory = $true)]
    [string]$ManifestPath
)

$ErrorActionPreference = 'Stop'
$rootPath = [System.IO.Path]::GetFullPath($Root)
$manifest = Get-Content -LiteralPath $ManifestPath -Raw -Encoding UTF8 | ConvertFrom-Json

foreach ($move in $manifest.moves) {
    $source = [System.IO.Path]::GetFullPath((Join-Path $rootPath ([string]$move.from)))
    $destination = [System.IO.Path]::GetFullPath((Join-Path $rootPath ([string]$move.to)))
    if (-not $source.StartsWith($rootPath, [System.StringComparison]::OrdinalIgnoreCase) -or
        -not $destination.StartsWith($rootPath, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Move escapes toolkit root: $($move.from)"
    }
    if (-not (Test-Path -LiteralPath $source -PathType Leaf)) {
        throw "Source file is missing: $($move.from)"
    }
    if (Test-Path -LiteralPath $destination) {
        throw "Destination already exists: $($move.to)"
    }
}

foreach ($move in $manifest.moves) {
    $source = Join-Path $rootPath ([string]$move.from)
    $destination = Join-Path $rootPath ([string]$move.to)
    if ($PSCmdlet.ShouldProcess($source, "Move to $destination")) {
        [System.IO.Directory]::CreateDirectory((Split-Path -Parent $destination)) | Out-Null
        Move-Item -LiteralPath $source -Destination $destination
    }
}
