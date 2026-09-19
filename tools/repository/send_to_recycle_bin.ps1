[CmdletBinding(SupportsShouldProcess = $true)]
param(
    [Parameter(Mandatory = $true)]
    [string[]]$Path,

    [Parameter(Mandatory = $true)]
    [string]$AllowedRoot
)

$ErrorActionPreference = 'Stop'
$root = [System.IO.Path]::GetFullPath((Resolve-Path -LiteralPath $AllowedRoot).Path)
$separator = [System.IO.Path]::DirectorySeparatorChar

$targets = foreach ($candidate in $Path) {
    $resolved = [System.IO.Path]::GetFullPath((Resolve-Path -LiteralPath $candidate).Path)
    if ($resolved.Equals($root, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to recycle the allowed root itself: $resolved"
    }
    if (-not $resolved.StartsWith($root + $separator, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Recycle target is outside AllowedRoot: $resolved"
    }
    Get-Item -LiteralPath $resolved -Force
}

Add-Type -AssemblyName Microsoft.VisualBasic
foreach ($target in $targets) {
    if (-not $PSCmdlet.ShouldProcess($target.FullName, 'Move to Windows Recycle Bin')) {
        continue
    }
    if ($target.PSIsContainer) {
        [Microsoft.VisualBasic.FileIO.FileSystem]::DeleteDirectory(
            $target.FullName,
            [Microsoft.VisualBasic.FileIO.UIOption]::OnlyErrorDialogs,
            [Microsoft.VisualBasic.FileIO.RecycleOption]::SendToRecycleBin
        )
    } else {
        [Microsoft.VisualBasic.FileIO.FileSystem]::DeleteFile(
            $target.FullName,
            [Microsoft.VisualBasic.FileIO.UIOption]::OnlyErrorDialogs,
            [Microsoft.VisualBasic.FileIO.RecycleOption]::SendToRecycleBin
        )
    }
    [pscustomobject]@{
        path = $target.FullName
        type = if ($target.PSIsContainer) { 'directory' } else { 'file' }
        destination = 'Windows Recycle Bin'
    }
}
