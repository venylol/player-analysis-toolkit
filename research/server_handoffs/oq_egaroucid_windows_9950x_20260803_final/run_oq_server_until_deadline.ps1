param(
    [string]$PythonExe = "python.exe",
    [int]$MinimumExpansionMinutes = 45,
    [int]$BilateralTarget = 4000
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
[Console]::InputEncoding = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Desktop = [Environment]::GetFolderPath("Desktop")
$RootFull = [IO.Path]::GetFullPath($Root)
$DesktopFull = [IO.Path]::GetFullPath($Desktop)
if (-not $RootFull.StartsWith($DesktopFull, [StringComparison]::OrdinalIgnoreCase)) {
    throw "Extract the server package under the Windows Desktop before running. Current root: $RootFull"
}

$ChinaTimeZone = [TimeZoneInfo]::FindSystemTimeZoneById("China Standard Time")
function China-Time-To-Utc([string]$Text) {
    $unspecified = [DateTime]::SpecifyKind([DateTime]::ParseExact($Text, "yyyy-MM-dd HH:mm:ss", $null), [DateTimeKind]::Unspecified)
    return [TimeZoneInfo]::ConvertTimeToUtc($unspecified, $ChinaTimeZone)
}
$HardStopUtc = China-Time-To-Utc "2026-08-04 01:44:00"
$DeliveryDeadlineUtc = China-Time-To-Utc "2026-08-04 01:49:00"

$Games = "input\games\games.csv"
$Moves = "input\games\move_times.csv"
$HintsDir = "input\hints"
$Hints = "input\hints\position_hints.csv"
$Users = "input\oq_reversi_5min_rating_2000_users.csv"
$Engine = "engine\Egaroucid_for_Console_7_8_1_AVX512_AMD.exe"
$Analyzer = "research\analyze_oq_reversi_5min_hints.py"
$Puller = "research\pull_oq_reversi_5min_elo2000_games.py"
$StatusScript = "research\oq_dataset_status.py"
$Logs = Join-Path $Root "logs"
New-Item -ItemType Directory -Path $Logs -Force | Out-Null

foreach ($required in ($Games, $Moves, $Hints, $Users, $Engine, $Analyzer, $Puller, $StatusScript)) {
    if (-not (Test-Path -LiteralPath (Join-Path $Root $required))) {
        throw "Missing required package file: $required"
    }
}

$script:Workloads = @()
function Start-HiddenPython([string]$Name, [string[]]$Arguments) {
    $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $stdout = Join-Path $Logs "$Name`_$stamp.stdout.log"
    $stderr = Join-Path $Logs "$Name`_$stamp.stderr.log"
    $process = Start-Process -FilePath $PythonExe -ArgumentList $Arguments -WorkingDirectory $Root -RedirectStandardOutput $stdout -RedirectStandardError $stderr -WindowStyle Hidden -PassThru
    $script:Workloads += $process
    Write-Host "started $Name pid=$($process.Id) stdout=$stdout"
    return $process
}

function Start-Analyzer([int]$FollowTarget) {
    $arguments = @(
        "-u", "-B", $Analyzer,
        "--games", $Games,
        "--target-moves", $Moves,
        "--out-dir", $HintsDir,
        "--engine", $Engine,
        "--all-game-positions",
        "--direct",
        "--write-batch-games", "10",
        "--retries", "5",
        "--level18-workers", "4",
        "--level18-threads", "4",
        "--hint1-level", "2"
    )
    if ($FollowTarget -gt 0) {
        $arguments += @("--follow-until-game-count", "$FollowTarget", "--follow-poll-seconds", "10")
    }
    return Start-HiddenPython "analyzer" $arguments
}

function Start-Puller {
    $arguments = @(
        "-u", "-B", $Puller,
        "--users", $Users,
        "--out-dir", "input\games",
        "--direct",
        "--balanced-bilateral-min-existing-games", "11",
        "--target-bilateral-game-count", "$BilateralTarget",
        "--retries", "5",
        "--continue-on-fetch-error"
    )
    return Start-HiddenPython "puller" $arguments
}

function Stop-ProcessTree([Diagnostics.Process]$Process) {
    if (-not $Process) { return }
    $live = Get-Process -Id $Process.Id -ErrorAction SilentlyContinue
    if (-not $live) { return }
    $children = @(Get-CimInstance Win32_Process | Where-Object { $_.ParentProcessId -eq $Process.Id })
    foreach ($child in $children) {
        Stop-Process -Id $child.ProcessId -ErrorAction SilentlyContinue
    }
    Stop-Process -Id $Process.Id -ErrorAction SilentlyContinue
}

function Stop-AllWorkloads {
    foreach ($process in $script:Workloads) { Stop-ProcessTree $process }
    $engineRoot = [IO.Path]::GetFullPath((Join-Path $Root "engine"))
    $leftovers = @(Get-CimInstance Win32_Process | Where-Object {
        ($_.Name -like "Egaroucid_for_Console*") -and
        $_.ExecutablePath -and
        ([IO.Path]::GetFullPath($_.ExecutablePath)).StartsWith($engineRoot, [StringComparison]::OrdinalIgnoreCase)
    })
    foreach ($leftover in $leftovers) {
        Stop-Process -Id $leftover.ProcessId -ErrorAction SilentlyContinue
    }
    Start-Sleep -Seconds 2
}

function Sleep-Poll {
    $remainingMs = ($HardStopUtc - [DateTime]::UtcNow).TotalMilliseconds
    if ($remainingMs -le 500) { return }
    $sleepMs = [int][Math]::Min(30000, [Math]::Max(100, $remainingMs - 500))
    Start-Sleep -Milliseconds $sleepMs
}

function Get-Status {
    $json = & $PythonExe -B (Join-Path $Root $StatusScript) `
        --games (Join-Path $Root $Games) `
        --moves (Join-Path $Root $Moves) `
        --hints (Join-Path $Root $Hints) `
        --users (Join-Path $Root $Users) `
        --bilateral-target $BilateralTarget
    if ($LASTEXITCODE -ne 0) { throw "status script failed" }
    return ($json | ConvertFrom-Json)
}

function Write-Delivery {
    Stop-AllWorkloads
    $status = Get-Status
    $delivery = [ordered]@{
        schema = "oq-egaroucid-server-delivery-v1"
        written_at_utc = [DateTime]::UtcNow.ToString("o")
        hard_stop_china = "2026-08-04 01:44:00"
        delivery_deadline_china = "2026-08-04 01:49:00"
        package_root_on_desktop = $RootFull
        status = $status
        processes_stopped = $true
        result_files = @(
            "input/games/games.csv",
            "input/games/move_times.csv",
            "input/games/game_player_summaries.csv",
            "input/hints/position_hints.csv",
            "input/hints/progress.json",
            "logs"
        )
    }
    $json = $delivery | ConvertTo-Json -Depth 6
    [IO.File]::WriteAllText((Join-Path $Root "DELIVERY_STATUS.json"), $json + "`n", [Text.UTF8Encoding]::new($false))
    Write-Host "Delivery is already on Desktop: $RootFull"
    Write-Host ($status | ConvertTo-Json -Depth 4)
}

try {
    if ([DateTime]::UtcNow -ge $HardStopUtc) {
        Write-Delivery
        exit 0
    }

    # Stage 1: finish all currently recorded games, both players, without recomputing existing keys.
    $analyzerProcess = Start-Analyzer 0
    while ((Get-Process -Id $analyzerProcess.Id -ErrorAction SilentlyContinue) -and [DateTime]::UtcNow -lt $HardStopUtc) {
        Sleep-Poll
    }

    if ([DateTime]::UtcNow -lt $HardStopUtc) {
        $status = Get-Status
        $minutesLeft = ($HardStopUtc - [DateTime]::UtcNow).TotalMinutes
        if (($status.complete_bilateral_hint_games -eq $status.strict_games) -and
            ($status.current_both_ranked_games -lt $BilateralTarget) -and
            ($minutesLeft -ge $MinimumExpansionMinutes)) {
            # Stage 2: new-rule pull and all-position analysis run concurrently.
            $pullProcess = Start-Puller
            $followTarget = [int]$status.expected_strict_total_at_bilateral_target
            $analyzerProcess = Start-Analyzer $followTarget
            while ([DateTime]::UtcNow -lt $HardStopUtc) {
                $pullLive = Get-Process -Id $pullProcess.Id -ErrorAction SilentlyContinue
                $analyzerLive = Get-Process -Id $analyzerProcess.Id -ErrorAction SilentlyContinue
                if (-not $pullLive -and -not $analyzerLive) { break }
                if (-not $pullLive -and $analyzerLive) {
                    $current = Get-Status
                    if ($current.complete_bilateral_hint_games -eq $current.strict_games) {
                        Stop-ProcessTree $analyzerProcess
                        break
                    }
                }
                Sleep-Poll
            }
        }
    }
}
finally {
    # This is reached on normal completion, errors, Ctrl+C, and the hard deadline.
    Write-Delivery
    if ([DateTime]::UtcNow -ge $DeliveryDeadlineUtc) {
        Write-Warning "Delivery completed at or after the 01:49 China deadline; inspect logs immediately."
    }
}
