param(
    [Parameter(Mandatory = $true)][string]$ServerRoot,
    [string]$PythonExe = '',
    [string]$AttemptName = 'oq_hint6_incremental_exact_index_snapshot_stm_v1',
    [switch]$PreflightOnly
)

$ErrorActionPreference = 'Stop'
$OutputEncoding = [Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
$PackageRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$ServerRoot = (Resolve-Path -LiteralPath $ServerRoot).Path
if ([string]::IsNullOrWhiteSpace($PythonExe)) {
    $python = Get-Command python.exe -ErrorAction SilentlyContinue | Select-Object -First 1
    if (-not $python) { throw 'python.exe not found; supply -PythonExe C:\path\to\python.exe' }
    $PythonExe = $python.Source
}
$PythonExe = (Resolve-Path -LiteralPath $PythonExe).Path

$Pipeline = Join-Path $PackageRoot 'scripts\incremental_hint6_pipeline.py'
$Materializer = Join-Path $PackageRoot 'scripts\materialize_oq_tcn_model_ready.py'
$Verifier = Join-Path $PackageRoot 'scripts\verify_package.py'
$Source = Join-Path $PackageRoot 'assets\source\handoff\raw_nodes_with_pass.csv'
$Handoff = Join-Path $PackageRoot 'assets\source\handoff'
$Snapshot = Join-Path $PackageRoot 'assets\source\source_snapshot'
$LocalW12 = Join-Path $PackageRoot 'evidence\hint6_local_w12'
$LocalW1 = Join-Path $PackageRoot 'evidence\hint6_local_w1'
$Legacy = Join-Path $PackageRoot 'evidence\legacy_hint6_exact_index_seed.jsonl'
$ServerHint1 = Join-Path $ServerRoot 'work\hint1'
$ServerHint6 = Join-Path $ServerRoot 'work\hint6'
$Engine = Join-Path $ServerRoot 'assets\engine\Egaroucid_for_Console_7_8_1_AVX512_AMD.exe'
$Runner = Join-Path $ServerRoot 'app\scripts\portable_safe_recompute.py'
$FinalValidator = Join-Path $ServerRoot 'app\scripts\validate_server_model_ready.py'
$Attempt = Join-Path $ServerRoot (Join-Path 'incremental_attempts' $AttemptName)
$PrepareManifest = Join-Path $Attempt 'prepare_manifest.json'
$MissingSource = Join-Path $Attempt 'missing_hint6_source.csv'
$NewHint6 = Join-Path $Attempt 'new_hint6_safe'
$AssembledDir = Join-Path $Attempt 'results\assembled'
$AssembledCsv = Join-Path $AssembledDir 'raw_nodes_with_pass_incremental_hints.csv'
$AssemblyManifest = Join-Path $AssembledDir 'assembly_manifest.json'
$ModelReadyDir = Join-Path $Attempt 'results\model_ready_snapshot_stm_v1'
$FinalValidation = Join-Path $ModelReadyDir 'server_final_validation.json'
$StatusPath = Join-Path $ServerRoot 'INCREMENTAL_DELIVERY_STATUS.json'

function Write-Status([string]$Status, [string]$Stage, [string]$Detail) {
    $value = [ordered]@{
        schema = 'oq-tcn-incremental-data-prep-status-v1'; status = $Status; stage = $Stage
        detail = $Detail; attempt = $Attempt; trainingStarted = $false; cudaUsed = $false
        workers = 12; threadsPerConsole = 16; hashLevel = 25; updatedAt = (Get-Date).ToString('o')
    } | ConvertTo-Json -Depth 10
    [System.IO.File]::WriteAllText($StatusPath, $value + [Environment]::NewLine, [System.Text.UTF8Encoding]::new($false))
}

function Invoke-Python {
    param([Parameter(ValueFromRemainingArguments = $true)][string[]]$Arguments)
    & $PythonExe @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Python failed ($LASTEXITCODE): $($Arguments -join ' ')" }
}

try {
    Write-Status 'running' 'preflight' 'verifying incremental package and requiring the old run to be stopped'
    Invoke-Python $Verifier --bundle-root $PackageRoot --manifest (Join-Path $PackageRoot 'package_manifest.json')
    Invoke-Python '-c' 'import numpy,pandas,torch,sklearn,scipy'
    $engines = @(Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'Egaroucid_for_Console_7_8_1_AVX512_AMD.exe' })
    if ($engines.Count -ne 0) {
        throw "Stop the existing run first. Egaroucid processes are still active: $($engines.ProcessId -join ',')"
    }
    foreach ($required in @($ServerHint1,$ServerHint6,$Engine,$Runner,$FinalValidator)) {
        if (-not (Test-Path -LiteralPath $required)) { throw "Required server-v3 asset missing: $required" }
    }

    if (-not (Test-Path -LiteralPath $PrepareManifest)) {
        if (Test-Path -LiteralPath $Attempt) { throw "Incomplete attempt directory exists without manifest: $Attempt" }
        Write-Status 'running' 'exact-index-merge-preparation' 'pairing exact game_id and original pass-inclusive move_index before any reuse'
        Invoke-Python $Pipeline prepare --source-csv $Source --server-hint6 $ServerHint6 --local-w12 $LocalW12 --local-w1 $LocalW1 --legacy-seed $Legacy --attempt-dir $Attempt
    }
    $prepared = Get-Content -Raw -Encoding UTF8 $PrepareManifest | ConvertFrom-Json
    if (-not $prepared.ok -or $prepared.indexContract.boardOnlyRemappingUsed) { throw 'exact-index preparation gate failed' }
    if ($PreflightOnly) {
        Write-Status 'preflight-complete' 'missing-work-ready' "Exact-index merge passed; $($prepared.missingRows) nodes remain. No Console launched."
        return
    }

    Write-Status 'running' 'hint6-missing-recompute' "Computing only $($prepared.missingRows) missing nodes with 12 Consoles x 16 threads"
    Invoke-Python $Runner run --source-csv $MissingSource --stage hint6 --engine $Engine --output-dir $NewHint6 --workers 12 --hash-level 25 --batch-size 128 --timeout 900 --max-attempts 2 --resume
    Invoke-Python $Runner audit --output-dir $NewHint6 --expected ([int64]$prepared.missingRows)

    if (-not (Test-Path -LiteralPath $AssemblyManifest)) {
        New-Item -ItemType Directory -Path $AssembledDir -Force | Out-Null
        Write-Status 'running' 'audited-assembly' 'server > local-w12 > local-w1 > exact-index legacy > new-compute; exact keys only'
        Invoke-Python $Pipeline assemble --source-csv $Source --server-hint6 $ServerHint6 --local-w12 $LocalW12 --local-w1 $LocalW1 --legacy-seed $Legacy --hint1-dir $ServerHint1 --new-hint6 $NewHint6 --output-csv $AssembledCsv --output-manifest $AssemblyManifest --merge-index (Join-Path $AssembledDir 'merge_index.sqlite3')
    }

    if (-not (Test-Path -LiteralPath $ModelReadyDir)) {
        Write-Status 'running' 'materialization' 'building 362 numeric features and 23 board channels; no training'
        Invoke-Python $Materializer --handoff-dir $Handoff --raw-nodes $AssembledCsv --split-manifest (Join-Path $Handoff 'split_manifest.csv') --output-dir $ModelReadyDir --base-checkpoint (Join-Path $Snapshot 'tcn_board_cnn_time_model_best.pt') --preprocessing (Join-Path $Snapshot 'preprocessing.json') --source-research (Join-Path $Snapshot 'official_research') --human-opening-book (Join-Path $Snapshot 'othelloquest_human_frequency_nodes_ply1_30_min5.runtime.json') --output-name 'model_ready_10000_snapshot_stm_v1.npz' --workers 12 --allow-screened-legacy-hint6
    }

    Write-Status 'running' 'final-data-contract' 'validating TCN-ready arrays and checkpoint compatibility on CPU'
    Invoke-Python $FinalValidator --data (Join-Path $ModelReadyDir 'model_ready_10000_snapshot_stm_v1.npz') --context (Join-Path $ModelReadyDir 'position_context_metadata.csv') --checkpoint (Join-Path $Snapshot 'tcn_board_cnn_time_model_best.pt') --preprocessing (Join-Path $Snapshot 'preprocessing.json') --materialization-manifest (Join-Path $ModelReadyDir 'materialization_manifest.json') --output $FinalValidation
    $final = Get-Content -Raw -Encoding UTF8 $FinalValidation | ConvertFrom-Json
    if (-not $final.ok -or [int]$final.inputFeatures -ne 362 -or [int]$final.boardChannels -ne 23) { throw 'final data contract failed' }
    Write-Status 'complete' 'data-ready' 'Merged full training data is ready for TCN use; no model training started'
}
catch {
    Write-Status 'failed' 'blocked' $_.Exception.ToString()
    throw
}
