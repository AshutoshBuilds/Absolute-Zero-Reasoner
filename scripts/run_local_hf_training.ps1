param(
    [int]$Epochs = 100,
    [string]$CheckpointDir = "hf_checkpoints_qwen3_5b",
    [int]$Seed = 42,
    [int]$SeedTasksPerType = 0,
    [double]$CpuCap = 20,
    [string]$PythonExe = ".\azr_venv\Scripts\python.exe",
    [switch]$NoRich,
    [switch]$RunBenchmark,
    [string]$BaselineModelForBenchmark = "models\Qwen3.5-0.8B",
    [int]$BenchmarkLimit = 100,
    [string[]]$BenchmarkList = @("humaneval", "mbpp", "gsm8k"),
    [int]$BenchmarkSeed = 20260319,
    [int]$BenchmarkSamplesPerTask = 1,
    [int]$BenchmarkPassk = 1,
    [double]$BenchmarkTemperature = 0.2,
    [double]$BenchmarkTopP = 0.95,
    [double]$BenchmarkCpuCap = 20,
    [switch]$NoBenchmarkRich,
    [string]$RunLogRoot = "training_run_logs",
[double]$GpuMemoryFraction = 0.85,
[string]$CudaAllocConfig = "max_split_size_mb:128,garbage_collection_threshold:0.8",
    [switch]$UseSeparateValueModel,
    [int]$ResourceSampleSeconds = 20,
[switch]$Use4Bit,
    [string]$PythonExeForBenchmark = "",
    [Parameter(ValueFromRemainingArguments=$true)]
    [string[]]$ExtraArgs
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

function Resolve-CheckpointDir {
    param([string]$PathValue)

    if ([string]::IsNullOrWhiteSpace($PathValue)) {
        return "hf_checkpoints\hf_trainer_qwen3_5b"
    }

    $trimmed = $PathValue.Trim()

    # Keep absolute paths and explicit relative paths unchanged.
    if ([System.IO.Path]::IsPathRooted($trimmed)) {
        return $trimmed
    }
    if ($trimmed -match "[\\/]" ) {
        return $trimmed
    }

    # Legacy flat names like hf_trainer_checkpoints_* and hf_checkpoints_* are now grouped under hf_checkpoints/.
    if ($trimmed -match '^hf_(trainer_)?checkpoints_') {
        return Join-Path "hf_checkpoints" $trimmed
    }

    return $trimmed
}

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$resolvedPython = Resolve-Path $PythonExe -ErrorAction SilentlyContinue

if (-not $resolvedPython) {
    throw "Python executable not found: $PythonExe"
}

$CheckpointDir = Resolve-CheckpointDir $CheckpointDir

$trainerScript = Join-Path $projectRoot "hf_trainer.py"
if (-not (Test-Path $trainerScript)) {
    throw "hf_trainer.py not found at: $trainerScript"
}

if (-not [string]::IsNullOrWhiteSpace($RunLogRoot)) {
    $runLogRoot = Join-Path $projectRoot $RunLogRoot
} else {
    $runLogRoot = Join-Path $projectRoot "training_run_logs"
}

$runTimestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$runDir = Join-Path $runLogRoot ("local_hf_train_" + $runTimestamp)
New-Item -ItemType Directory -Path $runDir -Force | Out-Null

$resolvedPythonForBenchmark = $resolvedPython.Path
if (-not [string]::IsNullOrWhiteSpace($PythonExeForBenchmark)) {
    $benchPython = Resolve-Path $PythonExeForBenchmark -ErrorAction SilentlyContinue
    if (-not $benchPython) {
        throw "Benchmark Python executable not found: $PythonExeForBenchmark"
    }
    $resolvedPythonForBenchmark = $benchPython.Path
} else {
    $resolvedPythonForBenchmark = $resolvedPython.Path
}

$originalPythonIOEncoding = $env:PYTHONIOENCODING
$originalPythonUtf8 = $env:PYTHONUTF8
$originalAllocConfig = $env:PYTORCH_CUDA_ALLOC_CONF
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"
$env:PYTORCH_CUDA_ALLOC_CONF = $CudaAllocConfig

Write-Host "AZR Local HF Training Launcher"
Write-Host "Project root: $projectRoot"
Write-Host "Run directory: $runDir"
Write-Host "Using Python: $($resolvedPython.Path)"
Write-Host "Epochs: $Epochs"
Write-Host "Checkpoint dir: $CheckpointDir"
Write-Host "CPU cap: $CpuCap%"
if ($RunBenchmark) {
    Write-Host "Auto benchmark: enabled"
    Write-Host "Benchmark baseline: $BaselineModelForBenchmark"
}

if (-not (Test-Path $runDir)) {
    throw "Failed to create run output directory: $runDir"
}

$trainerOut = Join-Path $runDir "hf_trainer_stdout.log"
$trainerErr = Join-Path $runDir "hf_trainer_stderr.log"
$resourceLog = Join-Path $runDir "resource_log.tsv"
$trainSummary = Join-Path $runDir "run_summary.md"
$runConfig = Join-Path $runDir "run_config.json"

function Get-ResourceSample {
    param(
        [int]$ProcessId
    )

    $ts = (Get-Date -Format "yyyy-MM-dd HH:mm:ss")
    $cpuValue = "NA"
    $memValue = "NA"
    $gpuValue = "NA"

    try {
        $cpuValue = (Get-Counter -Counter "\Processor(_Total)\% Processor Time").CounterSamples[0].CookedValue
        $cpuValue = [Math]::Round($cpuValue, 2)
    } catch {
        # ignore
    }

    try {
        $memValue = (Get-Counter -Counter "\Memory\Available MBytes").CounterSamples[0].CookedValue
        $memValue = [Math]::Round($memValue, 2)
    } catch {
        # ignore
    }

    try {
        $g = nvidia-smi --query-gpu=index,name,utilization.gpu,utilization.memory,memory.total,memory.used --format=csv,noheader,nounits | Out-String
        $lines = $g -split "`r?`n" | Where-Object { $_.Trim() -ne "" }
        if ($lines.Count -gt 0) {
            $gpuValue = ($lines | ForEach-Object { ($_ -replace ",", " | ") }) -join " || "
        }
    } catch {
        # ignore
    }

    "$ts, $ProcessId, $cpuValue, $memValue, $gpuValue"
}

$resolvedCheckpointDir = Join-Path $projectRoot $CheckpointDir
if (-not (Test-Path $resolvedCheckpointDir)) {
    New-Item -ItemType Directory -Path $resolvedCheckpointDir -Force | Out-Null
}

$trainerScriptResolved = (Resolve-Path $trainerScript).Path
$trainerArgs = @(
    "`"$trainerScriptResolved`"",
    "--epochs", [string]$Epochs,
    "--checkpoint-dir", $CheckpointDir,
    "--seed", [string]$Seed,
    "--seed-tasks-per-type", [string]$SeedTasksPerType,
    "--cpu-cap", [string]$CpuCap
)

if ($NoRich) { $trainerArgs += "--no-rich" } else { $trainerArgs += "--rich" }
if ($Use4Bit) { $trainerArgs += "--use-4bit" }
$trainerArgs += "--gpu-memory-fraction"
$trainerArgs += [string]$GpuMemoryFraction
$trainerArgs += "--cuda-alloc-config"
$trainerArgs += $CudaAllocConfig
if ($ExtraArgs) { $trainerArgs += $ExtraArgs }

$commandLine = "$($resolvedPython.Path) " + ($trainerArgs -join " ")
$cmdLineOut = "Command: $commandLine"
Set-Content -Path $trainSummary -Value $cmdLineOut
Add-Content -Path $resourceLog -Value "timestamp,pid,cpu_percent,available_mem_mb,gpu_sample"
Add-Content -Path $runConfig -Value "{" | Out-Null
Add-Content -Path $runConfig -Value "`"epochs`": $Epochs," | Out-Null
Add-Content -Path $runConfig -Value "`"checkpoint_dir`": `"$CheckpointDir`"," | Out-Null
Add-Content -Path $runConfig -Value "`"seed`": $Seed," | Out-Null
Add-Content -Path $runConfig -Value "`"seed_tasks_per_type`": $SeedTasksPerType," | Out-Null
Add-Content -Path $runConfig -Value "`"cpu_cap`": $CpuCap," | Out-Null
Add-Content -Path $runConfig -Value "`"gpu_memory_fraction`": $GpuMemoryFraction," | Out-Null
Add-Content -Path $runConfig -Value "`"cuda_alloc_config`": `"$CudaAllocConfig`"," | Out-Null
Add-Content -Path $runConfig -Value "`"use_4bit`": $($Use4Bit.ToString().ToLower())," | Out-Null
Add-Content -Path $runConfig -Value "`"timestamp`": `"$runTimestamp`"" | Out-Null
Add-Content -Path $runConfig -Value "}" | Out-Null

Write-Host $cmdLineOut
Write-Host "Logs: $runDir"
Write-Host "Starting process..."

$proc = Start-Process `
    -FilePath $resolvedPython.Path `
    -ArgumentList $trainerArgs `
    -PassThru `
    -NoNewWindow `
    -RedirectStandardOutput $trainerOut `
    -RedirectStandardError $trainerErr

if (-not $proc) {
    throw "Failed to start trainer process."
}

$trainerPid = $proc.Id
Write-Host "Trainer PID: $trainerPid"

try {
    while ($true) {
        $p = Get-Process -Id $trainerPid -ErrorAction SilentlyContinue
        if (-not $p) {
            break
        }
        Add-Content -Path $resourceLog -Value (Get-ResourceSample -ProcessId $trainerPid)
        Start-Sleep -Seconds $ResourceSampleSeconds
    }
} finally {
    Add-Content -Path $resourceLog -Value "ended,$trainerPid,$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
}

$proc.WaitForExit()
$proc.Refresh()
$exitCode = if ($proc.HasExited -and $null -ne $proc.ExitCode) { [int]$proc.ExitCode } else { -1 }

if ($exitCode -lt 0) {
    # Fallback for cases where Start-Process does not surface ExitCode reliably.
    # Treat explicit successful completion marker in trainer logs as success.
    $lastLine = Get-Content -Path $trainerOut -Tail 1 -ErrorAction SilentlyContinue
    if ($lastLine -like "*Training run finished.*") {
        $exitCode = 0
    }
}
Add-Content -Path $trainSummary -Value "Training exit code: $exitCode" 

if ($exitCode -ne 0) {
    Write-Host "Training failed (exit code $exitCode). Check logs:"
    Write-Host "stdout: $trainerOut"
    Write-Host "stderr: $trainerErr"
    throw "Training run failed with exit code $exitCode"
}

Write-Host "Training complete. Exit code: $exitCode"

if ($RunBenchmark) {
    $benchmarkScript = Join-Path $projectRoot "run_pre_post_benchmarks.py"
    if (-not (Test-Path $benchmarkScript)) {
        throw "Benchmark script not found: $benchmarkScript"
    }

    $resultRoot = Join-Path $projectRoot ("evaluation_results\comparison\run_local_" + $runTimestamp)
    New-Item -ItemType Directory -Path $resultRoot -Force | Out-Null

    $benchResultsDir = Join-Path $resultRoot ("comparison_" + $runTimestamp)
    New-Item -ItemType Directory -Path $benchResultsDir -Force | Out-Null

    $latestCheckpoint = Get-ChildItem -Path $resolvedCheckpointDir -Filter "checkpoint_epoch_*" |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1

    if (-not $latestCheckpoint) {
        throw "No checkpoint found in $resolvedCheckpointDir for post-run benchmark."
    }

    $improvedModel = $latestCheckpoint.FullName
    Write-Host "Selected improved model from latest checkpoint: $improvedModel"

    $benchmarkScriptResolved = (Resolve-Path $benchmarkScript).Path
    $benchOut = Join-Path $runDir "benchmark_stdout.log"
    $benchErr = Join-Path $runDir "benchmark_stderr.log"

    $benchArgs = @(
        "`"$benchmarkScriptResolved`"",
        "--baseline-model", $BaselineModelForBenchmark,
        "--improved-model", $improvedModel,
        "--results-root", $benchResultsDir,
        "--benchmarks"
    ) + $BenchmarkList + @(
        "--limit", [string]$BenchmarkLimit,
        "--samples-per-task", [string]$BenchmarkSamplesPerTask,
        "--passk", [string]$BenchmarkPassk,
        "--temperature", [string]$BenchmarkTemperature,
        "--top-p", [string]$BenchmarkTopP,
        "--seed", [string]$BenchmarkSeed,
        "--cpu-cap", [string]$BenchmarkCpuCap
    )

    if ($NoBenchmarkRich) { $benchArgs += "--no-rich" } else { $benchArgs += "--rich" }
    if ($UseSeparateValueModel) { $benchArgs += "--use-separate-value-model" }

    $benchCommand = "$($resolvedPythonForBenchmark) " + ($benchArgs -join " ")
    Add-Content -Path $trainSummary -Value "Benchmark command: $benchCommand"
    Write-Host "Benchmark command: $benchCommand"
    Write-Host "Running post-training benchmark..."

    $benchProc = Start-Process `
        -FilePath $resolvedPythonForBenchmark `
        -ArgumentList $benchArgs `
        -NoNewWindow `
        -Wait `
        -RedirectStandardOutput $benchOut `
        -RedirectStandardError $benchErr

    if ($benchProc.ExitCode -ne 0) {
        Write-Host "Benchmark failed (exit code $($benchProc.ExitCode))."
        Write-Host "Benchmark stdout: $benchOut"
        Write-Host "Benchmark stderr: $benchErr"
        throw "Benchmark run failed with exit code $($benchProc.ExitCode)"
    }

    Write-Host "Benchmark complete. Results in: $benchResultsDir"
    Write-Host "Benchmark logs: $benchOut / $benchErr"
}

Write-Host "Done. All outputs in: $runDir"

if ($null -eq $originalPythonIOEncoding) {
    Remove-Item Env:PYTHONIOENCODING -ErrorAction SilentlyContinue
} else {
    $env:PYTHONIOENCODING = $originalPythonIOEncoding
}

if ($null -eq $originalPythonUtf8) {
    Remove-Item Env:PYTHONUTF8 -ErrorAction SilentlyContinue
} else {
    $env:PYTHONUTF8 = $originalPythonUtf8
}

if ($null -eq $originalAllocConfig) {
    Remove-Item Env:PYTORCH_CUDA_ALLOC_CONF -ErrorAction SilentlyContinue
} else {
    $env:PYTORCH_CUDA_ALLOC_CONF = $originalAllocConfig
}
