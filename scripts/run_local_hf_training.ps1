# .SYNOPSIS
#   Launch local Hugging Face PPO-style training and optional benchmarks for AZR-style experiments.
# .DESCRIPTION
#   This script prefers configuration from `.env` (project root) and allows each value to be overridden
#   via explicit script parameters. It creates a per-run log directory, streams live trainer output,
#   optionally benchmarks the latest checkpoint, and writes both command/run metadata.
# .PARAMETER ForceRestart
#   When set, existing checkpoint_epoch_* directories inside the checkpoint root are deleted before launch
#   so training starts from a clean epoch 0 state.
# .PARAMETER RunBenchmark
#   When enabled, runs `run_pre_post_benchmarks.py` against the newly produced latest checkpoint.
param(
    [int]$Epochs = 100,
    [string]$CheckpointDir = "hf_checkpoints\\Qwen3-0.6B",
    [int]$Seed = 42,
    [int]$SeedTasksPerType = 0,
    [double]$CpuCap = 20,
    [string]$PythonExe = ".\azr_venv\Scripts\python.exe",
    [switch]$NoRich,
    [switch]$ForceRestart,
    [switch]$RunBenchmark,
    [string]$BaselineModelForBenchmark = "models\Qwen3-0.6B",
    [int]$BenchmarkLimit = 100,
    [string[]]$BenchmarkList = @("humaneval", "mbpp", "gsm8k"),
    [int]$BenchmarkSeed = 20260319,
    [int]$BenchmarkSamplesPerTask = 1,
    [int]$BenchmarkPassk = 1,
    [double]$BenchmarkTemperature = 0.2,
    [double]$BenchmarkTopP = 0.95,
    [double]$BenchmarkCpuCap = 20,
    [string]$BaselineProgrambenchRunDir = "",
    [string]$ImprovedProgrambenchRunDir = "",
    [switch]$NoBenchmarkRich,
    [string]$RunLogRoot = "training_run_logs",
    [double]$GpuMemoryFraction = 0.85,
    [string]$CudaAllocConfig = "max_split_size_mb:128,garbage_collection_threshold:0.8",
    [bool]$UseSeparateValueModel = $true,
    [ValidateSet("fp16","bf16","fp32","auto")]
    [string]$ModelDtype = "auto",
    [int]$ResourceSampleSeconds = 20,
    [switch]$Use4Bit,
    [string]$PythonExeForBenchmark = "",
    [Parameter(ValueFromRemainingArguments=$true)]
[string[]]$ExtraArgs
)

# Fail fast and keep progress UI clean during long training / benchmark runs.
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

# ----------------------------- [1/8] Input helpers ------------------------------
function Read-DotEnv {
    param(
        [string]$Path
    )

    $values = @{}
    if (-not (Test-Path $Path)) {
        return $values
    }

    foreach ($line in (Get-Content -Path $Path)) {
        if ([string]::IsNullOrWhiteSpace($line)) { continue }
        $trimmed = $line.Trim()
        if ($trimmed.StartsWith("#")) { continue }

        if ($trimmed -match "^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=(.*)$") {
            $key = $matches[1]
            $value = $matches[2].Trim()
            if ($value.Length -ge 2) {
                if (($value[0] -eq '"' -and $value[-1] -eq '"') -or ($value[0] -eq "'" -and $value[-1] -eq "'")) {
                    $value = $value.Substring(1, $value.Length - 2)
                }
            }
            $values[$key] = $value
        }
    }

    return $values
}

function Get-EnvSetting {
    param(
        [hashtable]$Source,
        [string]$Name
    )
    if ($Source.ContainsKey($Name)) {
        return $Source[$Name]
    }
    return $null
}

function Parse-EnvBool {
    param(
        [string]$Value,
        [string]$Name
    )
    if ([string]::IsNullOrWhiteSpace($Value)) {
        return $null
    }
    switch -Regex ($Value.Trim().ToLower()) {
        "^(1|true|yes|on)$" { return $true }
        "^(0|false|no|off)$" { return $false }
        default { throw "Invalid boolean value for ${Name}: '${Value}'. Use true/false, 1/0, yes/no, on/off." }
    }
}

function Parse-EnvStringList {
    param([string]$Value)
    if ([string]::IsNullOrWhiteSpace($Value)) { return @() }
    return $Value -split "," | ForEach-Object { $_.Trim() } | Where-Object { -not [string]::IsNullOrWhiteSpace($_) }
}

function Write-SectionDivider {
    param(
        [string]$Title
    )
    $line = "=" * 16
    Write-Host ""
    Write-Host "$line $Title $line"
}

# Build the Win32 command-line tail for CreateProcess using the same rules as the MSVC runtime
# (see Python subprocess.list2cmdline). Needed on Windows PowerShell 5.1 where Start-Process
# -ArgumentList can split tokens that contain spaces even when passed as a string array.
function ConvertTo-CmdLineFromTokens {
    param(
        [Parameter(Mandatory)]$Tokens
    )

    # Coerce to a flat list of strings (handles nested arrays from splatting and avoids
    # iterating a single string as char-by-char tokens).
    $flat = New-Object System.Collections.Generic.List[string]
    foreach ($outer in @($Tokens)) {
        if ($null -eq $outer) { continue }
        if ($outer -is [string]) {
            $flat.Add([string]$outer)
            continue
        }
        if ($outer -is [System.Collections.IEnumerable]) {
            foreach ($inner in $outer) {
                if ($null -ne $inner) { $flat.Add([string]$inner) }
            }
        }
        else {
            $flat.Add([string]$outer)
        }
    }

    $parts = New-Object System.Collections.Generic.List[string]
    foreach ($raw in $flat) {
        if ($null -eq $raw) { continue }
        $s = [string]$raw
        if ($s.Length -eq 0) {
            $parts.Add('""')
            continue
        }

        $mustQuote = $false
        foreach ($ch in $s.ToCharArray()) {
            if ($ch -eq ' ' -or $ch -eq "`t" -or $ch -eq '"') {
                $mustQuote = $true
                break
            }
        }

        if (-not $mustQuote) {
            $parts.Add($s)
            continue
        }

        $parts.Add('"' + ($s -replace '"', '""') + '"')
    }

    return [string]::Join(' ', $parts)
}

# Flattens nested splatted argv segments into a single string[] so human/MSVC command lines never
# glue flags to paths (e.g. checkpoint_epoch_19--results-root) and nested $ExtraArgs arrays do not
# stringify as a single token. Regression: PS 5.1 + Start-Process-style argv handling.
function ConvertTo-AzrFlatStringArgv {
    param(
        [Parameter(Mandatory = $true)]
        [AllowEmptyCollection()]
        [object[]]$Segments
    )

    $out = New-Object System.Collections.Generic.List[string]

    function WalkOne($node) {
        if ($null -eq $node) { return }
        if ($node -is [string]) {
            if ($node.Length -gt 0) { $null = $out.Add([string]$node) }
            return
        }
        if ($node -is [char]) {
            $null = $out.Add([string]$node)
            return
        }
        if ($node -is [System.Collections.IEnumerable]) {
            foreach ($child in $node) { WalkOne $child }
            return
        }
        $s = [string]$node
        if ($s.Length -gt 0) { $null = $out.Add($s) }
    }

    foreach ($top in @($Segments)) {
        WalkOne $top
    }

    return ,$out.ToArray()
}

# Starts a native child with RedirectStandardOutput/Error and copies both streams to disk
# asynchronously (avoids deadlocks). Prefer over Start-Process when argv tokens may contain spaces.
function Start-AzrRedirectedNativeProcess {
    param(
        [Parameter(Mandatory)][string]$FilePath,
        [Parameter(Mandatory)][string[]]$Tokens,
        [Parameter(Mandatory)][string]$WorkingDirectory,
        [Parameter(Mandatory)][string]$StdoutPath,
        [Parameter(Mandatory)][string]$StderrPath
    )

    $psi = [System.Diagnostics.ProcessStartInfo]::new()
    $psi.FileName = $FilePath
    $psi.WorkingDirectory = $WorkingDirectory
    $psi.UseShellExecute = $false
    $psi.RedirectStandardOutput = $true
    $psi.RedirectStandardError = $true
    $psi.CreateNoWindow = $true
    $psi.Arguments = ConvertTo-CmdLineFromTokens $Tokens

    $proc = [System.Diagnostics.Process]::new()
    $proc.EnableRaisingEvents = $true
    $proc.StartInfo = $psi

    $outFs = [System.IO.File]::Open(
        $StdoutPath,
        [System.IO.FileMode]::Create,
        [System.IO.FileAccess]::Write,
        [System.IO.FileShare]::ReadWrite
    )
    $errFs = [System.IO.File]::Open(
        $StderrPath,
        [System.IO.FileMode]::Create,
        [System.IO.FileAccess]::Write,
        [System.IO.FileShare]::ReadWrite
    )

    if (-not $proc.Start()) {
        try { $outFs.Close() } catch {}
        try { $errFs.Close() } catch {}
        throw "Failed to start process: $FilePath"
    }

    $copyOut = $proc.StandardOutput.BaseStream.CopyToAsync($outFs)
    $copyErr = $proc.StandardError.BaseStream.CopyToAsync($errFs)

    return [pscustomobject]@{
        Process  = $proc
        CopyOut  = $copyOut
        CopyErr  = $copyErr
        OutStream = $outFs
        ErrStream = $errFs
    }
}

function Complete-AzrRedirectedNativeProcess {
    param(
        [Parameter(Mandatory)]$Handle
    )

    $p = $Handle.Process
    if (-not $p.HasExited) {
        $null = $p.WaitForExit()
    }

    $Handle.CopyOut.GetAwaiter().GetResult() | Out-Null
    $Handle.CopyErr.GetAwaiter().GetResult() | Out-Null

    try { $Handle.OutStream.Flush() } catch {}
    try { $Handle.ErrStream.Flush() } catch {}
    try { $Handle.OutStream.Close() } catch {}
    try { $Handle.ErrStream.Close() } catch {}
}

# After Ctrl+C, wait briefly then terminate the native child and descendants (Windows job/tree).
# Nested Python worker processes are not always children of the launcher PID; those may survive.
function Stop-AzrNativeChildTree {
    param(
        [Parameter(Mandatory)][System.Diagnostics.Process]$Process,
        [int]$GraceMilliseconds = 3000
    )

    if ($null -eq $Process) { return }
    try { $Process.Refresh() } catch { }
    if ($Process.HasExited) { return }

    try {
        $null = $Process.WaitForExit([Math]::Max(0, $GraceMilliseconds))
    } catch { }

    if ($Process.HasExited) { return }

    try {
        $Process.Kill($true)
    } catch {
        try {
            $Process.Kill()
        } catch { }
    }

    if (-not $Process.HasExited) {
        $leafPid = $Process.Id
        Start-Process -FilePath "taskkill.exe" `
            -ArgumentList @("/PID", "$leafPid", "/T", "/F") `
            -Wait -NoNewWindow `
            -ErrorAction SilentlyContinue | Out-Null
    }

    try { $null = $Process.WaitForExit(15000) } catch { }
}

function Wait-AzrSleepCancellable {
    param(
        [Parameter(Mandatory)][double]$TotalSeconds,
        [int]$SliceMilliseconds = 200
    )

    $endUtc = [datetime]::UtcNow.AddSeconds($TotalSeconds)
    while ([datetime]::UtcNow -lt $endUtc) {
        if ($script:AzrCancelRequested) {
            return
        }
        $remainingMs = ($endUtc - [datetime]::UtcNow).TotalMilliseconds
        if ($remainingMs -le 0) {
            break
        }
        $sleepMs = [Math]::Min($SliceMilliseconds, [int][Math]::Ceiling($remainingMs))
        if ($sleepMs -gt 0) {
            Start-Sleep -Milliseconds $sleepMs
        }
    }
}

function Register-AzrConsoleCancel {
    $script:AzrCancelRequested = $false
    $script:AzrCancelPressCount = 0
    $script:AzrCancelKeyDelegate = $null

    # Use Console's event API (`add_CancelKeyPress` / `remove_CancelKeyPress`), not `CancelKeyPress +=`:
    # in Windows PowerShell 5.1 and pwsh, `CancelKeyPress` is not a settable property and `+=`
    # surfaces "property cannot be found". A stored delegate enables clean removal (no leak).
    $handlerScript = {
        param([object]$sender, [System.ConsoleCancelEventArgs]$e)

        $script:AzrCancelPressCount++
        if ($script:AzrCancelPressCount -le 1) {
            $e.Cancel = $true
            $script:AzrCancelRequested = $true
            Write-Host ""
            Write-Host "[azr] Cancel requested (Ctrl+C); stopping subprocess..." -ForegroundColor Yellow
            return
        }

        Write-Host "[azr] Second Ctrl+C: forcing shell exit." -ForegroundColor Red
        $e.Cancel = $false
    }

    try {
        $delegate = [System.ConsoleCancelEventHandler]$handlerScript
        [Console]::add_CancelKeyPress($delegate)
        $script:AzrCancelKeyDelegate = $delegate
    } catch {
        Write-Host ("[azr] Warning: could not attach Ctrl+C handler ({0}); interrupt behavior may be limited." -f $_.Exception.Message) -ForegroundColor DarkYellow
        $script:AzrCancelKeyDelegate = $null
    }
}

function Unregister-AzrConsoleCancel {
    if ($null -ne $script:AzrCancelKeyDelegate) {
        try {
            [Console]::remove_CancelKeyPress($script:AzrCancelKeyDelegate)
        } catch { }
        $script:AzrCancelKeyDelegate = $null
    }
}

function Restore-AzrLauncherProcessEnv {
    param(
        $OriginalPythonIOEncoding,
        $OriginalPythonUtf8,
        $OriginalAllocConfig
    )

    if ($null -eq $OriginalPythonIOEncoding) {
        Remove-Item Env:PYTHONIOENCODING -ErrorAction SilentlyContinue
    } else {
        $env:PYTHONIOENCODING = $OriginalPythonIOEncoding
    }

    if ($null -eq $OriginalPythonUtf8) {
        Remove-Item Env:PYTHONUTF8 -ErrorAction SilentlyContinue
    } else {
        $env:PYTHONUTF8 = $OriginalPythonUtf8
    }

    if ($null -eq $OriginalAllocConfig) {
        Remove-Item Env:PYTORCH_CUDA_ALLOC_CONF -ErrorAction SilentlyContinue
    } else {
        $env:PYTORCH_CUDA_ALLOC_CONF = $OriginalAllocConfig
    }
}

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$envConfig = Read-DotEnv -Path (Join-Path $projectRoot ".env")

# ------------------------- [2/8] Merge .env into parameters --------------------
# For every supported setting, keep CLI values when explicitly passed; otherwise
# inherit `.env` values for local-only workflows.
Write-SectionDivider -Title "ENV CONFIG RESOLUTION"
if (-not $PSBoundParameters.ContainsKey("PythonExe")) {
    $envValue = Get-EnvSetting -Source $envConfig -Name "AZR_PYTHON_EXE"
    if (-not [string]::IsNullOrWhiteSpace($envValue)) { $PythonExe = $envValue }
}

if (-not $PSBoundParameters.ContainsKey("Epochs")) {
    $envValue = Get-EnvSetting -Source $envConfig -Name "AZR_EPOCHS"
    if (-not [string]::IsNullOrWhiteSpace($envValue)) { $Epochs = [int]$envValue }
}

if (-not $PSBoundParameters.ContainsKey("CheckpointDir")) {
    $envValue = Get-EnvSetting -Source $envConfig -Name "AZR_CHECKPOINT_DIR"
    if (-not [string]::IsNullOrWhiteSpace($envValue)) { $CheckpointDir = $envValue }
}

if (-not $PSBoundParameters.ContainsKey("Seed")) {
    $envValue = Get-EnvSetting -Source $envConfig -Name "AZR_SEED"
    if (-not [string]::IsNullOrWhiteSpace($envValue)) { $Seed = [int]$envValue }
}

if (-not $PSBoundParameters.ContainsKey("SeedTasksPerType")) {
    $envValue = Get-EnvSetting -Source $envConfig -Name "AZR_SEED_TASKS_PER_TYPE"
    if (-not [string]::IsNullOrWhiteSpace($envValue)) { $SeedTasksPerType = [int]$envValue }
}

if (-not $PSBoundParameters.ContainsKey("CpuCap")) {
    $envValue = Get-EnvSetting -Source $envConfig -Name "AZR_CPU_CAP"
    if (-not [string]::IsNullOrWhiteSpace($envValue)) { $CpuCap = [double]$envValue }
}

if (-not $PSBoundParameters.ContainsKey("NoRich")) {
    $envValue = Get-EnvSetting -Source $envConfig -Name "AZR_NO_RICH"
    $parsed = Parse-EnvBool -Value $envValue -Name "AZR_NO_RICH"
    if ($null -ne $parsed) { $NoRich = $parsed }
}

if (-not $PSBoundParameters.ContainsKey("ForceRestart")) {
    $envValue = Get-EnvSetting -Source $envConfig -Name "AZR_FORCE_RESTART"
    $parsed = Parse-EnvBool -Value $envValue -Name "AZR_FORCE_RESTART"
    if ($null -ne $parsed) { $ForceRestart = $parsed }
}

if (-not $PSBoundParameters.ContainsKey("RunBenchmark")) {
    $envValue = Get-EnvSetting -Source $envConfig -Name "AZR_RUN_BENCHMARK"
    $parsed = Parse-EnvBool -Value $envValue -Name "AZR_RUN_BENCHMARK"
    if ($null -ne $parsed) { $RunBenchmark = $parsed }
}

if (-not $PSBoundParameters.ContainsKey("BaselineModelForBenchmark")) {
    $envValue = Get-EnvSetting -Source $envConfig -Name "AZR_BASELINE_MODEL_FOR_BENCHMARK"
    if (-not [string]::IsNullOrWhiteSpace($envValue)) { $BaselineModelForBenchmark = $envValue }
}

if (-not $PSBoundParameters.ContainsKey("BenchmarkLimit")) {
    $envValue = Get-EnvSetting -Source $envConfig -Name "AZR_BENCHMARK_LIMIT"
    if (-not [string]::IsNullOrWhiteSpace($envValue)) { $BenchmarkLimit = [int]$envValue }
}

if (-not $PSBoundParameters.ContainsKey("BenchmarkList")) {
    $envValue = Get-EnvSetting -Source $envConfig -Name "AZR_BENCHMARK_LIST"
    $parsed = Parse-EnvStringList -Value $envValue
    if ($parsed.Count -gt 0) { $BenchmarkList = $parsed }
}

if (-not $PSBoundParameters.ContainsKey("BenchmarkSeed")) {
    $envValue = Get-EnvSetting -Source $envConfig -Name "AZR_BENCHMARK_SEED"
    if (-not [string]::IsNullOrWhiteSpace($envValue)) { $BenchmarkSeed = [int]$envValue }
}

if (-not $PSBoundParameters.ContainsKey("BenchmarkSamplesPerTask")) {
    $envValue = Get-EnvSetting -Source $envConfig -Name "AZR_BENCHMARK_SAMPLES_PER_TASK"
    if (-not [string]::IsNullOrWhiteSpace($envValue)) { $BenchmarkSamplesPerTask = [int]$envValue }
}

if (-not $PSBoundParameters.ContainsKey("BenchmarkPassk")) {
    $envValue = Get-EnvSetting -Source $envConfig -Name "AZR_BENCHMARK_PASSK"
    if (-not [string]::IsNullOrWhiteSpace($envValue)) { $BenchmarkPassk = [int]$envValue }
}

if (-not $PSBoundParameters.ContainsKey("BenchmarkTemperature")) {
    $envValue = Get-EnvSetting -Source $envConfig -Name "AZR_BENCHMARK_TEMPERATURE"
    if (-not [string]::IsNullOrWhiteSpace($envValue)) { $BenchmarkTemperature = [double]$envValue }
}

if (-not $PSBoundParameters.ContainsKey("BenchmarkTopP")) {
    $envValue = Get-EnvSetting -Source $envConfig -Name "AZR_BENCHMARK_TOP_P"
    if (-not [string]::IsNullOrWhiteSpace($envValue)) { $BenchmarkTopP = [double]$envValue }
}

if (-not $PSBoundParameters.ContainsKey("BenchmarkCpuCap")) {
    $envValue = Get-EnvSetting -Source $envConfig -Name "AZR_BENCHMARK_CPU_CAP"
    if (-not [string]::IsNullOrWhiteSpace($envValue)) { $BenchmarkCpuCap = [double]$envValue }
}

# One argv token per benchmark name (e.g. env `humaneval,mbpp` or `-BenchmarkList humaneval,mbpp`
# must not become a single argparse value `humaneval,mbpp`, which skips all eval branches).
$expandedBenchmarks = New-Object System.Collections.Generic.List[string]
foreach ($b in @($BenchmarkList)) {
    if ([string]::IsNullOrWhiteSpace($b)) { continue }
    foreach ($piece in ($b -split ",")) {
        $t = $piece.Trim()
        if ($t) { $expandedBenchmarks.Add($t) }
    }
}
$BenchmarkList = @($expandedBenchmarks | Select-Object -Unique)

if (-not $PSBoundParameters.ContainsKey("NoBenchmarkRich")) {
    $envValue = Get-EnvSetting -Source $envConfig -Name "AZR_NO_BENCHMARK_RICH"
    $parsed = Parse-EnvBool -Value $envValue -Name "AZR_NO_BENCHMARK_RICH"
    if ($null -ne $parsed) { $NoBenchmarkRich = $parsed }
}

if (-not $PSBoundParameters.ContainsKey("RunLogRoot")) {
    $envValue = Get-EnvSetting -Source $envConfig -Name "AZR_RUN_LOG_ROOT"
    if (-not [string]::IsNullOrWhiteSpace($envValue)) { $RunLogRoot = $envValue }
}

if (-not $PSBoundParameters.ContainsKey("GpuMemoryFraction")) {
    $envValue = Get-EnvSetting -Source $envConfig -Name "AZR_GPU_MEMORY_FRACTION"
    if (-not [string]::IsNullOrWhiteSpace($envValue)) { $GpuMemoryFraction = [double]$envValue }
}

if (-not $PSBoundParameters.ContainsKey("CudaAllocConfig")) {
    $envValue = Get-EnvSetting -Source $envConfig -Name "AZR_CUDA_ALLOC_CONFIG"
    if (-not [string]::IsNullOrWhiteSpace($envValue)) {
        $CudaAllocConfig = $envValue
    } else {
        $ptAlloc = Get-EnvSetting -Source $envConfig -Name "PYTORCH_CUDA_ALLOC_CONF"
        if (-not [string]::IsNullOrWhiteSpace($ptAlloc)) {
            $CudaAllocConfig = $ptAlloc
        }
    }
}

if (-not $PSBoundParameters.ContainsKey("UseSeparateValueModel")) {
    $envValue = Get-EnvSetting -Source $envConfig -Name "AZR_USE_SEPARATE_VALUE_MODEL"
    $parsed = Parse-EnvBool -Value $envValue -Name "AZR_USE_SEPARATE_VALUE_MODEL"
    if ($null -ne $parsed) { $UseSeparateValueModel = $parsed }
}

if (-not $PSBoundParameters.ContainsKey("ModelDtype")) {
    $envValue = Get-EnvSetting -Source $envConfig -Name "AZR_MODEL_DTYPE"
    if (-not [string]::IsNullOrWhiteSpace($envValue)) { $ModelDtype = $envValue }
}

if (-not $PSBoundParameters.ContainsKey("ResourceSampleSeconds")) {
    $envValue = Get-EnvSetting -Source $envConfig -Name "AZR_RESOURCE_SAMPLE_SECONDS"
    if (-not [string]::IsNullOrWhiteSpace($envValue)) { $ResourceSampleSeconds = [int]$envValue }
}

if (-not $PSBoundParameters.ContainsKey("Use4Bit")) {
    $envValue = Get-EnvSetting -Source $envConfig -Name "AZR_USE_4BIT"
    $parsed = Parse-EnvBool -Value $envValue -Name "AZR_USE_4BIT"
    if ($null -ne $parsed) { $Use4Bit = $parsed }
}

if (-not $PSBoundParameters.ContainsKey("BaselineProgrambenchRunDir")) {
    $envValue = Get-EnvSetting -Source $envConfig -Name "AZR_BASELINE_PROGRAMBENCH_RUN_DIR"
    if (-not [string]::IsNullOrWhiteSpace($envValue)) { $BaselineProgrambenchRunDir = $envValue }
}

if (-not $PSBoundParameters.ContainsKey("ImprovedProgrambenchRunDir")) {
    $envValue = Get-EnvSetting -Source $envConfig -Name "AZR_IMPROVED_PROGRAMBENCH_RUN_DIR"
    if (-not [string]::IsNullOrWhiteSpace($envValue)) { $ImprovedProgrambenchRunDir = $envValue }
}

if (-not $PSBoundParameters.ContainsKey("PythonExeForBenchmark")) {
    $envValue = Get-EnvSetting -Source $envConfig -Name "AZR_PYTHON_EXE_FOR_BENCHMARK"
    if (-not [string]::IsNullOrWhiteSpace($envValue)) { $PythonExeForBenchmark = $envValue }
}

$envValue = Get-EnvSetting -Source $envConfig -Name "AZR_CUDA_VISIBLE_DEVICES"
if (-not [string]::IsNullOrWhiteSpace($envValue)) {
    $env:CUDA_VISIBLE_DEVICES = $envValue
}

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

# -------------------------- [3/8] Cleanup safety guard -----------------------------
# Ensure `AZR_FORCE_RESTART` cleanup cannot remove non-checkpoint paths.
function Test-CheckpointCleanupPath {
    param([string]$PathValue)

    if ([string]::IsNullOrWhiteSpace($PathValue)) {
        return $false
    }

    $fullPath = [System.IO.Path]::GetFullPath($PathValue)
    $rootPath = [System.IO.Path]::GetPathRoot($fullPath)
    if ([string]::IsNullOrWhiteSpace($rootPath)) {
        return $false
    }

    if ($fullPath.TrimEnd("\\/") -eq $rootPath.TrimEnd("\\/")) {
        return $false
    }

    $projectRootFull = [System.IO.Path]::GetFullPath($projectRoot).TrimEnd("\\/")
    $projectRootPrefix = $projectRootFull + [System.IO.Path]::DirectorySeparatorChar
    if (-not ($fullPath.ToLowerInvariant().StartsWith($projectRootPrefix.ToLowerInvariant(), [System.StringComparison]::OrdinalIgnoreCase))) {
        return $false
    }

    if ($fullPath.TrimEnd("\\/") -eq $projectRootFull) {
        return $false
    }

    return $true
}

$resolvedPython = Resolve-Path $PythonExe -ErrorAction SilentlyContinue

if (-not $resolvedPython) {
    throw "Python executable not found: $PythonExe"
}

# ----------------------- [4/8] Path and executable bootstrap ----------------------
# Validate key runtime assumptions (python path, model dtype, checkpoint location) early.
$validModelDtypes = @("fp16","bf16","fp32","auto")
if ($ModelDtype -notin $validModelDtypes) {
    throw "Unsupported ModelDtype '$ModelDtype'. Use one of: $($validModelDtypes -join ', ')."
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

# ------------------------------ [5/8] Runtime summary -----------------------------
# Print resolved runtime settings so every run captures the "as executed" state.
Write-SectionDivider -Title "RUNTIME CONFIG & STARTUP"
Write-Host "AZR Local HF Training Launcher"
Write-Host "Project root: $projectRoot"
Write-Host "Run directory: $runDir"
Write-Host "Using Python: $($resolvedPython.Path)"
Write-Host "Epochs: $Epochs"
Write-Host "Force restart: $ForceRestart"
Write-Host "Checkpoint dir: $CheckpointDir"
Write-Host "CPU cap: $CpuCap%"
Write-Host "Model dtype: $ModelDtype"
Write-Host "Separate actor/critic: $UseSeparateValueModel"
if ($RunBenchmark) {
    Write-Host "Auto benchmark: enabled"
    Write-Host "Benchmark baseline: $BaselineModelForBenchmark"
}
if ($ForceRestart) {
    Write-Host "Force restart: enabled (existing checkpoint_epoch_* artifacts will be removed before launch)"
} else {
    Write-Host "Force restart: disabled (resume behavior remains active when checkpoints exist)"
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

# --------------------------- [6/8] Optional checkpoint reset -----------------------
# If requested, keep directory structure but remove previous epoch checkpoint folders so
# next run starts at a clean logical epoch baseline.
Write-SectionDivider -Title "CHECKPOINT PREPARATION"
if ($ForceRestart) {
    # When force restart is set, clear only checkpoint_epoch_* directories so old checkpoints
    # do not influence the next epoch schedule. Core directory metadata and other artifacts
    # are kept intact for inspection/debugging.
    $checkpointCleanupMessage = "No checkpoint epoch folders found. Starting with a clean checkpoint directory."
    # Guardrail: validate path shape before deletion to avoid wiping arbitrary folders on typo.
    if (-not (Test-CheckpointCleanupPath -PathValue $resolvedCheckpointDir)) {
        throw "Refusing to apply ForceRestart to unsafe checkpoint path: $resolvedCheckpointDir"
    }

    $checkpointEpochFolders = @(Get-ChildItem -Path $resolvedCheckpointDir -Directory -Filter "checkpoint_epoch_*" -ErrorAction SilentlyContinue)
    foreach ($checkpointEpochFolder in $checkpointEpochFolders) {
        Remove-Item -Recurse -Force -Path $checkpointEpochFolder.FullName
    }
    if ($checkpointEpochFolders.Count -gt 0) {
        $checkpointCleanupMessage = "Removed existing checkpoint folders for a clean restart: $resolvedCheckpointDir"
    }
    Write-Host "AZR_FORCE_RESTART: $checkpointCleanupMessage"
} else {
    $checkpointCleanupMessage = "Force restart disabled. Existing checkpoints will be reused."
}

$trainerScriptResolved = (Resolve-Path $trainerScript).Path
$trainerArgs = @(
    $trainerScriptResolved,
    "--epochs", [string]$Epochs,
    "--checkpoint-dir", $CheckpointDir,
    "--seed", [string]$Seed,
    "--seed-tasks-per-type", [string]$SeedTasksPerType,
    "--cpu-cap", [string]$CpuCap
)

if ($NoRich) { $trainerArgs += "--no-rich" } else { $trainerArgs += "--rich" }
if ($Use4Bit) { $trainerArgs += "--use-4bit" } else { $trainerArgs += "--no-use-4bit" }
$trainerArgs += "--model-dtype"
$trainerArgs += $ModelDtype
if ($UseSeparateValueModel) {
    $trainerArgs += "--use-separate-value-model"
} else {
    $trainerArgs += "--no-use-separate-value-model"
}
$trainerArgs += "--gpu-memory-fraction"
$trainerArgs += [string]$GpuMemoryFraction
$trainerArgs += "--cuda-alloc-config"
$trainerArgs += $CudaAllocConfig
if ($ExtraArgs) { $trainerArgs += $ExtraArgs }

# Flatten + trim: nested $ExtraArgs / enumerable argv must never stringify as one token (avoids
# checkpoint_epoch_19--results-root, --epochs20, --no-rich--no-use-4bit in logged command lines).
$trainerArgsFlat = @(ConvertTo-AzrFlatStringArgv -Segments $trainerArgs | ForEach-Object { $_.Trim() } | Where-Object { $_.Length -gt 0 })
$trainerArgsArray = @($trainerArgsFlat)
$trainerMsvcTail = ConvertTo-CmdLineFromTokens -Tokens $trainerArgsArray
$commandLineHuman = ConvertTo-CmdLineFromTokens -Tokens (@($resolvedPython.Path) + $trainerArgsArray)
$commandLineMsvc = ConvertTo-CmdLineFromTokens -Tokens (@($resolvedPython.Path) + $trainerArgsArray)
$argvDebug = ($trainerArgsArray | ForEach-Object { "<$_>" }) -join " "
$cmdLineOut = "Command (human join): $commandLineHuman"
$cmdLineMsvcOut = "Command (MSVC args tail): $commandLineMsvc"
$checkpointSummary = "AZR_FORCE_RESTART: $ForceRestart"
$checkpointSummary += "`r`nCheckpoint cleanup: $checkpointCleanupMessage"

# ----------------------------- [7/8] Trainer dispatch -----------------------------
# Build the trainer command explicitly, persist metadata snapshots, then stream trainer
# logs and resource samples continuously until the process exits.
Set-Content -Path $trainSummary -Value $checkpointSummary
Add-Content -Path $trainSummary -Value $cmdLineOut
Add-Content -Path $trainSummary -Value $cmdLineMsvcOut
Add-Content -Path $resourceLog -Value "timestamp,pid,cpu_percent,available_mem_mb,gpu_sample"
Add-Content -Path $runConfig -Value "{" | Out-Null
Add-Content -Path $runConfig -Value "`"epochs`": $Epochs," | Out-Null
Add-Content -Path $runConfig -Value "`"force_restart`": $($ForceRestart.ToString().ToLower())," | Out-Null
Add-Content -Path $runConfig -Value "`"checkpoint_dir`": `"$CheckpointDir`"," | Out-Null
Add-Content -Path $runConfig -Value "`"seed`": $Seed," | Out-Null
Add-Content -Path $runConfig -Value "`"seed_tasks_per_type`": $SeedTasksPerType," | Out-Null
Add-Content -Path $runConfig -Value "`"cpu_cap`": $CpuCap," | Out-Null
Add-Content -Path $runConfig -Value "`"gpu_memory_fraction`": $GpuMemoryFraction," | Out-Null
Add-Content -Path $runConfig -Value "`"cuda_alloc_config`": `"$CudaAllocConfig`"," | Out-Null
Add-Content -Path $runConfig -Value "`"model_dtype`": `"$ModelDtype`"," | Out-Null
Add-Content -Path $runConfig -Value "`"use_separate_value_model`": $($UseSeparateValueModel.ToString().ToLower())," | Out-Null
Add-Content -Path $runConfig -Value "`"use_4bit`": $($Use4Bit.ToString().ToLower())," | Out-Null
Add-Content -Path $runConfig -Value "`"timestamp`": `"$runTimestamp`"" | Out-Null
Add-Content -Path $runConfig -Value "}" | Out-Null

Write-SectionDivider -Title "TRAINING START"
# Propagate select trainer-side env vars from .env into this process so the spawned Python
# child inherits them (hf_trainer does not load .env itself).
foreach ($trainerEnvKey in @(
        "AZR_GEN_LOGITS_FP32",
        "AZR_PPO_DISABLE_CUDA_AUTOCAST",
        "AZR_LAYERNORM_EPS",
        "AZR_TRAIN_STEP_LOG",
        "AZR_TRAIN_STEP_LOG_MAX_CHARS",
        "AZR_TRAIN_STEP_LOG_FILE",
        "AZR_PAPER_STYLE_DEFAULTS",
        "AZR_HF_LEARNING_RATE",
        "AZR_HF_CRITIC_LEARNING_RATE",
        "AZR_HF_GENERATION_STEPS_PER_EPOCH",
        "AZR_HF_BATCH_SIZE",
        "AZR_HF_PPO_UPDATE_THRESHOLD",
        "AZR_SEED_TASKS_PER_TYPE",
        "AZR_PPO_MICROBATCH_SIZE",
        "AZR_PPO_CE_CHUNK")) {
    $trainerEnvVal = Get-EnvSetting -Source $envConfig -Name $trainerEnvKey
    if (-not [string]::IsNullOrWhiteSpace($trainerEnvVal)) {
        Set-Item -Path ("Env:" + $trainerEnvKey) -Value $trainerEnvVal
    }
}
Write-Host $cmdLineOut
Write-Host $cmdLineMsvcOut
Write-Host ("Trainer argv tokens (verbatim, angle-bracket delimited): " + $argvDebug)
Write-Host "Logs: $runDir"
Write-Host "Starting process..."

$env:AZR_RUN_LOG_DIR = $runDir

Register-AzrConsoleCancel

$trainerHandle = Start-AzrRedirectedNativeProcess `
    -FilePath $resolvedPython.Path `
    -Tokens $trainerArgsArray `
    -WorkingDirectory $projectRoot `
    -StdoutPath $trainerOut `
    -StderrPath $trainerErr

$proc = $trainerHandle.Process

if (-not $proc) {
    throw "Failed to start trainer process."
}

$trainerPid = $proc.Id
Write-Host "Trainer PID: $trainerPid"
$lastTrainerStdoutLines = 0
$lastTrainerStderrLines = 0

try {
    while ($true) {
        if ($script:AzrCancelRequested) {
            break
        }

        $p = Get-Process -Id $trainerPid -ErrorAction SilentlyContinue
        if (-not $p) {
            break
        }

        if (Test-Path $trainerOut) {
            $stdoutLines = Get-Content -Path $trainerOut -ErrorAction SilentlyContinue
            $stdoutCount = $stdoutLines.Count
            if ($stdoutCount -gt $lastTrainerStdoutLines) {
                $stdoutLines |
                    Select-Object -Skip $lastTrainerStdoutLines |
                    ForEach-Object { Write-Host $_ }
                $lastTrainerStdoutLines = $stdoutCount
            }
        }

        if (Test-Path $trainerErr) {
            $stderrLines = Get-Content -Path $trainerErr -ErrorAction SilentlyContinue
            $stderrCount = $stderrLines.Count
            if ($stderrCount -gt $lastTrainerStderrLines) {
                $stderrLines |
                    Select-Object -Skip $lastTrainerStderrLines |
                    ForEach-Object { Write-Host "[trainer stderr] $_" -ForegroundColor DarkYellow }
                $lastTrainerStderrLines = $stderrCount
            }
        }

        Add-Content -Path $resourceLog -Value (Get-ResourceSample -ProcessId $trainerPid)
        Wait-AzrSleepCancellable -TotalSeconds ([double]$ResourceSampleSeconds)
        if ($script:AzrCancelRequested) {
            break
        }
    }
} finally {
    if ($script:AzrCancelRequested -and $proc -and -not $proc.HasExited) {
        Stop-AzrNativeChildTree -Process $proc
    }
    Complete-AzrRedirectedNativeProcess $trainerHandle

    if (Test-Path $trainerOut) {
        $stdoutLines = Get-Content -Path $trainerOut -ErrorAction SilentlyContinue
        $stdoutLines | Select-Object -Skip $lastTrainerStdoutLines | ForEach-Object { Write-Host $_ }
    }

    if (Test-Path $trainerErr) {
        $stderrLines = Get-Content -Path $trainerErr -ErrorAction SilentlyContinue
        $stderrLines | Select-Object -Skip $lastTrainerStderrLines | ForEach-Object { Write-Host "[trainer stderr] $_" -ForegroundColor DarkYellow }
    }

    Add-Content -Path $resourceLog -Value "ended,$trainerPid,$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
}

if ($script:AzrCancelRequested) {
    Add-Content -Path $trainSummary -Value "Training interrupted by user (Ctrl+C)."
    Unregister-AzrConsoleCancel
    Restore-AzrLauncherProcessEnv -OriginalPythonIOEncoding $originalPythonIOEncoding `
        -OriginalPythonUtf8 $originalPythonUtf8 `
        -OriginalAllocConfig $originalAllocConfig
    Write-Host "Training stopped by user. Logs: $runDir"
    exit 130
}

$proc.Refresh()
$exitCode = if ($proc.HasExited -and $null -ne $proc.ExitCode) { [int]$proc.ExitCode } else { -1 }

if ($exitCode -lt 0) {
    # Fallback when ExitCode is not surfaced reliably after redirected Process completion.
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
    Unregister-AzrConsoleCancel
    throw "Training run failed with exit code $exitCode"
}

Write-Host "Training complete. Exit code: $exitCode"

if ($RunBenchmark) {
    Write-SectionDivider -Title "POST-TRAIN BENCHMARK"
    foreach ($benchEnvKey in @(
            "AZR_BENCHMARK_FAST",
            "AZR_BENCHMARK_MAX_TASKS_PER_DATASET",
            "AZR_BENCHMARK_BATCH_SIZE",
            "AZR_GENU_LOG_WARN_CAP")) {
        $benchEnvVal = Get-EnvSetting -Source $envConfig -Name $benchEnvKey
        if (-not [string]::IsNullOrWhiteSpace($benchEnvVal)) {
            Set-Item -Path ("Env:" + $benchEnvKey) -Value $benchEnvVal
        }
    }
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

    # Path tokens stay separate array elements (no script.py--flag joins). The launcher builds a
    # MSVC-compatible ProcessStartInfo.Arguments tail so spaces in paths survive CreateProcess on
    # Windows PowerShell 5.1 (Start-Process -ArgumentList can still split spaced tokens there).
    $benchArgs = @(
        "-u",
        $benchmarkScriptResolved,
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
    if ($Use4Bit) { $benchArgs += "--use-4bit" } else { $benchArgs += "--no-use-4bit" }
    # run_pre_post_benchmarks.py only defines --use-separate-value-model (store_true); omit when unified.
    if ($UseSeparateValueModel) {
        $benchArgs += "--use-separate-value-model"
    } else {
        $benchArgs += "--no-use-separate-value-model"
    }
    $benchmarkIncludesProgrambench = $false
    foreach ($b in $BenchmarkList) {
        if ($b -ieq "programbench") {
            $benchmarkIncludesProgrambench = $true
            break
        }
    }
    if ($benchmarkIncludesProgrambench -and -not [string]::IsNullOrWhiteSpace($BaselineProgrambenchRunDir)) {
        $benchArgs += @("--baseline-programbench-run-dir", $BaselineProgrambenchRunDir)
    }
    if ($benchmarkIncludesProgrambench -and -not [string]::IsNullOrWhiteSpace($ImprovedProgrambenchRunDir)) {
        $benchArgs += @("--improved-programbench-run-dir", $ImprovedProgrambenchRunDir)
    }

    $benchArgsFlat = @(ConvertTo-AzrFlatStringArgv -Segments $benchArgs | ForEach-Object { $_.Trim() } | Where-Object { $_.Length -gt 0 })
    $benchArgsArray = @($benchArgsFlat)
    $benchMsvcTail = ConvertTo-CmdLineFromTokens -Tokens $benchArgsArray
    $benchCommandHuman = ConvertTo-CmdLineFromTokens -Tokens (@($resolvedPythonForBenchmark) + $benchArgsArray)
    $benchCommandMsvc = ConvertTo-CmdLineFromTokens -Tokens (@($resolvedPythonForBenchmark) + $benchArgsArray)
    $benchArgvDebug = ($benchArgsArray | ForEach-Object { "<$_>" }) -join " "
    Add-Content -Path $trainSummary -Value "Benchmark command (human join): $benchCommandHuman"
    Add-Content -Path $trainSummary -Value "Benchmark command (MSVC args tail): $benchCommandMsvc"
    Write-Host "Benchmark command (human join): $benchCommandHuman"
    Write-Host "Benchmark command (MSVC args tail): $benchCommandMsvc"
    Write-Host ("Benchmark argv tokens (verbatim, angle-bracket delimited): " + $benchArgvDebug)
    Write-Host "Running post-training benchmark..."

    $previousLocation = Get-Location
    $benchHandle = $null
    $benchProc = $null
    try {
        Set-Location $projectRoot
        $benchHandle = Start-AzrRedirectedNativeProcess `
            -FilePath $resolvedPythonForBenchmark `
            -Tokens $benchArgsArray `
            -WorkingDirectory $projectRoot `
            -StdoutPath $benchOut `
            -StderrPath $benchErr

        $benchProc = $benchHandle.Process

        if (-not $benchProc) {
            throw "Failed to start benchmark process."
        }

        $benchPid = $benchProc.Id
        Write-Host "Benchmark PID: $benchPid"
        $lastBenchmarkStdoutLines = 0
        $lastBenchmarkStderrLines = 0

        while ($true) {
            if ($script:AzrCancelRequested) {
                break
            }

            $p = Get-Process -Id $benchPid -ErrorAction SilentlyContinue
            if (-not $p) {
                break
            }

            if (Test-Path $benchOut) {
                $stdoutLines = Get-Content -Path $benchOut -ErrorAction SilentlyContinue
                $stdoutCount = $stdoutLines.Count
                if ($stdoutCount -gt $lastBenchmarkStdoutLines) {
                    $stdoutLines |
                        Select-Object -Skip $lastBenchmarkStdoutLines |
                        ForEach-Object { Write-Host $_ }
                    $lastBenchmarkStdoutLines = $stdoutCount
                }
            }

            if (Test-Path $benchErr) {
                $stderrLines = Get-Content -Path $benchErr -ErrorAction SilentlyContinue
                $stderrCount = $stderrLines.Count
                if ($stderrCount -gt $lastBenchmarkStderrLines) {
                    $stderrLines |
                        Select-Object -Skip $lastBenchmarkStderrLines |
                        ForEach-Object { Write-Host "[benchmark stderr] $_" -ForegroundColor DarkYellow }
                    $lastBenchmarkStderrLines = $stderrCount
                }
            }

            Wait-AzrSleepCancellable -TotalSeconds 2
            if ($script:AzrCancelRequested) {
                break
            }
        }
    }
    finally {
        Set-Location $previousLocation
        if ($benchHandle) {
            if ($script:AzrCancelRequested -and $benchProc -and -not $benchProc.HasExited) {
                Stop-AzrNativeChildTree -Process $benchProc
            }
            Complete-AzrRedirectedNativeProcess $benchHandle
            if (Test-Path $benchOut) {
                $stdoutLines = Get-Content -Path $benchOut -ErrorAction SilentlyContinue
                $stdoutLines |
                    Select-Object -Skip $lastBenchmarkStdoutLines |
                    ForEach-Object { Write-Host $_ }
            }
            if (Test-Path $benchErr) {
                $stderrLines = Get-Content -Path $benchErr -ErrorAction SilentlyContinue
                $stderrLines |
                    Select-Object -Skip $lastBenchmarkStderrLines |
                    ForEach-Object { Write-Host "[benchmark stderr] $_" -ForegroundColor DarkYellow }
            }
        }
    }

    $benchExitCode = -1
    if ($benchProc) {
        $benchProc.Refresh()
        if ($benchProc.HasExited -and $null -ne $benchProc.ExitCode) {
            $benchExitCode = [int]$benchProc.ExitCode
        }
    }

    if ($benchExitCode -lt 0 -and (Test-Path $benchOut)) {
        $tailText = (Get-Content -Path $benchOut -Tail 20 -ErrorAction SilentlyContinue) -join " "
        if ($tailText -match "Comparison JSON:|Comparison Report:|AZR Benchmark Comparison") {
            $benchExitCode = 0
        }
    }

    if ($script:AzrCancelRequested) {
        Add-Content -Path $trainSummary -Value "Benchmark interrupted by user (Ctrl+C)."
        Unregister-AzrConsoleCancel
        Restore-AzrLauncherProcessEnv -OriginalPythonIOEncoding $originalPythonIOEncoding `
            -OriginalPythonUtf8 $originalPythonUtf8 `
            -OriginalAllocConfig $originalAllocConfig
        Write-Host "Benchmark stopped by user. Logs: $runDir"
        exit 130
    }

    if ($benchExitCode -ne 0) {
        Write-Host "Benchmark failed (exit code $benchExitCode)."
        Write-Host "Benchmark stdout: $benchOut"
        Write-Host "Benchmark stderr: $benchErr"
        Unregister-AzrConsoleCancel
        throw "Benchmark run failed with exit code $benchExitCode"
    }

    Write-Host "Benchmark complete. Results in: $benchResultsDir"
    Write-Host "Benchmark logs: $benchOut / $benchErr"
}

Write-SectionDivider -Title "RUN COMPLETE"
Write-Host "Done. All outputs in: $runDir"

Unregister-AzrConsoleCancel
Restore-AzrLauncherProcessEnv -OriginalPythonIOEncoding $originalPythonIOEncoding `
    -OriginalPythonUtf8 $originalPythonUtf8 `
    -OriginalAllocConfig $originalAllocConfig
