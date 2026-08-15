[CmdletBinding()]
param(
    [ValidateRange(0, 255)]
    [int]$GpuDeviceId = 0,

    [switch]$ForcePtxJit,

    [switch]$KeepScratch,

    [switch]$Pause
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version 2.0

$BundleRoot = [System.IO.Path]::GetFullPath($PSScriptRoot)
$RunRoot = $null
$ScratchRoot = $null
$ExitCode = 1

function Assert-BundleIntegrity {
    $ManifestPath = Join-Path $BundleRoot "SHA256SUMS.txt"
    if (-not (Test-Path -LiteralPath $ManifestPath -PathType Leaf)) {
        throw "Bundle manifest is missing: $ManifestPath"
    }
    $BundlePrefix = $BundleRoot.TrimEnd("\") + "\"
    foreach ($Line in Get-Content -LiteralPath $ManifestPath) {
        if ([string]::IsNullOrWhiteSpace($Line)) {
            continue
        }
        if ($Line -notmatch "^(?<Hash>[0-9a-fA-F]{64})  (?<Name>.+)$") {
            throw "Invalid manifest entry: $Line"
        }
        $RelativePath = $Matches.Name.Replace("/", "\")
        $FilePath = [System.IO.Path]::GetFullPath((Join-Path $BundleRoot $RelativePath))
        if (-not $FilePath.StartsWith($BundlePrefix, [System.StringComparison]::OrdinalIgnoreCase)) {
            throw "Manifest path escapes the bundle: $RelativePath"
        }
        if (-not (Test-Path -LiteralPath $FilePath -PathType Leaf)) {
            throw "Bundle file is missing: $RelativePath"
        }
        $ActualHash = (Get-FileHash -LiteralPath $FilePath -Algorithm SHA256).Hash
        if ($ActualHash -ne $Matches.Hash) {
            throw "Bundle checksum mismatch: $RelativePath"
        }
    }
}

try {
    Assert-BundleIntegrity

    $MachineName = ($env:COMPUTERNAME -replace "[^A-Za-z0-9_.-]", "_")
    $RunId = "{0}-{1}-{2}" -f $MachineName, (Get-Date -Format "yyyyMMdd-HHmmss"), $PID
    $RunRoot = Join-Path $BundleRoot ("results\" + $RunId)
    $ScratchRoot = Join-Path $RunRoot "scratch"
    New-Item -ItemType Directory -Path $ScratchRoot -Force | Out-Null

    $PythonDirectory = Join-Path $BundleRoot "runtime\python"
    $PythonExecutable = Join-Path $PythonDirectory "python.exe"
    $SmokeTest = Join-Path $BundleRoot "tests\smoke_test.py"
    if (-not (Test-Path -LiteralPath $PythonExecutable -PathType Leaf)) {
        throw "Embedded Python is missing: $PythonExecutable"
    }
    if (-not (Test-Path -LiteralPath $SmokeTest -PathType Leaf)) {
        throw "Smoke test is missing: $SmokeTest"
    }

    # Every environment change below is local to this PowerShell process and its child.
    $env:PATH = $PythonDirectory + ";" + $env:PATH
    $env:TEMP = $ScratchRoot
    $env:TMP = $ScratchRoot
    $env:CUDA_CACHE_PATH = Join-Path $ScratchRoot "cuda-cache"
    $env:CUDA_CACHE_MAXSIZE = "1073741824"
    $env:PYTHONDONTWRITEBYTECODE = "1"
    $env:PYTHONNOUSERSITE = "1"
    $env:LIGHTGBM_TEST_GPU_DEVICE_ID = [string]$GpuDeviceId
    $env:LIGHTGBM_TEST_RESULT = Join-Path $RunRoot "result.json"
    Remove-Item Env:PYTHONHOME -ErrorAction SilentlyContinue
    Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue
    Remove-Item Env:CUDA_VISIBLE_DEVICES -ErrorAction SilentlyContinue
    Remove-Item Env:CUDA_FORCE_PTX_JIT -ErrorAction SilentlyContinue
    Remove-Item Env:CUDA_DISABLE_PTX_JIT -ErrorAction SilentlyContinue
    if ($ForcePtxJit) {
        $env:CUDA_FORCE_PTX_JIT = "1"
    }

    $Invocation = [ordered]@{
        run_id = $RunId
        started_local = (Get-Date).ToString("o")
        gpu_device_id = $GpuDeviceId
        force_ptx_jit = [bool]$ForcePtxJit
        bundle_root = $BundleRoot
    }
    $Invocation | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $RunRoot "invocation.json") -Encoding UTF8

    Write-Host "LightGBM Windows native CUDA portable test"
    Write-Host "Result directory: $RunRoot"
    Write-Host "GPU device ID:   $GpuDeviceId"
    Write-Host "Force PTX JIT:   $([bool]$ForcePtxJit)"

    & $PythonExecutable -I -s -E -B $SmokeTest 2>&1 |
        Tee-Object -FilePath (Join-Path $RunRoot "console.log")
    $ExitCode = $LASTEXITCODE
    if ($ExitCode -eq 0) {
        Write-Host "PASS - CUDA training completed."
    }
    else {
        Write-Host "FAIL - test process returned exit code $ExitCode."
    }
}
catch {
    Write-Host "FAIL - $($_.Exception.Message)"
    if ($RunRoot -and (Test-Path -LiteralPath $RunRoot -PathType Container)) {
        $_ | Out-String | Set-Content -LiteralPath (Join-Path $RunRoot "launcher-error.txt") -Encoding UTF8
    }
    $ExitCode = 1
}
finally {
    if (-not $KeepScratch -and $ScratchRoot -and (Test-Path -LiteralPath $ScratchRoot -PathType Container)) {
        $RunPrefix = [System.IO.Path]::GetFullPath($RunRoot).TrimEnd("\") + "\"
        $ScratchFullPath = [System.IO.Path]::GetFullPath($ScratchRoot)
        if ($ScratchFullPath.StartsWith($RunPrefix, [System.StringComparison]::OrdinalIgnoreCase) -and
            (Split-Path -Leaf $ScratchFullPath) -eq "scratch") {
            Remove-Item -LiteralPath $ScratchFullPath -Recurse -Force -ErrorAction SilentlyContinue
        }
    }
    if ($Pause) {
        Write-Host ""
        [void](Read-Host "Press Enter to close")
    }
}

exit $ExitCode
