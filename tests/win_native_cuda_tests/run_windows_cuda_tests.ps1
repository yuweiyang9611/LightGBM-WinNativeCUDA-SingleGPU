[CmdletBinding()]
param(
    [string]$BuildDirectory = ".builds\win-native-cuda",
    [string]$CudaArchitectures = "89-real;89-virtual",
    [string]$CMakeExecutable = "cmake",
    [string]$PythonExecutable = ""
)

$ErrorActionPreference = "Stop"

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$ResolvedBuildDirectory = [System.IO.Path]::GetFullPath((Join-Path $RepoRoot $BuildDirectory))
$ArtifactDirectory = Join-Path $ResolvedBuildDirectory "artifacts"

if ([string]::IsNullOrWhiteSpace($PythonExecutable)) {
    $PythonExecutable = Join-Path $RepoRoot ".venv\Scripts\python.exe"
    if (-not (Test-Path -LiteralPath $PythonExecutable)) {
        throw "Project Python environment was not found at $PythonExecutable. Create .venv or pass -PythonExecutable."
    }
}

if (-not (Get-Command $CMakeExecutable -ErrorAction SilentlyContinue)) {
    throw "CMake was not found. Pass -CMakeExecutable with the full path to cmake.exe."
}
if (-not (Get-Command $PythonExecutable -ErrorAction SilentlyContinue)) {
    throw "Python was not found. Pass -PythonExecutable with the full path to python.exe."
}
& $PythonExecutable -c "import narwhals, numpy, pytest, scipy"
if ($LASTEXITCODE -ne 0) {
    throw "The test Python environment needs narwhals, numpy, scipy, and pytest. Install them before running this script."
}

$Nvcc = Get-Command "nvcc.exe" -ErrorAction SilentlyContinue
if (-not $Nvcc -and $env:CUDA_PATH) {
    $NvccCandidate = Join-Path $env:CUDA_PATH "bin\nvcc.exe"
    if (Test-Path -LiteralPath $NvccCandidate) {
        $Nvcc = Get-Item -LiteralPath $NvccCandidate
    }
}
if (-not $Nvcc) {
    throw "CUDA Toolkit (nvcc.exe) was not found. Install it and expose CUDA_PATH or add its bin directory to PATH."
}
$NvccPath = if ($Nvcc.Source) { $Nvcc.Source } else { $Nvcc.FullName }
$NvccDirectory = Split-Path -Parent $NvccPath
$env:PATH = "$NvccDirectory;$env:PATH"
$Cuobjdump = Join-Path $NvccDirectory "cuobjdump.exe"
if (-not (Test-Path -LiteralPath $Cuobjdump -PathType Leaf)) {
    throw "cuobjdump.exe was not found next to nvcc.exe at $NvccDirectory."
}

$ConfigureArguments = @(
    "-S", $RepoRoot,
    "-B", $ResolvedBuildDirectory,
    "-A", "x64",
    "-DUSE_CUDA=ON",
    "-DUSE_NCCL=OFF",
    "-DBUILD_CLI=OFF",
    "-DCMAKE_CUDA_ARCHITECTURES=$CudaArchitectures",
    "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=$ArtifactDirectory",
    "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE=$ArtifactDirectory",
    "-DCMAKE_RUNTIME_OUTPUT_DIRECTORY=$ArtifactDirectory",
    "-DCMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE=$ArtifactDirectory"
)

& $CMakeExecutable @ConfigureArguments
if ($LASTEXITCODE -ne 0) {
    throw "CMake configuration failed with exit code $LASTEXITCODE."
}

& $CMakeExecutable --build $ResolvedBuildDirectory --config Release --target _lightgbm --parallel
if ($LASTEXITCODE -ne 0) {
    throw "CMake build failed with exit code $LASTEXITCODE."
}

$BuiltDllPath = Join-Path $ArtifactDirectory "lib_lightgbm.dll"
if (-not (Test-Path -LiteralPath $BuiltDllPath -PathType Leaf)) {
    throw "The build completed but lib_lightgbm.dll was not found at $BuiltDllPath."
}
$BuiltDll = Get-Item -LiteralPath $BuiltDllPath

$CubinListing = (& $Cuobjdump --list-elf $BuiltDll.FullName 2>&1 | Out-String)
if ($CubinListing -notmatch "ELF file\s+\d+:") {
    throw "The built DLL does not contain a CUDA cubin image.`n$CubinListing"
}
$PtxListing = (& $Cuobjdump --all-fatbin --list-ptx $BuiltDll.FullName 2>&1 | Out-String)
if ($PtxListing -notmatch "PTX file\s+\d+:") {
    throw "The built DLL does not contain PTX for forward-compatible JIT.`n$PtxListing"
}
Write-Host $CubinListing.Trim()
Write-Host "Relocatable PTX fallback found in the built DLL."

$env:LIGHTGBM_CUDA_DLL = $BuiltDll.FullName
& $PythonExecutable -m pytest -ra (Join-Path $PSScriptRoot "test_source_contract.py") (Join-Path $PSScriptRoot "test_cuda_runtime.py")
if ($LASTEXITCODE -ne 0) {
    throw "Windows native CUDA tests failed with exit code $LASTEXITCODE."
}
