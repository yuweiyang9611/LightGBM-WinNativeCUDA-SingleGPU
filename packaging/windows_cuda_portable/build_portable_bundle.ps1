[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$WheelhouseZip,

    [Parameter(Mandatory = $true)]
    [string]$PythonEmbedZip,

    [Parameter(Mandatory = $true)]
    [ValidatePattern("^[0-9a-fA-F]{64}$")]
    [string]$ExpectedPythonEmbedSha256,

    [Parameter(Mandatory = $true)]
    [string[]]$VcRuntimeDirectories,

    [string]$OutputRoot = (Join-Path $PSScriptRoot "out"),

    [string]$BundleName = "LightGBM-CUDA-Portable-cp311-win_amd64",

    [string]$BuildPythonExecutable = "py",

    [string[]]$BuildPythonArguments = @(),

    [ValidatePattern("^$|^[0-9a-fA-F]{64}$")]
    [string]$ExpectedWheelhouseSha256 = "26e9fdf66405ad9a626f3f19619f88ee2f0e4053e48b5a9d0bbf98188d214880"
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version 2.0

function Get-FullPath {
    param([Parameter(Mandatory = $true)][string]$Path)
    return [System.IO.Path]::GetFullPath($Path)
}

function Assert-FileHash {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$ExpectedHash
    )
    if ([string]::IsNullOrWhiteSpace($ExpectedHash)) {
        return
    }
    $ActualHash = (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash
    if ($ActualHash -ne $ExpectedHash) {
        throw "SHA-256 mismatch for '$Path'. Expected $ExpectedHash, got $ActualHash."
    }
}

function Write-Utf8NoBom {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$Content
    )
    $Encoding = New-Object System.Text.UTF8Encoding($false)
    [System.IO.File]::WriteAllText($Path, $Content, $Encoding)
}

$WheelhouseZip = Get-FullPath $WheelhouseZip
$PythonEmbedZip = Get-FullPath $PythonEmbedZip
$OutputRoot = Get-FullPath $OutputRoot

foreach ($RequiredFile in @($WheelhouseZip, $PythonEmbedZip)) {
    if (-not (Test-Path -LiteralPath $RequiredFile -PathType Leaf)) {
        throw "Input file not found: $RequiredFile"
    }
}
foreach ($RuntimeDirectory in $VcRuntimeDirectories) {
    if (-not (Test-Path -LiteralPath $RuntimeDirectory -PathType Container)) {
        throw "VC runtime directory not found: $RuntimeDirectory"
    }
}

Assert-FileHash -Path $WheelhouseZip -ExpectedHash $ExpectedWheelhouseSha256
Assert-FileHash -Path $PythonEmbedZip -ExpectedHash $ExpectedPythonEmbedSha256

if (-not (Get-Command $BuildPythonExecutable -ErrorAction SilentlyContinue)) {
    throw "Build Python command was not found: $BuildPythonExecutable"
}

New-Item -ItemType Directory -Path $OutputRoot -Force | Out-Null
$BundleRoot = Join-Path $OutputRoot $BundleName
$BundleZip = Join-Path $OutputRoot ($BundleName + ".zip")
if ((Test-Path -LiteralPath $BundleRoot) -or (Test-Path -LiteralPath $BundleZip)) {
    throw "Output already exists. Choose a new -BundleName or remove it explicitly: $BundleRoot"
}

$BuildScratch = Join-Path ([System.IO.Path]::GetTempPath()) ("lightgbm-portable-build-" + [Guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Path $BuildScratch | Out-Null

try {
    $WheelhouseDirectory = Join-Path $BuildScratch "wheelhouse"
    $PythonDirectory = Join-Path $BundleRoot "runtime\python"
    $SitePackages = Join-Path $PythonDirectory "Lib\site-packages"

    New-Item -ItemType Directory -Path $WheelhouseDirectory | Out-Null
    New-Item -ItemType Directory -Path $PythonDirectory -Force | Out-Null
    New-Item -ItemType Directory -Path $SitePackages -Force | Out-Null

    Expand-Archive -LiteralPath $WheelhouseZip -DestinationPath $WheelhouseDirectory
    Expand-Archive -LiteralPath $PythonEmbedZip -DestinationPath $PythonDirectory

    $EmbeddedPython = Join-Path $PythonDirectory "python.exe"
    if (-not (Test-Path -LiteralPath $EmbeddedPython -PathType Leaf)) {
        throw "The Python archive is not a Windows embeddable distribution: python.exe is missing."
    }
    $EmbeddedPth = @(Get-ChildItem -LiteralPath $PythonDirectory -Filter "python*._pth" -File)
    if ($EmbeddedPth.Count -ne 1) {
        throw "Expected exactly one python*._pth file in the embeddable distribution."
    }
    if ($EmbeddedPth[0].Name -notmatch "^python311\._pth$") {
        throw "This bundle requires CPython 3.11, but found $($EmbeddedPth[0].Name)."
    }

    $PthLines = New-Object System.Collections.Generic.List[string]
    foreach ($Line in Get-Content -LiteralPath $EmbeddedPth[0].FullName) {
        if ($Line.Trim() -eq "import site") {
            $PthLines.Add("#import site")
        }
        else {
            $PthLines.Add($Line)
        }
    }
    if (-not ($PthLines -contains "Lib\site-packages")) {
        $PthLines.Add("Lib\site-packages")
    }
    Set-Content -LiteralPath $EmbeddedPth[0].FullName -Value $PthLines -Encoding ASCII

    $LightGbmWheel = @(Get-ChildItem -LiteralPath $WheelhouseDirectory -Filter "lightgbm-*-win_amd64.whl" -File |
        Sort-Object Name)
    if ($LightGbmWheel.Count -ne 1) {
        throw "Expected exactly one LightGBM win_amd64 wheel in the wheelhouse."
    }

    & $BuildPythonExecutable @BuildPythonArguments -m pip install `
        --isolated `
        --no-index `
        --no-cache-dir `
        --find-links $WheelhouseDirectory `
        --platform win_amd64 `
        --python-version 3.11 `
        --implementation cp `
        --abi cp311 `
        --only-binary=:all: `
        --ignore-installed `
        --no-compile `
        --target $SitePackages `
        $LightGbmWheel[0].FullName
    if ($LASTEXITCODE -ne 0) {
        throw "Offline target installation failed with exit code $LASTEXITCODE."
    }

    foreach ($RuntimeDirectory in $VcRuntimeDirectories) {
        Get-ChildItem -LiteralPath $RuntimeDirectory -Filter "*.dll" -File -Recurse | ForEach-Object {
            # python.exe's directory is first in the normal Windows DLL search order. Copying here
            # also replaces the older vcruntime140 DLLs included in some embeddable Python ZIPs.
            Copy-Item -LiteralPath $_.FullName -Destination (Join-Path $PythonDirectory $_.Name) -Force
        }
    }
    foreach ($RequiredDll in @("MSVCP140.dll", "VCRUNTIME140.dll", "VCRUNTIME140_1.dll", "VCOMP140.dll")) {
        $RequiredDllPath = Join-Path $PythonDirectory $RequiredDll
        if (-not (Test-Path -LiteralPath $RequiredDllPath -PathType Leaf)) {
            throw "Required app-local runtime is missing: $RequiredDll"
        }
        $RuntimeVersionText = (Get-Item -LiteralPath $RequiredDllPath).VersionInfo.FileVersion
        $RuntimeVersion = New-Object System.Version($RuntimeVersionText)
        if ($RuntimeVersion -lt (New-Object System.Version("14.51.0.0"))) {
            throw "App-local runtime $RequiredDll is too old: $RuntimeVersionText; version 14.51 or newer is required."
        }
    }

    Copy-Item -LiteralPath (Join-Path $PSScriptRoot "template\START_TEST.cmd") -Destination $BundleRoot
    Copy-Item -LiteralPath (Join-Path $PSScriptRoot "template\run_portable_test.ps1") -Destination $BundleRoot
    Copy-Item -LiteralPath (Join-Path $PSScriptRoot "template\README.md") -Destination $BundleRoot
    Copy-Item -LiteralPath (Join-Path $PSScriptRoot "template\THIRD_PARTY_NOTICES.md") -Destination $BundleRoot
    Copy-Item -LiteralPath (Join-Path $PSScriptRoot "template\tests") -Destination $BundleRoot -Recurse
    New-Item -ItemType Directory -Path (Join-Path $BundleRoot "results") | Out-Null

    $BundleInfo = [ordered]@{
        format_version = 1
        created_utc = [DateTime]::UtcNow.ToString("o")
        lightgbm_wheel = $LightGbmWheel[0].Name
        lightgbm_wheel_sha256 = (Get-FileHash -LiteralPath $LightGbmWheel[0].FullName -Algorithm SHA256).Hash.ToLowerInvariant()
        wheelhouse_zip = [System.IO.Path]::GetFileName($WheelhouseZip)
        wheelhouse_zip_sha256 = (Get-FileHash -LiteralPath $WheelhouseZip -Algorithm SHA256).Hash.ToLowerInvariant()
        python_embed_zip = [System.IO.Path]::GetFileName($PythonEmbedZip)
        python_embed_zip_sha256 = (Get-FileHash -LiteralPath $PythonEmbedZip -Algorithm SHA256).Hash.ToLowerInvariant()
        isolation = "CPython _pth plus -I -s -E -B; VC runtime beside python.exe; process-local environment"
    }
    Write-Utf8NoBom -Path (Join-Path $BundleRoot "bundle_info.json") -Content ($BundleInfo | ConvertTo-Json -Depth 4)

    $ManifestLines = New-Object System.Collections.Generic.List[string]
    Get-ChildItem -LiteralPath $BundleRoot -File -Recurse |
        Where-Object { $_.Name -ne "SHA256SUMS.txt" -and $_.FullName -notlike (Join-Path $BundleRoot "results\*") } |
        Sort-Object FullName |
        ForEach-Object {
            $RelativePath = $_.FullName.Substring($BundleRoot.Length + 1).Replace("\", "/")
            $Hash = (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
            $ManifestLines.Add("$Hash  $RelativePath")
        }
    Write-Utf8NoBom -Path (Join-Path $BundleRoot "SHA256SUMS.txt") -Content (($ManifestLines -join "`n") + "`n")

    $OriginalPath = $env:PATH
    try {
        $env:PATH = $PythonDirectory + ";" + $OriginalPath
        & $EmbeddedPython -I -s -E -B -c "import lightgbm, numpy, scipy, narwhals; print(lightgbm.__version__, numpy.__version__, scipy.__version__, narwhals.__version__)"
        if ($LASTEXITCODE -ne 0) {
            throw "The assembled embedded Python runtime could not import all required packages."
        }
    }
    finally {
        $env:PATH = $OriginalPath
    }

    Compress-Archive -Path (Join-Path $BundleRoot "*") -DestinationPath $BundleZip -CompressionLevel Optimal
    $BundleZipHash = (Get-FileHash -LiteralPath $BundleZip -Algorithm SHA256).Hash.ToLowerInvariant()
    Write-Host "Portable bundle: $BundleZip"
    Write-Host "SHA-256:        $BundleZipHash"
}
finally {
    $ScratchFullPath = Get-FullPath $BuildScratch
    $TempRoot = (Get-FullPath ([System.IO.Path]::GetTempPath())).TrimEnd("\") + "\"
    if ($ScratchFullPath.StartsWith($TempRoot, [System.StringComparison]::OrdinalIgnoreCase) -and
        (Split-Path -Leaf $ScratchFullPath) -like "lightgbm-portable-build-*") {
        Remove-Item -LiteralPath $ScratchFullPath -Recurse -Force -ErrorAction SilentlyContinue
    }
}
