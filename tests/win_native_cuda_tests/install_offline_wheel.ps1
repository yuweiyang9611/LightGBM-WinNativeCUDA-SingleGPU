[CmdletBinding()]
param(
    [string]$PythonExecutable = "python"
)

$ErrorActionPreference = "Stop"

$ChecksumManifest = Join-Path $PSScriptRoot "SHA256SUMS.txt"
if (-not (Test-Path -LiteralPath $ChecksumManifest -PathType Leaf)) {
    throw "Offline package checksum manifest is missing: $ChecksumManifest"
}
foreach ($Line in Get-Content -LiteralPath $ChecksumManifest) {
    if ([string]::IsNullOrWhiteSpace($Line)) {
        continue
    }
    if ($Line -notmatch "^(?<Hash>[0-9a-fA-F]{64})  (?<Name>.+)$") {
        throw "Invalid checksum manifest entry: $Line"
    }
    $FilePath = Join-Path $PSScriptRoot $Matches.Name
    if (-not (Test-Path -LiteralPath $FilePath -PathType Leaf)) {
        throw "Offline package file is missing: $($Matches.Name)"
    }
    $ActualHash = (Get-FileHash -LiteralPath $FilePath -Algorithm SHA256).Hash
    if ($ActualHash -ne $Matches.Hash) {
        throw "Checksum mismatch for $($Matches.Name)."
    }
}

$Wheel = Get-ChildItem -LiteralPath $PSScriptRoot -Filter "lightgbm-*-win_amd64.whl" |
    Sort-Object Name |
    Select-Object -Last 1
if (-not $Wheel) {
    throw "No Windows x64 LightGBM wheel was found in $PSScriptRoot."
}

& $PythonExecutable -m pip install `
    --no-index `
    --find-links $PSScriptRoot `
    --force-reinstall `
    $Wheel.FullName
if ($LASTEXITCODE -ne 0) {
    throw "Offline LightGBM installation failed with exit code $LASTEXITCODE."
}

& $PythonExecutable -c "import lightgbm; print('LightGBM', lightgbm.__version__, 'installed from offline wheelhouse')"
if ($LASTEXITCODE -ne 0) {
    throw "LightGBM was installed but could not be imported."
}
