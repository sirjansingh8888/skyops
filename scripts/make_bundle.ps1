<#
Pack the project for another laptop (USB / Drive / chat). Uses Windows' built-in tar (bsdtar).
  .\scripts\make_bundle.ps1                  source + git history (~5 MB)
  .\scripts\make_bundle.ps1 -IncludeData     + data\raw (~550 MB with the default subsets)
  .\scripts\make_bundle.ps1 -IncludeModels   + models\ (trained weights)
Output: ..\skyops_bundle_<yyyyMMdd_HHmm>.zip. Never includes .venv, runs\, caches or .env.
#>
param([switch]$IncludeData, [switch]$IncludeModels, [string]$OutDir = "")
$ErrorActionPreference = "Stop"
$root = Resolve-Path (Join-Path $PSScriptRoot "..")
$parent = Split-Path $root -Parent
$name = Split-Path $root -Leaf
if ($OutDir -eq "") { $OutDir = $parent }
$stamp = Get-Date -Format "yyyyMMdd_HHmm"
$zip = Join-Path $OutDir "skyops_bundle_$stamp.zip"

$ex = @("--exclude=$name/.venv", "--exclude=$name/runs", "--exclude=$name/.pytest_cache", "--exclude=$name/.env",
        "--exclude=*/__pycache__", "--exclude=$name/data/processed", "--exclude=$name/*.zip", "--exclude=$name/yolov8*.pt")
if (-not $IncludeData)   { $ex += "--exclude=$name/data/raw" }
if (-not $IncludeModels) { $ex += "--exclude=$name/models" }

Push-Location $parent
try {
  & tar -a -c -f $zip @ex $name
} finally { Pop-Location }
$mb = [math]::Round((Get-Item $zip).Length / 1MB, 1)
Write-Host "Bundle written: $zip ($mb MB)"
Write-Host "On the other laptop: unzip, then .\scripts\setup.ps1 -Cuda ; python scripts\download_data.py --all (unless -IncludeData) ; .\scripts\run.ps1"
