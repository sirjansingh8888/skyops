<#
Create the virtual environment and install dependencies.
  .\scripts\setup.ps1          CPU PyTorch (any laptop)
  .\scripts\setup.ps1 -Cuda    CUDA 12.8 PyTorch (RTX 50-series / any recent NVIDIA driver)
#>
param([switch]$Cuda)
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")

if (-not (Test-Path ".venv")) { python -m venv .venv }
$py = ".\.venv\Scripts\python.exe"
& $py -m pip install --upgrade pip
if ($Cuda) { & $py -m pip install -r requirements-torch-cu128.txt } else { & $py -m pip install -r requirements-torch-cpu.txt }
& $py -m pip install -r requirements.txt
& $py -c "import torch, ultralytics; print('torch', torch.__version__, '| cuda available:', torch.cuda.is_available(), '| ultralytics', ultralytics.__version__)"
if (-not (Test-Path ".env")) { Copy-Item .env.example .env; Write-Host "Created .env from .env.example - add ANTHROPIC_API_KEY for the assistant." }
Write-Host "`nNext: $py scripts\download_data.py --all   then   .\scripts\run.ps1"
