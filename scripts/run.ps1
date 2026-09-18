# Start the SkyOps API + console on http://127.0.0.1:8000
param([int]$Port = 8000, [switch]$Reload)
Set-Location (Join-Path $PSScriptRoot "..")
$args_ = @("-m", "uvicorn", "skyops.api.main:app", "--host", "127.0.0.1", "--port", "$Port")
if ($Reload) { $args_ += "--reload" }
& ".\.venv\Scripts\python.exe" @args_
