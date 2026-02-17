# Paper Trading Bot Launcher
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONLEGACYWINDOWSSTDIO = "utf-8"

# Read from persistent user env vars (set via setx)
if (-not $env:ASTER_API_KEY) {
    $env:ASTER_API_KEY = [System.Environment]::GetEnvironmentVariable('ASTER_API_KEY', 'User')
}
if (-not $env:ASTER_API_SECRET) {
    $env:ASTER_API_SECRET = [System.Environment]::GetEnvironmentVariable('ASTER_API_SECRET', 'User')
}

if (-not $env:ASTER_API_KEY -or -not $env:ASTER_API_SECRET) {
    Write-Host "WARNING: ASTER_API_KEY and ASTER_API_SECRET not found. Paper trading needs them for market data." -ForegroundColor Yellow
}

Write-Host "Starting Paper Trading Bot... API Key: $($env:ASTER_API_KEY.Substring(0,8))..." -ForegroundColor Green

Set-Location C:\Users\Never\.openclaw\workspace
& C:\Users\Never\AppData\Local\Programs\Python\Python312\python.exe -m trading.run_paper --capital 10000 --profile medium