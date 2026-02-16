Stop-ScheduledTask -TaskName 'AsterTradingBot' -ErrorAction SilentlyContinue
Start-Sleep 2
# Kill all python processes except dashboard (PID 13192)
Get-Process python* -ErrorAction SilentlyContinue | ForEach-Object {
    if ($_.Id -ne 13192) {
        Write-Output "Killing PID $($_.Id)"
        Stop-Process -Id $_.Id -Force
    }
}
Start-Sleep 2
Write-Output "All bots stopped."
Get-Process python* -ErrorAction SilentlyContinue | Select-Object Id, StartTime
