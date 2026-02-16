$action = New-ScheduledTaskAction -Execute 'wscript.exe' -Argument '"C:\Users\Never\.openclaw\workspace\trading\sync_dashboard_silent.vbs"'
Set-ScheduledTask -TaskName 'AIT_DashboardSync' -Action $action
Write-Output "Task updated successfully"
