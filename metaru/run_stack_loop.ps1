param([string]$order = "b-xm")
$py = 'E:\minimax_h3_run\.venv\Scripts\python.exe'
$done = "F:\OpenASH2605\metaru\recall_stack_$order.done"
$log = "F:\OpenASH2605\metaru\run_$order.log"
for ($i = 0; $i -lt 30; $i++) {
    if (Test-Path $done) { Write-Output "DONE"; break }
    Add-Content $log "=== $order 第 $i 次 ==="
    & $py -u -X utf8 "F:\OpenASH2605\metaru\stack_test.py" $order *>> $log
    $rc = $LASTEXITCODE
    Add-Content $log "=== 退出码 $rc @ $(Get-Date) ==="
    if ($rc -eq 0) { Write-Output "clean"; break }
    Start-Sleep -Seconds 2
}
