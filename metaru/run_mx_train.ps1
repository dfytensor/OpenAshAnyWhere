# 循环重启 stack_test.py m-x 直到召回 done
$py = 'E:\minimax_h3_run\.venv\Scripts\python.exe'
$done = 'F:\OpenASH2605\metaru\recall_stack_m-x.done'
for ($i = 0; $i -lt 20; $i++) {
    if (Test-Path $done) { Write-Output "DONE"; break }
    Add-Content 'F:\OpenASH2605\metaru\run_mx.log' "=== mx 训练第 $i 次 ==="
    & $py -u -X utf8 'F:\OpenASH2605\metaru\stack_test.py' m-x *>> 'F:\OpenASH2605\metaru\run_mx.log'
    $rc = $LASTEXITCODE
    Add-Content 'F:\OpenASH2605\metaru\run_mx.log' "=== 退出码 $rc @ $(Get-Date) ==="
    Start-Sleep -Seconds 2
}
