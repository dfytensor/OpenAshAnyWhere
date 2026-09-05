# 循环重启 stack_eval.py m-x 直到 LM ckpt 完成
$py = 'E:\minimax_h3_run\.venv\Scripts\python.exe'
for ($i = 0; $i -lt 20; $i++) {
    Add-Content 'F:\OpenASH2605\metaru\run_mx.log' "=== 第 $i 次 ==="
    & $py -u -X utf8 'F:\OpenASH2605\metaru\stack_eval.py' m-x *>> 'F:\OpenASH2605\metaru\run_mx.log'
    $rc = $LASTEXITCODE
    Add-Content 'F:\OpenASH2605\metaru\run_mx.log' "=== 退出码 $rc @ $(Get-Date) ==="
    if ($rc -eq 0) { Write-Output "clean"; break }
    Start-Sleep -Seconds 2
}
