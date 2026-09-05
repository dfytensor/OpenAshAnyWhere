# 循环重启 stack_test x-m 直到完成 (处理环境杀进程)
$py = 'E:\minimax_h3_run\.venv\Scripts\python.exe'
$done = 'F:\OpenASH2605\metaru\recall_stack_x-m.done'
$out = 'F:\OpenASH2605\metaru\run_stack_xm.log'
for ($i = 0; $i -lt 12; $i++) {
    if (Test-Path $done) { Write-Output "DONE"; break }
    Add-Content $out "=== 第 $i 次尝试 ==="
    & $py -u -X utf8 'F:\OpenASH2605\metaru\stack_test.py' x-m *>> $out
    $rc = $LASTEXITCODE
    Add-Content $out "=== 退出码 $rc @ $(Get-Date) ==="
    if ($rc -eq 0) { Write-Output "clean exit"; break }
    Start-Sleep -Seconds 3
}
