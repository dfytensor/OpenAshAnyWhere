# 过夜 100M 级验证: d=768 L=12 (~88M 参数), ctx=512, bs=16, 15000 步 (~123M token, 1.5 epoch)
# 三方案同种子同数据序: rope / alibi / ballistic2, 三路并发
$py = "F:\OpenASH2605\.venv\Scripts\python.exe"
Set-Location "F:\OpenASH2605\v6_attbias"

function Start-Run($bias, $name) {
    $args = @('-u','train6.py','--bias',$bias,'--src','sub','--sub_path','sub_stream_big.npy','--d','768','--n_layer','12','--steps','15000','--bs','16','--ctx','512','--lr','2e-4','--seed','1','--rundir',"runs/big_$name")
    return Start-Process -FilePath $py -ArgumentList $args -RedirectStandardOutput "runs/big_$name.log" -RedirectStandardError "runs/big_$name.err" -PassThru -NoNewWindow
}
function Wait-Wave($jobs) { foreach ($j in $jobs) { Wait-Process -Id $j.Id } }

$w = @( (Start-Run 'rope' 'rope_s1'), (Start-Run 'alibi' 'alibi_s1'), (Start-Run 'ballistic2' 'ballistic2_s1') )
Wait-Wave $w
Write-Output "ALL DONE"
