# 长上下文外推: ctx=512 训练, 1024/2048/4096 零样本评估. 3 方案 × 2 种子, 两波 × 3 并行
$py = "F:\OpenASH2605\.venv\Scripts\python.exe"
Set-Location "F:\OpenASH2605\v6_attbias"

function Start-Run($bias, $seed, $name) {
    $args = @('-u','train6.py','--bias',$bias,'--src','sub','--d','384','--n_layer','6','--steps','6000','--bs','32','--ctx','512','--seed',"$seed",'--eval_ctxs','512,1024,2048,4096','--rundir',"runs/$name")
    return Start-Process -FilePath $py -ArgumentList $args -RedirectStandardOutput "runs/$name.log" -RedirectStandardError "runs/$name.err" -PassThru -NoNewWindow
}
function Wait-Wave($jobs) { foreach ($j in $jobs) { Wait-Process -Id $j.Id } }

$w1 = @( (Start-Run 'rope' 1 'lc_rope_s1'), (Start-Run 'alibi' 1 'lc_alibi_s1'), (Start-Run 'ballistic2' 1 'lc_ballistic2_s1') )
Wait-Wave $w1
Write-Output "wave1 done"
$w2 = @( (Start-Run 'rope' 2 'lc_rope_s2'), (Start-Run 'alibi' 2 'lc_alibi_s2'), (Start-Run 'ballistic2' 2 'lc_ballistic2_s2') )
Wait-Wave $w2
Write-Output "ALL DONE"
