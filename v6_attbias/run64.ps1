# v6.4 决定性实验: rope_abs (RoPE+可学习绝对先验) vs rope 基线, 各 2 种子, 外加 rope s3 加固
$py = "F:\OpenASH2605\.venv\Scripts\python.exe"
Set-Location "F:\OpenASH2605\v6_attbias"

function Start-Run($bias, $seed, $name) {
    $args = @('-u','train6.py','--bias',$bias,'--src','sub','--d','384','--n_layer','6','--steps','6000','--bs','32','--ctx','512','--seed',"$seed",'--eval_ctxs','512,1024,2048','--rundir',"runs/$name")
    return Start-Process -FilePath $py -ArgumentList $args -RedirectStandardOutput "runs/$name.log" -RedirectStandardError "runs/$name.err" -PassThru -NoNewWindow
}
function Wait-Wave($jobs) { foreach ($j in $jobs) { Wait-Process -Id $j.Id } }

$w1 = @( (Start-Run 'rope_abs' 1 'sub64_ropeabs_s1'), (Start-Run 'rope_abs' 2 'sub64_ropeabs_s2'), (Start-Run 'rope' 3 'sub_rope_s3') )
Wait-Wave $w1
Write-Output "wave1 done"
$w2 = @( (Start-Run 'rope_abs' 3 'sub64_ropeabs_s3'), (Start-Run 'rope' 4 'sub_rope_s4'), (Start-Run 'ballistic_abs' 1 'sub64_ballisticabs_s1') )
Wait-Wave $w2
Write-Output "ALL DONE"
