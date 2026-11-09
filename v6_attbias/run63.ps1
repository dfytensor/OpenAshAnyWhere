# v6.3 修正索引后的对照: 子词级 11.4M × {alibi, ballistic2, ballistic3} × 2 种子, 两波 × 3 并行
$py = "F:\OpenASH2605\.venv\Scripts\python.exe"
Set-Location "F:\OpenASH2605\v6_attbias"

function Start-Run($bias, $seed, $name, $extra) {
    $args = @('-u','train6.py','--bias',$bias,'--src','sub','--d','384','--n_layer','6','--steps','6000','--seed',"$seed",'--rundir',"runs/$name")
    if ($extra) { $args += @('--bias_init','bias_init_sub.json') }
    return Start-Process -FilePath $py -ArgumentList $args -RedirectStandardOutput "runs/$name.log" -RedirectStandardError "runs/$name.err" -PassThru -NoNewWindow
}
function Wait-Wave($jobs) { foreach ($j in $jobs) { Wait-Process -Id $j.Id } }

$w1 = @( (Start-Run 'alibi' 1 'subd_alibi_s1' $false), (Start-Run 'ballistic2' 1 'subd_ballistic2_s1' $false), (Start-Run 'ballistic3' 1 'subd_ballistic3_s1' $false) )
Wait-Wave $w1
Write-Output "wave1 done"

$w2 = @( (Start-Run 'alibi' 2 'subd_alibi_s2' $false), (Start-Run 'ballistic2' 2 'subd_ballistic2_s2' $false), (Start-Run 'ballistic3' 2 'subd_ballistic3_s2' $false) )
Wait-Wave $w2
Write-Output "ALL DONE"
