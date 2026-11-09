# 波次脚本: 子词级 24M 档 (d=512, L=8, 6000 步), 两波 × 各 2 并行
$py = "F:\OpenASH2605\.venv\Scripts\python.exe"
Set-Location "F:\OpenASH2605\v6_attbias"

function Wait-Wave($jobs) {
    foreach ($j in $jobs) { Wait-Process -Id $j.Id }
}

# 波 1: rope + alibi
$p1 = Start-Process -FilePath $py -ArgumentList '-u','train6.py','--bias','rope','--src','sub','--d','512','--n_layer','8','--steps','6000','--seed','1','--rundir','runs/sub24m_rope_s1' -RedirectStandardOutput 'runs/sub24m_rope_s1.log' -RedirectStandardError 'runs/sub24m_rope_s1.err' -PassThru -NoNewWindow
$p2 = Start-Process -FilePath $py -ArgumentList '-u','train6.py','--bias','alibi','--src','sub','--d','512','--n_layer','8','--steps','6000','--seed','1','--rundir','runs/sub24m_alibi_s1' -RedirectStandardOutput 'runs/sub24m_alibi_s1.log' -RedirectStandardError 'runs/sub24m_alibi_s1.err' -PassThru -NoNewWindow
Wait-Wave @($p1, $p2)
Write-Output "wave1 done"

# 波 2: ballistic2 (random) + ballistic2 (fitted)
$p3 = Start-Process -FilePath $py -ArgumentList '-u','train6.py','--bias','ballistic2','--src','sub','--d','512','--n_layer','8','--steps','6000','--seed','1','--rundir','runs/sub24m_ballistic2_s1' -RedirectStandardOutput 'runs/sub24m_ballistic2_s1.log' -RedirectStandardError 'runs/sub24m_ballistic2_s1.err' -PassThru -NoNewWindow
$p4 = Start-Process -FilePath $py -ArgumentList '-u','train6.py','--bias','ballistic2','--src','sub','--d','512','--n_layer','8','--steps','6000','--seed','1','--bias_init','bias_init_sub.json','--rundir','runs/sub24m_ballistic2fit_s1' -RedirectStandardOutput 'runs/sub24m_ballistic2fit_s1.log' -RedirectStandardError 'runs/sub24m_ballistic2fit_s1.err' -PassThru -NoNewWindow
Wait-Wave @($p3, $p4)
Write-Output "ALL DONE"
