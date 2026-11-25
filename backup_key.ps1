# backup_key.ps1 — 关键产物异地备份（F: 磁盘1 -> E: 磁盘0），复制后逐个 CRC 校验
$ErrorActionPreference = 'Stop'
$src = 'F:\OpenASH2605'
$dst = 'E:\OpenASH_ckpt_backup_20261126'
New-Item -ItemType Directory -Force -Path $dst | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $dst 'minimind_data') | Out-Null

$keys = @(
  'full_v53dms_80k', 'full_v53dme_80k', 'full_v53dmh_80k',
  'full_gdn_160k', 'full_gdn_80k_fixed',
  'full_v53dm_v2_40k', 'full_v53dms_40k', 'full_v53dmh_40k',
  'full_v53kr_40k', 'full_v53f1_40k', 'full_v53e2_40k', 'full_v53d7_40k',
  'full_v53wr_40k', 'full_v53dme_40k_t1024', 'full_v53dmg_40k',
  'sft_v53f1', 'grpo_v53f1', 'grpo_v53f1_c'
)
$t0 = Get-Date
$copied = @()
foreach ($k in $keys) {
  $d = Join-Path $src $k
  if (Test-Path (Join-Path $d 'model.pt')) {
    robocopy $d (Join-Path $dst $k) /E /NJH /NJS /NDL /NP /R:2 /W:2 | Out-Null
    $copied += $k
  } else {
    Write-Output "SKIP (no model.pt): $k"
  }
}
# 数据缓存（最重要的单一产物）
robocopy (Join-Path $src 'minimind_data') (Join-Path $dst 'minimind_data') pretrain_cached_1270238_384.pt /NJH /NJS /NDL /NP /R:2 /W:2 | Out-Null
Write-Output ("copied {0} checkpoint dirs + data cache in {1:N0}s" -f $copied.Count, ((Get-Date) - $t0).TotalSeconds)

# 校验
$verify = @($copied | ForEach-Object { 'E:\OpenASH_ckpt_backup_20261126\' + $_ })
& 'F:\OpenASH2605\.venv\Scripts\python.exe' -u 'F:\OpenASH2605\verify_ckpts.py' @verify
& 'F:\OpenASH2605\.venv\Scripts\python.exe' -u -c "import zipfile;z=zipfile.ZipFile(r'E:\OpenASH_ckpt_backup_20261126\minimind_data\pretrain_cached_1270238_384.pt');print('data copy testzip:', z.testzip())"
$total = (Get-ChildItem $dst -Recurse -File | Measure-Object Length -Sum).Sum
Write-Output ("backup total: {0:N1} GB at {1}" -f ($total/1GB), $dst)
