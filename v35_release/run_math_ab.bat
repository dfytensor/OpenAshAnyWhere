@echo off
cd /d F:\OpenASH2605
echo === SFT-A v53f1 (no read) ===
.venv\Scripts\python.exe -u train_sft.py --base F:\OpenASH2605\full_v53f1_40k\model.pt --arch v53f1 --data F:\OpenASH2605\minimind_data\math_cot_cached.pt --out F:\OpenASH2605\mathsft_v53f1 --steps 3000
echo === SFT-B v53rf1 (read channel, fresh read params) ===
.venv\Scripts\python.exe -u train_sft.py --base F:\OpenASH2605\full_v53f1_40k\model.pt --arch v53rf1 --data F:\OpenASH2605\minimind_data\math_cot_cached.pt --out F:\OpenASH2605\mathsft_v53rf1 --steps 3000
echo === EVAL-A v53f1 ===
.venv\Scripts\python.exe -u eval_math.py --ckpt F:\OpenASH2605\mathsft_v53f1\model.pt --arch v53f1
echo === EVAL-B v53rf1 ===
.venv\Scripts\python.exe -u eval_math.py --ckpt F:\OpenASH2605\mathsft_v53rf1\model.pt --arch v53rf1
echo === ALL DONE ===
