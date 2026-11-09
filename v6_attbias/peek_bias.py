# -*- coding: utf-8 -*-
import json, numpy as np, sys, glob, os
sys.stdout.reconfigure(encoding='utf-8')
for rp in sorted(glob.glob(r'F:\OpenASH2605\v6_attbias\runs\ballistic*\result.json')):
    r = json.load(open(rp, encoding='utf-8'))
    bp = r['bias_params']
    run = os.path.basename(os.path.dirname(rp))
    print(run, 'val_loss', round(r['final_val_loss'], 3))
    print('  scale:', np.round(bp['scale'], 2).tolist())
    for h in range(len(bp['R'])):
        print(f'  head{h}: R={np.round(bp["R"][h],1).tolist()} sig={np.round(bp["sig"][h],1).tolist()} w={np.round(bp["w"][h],3).tolist()}')
    b = np.load(rp.replace('result.json', 'mix_bias.npy'))
    print('  bias[h0] @ d=1,10,50,256:', np.round(b[0][[0, 9, 49, 255]], 2).tolist())
