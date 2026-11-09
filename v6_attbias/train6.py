# -*- coding: utf-8 -*-
"""v6 训练: 四种位置方案在小字符 LM 上的对照
  python train6.py --bias rope --seed 1 --steps 4000 --rundir runs/rope_s1
评估: 每 250 步在固定 val crops 上测 loss/ppl; 结束后导出弹道核参数 (R, σ, w, scale)
"""
import sys, os, json, time, argparse, math
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import torch
import torch.nn.functional as F
from bmodel import TinyLM, BallisticBias
from data6 import CharStream, SubStream, V as V_CHAR


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bias', required=True,
                    choices=['rope', 'alibi', 'ballistic', 'ballistic_ng', 'ballistic2', 'rope_ballistic',
                             'ballistic3', 'ballistic_abs', 'rope_abs'])
    ap.add_argument('--src', default='char', choices=['char', 'sub'])
    ap.add_argument('--V', type=int, default=0, help='0 = 按 src 默认 (char 1500 / sub 2048)')
    ap.add_argument('--bias_init', default=None, help='autoballistic 导出的 bias_init.json')
    ap.add_argument('--eval_ctxs', default='', help='逗号分隔的评估长度, 如 512,1024,2048 (零样本长度外推)')
    ap.add_argument('--sub_path', default='', help='子词流 npy 路径 (默认 sub_stream.npy)')
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--steps', type=int, default=4000)
    ap.add_argument('--bs', type=int, default=64)
    ap.add_argument('--ctx', type=int, default=256)
    ap.add_argument('--d', type=int, default=256)
    ap.add_argument('--n_layer', type=int, default=4)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--rundir', default=None)
    a = ap.parse_args()
    run_name = a.rundir or f'runs/{a.bias}_s{a.seed}'
    os.makedirs(run_name, exist_ok=True)
    torch.manual_seed(a.seed)
    dev = 'cuda'
    t0 = time.time()

    print(f'== bias={a.bias} seed={a.seed} steps={a.steps} bs={a.bs} ctx={a.ctx} d={a.d} L={a.n_layer} src={a.src}')
    if a.src == 'char':
        data = CharStream()
        Vv = a.V or V_CHAR
    else:
        data = SubStream(a.sub_path or None)
        Vv = a.V or 2048
        print(f'  sub stream: train {len(data.train):,} / val {len(data.val):,}')
    g = torch.Generator().manual_seed(a.seed)
    binit = None
    if a.bias_init:
        binit = json.load(open(a.bias_init, encoding='utf-8'))
        print(f"  bias_init: J={binit['J']} R={binit['R']} sig={np.round(binit['sig'],2)}")
    model = TinyLM(Vv, d=a.d, n_layer=a.n_layer, h=4, bias_kind=a.bias,
                   ng_init=(a.bias == 'ballistic_ng'), ctx=a.ctx, bias_init=binit).to(dev)
    nparam = sum(p.numel() for p in model.parameters())
    print(f'params {nparam:,}  V={Vv}')
    log = open(os.path.join(run_name, 'log.jsonl'), 'w', encoding='utf-8')

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.95), weight_decay=0.1)
    warm = 200

    def lr_at(s):
        if s < warm:
            return s / warm
        t = (s - warm) / max(1, a.steps - warm)
        return 0.1 + 0.45 * (1 + math.cos(math.pi * t))   # 0.55→0.1

    # 固定 val crops
    vg = torch.Generator().manual_seed(123)
    val_x = data.crop(data.val, 24, a.ctx, vg).to(dev)
    val_y = torch.cat([val_x[:, 1:], val_x[:, -1:]], 1)

    def eval_val():
        model.eval()
        with torch.no_grad():
            loss = 0.0
            for i in range(0, len(val_x), 8):
                out = model(val_x[i:i + 8])
                loss += F.cross_entropy(out[:, :-1].reshape(-1, Vv), val_y[i:i + 8, :-1].reshape(-1)).item() * 8
        model.train()
        return loss / len(val_x)

    best = 1e9
    for s in range(a.steps):
        for p in opt.param_groups:
            p['lr'] = a.lr * lr_at(s)
        x = data.crop(data.train, a.bs, a.ctx, g).to(dev)
        y = torch.cat([x[:, 1:], x[:, -1:]], 1)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            out = model(x)
            loss = F.cross_entropy(out[:, :-1].reshape(-1, Vv), y[:, :-1].reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if (s + 1) % 250 == 0 or s == 0:
            vl = eval_val()
            best = min(best, vl)
            rec = dict(step=s + 1, loss=float(loss.item()), val_loss=float(vl), val_ppl=float(np.exp(vl)))
            log.write(json.dumps(rec) + '\n'); log.flush()
            print(f'  step {s+1:>5}  loss {loss.item():.3f}  val {vl:.3f}  ppl {np.exp(vl):7.2f}  [{time.time()-t0:.0f}s]', flush=True)

    result = dict(bias=a.bias, seed=a.seed, steps=a.steps, params=nparam,
                  final_val_loss=float(eval_val()), best_val_loss=float(best),
                  final_ppl=float(np.exp(eval_val())), elapsed=float(time.time() - t0),
                  ctx=a.ctx, src=a.src)

    # 零样本长度外推评估: 在训练 ctx 之外的长度上测 val loss
    if a.eval_ctxs:
        le = {}
        for Ls in a.eval_ctxs.split(','):
            L = int(Ls)
            if L > len(data.val) - 2:
                continue
            Bc = max(2, min(8, int(3e8 / (4 * L * L))))
            g2 = torch.Generator().manual_seed(1000 + L)
            vx = data.crop(data.val, Bc, L, g2).to(dev)
            vy = torch.cat([vx[:, 1:], vx[:, -1:]], 1)
            model.eval()
            with torch.no_grad():
                tot = 0.0
                out = model(vx)
                tot = F.cross_entropy(out[:, :-1].reshape(-1, Vv), vy[:, :-1].reshape(-1)).item()
            model.train()
            le[str(L)] = float(tot)
            print(f'  [len-eval] ctx={L}: loss {tot:.3f} ppl {np.exp(tot):.2f} (B={Bc})', flush=True)
        result['length_eval'] = le
        torch.save(model.state_dict(), os.path.join(run_name, 'model.pt'))

    bj = model.ngram_bias_json()
    if bj:
        result['bias_params'] = bj
        # 保存逐头混合核偏置曲线 (H, ctx)
        with torch.no_grad():
            b = model.bias.bias().detach().cpu().numpy()
        np.save(os.path.join(run_name, 'mix_bias.npy'), b)
    with open(os.path.join(run_name, 'result.json'), 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=1)
    log.close()
    print(json.dumps({k: v for k, v in result.items() if k != 'bias_params'}, indent=1))


if __name__ == '__main__':
    main()
