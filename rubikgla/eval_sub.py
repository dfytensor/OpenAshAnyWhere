#!/usr/bin/env python3
"""子进程终评: 隔离主进程 CUDA 状态 (根治三连崩).
用法: python eval_sub.py --arch {v2,v3} --ckpt PATH --lens 4096,16384,65536
输出: 每长度三方后缀 NLL — 零state / relay-s0链(廉价s0接力capture) / v36-MS.
"""
import sys, os, argparse, random
os.environ.setdefault("DELTA_STAGES", "1")
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import deqlm_v35_relay as M
from deqlm_v5_30m import PT_CACHE

DEV = "cuda"
D, G, DH = M.D, M.KV_H * M.SLOTS, M.DH
STATIONS = 5
VAL = []


def capture_segmented(net, ids, chunk, max_seg=2048):
    """分段 capture + s0 链: 单内核 ≤2048 步, 避开 Windows TDR."""
    L = ids.shape[1]
    bounds_all = list(range(chunk - 1, L, chunk))
    banks = [[] for _ in range(STATIONS)]
    cur = None
    i = 0
    while i < L:
        j = min(i + max_seg, L)
        seg = ids[:, i:j]
        bnd = [b - i for b in bounds_all if i <= b < j]
        ends, bs = net.capture_states(seg, boundary_idx=bnd, s0_list=cur)
        for k in range(STATIONS):
            banks[k].append(bs[k])
        cur = ends
        i = j
    return [torch.cat(b, dim=1) for b in banks]


class MSReadoutV2(nn.Module):
    """v2 单点末端注入读出 (冠军 checkpoint 的架构)."""

    def __init__(self):
        super().__init__()
        self.ms_wq = nn.ModuleList([nn.Linear(D, D, bias=False) for _ in range(STATIONS)])
        self.ms_wscore = nn.ModuleList([nn.Linear(DH, DH) for _ in range(STATIONS)])
        self.ms_down = nn.ModuleList([nn.Linear(G * DH, D, bias=False) for _ in range(STATIONS)])
        self.ms_ln = M.MARMSNorm(D)

    def forward(self, h, banks, chunk):
        B, L, _ = h.shape
        K = banks[0].shape[1]
        cidx = (torch.arange(L, device=h.device) // chunk).clamp(max=K - 1)
        y_ms = torch.zeros_like(h)
        for j in range(STATIONS):
            Sst = banks[j].float()
            q = self.ms_wq[j](h).view(B, L, G, DH).float()
            ws = self.ms_wscore[j](Sst.float().mean(dim=-1))
            scores = torch.einsum("blgd,bkgd->blk", q, ws)
            mask = torch.arange(K, device=h.device).view(1, 1, K) >= cidx.view(1, L, 1)
            w = torch.softmax(scores.masked_fill(mask, -1e9), dim=-1) * (cidx > 0).float().view(1, L, 1)
            y = torch.zeros(B, L, G, DH, device=h.device, dtype=torch.float32)
            for k in range(K):                         # K 循环: 防 TDR (16k+ 大中间张量)
                y = y + w[:, :, k].unsqueeze(-1).unsqueeze(-1) * torch.einsum(
                    "blgd,bgdt->blgt", q, Sst[:, k])
            y_ms = y_ms + self.ms_down[j](y.reshape(B, L, G * DH).to(h.dtype))
        return self.ms_ln(y_ms)


class V2Model(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net
        self.ms = MSReadoutV2()

    def forward(self, ids, chunk=128, ramp=1.0):
        was_training = self.net.training
        banks = capture_segmented(self.net, ids, chunk)
        if was_training:
            self.net.train()
        banks = [b.detach() for b in banks]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            hh = self.net.emb(ids)
            for st in (self.net.pre, self.net.loop_a, self.net.bridge, self.net.loop_c, self.net.post):
                hh = st(hh)
        h = hh.float()
        if ramp != 0.0:
            h = h + ramp * self.ms(h, banks, chunk)
        return self.net.head(self.net.ln_f(h))


class V3Model(nn.Module):
    """v3 每站中途注入 (与当前训练同构)."""

    def __init__(self, net):
        super().__init__()
        self.net = net
        from v36_multistate import StationReadout
        self.ms = nn.ModuleList([StationReadout() for _ in range(STATIONS)])

    def forward(self, ids, chunk=128, ramp=1.0):
        was_training = self.net.training
        banks = capture_segmented(self.net, ids, chunk)
        if was_training:
            self.net.train()
        banks = [b.detach() for b in banks]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            hh = self.net.emb(ids)
            for j, st in enumerate((self.net.pre, self.net.loop_a,
                                    self.net.bridge, self.net.loop_c, self.net.post)):
                hh = st(hh)
                if ramp != 0.0:
                    hh = hh + ramp * self.ms[j](hh, banks[j], chunk)
        return self.net.head(self.net.ln_f(hh.float()))


def build_block(n, ln, rng):
    out = []
    while len(out) < n:
        buf, tot = [], 0
        while tot < ln:
            s = VAL[rng.randrange(len(VAL))]
            buf.append(s)
            tot += len(s)
        out.append(torch.cat(buf)[:ln])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=["v2", "v3"], required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--lens", default="4096,16384,65536")
    args = ap.parse_args()
    torch.manual_seed(0)
    rng = random.Random(7)
    seqs = torch.load(PT_CACHE, map_location="cpu", weights_only=True)
    VAL.extend(s for s in seqs[int(0.95 * len(seqs)):][:2000] if len(s) >= 256)

    net = M.DeltaLoopLM()
    sd = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    missing, unexpected = net.load_state_dict(sd, strict=False)
    print("加载 %s: missing %d | unexpected %d" % (args.arch, len(missing), len(unexpected)), flush=True)
    gate = args.arch == "v3"
    for blk in (net.pre, net.loop_a.attn, net.bridge, net.loop_c.attn, net.post):
        blk.attn.use_gate = gate
        blk.attn.use_sel = False
    net = net.to(DEV).eval()
    model = (V2Model(net) if args.arch == "v2" else V3Model(net)).to(DEV).eval()

    for L in [int(s) for s in args.lens.split(",")]:
        n = max(2, min(8, (len(VAL) * 250) // L))
        ids = torch.stack(build_block(n, L, rng)).to(DEV)
        SUFF = min(2048, L // 2)
        tgt = ids[:, L - SUFF + 1:L]

        def suffix(logits, start):
            lo = logits[:, start:start + SUFF - 1].float()
            return F.cross_entropy(lo.reshape(-1, lo.shape[-1]), tgt.reshape(-1), ignore_index=0).item()

        with torch.no_grad():
            lo_zero = net(ids[:, L - SUFF:])
            cur_states = None
            lo_last = None
            n_seg = L // SUFF
            for i in range(n_seg):
                seg = ids[:, i * SUFF:(i + 1) * SUFF]
                cur_states = net.capture_states(seg, s0_list=cur_states)   # 廉价s0接力
                if i == n_seg - 1:
                    lo_last = net(seg, s0_list=cur_states)
            lo_ms = model(ids, chunk=min(128, L // 4), ramp=1.0)
        print("L=%6d 后缀NLL: 零state %.4f | relay-s0链 %.4f | v36-MS %.4f"
              % (L, suffix(lo_zero, 0), suffix(lo_last, 0), suffix(lo_ms, L - SUFF)), flush=True)
    print("EVAL DONE", flush=True)


if __name__ == "__main__":
    main()
