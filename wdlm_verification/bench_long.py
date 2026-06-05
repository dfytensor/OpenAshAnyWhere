import torch, time, sys
sys.path.insert(0,'F:/OpenASH2605'); sys.path.insert(0,'F:/OpenASH2605/wdlm_verification')
from open_ash_v2 import OpenASH_V2
from wdlm_real import WaveDynamicsLM_Real
import torch.nn as nn, torch.nn.functional as F

V,H=5000,128; L=2; B=1; d='cuda'

class TBlk(nn.Module):
    def __init__(self):
        super().__init__()
        self.atn = nn.MultiheadAttention(128, 4, batch_first=True, bias=False)
        self.ffn = nn.Sequential(nn.Linear(128, 512), nn.GELU(), nn.Linear(512, 128))
        self.n1 = nn.LayerNorm(128)
        self.n2 = nn.LayerNorm(128)

    def forward(self, x):
        S = x.size(1)
        m = torch.triu(torch.ones(S, S, device=x.device) * float('-inf'), 1)
        a, _ = self.atn(x, x, x, attn_mask=m)
        x = self.n1(x + a)
        return self.n2(x + self.ffn(x))

class TLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(V, 128)
        self.layers = nn.ModuleList([TBlk() for _ in range(L)])
        self.head = nn.Linear(128, V, bias=False)

    def forward(self, x):
        h = self.emb(x)
        for l in self.layers:
            h = l(h)
        return self.head(h)

torch.manual_seed(42)
ash = OpenASH_V2(V, 128, 4, L).cuda().eval()
wdl = WaveDynamicsLM_Real(V, 128, L, 1).cuda().eval()
tr = TLM().cuda().eval()

@torch.no_grad()
def gen_state(model, prefix, n):
    ids = prefix.clone()
    state = None
    for _ in range(n):
        out = model(ids[:, -1:], state)
        if isinstance(out, tuple):
            o = out[0]; state = out[-1]  # state is last element
        else:
            o = out
        nt = o[:, -1, :].argmax(-1)
        ids = torch.cat([ids, nt.unsqueeze(1)], 1)
    return ids

@torch.no_grad()
def gen_naive(model, prefix, n, max_ctx=512):
    ids = prefix.clone()
    for _ in range(n):
        out = model(ids[:, -max_ctx:])
        if isinstance(out, tuple):
            o = out[0]
        else:
            o = out
        nt = o[:, -1, :].argmax(-1)
        ids = torch.cat([ids, nt.unsqueeze(1)], 1)
    return ids

prefix_len = 256
new_tokens = 128
px = torch.randint(0, V, (B, prefix_len), device=d)

naive_tokens = sum(prefix_len + i for i in range(new_tokens))
state_tokens = prefix_len + new_tokens

print(f'Generate {new_tokens} tokens from {prefix_len}-token prefix:')
print(f'State: {state_tokens} tokens total | Naive: {naive_tokens:,} tokens ({naive_tokens/state_tokens:.0f}x more)')
print()

models = [
    ('Transformer (naive)', tr, gen_naive),
    ('OpenASH_V2 (state)', ash, gen_state),
    ('WDLM-Real (state)', wdl, gen_state),
    ('OpenASH_V2 (naive)', ash, gen_naive),
    ('WDLM-Real (naive)', wdl, gen_naive),
]

print(f'{"Model":25s} {"Time":>8s} {"Tokens/s":>10s}')
for name, model, fn in models:
    is_state = 'state' in name
    torch.cuda.synchronize(); t0 = time.time()
    fn(model, px, new_tokens)
    torch.cuda.synchronize(); t = time.time() - t0
    raw_tok = naive_tokens / t if not is_state else state_tokens / t
    print(f'{name:25s} {t:>7.3f}s  {raw_tok:>9.0f}')

print(f'\nAt 1024 prefix + 256 new: naive processes {sum(1024+i for i in range(256)) // (1024+256)}x more tokens')
