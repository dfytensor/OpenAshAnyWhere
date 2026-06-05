"""20M Benchmark: Perplexity, Generation, State speed, Long-context"""
import torch, time, sys, math, os, json
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

sys.path.insert(0, 'F:/OpenASH2605')
sys.path.insert(0, 'F:/OpenASH2605/wdlm_verification')
os.chdir('F:/OpenASH2605')
from open_ash_voc import OpenASHVoc
from config import agent_voc_path
from open_ash_v2 import OpenASH_V2
from wdlm_neural import WaveDynamicsLanguageModel as WN
from wdlm_real import WaveDynamicsLM_Real as WR
import torch.nn as nn

voc = OpenASHVoc(agent_voc_path=agent_voc_path)
vs = len(voc.token_to_id) + 1
dev = torch.device('cuda:0')
OUT_DIR = './train_20m'

# ============================================================
# Models (same config as training)
# ============================================================
MODEL_CONFIGS = {
    'WDLM-Neural+gen': lambda: WN(vs, hidden_dim=256, num_layers=9),
    'WDLM-Real':       lambda: WR(vs, hidden_dim=256, num_layers=12, evo_steps=1),
    'OpenASH_V2':      lambda: OpenASH_V2(vs, hidden_size=288, num_heads=9, num_layers=12),
    'Transformer':     None,
}

class TBlk(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        assert dim % heads == 0
        self.atn = nn.MultiheadAttention(dim, heads, batch_first=True, bias=False)
        self.ffn = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))
        self.n1 = nn.LayerNorm(dim); self.n2 = nn.LayerNorm(dim)
    def forward(self, x):
        S = x.size(1)
        m = torch.triu(torch.ones(S, S, device=x.device) * float('-inf'), 1)
        a, _ = self.atn(x, x, x, attn_mask=m)
        return self.n2(self.n1(x + a) + self.ffn(self.n1(x + a)))

class TLM(nn.Module):
    def __init__(self, vs, dim, heads, layers):
        super().__init__()
        self.emb = nn.Embedding(vs, dim)
        self.layers = nn.ModuleList([TBlk(dim, heads) for _ in range(layers)])
        self.head = nn.Linear(dim, vs, bias=False)
    def forward(self, x):
        h = self.emb(x)
        for l in self.layers: h = l(h)
        return self.head(h)


def load_model(name):
    if name == 'Transformer':
        m = TLM(vs, 256, 8, 10).to(dev)
    else:
        m = MODEL_CONFIGS[name]().to(dev)
    try:
        m.load_state_dict(torch.load(f'{OUT_DIR}/{name}_pretrain.pth', map_location=dev))
        print(f'Loaded {name} PT weights')
    except:
        print(f'No pretrain weights for {name}, using random init')
    return m


def count(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


# ============================================================
# 1. Perplexity on validation set
# ============================================================
@torch.no_grad()
def calc_ppl(model, data_path, max_len, n_samples=200):
    model.eval()
    tk = voc
    is_ = tk.token_to_id.get('<|im_start|>'); ie_ = tk.token_to_id.get('<|im_end|>')
    uid_ = tk.token_to_id.get('<|user|>'); aid_ = tk.token_to_id.get('<|agent|>')
    ts_ = tk.token_to_id.get('<|think|>'); te_ = tk.token_to_id.get('<|end_think|>')

    # Load last n_samples from SFT data
    lines = []
    with open(data_path, encoding='utf-8') as f:
        all_lines = [l.strip() for l in f if l.strip()]
        lines = all_lines[-5000:]

    total_loss = 0; total_tokens = 0
    for line in lines[:n_samples]:
        convs = json.loads(line).get('conversations', [])
        m = []
        for msg in convs:
            role = msg.get('role', '')
            ct = msg.get('content', '')
            if role == 'user':
                m += [is_, uid_] + tk.encode(ct) + [ie_]
            elif role == 'assistant':
                m += [is_, aid_]
                if msg.get('reasoning_content'):
                    m += [ts_] + tk.encode(msg['reasoning_content']) + [te_]
                m += tk.encode(ct) + [ie_]
        if len(m) < 4: continue
        if len(m) > max_len + 1: m = m[:max_len + 1]
        ids = torch.tensor(m, dtype=torch.long).unsqueeze(0).to(dev)
        inp, tgt = ids[:, :-1], ids[:, 1:]
        out = model(inp)
        if isinstance(out, tuple): out = out[0]
        loss = F.cross_entropy(out.view(-1, vs), tgt.view(-1), ignore_index=0)
        total_loss += loss.item() * (tgt != 0).sum().item()
        total_tokens += (tgt != 0).sum().item()
    return math.exp(total_loss / total_tokens) if total_tokens > 0 else float('inf')


# ============================================================
# 2. Generation quality test
# ============================================================
@torch.no_grad()
def generate(model, prompt_ids, max_new=50, temp=0.8, top_k=40):
    model.eval()
    ids = prompt_ids.clone()
    for _ in range(max_new):
        ctx = ids[:, -256:] if ids.size(1) > 256 else ids
        out = model(ctx)
        if isinstance(out, tuple): out = out[0]
        logits = out[:, -1, :] / temp
        if top_k:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits = logits.masked_fill(logits < v[:, [-1]], float('-inf'))
        probs = F.softmax(logits, dim=-1)
        nt = torch.multinomial(probs, 1)
        ids = torch.cat([ids, nt], dim=1)
    return ids


# ============================================================
# 3. State mode generation speed
# ============================================================
@torch.no_grad()
def gen_state_speed(model, prompt, new_tokens=64):
    model.eval()
    ids = prompt.clone()
    state = None
    torch.cuda.synchronize(); t0 = time.time()
    for _ in range(new_tokens):
        out = model(ids[:, -1:], state)
        if isinstance(out, tuple):
            o = out[0]; state = out[-1] if len(out) > 1 else None
        else:
            o = out
        nt = o[:, -1, :].argmax(-1)
        ids = torch.cat([ids, nt.unsqueeze(1)], 1)
    torch.cuda.synchronize(); t = time.time() - t0
    return t, ids


@torch.no_grad()
def gen_naive_speed(model, prompt, new_tokens=64, max_ctx=512):
    model.eval()
    ids = prompt.clone()
    torch.cuda.synchronize(); t0 = time.time()
    for _ in range(new_tokens):
        ctx = ids[:, -max_ctx:]
        out = model(ctx)
        if isinstance(out, tuple): out = out[0]
        nt = out[:, -1, :].argmax(-1)
        ids = torch.cat([ids, nt.unsqueeze(1)], 1)
    torch.cuda.synchronize(); t = time.time() - t0
    return t, ids


# ============================================================
# Run all benchmarks
# ============================================================
if __name__ == '__main__':
    print(f'{"="*70}')
    print(f'  20M Models Benchmark')
    print(f'{"="*70}\n')

    results = {}
    prompt_text = '人工智能技术的未来发展趋势包括'
    prompt_ids = torch.tensor([voc.encode(prompt_text)], dtype=torch.long).to(dev)
    print(f'Prompt: "{prompt_text}" ({prompt_ids.size(1)} tokens)\n')

    sft_path = './minimind_data/sft_t2t_mini.jsonl'

    for name in ['WDLM-Neural+gen', 'WDLM-Real', 'OpenASH_V2', 'Transformer']:
        print(f'--- {name} ---')
        model = load_model(name)
        p = count(model)

        # Perplexity
        ppl = calc_ppl(model, sft_path, 512, n_samples=100)
        print(f'  PPL: {ppl:.2f}')

        # Generation quality (just generate, human judges)
        gen_ids = generate(model, prompt_ids, max_new=30, temp=0.8)
        gen_text = voc.decode(gen_ids[0].tolist())[:200]
        print(f'  Gen: {gen_text[:120]}...')

        # State mode speed (not for Transformer)
        if name != 'Transformer':
            t_state, _ = gen_state_speed(model, prompt_ids, 64)
            print(f'  State gen: {t_state:.3f}s ({64/t_state:.0f} tok/s)')

        # Naive generation speed
        t_naive, _ = gen_naive_speed(model, prompt_ids, 64)
        total_tokens = sum(prompt_ids.size(1) + i for i in range(64))
        print(f'  Naive gen: {t_naive:.3f}s ({total_tokens/t_naive:.0f} tok/s)')

        results[name] = {'ppl': ppl, 'params': p}
        torch.cuda.empty_cache()
        print()

    # Summary table
    print(f'{"="*70}')
    print(f'{"Model":>15s} {"Params":>10s} {"PPL":>8s} {"State":>8s}')
    for name, r in results.items():
        print(f'{name:>15s} {r["params"]:>10,} {r["ppl"]:>8.2f}', end='')
        if name != 'Transformer':
            t_s, _ = gen_state_speed(load_model(name), prompt_ids, 32)
            print(f' {32/t_s:>7.0f} t/s', end='')
        print()
