"""架构效率基准: ppl 打平时, 比 算力/速度/长程依赖.
A. 吞吐 (train fwd+bwd, infer fwd) tok/s + 峰值显存 —— 5 架构 @ seq512
B. 序列长度 scaling: softmax vs GLA 的 前向耗时 + 峰值显存 @ 512/1024/2048/4096 (O(T²) vs O(T))
C. 长上下文质量: GLA vs softmax 训练@512, 评估 ppl @ 512/1024/2048 (更长上下文能否降 loss)
"""
import torch, torch.nn.functional as F, math, os, sys, time, csv, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from arch_compare import LM, VOCAB
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

DEV = 'cuda'; OUT = os.path.dirname(os.path.abspath(__file__))
CACHE = os.environ.get('KINT_CACHE', r'F:\OpenASH2605\train_60m\cache\pt_cache_openash_512_openash.pt')
ARCH={'vanilla':('softmax','dense'),'sgr':('softmax','sgr'),'moe':('softmax','moe'),
      'gla':('gla','dense'),'gla_sgr':('gla','sgr')}


def build(arch, max_len):
    attn, ffn = ARCH[arch]
    m = LM(VOCAB, d=256, h=8, L=6, attn_t=attn, ffn_t=ffn, d_ffn=1024, max_len=max_len+8).to(DEV)
    return m, attn, ffn

def flush(): torch.cuda.synchronize()
def peak_mb(): return torch.cuda.max_memory_allocated()/1024/1024

# ============ A. 吞吐 ============
def bench_speed(arch, seq=512, batch=32, n_iter=20):
    m,_,_ = build(arch, seq); m.train()
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    x = torch.randint(0, VOCAB, (batch, seq), device=DEV)
    # warmup
    for _ in range(3):
        lo,_=m(x); F.cross_entropy(lo.reshape(-1,VOCAB),x.reshape(-1)).backward(); opt.zero_grad()
    flush(); torch.cuda.reset_peak_memory_stats(); flush()
    t0=time.time()
    for _ in range(n_iter):
        lo,_=m(x); loss=F.cross_entropy(lo.reshape(-1,VOCAB),x.reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
    flush(); dt=time.time()-t0
    train_tps = batch*seq*n_iter/dt
    mem=peak_mb()
    # infer
    m.eval()
    for _ in range(3):
        with torch.no_grad(): m(x)
    flush(); torch.cuda.reset_peak_memory_stats(); flush(); t0=time.time()
    with torch.no_grad():
        for _ in range(n_iter): m(x)
    flush(); dti=time.time()-t0
    inf_tps = batch*seq*n_iter/dti
    nparam=sum(p.numel() for p in m.parameters())
    print(f'{arch:8} params={nparam/1e6:.1f}M  train={train_tps/1e3:.1f}k inf={inf_tps/1e3:.1f}k tok/s  mem={mem:.0f}MB',flush=True)
    return arch,nparam,train_tps,inf_tps,mem

# ============ B. 序列长度 scaling ============
def bench_scaling(arch, seq, batch=8):
    try:
        m,_,_=build(arch, seq); m.eval()
        x=torch.randint(0,VOCAB,(batch,seq),device=DEV)
        for _ in range(2):
            with torch.no_grad(): m(x)
        flush(); torch.cuda.reset_peak_memory_stats(); flush(); t0=time.time()
        with torch.no_grad():
            for _ in range(5): m(x)
        flush(); dt=(time.time()-t0)/5; mem=peak_mb()
        del m,x; torch.cuda.empty_cache()
        return dt, mem, None
    except Exception as e:
        return None, None, str(e)[:40]

# ============ C. 长上下文质量 ============
class DS(Dataset):
    def __init__(s,d,se): s.d,s.se=d,se
    def __len__(s): return len(s.d)
    def __getitem__(s,i): return s.d[i][:s.se+1]
    @staticmethod
    def collate(it):
        p=pad_sequence(it,batch_first=True,padding_value=0); return p[:,:-1],p[:,1:]

def train_and_longeval(arch, train_seq=512, steps=800, eval_seqs=(512,1024,2048)):
    torch.manual_seed(0)
    raw=torch.load(CACHE,weights_only=False); val_raw,tr_raw=raw[:2000],raw[2000:60000]
    # 拼成长流 (真实连续文本), 再切块 —— 这样更长 eval 真的用更多上下文
    tr_flat=torch.cat([t[t!=0] for t in tr_raw]); val_flat=torch.cat([t[t!=0] for t in val_raw])
    maxL=max(eval_seqs)+4
    def chunks(flat,L):  # 非重叠定长块
        n=(flat.size(0)-1)//L; return [flat[i*L:i*L+L+1] for i in range(n)]
    m,attn,ffn=build(arch, max(eval_seqs)); m.train()
    opt=torch.optim.AdamW(m.parameters(),lr=3e-4,weight_decay=0.1)
    warm=int(0.05*steps)
    tr_chunks=chunks(tr_flat,train_seq)
    def collate(it):
        p=pad_sequence(it,batch_first=True,padding_value=0); return p[:,:-1],p[:,1:]
    from torch.utils.data import DataLoader
    tr=DataLoader(tr_chunks,batch_size=32,shuffle=True,num_workers=0,collate_fn=collate,drop_last=True)
    for st in range(1,steps+1):
        for g in opt.param_groups: g['lr']=3e-4*min(1.0,st/warm)
        x,y=next(iter(tr)); x,y=x.to(DEV),y.to(DEV)
        lo,_=m(x); loss=F.cross_entropy(lo.reshape(-1,VOCAB),y.reshape(-1),ignore_index=0)
        opt.zero_grad(); loss.backward(); opt.step()
    m.eval(); res={}
    for es in eval_seqs:
        vl=DataLoader(chunks(val_flat,es),batch_size=16,shuffle=False,num_workers=0,collate_fn=collate)
        t=0.0;c=0
        try:
            with torch.no_grad():
                for x,y in vl:
                    x,y=x.to(DEV),y.to(DEV); lo,_=m(x)
                    l=F.cross_entropy(lo.reshape(-1,VOCAB),y.reshape(-1),ignore_index=0,reduction='sum'); t+=float(l); c+=int((y!=0).sum())
            res[es]=math.exp(t/c)
        except Exception as e:
            res[es]=f'OOM'
        print(f'   {arch} eval_seq={es}: ppl={res[es] if isinstance(res[es],str) else round(res[es],2)}',flush=True)
    return res

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--part',default='all')
    a=ap.parse_args()
    if a.part in ('all','speed'):
        print('=== A. 吞吐 (seq=512, bs=32) ===',flush=True)
        for arch in ['vanilla','sgr','gla','gla_sgr','moe']:
            bench_speed(arch)
    if a.part in ('all','scaling'):
        print('\n=== B. 序列长度 scaling (前向, bs=8) ===',flush=True)
        print(f'{"arch":8} '+' '.join(f'{s:>16}' for s in ['512','1024','2048','4096']))
        for arch in ['vanilla','gla']:
            row=[arch]
            for s in [512,1024,2048,4096]:
                dt,mem,err=bench_scaling(arch,s)
                if err: row.append(f'OOM')
                else: row.append(f'{dt*1000:.0f}ms/{mem:.0f}MB')
            print(' '.join(f'{c:>16}' for c in row),flush=True)
    if a.part in ('all','longctx'):
        print('\n=== C. 长上下文质量 (训练@512, 评估@多长度) ===',flush=True)
        for arch in ['gla','vanilla']:
            train_and_longeval(arch)
