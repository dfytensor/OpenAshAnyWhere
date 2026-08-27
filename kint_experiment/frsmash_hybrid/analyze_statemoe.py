"""分析 State-MoE 为什么没用: 路由分布/专家负载/状态范数/输出贡献/vanilla 对比."""
import torch, torch.nn.functional as F, math, os, sys
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
DEV='cuda'; OUT=os.path.dirname(os.path.abspath(__file__))
CACHE=os.environ.get('KINT_CACHE', r'F:\OpenASH2605\train_60m\cache\pt_cache_openash_512_openash.pt')
VOCAB=23005
sys.path.insert(0, r'F:\rwkv\frsmash_v36'); sys.path.insert(0, OUT)
from frsmash_v36 import FRSMASHv36
from frsmash_statemoe import FRSMASHStateMoE

class DS(Dataset):
    def __init__(s,d,se): s.d,s.se=d,se
    def __len__(s): return len(s.d)
    def __getitem__(s,i): return s.d[i][:s.se+1]
    @staticmethod
    def collate(it): p=pad_sequence(it,batch_first=True,padding_value=0); return p[:,:-1],p[:,1:]

def fwd(model,x):
    o=model(x); return o[0] if isinstance(o,tuple) else o

def train_brief(model, steps=500):
    data=torch.load(CACHE,weights_only=False); tr=data[3000:300000]
    opt=torch.optim.AdamW(model.parameters(),lr=5e-4,weight_decay=0.01,betas=(0.9,0.95))
    sc=torch.amp.GradScaler()
    dl=DataLoader(DS(tr,512),batch_size=32,shuffle=True,collate_fn=DS.collate,drop_last=True)
    for st in range(steps):
        x,y=next(iter(dl)); x=x.clamp(0,VOCAB-1).to(DEV); y=y.clamp(0,VOCAB-1).to(DEV)
        with torch.amp.autocast('cuda',dtype=torch.bfloat16):
            o=model(x); o,a=(o if isinstance(o,tuple) else (o,torch.tensor(0.,device=DEV)))
            loss=F.cross_entropy(o.reshape(-1,VOCAB),y.reshape(-1),ignore_index=0)+0.01*a
        opt.zero_grad(set_to_none=True); sc.scale(loss).backward(); sc.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); sc.step(opt); sc.update()
    print(f'trained {steps} steps',flush=True)

def main():
    data=torch.load(CACHE,weights_only=False); val=data[:64]
    vl=DataLoader(DS(val,512),batch_size=8,shuffle=False,collate_fn=DS.collate)
    x,y=next(iter(vl)); x=x.clamp(0,VOCAB-1).to(DEV)

    print('=== 1. StateMoE 路由/状态分析(500步训练后) ===')
    m=FRSMASHStateMoE(VOCAB,512,8,8,4,n_state_experts=10).to(DEV); train_brief(m,500)
    m.eval()
    with torch.no_grad():
        with torch.amp.autocast('cuda',dtype=torch.bfloat16): _=fwd(m,x)
    d=m.slow_cell.diag
    N=m.slow_cell.N
    print(f'  路由熵 = {d["route_ent"]:.3f} / log({N})={math.log(N):.3f}  (越接近 log N = 越均衡; 低=坍缩)')
    load=d['load']; print(f'  各专家 token 负载:')
    for i in range(N): print(f'    专家{i}: {load[i].item()*100:5.1f}%   状态范数={d["state_norm"][i].item():.2f}   输出贡献={d["out_norm"][i].item():.2f}')
    print(f'  负载熵(均匀={math.log(N):.3f}): {-(load.clamp(1e-9)*load.clamp(1e-9).log()).sum().item():.3f}')
    sn=d['state_norm']; print(f'  状态范数: 均值={sn.mean():.2f} 标准差={sn.std():.2f}  (std小=状态冗余/雷同)')

    print('\n=== 2. vanilla FRSMASH 的 SlowMemory 状态(对照) ===')
    mv=FRSMASHv36(VOCAB,512,8,8,4).to(DEV); train_brief(mv,500); mv.eval()
    with torch.no_grad():
        x_emb=mv.em(x)+mv.pe[:x.size(1)]
        inp=mv.mem_input_proj(x_emb)
        H,h=mv.slow_cell(inp, torch.zeros(x.size(0),512,device=DEV,dtype=x_emb.dtype))
        print(f'  vanilla SlowMemory 状态范数 = {h.float().norm().item():.2f}  (单一状态, 整合全序列)')
        print(f'  vanilla SlowMemory 输出范数 = {H.float().norm().item():.2f}')

    print('\n=== 3. 诊断结论 ===')
    sn=d['state_norm']; 
    cv = (sn.std()/sn.mean()).item()
    print(f'  StateMoE 状态范数变异系数 CV={cv:.3f}  ({"雷同/冗余" if cv<0.3 else "有分化"})')
    print(f'  路由熵 {d["route_ent"]:.2f}/{math.log(N):.2f} = {"均衡" if d["route_ent"]>math.log(N)*0.7 else "坍缩"}')
    # 单专家状态 vs vanilla 状态
    print(f'  StateMoE 单专家状态范数均值={sn.mean():.1f}  vs vanilla 单一状态范数={h.float().norm().item():.1f}')
    print(f'  => 每专家状态{"远小于" if sn.mean()<h.float().norm().item()*0.5 else "接近"} vanilla 全状态 => {"欠激励(只整合1/N token)" if sn.mean()<h.float().norm().item()*0.5 else "正常"}')

if __name__=='__main__': main()
