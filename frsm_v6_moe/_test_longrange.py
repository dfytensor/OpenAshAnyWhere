"""
长期依赖 PPL 稳定性测试 — 按位置分段计算 loss
OpenASH(cummax 不遗忘) vs Slow(内容门控选择性) vs Full(融合)
"""
import torch,torch.nn.functional as F,sys,time,math
sys.path.insert(0,'.');sys.path.insert(0,'..')
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from frsmash import FRSMASH

dev='cuda';V=23005;H=512;L=4;B=32;T=384;steps=200
data=torch.load('../minimind_data/pretrain_cached_30000_384.pt',weights_only=True)
class DS(torch.utils.data.Dataset):
    def __len__(s):return len(data)
    def __getitem__(s,i):d=data[i];return d[:T+1] if len(d)>T+1 else d
    @staticmethod
    def collate_fn(items):p=pad_sequence(items,batch_first=True,padding_value=0);return p[:,:-1],p[:,1:]
loader=DataLoader(DS(),batch_size=B,shuffle=True,collate_fn=DS.collate_fn,drop_last=True)

def train(model,name):
    print(f'  training {name}...')
    m=model.to(dev);m.train();opt=torch.optim.AdamW(m.parameters(),lr=5e-4,weight_decay=0.01)
    it=iter(loader);st=0;t0=time.time()
    while st<steps:
        x,t=next(it,(None,None))
        if x is None:it=iter(loader);x,t=next(it)
        x,t=x.to(dev),t.to(dev);lg=m(x)
        loss=F.cross_entropy(lg.reshape(-1,V),t.reshape(-1),ignore_index=0)
        if torch.isnan(loss):continue
        opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1.0);opt.step();st+=1
    print(f'  {name} done {time.time()-t0:.0f}s')
    return m

def eval_position_loss(model, loader, ranges):
    """按位置分段计算 loss"""
    m=model.to(dev);m.eval()
    results={k:[] for k in ranges}
    total_tokens=0
    with torch.no_grad():
        for _ in range(5):  # 5 batches
            x,t=next(iter(loader))
            x,t=x.to(dev),t.to(dev)
            lg=m(x)  # (B,T,V)
            loss_per_pos=F.cross_entropy(lg.reshape(-1,V),t.reshape(-1),ignore_index=0,reduction='none')
            loss_2d=loss_per_pos.reshape(B,-1)  # (B,T)
            mask=(t!=0).float()
            for name,(lo,hi) in ranges.items():
                pos_mask=mask[:,lo:hi]
                pos_loss=(loss_2d[:,lo:hi]*pos_mask).sum()/(pos_mask.sum()+1e-8)
                results[name].append(pos_loss.item())
    return {k:sum(v)/len(v) for k,v in results.items()}

# ====== 训练3个模型 ======
print('Training models...')
mf=train(FRSMASH(V,H,8,L,K=2),'K=2');torch.cuda.empty_cache()
ma=train(FRSMASH(V,H,8,L,K=999999),'NoSlow(≈ASH only)');torch.cuda.empty_cache()
ms=train(FRSMASH(V,H,8,0,K=1),'NoASH(≈Slow only)');torch.cuda.empty_cache()

# ====== 位置分段 loss 测试 ======
ranges={'near(0-64)':(0,64),'mid(128-192)':(128,192),'far(320-384)':(320,384),'all(0-384)':(0,384)}
print('\n=== 长期依赖 PPL 稳定性 ===')
print(f"{'Model':<20} {'near':>8} {'mid':>8} {'far':>8} {'all':>8} {'far/near':>10} {'stability':>10}")
print('-'*70)
mf=train(FRSMASH(V,H,8,L,K=2),'K=2'); torch.cuda.empty_cache()
ma=train(FRSMASH(V,H,8,L,K=999999),'NoSlow'); torch.cuda.empty_cache()
# eval
for model,tag in [(mf,'K=2(完整)'),(ma,'NoSlow(无记忆)')]:
    r=eval_position_loss(model,loader,ranges)
    n=math.exp(r['near(0-64)']);m=math.exp(r['mid(128-192)']);f=math.exp(r['far(320-384)']);a=math.exp(r['all(0-384)']);fn=f/n;st=1-abs(f-n)/a
    print(f"{tag:<20} {n:>7.1f} {m:>7.1f} {f:>7.1f} {a:>7.1f} {fn:>9.2f}x {st:>9.3f}")
