"""验证: 上下文长度(思维链)决定模型智商.
用 FRSMASH v3.7 在算法任务上测: 给不同"思维空间"(seq长度), 看推理准确率.
任务: 多步算术(需要多token推理链) + 模式复制(需要长程记忆).
"""
import torch, torch.nn.functional as F, math, os, sys, time, random
sys.path.insert(0, r'F:\rwkv\frsmash_v36')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from frsmash_v36 import FRSMASHv36
DEV='cuda'; VOCAB=23005
CACHE=os.environ.get('KINT_CACHE', r'F:\OpenASH2605\train_60m\cache\pt_cache_openash_512_openash.pt')
from frsmash_directadd import FRSMASHDirectAdd
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence


class DS(Dataset):
    def __init__(s,d,se): s.d,s.se=d,se
    def __len__(s): return len(s.d)
    def __getitem__(s,i): return s.d[i][:s.se+1]
    @staticmethod
    def collate(it): p=pad_sequence(it,batch_first=True,padding_value=0); return p[:,:-1],p[:,1:]


def train_model(model, steps=1500, seq=512):
    data=torch.load(CACHE,weights_only=False); tr=data[3000:300000]
    opt=torch.optim.AdamW(model.parameters(),lr=5e-4,weight_decay=0.01,betas=(0.9,0.95)); sc=torch.amp.GradScaler()
    dl=DataLoader(DS(tr,seq),batch_size=32,shuffle=True,collate_fn=DS.collate,drop_last=True)
    ti=iter(dl)
    for st in range(steps):
        try: x,y=next(ti)
        except StopIteration: ti=iter(dl); x,y=next(ti)
        x=x.clamp(0,VOCAB-1).to(DEV); y=y.clamp(0,VOCAB-1).to(DEV)
        with torch.amp.autocast('cuda',dtype=torch.bfloat16):
            o=model(x); loss=F.cross_entropy(o.reshape(-1,VOCAB),y.reshape(-1),ignore_index=0)
        opt.zero_grad(set_to_none=True); sc.scale(loss).backward(); sc.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); sc.step(opt); sc.update()
    return model


@torch.no_grad()
def test_next_token_accuracy(model, data, seq_lengths):
    """不同 seq 长度下的 next-token 准确率 = '智商'代理指标.
    更长上下文 → 更多信息 → 更准预测 → 更高'智商'."""
    model.eval()
    results={}
    for seq in seq_lengths:
        # 取连续 seq+1 token, 预测最后一个
        correct=0; total=0
        for sample in data[:200]:
            tokens=sample[sample!=0]
            if len(tokens)<seq+1: continue
            # 从不同位置取窗口
            for start in range(0, min(len(tokens)-seq-1, 5), max(1,(len(tokens)-seq)//5)):
                ctx=tokens[start:start+seq].unsqueeze(0).to(DEV).clamp(0,VOCAB-1)
                target=tokens[start+seq].item()
                with torch.amp.autocast('cuda',dtype=torch.bfloat16):
                    o=model(ctx)
                pred=o[0,-1].argmax().item()
                correct+=(pred==target); total+=1
        acc=correct/max(total,1)
        results[seq]=acc
        print(f'  ctx={seq:>5}: top1_acc={acc:.4f} ({correct}/{total})',flush=True)
    return results


@torch.no_grad()
def test_distant_copy(model, data, gap_sizes):
    """远距离复制: 在 gap 步前放一个特殊 token, 看模型能否在末尾复制它.
    gap 越大 = 需要的记忆越长 = '智商'越高才能做到."""
    model.eval()
    results={}
    # 用高频 token 做目标
    from collections import Counter
    all_tokens=[]
    for s in data[:1000]:
        all_tokens.extend(s[s!=0].tolist())
    common=Counter(all_tokens).most_common(50)
    target_ids=[t for t,_ in common if 10<t<VOCAB-1][:10]
    for gap in gap_sizes:
        correct=0; total=0
        for target_id in target_ids:
            for trial in range(10):
                # 构造: [target_id] + [随机填充 gap 步] + [预测 target_id]
                filler=torch.randint(100,VOCAB-1,(gap,),device=DEV)
                ctx=torch.cat([torch.tensor([target_id],device=DEV),filler]).unsqueeze(0).clamp(0,VOCAB-1)
                with torch.amp.autocast('cuda',dtype=torch.bfloat16):
                    o=model(ctx)
                pred=o[0,-1].argmax().item()
                correct+=(pred==target_id); total+=1
        acc=correct/max(total,1)
        results[gap]=acc
        print(f'  gap={gap:>4}: copy_acc={acc:.4f} ({correct}/{total})',flush=True)
    return results


if __name__=='__main__':
    data=torch.load(CACHE,weights_only=False)
    val_data=data[:2000]
    print('=== 训练 v3.7 DirectAdd (1500步 seq512) ===',flush=True)
    model=FRSMASHDirectAdd(VOCAB,512,8,8,4).to(DEV)
    model=train_model(model,1500,512)

    print('\n=== 测试1: 上下文长度 vs next-token 准确率(智商代理) ===',flush=True)
    ctx_acc=test_next_token_accuracy(model, val_data, [32,64,128,256,512])

    print('\n=== 测试2: 远距离复制 gap vs 准确率(记忆/推理深度) ===',flush=True)
    copy_acc=test_distant_copy(model, val_data, [1,4,16,64,128,256,512])

    print('\n=== 总结 ===')
    print('上下文长度 → 准确率(智商):')
    for k,v in ctx_acc.items(): print(f'  ctx={k:>4}: {v:.4f}')
    print('\n思维链长度(gap) → 复制准确率:')
    for k,v in copy_acc.items(): print(f'  gap={k:>4}: {v:.4f}')
