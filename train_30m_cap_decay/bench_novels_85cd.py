import os, sys, math, torch, torch.nn.functional as F, time
ROOT = r"F:\OpenASH2605"; BENCH = os.path.join(ROOT, "experiment_openash_vs_wdlm", "bench")
sys.path.insert(0, ROOT); sys.path.insert(0, BENCH); os.chdir(ROOT)
from open_ash import OpenASH; from open_ash_voc import OpenASHVoc; from config import agent_voc_path
DEV="cuda"; CHUNK=64; CAP=150; DECAY=0.97
voc = OpenASHVoc(agent_voc_path=agent_voc_path); vs = len(voc.token_to_id)+1

m30 = OpenASH(vs, hidden_size=432, num_heads=8, num_layers=8, model_flag="train")
m30.load_state_dict(torch.load(os.path.join(ROOT,"train_30m_cap_decay","openash30m_cd_sft_final.pth"), map_location=DEV)["model"])
m30.to(DEV).eval()

m85 = OpenASH(vs, hidden_size=768, num_heads=8, num_layers=12, model_flag="train")
m85.load_state_dict(torch.load(os.path.join(BENCH,"full_sft_768_12.pth"), map_location=DEV))
m85.to(DEV).eval()

for m in [m30,m85]:
    with torch.no_grad(): m(torch.randint(1,100,(1,128),device=DEV), state=None)
torch.cuda.synchronize()

def novel_ppl(model, text, sl, use_cd=False):
    ids = voc.encode(text)
    if len(ids) < sl+10: return None
    ids = ids[:sl]; nl = len(model.decoder_layers)
    x = torch.tensor([ids[:-1]], dtype=torch.long).to(DEV).clamp(0,vs-1)
    t = torch.tensor([ids[1:]], dtype=torch.long).to(DEV).clamp(0,vs-1)
    with torch.no_grad():
        st = [None]*nl; nll=0.0; nt=0
        for c0 in range(0, x.size(1), CHUNK):
            c=x[:,c0:c0+CHUNK]; tc=t[:,c0:c0+CHUNK]
            h = model.em(c)
            for i,la in enumerate(model.decoder_layers):
                h2,s = la(h, st[i]); h=h2+h; st[i]=s
                if use_cd and s is not None:
                    sn=s.norm()
                    if sn>CAP: s=s*(CAP/sn)
                    st[i]=s*DECAY
            lo=model.head_score(h)
            nll+=F.cross_entropy(lo.reshape(-1,lo.size(-1)),tc.reshape(-1),ignore_index=0,reduction="sum").item()
            nt+=(tc!=0).sum().item()
    return math.exp(nll/max(nt,1))

NOVEL_DIR = r"F:\小说\女生小说"
novels = ["傲世九重天-风凌天下.txt","奥术神座-爱潜水的乌贼.txt","百炼成仙-幻雨.txt","八零小富婆-风夜晚晚.txt","霸道总裁宠鲜妻-衣林夕.txt"]
seqs = [512, 1024, 4096, 16384, 65536]

hdr = "  {:>25}  {:>5}  {:>8}  {:>8}  {:>8}  {:>8}  {:>8}  {:>4}".format("Novel","Seq","30M-cd","85M","85M+cd","30Mdeg","85Mdeg","time")
print(hdr)
print("  "+"-"*len(hdr))

for name in novels:
    path = os.path.join(NOVEL_DIR, name)
    if not os.path.exists(path): continue
    with open(path, encoding="utf-8", errors="ignore") as f: text = f.read(200000)
    nt = len(voc.encode(text))
    base30=None; base85=None
    for sl in seqs:
        if nt < sl: continue
        t0=time.time()
        p30 = novel_ppl(m30, text, sl, use_cd=True)
        p85 = novel_ppl(m85, text, sl, use_cd=False)
        p85cd = novel_ppl(m85, text, sl, use_cd=True)
        if base30 is None: base30=p30
        if base85 is None: base85=p85
        d30 = "{:.2f}x".format(p30/base30)
        d85 = "{:.2f}x".format(p85cd/base85)
        lb = "{}K".format(sl//1024) if sl>=1024 else str(sl)
        el = int(time.time()-t0)
        short = name[:25]
        print("  {:>25}  {:>5}  {:>8.1f}  {:>8.1f}  {:>8.1f}  {:>8}  {:>8}  {:>4}s".format(short,lb,p30,p85,p85cd,d30,d85,el))
        sys.stdout.flush()
    print()
print("Done.")
