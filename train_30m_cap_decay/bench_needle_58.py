import os, sys, math, torch, torch.nn.functional as F, time, random
ROOT = r"F:\OpenASH2605"
BENCH = os.path.join(ROOT, "experiment_openash_vs_wdlm", "bench")
sys.path.insert(0, ROOT); sys.path.insert(0, BENCH); os.chdir(ROOT)
from open_ash import OpenASH; from open_ash_voc import OpenASHVoc; from config import agent_voc_path
from open_ash_infer import _sp
DEV="cuda"; CHUNK=64; CAP=150; DECAY=0.97
voc = OpenASHVoc(agent_voc_path=agent_voc_path); vs = len(voc.token_to_id)+1
sp = _sp(voc)

m58 = OpenASH(vs, hidden_size=640, num_heads=8, num_layers=10, model_flag="train")
m58.load_state_dict(torch.load(os.path.join(BENCH,"openash60m_sft_final.pth"), map_location=DEV)["model"])
m58.to(DEV).eval()
with torch.no_grad(): m58(torch.randint(1,100,(1,128),device=DEV), state=None)
torch.cuda.synchronize()

NEEDLES = [
    ("我的手机密码是8473，请记住这个密码。", "我的手机密码是什么？", "8473"),
    ("今天是2026年6月12日，天气晴朗。", "今天的日期是哪天？", "2026"),
    ("小明家的猫叫橘子，是一只橘色的胖猫。", "小明家的猫叫什么名字？", "橘子"),
    ("这本书的作者是李华，出版社是清华大学出版社。", "这本书的作者是谁？", "李华"),
    ("密码箱的密码是9527，千万不要忘记。", "密码箱的密码是多少？", "9527"),
    ("会议室在三楼302房间，下午两点开会。", "会议室在哪个房间？", "302"),
    ("张三的银行卡号是6225880137123456。", "张三的银行卡号后四位是多少？", "3456"),
    ("钥匙藏在门口花盆下面第三个位置。", "钥匙藏在哪里？", "花盆"),
]

def oa_gen(model, ids, max_new=60, use_cd=False):
    nl = len(model.decoder_layers)
    with torch.no_grad():
        states = [None]*nl; generated = ids
        for _ in range(max_new):
            ctx = generated[:,-768:]
            c = ctx[:,-CHUNK:] if ctx.size(1)>CHUNK else ctx
            h = model.em(c)
            for i,la in enumerate(model.decoder_layers):
                h2,s = la(h, states[i]); h=h2+h; states[i]=s
                if use_cd and s is not None:
                    sn=s.norm()
                    if sn>CAP: s=s*(CAP/sn)
                    states[i]=s*DECAY
            logits = model.head_score(h)[:,-1,:]/0.7
            v,_ = torch.topk(logits,40)
            logits = logits.masked_fill(logits < v[:,[-1]], float("-inf"))
            nt = torch.multinomial(F.softmax(logits,dim=-1),1)
            generated = torch.cat([generated,nt],dim=1)
            if nt.item()==sp["im_end"]: break
    return generated[0].tolist()

novel_path = os.path.join(r"F:\小说\女生小说", "傲世九重天-风凌天下.txt")
with open(novel_path, encoding="utf-8", errors="ignore") as f: novel_text = f.read(1000000)
novel_tokens = voc.encode(novel_text)

for use_cd in [False, True]:
    tag = "58M+cd" if use_cd else "58M-base"
    print("  --- {} ---".format(tag))
    for cl in [512, 2048, 8192]:
        for d in [0, 50, 100]:
            hits = 0
            for trial in range(10):
                needle_stmt, question, answer = random.choice(NEEDLES)
                needle_ids = voc.encode(needle_stmt)
                question_ids = voc.encode(question)
                insert_pos = int(cl*d/100)
                insert_pos = max(0, min(insert_pos, cl-len(needle_ids)))
                before = novel_tokens[:insert_pos]
                after = novel_tokens[insert_pos:cl-len(needle_ids)]
                qa_prefix = [sp["im_start"],sp["user"]]+question_ids+[sp["im_end"],sp["im_start"],sp["agent"]]
                full_ids = before+needle_ids+after+qa_prefix
                if len(full_ids)>cl: full_ids=full_ids[:cl]
                while len(full_ids)<64: full_ids.append(0)
                x = torch.tensor([full_ids],dtype=torch.long,device=DEV).clamp(0,vs-1)
                resp_ids = oa_gen(m58, x, max_new=60, use_cd=use_cd)
                resp_text = voc.decode(resp_ids[len(full_ids):]).strip()[:200].replace("\n"," ")
                hit = 1 if answer in resp_text else 0
                hits += hit
                mark = "HIT" if hit else "miss"
                lb = "{}K@{}%".format(cl//1024 if cl>=1024 else cl, d)
                print("  [{:>10}] {:>8} t{:>2} {}  ans={:<6} resp={:.60}".format(tag,lb,trial+1,mark,answer,resp_text))
                sys.stdout.flush()
            acc = hits/10
            print("  [{:>10}] {:>8} ACC={:.0%}".format(tag, lb, acc))
            print()
    print()
print("Done.")
