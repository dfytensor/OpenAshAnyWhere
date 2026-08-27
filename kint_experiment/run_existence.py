"""存在性检查: 用训练后的 K_int 模型生成文本, 验证"智能态"成立. 仅 pretrain 故只续写."""
import torch, sys, os
sys.path.insert(0, r'F:\rwkv\frsmash_v36')
sys.path.insert(0, r'F:\OpenASH2605')
from frsmash_v36 import FRSMASHv36
import frsmash_v36_infer as inf
from open_ash_voc import OpenASHVoc

VOCAB = 23005
DEV = 'cuda'
HERE = os.path.dirname(os.path.abspath(__file__))


def load_model(cond, hidden=432, heads=8, layers=8):
    m = FRSMASHv36(VOCAB, hidden, heads, layers, n_slots=4).to(DEV)
    ck = torch.load(os.path.join(HERE, f'kint_{cond}_final.pth'), map_location=DEV, weights_only=False)
    m.load_state_dict(ck['model']); m.eval()
    return m


def gen(m, voc, prompt, max_new=80, temperature=0.8, top_k=40, rep_penalty=1.1):
    ids = voc.encode(prompt)
    x = torch.tensor([ids], device=DEV)
    out = inf.generate(m, x, max_new=max_new, top_k=top_k, top_p=1.0,
                       rep_penalty=rep_penalty, temperature=temperature, eos=None)
    return voc.decode(ids + out)


if __name__ == '__main__':
    voc = OpenASHVoc(agent_voc_path=r'F:\OpenASH2605\open_ash_voc_agent.json',
                     flag=False, voc_size=20000, two=200)
    prompts = [
        '春天的景色是',
        '人工智能的未来',
        '请写一首关于月亮的诗。',
    ]
    for cond in ['baseline', 'heated']:
        print(f'\n========== {cond} ==========')
        m = load_model(cond)
        for p in prompts:
            print(f'\n[Prompt] {p}')
            print(gen(m, voc, p, max_new=70))
        del m; torch.cuda.empty_cache()
