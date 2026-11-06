"""SFT 后生成质量定性对比：sft_v35 vs sft_tf 同 prompt 生成。"""
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, '.')
import torch
from open_ash_v3 import OpenASH_V3
from baseline_tf import TransformerLM
from open_ash_voc import OpenASHVoc
from config import agent_voc_path

DEV = 'cuda'
voc = OpenASHVoc(agent_voc_path=agent_voc_path)


def load_v35():
    m = OpenASH_V3(23004, 768, 8, 12, max_seq_len=2048, model_flag='train', use_fused=True,
                   use_decay=True, pre_norm=True, ffn_act='silu',
                   decay_init=(0.0001, 0.35), decay_v_std=0.04,
                   kv_channel=True, kv_decay=0.98, kv_norm_keys=True, kv_beta=True).to(DEV)
    m.load_state_dict(torch.load(r'sft_v35\model.pt', map_location='cpu', weights_only=True))
    return m.eval()


def load_tf():
    m = TransformerLM(23004, 768, 8, 11, max_seq_len=2048).to(DEV)
    m.load_state_dict(torch.load(r'sft_tf\model.pt', map_location='cpu', weights_only=True))
    return m.eval()


@torch.no_grad()
def gen(m, prompt, n_new=120, temperature=0.8, top_k=50, style='sft'):
    ids = voc.encode(prompt)
    x = torch.tensor([ids], device=DEV)
    out = x
    gen_ids = []
    # SFT 格式：模型在预训练+SFT 数据里学过 "用户：...助手：..." 文本流；
    # 生成时把 prompt 包装为对话开头，让模型续写助手回答
    if style == 'sft':
        full = '用户：' + prompt + '\n助手：'
        full_ids = voc.encode(full)
        out = torch.tensor([full_ids], device=DEV)
        # prefill
        logits, _ = m(out)
        cur = logits
    else:
        full_ids = ids
        logits, _ = m(out)
        cur = logits
    for i in range(n_new):
        logits = cur[0, -1].float() / temperature
        kk = min(top_k, logits.shape[-1])
        vv, _ = torch.topk(logits, kk)
        logits[logits < vv[-1]] = -float('inf')
        nxt = torch.multinomial(torch.softmax(logits, -1), 1).item()
        gen_ids.append(nxt)
        if nxt == voc.encode('\n')[0] and i > 8:
            break
        cur, _ = m(torch.tensor([[nxt]], device=DEV),
                   pos_offset=len(full_ids) + i)
    return voc.decode(gen_ids)


m35 = load_v35()
mtf = load_tf()
prompts = ['人工智能是什么？', '请介绍一下中国的四大发明。', '写一句关于春天的诗。']
for p in prompts:
    print(f'== {p} ==')
    print('v35:', gen(m35, p, style='sft')[:200])
    print('tf :', gen(mtf, p, style='sft')[:200])
    print()
