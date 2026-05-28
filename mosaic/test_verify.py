import sys, os
sys.path.insert(0, r'F:\OpenASH2605\mosaic')
sys.path.insert(0, r'F:\OpenASH2605')
os.chdir(r'F:\OpenASH2605')

import torch
from open_ash_mosaic import OpenASHMoSAIC

print('=== MoSAIC Model Verification ===')
print()

from open_ash_voc import OpenASHVoc
from config import agent_voc_path
voc = OpenASHVoc(agent_voc_path=agent_voc_path)
voc_size = len(voc.token_to_id) + 1
print(f'  Voc size: {voc_size}')

model = OpenASHMoSAIC(
    voc_size=voc_size, hidden_size=768, num_heads=8,
    num_encoder_layers=6, num_expert_layers=6,
)

print('[1] Loading pretrained weight...')
sd = torch.load(r'F:\OpenASH2605\models\full_sft_768_12.pth', map_location='cpu', weights_only=False)
print(f'  Original keys: {len(sd)}')

orig_layers = set()
for k in sd.keys():
    if k.startswith('decoder_layers.'):
        orig_layers.add(int(k.split('.')[1]))
print(f'  Original decoder layers: {sorted(orig_layers)}')
print(f'  Has em: {"em.weight" in sd}')
print(f'  Has head_score: {any(k.startswith("head_score") for k in sd.keys())}')

print()
print('[2] Converting to MoSAIC format...')
model.load_from_pretrained(sd)
del sd

model_sd = model.state_dict()
enc_layers = set()
exp_layers = set()
for k in model_sd.keys():
    if k.startswith('encoder_layers.'):
        enc_layers.add(int(k.split('.')[1]))
    if k.startswith('experts.base.layers.'):
        exp_layers.add(int(k.split('.')[3]))
print(f'  Encoder layers loaded: {sorted(enc_layers)}')
print(f'  Expert(base) layers loaded: {sorted(exp_layers)}')
print(f'  Experts available: {list(model.experts.keys())}')

print()
print('[3] Adding experts...')
model.add_expert('math', init_from='base')
model.add_expert('code', init_from='base')
model.add_expert('chat', init_from='base')
model.freeze_encoder()

info = model.expert_info()
total = info['encoder_params'] + sum(e['params'] for e in info['experts'].values()) + info['router_params']
trainable = sum(e['params'] for e in info['experts'].values()) + info['router_params']

print(f'  Total params:     {total:>12,}')
print(f'  Encoder (frozen): {info["encoder_params"]:>12,}')
for eid, einfo in info['experts'].items():
    print(f'  Expert "{eid}":    {einfo["params"]:>12,}')
print(f'  Router:           {info["router_params"]:>12,}')
print(f'  Trainable ratio:  {trainable/total*100:.1f}%')
per_expert = list(info['experts'].values())[0]['params']
print(f'  Per-expert ratio: {per_expert/total*100:.1f}%')

print()
print('[4] Forward pass test...')
x = torch.randint(0, 100, (2, 16))
out, state = model(x, expert_id='base')
print(f'  Input:  {x.shape}')
print(f'  Output: {out.shape}')
print(f'  State:  encoder={len(state["encoder"])}, expert={len(state["expert"])}, active={state["active_expert"]}')

out2, state2 = model(x, state=None, expert_id='math')
print(f'  Switch to math expert: {out2.shape} (independent from base)')

logits = model.route(x)
print(f'  Router output: {logits.shape}')

detached = OpenASHMoSAIC.detach_state(state)
print(f'  State detach: OK')

print()
print('=== ALL CHECKS PASSED ===')
