"""
FRSM V6 Dense MoE — 质量验证
在 minimind_data 上训练少量步数,验证 loss 收敛和生成质量。
"""
import os, sys, time, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch, torch.nn.functional as F
from torch.optim import AdamW
from frsm.config import FRSMConfig
from frsm.dataset import create_dataloaders
from frsm_v6a_dense_moe import FRSM_V6_DenseMoE
from config import agent_voc_path
from open_ash_voc import OpenASHVoc

torch.set_float32_matmul_precision('high')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"设备: {device}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")

# 加载词表
voc = OpenASHVoc(agent_voc_path=agent_voc_path)
vs = len(voc.token_to_id) + 1
print(f"词表大小: {vs}")

# 配置训练参数
class Args: pass
args = Args()
args.d_model = 256
args.num_scales = 4
args.n_experts = 16
args.n_shared = 1
args.batch_size = 16
args.max_seq_len = 256
args.max_steps = 500
args.learning_rate = 5e-4
args.weight_decay = 0.01
args.warmup_steps = 50
args.log_interval = 50
args.eval_interval = 200
args.save_interval = 500
args.data_dir = "../minimind_data"
args.output_dir = "frsm_v6_dense_moe_val"
args.max_lines = 50000
args.num_workers = 0
args.chunk_size = 16

# 创建 Dense MoE 模型
model = FRSM_V6_DenseMoE(
    vocab_size=vs, d_model=args.d_model, num_scales=args.num_scales,
    n_experts=args.n_experts, n_shared=args.n_shared, chunk_size=args.chunk_size
).to(device)
param_count = sum(p.numel() for p in model.parameters())
print(f"Dense MoE 参数: {param_count:,} ({param_count/1e6:.1f}M)")

config = FRSMConfig(
    d_model=args.d_model, num_scales=args.num_scales,
    batch_size=args.batch_size, max_seq_len=args.max_seq_len,
    max_steps=args.max_steps, learning_rate=args.learning_rate,
    max_pretrain_lines=args.max_lines, output_dir=args.output_dir,
    data_dir=args.data_dir,
)
config.n_experts = args.n_experts
config.n_shared = args.n_shared

# 数据加载器
train_loader = create_dataloaders(voc, mode='pretrain', config=config)
print(f"数据集大小: {len(train_loader.dataset)} 样本")

# 优化器
optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay, betas=(0.9, 0.95))
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step:
    min(step/max(1, config.warmup_steps), 1.0) if step < config.warmup_steps
    else max(0, 0.5*(1+math.cos(math.pi*(step-config.warmup_steps)/(config.max_steps-config.warmup_steps)))))

# 训练
model.train()
global_step = 0
loss_accum = 0.0
start_time = time.time()
data_iter = iter(train_loader)

print(f"\n开始训练 {config.max_steps} 步...\n{'='*50}")
for i in range(100):
    # 打印初始 loss
    if i == 0:
        with torch.no_grad():
            try: x0, t0 = next(data_iter)
            except StopIteration: data_iter = iter(train_loader); x0, t0 = next(data_iter)
            x0, t0 = x0.to(device), t0.to(device)
            logits = model(x0, return_state=False)
            init_loss = F.cross_entropy(logits.reshape(-1, vs), t0.reshape(-1), ignore_index=0)
            print(f"初始 loss: {init_loss.item():.4f} (ppl: {math.exp(init_loss.item()):.2f})")

while global_step < config.max_steps:
    try: x, t = next(data_iter)
    except StopIteration: data_iter = iter(train_loader); x, t = next(data_iter)
    x, t = x.to(device), t.to(device)

    logits = model(x, return_state=False)
    loss = F.cross_entropy(logits.reshape(-1, vs), t.reshape(-1), ignore_index=0)
    total = loss + 0.01 * model.aux_loss

    if torch.isnan(loss) or torch.isinf(loss):
        print(f"WARNING: NaN/Inf loss at step {global_step}, skipping"); continue

    optimizer.zero_grad()
    total.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    scheduler.step()

    global_step += 1
    loss_accum += loss.item()

    if global_step % config.log_interval == 0 or global_step == 1:
        avg_loss = loss_accum / config.log_interval
        ppl = math.exp(min(avg_loss, 20))
        lr = optimizer.param_groups[0]['lr']
        elapsed = time.time() - start_time
        tok_per_sec = global_step * x.size(1) * x.size(0) / elapsed if elapsed > 0 else 0
        print(f"step {global_step:5d}/{config.max_steps} | loss: {avg_loss:.4f} | ppl: {ppl:.2f} | aux: {model.aux_loss.item():.4f} | lr: {lr:.2e} | {tok_per_sec:.0f} tok/s")
        loss_accum = 0.0

# 最终 loss
final_loss = avg_loss if 'avg_loss' in dir() else 0
print(f"\n{'='*50}")
print(f"训练完成! 最终 loss: {final_loss:.4f} (ppl: {math.exp(min(final_loss,20)):.2f})")

# 简单推理测试
print(f"\n{'='*50}")
print(f"生成测试:")
model.eval()
prompt = "给我讲一个"
prompt_ids = voc.encode(prompt)
input_ids = torch.tensor([prompt_ids], device=device)

with torch.no_grad():
    # 初始状态
    H = torch.zeros(model.n_experts, 1, model.num_scales, model.d_model, device=device)
    Hs = torch.zeros(model.n_shared, 1, model.num_scales, model.d_model, device=device) if model.n_shared > 0 else None

    # 编码 prompt
    all_ids = prompt_ids.copy()
    for token_id in prompt_ids:
        tok = torch.tensor([[token_id]], device=device)
        logits, (H, Hs) = model.generate_step(tok, (H, Hs))

    # 生成新 token
    generated = prompt
    for i in range(50):
        tok = logits.argmax(dim=-1, keepdim=True)
        token_id = tok.item()
        all_ids.append(token_id)
        if token_id == 0: break
        decoded = voc.decode([token_id])
        generated += decoded
        logits, (H, Hs) = model.generate_step(tok, (H, Hs))

print(f"\nPrompt: {prompt}")
print(f"生成: {generated[:200]}")
