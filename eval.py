"""
FRSM Evaluation & Generation Script
验证模型效果：计算困惑度 + 交互式对话生成。
"""
import os
import sys
import math
import argparse

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from frsm.config import FRSMConfig
from frsm.model import FractalRecursiveStateMachine
from frsm.dataset import create_dataloaders
from config import agent_voc_path
from open_ash_voc import OpenASHVoc


@torch.no_grad()
def evaluate_perplexity(model, loader, device, vs, max_batches=20):
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    for i, (x, t) in enumerate(loader):
        if i >= max_batches:
            break
        x = x.to(device)
        t = t.to(device)
        logits = model(x)
        loss = F.cross_entropy(
            logits.reshape(-1, vs), t.reshape(-1),
            ignore_index=0, reduction='sum'
        )
        non_pad = (t != 0).sum().item()
        total_loss += loss.item()
        total_tokens += non_pad

    avg_loss = total_loss / max(1, total_tokens)
    ppl = math.exp(avg_loss) if avg_loss < 20 else float('inf')
    return avg_loss, ppl


@torch.no_grad()
def generate_response(model, voc, prompt, max_new_tokens=128, temperature=0.8, device='cuda'):
    model.eval()
    input_ids = voc.encode(prompt)
    if len(input_ids) == 0:
        return ""

    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

    h = None
    generated = list(input_ids)

    for _ in range(max_new_tokens):
        if h is None:
            logits_seq, h, _ = model(input_tensor, return_state=True, compute_critical_loss=False)
            logits = logits_seq[:, -1, :]
        else:
            last_token = torch.tensor([[generated[-1]]], dtype=torch.long, device=device)
            logits, h = model.generate_step(last_token, h)

        logits = logits / temperature
        probs = F.softmax(logits, dim=-1)

        top_k = min(50, probs.size(-1))
        top_probs, top_indices = torch.topk(probs, top_k, dim=-1)
        top_probs = top_probs / top_probs.sum(dim=-1, keepdim=True)

        next_token = torch.multinomial(top_probs, num_samples=1)
        next_token_id = top_indices[0, next_token[0, 0]].item()

        im_end = voc.token_to_id.get('<|im_end|>')
        if next_token_id == im_end:
            break
        if next_token_id == 0:
            break

        generated.append(next_token_id)

    response = voc.decode(generated[len(input_ids):])
    return response


def interactive_chat(model, voc, device):
    print("\n" + "=" * 60)
    print("FRSM Interactive Chat")
    print("Type 'exit' to quit, 'reset' to clear context")
    print("=" * 60)
    print(f"Model: d_model={model.d_model}, num_scales={model.num_scales}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    while True:
        try:
            user_input = input("\n用户: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if user_input.lower() in ('exit', 'quit'):
            print("Goodbye!")
            break
        if user_input.lower() == 'reset':
            print("Context cleared.")
            continue
        if not user_input:
            continue

        prompt = f"<|im_start|><|user|>{user_input}<|im_end|><|im_start|><|agent|>"
        response = generate_response(model, voc, prompt, max_new_tokens=200, temperature=0.8, device=device)
        print(f"助手: {response}")


def main():
    parser = argparse.ArgumentParser(description="FRSM Evaluation")
    parser.add_argument("--ckpt", type=str, required=True, help="Model checkpoint path")
    parser.add_argument("--mode", type=str, default="chat", choices=["chat", "ppl", "both"],
                        help="Evaluation mode")
    parser.add_argument("--max_eval_batches", type=int, default=20, help="Max batches for PPL eval")
    args = parser.parse_args()

    ckpt = torch.load(args.ckpt, map_location='cpu')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    voc = OpenASHVoc(agent_voc_path=agent_voc_path)
    vs = len(voc.token_to_id) + 1
    print(f"Vocabulary size: {vs}")

    d_model = ckpt.get('config_d_model', 256)
    num_scales = ckpt.get('config_num_scales', 4)

    model = FractalRecursiveStateMachine(
        vocab_size=vs,
        d_model=d_model,
        num_scales=num_scales,
    )
    model.load_state_dict(ckpt['model_state_dict'], strict=False)
    model = model.to(device)
    model.eval()

    if args.mode in ("ppl", "both"):
        eval_config = FRSMConfig(
            d_model=d_model, num_scales=num_scales,
            max_seq_len=256, batch_size=4,
            max_pretrain_lines=2000,
        )
        eval_loader = create_dataloaders(voc, mode='pretrain', config=eval_config)

        print("\nEvaluating perplexity on pretrain data...")
        avg_loss, ppl = evaluate_perplexity(model, eval_loader, device, vs, args.max_eval_batches)
        print(f"  Average loss: {avg_loss:.4f}")
        print(f"  Perplexity: {ppl:.2f}")

    if args.mode in ("chat", "both"):
        interactive_chat(model, voc, device)


if __name__ == "__main__":
    main()
