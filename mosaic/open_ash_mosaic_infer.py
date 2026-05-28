import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if sys.stdout.encoding != "utf-8":
    sys.stdout = __import__("io").TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if sys.stderr.encoding != "utf-8":
    sys.stderr = __import__("io").TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import json
import torch
from open_ash_mosaic import OpenASHMoSAIC
from open_ash_voc import OpenASHVoc
from config import agent_voc_path


WEIGHT_DIR = r"F:\OpenASH\out_mosaic"
HIDDEN_SIZE = 768
NUM_LAYERS = 12
NUM_HEADS = 8
NUM_ENCODER_LAYERS = 6
MAX_SEQ_LEN = 8192
NUM_EXPERT_LAYERS = NUM_LAYERS - NUM_ENCODER_LAYERS


def _sp(tokenizer):
    return {
        "pad": tokenizer.token_to_id.get("<|pad|>"),
        "im_start": tokenizer.token_to_id.get("<|im_start|>"),
        "im_end": tokenizer.token_to_id.get("<|im_end|>"),
        "think_s": tokenizer.token_to_id.get("<|think|>"),
        "think_e": tokenizer.token_to_id.get("<|end_think|>"),
        "user": tokenizer.token_to_id.get("194"),
        "agent": tokenizer.token_to_id.get("<|agent|>"),
        "system": tokenizer.token_to_id.get("195"),
    }


def sample_next_token(logits, generated_ids, temperature=0.5, top_k=30, top_p=0.85,
                      repetition_penalty=1.35, repetition_window=64):
    logits = logits.clone()
    if generated_ids and repetition_penalty > 1.0:
        recent = set(generated_ids[-min(repetition_window, len(generated_ids)):])
        for tid in recent:
            if tid < logits.size(0):
                if logits[tid] > 0:
                    logits[tid] /= repetition_penalty
                else:
                    logits[tid] *= repetition_penalty
    if temperature < 1e-6:
        return logits.argmax().item()
    logits = logits / temperature
    if torch.isnan(logits).any() or torch.isinf(logits).any():
        logits = torch.nan_to_num(logits, nan=0.0, posinf=1e8, neginf=-1e8)
    if top_k is not None and top_k > 0:
        top_k_val = min(top_k, logits.size(-1))
        topk_vals, topk_idx = torch.topk(logits, top_k_val)
        filtered = torch.full_like(logits, float("-inf"))
        filtered.scatter_(0, topk_idx, topk_vals)
        logits = filtered
    if top_p is not None and 0 < top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        cumulative_probs = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
        remove_mask = cumulative_probs > top_p
        remove_mask[1:] = remove_mask[:-1].clone()
        remove_mask[0] = False
        sorted_logits[remove_mask] = float("-inf")
        logits.scatter_(0, sorted_idx, sorted_logits)
    probs = torch.softmax(logits, dim=-1)
    prob_sum = probs.sum()
    if prob_sum <= 0:
        probs = torch.ones_like(probs) / probs.numel()
    else:
        probs = probs / prob_sum
    return torch.multinomial(probs.cpu(), num_samples=1).item()


def build_user_prompt(tokenizer, user_text, system_text=None):
    sp = _sp(tokenizer)
    ids = []
    if system_text:
        ids += [sp["im_start"], sp["system"]]
        ids += tokenizer.encode(system_text)
        ids += [sp["im_end"]]
    ids += [sp["im_start"], sp["user"]]
    ids += tokenizer.encode(user_text)
    ids += [sp["im_end"]]
    ids += [sp["im_start"], sp["agent"]]
    return ids


def generate(model, tokenizer, prompt_ids, expert_id="base",
             max_new_tokens=512, temperature=0.5, top_k=30, top_p=0.85,
             repetition_penalty=1.35):
    device = next(model.parameters()).device
    sp = _sp(tokenizer)
    stop_ids = {sp["im_end"], sp["pad"]}

    input_tensor = torch.tensor([prompt_ids], dtype=torch.long).to(device)
    if input_tensor.size(1) > MAX_SEQ_LEN:
        input_tensor = input_tensor[:, -MAX_SEQ_LEN:]

    new_ids = []
    model.eval()
    with torch.no_grad():
        state = None
        input_chunk = input_tensor
        for step in range(max_new_tokens):
            if input_chunk.size(1) > MAX_SEQ_LEN:
                break
            outputs, state = model(input_chunk, state=state, expert_id=expert_id)
            logits = outputs[0, -1, :]
            next_id = sample_next_token(logits, new_ids, temperature, top_k, top_p, repetition_penalty)
            if next_id in stop_ids:
                break
            new_ids.append(next_id)
            input_chunk = torch.tensor([[next_id]], dtype=torch.long, device=device)
            state = OpenASHMoSAIC.detach_state(state)
    return new_ids


def auto_route(model, tokenizer, prompt_ids):
    device = next(model.parameters()).device
    input_tensor = torch.tensor([prompt_ids], dtype=torch.long).to(device)
    if input_tensor.size(1) > MAX_SEQ_LEN:
        input_tensor = input_tensor[:, -MAX_SEQ_LEN:]

    model.eval()
    with torch.no_grad():
        logits = model.route(input_tensor)
        expert_idx = logits.argmax(dim=-1).item()

    idx_to_expert = {i: eid for i, eid in enumerate(sorted(model.experts.keys()))}
    chosen = idx_to_expert.get(expert_idx, "base")
    return chosen


def interactive_chat(model, tokenizer, expert_id=None,
                     temperature=0.5, top_k=30, top_p=0.85,
                     repetition_penalty=1.35, max_new_tokens=512):
    sp = _sp(tokenizer)
    messages = []

    available = list(model.experts.keys())
    print("\n" + "=" * 60)
    print("OpenASH MoSAIC Chat")
    print(f"Available experts: {available}")
    if expert_id == "auto":
        print("Mode: AUTO ROUTING")
    elif expert_id:
        print(f"Mode: Manual expert = '{expert_id}'")
    print("Commands: 'quit' exit | 'clear' reset | 'expert <name>' switch")
    print("=" * 60)

    while True:
        try:
            user_input = input("\nUser: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not user_input:
            continue
        if user_input.lower() == "quit":
            break
        if user_input.lower() == "clear":
            messages = []
            print("[History cleared]")
            continue
        if user_input.lower().startswith("expert "):
            new_expert = user_input.split(" ", 1)[1].strip()
            if new_expert in model.experts:
                expert_id = new_expert
                print(f"[Switched to expert '{expert_id}']")
            else:
                print(f"[Unknown expert. Available: {available}]")
            continue

        messages.append({"role": "user", "content": user_input})
        prompt_ids = build_user_prompt(tokenizer, user_input)

        if expert_id == "auto":
            chosen = auto_route(model, tokenizer, prompt_ids)
            print(f"  [Router -> Expert '{chosen}']")
        else:
            chosen = expert_id

        new_ids = generate(model, tokenizer, prompt_ids, expert_id=chosen,
                           max_new_tokens=max_new_tokens, temperature=temperature,
                           top_k=top_k, top_p=top_p, repetition_penalty=repetition_penalty)
        response_text = tokenizer.decode(new_ids)
        print(f"Assistant ({chosen}): {response_text}")

        messages.append({"role": "assistant", "content": response_text})


def main():
    import argparse
    parser = argparse.ArgumentParser(description="OpenASH MoSAIC Inference")
    parser.add_argument("--weight", type=str, default=None, help="MoSAIC full model weight path")
    parser.add_argument("--pretrained_weight", type=str, default=None,
                        help="Original monolithic weight (auto-convert)")
    parser.add_argument("--expert", type=str, default="auto",
                        help="Expert to use (name, 'auto' for routing, 'list' to see available)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--top_k", type=int, default=30)
    parser.add_argument("--top_p", type=float, default=0.85)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    args = parser.parse_args()

    print("=" * 60)
    print("  OpenASH MoSAIC Inference")
    print("=" * 60)

    print("\n[1/4] Loading vocabulary...")
    voc = OpenASHVoc(agent_voc_path=agent_voc_path)
    voc_size = len(voc.token_to_id) + 1
    print(f"  Voc size: {voc_size}")

    print("\n[2/4] Building MoSAIC model...")
    model = OpenASHMoSAIC(
        voc_size=voc_size, hidden_size=HIDDEN_SIZE, num_heads=NUM_HEADS,
        num_encoder_layers=NUM_ENCODER_LAYERS, num_expert_layers=NUM_EXPERT_LAYERS,
        model_flag="infer",
    )

    if args.weight and os.path.exists(args.weight):
        print(f"  Loading MoSAIC weight: {args.weight}")
        model.load_state_dict(torch.load(args.weight, map_location="cpu", weights_only=False), strict=False)
    elif args.pretrained_weight and os.path.exists(args.pretrained_weight):
        print(f"  Converting from monolithic weight: {args.pretrained_weight}")
        sd = torch.load(args.pretrained_weight, map_location="cpu", weights_only=False)
        model.load_from_pretrained(sd)
        del sd
    else:
        default_paths = [
            os.path.join(WEIGHT_DIR, f"mosaic_base_{HIDDEN_SIZE}_{NUM_LAYERS}.pth"),
            os.path.join(WEIGHT_DIR, f"full_sft_{HIDDEN_SIZE}_{NUM_LAYERS}.pth"),
            r"F:\OpenASH\out\full_sft_768_12.pth",
        ]
        loaded = False
        for p in default_paths:
            if os.path.exists(p):
                print(f"  Auto-loading: {p}")
                sd = torch.load(p, map_location="cpu", weights_only=False)
                if "experts.base.layers.0.self_attention_linear.combined.weight" in sd if isinstance(sd, dict) else False:
                    model.load_state_dict(sd, strict=False)
                else:
                    model.load_from_pretrained(sd)
                del sd
                loaded = True
                break
        if not loaded:
            print("  WARNING: No weight found, using random init")

    model.to(args.device)
    model.eval()

    info = model.expert_info()
    total_params = info["encoder_params"] + sum(e["params"] for e in info["experts"].values()) + info["router_params"]
    print(f"  Parameters: {total_params:,}")
    print(f"  Experts: {list(model.experts.keys())}")

    if args.expert == "list":
        print("\nAvailable experts:")
        for eid, einfo in info["experts"].items():
            print(f"  '{eid}': {einfo['params']:,} params")
        return

    print(f"\n[3/4] Running test inference...")
    test_prompts = [
        "你好，请介绍一下你自己。",
        "什么是人工智能？",
    ]
    for prompt in test_prompts:
        prompt_ids = build_user_prompt(voc, prompt)
        if args.expert == "auto":
            chosen = auto_route(model, voc, prompt_ids)
        else:
            chosen = args.expert
        new_ids = generate(model, voc, prompt_ids, expert_id=chosen,
                           max_new_tokens=args.max_new_tokens,
                           temperature=args.temperature, top_k=args.top_k, top_p=args.top_p)
        response = voc.decode(new_ids)
        print(f"\n  Q: {prompt}")
        print(f"  A ({chosen}): {response[:200]}")

    print(f"\n[4/4] Entering interactive chat...")
    interactive_chat(model, voc, expert_id=args.expert,
                     temperature=args.temperature, top_k=args.top_k, top_p=args.top_p,
                     max_new_tokens=args.max_new_tokens)


if __name__ == "__main__":
    main()
