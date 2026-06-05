#!/usr/bin/env python3
import os
import sys
import io
import torch

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

sys.path.insert(0, 'F:/OpenASH2605')
sys.path.insert(0, 'F:/OpenASH2605/wdlm_verification')
os.chdir('F:/OpenASH2605')

from open_ash_voc import OpenASHVoc
from config import agent_voc_path
from wdlm_neural import WaveDynamicsLanguageModel

WEIGHT_DIR = r"F:\OpenASH2605\train_60m"
HIDDEN_DIM = 512
NUM_LAYERS = 10
MAX_SEQ_LEN = 1024


def _sp(tokenizer):
    return {
        "pad": tokenizer.token_to_id.get('<|pad|>', 0),
        "im_start": tokenizer.token_to_id.get('<|im_start|>', 1),
        "im_end": tokenizer.token_to_id.get('<|im_end|>', 2),
        "think_s": tokenizer.token_to_id.get('<|think|>', 3),
        "think_e": tokenizer.token_to_id.get('<|end_think|>', 4),
        "user": 5,
        "agent": tokenizer.token_to_id.get('<|agent|>', 6),
        "system": 7,
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
        filtered = torch.full_like(logits, float('-inf'))
        filtered.scatter_(0, topk_idx, topk_vals)
        logits = filtered

    if top_p is not None and 0 < top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        cumulative_probs = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
        remove_mask = cumulative_probs > top_p
        remove_mask[1:] = remove_mask[:-1].clone()
        remove_mask[0] = False
        sorted_logits[remove_mask] = float('-inf')
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


def build_chat_prompt(tokenizer, messages):
    sp = _sp(tokenizer)
    ids = []
    for msg in messages:
        role = msg["role"]
        if role == "system":
            ids += [sp["im_start"], sp["system"]]
            ids += tokenizer.encode(msg.get("content", ""))
            ids += [sp["im_end"]]
        elif role == "user":
            ids += [sp["im_start"], sp["user"]]
            ids += tokenizer.encode(msg["content"])
            ids += [sp["im_end"]]
        elif role == "assistant":
            ids += [sp["im_start"], sp["agent"]]
            if msg.get("reasoning_content"):
                ids += [sp["think_s"]]
                ids += tokenizer.encode(msg["reasoning_content"])
                ids += [sp["think_e"]]
            if msg.get("content", "") != "":
                ids += tokenizer.encode(msg["content"])
            ids += [sp["im_end"]]
    return ids


def _split_by_special(token_ids, sp):
    stop_ids = {sp["im_end"], sp["pad"]}
    sections = {"thinking": [], "content": []}
    current = "content"
    buf = []

    def flush():
        if buf:
            sections[current].append(list(buf))
            buf.clear()

    for tid in token_ids:
        if tid == sp["think_s"]:
            flush(); current = "thinking"
        elif tid == sp["think_e"]:
            flush(); current = "content"
        elif tid in stop_ids:
            flush(); break
        else:
            buf.append(tid)
    flush()
    return sections


def format_response(tokenizer, token_ids):
    sp = _sp(tokenizer)
    sections = _split_by_special(token_ids, sp)
    result = {}
    for key in ("thinking", "content"):
        all_ids = []
        for chunk in sections[key]:
            all_ids.extend(chunk)
        if all_ids:
            result[key] = tokenizer.decode(all_ids)
    return result


def generate(model, tokenizer, prompt_ids, max_new_tokens=512,
             temperature=0.5, top_k=30, top_p=0.85, repetition_penalty=1.35):
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
            outputs, state = model(input_chunk, state=state)
            logits = outputs[0, -1, :]
            next_id = sample_next_token(
                logits, new_ids,
                temperature=temperature, top_k=top_k, top_p=top_p,
                repetition_penalty=repetition_penalty
            )
            if next_id in stop_ids:
                break
            new_ids.append(next_id)
            input_chunk = torch.tensor([[next_id]], dtype=torch.long, device=device)
            state = [s.detach() for s in state]
    return new_ids


def generate_stream(model, tokenizer, prompt_ids, max_new_tokens=512,
                    temperature=0.5, top_k=30, top_p=0.85, repetition_penalty=1.35):
    device = next(model.parameters()).device
    sp = _sp(tokenizer)
    stop_ids = {sp["im_end"], sp["pad"]}

    input_tensor = torch.tensor([prompt_ids], dtype=torch.long).to(device)
    if input_tensor.size(1) > MAX_SEQ_LEN:
        input_tensor = input_tensor[:, -MAX_SEQ_LEN:]

    new_ids = []
    prev_decoded_len = 0
    model.eval()
    with torch.no_grad():
        state = None
        input_chunk = input_tensor
        for step in range(max_new_tokens):
            if input_chunk.size(1) > MAX_SEQ_LEN:
                break
            outputs, state = model(input_chunk, state=state)
            logits = outputs[0, -1, :]
            next_id = sample_next_token(
                logits, new_ids,
                temperature=temperature, top_k=top_k, top_p=top_p,
                repetition_penalty=repetition_penalty
            )
            if next_id in stop_ids:
                break
            new_ids.append(next_id)
            text = tokenizer.decode(new_ids[prev_decoded_len:])
            if text:
                prev_decoded_len = len(new_ids)
                yield text, new_ids
            input_chunk = torch.tensor([[next_id]], dtype=torch.long, device=device)
            state = [s.detach() for s in state]
    if prev_decoded_len < len(new_ids):
        remaining = tokenizer.decode(new_ids[prev_decoded_len:])
        if remaining:
            yield remaining, new_ids


def interactive_chat(model, tokenizer, system_prompt=None,
                     temperature=0.5, top_k=30, top_p=0.85, repetition_penalty=1.35,
                     max_new_tokens=512):
    sp = _sp(tokenizer)
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    print("\n" + "=" * 60)
    print("WDLM-Neural 60M Chat (type 'quit' to exit, 'clear' to reset)")
    print(f"temperature={temperature}, top_k={top_k}, top_p={top_p}")
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
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            print("[History cleared]")
            continue

        messages.append({"role": "user", "content": user_input})
        prompt_ids = build_chat_prompt(tokenizer, messages)
        prompt_ids += [sp["im_start"], sp["agent"]]

        print("Assistant: ", end="", flush=True)
        full_ids = []
        for text_chunk, ids_batch in generate_stream(
            model, tokenizer, prompt_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature, top_k=top_k, top_p=top_p,
            repetition_penalty=repetition_penalty
        ):
            full_ids = ids_batch
            print(text_chunk, end="", flush=True)
        print()

        result = format_response(tokenizer, full_ids)
        messages.append({
            "role": "assistant",
            "content": result.get("content", ""),
            "reasoning_content": result.get("thinking", ""),
        })


def main():
    print("=" * 60)
    print(f"WDLM-Neural 60M Inference  H={HIDDEN_DIM}  L={NUM_LAYERS}")
    print("=" * 60)

    print(f"\n[1/3] Loading vocabulary...")
    voc = OpenASHVoc(agent_voc_path=agent_voc_path)
    vs = len(voc.token_to_id) + 1
    print(f"Vocabulary size: {vs}")

    print(f"\n[2/3] Loading model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = WaveDynamicsLanguageModel(vs, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS)

    sft_path = os.path.join(WEIGHT_DIR, "wdlm60m_sft_final.pth")
    pt_path = os.path.join(WEIGHT_DIR, "wdlm60m_pretrain_final.pth")

    if os.path.exists(sft_path):
        print(f"Loading SFT weights: {sft_path}")
        ckp = torch.load(sft_path, map_location=device)
        model.load_state_dict(ckp['model'] if 'model' in ckp else ckp)
    elif os.path.exists(pt_path):
        print(f"Loading pretrain weights: {pt_path}")
        ckp = torch.load(pt_path, map_location=device)
        model.load_state_dict(ckp['model'] if 'model' in ckp else ckp)
    else:
        print("WARNING: No weights found, using random init!")

    params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {params:,}")
    model.to(device)
    model.eval()

    print(f"\n[3/3] Running inference...")
    print("=" * 60)

    test_cases = [
        {"messages": [{"role": "user", "content": "你好，请介绍一下你自己。"}]},
        {"messages": [{"role": "user", "content": "什么是人工智能？"}]},
        {
            "messages": [
                {"role": "system", "content": "你是一个有用的编程助手。"},
                {"role": "user", "content": "请用Python写一个冒泡排序算法。"},
            ],
        },
    ]

    cfg = {"temperature": 0.5, "top_k": 30, "top_p": 0.85, "repetition_penalty": 1.35}

    for case in test_cases:
        msgs = case["messages"]
        user_text = msgs[-1]["content"]
        system_text = None
        for m in msgs:
            if m["role"] == "system":
                system_text = m["content"]

        print(f"\n{'=' * 60}")
        print(f"Input: {user_text}")
        if system_text:
            print(f"System: {system_text}")
        print("-" * 60)

        prompt_ids = build_user_prompt(voc, user_text, system_text=system_text)
        new_ids = generate(model, voc, prompt_ids, max_new_tokens=256, **cfg)
        result = format_response(voc, new_ids)

        parts = []
        if result.get("thinking"):
            parts.append(f"[Think] {result['thinking']}")
        if result.get("content"):
            parts.append(result["content"])
        output = " | ".join(parts) if parts else "(empty)"
        print(f"  {output[:500]}")
        print(f"  {'-' * 40}")


if __name__ == "__main__":
    main()
