"""对比任务数据: adding / copy / char-LM (语料取自 minimind sft jsonl)."""
import json
import torch


def adding_batch(b, T, device):
    """adding problem: 输入 [值, 标记], 输出两个标记位置的值之和."""
    vals = torch.rand(b, T, device=device) * 0.2 - 0.1
    marks = (torch.rand(b, T, device=device) < 2.0 / T).float()
    rows = (marks.sum(1) < 2).nonzero(as_tuple=True)[0]
    for r in rows.tolist():
        need = 2 - int(marks[r].sum().item())
        idx = torch.randint(0, T, (need,), device=device)
        marks[r, idx] = 1.0
    x = torch.stack([vals, marks], -1)
    y = (vals * marks).sum(1)
    return x, y


def copy_batch(b, T, vocab, device):
    """copy task: T 个 token + 分隔符 + 重现序列, 预测重现段."""
    seq = torch.randint(2, vocab, (b, T), device=device)
    delim = torch.full((b, 1), 1, device=device, dtype=torch.long)
    x = torch.cat([seq, delim, torch.full((b, T), 0, device=device)], 1)
    y = torch.cat([torch.full((b, T + 1), -100, device=device, dtype=torch.long), seq], 1)
    return x, y


class CharLM:
    """字符级 LM: 语料取自 minimind sft 对话文本 (构建一次, 缓存 ids)."""

    CACHE = r"F:\OpenASH2605\metaru\corpus_cache.pt"

    def __init__(self, path=r"F:\OpenASH2605\minimind_data\sft_t2t_mini.jsonl",
                 max_chars=1_500_000, ctx=128, device="cuda"):
        import os
        if os.path.exists(self.CACHE):
            blob = torch.load(self.CACHE, map_location="cpu", weights_only=False)
            ids, chars = blob["ids"], blob["chars"]
        else:
            texts, total = [], 0
            with open(path, encoding="utf-8") as f:
                for line in f:
                    try:
                        conv = json.loads(line)["conversations"]
                        texts.append("".join(t.get("content", "") for t in conv))
                        total += len(texts[-1])
                    except Exception:
                        continue
                    if total > max_chars:
                        break
            corpus = "\n".join(texts)[:max_chars]
            chars = sorted(set(corpus))
            ids = torch.tensor([chars.index(c) for c in corpus], dtype=torch.long)
            torch.save({"ids": ids, "chars": chars}, self.CACHE)
        self.stoi = {c: i for i, c in enumerate(chars)}
        self.itos = chars
        self.data = ids.to(device)
        self.vocab = len(chars)
        self.ctx = ctx
        self.device = device

    def batch(self, b):
        i = torch.randint(0, len(self.data) - self.ctx - 1, (b,), device=self.device)
        x = torch.stack([self.data[j:j + self.ctx] for j in i])
        y = torch.stack([self.data[j + 1:j + self.ctx + 1] for j in i])
        return x, y
