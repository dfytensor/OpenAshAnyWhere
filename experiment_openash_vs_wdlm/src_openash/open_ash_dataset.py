from torch.utils.data import Dataset, DataLoader
import torch
import json
import os
import random
from torch.nn.utils.rnn import pad_sequence

from tqdm import tqdm

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def split_sequence(x: int, l_max: int) -> list[int]:
    """
    将序列长度x分割成若干份，每份不超过l_max，且各份长度尽可能均匀

    Args:
        x: 序列总长度
        l_max: 每份最大长度

    Returns:
        长度列表，各份长度差距最小（最多相差1）

    Example:
        >>> split_sequence(100, 30)
        [25, 25, 25, 25]
        >>> split_sequence(100, 35)
        [34, 33, 33]
        >>> split_sequence(10, 3)
        [2, 2, 2, 2, 2]
    """
    import math

    # 计算需要的份数
    n = math.ceil(x / l_max)

    # 基础长度
    base = x // n

    # 前remainder份长度为base+1，其余为base
    remainder = x % n

    # 前 remainder 份长度为 base + 1
    # 后 n - remainder 份长度为 base
    lengths = [base + 1] * remainder + [base] * (n - remainder)

    return lengths
# 模型初始化

def pre_processing_chat(conversations, add_system_ratio=0.2):
    # tool use 数据完整保留不做处理
    if any(conv.get("tools") for conv in conversations):
        return conversations

    SYSTEM_PROMPTS = [
        "你是一个知识丰富的AI，尽力为用户提供准确的信息。",
        "你是openash，一个小巧但有用的语言模型。",
        "你是一个专业的AI助手，请提供有价值的回答。",
        "你是openash，请尽力帮助用户解决问题。",
        "你是一个可靠的AI，请给出准确的回答。",
        "You are a helpful AI assistant.",
        "You are openash, a lightweight intelligent assistant.",
        "You are a friendly chatbot. Please answer the user's questions carefully.",
        "You are a knowledgeable AI. Try your best to provide accurate information.",
        "You are openash, a small but useful language model.",
    ]
    # 概率性添加system
    if conversations[0].get("role") != "system":
        if random.random() < add_system_ratio:
            return [
                {"role": "system", "content": random.choice(SYSTEM_PROMPTS)}
            ] + conversations
    return conversations


import struct
import numpy as np


class PretrainDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=None):
        super().__init__()
        self.max_length = max_length
        cache_path = data_path + ".cache"

        if os.path.exists(cache_path):
            print(f"Loading cached tokens from {cache_path}")
            with open(cache_path, "rb") as f:
                num_offsets = struct.unpack("<Q", f.read(8))[0]
                offset_data = f.read(num_offsets * 8)
                offsets = struct.unpack(f"<{num_offsets}Q", offset_data)
                f.seek(8 + num_offsets * 8)
                all_ids = np.frombuffer(f.read(), dtype=np.uint16)
            self._offsets = np.array(offsets, dtype=np.int64) // 2
            self._all_ids = all_ids
            self._use_cache = True
            print(f"Cached {len(self._offsets) - 1} samples loaded, {len(all_ids)} tokens total")
        else:
            print(f"No cache found, loading raw data from {data_path}")
            self.tokenizer = tokenizer
            with open(data_path, "r", encoding="utf-8") as f:
                self.samples = f.readlines()
            self._use_cache = False

    def __len__(self):
        if self._use_cache:
            return len(self._offsets) - 1
        return len(self.samples)

    def __getitem__(self, index):
        if self._use_cache:
            start = self._offsets[index]
            end = self._offsets[index + 1]
            length = end - start

            if self.max_length and length > self.max_length:
                length = self.max_length
            return torch.from_numpy(self._all_ids[start:start + length].copy()).long()
        else:
            sample = self.samples[index]
            sample = json.loads(sample)["text"]
            sample = self.tokenizer.encode(sample)
            if self.max_length and len(sample) > self.max_length:
                sample = sample[:self.max_length]
            return torch.tensor(sample, dtype=torch.long)

    def pretrain_padding_func(self, items):
        padded_batch = pad_sequence(items, batch_first=True, padding_value=0)
        return padded_batch[:, :-1], padded_batch[:, 1:]



class SFTDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer
        with open(jsonl_path, "r", encoding="utf-8") as f:
            self.samples = f.readlines()

    def __len__(self):
        return len(self.samples)

    def create_chat_prompt(self, conversations):
        messages = []

        im_start = self.tokenizer.token_to_id.get("<|im_start|>")
        im_end = self.tokenizer.token_to_id.get("<|im_end|>")
        tool_call = self.tokenizer.token_to_id.get("<tool_call>")
        tool_call_end = self.tokenizer.token_to_id.get("</tool_call>")
        tools = self.tokenizer.token_to_id.get("<|tools|>")
        tools_end = self.tokenizer.token_to_id.get("<|end_tools|>")
        user = self.tokenizer.token_to_id.get("<|user|>")
        agent = self.tokenizer.token_to_id.get("<|agent|>")
        system = self.tokenizer.token_to_id.get("<|system|>")
        think_start = self.tokenizer.token_to_id.get("<|think|>")
        think_end = self.tokenizer.token_to_id.get("<|end_think|>")
        for message in conversations:
            message = dict(message)
            role = message.get("role")
            if role == "system":
                messages += [im_start, system]
                if message.get("content") != "":
                    messages += self.tokenizer.encode(message.get("content"))
                if "tools" in message:
                    messages += (
                        [tools]
                        + self.tokenizer.encode(message.get("tools"))
                        + [tools_end]
                    )
                messages += [im_end]

            elif role == "user":
                messages += (
                    [im_start, user]
                    + self.tokenizer.encode(message.get("content"))
                    + [im_end]
                )
            elif role == "assistant":
                messages += [im_start, agent]
                if "reasoning_content" in message:
                    if message["reasoning_content"]:
                        messages += (
                            [think_start]
                            + self.tokenizer.encode(message["reasoning_content"])
                            + [think_end]
                        )
                if "tool_calls" in message:
                    messages += (
                        [tool_call]
                        + self.tokenizer.encode(message["tool_calls"])
                        + [tool_call_end]
                    )
                if message.get("content") != "":
                    messages += self.tokenizer.encode(message["content"])

                messages += [im_end]
        print(self.tokenizer.decode(messages))
        return messages

    def __getitem__(self, index):
        sample = self.samples[index]
        sample = json.loads(sample)
        conversations = pre_processing_chat(sample["conversations"])
        prompt = self.create_chat_prompt(conversations)


        return torch.tensor(prompt, dtype=torch.long)


    def sft_padding_func(self, items):
        padded_batch = pad_sequence(items, batch_first=True, padding_value=0)
        return padded_batch[:, :-1], padded_batch[:, 1:]


class DPODataset(Dataset):
    def __init__(self, file_path, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer
        with open(file_path, "r", encoding="utf-8") as f:
            self.samples = f.readlines()
        self.im_start = self.tokenizer.token_to_id.get("<|im_start|>")
        self.im_end = self.tokenizer.token_to_id.get("<|im_end|>")
        self.user = self.tokenizer.token_to_id.get("user")
        self.agent = self.tokenizer.token_to_id.get("<|agent|>")

    def __len__(self):
        return len(self.samples)

    def create_chat_prompt(self, conversations):
        messages = []
        for message in conversations:
            message = dict(message)
            role = message.get("role")
            if role == "user":
                messages += (
                    [self.im_start, self.user]
                    + self.tokenizer.encode(message.get("content"))
                    + [self.im_end]
                )
            elif role == "assistant":
                messages += (
                    [self.im_start, self.agent]
                    + self.tokenizer.encode(message.get("content"))
                    + [self.im_end]
                )
        return messages

    def __getitem__(self, index):
        sample = self.samples[index]
        sample = json.loads(sample)
        chosen = self.create_chat_prompt(sample["chosen"])
        rejected = self.create_chat_prompt(sample["rejected"])
        return torch.tensor(chosen, dtype=torch.long), torch.tensor(
            rejected, dtype=torch.long
        )

    def dpo_padding_func(self, items):
        chosen_items = [item[0] for item in items]
        rejected_items = [item[1] for item in items]
        padded_chosen = pad_sequence(chosen_items, batch_first=True, padding_value=0)
        padded_rejected = pad_sequence(
            rejected_items, batch_first=True, padding_value=0
        )
        return (
            padded_chosen[:, :-1],
            padded_chosen[:, 1:],
            padded_rejected[:, :-1],
            padded_rejected[:, 1:],
        )


class RLAIFDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, thinking_ratio=0.5):
        super().__init__()
        self.tokenizer = tokenizer
        self.thinking_ratio = thinking_ratio
        with open(jsonl_path, "r", encoding="utf-8") as f:
            self.samples = f.readlines()
        self.im_start = self.tokenizer.token_to_id.get("<|im_start|>")
        self.im_end = self.tokenizer.token_to_id.get("<|im_end|>")
        self.user = self.tokenizer.token_to_id.get("user")
        self.agent = self.tokenizer.token_to_id.get("<|agent|>")
        self.think_start = self.tokenizer.token_to_id.get("<|think|>")
        self.think_end = self.tokenizer.token_to_id.get("<|end_think|>")

    def __len__(self):
        return len(self.samples)

    def create_chat_prompt(self, conversations, use_thinking=False):
        messages = []
        for message in conversations:
            message = dict(message)
            role = message.get("role")
            if role == "user":
                messages += (
                    [self.im_start, self.user]
                    + self.tokenizer.encode(message.get("content"))
                    + [self.im_end]
                )
            elif role == "assistant":
                messages += [self.im_start, self.agent]
                if (
                    use_thinking
                    and "reasoning_content" in message
                    and message["reasoning_content"]
                ):
                    messages += (
                        [self.think_start]
                        + self.tokenizer.encode(message["reasoning_content"])
                        + [self.think_end]
                    )
                if message.get("content"):
                    messages += self.tokenizer.encode(message.get("content"))
                messages += [self.im_end]
        return messages

    def __getitem__(self, index):
        sample = self.samples[index]
        sample = json.loads(sample)
        conversations = pre_processing_chat(sample["conversations"])
        use_thinking = random.random() < self.thinking_ratio
        prompt = self.create_chat_prompt(conversations[:-1], use_thinking=use_thinking)
        return torch.tensor(prompt, dtype=torch.long)

    def rlaif_padding_func(self, items):
        padded_batch = pad_sequence(items, batch_first=True, padding_value=0)
        return padded_batch[:, :-1], padded_batch[:, 1:]


class AgentRLDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer
        with open(jsonl_path, "r", encoding="utf-8") as f:
            self.samples = f.readlines()
        self.im_start = self.tokenizer.token_to_id.get("<|im_start|>")
        self.im_end = self.tokenizer.token_to_id.get("<|im_end|>")
        self.tool_call = self.tokenizer.token_to_id.get("<tool_call>")
        self.tool_call_end = self.tokenizer.token_to_id.get("1954")
        self.tools = self.tokenizer.token_to_id.get("<|tools|>")
        self.tools_end = self.tokenizer.token_to_id.get("<|end_tools|>")
        self.user = self.tokenizer.token_to_id.get("user")
        self.agent = self.tokenizer.token_to_id.get("<|agent|>")
        self.system = self.tokenizer.token_to_id.get("system")

    def __len__(self):
        return len(self.samples)

    def create_chat_prompt(self, conversations):
        messages = []
        for message in conversations:
            message = dict(message)
            role = message.get("role")
            if role == "system":
                messages += [self.im_start, self.system]
                if message.get("content"):
                    messages += self.tokenizer.encode(message.get("content"))
                if message.get("tools"):
                    messages += (
                        [self.tools]
                        + self.tokenizer.encode(message.get("tools"))
                        + [self.tools_end]
                    )
                messages += [self.im_end]
            elif role == "user":
                messages += (
                    [self.im_start, self.user]
                    + self.tokenizer.encode(message.get("content"))
                    + [self.im_end]
                )
            elif role == "assistant":
                messages += [self.im_start, self.agent]
                if message.get("tool_calls"):
                    messages += (
                        [self.tool_call]
                        + self.tokenizer.encode(message.get("tool_calls"))
                        + [self.tool_call_end]
                    )
                if message.get("content"):
                    messages += self.tokenizer.encode(message.get("content"))
                messages += [self.im_end]
        return messages

    def __getitem__(self, index):
        sample = self.samples[index]
        sample = json.loads(sample)
        conversations = sample["conversations"][:-1]
        prompt = self.create_chat_prompt(conversations)
        return torch.tensor(prompt, dtype=torch.long)

    def agent_rl_padding_func(self, items):
        padded_batch = pad_sequence(items, batch_first=True, padding_value=0)
        return padded_batch[:, :-1], padded_batch[:, 1:]


if __name__ == "__main__":
    from  open_ash_voc import OpenASHVoc

    vocab = OpenASHVoc
    data = PretrainDataset(
        "D:/jie_ma/minimind_dataset/pretrain_t2t_mini.jsonl", vocab,
    )
    dataloader_data = DataLoader(
        data,
        shuffle=True,
        num_workers=0,
        collate_fn=data.pretrain_padding_func,
        batch_size=10,
    )
    #
    # data = SFTDataset("D:/jie_ma/minimind_dataset/sft_t2t_mini.jsonl", vocab)
    # dataloader_data = DataLoader(
    #     data,
    #     shuffle=True,
    #     num_workers=0,
    #     collate_fn=data.sft_padding_func,
    #     batch_size=10,
    # )

    # data = DPODataset("D:/jie_ma/minimind_dataset/dpo.jsonl", vocab)
    # dataloader_data = DataLoader(
    #     data,
    #     shuffle=True,
    #     num_workers=0,
    #     collate_fn=data.dpo_padding_func,
    #     batch_size=10,
    # )
    # data = RLAIFDataset("D:/jie_ma/minimind_dataset/rlaif.jsonl", vocab)
    # dataloader_data = DataLoader(
    #     data,
    #     shuffle=True,
    #     num_workers=0,
    #     collate_fn=data.rlaif_padding_func,
    #     batch_size=10,
    # )

    # data = AgentRLDataset("D:/jie_ma/minimind_dataset/agent_rl.jsonl", vocab)
    # dataloader_data = DataLoader(
    #     data,
    #     shuffle=True,
    #     num_workers=0,
    #     collate_fn=data.agent_rl_padding_func,
    #     batch_size=10,
    # )
    #
    for i in tqdm(dataloader_data):
        print()