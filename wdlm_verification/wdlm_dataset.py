r"""
WDLM 训练数据集
适配 F:\OpenASH2605\data 的数据格式
数据格式: {"conversations": [{"from": "human", "value": "..."}, {"from": "assistant", "value": "..."}]}
"""

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import json
import os
import re
import random

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class WDLMDataset(Dataset):
    """WDLM SFT 数据集，处理 conversation 格式数据"""
    def __init__(self, jsonl_path, tokenizer, max_seq_len=1024):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

        with open(jsonl_path, "r", encoding="utf-8") as f:
            self.samples = f.readlines()
        print(f"加载 {len(self.samples)} 条数据 from {jsonl_path}")

        self.im_start = self.tokenizer.token_to_id.get("<|im_start|>")
        self.im_end = self.tokenizer.token_to_id.get("<|im_end|>")
        self.user_id = self.tokenizer.token_to_id.get("<|user|>")
        self.agent_id = self.tokenizer.token_to_id.get("<|agent|>")
        self.system_id = self.tokenizer.token_to_id.get("<|system|>")
        self.think_start = self.tokenizer.token_to_id.get("<|think|>")
        self.think_end = self.tokenizer.token_to_id.get("<|end_think|>")
        self.unk_id = self.tokenizer.token_to_id.get("<|unk|>")

    def __len__(self):
        return len(self.samples)

    def _split_think_content(self, text):
        """分离 <think> 思考内容和回复内容"""
        # 匹配 <think>...</think> 或 <｜end▁of▁thinking｜> 格式
        think_pattern = r'<think>(.*?)</think>'
        response_pattern = r'<\s*response\s*>(.*?)$'

        think_match = re.search(think_pattern, text, re.DOTALL)
        think_content = None
        main_content = text

        if think_match:
            think_content = think_match.group(1).strip()
            main_content = text[:think_match.start()] + text[think_match.end():]
            main_content = main_content.strip()

        response_match = re.search(response_pattern, text, re.DOTALL | re.IGNORECASE)
        if response_match:
            main_content = response_match.group(1).strip()

        return think_content, main_content

    def create_chat_prompt(self, conversations):
        messages = []

        for message in conversations:
            msg = dict(message)
            from_role = msg.get("from", msg.get("role", ""))
            content = msg.get("value", msg.get("content", ""))

            if from_role in ("human", "user"):
                messages += [self.im_start, self.user_id]
                messages += self.tokenizer.encode(content)
                messages += [self.im_end]

            elif from_role in ("gpt", "assistant", "agent"):
                messages += [self.im_start, self.agent_id]

                think_content, main_content = self._split_think_content(content)

                if think_content:
                    messages += [self.think_start]
                    messages += self.tokenizer.encode(think_content)
                    messages += [self.think_end]

                if main_content:
                    messages += self.tokenizer.encode(main_content)

                messages += [self.im_end]

            elif from_role == "system":
                messages += [self.im_start, self.system_id]
                messages += self.tokenizer.encode(content)
                messages += [self.im_end]

        return messages

    def __getitem__(self, index):
        sample = json.loads(self.samples[index])
        conversations = sample.get("conversations", [])

        prompt = self.create_chat_prompt(conversations)

        if len(prompt) > self.max_seq_len:
            prompt = prompt[:self.max_seq_len]

        return torch.tensor(prompt, dtype=torch.long)

    @staticmethod
    def sft_padding_func(items):
        """input = all[:-1], target = all[1:]"""
        padded_batch = pad_sequence(items, batch_first=True, padding_value=0)
        return padded_batch[:, :-1], padded_batch[:, 1:]
