import json
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence


class PretrainDataset(Dataset):
    def __init__(self, path, voc, max_len=384, max_lines=50000):
        self.max_len = max_len
        self.data = []
        with open(path, encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= max_lines:
                    break
                line = line.strip()
                if not line:
                    continue
                text = json.loads(line).get('text', '')
                ids = voc.encode(text)
                if len(ids) >= 4:
                    self.data.append(torch.tensor(ids, dtype=torch.long))
        print(f'Pretrain: {len(self.data)} samples from {path} (max_lines={max_lines})')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        ids = self.data[i]
        if len(ids) > self.max_len + 1:
            ids = ids[:self.max_len + 1]
        return ids

    @staticmethod
    def collate_fn(items):
        padded = pad_sequence(items, batch_first=True, padding_value=0)
        return padded[:, :-1], padded[:, 1:]


class SFTDataset(Dataset):
    def __init__(self, path, voc, max_len=512, max_lines=30000):
        self.max_len = max_len
        self.data = []
        is_tok = voc.token_to_id.get('<|im_start|>')
        ie_tok = voc.token_to_id.get('<|im_end|>')
        uid_tok = voc.token_to_id.get('<|user|>')
        aid_tok = voc.token_to_id.get('<|agent|>')

        with open(path, encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= max_lines:
                    break
                line = line.strip()
                if not line:
                    continue
                convs = json.loads(line).get('conversations', [])
                m = []
                for msg in convs:
                    role = msg.get('role', '')
                    ct = msg.get('content', '')
                    if role == 'user':
                        m += [is_tok, uid_tok] + voc.encode(ct) + [ie_tok]
                    elif role == 'assistant':
                        m += [is_tok, aid_tok]
                        if msg.get('reasoning_content'):
                            ts = voc.token_to_id.get('<|think|>')
                            te = voc.token_to_id.get('<|end_think|>')
                            m += [ts] + voc.encode(msg['reasoning_content']) + [te]
                        m += voc.encode(ct) + [ie_tok]
                if len(m) >= 4:
                    if len(m) > self.max_len + 1:
                        m = m[:self.max_len + 1]
                    self.data.append(torch.tensor(m, dtype=torch.long))
        print(f'SFT: {len(self.data)} samples from {path} (max_lines={max_lines})')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]

    @staticmethod
    def collate_fn(items):
        padded = pad_sequence(items, batch_first=True, padding_value=0)
        return padded[:, :-1], padded[:, 1:]


def create_dataloaders(voc, mode='pretrain', config=None):
    if mode == 'pretrain':
        dataset = PretrainDataset(
            str(config.data_dir / "pretrain_t2t_mini.jsonl"),
            voc,
            max_len=config.max_seq_len,
            max_lines=config.max_pretrain_lines,
        )
    elif mode == 'sft':
        dataset = SFTDataset(
            str(config.data_dir / "sft_t2t_mini.jsonl"),
            voc,
            max_len=config.max_seq_len,
            max_lines=config.max_sft_lines,
        )
    else:
        raise ValueError(f"Unknown mode: {mode}")

    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=dataset.collate_fn,
        drop_last=True,
    )
    return loader
