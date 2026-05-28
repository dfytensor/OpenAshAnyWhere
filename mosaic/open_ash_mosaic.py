import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch import nn
from open_ash import DecoderLayer


class ExpertModule(nn.Module):
    def __init__(self, hidden_size, num_heads, num_layers, voc_size, model_flag="train"):
        super().__init__()
        self.layers = nn.ModuleList(
            [DecoderLayer(hidden_size, num_heads, model_flag) for _ in range(num_layers)]
        )
        self.head = nn.Linear(hidden_size, voc_size, bias=False)

    def forward(self, x, states=None):
        if states is None:
            states = [None] * len(self.layers)
        for i, layer in enumerate(self.layers):
            x1, states[i] = layer(x, states[i])
            x = x1 + x
        return self.head(x), states


class OpenASHMoSAIC(nn.Module):
    def __init__(self, voc_size, hidden_size, num_heads,
                 num_encoder_layers, num_expert_layers,
                 model_flag="train", max_experts=10):
        super().__init__()
        self.voc_size = voc_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_encoder_layers = num_encoder_layers
        self.num_expert_layers = num_expert_layers
        self.model_flag = model_flag
        self.max_experts = max_experts

        self.em = nn.Embedding(voc_size, hidden_size, padding_idx=0)
        self.encoder_layers = nn.ModuleList(
            [DecoderLayer(hidden_size, num_heads, model_flag) for _ in range(num_encoder_layers)]
        )

        self.router = nn.Linear(hidden_size, max_experts, bias=False)

        self.experts = nn.ModuleDict()

    def add_expert(self, expert_id, init_from=None):
        eid = str(expert_id)
        if eid in self.experts:
            return
        device = next(self.parameters()).device
        expert = ExpertModule(
            self.hidden_size, self.num_heads, self.num_expert_layers,
            self.voc_size, self.model_flag
        ).to(device)

        if init_from is not None and str(init_from) in self.experts:
            src = self.experts[str(init_from)]
            expert.load_state_dict(src.state_dict())

        self.experts[eid] = expert

    def freeze_encoder(self):
        self.em.weight.requires_grad = False
        for param in self.encoder_layers.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self):
        self.em.weight.requires_grad = True
        for param in self.encoder_layers.parameters():
            param.requires_grad = True

    def expert_parameters(self, expert_id):
        eid = str(expert_id)
        return list(self.experts[eid].parameters())

    def trainable_parameters(self):
        params = []
        for expert in self.experts.values():
            params += list(expert.parameters())
        params += list(self.router.parameters())
        return params

    def load_from_pretrained(self, state_dict):
        self.add_expert("base")
        new_sd = {}
        for key, val in state_dict.items():
            if key == "em.weight":
                new_sd[key] = val
            elif key.startswith("decoder_layers."):
                parts = key.split(".")
                layer_idx = int(parts[1])
                rest = ".".join(parts[2:])
                if layer_idx < self.num_encoder_layers:
                    new_sd[f"encoder_layers.{layer_idx}.{rest}"] = val
                else:
                    eidx = layer_idx - self.num_encoder_layers
                    new_sd[f"experts.base.layers.{eidx}.{rest}"] = val
            elif key.startswith("head_score."):
                suffix = key[len("head_score."):]
                new_sd[f"experts.base.head.{suffix}"] = val
        self.load_state_dict(new_sd, strict=False)

    def forward(self, x, state=None, expert_id="base"):
        x = self.em(x)
        if state is None:
            state = {
                "encoder": [None] * self.num_encoder_layers,
                "expert": None,
                "active_expert": None,
            }

        for i, layer in enumerate(self.encoder_layers):
            x1, state["encoder"][i] = layer(x, state["encoder"][i])
            x = x1 + x

        eid = str(expert_id)
        if eid not in self.experts:
            raise KeyError(f"Expert '{eid}' not found. Available: {list(self.experts.keys())}")

        if state["active_expert"] != eid:
            state["expert"] = [None] * self.num_expert_layers
            state["active_expert"] = eid

        x_out, state["expert"] = self.experts[eid](x, state["expert"])
        return x_out, state

    def forward_encoder(self, x):
        x = self.em(x)
        for layer in self.encoder_layers:
            x1, _ = layer(x, None)
            x = x1 + x
        return x

    def route(self, x):
        features = self.forward_encoder(x)
        pooled = features.mean(dim=1)
        return self.router(pooled)

    @staticmethod
    def detach_state(state):
        if state is None:
            return None
        new_state = {
            "encoder": [s.detach() if s is not None else None for s in state["encoder"]],
            "expert": [s.detach() if s is not None else None for s in state["expert"]] if state["expert"] else None,
            "active_expert": state["active_expert"],
        }
        return new_state

    def save_expert(self, expert_id, path):
        eid = str(expert_id)
        if eid not in self.experts:
            raise KeyError(f"Expert '{eid}' not found")
        torch.save(self.experts[eid].state_dict(), path)

    def load_expert(self, expert_id, path):
        eid = str(expert_id)
        if eid not in self.experts:
            self.add_expert(eid)
        self.experts[eid].load_state_dict(torch.load(path, map_location="cpu"))

    def save_full(self, path):
        torch.save(self.state_dict(), path)

    def expert_info(self):
        info = {"encoder_layers": self.num_encoder_layers, "expert_layers": self.num_expert_layers, "experts": {}}
        enc_params = sum(p.numel() for p in self.encoder_layers.parameters()) + self.em.num_embeddings * self.em.embedding_dim
        info["encoder_params"] = enc_params
        for eid, expert in self.experts.items():
            params = sum(p.numel() for p in expert.parameters())
            frozen = sum(p.numel() for p in expert.parameters() if not p.requires_grad)
            info["experts"][eid] = {"params": params, "frozen": frozen}
        info["router_params"] = sum(p.numel() for p in self.router.parameters())
        return info


if __name__ == "__main__":
    voc_size = 12506
    hidden_size = 768
    num_layers = 12
    num_heads = 8
    num_enc = 6
    num_exp = 6

    model = OpenASHMoSAIC(
        voc_size=voc_size, hidden_size=hidden_size, num_heads=num_heads,
        num_encoder_layers=num_enc, num_expert_layers=num_exp,
    )

    model.add_expert("base")
    model.add_expert("math", init_from="base")
    model.add_expert("code", init_from="base")
    model.freeze_encoder()

    info = model.expert_info()
    total = info["encoder_params"] + sum(e["params"] for e in info["experts"].values()) + info["router_params"]
    trainable = sum(e["params"] for e in info["experts"].values()) + info["router_params"]

    print(f"Total params:       {total:,}")
    print(f"Encoder (frozen):   {info['encoder_params']:,}")
    print(f"Trainable:          {trainable:,}")
    for eid, einfo in info["experts"].items():
        print(f"  Expert '{eid}':   {einfo['params']:,}")
    print(f"Router:             {info['router_params']:,}")
    print(f"Trainable ratio:    {trainable/total*100:.1f}%")
