import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import time
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

from open_ash_mosaic import OpenASHMoSAIC
from open_ash_voc import OpenASHVoc
from open_ash_dataset import SFTDataset, PretrainDataset
from config import agent_voc_path


def get_lr(current_step, total_steps, lr):
    import math
    return lr * (0.1 + 0.45 * (1 + math.cos(math.pi * current_step / total_steps)))


def train_expert_epoch(epoch, model, loader, optimizer, scaler, criterion, args,
                       expert_id, voc_size):
    model.train()
    for param in model.encoder_layers.parameters():
        param.requires_grad = False
    model.em.weight.requires_grad = False

    iters = len(loader)
    avg_loss = 0
    steps = 0
    start_time = time.time()
    last_step = 0

    bar = tqdm(loader, desc=f"Epoch {epoch + 1}/{args.epochs}")
    for step, batch_tokens in enumerate(bar, start=1):
        last_step = step
        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        inputs, targets = batch_tokens
        if isinstance(inputs, (list, tuple)):
            inputs = pad_sequence(inputs, batch_first=True, padding_value=0)
            targets = pad_sequence(targets, batch_first=True, padding_value=0)

        inputs = inputs[:, :args.max_seq_len].to(args.device, non_blocking=True)
        targets = targets[:, :args.max_seq_len].to(args.device, non_blocking=True)

        if inputs.numel() == 0:
            continue

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            outputs, _ = model(inputs, expert_id=expert_id)
            B, S, V = outputs.size()
            loss = criterion(outputs.view(B * S, V), targets.view(B * S)) / args.accumulation_steps

        steps += 1
        avg_loss += loss.item()
        bar.set_description(f"Epoch {epoch + 1}/{args.epochs} loss: {avg_loss / steps:.4f}")

        scaler.scale(loss).backward()

        if step % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.expert_parameters(expert_id), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        if step % args.log_interval == 0:
            spend = time.time() - start_time
            eta = spend / max(step, 1) * (iters - step) // 60
            print(f"  [{step}/{iters}] lr={lr:.8f} eta={eta:.0f}min")

    if last_step > 0 and last_step % args.accumulation_steps != 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.expert_parameters(expert_id), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)


def train_router(model, data_loaders, args, epochs=5):
    """
    Train router using data from each expert's domain.
    data_loaders: dict {expert_id: (loader, dataset_class)}
    """
    print("\n[Router Training]")
    model.train()
    for param in model.encoder_layers.parameters():
        param.requires_grad = False
    model.em.weight.requires_grad = False
    for expert in model.experts.values():
        for param in expert.parameters():
            param.requires_grad = False

    optimizer = optim.AdamW(model.router.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    expert_ids = sorted(data_loaders.keys())
    expert_to_idx = {eid: i for i, eid in enumerate(expert_ids)}

    features_cache = []
    labels_cache = []

    print("  Extracting encoder features...")
    model.eval()
    with torch.no_grad():
        for eid in expert_ids:
            loader, _ = data_loaders[eid]
            idx = expert_to_idx[eid]
            count = 0
            for batch in loader:
                if isinstance(batch, (list, tuple)):
                    inputs = batch[0]
                else:
                    inputs = batch
                if isinstance(inputs, (list, tuple)):
                    inputs = pad_sequence(inputs, batch_first=True, padding_value=0)
                inputs = inputs[:, :args.max_seq_len].to(args.device)
                feats = model.forward_encoder(inputs)
                pooled = feats.mean(dim=1)
                features_cache.append(pooled.cpu())
                labels_cache.append(torch.full((pooled.size(0),), idx, dtype=torch.long))
                count += pooled.size(0)
                if count >= 5000:
                    break
            print(f"    Expert '{eid}': {count} samples")

    all_features = torch.cat(features_cache, dim=0).to(args.device)
    all_labels = torch.cat(labels_cache, dim=0).to(args.device)
    dataset = torch.utils.data.TensorDataset(all_features, all_labels)
    router_loader = DataLoader(dataset, batch_size=128, shuffle=True)

    print(f"  Training router for {epochs} epochs, {len(expert_ids)} experts...")
    model.train()
    for epoch in range(epochs):
        correct = 0
        total = 0
        for feats, labels in router_loader:
            optimizer.zero_grad()
            logits = model.router(feats)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            correct += (logits.argmax(dim=-1) == labels).sum().item()
            total += labels.size(0)
        print(f"    Epoch {epoch + 1}/{epochs}: router acc = {correct / total * 100:.2f}%")

    model.eval()
    print("  Router training complete.")


def main():
    parser = argparse.ArgumentParser(description="OpenASH MoSAIC Expert Training")
    parser.add_argument("--pretrained_weight", type=str, required=True,
                        help="Path to pretrained monolithic weight (.pth)")
    parser.add_argument("--expert_name", type=str, required=True,
                        help="Name for the new expert (e.g. 'math', 'code', 'chat')")
    parser.add_argument("--init_from", type=str, default="base",
                        help="Initialize new expert from which expert (default: base)")
    parser.add_argument("--data_path", type=str, required=True,
                        help="Training data path (jsonl)")
    parser.add_argument("--data_type", type=str, default="sft", choices=["sft", "pretrain"],
                        help="Dataset type")
    parser.add_argument("--save_dir", type=str, default="../out_mosaic",
                        help="Save directory")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--accumulation_steps", type=int, default=1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--num_layers", type=int, default=12)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--num_encoder_layers", type=int, default=6)
    parser.add_argument("--train_router", type=int, default=0, choices=[0, 1],
                        help="Train router after expert training")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    num_expert_layers = args.num_layers - args.num_encoder_layers

    print("=" * 60)
    print("  OpenASH MoSAIC - Expert Training")
    print("=" * 60)
    print(f"  Expert name:      {args.expert_name}")
    print(f"  Init from:        {args.init_from}")
    print(f"  Encoder layers:   {args.num_encoder_layers}")
    print(f"  Expert layers:    {num_expert_layers}")
    print(f"  Data:             {args.data_path}")
    print(f"  Device:           {args.device}")

    print("\n[1/5] Loading vocabulary...")
    voc = OpenASHVoc(agent_voc_path=agent_voc_path)
    voc_size = len(voc.token_to_id) + 1
    print(f"  Voc size: {voc_size}")

    print("\n[2/5] Building MoSAIC model...")
    model = OpenASHMoSAIC(
        voc_size=voc_size, hidden_size=args.hidden_size, num_heads=args.num_heads,
        num_encoder_layers=args.num_encoder_layers, num_expert_layers=num_expert_layers,
    )

    print(f"  Loading pretrained weights: {args.pretrained_weight}")
    pretrained_sd = torch.load(args.pretrained_weight, map_location="cpu", weights_only=False)
    model.load_from_pretrained(pretrained_sd)
    del pretrained_sd

    model.add_expert(args.expert_name, init_from=args.init_from if args.init_from != "none" else None)
    model.freeze_encoder()
    model.to(args.device)

    enc_params = sum(p.numel() for p in model.encoder_layers.parameters()) + model.em.num_embeddings * model.em.embedding_dim
    expert_params = sum(p.numel() for p in model.experts[str(args.expert_name)].parameters())
    print(f"  Encoder params (frozen): {enc_params:,}")
    print(f"  Expert '{args.expert_name}' params (trainable): {expert_params:,}")

    print("\n[3/5] Loading dataset...")
    if args.data_type == "sft":
        dataset = SFTDataset(args.data_path, voc)
        collate_fn = dataset.sft_padding_func
    else:
        dataset = PretrainDataset(args.data_path, voc, max_length=args.max_seq_len)
        collate_fn = dataset.pretrain_padding_func
    print(f"  Samples: {len(dataset)}")

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=4, pin_memory=True, collate_fn=collate_fn)

    print("\n[4/5] Training expert...")
    optimizer = optim.AdamW(model.expert_parameters(args.expert_name), lr=args.learning_rate)
    scaler = torch.amp.GradScaler()
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    for epoch in range(args.epochs):
        train_expert_epoch(epoch, model, loader, optimizer, scaler, criterion,
                           args, args.expert_name, voc_size)

    print("\n[5/5] Saving...")
    full_path = os.path.join(args.save_dir, f"mosaic_{args.expert_name}_{args.hidden_size}_{args.num_layers}.pth")
    model.save_full(full_path)
    print(f"  Full model saved: {full_path}")

    expert_path = os.path.join(args.save_dir, f"expert_{args.expert_name}.pth")
    model.save_expert(args.expert_name, expert_path)
    print(f"  Expert only saved: {expert_path}")

    if args.train_router:
        print("\n[Router] Loading data for router training...")
        router_data = {}
        for eid in model.experts.keys():
            ds = SFTDataset(args.data_path, voc) if args.data_type == "sft" else PretrainDataset(args.data_path, voc)
            ld = DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
            router_data[eid] = (ld, type(ds))
        train_router(model, router_data, args)

    print("\nDone.")


if __name__ == "__main__":
    main()
