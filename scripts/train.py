import os
import math
import argparse
import time

import psutil
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import wandb
from transformers import PreTrainedTokenizerFast

from transformer.model import TransformerLM
from transformer.data import CausalMemmapDataset
from transformer.optimizer import SGD, AdamW, ConsinSchedule
from transformer.nn_utils import CrossEntropy, gradient_clipping
from transformer.serialization import save_checkpoint, load_checkpoint



def evaluate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs.view(-1, outputs.size(-1)), labels.view(-1))
            total_loss += loss.item()
    return total_loss / len(val_loader)

def train(args, model, train_loader, val_loader, optimizer, criterion, device):
    model.train()

    loss_history = []
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            inputs, labels = batch
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.view(-1, outputs.size(-1)), labels.view(-1))
            loss.backward()
            gradient_clipping(model.parameters(), max_l2_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()

        avg_epoch_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch+1} Loss: {avg_epoch_loss:.4f}")

        val_loss = evaluate(model, val_loader, criterion, device)
        print(f"Epoch {epoch+1} Validation Loss: {val_loss:.4f}")

        loss_history.append((avg_epoch_loss, val_loss))
        if (epoch + 1) % args.checkpoint_interval == 0:
            checkpoint_path = os.path.join(args.checkpoint_dir, f"model_epoch_{epoch+1}.pt")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")
            
    return loss_history
    


def main(args):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using Device: {device}")
    os.makedirs(args.checkpoint_dir, exist_ok=True)


    total_ds = CausalMemmapDataset(args.data_path, args.context_length)
    total_blocks = len(total_ds)
    split = 0.8
    split_block = int(total_blocks * split)

    train_ds = CausalMemmapDataset(
        args.data_path,
        args.context_length,
        start_block=0,
        end_block=split_block
    )

    val_ds = CausalMemmapDataset(
        args.data_path,
        args.context_length,
        start_block=split_block,
        end_block=total_blocks
    )

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=args.tokenizer_path,
    )
    args.vocab_size = tokenizer.vocab_size

    # 确保dataset不为空
    if len(train_ds) == 0:
        raise ValueError("训练数据集为空，增加数据量或减小上下文长度。")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False
    )


    # 初始化模型
    model = TransformerLM(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        max_seq_len=args.context_length
    ).to(device)

    print(f"模型参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)
    loss_fn = CrossEntropy

    loss_history = train(args, model, train_loader, val_loader, optimizer, loss_fn, device)

    



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--checkpoint_interval", type=int, default=1)
    
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min_lr", type=float, default=3e-5)

    # 模型超参数
    parser.add_argument("--vocab_size", type=int, default=50257)
    parser.add_argument("--num_layers", type=int, default=6)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--d_ff", type=int, default=1024)
    parser.add_argument("--use_rope", action="store_true")
    parser.add_argument("--context_length", type=int, default=128)
    parser.add_argument("--theta", type=float, default=100000.0)
    

    parser.add_argument("--checkpoint_dir", type=str, default="../checkpoints")
    parser.add_argument("--data_path", type=str, default="data.bin")
    parser.add_argument("--tokenizer_path", type=str, default="bpe_tokenizer.json")

    args = parser.parse_args()

    main(args)