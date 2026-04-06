import os
import math
import argparse
import time
import logging

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
from transformer.optimizer import AdamW, CosineSchedule
from transformer.nn_utils import CrossEntropy, gradient_clipping
from transformer.serialization import save_checkpoint, load_checkpoint


def setup_logging(log_dir):
    """设置日志配置"""
    os.makedirs(log_dir, exist_ok=True)
    
    logger = logging.getLogger("TransformerLM")
    logger.setLevel(logging.DEBUG)
    
    # 文件处理器
    fh = logging.FileHandler(
        os.path.join(log_dir, f"train_{time.strftime('%Y%m%d_%H%M%S')}.log")
    )
    fh.setLevel(logging.INFO)
    
    # 控制台处理器
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    # 日志格式
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger


def plot_training_history(loss_history, checkpoint_dir):
    """绘制训练和验证loss曲线"""
    train_losses = [h[0] for h in loss_history]
    val_losses = [h[1] for h in loss_history]
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Training Loss", marker='o')
    plt.plot(val_losses, label="Validation Loss", marker='s')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plot_path = os.path.join(checkpoint_dir, "loss_curve.png")
    plt.savefig(plot_path, dpi=100, bbox_inches='tight')
    print(f"Saved loss curve to: {plot_path}")
    
    # 上传到wandb
    wandb.log({"loss_curve": wandb.Image(plot_path)})
    plt.close()


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
             
    model.train()
    return total_loss / len(val_loader)


def train(args, model, train_loader, val_loader, optimizer, scheduler, criterion, logger):
    model.train()

    loss_history = []
    global_step = 0
    
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for inputs, labels in pbar:
            inputs, labels = inputs.to(args.device), labels.to(args.device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.view(-1, outputs.size(-1)), labels.view(-1))
            loss.backward()
            gradient_clipping(model.parameters(), max_l2_norm=1.0)
            
            # 使用调度器计算当前学习率并更新优化器
            current_lr = scheduler.get_lr(global_step)
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr
            
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
            global_step += 1
            
            # 每100步记录一次训练loss
            if global_step % 100 == 0:
                wandb.log({
                    "train/loss": loss.item(),
                    "train/learning_rate": current_lr,
                    "train/global_step": global_step,
                }, step=global_step)
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{current_lr:.2e}"}, refresh=True)

        avg_epoch_loss = epoch_loss / num_batches
        logger.info(f"Epoch {epoch+1} Average Loss: {avg_epoch_loss:.4f}")

        val_loss = evaluate(model, val_loader, criterion, args.device)
        logger.info(f"Epoch {epoch+1} Validation Loss: {val_loss:.4f}")

        # 记录epoch级别的指标
        wandb.log({
            "epoch/train_loss": avg_epoch_loss,
            "epoch/val_loss": val_loss,
            "epoch": epoch + 1,
        }, step=epoch + 1)

        loss_history.append((avg_epoch_loss, val_loss))
        
        # 保存checkpoint
        if (epoch + 1) % args.checkpoint_interval == 0:
            checkpoint_path = os.path.join(args.checkpoint_dir, f"model_epoch_{epoch+1}.pt")
            save_checkpoint(model, optimizer, epoch+1, checkpoint_path)
            logger.info(f"Checkpoint saved: {checkpoint_path}")
            
    return loss_history
    


def main(args):
    # 设置日志
    logger = setup_logging(args.log_dir)
    logger.info(f"Starting training with args: {vars(args)}")
    
    # 初始化wandb
    os.makedirs(args.log_dir, exist_ok=True)
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=f"exp_{time.strftime('%Y%m%d_%H%M%S')}",
        config=vars(args),
        dir=args.log_dir,
    )
    logger.info("Initialized Weights & Biases")
    
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using Device: {args.device}")
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # 记录系统信息
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    logger.info(f"GPU Count: {num_gpus}")
    wandb.log({
        "system/device": args.device,
        "system/cuda_available": torch.cuda.is_available(),
        "system/num_gpus": num_gpus,
    })


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
    logger.info(f"Tokenizer vocab size: {args.vocab_size}")

    # 确保dataset不为空
    if len(train_ds) == 0:
        logger.error("训练数据集为空")
        raise ValueError("训练数据集为空，增加数据量或减小上下文长度。")
    
    logger.info(f"Train dataset size: {len(train_ds)}, Val dataset size: {len(val_ds)}")

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
        num_layers=args.num_layers,
        d_model=args.d_model,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        use_rope=args.use_rope,
        context_length=args.context_length,
        theta=args.theta,
        device=args.device,
        dtype=torch.float32  
    ) 
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total Parameters: {total_params / 1e6:.2f}M")
    logger.info(f"Trainable Parameters: {trainable_params / 1e6:.2f}M")

    # 记录模型信息
    wandb.log({
        "model/total_parameters": total_params,
        "model/trainable_parameters": trainable_params,
    })

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)
    loss_fn = CrossEntropy
    # 初始化学习率调度器
    total_steps = len(train_loader) * args.epochs
    scheduler = CosineSchedule(
        max_learning_rate=args.lr,
        min_learning_rate=args.min_lr,
        warmup_iters=args.warmup_iters,
        cosine_cycle_iters=args.cosine_cycle_iters,
    )

    loss_history = train(args, model, train_loader, val_loader, optimizer, scheduler, loss_fn, logger)
    
    # 训练完成后的总结
    logger.info("\n=" * 50)
    logger.info("Training Completed!")
    logger.info(f"Total epochs: {args.epochs}")
    logger.info(f"Final training loss: {loss_history[-1][0]:.4f}")
    logger.info(f"Final validation loss: {loss_history[-1][1]:.4f}")
    logger.info("=" * 50)
    
    # 最终的checkpoint
    final_checkpoint_path = os.path.join(args.checkpoint_dir, "model_final.pt")
    save_checkpoint(model, optimizer, args.epochs, final_checkpoint_path)
    logger.info(f"Saved final checkpoint: {final_checkpoint_path}")
    wandb.log({"final_checkpoint": final_checkpoint_path})
    
    # 绘制loss曲线
    plot_training_history(loss_history, args.checkpoint_dir)
    
    wandb.finish()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--checkpoint_interval", type=int, default=5)
    
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--min_lr", type=float, default=3e-4)
    parser.add_argument("--max_lr", type=float, default=3e-3)
    parser.add_argument("--warmup_iters", type=int, default=100)
    parser.add_argument("--cosine_cycle_iters", type=int, default=1000)

    # 模型超参数
    parser.add_argument("--vocab_size", type=int, default=50257)
    parser.add_argument("--num_layers", type=int, default=6)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--d_ff", type=int, default=1024)
    parser.add_argument("--use_rope", action="store_true")
    parser.add_argument("--context_length", type=int, default=128)
    parser.add_argument("--theta", type=float, default=100000.0)
    
    # 数据和路径参数
    parser.add_argument("--checkpoint_dir", type=str, default="/mnt/miaohua/lizehua/workspace/basic/checkpoints")
    parser.add_argument("--data_path", type=str, default="/mnt/miaohua/lizehua/workspace/basic/data/TinyStoriesV2.bin")
    parser.add_argument("--tokenizer_path", type=str, default="/mnt/miaohua/lizehua/workspace/basic/data/bpe_tokenizer/tokenizer.json")
    
    # W&B参数
    parser.add_argument("--wandb_project", type=str, default="transformer-lm", help="Weights & Biases project name")
    parser.add_argument("--wandb_entity", type=str, default=None, help="Weights & Biases entity name")
    parser.add_argument("--log_dir", type=str, default="../logs", help="Directory for logs")

    args = parser.parse_args()

    main(args)