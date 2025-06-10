#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import json
import random
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision import transforms
from tqdm import tqdm
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.cuda.amp import autocast, GradScaler

# 确保能 import 本地的 get_tokenizer
sys.path.append(os.getcwd())
from run_radfm_inference import get_tokenizer

from Model.RadFM.multimodality_model import MultiLLaMAForCausalLM
from Dataset.variance_slice_dataset import VarianceSliceVQADataset

def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def parse_args():
    p = argparse.ArgumentParser(description="Fine-tune visual adapter for RadFM (batch 修复版)")
    p.add_argument("--lang_model_path",   required=True,
                   help="RadFM 原始 LLaMA 模型目录（含 config.json/tokenizer）")
    p.add_argument("--ckpt_path",         required=True,
                   help="RadFM 微调后的 pytorch_model.bin 路径")
    p.add_argument("--samples_dir",       required=True,
                   help="processed_samples 根目录")
    p.add_argument("--annotations_file",  required=True,
                   help="train.json 数据标注文件")
    p.add_argument("--device",            default="cuda",
                   help="设备，默认为 cuda")
    p.add_argument("--num_slices",  type=int, default=10,
                   help="每个样本选取的 slice 数量")
    p.add_argument("--batch_size",  type=int, default=1,
                   help="训练 batch size")
    p.add_argument("--lr",          type=float, default=5e-5,
                   help="学习率（可尝试 5e-5~1e-4）")
    p.add_argument("--epochs",      type=int,   default=5,
                   help="训练轮数")
    p.add_argument("--warmup_ratio",type=float, default=0.1,
                   help="学习率预热比例")
    p.add_argument("--accum_steps", type=int,   default=4,
                   help="梯度累积步数，用于模拟更大 batch")
    p.add_argument("--log_interval",type=int,   default=200,
                   help="每隔多少更新步打印一次 loss")
    p.add_argument("--seed",        type=int,   default=42,
                   help="随机种子，保证可复现")
    p.add_argument("--resume_adapter", type=str, default=None,
                   help="可选：中断后继续训练时，加载已有 adapter 权重（.pth）")
    return p.parse_args()

def main():
    args   = parse_args()
    set_seed(args.seed)

    # 设备 & cuDNN 加速
    device = torch.device(args.device)
    cudnn.benchmark = True

    # 1) 离线加载 tokenizer & image_padding_tokens
    print(">>> Loading tokenizer …")
    tokenizer, image_padding_tokens = get_tokenizer(args.lang_model_path)

    # 2) 构造模型，并在 __init__ 中加载 RadFM 权重
    print(">>> Loading RadFM model …")
    model = MultiLLaMAForCausalLM(
        lang_model_path = args.lang_model_path,
        ckpt_path       = args.ckpt_path
    )
    # 关闭 use_cache
    try:
        model.lang_model.config.use_cache = False
    except:
        pass
    model.to(device)

    # 如果要从已有 adapter 断点恢复
    if args.resume_adapter:
        print(f">>> Resuming adapter from {args.resume_adapter}")
        sd = torch.load(args.resume_adapter, map_location="cpu")
        model.embedding_layer.adapter.load_state_dict(sd)

    # 3) 冻结除 adapter & 多模态融合头外所有参数
    for name, p in model.named_parameters():
        if not any(k in name for k in ("adapter", "fc_proj", "cls_head", "mm_head")):
            p.requires_grad = False

    print(">>> Trainable parameters:")
    for name, p in model.named_parameters():
        if p.requires_grad:
            print("   ", name)
    print("="*60)

    # 4) Dataset & DataLoader
    clip_transforms = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711)
        ),
    ])
    ds = VarianceSliceVQADataset(
        samples_root     = args.samples_dir,
        annotations_file = args.annotations_file,
        num_slices       = args.num_slices,
        transforms       = clip_transforms
    )
    loader = DataLoader(
        ds,
        batch_size        = args.batch_size,
        shuffle           = True,
        num_workers       = min(8, os.cpu_count()),
        pin_memory        = True,
        persistent_workers= True
    )

    # 5) 优化器 & 调度器
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=1e-2
    )
    total_updates = (len(loader) + args.accum_steps - 1) // args.accum_steps * args.epochs
    warmup_steps  = int(total_updates * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps   = warmup_steps,
        num_training_steps = total_updates
    )

    # 6) AMP & 梯度 scaler
    scaler = GradScaler()

    loss_history = []
    update_step  = 0

    # 7) 训练循环
    for ep in range(1, args.epochs + 1):
        torch.cuda.empty_cache()
        model.train()
        running_loss = 0.0
        pbar = tqdm(loader, desc=f"Epoch {ep}/{args.epochs}", leave=False)

        optimizer.zero_grad()
        for step, (slices, question, answer) in enumerate(pbar, start=1):
            slices = slices.to(device, non_blocking=True)

            # —— 核心修改：用整个 batch 拼 text —— #
            text = [q + " " + a for q, a in zip(question, answer)]
            enc  = tokenizer(
                text,
                max_length     = 512,
                truncation     = True,
                padding        = "longest",
                return_tensors = "pt"
            ).to(device)

            lang_x    = enc.input_ids      # [B, L]
            attn_mask = enc.attention_mask # [B, L]
            labels    = lang_x.clone()
            labels[labels == tokenizer.pad_token_id] = -100
            loss_rw   = torch.ones_like(labels, device=device, dtype=torch.float)

            # 混合精度前向
            with autocast():
                out  = model(
                    lang_x         = lang_x,
                    vision_x       = slices,
                    attention_mask = attn_mask,
                    labels         = labels,
                    loss_reweight  = loss_rw,
                    key_words_query= None
                )
                loss = out["loss"] / args.accum_steps

            # 反向 + 累积
            scaler.scale(loss).backward()

            if step % args.accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

                update_step += 1
                running_loss += (loss.item() * args.accum_steps)
                avg_loss = running_loss / update_step
                loss_history.append(avg_loss)

                if update_step % args.log_interval == 0:
                    pbar.set_postfix({
                        "step": update_step,
                        "lr":    f"{scheduler.get_last_lr()[0]:.2e}",
                        "loss":  f"{avg_loss:.4f}"
                    })

        # 每个 epoch 结束
        ep_avg = running_loss / update_step
        print(f"\n--- Epoch {ep} done.  Avg loss: {ep_avg:.4f} ---\n")
        torch.save(model.embedding_layer.adapter.state_dict(), f"adapter_ep{ep}.pth")

    # 8) 最终保存
    with open("loss_history.json", "w") as f:
        json.dump(loss_history, f, indent=2)
    torch.save(model.embedding_layer.adapter.state_dict(), "adapter_finetuned.pth")
    print("训练结束，adapter 和 loss 曲线已保存。")

if __name__ == "__main__":
    main()


