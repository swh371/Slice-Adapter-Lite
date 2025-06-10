#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_radfm_inference.py

对 processed_samples/<sample_id>/*.jpg 组合 question_en，用 Rad-FM（MultiLLaMAForCausalLM）做推理，
将 quiz_id, sample_id, question_en, answer_en, pred_text 全部输出到 CSV。
"""
import os
import sys
import glob
import json
import csv
import argparse

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

# 确保可以 import Model.RadFM
sys.path.append(os.getcwd())

from Model.RadFM.multimodality_model import MultiLLaMAForCausalLM
from transformers import LlamaTokenizer

def get_tokenizer(tokenizer_path: str,
                  max_img_size: int = 100,
                  image_num: int    = 32):
    """
    初始化 tokenizer，并生成 image_padding_tokens 列表。
    max_img_size: 最多支持多少“不同的 image 段”（这里我们只用到第 0 段）
    image_num: 每段对应多少个 patch token（这里 32）
    """
    text_tokenizer = LlamaTokenizer.from_pretrained(tokenizer_path)
    # 构造所有可能的 <image0>..<image{max_img_size*image_num-1}> token
    specials = ["<image>","</image>"]
    for i in range(max_img_size * image_num):
        specials.append(f"<image{i}>")
    text_tokenizer.add_special_tokens({"additional_special_tokens": specials})
    # LLaMA 通用 id
    text_tokenizer.pad_token_id = 0
    text_tokenizer.bos_token_id = 1
    text_tokenizer.eos_token_id = 2

    # 构造 padding token 字符串列表，每个 entry 是 image_num 个连续 patch token
    image_padding_tokens = []
    for seg in range(max_img_size):
        toks = [f"<image{seg*image_num + j}>" for j in range(image_num)]
        image_padding_tokens.append("".join(toks))
    return text_tokenizer, image_padding_tokens

def combine_and_preprocess(question: str,
                           slice_paths:  list,
                           image_padding_tokens: list,
                           device: torch.device):
    """
    1) 读取所有 slice_paths，对每张图做 ToTensor->unsqueeze->[1,C,1,H0,W0]
    2) 沿第 2 维（depth）cat，得到 [1,C,D,H0,W0]
    3) 插值到 [1,C,32,256,256]
    4) 构造含一个 <image>…</image> 占位符的 text
    """
    # 只取 32 张 slice（如果更多，前 32；不足，pad 最后一帧）
    slice_paths = sorted(slice_paths)[:32]
    # 基本 resize 到 256x256
    transform = transforms.Compose([
        transforms.ToTensor(),            # [C,H0,W0]
        transforms.Resize((256,256)),     # [C,256,256]
    ])

    slice_tensors = []
    for p in slice_paths:
        im = Image.open(p).convert('RGB')
        t  = transform(im)           # [3,256,256]
        t  = t.unsqueeze(0).unsqueeze(2)  # [1,3,1,256,256]
        slice_tensors.append(t.to(device))

    # 如果不足 32，就用最后一帧重复 pad
    while len(slice_tensors) < 32:
        slice_tensors.append(slice_tensors[-1])

    # cat depth -> [1,3,32,256,256]
    vol = torch.cat(slice_tensors, dim=2)

    # 确保数值范围
    vol = vol.clamp(0.0, 1.0)

    # 插值（trilinear）到精确 (32,256,256) 格式（其实已经是了，但保证通用）
    vol = F.interpolate(
        vol,
        size=(32,256,256),
        mode='trilinear',
        align_corners=False
    )

    # 文本里只插一次 <image> ... </image>
    pad_tokens = image_padding_tokens[0]  # 32 个 <image0>..<image31>
    text = f"{question}<image>{pad_tokens}</image>"

    return text, vol.unsqueeze(0)  # [1,3,32,256,256] -> [1,3,32,256,256]

def main(lang_model_path, ckpt_path,
         samples_dir, json_file,
         output_file, device_str):
    device = torch.device(device_str)

    # 1. tokenizer & padding tokens
    print(">>> Loading tokenizer …")
    tokenizer, image_padding_tokens = get_tokenizer(lang_model_path)

    # 2. model
    print(">>> Loading Rad-FM model …")
    model = MultiLLaMAForCausalLM(lang_model_path)
    sd = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(sd)
    model.to(device).eval()

    # 3. 读 json
    with open(json_file, 'r', encoding='utf-8') as f:
        samples = json.load(f)

    # 4. 准备输出 CSV
    os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
    with open(output_file, 'w', newline='', encoding='utf-8') as outf:
        writer = csv.writer(outf)
        writer.writerow(['quiz_id','sample_id','question_en','answer_en','pred_text'])

        # 5. 遍历所有样本
        for s in samples:
            sid     = str(s['sample_id'])
            quiz_id = s.get('quiz_id','')
            q_en    = s['question_en']
            a_en    = s['answer_en']
            folder  = os.path.join(samples_dir, sid)
            if not os.path.isdir(folder):
                print(f"[{sid}] 文件夹不存在，跳过")
                continue

            # 收集 jpg 切片
            jpgs = glob.glob(os.path.join(folder, '*.jpg'))
            if len(jpgs)==0:
                print(f"[{sid}] 没有找到任何 .jpg，跳过")
                continue

            # 组合文本 & 体数据
            text, vision_x = combine_and_preprocess(
                q_en, jpgs, image_padding_tokens, device
            )

            # 文本 tokenize
            enc    = tokenizer(
                text,
                max_length=2048,
                truncation=True,
                return_tensors='pt'
            )
            lang_x = enc['input_ids'].to(device)

            # 推理
            with torch.no_grad():
                gen = model.generate(lang_x, vision_x)
                pred = tokenizer.batch_decode(gen, skip_special_tokens=True)[0]

            print(f"[{sid}] Q: {q_en}\n      Ref: {a_en}\n     Pred: {pred}")
            writer.writerow([quiz_id, sid, q_en, a_en, pred])

    print(">>> Done. 结果已写入", output_file)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--lang_model_path', required=True,
                   help='例如 ./Language_files')
    p.add_argument('--ckpt_path',       required=True,
                   help='例如 ./pytorch_model.bin')
    p.add_argument('--samples_dir',     required=True,
                   help='processed_samples 根目录')
    p.add_argument('--json_file',       required=True,
                   help='train.json 路径')
    p.add_argument('--output_file',     required=True,
                   help='输出 CSV 文件名')
    p.add_argument('--device', default='cuda')
    args = p.parse_args()

    main(
      args.lang_model_path,
      args.ckpt_path,
      args.samples_dir,
      args.json_file,
      args.output_file,
      args.device
    )


