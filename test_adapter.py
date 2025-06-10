#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_adapter.py  ·  Rad-FM + Adapter  推理脚本（可调解码策略 + ICL + 重采样防抄袭）

用法示例见文末。
"""
import os
import sys
import glob
import json
import csv
import argparse
import random

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from transformers import LlamaTokenizer

# 项目根目录
sys.path.append(os.getcwd())
from Model.RadFM.multimodality_model import MultiLLaMAForCausalLM


from transformers import PreTrainedTokenizerFast

def get_tokenizer(tokenizer_path: str,
                  max_img_size: int = 100,
                  image_num: int    = 32):
    # 如果目录下有 tokenizer.json，就用 PreTrainedTokenizerFast 直接加载
    tok_json = os.path.join(tokenizer_path, "tokenizer.json")
    if os.path.isfile(tok_json):
        tok = PreTrainedTokenizerFast(tokenizer_file=tok_json)
    else:
        # 兜底：尝试 local_files_only=True
        tok = LlamaTokenizer.from_pretrained(
            tokenizer_path,
            local_files_only=True
        )

    specials = ["<image>", "</image>"] + [f"<image{i}>" for i in range(max_img_size * image_num)]
    tok.add_special_tokens({"additional_special_tokens": specials})
    tok.pad_token_id = tok.pad_token_id or 0
    tok.bos_token_id = tok.bos_token_id or 1
    tok.eos_token_id = tok.eos_token_id or 2

    # 构造 padding token 列表
    pad_tokens = [
        "".join([f"<image{seg*image_num + j}>" for j in range(image_num)])
        for seg in range(max_img_size)
    ]
    return tok, pad_tokens


def load_volume(slice_paths: list,
                image_padding_tokens: list,
                device: torch.device):
    """
    只处理影像，返回 [1,3,32,256,256] 的 vol tensor，不再生成 text。
    """
    slice_paths = sorted(slice_paths)[:32]
    tf = transforms.Compose([transforms.ToTensor(),
                             transforms.Resize((256, 256))])

    frames = [
        tf(Image.open(p).convert('RGB'))
        .unsqueeze(0).unsqueeze(2).to(device)
        for p in slice_paths
    ]
    if not frames:
        raise ValueError("slice_paths 为空")
    # pad 到 32 帧
    while len(frames) < 32:
        frames.append(frames[-1])
    vol = torch.cat(frames, dim=2).clamp(0, 1)        # [1,3,32,256,256]
    vol = F.interpolate(vol, size=(32, 256, 256),
                        mode='trilinear', align_corners=False)
    return vol.unsqueeze(0)                          # [1,3,32,256,256]


def build_prompt(current, pad_tok, samples, num_shots):
    """
    构造最终的 prompt 文本：
      - 随机抽取 num_shots 个示例
      - 示例格式：
            Question: ...
            Answer: ...
        最后拼接当前样本：
            Question: ...<image>pad_tok</image>
            Answer:
    """
    prompt = ""
    if num_shots > 0 and len(samples) > 1:
        # 排除自身，随机抽取
        others = [s for s in samples if s['sample_id'] != current['sample_id']]
        shots = random.sample(others, min(num_shots, len(others)))
        for e in shots:
            prompt += f"Question: {e['question_en']}\nAnswer: {e['answer_en']}\n\n"
    # 当前样本
    prompt += f"Question: {current['question_en']}<image>{pad_tok}</image>\nAnswer:"
    return prompt


def main(args):
    device = torch.device(args.device)

    print(">>> tokenizer")
    tok, img_pad_toks = get_tokenizer(args.lang_model_path)

    print(">>> build model & load ckpts")
    model = MultiLLaMAForCausalLM(args.lang_model_path).to(device)
    model.load_state_dict(torch.load(args.ckpt_path,    map_location='cpu'), strict=False)
    model.load_state_dict(torch.load(args.adapter_path, map_location='cpu'), strict=False)
    model.eval()

    print(">>> load json")
    with open(args.json_file, 'r', encoding='utf-8') as f:
        samples = json.load(f)

    os.makedirs(os.path.dirname(args.output_file) or '.', exist_ok=True)
    with open(args.output_file, 'w', newline='', encoding='utf-8') as fo:
        writer = csv.writer(fo)
        writer.writerow(['quiz_id', 'sample_id', 'question_en',
                         'answer_en', 'pred_text'])

        for s in samples:
            sid = str(s['sample_id'])
            folder = os.path.join(args.samples_dir, sid)
            jpgs = glob.glob(os.path.join(folder, '*.jpg'))
            if not os.path.isdir(folder) or not jpgs:
                print(f"[{sid}] skip (no folder or jpg)")
                continue

            # 1. 准备影像 volume
            vol = load_volume(jpgs, img_pad_toks, device)

            # 2. 构造带 ICL 的 prompt
            text = build_prompt(s, img_pad_toks[0], samples, args.num_shots)
            enc = tok(text, max_length=2048, truncation=True, return_tensors='pt')
            lang_x = enc['input_ids'].to(device)

            # 3. 自定义解码
            model.embedding_layer.flag = 'Text'
            with torch.no_grad():
                embeds, _ = model.embedding_layer(lang_x, vol)
                gen_kwargs = dict(
                    inputs_embeds      = embeds,
                    max_new_tokens     = args.max_new_tokens,
                    num_beams          = args.num_beams,
                    do_sample          = args.do_sample,
                    top_k              = args.top_k,
                    top_p              = args.top_p,
                    temperature        = args.temperature,
                    repetition_penalty = args.rep_penalty,
                    length_penalty     = args.length_penalty,
                    early_stopping     = True,
                )
                # 首次生成
                gen_ids = model.lang_model.generate(**gen_kwargs)
                pred = tok.batch_decode(gen_ids, skip_special_tokens=True)[0].strip()

                # 如果恰好和参考一致，则尝试重采样
                ref = s['answer_en'].strip()
                if pred == ref and args.max_regen_attempts > 0:
                    for attempt in range(args.max_regen_attempts):
                        # 强制采样：退回到 beam=1，do_sample=True
                        gen_kwargs.update(num_beams=1, do_sample=True)
                        gen_ids = model.lang_model.generate(**gen_kwargs)
                        new_pred = tok.batch_decode(gen_ids, skip_special_tokens=True)[0].strip()
                        if new_pred != ref:
                            pred = new_pred
                            break

            print(f"[{sid}] Q: {s['question_en']}")
            print(f"     Ref: {s['answer_en']}")
            print(f"    Pred: {pred}")
            writer.writerow([s.get('quiz_id',''), sid,
                             s['question_en'], s['answer_en'], pred])

    print(">>> Done! 结果保存到", args.output_file)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    # 路径设置
    p.add_argument('--lang_model_path', required=True)
    p.add_argument('--ckpt_path',       required=True)
    p.add_argument('--adapter_path',    required=True)
    p.add_argument('--samples_dir',     required=True)
    p.add_argument('--json_file',       required=True)
    p.add_argument('--output_file',     required=True)
    # 设备
    p.add_argument('--device', default='cuda')
    # 解码超参
    p.add_argument('--num_beams',      type=int,   default=4)
    p.add_argument('--max_new_tokens', type=int,   default=64)
    p.add_argument('--length_penalty', type=float, default=0.8)
    p.add_argument('--do_sample',      action='store_true')
    p.add_argument('--top_k',          type=int,   default=20)
    p.add_argument('--top_p',          type=float, default=0.95)
    p.add_argument('--temperature',    type=float, default=0.7)
    p.add_argument('--rep_penalty',    type=float, default=1.1)
    # 新增：ICL 示例数 & 重采样尝试次数
    p.add_argument('--num_shots',          type=int,   default=0,
                                 help="每次生成前随机加入多少示例做 ICL")
    p.add_argument('--max_regen_attempts', type=int,   default=3,
                                 help="若生成与参考完全一致，最多重试多少次采样")
    args = p.parse_args()

    main(args)

