#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from transformers import LlamaForCausalLM, LlamaTokenizer

def main():
    # 改成你本地存放 tokenizer + model 的目录
    model_dir = "./Language_files"

    print(f">>> Loading tokenizer from {model_dir}")
    tokenizer = LlamaTokenizer.from_pretrained(model_dir,
                                                local_files_only=True)
    print(f">>> Loading model from {model_dir}")
    model = LlamaForCausalLM.from_pretrained(model_dir,
                                             local_files_only=True).cuda().eval()

    prompt = "Q: What is your name?\nA:"
    inputs = tokenizer(prompt,
                       return_tensors="pt").to("cuda")

    print(">>> Generating text …")
    out = model.generate(
        **inputs,
        max_new_tokens=20,
        num_beams=2,
        temperature=0.7,
        top_p=0.9,
        eos_token_id=tokenizer.eos_token_id,
    )

    print(">>> Output:")
    print(tokenizer.decode(out[0], skip_special_tokens=True))

if __name__ == "__main__":
    main()
