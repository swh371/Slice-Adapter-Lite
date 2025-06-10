#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluate_per_sample.py

读取 results.csv（含 quiz_id, sample_id, question_en, answer_en, pred_text），
对每个样本计算 BLEU-1/2/3/4、METEOR、ROUGE-L、CIDEr，将结果写入 output_csv。
依赖：nltk、rouge-score、pycocoevalcap（仅 CIDEr）、numpy
"""

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Compute per-sample BLEU / METEOR / ROUGE-L / CIDEr
with text normalisation, synonym mapping and smoothing.
"""

import argparse
import csv
import re
import string
from collections import defaultdict

import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from pycocoevalcap.cider.cider import Cider


# ---------- 文本预处理 ---------- #

PUNCT = re.compile(rf"[{re.escape(string.punctuation)}]")
MULTISPACE = re.compile(r"\s+")

# 可自行扩充
SYNONYMS = {
    "none": "no",
    "no.": "no",
    "effaced": "compressed",
    "obscured": "compressed",
    "high": "elevated",
    "low": "decreased",
    "diffuse": "widespread",
    # ...
}

def normalise(txt: str) -> str:
    """大小写统一 + 去标点 + 合并空格 + 同义词替换"""
    txt = txt.lower()
    txt = PUNCT.sub("", txt)
    txt = MULTISPACE.sub(" ", txt).strip()
    # 同义词逐词映射
    tokens = [SYNONYMS.get(t, t) for t in txt.split()]
    return " ".join(tokens)


# ---------- 读取数据 ---------- #

def load_rows(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# ---------- 指标计算 ---------- #

smooth = SmoothingFunction().method4     # 更稳健的平滑

def compute_metrics_per_sample(hyps, refs):
    N = len(hyps)

    bleu1 = []; bleu2 = []; bleu3 = []; bleu4 = []
    meteor_scores = []; rouge_l = []

    rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

    # 先把文本统一好
    hyps_norm = [normalise(h) for h in hyps]
    refs_norm = [normalise(r) for r in refs]

    # BLEU / METEOR / ROUGE-L
    for h, r in zip(hyps_norm, refs_norm):
        if len(h) == 0 or len(r) == 0:       # 空串保护
            bleu1.append(0); bleu2.append(0); bleu3.append(0); bleu4.append(0)
            meteor_scores.append(0); rouge_l.append(0)
            continue
        h_tok = h.split()
        r_tok = r.split()
        bleu1.append(sentence_bleu([r_tok], h_tok, weights=(1,0,0,0),     smoothing_function=smooth)*100)
        bleu2.append(sentence_bleu([r_tok], h_tok, weights=(0.5,0.5,0,0), smoothing_function=smooth)*100)
        bleu3.append(sentence_bleu([r_tok], h_tok, weights=(1/3,)*3+(0,), smoothing_function=smooth)*100)
        bleu4.append(sentence_bleu([r_tok], h_tok, weights=(0.25,)*4,     smoothing_function=smooth)*100)

        meteor_scores.append(meteor_score([r_tok], h_tok) * 100)             # 直接传字符串
        rouge_l.append(rouge.score(r, h)['rougeL'].fmeasure*100)

    # CIDEr（整合时用归一化文本）
    gts = {i:[refs_norm[i]] for i in range(N)}
    res = {i:[hyps_norm[i]] for i in range(N)}
    cider_scorer = Cider()
    _, cider_scores = cider_scorer.compute_score(gts, res)

    return {
        'BLEU-1': bleu1, 'BLEU-2': bleu2, 'BLEU-3': bleu3, 'BLEU-4': bleu4,
        'METEOR': meteor_scores, 'ROUGE-L': rouge_l, 'CIDEr': cider_scores
    }


# ---------- 主流程 ---------- #

def main(input_csv, output_csv):
    rows = load_rows(input_csv)
    hyps = [r['pred_text'].strip()  for r in rows]
    refs = [r['answer_en'].strip() for r in rows]

    metrics = compute_metrics_per_sample(hyps, refs)

    fieldnames = list(rows[0].keys()) + ['BLEU-1','BLEU-2','BLEU-3','BLEU-4','METEOR','ROUGE-L','CIDEr']
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(fieldnames)
        for i, row in enumerate(rows):
            writer.writerow([
                *[row[k] for k in rows[0].keys()],
                f"{metrics['BLEU-1'][i]:.2f}",
                f"{metrics['BLEU-2'][i]:.2f}",
                f"{metrics['BLEU-3'][i]:.2f}",
                f"{metrics['BLEU-4'][i]:.2f}",
                f"{metrics['METEOR'][i]:.2f}",
                f"{metrics['ROUGE-L'][i]:.2f}",
                f"{metrics['CIDEr'][i]:.4f}"
            ])

    print(f"[✓] Done! Per-sample metrics written to {output_csv}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Compute per-sample BLEU / METEOR / ROUGE-L / CIDEr (with normalisation & smoothing)"
    )
    parser.add_argument('--input_csv',  required=True, help="原始结果 CSV")
    parser.add_argument('--output_csv', required=True, help="输出带指标的 CSV")
    args = parser.parse_args()
    main(args.input_csv, args.output_csv)


