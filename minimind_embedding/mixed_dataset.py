"""
minimind_embedding/mixed_dataset.py — 多源混合数据集(扩充 Stage1)

把多个中文语料转成统一的 (query, positive, negative) 三元组格式,
大幅扩充 Stage1 弱监督数据量。

数据源转换:
  * t2ranking triplet (9万)  —— 检索,已有格式
  * XNLI-zh (39万)           —— NLI: premise=query, entailment=正, contradiction=负
  * AFQMC (3.4万)            —— STS: score>0.6=正对, score<0.4=负对(跨query采样负)
  * STS-B-zh (0.5万)         —— 同 AFQMC

合计 ~50 万条,是原 Stage1(9万)的 ~5 倍。

用法:
  python -m minimind_embedding.mixed_dataset  # 预处理并缓存到 data/stage1_mixed.jsonl
"""
import os
import sys
import json
import random
from typing import List, Optional

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import torch
from torch.utils.data import Dataset


def convert_xnli_to_triplets(ds_split, max_samples: int = 0) -> List[dict]:
    """XNLI-zh → 三元组。

    XNLI 每条:premise(前提), hypothesis(假设), label(0=entailment蕴含,1=neutral,2=contradiction矛盾)
    转:(premise, entailment_hypothesis, contradiction_hypothesis)

    但 XNLI 原始结构是每个 premise 对应 3 条 hypothesis(label 0/1/2 各一)。
    我们把同一 premise 的 entailment 当 positive,contradiction 当 negative。
    """
    # 按 premise 分组
    from collections import defaultdict
    premise_map = defaultdict(lambda: {"entail": [], "contradict": []})
    for row in ds_split:
        p = row["premise"]
        h = row["hypothesis"]
        lab = row["label"]
        if lab == 0:  # entailment
            premise_map[p]["entail"].append(h)
        elif lab == 2:  # contradiction
            premise_map[p]["contradict"].append(h)

    samples = []
    for p, groups in premise_map.items():
        if groups["entail"] and groups["contradict"]:
            pos = random.choice(groups["entail"])
            neg = random.choice(groups["contradict"])
            samples.append({"query": p, "positive": pos, "negative": neg})
    if max_samples > 0:
        random.shuffle(samples)
        samples = samples[:max_samples]
    return samples


def convert_sts_to_triplets(ds_split, pos_thresh: float = 0.6,
                            neg_thresh: float = 0.3,
                            max_samples: int = 0) -> List[dict]:
    """STS 数据(AFQMC/STSB) → 三元组。

    STS 每条:sentence1, sentence2, score
    高分对(score>pos_thresh)作为正样本对。
    负样本:从其他 query 的 sentence2 里随机采样(score<neg_thresh 的优先)。
    """
    # 收集正样本对
    pos_pairs = [(r["sentence1"], r["sentence2"]) for r in ds_split
                 if float(r["score"]) >= pos_thresh]
    # 收集候选负样本(低分对的 sentence2)
    neg_pool = [r["sentence2"] for r in ds_split if float(r["score"]) <= neg_thresh]
    # 兜底:用所有 sentence2 作负样本池
    if len(neg_pool) < 100:
        neg_pool = [r["sentence2"] for r in ds_split]

    samples = []
    for q, pos in pos_pairs:
        neg = random.choice(neg_pool)
        # 确保负样本不等于正样本
        for _ in range(5):
            if neg != pos:
                break
            neg = random.choice(neg_pool)
        samples.append({"query": q, "positive": pos, "negative": neg})
    if max_samples > 0:
        random.shuffle(samples)
        samples = samples[:max_samples]
    return samples


def build_mixed_stage1(output_path: str, cache_dir: Optional[str] = None,
                       seed: int = 42):
    """构建混合 Stage1 数据集,保存为 jsonl。"""
    random.seed(seed)
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    from datasets import load_dataset

    all_samples = []

    # 1) t2ranking triplet (9万,已有 anchor/positive/negative)
    print("加载 t2ranking triplet ...")
    ds = load_dataset("sentence-transformers/t2ranking", "triplet", split="train", cache_dir=cache_dir)
    for r in ds:
        all_samples.append({"query": r["anchor"], "positive": r["positive"], "negative": r["negative"]})
    print(f"  t2ranking: {len(ds)} 条")

    # 2) XNLI-zh (39万,NLI 转三元组)
    print("加载 XNLI-zh ...")
    ds = load_dataset("facebook/xnli", "zh", split="train", cache_dir=cache_dir)
    xnli_samples = convert_xnli_to_triplets(ds)
    all_samples.extend(xnli_samples)
    print(f"  XNLI-zh: {len(xnli_samples)} 条(转三元组后)")

    # 3) AFQMC (3.4万,STS 转三元组)
    print("加载 AFQMC ...")
    try:
        ds = load_dataset("C-MTEB/AFQMC", split="train", cache_dir=cache_dir)
        afqmc_samples = convert_sts_to_triplets(ds)
        all_samples.extend(afqmc_samples)
        print(f"  AFQMC: {len(afqmc_samples)} 条(转三元组后)")
    except Exception as e:
        print(f"  AFQMC 跳过: {e}")

    # 4) STS-B-zh (0.5万,STS 转三元组)
    print("加载 STS-B-zh ...")
    try:
        ds = load_dataset("C-MTEB/STSB", split="train", cache_dir=cache_dir)
        stsb_samples = convert_sts_to_triplets(ds)
        all_samples.extend(stsb_samples)
        print(f"  STS-B-zh: {len(stsb_samples)} 条(转三元组后)")
    except Exception as e:
        print(f"  STS-B-zh 跳过: {e}")

    # 打乱并保存
    random.shuffle(all_samples)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for s in all_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"\n✓ 混合数据集构建完成: {output_path}")
    print(f"  总计 {len(all_samples)} 条(原 Stage1 的 {len(all_samples)/90467:.1f}x)")
    return len(all_samples)


class MixedTripletDataset(Dataset):
    """读取混合 Stage1 jsonl(格式同 TripletJsonlDataset)。"""

    def __init__(self, jsonl_path: str, max_negatives: int = 1):
        self.path = jsonl_path
        self.max_neg = max_negatives
        self.samples = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                q = obj.get("query", "")
                pos = obj.get("positive", "")
                neg = obj.get("negative", "")
                if q and pos:
                    self.samples.append({
                        "query": q, "positive": pos,
                        "negatives": [neg] if neg else [],
                    })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return s["query"], s["positive"], s["negatives"]


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--output", default=os.path.join(_ROOT, "data/stage1_mixed.jsonl"))
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    build_mixed_stage1(args.output, seed=args.seed)
