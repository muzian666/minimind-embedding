"""
scripts/eval_sts_simple.py — 轻量 STS 评测(绕过 mteb fast_load 兼容问题)

mteb v2.18.7 的 fast_load 对 C-MTEB 老任务(ATEC/BQ/LCQMC 等)有 "lang 列缺失" bug,
导致 evaluate() 内部数据加载失败。本脚本绕过 mteb 的 evaluate 框架,直接:
  1. 用 datasets 拉 STS 数据(ATEC/BQ/LCQMC/STSB 的 test split)
  2. 自己调 model.encode 编码 sentence1/sentence2
  3. 算 Spearman 相关性(标准 STS 指标)

这样能快速验证整个 embedding→STS pipeline 出分数。正式 mteb 集成(M2 全量阶段)
再修 fast_load 问题或等 mteb 更新。

用法:
  python scripts/eval_sts_simple.py \\
      --checkpoint checkpoints/embedding/embedding_stage2_768.pth \\
      --tasks ATEC,BQ,LCQMC
"""
import os
import sys
import argparse
import json

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
import torch
from scipy.stats import spearmanr

from shared import load_tokenizer, get_config
from shared.tokenizer import build_embedding_inputs, DEFAULT_QUERY_INSTRUCTION
from shared.model import MiniMindForEmbedding


# C-MTEB STS 任务 → HuggingFace 仓库映射(直接用 datasets 加载,绕过 mteb)
STS_DATASETS = {
    "ATEC": ("C-MTEB/ATEC", None),
    "BQ": ("C-MTEB/BQ", None),
    "LCQMC": ("C-MTEB/LCQMC", None),
    "PAWSX": ("C-MTEB/PAWSX", "zh"),
    "STSB": ("C-MTEB/STSB", None),
    "AFQMC": ("C-MTEB/AFQMC", None),
    "QBQTC": ("C-MTEB/QBQTC", None),
}


class Embedder:
    """轻量封装:加载模型,提供 encode(texts) -> np.ndarray。"""

    def __init__(self, checkpoint, config_name, device="cuda", max_length=256, batch_size=64):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        self.batch_size = batch_size

        config = get_config(config_name)
        self.model = MiniMindForEmbedding(config)
        if checkpoint and os.path.exists(checkpoint):
            state = torch.load(checkpoint, map_location="cpu")
            self.model.load_state_dict(state, strict=False)
            print(f"[Embedder] 加载 {checkpoint}")
        self.model.to(self.device).eval()
        self.tokenizer = load_tokenizer(padding_side="left")

    @torch.inference_mode()
    def encode(self, texts, is_query=False):
        vecs = []
        for i in range(0, len(texts), self.batch_size):
            chunk = texts[i:i + self.batch_size]
            enc = build_embedding_inputs(
                self.tokenizer, chunk, is_query=is_query,
                instruction=DEFAULT_QUERY_INSTRUCTION if is_query else None,
                max_length=self.max_length,
            )
            ids = enc["input_ids"].to(self.device)
            mask = enc["attention_mask"].to(self.device)
            z = self.model(ids, attention_mask=mask, output_mrl=False)
            vecs.append(z.cpu().float().numpy())
        return np.concatenate(vecs, axis=0) if vecs else np.zeros((0, 768), dtype=np.float32)


def load_sts_split(task_name, split="test"):
    """加载 STS 任务的 test split,返回 (sentence1_list, sentence2_list, scores_list)。"""
    from datasets import load_dataset
    if task_name not in STS_DATASETS:
        raise ValueError(f"未知 STS 任务 {task_name},支持: {list(STS_DATASETS.keys())}")
    repo, config = STS_DATASETS[task_name]
    print(f"  加载 {repo} (config={config}) split={split} ...")
    ds = load_dataset(repo, config) if config else load_dataset(repo)

    # 取目标 split(优先 test,没有则 validation)
    if split in ds:
        data = ds[split]
    elif "validation" in ds:
        data = ds["validation"]
        print(f"    (无 {split},用 validation)")
    else:
        data = ds[list(ds.keys())[0]]

    s1 = data["sentence1"]
    s2 = data["sentence2"]
    scores = data["score"]
    # 归一化分数到 [0,1](部分任务的 score 范围不同,如 0-5 或 0-1)
    scores_arr = np.array(scores, dtype=np.float32)
    if scores_arr.max() > 1.0:
        scores_arr = scores_arr / scores_arr.max()
    return s1, s2, scores_arr.tolist()


def eval_sts_task(embedder, task_name):
    """评测单个 STS 任务,返回 Spearman 相关系数。"""
    s1, s2, gold = load_sts_split(task_name)
    print(f"    样本数: {len(s1)}")

    e1 = embedder.encode(s1)
    e2 = embedder.encode(s2)

    # cosine 相似度(向量已 L2 归一化,直接点积)
    cos = (e1 * e2).sum(axis=1)
    # Spearman 相关性
    rho, _ = spearmanr(cos, gold)
    return rho


def parse_args():
    p = argparse.ArgumentParser(description="轻量 STS 评测")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--config", default="embed_dense_64m")
    p.add_argument("--tasks", default="ATEC,BQ,LCQMC,STSB",
                   help="逗号分隔的 STS 任务名")
    p.add_argument("--device", default="cuda")
    p.add_argument("--max_length", type=int, default=128)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--output", default="results/sts_scores.json")
    return p.parse_args()


def main():
    args = parse_args()
    print(f"== MiniMind-Embedding STS 评测(轻量版)==")
    print(f"checkpoint: {args.checkpoint}")
    print(f"tasks: {args.tasks}")

    embedder = Embedder(
        checkpoint=args.checkpoint, config_name=args.config,
        device=args.device, max_length=args.max_length, batch_size=args.batch_size,
    )

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    results = {}
    print(f"\n{'任务':<12} {'Spearman':<12}")
    print("-" * 26)
    rhos = []
    for task in tasks:
        try:
            rho = eval_sts_task(embedder, task)
            results[task] = float(rho)
            rhos.append(float(rho))
            print(f"{task:<12} {rho:.4f}")
        except Exception as e:
            print(f"{task:<12} 失败: {type(e).__name__}: {str(e)[:60]}")
            results[task] = None

    if rhos:
        avg = sum(rhos) / len(rhos)
        print("-" * 26)
        print(f"{'平均':<12} {avg:.4f}")
        results["_average"] = avg

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump({"checkpoint": args.checkpoint, "scores": results}, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存: {args.output}")


if __name__ == "__main__":
    main()
