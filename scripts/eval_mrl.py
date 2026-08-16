"""
scripts/eval_mrl.py — MRL 多维度评测

测试 Matryoshka 表示学习的效果:同一个模型输出截断到不同维度
(768/512/256/128/64)后,STS 分数的变化。MRL 训练的目标就是让
截断后的向量仍保持良好语义。

用法:
  python scripts/eval_mrl.py \\
      --checkpoint checkpoints/embedding/embedding_stage3_768.pth \\
      --config embed_dense_64m \\
      --dims 768 512 256 128 64 \\
      --tasks ATEC,BQ,LCQMC,STSB \\
      --device cuda
"""
import os
import sys
import json
import argparse

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr

from shared import load_tokenizer, get_config
from shared.tokenizer import build_embedding_inputs, DEFAULT_QUERY_INSTRUCTION
from shared.model import MiniMindForEmbedding


class MRLEmbedder:
    """加载模型,支持输出指定维度的截断向量。"""

    def __init__(self, checkpoint, config_name, device="cuda", max_length=128, batch_size=64):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        self.batch_size = batch_size

        config = get_config(config_name)
        self.model = MiniMindForEmbedding(config)
        if checkpoint and os.path.exists(checkpoint):
            state = torch.load(checkpoint, map_location="cpu")
            self.model.load_state_dict(state, strict=False)
            print(f"[MRL] 加载 {checkpoint}")
        self.model.to(self.device).eval()
        self.tokenizer = load_tokenizer(padding_side="left")

    @torch.inference_mode()
    def encode(self, texts, is_query=False):
        """编码,返回完整维度向量 [N, full_dim](已 L2 归一化)。"""
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

    def encode_truncated(self, texts, dim, is_query=False):
        """编码并截断到 dim 维,重新 L2 归一化(模拟 MRL 推理)。"""
        full = self.encode(texts, is_query=is_query)
        truncated = full[:, :dim]
        # 截断后重新归一化(MRL 推理标准做法)
        norms = np.linalg.norm(truncated, axis=1, keepdims=True) + 1e-12
        return truncated / norms


STS_DATASETS = {
    "ATEC": ("C-MTEB/ATEC", None),
    "BQ": ("C-MTEB/BQ", None),
    "LCQMC": ("C-MTEB/LCQMC", None),
    "STSB": ("C-MTEB/STSB", None),
}


def load_sts_split(task_name, split="test"):
    from datasets import load_dataset
    repo, config = STS_DATASETS[task_name]
    ds = load_dataset(repo, config) if config else load_dataset(repo)
    if split in ds:
        data = ds[split]
    elif "validation" in ds:
        data = ds["validation"]
    else:
        data = ds[list(ds.keys())[0]]
    s1 = data["sentence1"]
    s2 = data["sentence2"]
    scores = np.array(data["score"], dtype=np.float32)
    if scores.max() > 1.0:
        scores = scores / scores.max()
    return s1, s2, scores.tolist()


def eval_sts_all_dims(embedder, task_name, dims):
    """某任务一次编码,评测所有维度的 STS Spearman。

    编码是最贵的操作(每个样本过一遍 transformer),而 MRL 截断只是
    numpy 切片。所以先编码缓存完整向量,再对每个维度切片评测。
    """
    s1, s2, gold = load_sts_split(task_name)
    e1_full = embedder.encode(s1, is_query=False)   # [N, 768]
    e2_full = embedder.encode(s2, is_query=False)
    out = {}
    for dim in dims:
        t1 = e1_full[:, :dim]
        t2 = e2_full[:, :dim]
        # 截断后重新 L2 归一化(MRL 推理标准做法)
        t1 = t1 / (np.linalg.norm(t1, axis=1, keepdims=True) + 1e-12)
        t2 = t2 / (np.linalg.norm(t2, axis=1, keepdims=True) + 1e-12)
        cos = (t1 * t2).sum(axis=1)
        rho, _ = spearmanr(cos, gold)
        out[dim] = float(rho)
    return out


def parse_args():
    p = argparse.ArgumentParser(description="MRL 多维度评测")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--config", default="embed_dense_64m")
    p.add_argument("--dims", nargs="+", type=int, default=[768, 512, 256, 128, 64])
    p.add_argument("--tasks", default="ATEC,BQ,LCQMC,STSB")
    p.add_argument("--device", default="cuda")
    p.add_argument("--max_length", type=int, default=128)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--output", default="results/mrl_eval.json")
    return p.parse_args()


def main():
    args = parse_args()
    print(f"== MRL 多维度评测 ==")
    print(f"checkpoint: {args.checkpoint}")
    print(f"dims: {args.dims}")

    embedder = MRLEmbedder(
        checkpoint=args.checkpoint, config_name=args.config,
        device=args.device, max_length=args.max_length, batch_size=args.batch_size,
    )

    tasks = [t.strip() for t in args.tasks.split(",")]
    results = {}  # {dim: {task: score}}

    # 每任务编码一次,得到所有维度的分数 {task: {dim: rho}}
    task_scores = {}
    for task in tasks:
        print(f"\n编码 {task} ...", flush=True)
        task_scores[task] = eval_sts_all_dims(embedder, task, args.dims)

    print(f"\n{'维度':<8}", end="")
    for t in tasks:
        print(f"{t:<12}", end="")
    print(f"{'平均':<10}")
    print("-" * (8 + 12 * len(tasks) + 10))

    for dim in args.dims:
        results[dim] = {}
        print(f"{dim:<8}", end="")
        scores = []
        for task in tasks:
            rho = task_scores[task][dim]
            results[dim][task] = rho
            scores.append(rho)
            print(f"{rho:<12.4f}", end="")
        avg = sum(scores) / len(scores)
        results[dim]["_average"] = avg
        print(f"{avg:<10.4f}")

    # 保存
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump({"checkpoint": args.checkpoint, "dims": args.dims, "results": results}, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存: {args.output}")

    # 总结:MRL 保持率(各维度相对 768 的百分比)
    if 768 in results:
        base = results[768]["_average"]
        print(f"\n=== MRL 保持率(相对 768 维 baseline)===")
        for dim in args.dims:
            r = results[dim]["_average"]
            print(f"  {dim:>4} 维: {r:.4f} ({r/base*100:.1f}%)")
    elif args.dims:
        base = results[max(args.dims)]["_average"]
        print(f"\n(无 768 维 baseline,以最大维度 {max(args.dims)} 为基准)")


if __name__ == "__main__":
    main()
