"""
scripts/eval_rerank_simple.py — 轻量 Rerank 评测

绕过 mteb v2 的 fast_load 兼容问题,直接:
  1. 用 datasets 从 hf-mirror 拉 C-MTEB/T2Reranking
  2. 对每个 query,用模型对其所有候选 doc 打分(yes 概率)
  3. 按 score 降序排,算 MAP@10(Mean Average Precision,C-MTEB 标准)

用法:
  python scripts/eval_rerank_simple.py \\
      --checkpoint checkpoints/rerank/rerank_768.pth \\
      --config rerank_dense_64m \\
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

from shared import load_tokenizer, get_config
from shared.tokenizer import build_rerank_inputs
from shared.model import MiniMindForRerank


class Reranker:
    """加载 rerank 模型,提供 score(queries, docs) -> np.ndarray。"""

    def __init__(self, checkpoint, config_name, device="cuda", max_length=512, batch_size=64):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        self.batch_size = batch_size

        config = get_config(config_name)
        self.model = MiniMindForRerank(config)
        if checkpoint and os.path.exists(checkpoint):
            state = torch.load(checkpoint, map_location="cpu")
            self.model.load_state_dict(state, strict=False)
            print(f"[Reranker] 加载 {checkpoint}")
        self.model.to(self.device).eval()
        self.tokenizer = load_tokenizer(padding_side="left")

    @torch.inference_mode()
    def score(self, queries, documents):
        """对 (query, doc) 对打分,返回相关性分数 [N]。

        用 logit 差值(yes_logit - no_logit)而非 softmax 概率。
        原因:softmax 会压缩区分度(当 yes/no logit 都很大时概率差异变小),
              而 logit 差值保留了原始的排序信号,对 rerank 更有效。
        """
        scores = []
        for i in range(0, len(queries), self.batch_size):
            qs = queries[i:i + self.batch_size]
            ds = documents[i:i + self.batch_size]
            enc = build_rerank_inputs(self.tokenizer, qs, ds, max_length=self.max_length)
            ids = enc["input_ids"].to(self.device)
            mask = enc["attention_mask"].to(self.device)
            # forward 返回 [B, 2] (yes_logit, no_logit),用差值作分数
            yn_logits = self.model(ids, attention_mask=mask)
            logit_diff = (yn_logits[:, 0] - yn_logits[:, 1]).cpu().float().numpy()
            scores.append(logit_diff)
        return np.concatenate(scores) if scores else np.zeros(0)


def compute_map_at_k(ranked_relevance, k=10):
    """单 query 的 AP@k。ranked_relevance: 按预测分数降序排列的 relevance(0/1)列表。"""
    if not ranked_relevance:
        return 0.0
    hits, sum_prec = 0, 0.0
    for i, rel in enumerate(ranked_relevance[:k]):
        if rel:
            hits += 1
            sum_prec += hits / (i + 1)
    n_rel = sum(ranked_relevance)
    return sum_prec / min(n_rel, k) if n_rel > 0 else 0.0


def eval_t2reranking(reranker, split_ratio=0.0, split_part="test"):
    """评测 C-MTEB/T2Reranking。返回 MAP@10。

    split_ratio>0 时,只评测切分后的指定部分(避免与训练数据重叠)。
    """
    from datasets import load_dataset
    print("  加载 C-MTEB/T2Reranking ...")
    ds = load_dataset("C-MTEB/T2Reranking")["dev"]
    if split_ratio > 0 and split_ratio < 1:
        n = len(ds)
        cut = int(n * split_ratio)
        if split_part == "test":
            ds = ds.select(range(cut, n))
            print(f"  测试集切分: 后 {n-cut}/{n} 条(避免泄露)")
        else:
            ds = ds.select(range(0, cut))
    print(f"  样本(query)数: {len(ds)}")

    # 用 ast.literal_eval 解析 list 字段(positive/negative 是字符串形式 list)
    import ast
    def to_list(v):
        if isinstance(v, list): return v
        try: return ast.literal_eval(v)
        except: return [v] if isinstance(v, str) else []

    ap_list = []
    for idx in range(len(ds)):
        row = ds[idx]
        q = row["query"]
        pos = to_list(row["positive"])
        neg = to_list(row["negative"])
        if not pos or not neg:
            continue
        # 构造候选:正样本在前(标1),负样本在后(标0)
        docs = pos + neg
        rels = [1] * len(pos) + [0] * len(neg)
        # 打分
        queries = [q] * len(docs)
        scores = reranker.score(queries, docs)
        # 按分数降序排
        order = np.argsort(-scores)
        ranked_rel = [rels[i] for i in order]
        ap_list.append(compute_map_at_k(ranked_rel, k=10))
        if (idx + 1) % 1000 == 0:
            print(f"    已评 {idx+1}/{len(ds)}, 当前 MAP@10={np.mean(ap_list):.4f}")

    map10 = float(np.mean(ap_list))
    return map10, len(ap_list)


def parse_args():
    p = argparse.ArgumentParser(description="轻量 Rerank 评测")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--config", default="rerank_dense_64m",
                   choices=["rerank_dense_64m", "rerank_moe_198m"])
    p.add_argument("--device", default="cuda")
    p.add_argument("--max_length", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--output", default="results/rerank_scores.json")
    p.add_argument("--split_ratio", type=float, default=0.0,
                   help="与训练一致的切分比例(>0 时只评测 test 部分,避免泄露)")
    return p.parse_args()


def main():
    args = parse_args()
    print(f"== MiniMind-Rerank 评测(轻量版)==")
    print(f"checkpoint: {args.checkpoint}")

    reranker = Reranker(
        checkpoint=args.checkpoint, config_name=args.config,
        device=args.device, max_length=args.max_length, batch_size=args.batch_size,
    )

    map10, n = eval_t2reranking(reranker, split_ratio=args.split_ratio, split_part="test")
    print("\n" + "=" * 40)
    print(f"C-MTEB/T2Reranking MAP@10: {map10:.4f}  (n={n})")
    print("=" * 40)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump({"checkpoint": args.checkpoint, "T2Reranking_MAP@10": map10, "n": n}, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存: {args.output}")


if __name__ == "__main__":
    main()
