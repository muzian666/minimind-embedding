"""
scripts/eval_mteb.py — MTEB v2 评测脚本

把训练好的 MiniMind-Embedding 模型包装成 mteb v2 的 EncoderProtocol,跑 C-MTEB 任务。

mteb v2 接口要点(2.18.7):
  * 模型需 isinstance(model, EncoderProtocol) —— 通过继承协议类(Protocol)满足
  * encode 签名:
      encode(inputs: DataLoader[BatchedInput], *, task_metadata, hf_split,
             hf_subset, prompt_type=None, **kwargs) -> Array
    其中 BatchedInput 是 dict-like,含 "text" 字段(list[str])
  * DataLoader[BatchedInput] 可迭代,每次 yield 一个 batch(dict)

用法:
  python scripts/eval_mteb.py \\
      --checkpoint checkpoints/embedding/embedding_stage2_768.pth \\
      --config embed_dense_64m \\
      --tasks ATEC,BQ,LCQMC,STSB,TNews
"""
import os
import sys
import argparse
import time
import json
from typing import Any

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
import torch

from shared import load_tokenizer, get_config
from shared.tokenizer import build_embedding_inputs, DEFAULT_QUERY_INSTRUCTION
from shared.model import MiniMindForEmbedding

# mteb v2 的协议类(用于 isinstance 检查)
from mteb.models.models_protocols import EncoderProtocol


class MiniMindMTEBEncoder(EncoderProtocol):
    """把 MiniMindForEmbedding 包装成 mteb v2 Encoder。

    继承 EncoderProtocol 以通过 isinstance 检查(协议类,runtime 可继承)。
    实现 v2 的 encode(inputs: DataLoader[BatchedInput], ...) 接口。
    """

    def __init__(self, checkpoint_path: str, config_name: str = "embed_dense_64m",
                 device: str = "cuda", max_length: int = 512, batch_size: int = 32):
        self.device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
        self.max_length = max_length
        self.batch_size = batch_size

        config = get_config(config_name)
        self.model = MiniMindForEmbedding(config)
        if checkpoint_path and os.path.exists(checkpoint_path):
            state = torch.load(checkpoint_path, map_location="cpu")
            missing, unexpected = self.model.load_state_dict(state, strict=False)
            print(f"[MiniMindMTEBEncoder] 加载 {checkpoint_path} (missing={len(missing)}, unexpected={len(unexpected)})")
        else:
            print(f"[MiniMindMTEBEncoder] 警告:未加载 checkpoint,用随机权重!")
        self.model.to(self.device).eval()

        self.tokenizer = load_tokenizer(padding_side="left")
        # 注:mteb_model_meta 在 EncoderProtocol 里是只读 property,评测阶段不需要设置,
        #     仅在 HF 打包上传(M4)时通过 ModelMeta 注册。这里跳过。

    def encode(
        self,
        inputs,  # DataLoader[BatchedInput] —— 可迭代,每次 yield 一个 batch dict
        *,
        task_metadata=None,
        hf_split="test",
        hf_subset="default",
        prompt_type=None,
        **kwargs,
    ) -> np.ndarray:
        """编码,返回 [N, D] numpy(已 L2 归一化)。"""
        # prompt_type: mteb 对 retrieval 任务的 query 传 "query",passage 不传
        is_query = prompt_type in ("query", "Question")
        instruction = DEFAULT_QUERY_INSTRUCTION if is_query else None

        all_vecs = []
        for batch in inputs:
            # BatchedInput 是 dict,文本在 "text" 字段(可能是 list 或 dict)
            texts = self._extract_texts(batch)
            for i in range(0, len(texts), self.batch_size):
                chunk = texts[i:i + self.batch_size]
                enc = build_embedding_inputs(
                    self.tokenizer, chunk, is_query=is_query,
                    instruction=instruction, max_length=self.max_length,
                )
                ids = enc["input_ids"].to(self.device)
                mask = enc["attention_mask"].to(self.device)
                with torch.inference_mode():
                    z = self.model(ids, attention_mask=mask, output_mrl=False)
                all_vecs.append(z.cpu().float().numpy())

        if not all_vecs:
            return np.zeros((0, self.model.config.embed_dim), dtype=np.float32)
        return np.concatenate(all_vecs, axis=0)

    def _extract_texts(self, batch) -> list:
        """从 mteb 的 BatchedInput(dict)中提取文本列表。"""
        if isinstance(batch, dict):
            for key in ("text", "texts", "sentence", "sentences"):
                if key in batch:
                    val = batch[key]
                    return list(val) if not isinstance(val, str) else [val]
            # 兜底:取第一个 list 值
            for v in batch.values():
                if isinstance(v, (list, tuple)):
                    return list(v)
                if isinstance(v, str):
                    return [v]
        if isinstance(batch, (list, tuple)):
            return list(batch)
        if isinstance(batch, str):
            return [batch]
        return []


# ---------------------------------------------------------------------------
# 任务分组
# ---------------------------------------------------------------------------
TASK_GROUPS = {
    "STS": ["ATEC", "BQ", "LCQMC", "PAWSX", "STSB", "AFQMC", "QBQTC"],
    "Retrieval-zh": ["T2Retrieval", "MMarcoRetrieval", "DuRetrieval"],
    "Classification-zh": ["TNews", "IFlyTek", "MultilingualSentiment", "JDReview"],
    "C-MTEB-fast": ["ATEC", "BQ", "LCQMC", "STSB", "TNews"],
}


def parse_args():
    p = argparse.ArgumentParser(description="MiniMind-Embedding MTEB 评测")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--config", default="embed_dense_64m",
                   choices=["embed_dense_64m", "embed_moe_198m"])
    p.add_argument("--tasks", default="C-MTEB-fast",
                   help="任务组名(STS/Retrieval-zh/Classification-zh/C-MTEB-fast)或逗号分隔的任务名")
    p.add_argument("--output_folder", default="results/eval")
    p.add_argument("--device", default="cuda")
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=64)
    return p.parse_args()


def resolve_tasks(tasks_arg: str):
    if tasks_arg in TASK_GROUPS:
        return TASK_GROUPS[tasks_arg]
    return [t.strip() for t in tasks_arg.split(",") if t.strip()]


def main():
    args = parse_args()
    print(f"== MiniMind-Embedding MTEB 评测 ==")
    print(f"checkpoint: {args.checkpoint}")

    import mteb

    encoder = MiniMindMTEBEncoder(
        checkpoint_path=args.checkpoint,
        config_name=args.config,
        device=args.device,
        max_length=args.max_length,
        batch_size=args.batch_size,
    )

    task_names = resolve_tasks(args.tasks)
    print(f"任务列表: {task_names}")

    tasks = mteb.get_tasks(tasks=task_names)
    print(f"解析到 {len(tasks)} 个任务: {[t.metadata.name for t in tasks]}")
    if not tasks:
        print("❌ 没有解析到任务"); return

    os.makedirs(args.output_folder, exist_ok=True)
    print(f"开始评测...")

    try:
        model_result = mteb.evaluate(
            encoder, tasks=tasks,
            prediction_folder=os.path.join(args.output_folder, "predictions"),
            raise_error=True,
            show_progress_bar=True,
        )
    except TypeError as e:
        print(f"evaluate 参数问题,fallback: {e}")
        model_result = mteb.evaluate(encoder, tasks=tasks)

    # 汇总分数
    print("\n" + "=" * 60)
    print("评测结果汇总:")
    print("=" * 60)
    try:
        result_dict = model_result.to_dict()
    except Exception:
        result_dict = model_result if isinstance(model_result, dict) else {}

    scores_summary = []
    for task_name, task_res in result_dict.items():
        if isinstance(task_res, dict):
            scores = task_res.get("scores", {})
            for split, split_scores in scores.items():
                if isinstance(split_scores, dict):
                    for subset, sc in split_scores.items():
                        if isinstance(sc, dict) and "main_score" in sc:
                            ms = sc["main_score"]
                            tag = f"[{split}/{subset}]" if subset != "default" else f"[{split}]"
                            print(f"  {task_name} {tag}: {ms:.4f}")
                            scores_summary.append((task_name, split, subset, ms))

    summary_path = os.path.join(args.output_folder, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({"scores": scores_summary, "checkpoint": args.checkpoint}, f, ensure_ascii=False, indent=2)
    print(f"\n汇总已保存: {summary_path}")


if __name__ == "__main__":
    main()
