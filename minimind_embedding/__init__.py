"""minimind_embedding — 基于 minimind 底座的对比学习 Embedding 模型。

路线:Qwen3-Embedding
  causal decoder-only + last-token pooling + InfoNCE(带假负样本 mask) + MRL
"""
from minimind_embedding.dataset import (
    TripletJsonlDataset,
    HFEmbeddingDataset,
    EmbeddingCollator,
    write_demo_jsonl,
)

__all__ = [
    "TripletJsonlDataset",
    "HFEmbeddingDataset",
    "EmbeddingCollator",
    "write_demo_jsonl",
]
