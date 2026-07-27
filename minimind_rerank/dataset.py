"""
minimind_rerank/dataset.py — Rerank 训练数据加载

统一数据格式(每行一个 JSON):
    {"query": str, "document": str, "label": int}   # label: 1=相关, 0=不相关

支持来源:
  1. 本地 jsonl
  2. HuggingFace datasets:
     - C-MTEB/T2Reranking (query, positive, negative → 转 pairwise)
     - sentence-transformers/msmarco labeled-list
     - unicamp-dl/mmarco (parquet, query/positive/negative)
"""
import os
import json
import random
from typing import Optional, List

import torch
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# 本地 jsonl 数据集
# ---------------------------------------------------------------------------
class RerankJsonlDataset(Dataset):
    """读取 {query, document, label} jsonl。"""

    def __init__(self, jsonl_path: str):
        self.path = jsonl_path
        self.samples: List[dict] = []
        if not os.path.exists(jsonl_path):
            raise FileNotFoundError(f"数据文件不存在: {jsonl_path}")
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                q = obj.get("query") or obj.get("q") or ""
                d = obj.get("document") or obj.get("doc") or obj.get("positive") or obj.get("passage") or ""
                lab = obj.get("label", obj.get("score", obj.get("relevant", 0)))
                if not q or not d:
                    continue
                # label 归一化为 0/1(>0 视为相关)
                lab = 1 if (int(lab) if str(lab).lstrip("-").isdigit() else 0) > 0 else 0
                self.samples.append({"query": q, "document": d, "label": lab})

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return s["query"], s["document"], s["label"]


# ---------------------------------------------------------------------------
# HF datasets 适配器
# ---------------------------------------------------------------------------
class HFRerankDataset(Dataset):
    """从 HF datasets 拉取并转成 (query, document, label) 格式。

    支持:
      * "t2reranking"   —— C-MTEB/T2Reranking (query, positive[list], negative[list])
                            → 拆成正/负样本对
      * "mmarco-zh"     —— sentence-transformers/msmarco triplets-zh (query,pos,neg)
                            → pairwise
      * "cmarco-en"     —— sentence-transformers/msmarco triplets
    """

    _ADAPTERS = {
        "t2reranking": "_adapt_t2reranking",
        "mmarco-zh": "_adapt_mmarco_triplet",
        "mmarco-en": "_adapt_mmarco_triplet",
    }

    def __init__(self, dataset_name: str, split: str = "dev",
                 cache_dir: Optional[str] = None):
        from datasets import load_dataset
        if dataset_name not in self._ADAPTERS:
            raise ValueError(f"未知 dataset_name '{dataset_name}',支持: {list(self._ADAPTERS.keys())}")
        self.dataset_name = dataset_name
        self._adapter = getattr(self, self._ADAPTERS[dataset_name])

        repo_configs = {
            "t2reranking": ("C-MTEB/T2Reranking", None),
            "mmarco-zh": ("sentence-transformers/msmarco", "triplets-zh"),
            "mmarco-en": ("sentence-transformers/msmarco", "triplets"),
        }
        repo, config = repo_configs[dataset_name]
        print(f"[HFRerankDataset] 加载 {repo} (config={config}) split={split} ...")
        self.ds = load_dataset(repo, config) if config else load_dataset(repo)
        # 取目标 split
        if split in self.ds:
            self.ds = self.ds[split]
        else:
            self.ds = self.ds[list(self.ds.keys())[0]]
        # 预处理:展开成 (q, d, label) 列表(adapter 可能 1 条→多条)
        self._samples = []
        for row in self.ds:
            for q, d, lab in self._adapter(row):
                self._samples.append((q, d, lab))
        print(f"[HFRerankDataset] 展开后共 {len(self._samples)} 条 (q,d,label) 对")

    def __len__(self):
        return len(self._samples)

    def __getitem__(self, idx):
        return self._samples[idx]

    # ---- adapters ----
    def _adapt_t2reranking(self, row):
        """C-MTEB/T2Reranking: {query, positive:[...], negative:[...]} → 每条展开。"""
        q = row.get("query", "")
        for pos in row.get("positive", []):
            yield q, pos, 1
        for neg in row.get("negative", []):
            yield q, neg, 0

    def _adapt_mmarco_triplet(self, row):
        """(query, positive, negative) → 2 条:正样本+负样本。"""
        yield row.get("query", ""), row.get("positive", ""), 1
        yield row.get("query", ""), row.get("negative", ""), 0


# ---------------------------------------------------------------------------
# Collate:批量 tokenize rerank prompt
# ---------------------------------------------------------------------------
class RerankCollator:
    """把一个 batch 的 (query, document, label) 转成模型输入。

    使用 shared.tokenizer.RERANK_PROMPT_TEMPLATE 构造输入,左 padding。
    """

    def __init__(self, tokenizer, max_length: int = 512):
        self.tok = tokenizer
        self.max_length = max_length

    def __call__(self, batch):
        queries = [b[0] for b in batch]
        documents = [b[1] for b in batch]
        labels = torch.tensor([b[2] for b in batch], dtype=torch.long)

        from shared.tokenizer import build_rerank_inputs
        enc = build_rerank_inputs(self.tok, queries, documents, max_length=self.max_length)
        return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"], "labels": labels}


# ---------------------------------------------------------------------------
# demo 数据生成(smoke test)
# ---------------------------------------------------------------------------
_DEMO_PAIRS = [
    ("天空为什么是蓝色的", "太阳光穿过大气层时,蓝光波长较短被散射,所以天空呈蓝色。", 1),
    ("天空为什么是蓝色的", "水在0度以下会结成冰,这是物理变化。", 0),
    ("中国首都是哪里", "中华人民共和国的首都是北京,位于华北平原北部。", 1),
    ("中国首都是哪里", "上海是中国最大的经济中心城市,位于长江入海口。", 0),
    ("水的沸点", "在标准大气压下,纯水加热到100摄氏度时会沸腾。", 1),
    ("水的沸点", "光合作用是植物利用阳光合成有机物的过程。", 0),
    ("感冒怎么治", "普通感冒通常由病毒引起,需要多休息多喝水,一般7天自愈。", 1),
    ("感冒怎么治", "地球围绕太阳公转一周约需365天。", 0),
    ("光合作用是什么", "绿色植物利用光能将二氧化碳和水转化为葡萄糖和氧气。", 1),
    ("光合作用是什么", "声音在空气中传播速度约为340米每秒。", 0),
    ("DNA是什么", "DNA分子由两条互补的核苷酸链组成双螺旋结构。", 1),
    ("DNA是什么", "长江是中国第一长河,全长约6300公里。", 0),
]


def write_demo_rerank_jsonl(path: str, n: int = 128):
    """生成 n 条 (query, document, label) 三元组(smoke test 用)。"""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for i in range(n):
            q, d, lab = _DEMO_PAIRS[i % len(_DEMO_PAIRS)]
            f.write(json.dumps({"query": q, "document": d, "label": lab}) + "\n")
    print(f"[write_demo_rerank_jsonl] 已生成 {path} ({n} 条)")
