"""
minimind_embedding/dataset.py — Embedding 训练数据加载

统一数据格式(每行一个 JSON):
    {"query": str, "positive": str, "hard_negatives": [str, ...]}

支持两种数据来源:
  1. 本地 jsonl 文件(用户预处理好的三元组)
  2. HuggingFace datasets(load_dataset 直接拉,内置适配器)

为兼容 in-batch InfoNCE,每个 batch 内会做:
  * query  + EOS  → encode
  * positive + EOS → encode
  * 每个 query 的 K 个 hard negatives + EOS → encode
"""
import os
import json
import random
from typing import Optional, List, Dict

import torch
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# 本地 jsonl 数据集(最快验证路径,无网络依赖)
# ---------------------------------------------------------------------------
class TripletJsonlDataset(Dataset):
    """读取本地 {query, positive, hard_negatives} jsonl。

    每条样本返回原始文本三元组(tokenize 由 collate_fn 负责)。
    """

    def __init__(
        self,
        jsonl_path: str,
        max_negatives: int = 7,
        min_negatives: int = 0,
    ):
        self.path = jsonl_path
        self.max_neg = max_negatives
        self.min_neg = min_negatives
        self.samples: List[Dict] = []
        if not os.path.exists(jsonl_path):
            raise FileNotFoundError(f"数据文件不存在: {jsonl_path}")
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                # 字段兼容:query/pos/positive, hard_negatives/neg/negatives
                q = obj.get("query") or obj.get("q") or ""
                pos = obj.get("positive") or obj.get("pos") or ""
                negs = (
                    obj.get("hard_negatives")
                    or obj.get("negatives")
                    or obj.get("neg")
                    or []
                )
                if isinstance(negs, str):
                    negs = [negs]
                if not q or not pos:
                    continue
                # 采样/补齐负样本数
                if len(negs) > self.max_neg:
                    negs = random.sample(negs, self.max_neg)
                self.samples.append({"query": q, "positive": pos, "negatives": negs})

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return s["query"], s["positive"], s["negatives"]


# ---------------------------------------------------------------------------
# HuggingFace datasets 适配器(M2 阶段使用,接 T2Ranking / mMARCO 等)
# ---------------------------------------------------------------------------
class HFEmbeddingDataset(Dataset):
    """从 HF datasets 拉取并转成统一三元组格式。

    支持的数据集 schema(通过 dataset_name 路由到对应转换器):
      * "t2ranking"   —— sentence-transformers/t2ranking,有 query/pos/neg
      * "mmarco-zh"   —— unicamp-dl/mmarco chinese,有 query/positive/negative
      * "mmarco-en"   —— unicamp-dl/mmarco english
      * "nli-zh"      —— shibing624/nli-zh-all,有 premise+hypothesis(正/负)
    """

    _ADAPTERS = {
        "t2ranking": "_adapt_st_triplet",
        "t2ranking-15": "_adapt_st_triplet15",
        "mmarco-zh": "_adapt_mmarco",
        "mmarco-en": "_adapt_mmarco",
        "nli-zh": "_adapt_nli_zh",
    }

    def __init__(
        self,
        dataset_name: str,
        split: str = "train",
        max_negatives: int = 7,
        streaming: bool = False,
        cache_dir: Optional[str] = None,
    ):
        from datasets import load_dataset

        if dataset_name not in self._ADAPTERS:
            raise ValueError(
                f"未知 dataset_name '{dataset_name}',支持: {list(self._ADAPTERS.keys())}"
            )
        self.dataset_name = dataset_name
        self.max_neg = max_negatives
        adapter = self._ADAPTERS[dataset_name]
        self._adapter = getattr(self, adapter)

        # 按数据集选择 HF 仓库与 config(datasets 5.0 不再支持 script 格式,
        # 故 mmarco 用不了 unicamp-dl 老仓库;改用 parquet 仓库)
        repo_configs = {
            "t2ranking": ("sentence-transformers/t2ranking", "triplet"),
            "t2ranking-15": ("sentence-transformers/t2ranking", "triplet-15"),
            "mmarco-zh": ("sentence-transformers/mmarco", "triplets-zh"),
            "mmarco-en": ("sentence-transformers/msmarco", "triplets"),
            "nli-zh": ("shibing624/nli-zh-all", None),
        }
        repo, config = repo_configs[dataset_name]
        print(f"[HFEmbeddingDataset] 加载 {repo} (config={config}) split={split} ...")
        if streaming:
            self.ds = load_dataset(repo, config, split=split, streaming=True, cache_dir=cache_dir)
            self._len = None
        else:
            self.ds = load_dataset(repo, config, split=split, cache_dir=cache_dir)
            self._len = len(self.ds)
            print(f"[HFEmbeddingDataset] 共 {self._len} 条")

    def __len__(self):
        return self._len if self._len is not None else 100_000_000  # streaming 模式无法预知长度

    def __getitem__(self, idx):
        row = self.ds[idx]
        return self._adapter(row)

    # ---- 各数据集转换器 ----
    def _adapt_st_triplet(self, row):
        # sentence-transformers/t2ranking "triplet" config:{anchor, positive, negative}
        q = row.get("anchor") or row.get("query") or ""
        pos = row.get("positive") or row.get("pos") or ""
        neg = row.get("negative") or row.get("neg") or ""
        negs = [neg] if isinstance(neg, str) else (neg or [])
        return q, pos, negs

    def _adapt_st_triplet15(self, row):
        # sentence-transformers/t2ranking "triplet-15" config:
        # {anchor, positive, negative_1, negative_2, ..., negative_15} —— 15 个 hard negatives
        q = row.get("anchor", "")
        pos = row.get("positive", "")
        negs = [row[f"negative_{i}"] for i in range(1, 16) if row.get(f"negative_{i}")]
        return q, pos, negs

    def _adapt_mmarco(self, row):
        # unicamp-dl/mmarco: {"query","positive","negative"}
        q = row.get("query", "")
        pos = row.get("positive", "")
        neg = row.get("negative", "")
        return q, pos, [neg] if neg else []

    def _adapt_nli_zh(self, row):
        # NLI:premise 为 query,entailment 为正,contradiction 为负
        q = row.get("premise", "")
        pos = row.get("hypothesis", "") if row.get("label", 0) == 0 else ""
        neg = row.get("hypothesis", "") if row.get("label", 0) == 2 else ""
        return q, pos, [neg] if neg else []


# ---------------------------------------------------------------------------
# Collate:批量 tokenize + 左 padding + EOS
# ---------------------------------------------------------------------------
class EmbeddingCollator:
    """把一个 batch 的 (query, positive, [negatives]) 文本转成模型输入。

    输出 dict:
        query:        {input_ids, attention_mask}    [B, Lq]
        positive:     {input_ids, attention_mask}    [B, Lp]
        negatives:    {input_ids, attention_mask}    [B*K, Ln]  (展平)
        n_neg_per_q:  每条 query 的负样本数(变长时用于分离)
    """

    def __init__(self, tokenizer, max_length: int = 512, max_negatives: int = 7,
                 instruction: Optional[str] = None):
        self.tok = tokenizer
        self.max_length = max_length
        self.max_neg = max_negatives
        self.instruction = instruction

    def _encode(self, texts, is_query):
        from shared.tokenizer import build_embedding_inputs
        return build_embedding_inputs(
            self.tok, texts, is_query=is_query,
            instruction=self.instruction, max_length=self.max_length,
        )

    def __call__(self, batch):
        queries = [b[0] for b in batch]
        positives = [b[1] for b in batch]
        # 把每条 query 的 negatives 截断到 max_neg,展平
        negs_flat = []
        n_neg_per_q = []
        for b in batch:
            negs = b[2][: self.max_neg] if b[2] else []
            negs_flat.extend(negs)
            n_neg_per_q.append(len(negs))

        out = {
            "query": self._encode(queries, is_query=True),
            "positive": self._encode(positives, is_query=False),
            "n_neg_per_q": torch.tensor(n_neg_per_q, dtype=torch.long),
        }
        if negs_flat:
            out["negatives"] = self._encode(negs_flat, is_query=False)
        else:
            out["negatives"] = None
        return out


# ---------------------------------------------------------------------------
# 小工具:生成 smoke test 用的迷你三元组
# ---------------------------------------------------------------------------
_DEMO_TOPICS = [
    ("天空的颜色", "太阳光穿过大气层时,蓝光波长较短被散射,所以天空呈蓝色。", "水在0度以下会结成冰。"),
    ("水的沸点", "在标准大气压下,纯水加热到100摄氏度时会沸腾。", "光合作用是植物利用阳光合成有机物的过程。"),
    ("中国首都", "中华人民共和国的首都是北京,位于华北平原北部。", "上海是中国最大的经济中心城市之一。"),
    ("感冒的治疗", "普通感冒通常由病毒引起,需要多休息多喝水,一般7天自愈。", "地球围绕太阳公转一周约需365天。"),
    ("光合作用", "绿色植物利用光能将二氧化碳和水转化为葡萄糖和氧气。", "声音在空气中传播速度约为340米每秒。"),
    ("地球公转", "地球围绕太阳公转一周大约需要365.25天,形成四季更替。", "铁在潮湿空气中容易生锈。"),
    ("光速", "光在真空中的传播速度约为每秒30万公里,是宇宙中最快的速度。", "人体的正常体温约为37摄氏度。"),
    (" DNA结构", "DNA分子由两条互补的核苷酸链组成双螺旋结构。", "长江是中国第一长河,全长约6300公里。"),
    ("化学反应", "化学反应中,反应物分子通过断键和成键转化为生成物。", "月亮本身不发光,而是反射太阳光。"),
    ("血液循环", "人体血液循环系统由心脏泵血,经动脉输送氧气到全身。", "珠穆朗玛峰是世界最高峰,海拔约8848米。"),
    ("声音传播", "声音需要介质传播,在空气中速度约340米每秒,真空中无法传播。", "水的分子式是H2O,由两个氢原子和一个氧原子组成。"),
    ("重力", "地球引力使物体产生重力加速度,约为9.8米每秒平方。", "蜜蜂通过舞蹈传递花蜜位置信息。"),
    ("火山", "火山是地壳内部岩浆喷出地表形成的地质构造。", "钢琴有88个琴键,包括52个白键和36个黑键。"),
    ("氧气", "氧气是无色无味气体,占大气约21%,是呼吸作用必需。", "围棋棋盘有19条横线和19条纵线。"),
    ("海洋", "地球表面约71%被海洋覆盖,太平洋是最大的海洋。", "黄金的化学符号是Au,原子序数79。"),
    ("地震", "地震是地壳板块运动释放能量引起的地面震动现象。", "光的三原色是红绿蓝,颜料三原色是红黄蓝。"),
]


def write_demo_jsonl(path: str, n: int = 256):
    """生成 n 条 query-positive-negative 三元组(用于 smoke test 验证 loss 能降)。

    注:无预训练底座时,纯随机初始化的模型在极简数据上会"塌缩"(把所有向量映射到
    同一点使 loss=0)。因此 demo 数据必须足够多样(每条 query 尽量不重复),
    才能观察到真实的 loss 下降。真实训练请用 HFEmbeddingDataset 拉取 T2Ranking / mMARCO。
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    import random as _r
    with open(path, "w", encoding="utf-8") as f:
        for i in range(n):
            q, pos, _ = _DEMO_TOPICS[i % len(_DEMO_TOPICS)]
            # 随机选 1-3 个其他 topic 的正样本描述作为负样本(语义不相关)
            n_neg = _r.randint(1, 3)
            other_topics = [t for j, t in enumerate(_DEMO_TOPICS) if j != (i % len(_DEMO_TOPICS))]
            chosen = _r.sample(other_topics, min(n_neg, len(other_topics)))
            negs = [t[1] for t in chosen]  # 用其他 topic 的正样本描述当负样本
            f.write(json.dumps({"query": q, "positive": pos, "hard_negatives": negs}) + "\n")
    print(f"[write_demo_jsonl] 已生成 {path} ({n} 条, {len(_DEMO_TOPICS)} 个不同 query)")
