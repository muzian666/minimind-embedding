"""
shared/model.py — Embedding / Rerank 共享底座

设计思路
========
minimind 的 decoder-only causal LM 已经实现了全部底层组件
(RMSNorm / RoPE / GQA-Attention / SwiGLU FFN / MoE),代码质量很高。
本项目不重写这些组件,而是直接复用 `minimind.model.model_minimind`,
在其之上构建两个面向表示学习的 head:

  * `MiniMindForEmbedding` —— 对比学习 embedding 模型
      走 causal attention,末尾追加 EOS,取 last token 的 hidden state,
      经投影 + L2 归一化得到句向量。完全对齐 Qwen3-Embedding 路线。

  * `MiniMindForRerank` —— pointwise yes/no 重排模型
      直接复用 lm_head,把"判断 query-doc 是否相关"建模成"下一个 token
      是 <yes> 还是 <no>"的二分类。完全对齐 Qwen3-Reranker 路线。

为什么是 causal + last-token 而不是双向 + mean-pool?
  这是为了最大化复用预训练权重——causal LM 的预训练分布与本结构一致,
  无需重新预热。Qwen3-Embedding 实测证明 causal + last-token 路线效果优秀。
"""
import os
import sys
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel

# ---------------------------------------------------------------------------
# 把上游 minimind 包加入 import 路径(只读引用,不改动原项目)
# ---------------------------------------------------------------------------
_MINIMIND_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, "minimind")
)
if _MINIMIND_ROOT not in sys.path:
    sys.path.insert(0, _MINIMIND_ROOT)

from model.model_minimind import (  # noqa: E402
    MiniMindConfig,
    MiniMindModel,
    MOEFeedForward,
)

from shared.utils import last_token_pool  # noqa: E402


class MiniMindEmbedConfig(MiniMindConfig):
    """Embedding 专用配置(继承自 MiniMindConfig,新增 embedding 相关字段)。"""

    def __init__(
        self,
        embed_dim: Optional[int] = None,           # 投影后输出维度,None 表示等于 hidden_size
        mrl_dims: Optional[list] = None,           # Matryoshka 训练维度列表
        pooling: str = "last_token",               # 池化方式(本项目固定 last_token)
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim if embed_dim is not None else self.hidden_size
        self.mrl_dims = mrl_dims or [self.hidden_size]
        self.pooling = pooling
        # embedding 模型不需要 lm_head
        self.tie_word_embeddings = False


class MiniMindRerankConfig(MiniMindConfig):
    """Rerank 专用配置(继承自 MiniMindConfig,新增 rerank 相关字段)。"""

    def __init__(
        self,
        yes_token_id: int = 6400,
        no_token_id: int = 6401,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.yes_token_id = yes_token_id
        self.no_token_id = no_token_id
        # rerank 复用 lm_head,保留 tie 选项(默认 True,与预训练一致)
        self.tie_word_embeddings = kwargs.get("tie_word_embeddings", True)


class MiniMindForEmbedding(PreTrainedModel):
    """基于 minimind 底座的 Embedding 模型。

    流程:input_ids → MiniMindModel(causal) → last_token_pool → 投影 → L2 归一化。
    """

    config_class = MiniMindEmbedConfig

    def __init__(self, config: MiniMindEmbedConfig):
        self.config = config
        super().__init__(config)
        self.model = MiniMindModel(config)
        # 投影头:Linear + (可选)LayerNorm。Qwen3-Embedding 也是简单的投影。
        self.head = nn.Sequential(
            nn.Linear(config.hidden_size, config.embed_dim, bias=False),
            nn.LayerNorm(config.embed_dim, eps=config.rms_norm_eps),
        )
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_mrl: bool = False,
        **kwargs,
    ):
        """前向。

        参数:
            input_ids: [B, T]
            attention_mask: [B, T],padding side 推荐 left
            output_mrl: 若为 True,返回每个 mrl_dim 的切片向量列表(用于 MRL 训练)。

        返回:
            output_mrl=False → z: [B, embed_dim](已 L2 归一化)
            output_mrl=True  → list of [B, mrl_dim](每个切片独立 L2 归一化)
        """
        hidden_states, _, aux_loss = self.model(
            input_ids, attention_mask=attention_mask, use_cache=False, **kwargs
        )
        pooled = last_token_pool(hidden_states, attention_mask)  # [B, hidden]
        z = self.head(pooled)

        if not output_mrl:
            return F.normalize(z, p=2, dim=1)

        # Matryoshka:对前 mrl_dim 维切片后各自归一化
        out = []
        for d in self.config.mrl_dims:
            out.append(F.normalize(z[..., :d], p=2, dim=1))
        return out

    @torch.inference_mode()
    def encode(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """推理便捷接口,返回单一句向量(已归一化)。"""
        self.eval()
        return self.forward(
            input_ids, attention_mask=attention_mask, output_mrl=False, **kwargs
        )


class MiniMindForRerank(PreTrainedModel):
    """基于 minimind 底座的 Pointwise yes/no 重排模型。

    直接复用 lm_head,把"query-doc 是否相关"建模为对末尾 token 的二分类
    (下一个 token 是 <yes> 还是 <no>)。无新增参数(相对预训练底座)。

    forward 返回 [B, 2] 的原始 logits(yes 在前,no 在后),由调用方决定
    softmax 还是直接 cross_entropy。`predict` 返回 0~1 的 yes 概率。
    """

    config_class = MiniMindRerankConfig
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

    def __init__(self, config: MiniMindRerankConfig):
        self.config = config
        super().__init__(config)
        self.model = MiniMindModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """前向,返回 [B, 2] 的 (yes_logit, no_logit)。"""
        hidden_states, _, aux_loss = self.model(
            input_ids, attention_mask=attention_mask, use_cache=False, **kwargs
        )
        last_hidden = last_token_pool(hidden_states, attention_mask)  # [B, hidden]
        logits = self.lm_head(last_hidden)  # [B, vocab]
        # 只取 yes / no 两个 token 的 logits
        return logits[:, [self.config.yes_token_id, self.config.no_token_id]]

    @torch.inference_mode()
    def predict(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """推理便捷接口,返回 [B] 的 yes 概率(0~1)作为相关性分数。"""
        self.eval()
        yn_logits = self.forward(input_ids, attention_mask=attention_mask, **kwargs)
        return F.softmax(yn_logits, dim=-1)[:, 0]
