"""MiniMind-Embedding & Rerank 共享层。

包含:
  * model     —— MiniMindForEmbedding / MiniMindForRerank
  * configs   —— 4 个预设配置(Dense/MoE × Embedding/Rerank)
  * tokenizer —— minimind BPE + <yes>/<no> 扩展 + 输入模板
  * utils     —— last_token_pool / InfoNCE / 权重加载 / 学习率
"""
from shared.model import (
    MiniMindEmbedConfig,
    MiniMindRerankConfig,
    MiniMindForEmbedding,
    MiniMindForRerank,
)
from shared.utils import (
    last_token_pool,
    infonce_with_false_neg_mask,
    matryoshka_infonce_loss,
    load_pretrained_backbone,
    compute_num_params,
    get_lr,
)
from shared.configs import (
    embed_dense_64m,
    embed_moe_198m,
    rerank_dense_64m,
    rerank_moe_198m,
    get_config,
    CONFIG_REGISTRY,
    MRL_DIMS,
    YES_TOKEN_ID,
    NO_TOKEN_ID,
    EXTENDED_VOCAB_SIZE,
)
from shared.tokenizer import (
    load_tokenizer,
    build_embedding_inputs,
    build_rerank_inputs,
    RERANK_PROMPT_TEMPLATE,
    DEFAULT_QUERY_INSTRUCTION,
)

__all__ = [
    # model
    "MiniMindEmbedConfig", "MiniMindRerankConfig",
    "MiniMindForEmbedding", "MiniMindForRerank",
    # utils
    "last_token_pool", "infonce_with_false_neg_mask", "matryoshka_infonce_loss",
    "load_pretrained_backbone", "compute_num_params", "get_lr",
    # configs
    "embed_dense_64m", "embed_moe_198m", "rerank_dense_64m", "rerank_moe_198m",
    "get_config", "CONFIG_REGISTRY", "MRL_DIMS",
    "YES_TOKEN_ID", "NO_TOKEN_ID", "EXTENDED_VOCAB_SIZE",
    # tokenizer
    "load_tokenizer", "build_embedding_inputs", "build_rerank_inputs",
    "RERANK_PROMPT_TEMPLATE", "DEFAULT_QUERY_INSTRUCTION",
]
