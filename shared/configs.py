"""
shared/configs.py — 模型配置预设

4 个核心配置(Dense/MoE × Embedding/Rerank),全部基于 minimind-3 底座
(dim=768, n_layers=8, q_heads=8, kv_heads=4, head_dim=96, vocab=6400)。
"""
import os

from shared.model import MiniMindEmbedConfig, MiniMindRerankConfig

# Rerank 的 yes/no 目标 token:
# 改用词表里现成的 "是"(id=357)/"否"(id=1332),而非新增 <yes>/<no>。
# 原因:新增 token 的 embedding 是随机初始化的,预训练从未见过,
#       模型难以在低 lr 下学好"相关性 → 新 token"的映射(实测会学反)。
#       "是"/"否" 是单 token、语义明确、预训练见过,直接复用最稳。
YES_TOKEN_ID = 357   # "是"
NO_TOKEN_ID = 1332   # "否"
# 词表大小保持 minimind 原生 6400(不扩展)
EXTENDED_VOCAB_SIZE = 6400

# 通用底座超参(对齐 minimind-3)
_BASE = dict(
    hidden_size=768,
    num_hidden_layers=8,
    num_attention_heads=8,
    num_key_value_heads=4,
    head_dim=96,
    vocab_size=EXTENDED_VOCAB_SIZE,  # 6400,不扩展
    bos_token_id=1,
    eos_token_id=2,
    flash_attn=True,
    max_position_embeddings=32768,   # 原生支持,8192 训练无需外推
    rope_theta=1e6,
    rms_norm_eps=1e-6,
    dropout=0.0,
    intermediate_size=2432,          # ceil(768*π/64)*64 = 38*64 = 2432(对齐 minimind-3 实际发布权重)
)

# MoE 专属
_MOE = dict(
    use_moe=True,
    num_experts=4,
    num_experts_per_tok=1,
    moe_intermediate_size=2432,      # 与 dense 一致(对齐 minimind-3-moe)
    norm_topk_prob=True,
    router_aux_loss_coef=5e-4,
)

# Matryoshka 维度(全系列支持,768 为完整维度)
MRL_DIMS = [768, 512, 256, 128, 64]


# ---------------------------------------------------------------------------
# Embedding 配置
# ---------------------------------------------------------------------------
def embed_dense_64m() -> MiniMindEmbedConfig:
    """Dense Embedding,64M 底座,768 维输出。"""
    return MiniMindEmbedConfig(
        **_BASE,
        embed_dim=768,
        mrl_dims=MRL_DIMS,
        pooling="last_token",
    )


def embed_moe_198m() -> MiniMindEmbedConfig:
    """MoE Embedding,198M-A64M,768 维输出。"""
    return MiniMindEmbedConfig(
        **_BASE,
        **_MOE,
        embed_dim=768,
        mrl_dims=MRL_DIMS,
        pooling="last_token",
    )


# ---------------------------------------------------------------------------
# Rerank 配置
# ---------------------------------------------------------------------------
def rerank_dense_64m() -> MiniMindRerankConfig:
    """Dense Rerank,64M,pointwise yes/no。"""
    return MiniMindRerankConfig(
        **_BASE,
        yes_token_id=YES_TOKEN_ID,
        no_token_id=NO_TOKEN_ID,
    )


def rerank_moe_198m() -> MiniMindRerankConfig:
    """MoE Rerank,198M-A64M,pointwise yes/no。"""
    return MiniMindRerankConfig(
        **_BASE,
        **_MOE,
        yes_token_id=YES_TOKEN_ID,
        no_token_id=NO_TOKEN_ID,
    )


# 名称 → 构造函数 的注册表(供 CLI 解析)
CONFIG_REGISTRY = {
    "embed_dense_64m": embed_dense_64m,
    "embed_moe_198m": embed_moe_198m,
    "rerank_dense_64m": rerank_dense_64m,
    "rerank_moe_198m": rerank_moe_198m,
}


def get_config(name: str):
    """按名称取配置实例。"""
    if name not in CONFIG_REGISTRY:
        raise ValueError(
            f"未知配置名 '{name}',可选: {list(CONFIG_REGISTRY.keys())}"
        )
    return CONFIG_REGISTRY[name]()
