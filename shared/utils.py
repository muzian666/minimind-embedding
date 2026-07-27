"""
shared/utils.py — 通用工具

包含:
  * last_token_pool      —— Qwen3 风格的最后非 pad token 提取
  * InfoNCE 带假负样本 mask —— Qwen3-Embedding 损失函数逐字实现
  * load_pretrained_backbone —— 从 minimind 预训练 .pth 加载底座
  * compute_num_params   —— 统计参数量(支持 MoE 的 active 计算)
"""
from __future__ import annotations

import os
import sys
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

_MINIMIND_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, "minimind")
)
if _MINIMIND_ROOT not in sys.path:
    sys.path.insert(0, _MINIMIND_ROOT)


# ---------------------------------------------------------------------------
# Last-token pooling(严格对齐 Qwen3-Embedding-0.6B 官方实现)
# ---------------------------------------------------------------------------
def last_token_pool(
    last_hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """从 hidden states 中取出每条序列最后一个非 pad token 的表示。

    关键陷阱:
      * 左 padding(recommend):所有真实 token 在右侧,序列末尾 [:, -1] 就是
        最后一个真实 token,O(1) 取值。本项目统一用左 padding。
      * 右 padding:必须按 attention_mask.sum(dim=1)-1 计算每条序列真实长度索引再 gather。
        直接取 [:,-1] 会拿到 pad token 表示(错误!)。

    参数:
        last_hidden_states: [B, T, H]
        attention_mask:     [B, T]
    返回:
        [B, H]
    """
    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if left_padding:
        return last_hidden_states[:, -1]
    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_size = last_hidden_states.shape[0]
    return last_hidden_states[
        torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths
    ]


# ---------------------------------------------------------------------------
# InfoNCE + 假负样本 mask(Qwen3-Embedding 报告公式逐字实现)
# ---------------------------------------------------------------------------
def infonce_with_false_neg_mask(
    q: torch.Tensor,              # [B, D]
    d_pos: torch.Tensor,          # [B, D]
    d_neg: Optional[torch.Tensor] = None,  # [B, K, D] 或 None
    *,
    temp: float = 0.02,
    margin: float = 0.1,
) -> torch.Tensor:
    """带假负样本 mask 的 InfoNCE 损失。

    参考:Qwen3 Embedding 技术报告 (arXiv:2506.05176)。

    核心思想(报告公式):
        分母聚合: 正样本 + hard negatives + in-batch(query-query, doc-doc) negatives。
        mask factor m_ij:
            m_ij = 0  若 s(q_i, candidate_j) > s(q_i, d_i^+) + 0.1
                       (该候选与 query 的相似度甚至超过正样本 → 很可能是假负样本 → 屏蔽)
            m_ij = 1  否则

    参数:
        q:     query 向量(已 L2 归一化),[B, D]
        d_pos: 正样本 doc 向量(已归一化),[B, D]
        d_neg: hard negative doc 向量(已归一化),[B, K, D];无则 None
        temp:  温度系数 τ。Qwen3 未公布,社区常用 0.02~0.05,这里默认 0.02。
        margin: 假负样本判定的相似度裕度,报告取 0.1。

    返回:
        标量 loss。

    实现说明:
        相似度用 cosine(q, d) = q·d(因输入已 L2 归一化,等价)。
        对 [B, B] 的 query-query / doc-doc 矩阵,对角线(自身)置 0。
    """
    B = q.size(0)
    device = q.device

    # 1) 正样本相似度 [B, 1]
    s_pos = (q * d_pos).sum(dim=-1, keepdim=True)

    # 2) hard negatives [B, K]
    neg_parts, neg_masks = [], []
    if d_neg is not None and d_neg.numel() > 0:
        s_neg = (q.unsqueeze(1) * d_neg).sum(dim=-1)  # [B, K]
        mask_neg = (s_neg <= s_pos + margin).to(s_neg.dtype)  # 超过的当假负样本屏蔽
        neg_parts.append((s_neg * mask_neg) / temp)
        neg_masks.append(mask_neg)

    # 3) in-batch query-query negatives [B, B]
    s_qq = q @ q.t()  # [B, B]
    mask_qq = (s_qq <= s_pos + margin).to(s_qq.dtype)
    mask_qq.fill_diagonal_(0.0)  # 自身不算负样本

    # 4) in-batch doc-doc negatives [B, B]
    s_dd = d_pos @ d_pos.t()  # [B, B]
    mask_dd = (s_dd <= s_pos + margin).to(s_dd.dtype)
    mask_dd.fill_diagonal_(0.0)

    # 5) 聚合分母:[正样本, neg..., qq, dd],正样本在第 0 列
    columns = [s_pos / temp]
    columns += neg_parts
    columns += [(s_qq * mask_qq) / temp, (s_dd * mask_dd) / temp]
    logits = torch.cat(columns, dim=1)  # [B, 1+K+2B]

    # 标签:每个 query 的正样本都在第 0 列
    labels = torch.zeros(B, dtype=torch.long, device=device)
    return F.cross_entropy(logits, labels)


def matryoshka_infonce_loss(
    q_list,                       # list of [B, D_k](各 mrl 维度切片,已归一化)
    d_pos_list,                   # list of [B, D_k]
    d_neg_list=None,              # list of [B, K, D_k] 或 None
    *,
    temp: float = 0.02,
    margin: float = 0.1,
    weights=None,                 # 各维度 loss 权重,None 表示等权
) -> torch.Tensor:
    """MRL:对每个维度切片分别算 InfoNCE,加权求平均。

    参考:Matryoshka Representation Learning (Kusupati et al., 2022)。
    Qwen3-Embedding 全系列支持 MRL,允许输出维度灵活截断(如 768/512/256/128/64)。
    """
    losses = []
    for i, (q, dp) in enumerate(zip(q_list, d_pos_list)):
        dn = d_neg_list[i] if d_neg_list is not None else None
        losses.append(infonce_with_false_neg_mask(q, dp, dn, temp=temp, margin=margin))
    if weights is None:
        return torch.stack(losses).mean()
    return sum(w * l for w, l in zip(weights, losses))


# ---------------------------------------------------------------------------
# 权重加载:从 minimind 预训练 .pth 初始化底座
# ---------------------------------------------------------------------------
def load_pretrained_backbone(
    model: nn.Module,
    hidden_size: int = 768,
    use_moe: bool = False,
    from_weight: str = "pretrain",
    save_dir: Optional[str] = None,
    device: str = "cuda",
):
    """把 minimind 预训练权重(.pth)加载到 embedding/rerank 模型的底座。

    minimind 的权重命名约定:{weight}_{hidden_size}[_moe].pth,保存为 half().cpu()。
    我们的模型把底座包在 self.model(MiniMindModel)下,因此 key 前缀对齐
    (PreTrainedModel 默认顶层 model.*,与 minimind 的 ForCausalLM 结构一致)。

    参数:
        model: MiniMindForEmbedding 或 MiniMindForRerank 实例
        from_weight: 'pretrain' / 'full_sft' / 'none'(none 表示不加载)
    """
    if from_weight == "none":
        return model

    if save_dir is None:
        # 默认指向 minimind 项目的 out/ 目录
        save_dir = os.path.join(_MINIMIND_ROOT, "out")
    moe_suffix = "_moe" if use_moe else ""
    weight_path = os.path.join(save_dir, f"{from_weight}_{hidden_size}{moe_suffix}.pth")

    if not os.path.exists(weight_path):
        raise FileNotFoundError(
            f"预训练权重未找到: {weight_path}\n"
            f"请先参考 minimind 项目完成预训练,或下载社区权重到该路径。"
        )

    state_dict = torch.load(weight_path, map_location=device)
    model_sd = model.state_dict()

    # 手动逐 key 拷贝,处理两类不匹配:
    #   1) embed_tokens/lm_head:我们扩展了 vocab(6400→6402),形状 [6402,768] vs [6400,768]
    #      → 把前 6400 行拷过来,新增的 2 行保留随机初始化
    #   2) 大小完全不匹配的 key(理论上不应出现)→ 跳过并警告
    loaded, skipped, partial = [], [], []
    for k, v in state_dict.items():
        if k not in model_sd:
            skipped.append(k)  # 模型里没有(如 embedding 模型的 lm_head)
            continue
        target = model_sd[k]
        if target.shape == v.shape:
            target.copy_(v)
            loaded.append(k)
        elif target.dim() == v.dim() and target.shape[1:] == v.shape[1:]:
            # 第 0 维不同(vocab 扩展):拷贝重叠部分
            n = min(target.shape[0], v.shape[0])
            target[:n].copy_(v[:n])
            partial.append((k, tuple(v.shape), tuple(target.shape)))
        else:
            skipped.append(k)

    # 把手动修改过的 tensor 写回模型(因为 copy_ 是原地操作,已生效,但保险起见)
    print(f"[load_pretrained_backbone] 已加载 {weight_path}")
    print(f"  完整加载: {len(loaded)} 个参数")
    print(f"  部分加载: {len(partial)} 个(vocab 扩展,前 6400 行已拷贝)")
    for k, src, dst in partial:
        print(f"    {k}: {src} → {dst}")
    print(f"  跳过:     {len(skipped)} 个(embedding 模型忽略 lm_head / rerank 缺 head 属正常)")
    return model


# ---------------------------------------------------------------------------
# 参数统计(支持 MoE 的 total-active 表示)
# ---------------------------------------------------------------------------
def compute_num_params(model: nn.Module, config) -> str:
    """统计参数量,MoE 时输出 '198.2M-A64.0M' 形式。"""
    total = sum(p.numel() for p in model.parameters()) / 1e6
    use_moe = getattr(config, "use_moe", False)
    if not use_moe:
        return f"{total:.1f}M"
    n_routed = getattr(config, "num_experts", 0)
    n_active = getattr(config, "num_experts_per_tok", 0)
    expert = (
        sum(p.numel() for n, p in model.named_parameters() if "mlp.experts.0." in n) / 1e6
    )
    base = total - expert * n_routed
    active = base + expert * n_active
    return f"{total:.1f}M-A{active:.1f}M"


# ---------------------------------------------------------------------------
# 学习率调度(复用 minimind 的余弦调度,保持训练一致)
# ---------------------------------------------------------------------------
def get_lr(current_step: int, total_steps: int, lr: float) -> float:
    """余弦调度:起点 lr,终点 lr*0.1。无 warmup(与 minimind 一致)。

    对于 embedding 的 stage1/stage2,可在脚本里自行加 warmup 后调用此函数。
    """
    return lr * (0.1 + 0.45 * (1 + math.cos(math.pi * current_step / total_steps)))
