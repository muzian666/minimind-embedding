"""
minimind_embedding/train.py — Embedding 模型训练脚本

支持两阶段(对齐 Qwen3-Embedding 报告):
  * stage1 弱监督预训练:大 batch,纯 in-batch InfoNCE,不加 hard negative。
                       用大规模 BM25/triplet 数据。
  * stage2 监督微调   :带 K 个 hard negative + 假负样本 mask + MRL 多维度。
                       用高质量标注数据。

用法示例(smoke test):
  python -m minimind_embedding.train \\
      --config embed_dense_64m \\
      --stage 2 \\
      --data_type demo \\
      --batch_size 8 \\
      --accumulation_steps 2 \\
      --epochs 3 \\
      --max_length 128 \\
      --device cpu             # GPU 上换成 cuda

真实训练(stage2,T2Ranking 数据):
  python -m minimind_embedding.train \\
      --config embed_dense_64m --stage 2 --data_type hf \\
      --hf_dataset t2ranking --batch_size 64 --epochs 1 --max_length 512 --device cuda
"""
import os
import sys
import argparse
import math

# 确保项目根在 path 上
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import torch
from torch.utils.data import DataLoader

from shared import (
    get_config, load_tokenizer, load_pretrained_backbone, compute_num_params,
    get_lr, infonce_with_false_neg_mask, matryoshka_infonce_loss,
    DEFAULT_QUERY_INSTRUCTION, MRL_DIMS,
)
from shared.utils import last_token_pool
from minimind_embedding.dataset import (
    TripletJsonlDataset, HFEmbeddingDataset, EmbeddingCollator, write_demo_jsonl,
)


def parse_args():
    p = argparse.ArgumentParser(description="MiniMind-Embedding 训练")
    # 模型
    p.add_argument("--config", default="embed_dense_64m",
                   choices=["embed_dense_64m", "embed_moe_198m"])
    p.add_argument("--from_weight", default="none",
                   help="预训练底座来源:pretrain/full_sft/none")
    p.add_argument("--backbone_dir", default=None,
                   help="预训练权重目录(默认 minimind/out)")
    p.add_argument("--init_new_head", action="store_true",
                   help="从头初始化 head(默认会尝试加载预训练投影,若有)")
    # 数据
    p.add_argument("--stage", type=int, default=2, choices=[1, 2],
                   help="1=弱监督(纯in-batch), 2=监督(hard neg+MRL)")
    p.add_argument("--data_type", default="demo", choices=["demo", "jsonl", "hf"])
    p.add_argument("--jsonl_path", default="data/embedding_train.jsonl")
    p.add_argument("--hf_dataset", default="t2ranking")
    p.add_argument("--max_negatives", type=int, default=7)
    p.add_argument("--max_train_samples", type=int, default=0,
                   help="限制训练样本数(0=用全部;mini 验证用如 2000)")
    # 训练
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--accumulation_steps", type=int, default=1)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--lr", type=float, default=None,
                   help="学习率。None 时按 stage 自动取(stage1=2e-4, stage2=1e-5)")
    p.add_argument("--warmup_steps", type=int, default=0)
    p.add_argument("--temp", type=float, default=0.02, help="InfoNCE 温度 τ")
    p.add_argument("--margin", type=float, default=0.1, help="假负样本 mask margin")
    p.add_argument("--use_mrl", action="store_true",
                   help="stage2 启用 MRL 多维度损失")
    p.add_argument("--save_dir", default="checkpoints/embedding")
    p.add_argument("--save_steps", type=int, default=500)
    p.add_argument("--log_steps", type=int, default=10)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    # wandb 日志(失败时优雅降级为纯 stdout)
    p.add_argument("--wandb_project", default="minimind-embedding")
    p.add_argument("--wandb_run_name", default=None)
    p.add_argument("--no_wandb", action="store_true", help="禁用 wandb")
    return p.parse_args()


def init_wandb(args, config):
    """初始化 wandb。失败(无 key/离线)时返回 None,训练继续走纯 stdout。"""
    if args.no_wandb:
        return None
    try:
        import wandb
        run_name = args.wandb_run_name or f"{args.config}_stage{args.stage}"
        run = wandb.init(
            project=args.wandb_project,
            name=run_name,
            config={
                "config": args.config,
                "stage": args.stage,
                "use_moe": config.use_moe,
                "batch_size": args.batch_size,
                "accumulation_steps": args.accumulation_steps,
                "max_length": args.max_length,
                "max_negatives": args.max_negatives,
                "lr": args.lr,
                "temp": args.temp,
                "margin": args.margin,
                "use_mrl": args.use_mrl,
                "epochs": args.epochs,
                "intermediate_size": config.intermediate_size,
            },
        )
        print(f"[wandb] 已启用,project={args.wandb_project} run={run_name}")
        return run
    except Exception as e:
        print(f"[wandb] 初始化失败,降级为纯 stdout: {e}")
        return None


def build_dataset(args):
    """根据 data_type 构造数据集。"""
    if args.data_type == "demo":
        args.jsonl_path = os.path.join(_ROOT, args.jsonl_path)
        write_demo_jsonl(args.jsonl_path, n=256)
        ds = TripletJsonlDataset(args.jsonl_path, max_negatives=args.max_negatives)
    elif args.data_type == "jsonl":
        ds = TripletJsonlDataset(args.jsonl_path, max_negatives=args.max_negatives)
    elif args.data_type == "hf":
        ds = HFEmbeddingDataset(args.hf_dataset, max_negatives=args.max_negatives)
    else:
        raise ValueError(args.data_type)
    # 子集采样(mini 验证用)
    if args.max_train_samples > 0:
        from torch.utils.data import Subset
        n = min(args.max_train_samples, len(ds))
        ds = Subset(ds, list(range(n)))
        print(f"[build_dataset] 子集采样:只用前 {n} 条")
    return ds


def encode_batch(model, batch_enc, device):
    """对一批 tokenize 后的输入跑模型,返回 L2 归一化向量。

    batch_enc: dict with input_ids / attention_mask
    返回: [N, D](已归一化)
    """
    ids = batch_enc["input_ids"].to(device)
    mask = batch_enc["attention_mask"].to(device)
    z = model(ids, attention_mask=mask, output_mrl=False)
    return z


def encode_batch_mrl(model, batch_enc, device, mrl_dims):
    """MRL 版:返回各维度切片向量列表。"""
    ids = batch_enc["input_ids"].to(device)
    mask = batch_enc["attention_mask"].to(device)
    return model(ids, attention_mask=mask, output_mrl=True)


def compute_stage1_loss(model, batch, device, temp):
    """Stage1:纯 in-batch InfoNCE,正样本就是 batch 内对齐的 positive。

    不使用 hard negatives,依赖大 batch 提供足够负样本。
    重要:stage1(弱监督)不加假负样本 mask —— Qwen3 报告里 mask 仅用于 stage2
    的精细监督。弱监督阶段数据噪声大、模型未充分训练,mask 会误伤真负样本,
    因此 stage1 用标准 InfoNCE(对所有 in-batch 样本一视同仁)。
    """
    q = encode_batch(model, batch["query"], device)        # [B, D]
    d_pos = encode_batch(model, batch["positive"], device)  # [B, D]
    # 标准 InfoNCE:q·d_pos^T / temp,对角线为正,in-batch 其他全为负
    logits = (q @ d_pos.t()) / temp
    labels = torch.arange(q.size(0), device=device)
    return torch.nn.functional.cross_entropy(logits, labels)


def compute_stage2_loss(model, batch, device, args):
    """Stage2:带 hard negatives + 假负样本 mask +(可选)MRL。"""
    if args.use_mrl:
        # MRL:各维度切片分别算 loss
        q_list = encode_batch_mrl(model, batch["query"], device, MRL_DIMS)
        dp_list = encode_batch_mrl(model, batch["positive"], device, MRL_DIMS)
        if batch["negatives"] is not None:
            # 负样本只编码一次(完整维度),然后切片
            from shared.utils import last_token_pool
            ids = batch["negatives"]["input_ids"].to(device)
            mask = batch["negatives"]["attention_mask"].to(device)
            hidden = model.model(ids, attention_mask=mask, use_cache=False)[0]
            pooled = last_token_pool(hidden, mask)
            neg_full = model.head(pooled)  # [N, D] 未归一化
            n_neg = pooled.size(0)
            B = q_list[0].size(0)
            n_per_q = batch["n_neg_per_q"].tolist()
            # 切成 [B, K, D](K=max_neg)
            dn_list = []
            for d in MRL_DIMS:
                neg_norm = torch.nn.functional.normalize(neg_full[..., :d], p=2, dim=1)
                # 简化:假设所有 query 负样本数相同(max_neg),否则需 padding
                if len(set(n_per_q)) == 1 and n_per_q[0] > 0:
                    K = n_per_q[0]
                    dn_list.append(neg_norm.view(B, K, d))
                else:
                    dn_list.append(None)
            return matryoshka_infonce_loss(
                q_list, dp_list, dn_list, temp=args.temp, margin=args.margin
            )
        else:
            return matryoshka_infonce_loss(
                q_list, dp_list, None, temp=args.temp, margin=args.margin
            )
    else:
        q = encode_batch(model, batch["query"], device)
        d_pos = encode_batch(model, batch["positive"], device)
        d_neg = None
        if batch["negatives"] is not None:
            d_neg_flat = encode_batch(model, batch["negatives"], device)  # [N, D]
            n_per_q = batch["n_neg_per_q"].tolist()
            if len(set(n_per_q)) == 1 and n_per_q[0] > 0:
                B = q.size(0)
                K = n_per_q[0]
                d_neg = d_neg_flat.view(B, K, -1)
            # 变长负样本:退化为 stage1(只用 in-batch),保证不出错
        return infonce_with_false_neg_mask(
            q, d_pos, d_neg, temp=args.temp, margin=args.margin
        )


def save_checkpoint(model, args, config, step, tokenizer):
    os.makedirs(args.save_dir, exist_ok=True)
    moe = "_moe" if config.use_moe else ""
    name = f"embedding_stage{args.stage}_{config.hidden_size}{moe}.pth"
    path = os.path.join(args.save_dir, name)
    raw = model.module if hasattr(model, "module") else model
    state = {k: v.half().cpu() for k, v in raw.state_dict().items()}
    torch.save(state, path)
    print(f"[save] {path} (step {step})")


def main():
    args = parse_args()
    # stage 自动决定默认 lr
    if args.lr is None:
        args.lr = 2e-4 if args.stage == 1 else 1e-5
    torch.manual_seed(args.seed)

    device = torch.device(args.device)
    print(f"== MiniMind-Embedding 训练 (stage{args.stage}, {args.config}) ==")

    # 1) 配置 + tokenizer
    config = get_config(args.config)
    tokenizer = load_tokenizer(padding_side="left")
    # 确认模型词表与 tokenizer 实际长度一致
    # (HF tokenizer 的 .vocab_size 不随 add_special_tokens 更新,用 len() 才准)
    assert config.vocab_size == len(tokenizer), \
        f"vocab_size 不匹配: config={config.vocab_size} vs tokenizer len={len(tokenizer)}"

    # 2) 模型
    from shared.model import MiniMindForEmbedding
    model = MiniMindForEmbedding(config)
    # resize embedding 以容纳 <yes>/<no>(即使 embedding 不用,底座共享对齐 rerank)
    model.model.embed_tokens.weight.data.normal_(mean=0, std=0.02)  # 默认初始化
    # 加载预训练底座(若指定)
    if args.from_weight != "none":
        load_pretrained_backbone(
            model, hidden_size=config.hidden_size, use_moe=config.use_moe,
            from_weight=args.from_weight, save_dir=args.backbone_dir, device=args.device,
        )
    model.to(device)
    print(f"参数量: {compute_num_params(model, config)}")

    # 3) 数据
    ds = build_dataset(args)
    collator = EmbeddingCollator(
        tokenizer, max_length=args.max_length, max_negatives=args.max_negatives,
        instruction=DEFAULT_QUERY_INSTRUCTION if args.stage == 1 else None,
    )
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=True, collate_fn=collator,
        num_workers=0, drop_last=True,
    )

    # 4) 优化器
    total_steps = math.ceil(len(loader) / args.accumulation_steps) * args.epochs
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)
    print(f"总步数: {total_steps}, batch={args.batch_size}, accum={args.accumulation_steps}")

    # wandb
    wandb_run = init_wandb(args, config)

    # 5) 训练循环
    model.train()
    global_step = 0
    accum_step = 0
    optim.zero_grad()
    running_loss = 0.0

    for epoch in range(args.epochs):
        for batch in loader:
            # 前向 + 损失
            if args.stage == 1:
                loss = compute_stage1_loss(model, batch, device, args.temp)
            else:
                loss = compute_stage2_loss(model, batch, device, args)

            # 反向(梯度累积)
            (loss / args.accumulation_steps).backward()
            running_loss += loss.item()
            accum_step += 1

            if accum_step % args.accumulation_steps == 0:
                # warmup + 余弦
                if global_step < args.warmup_steps:
                    lr_scale = (global_step + 1) / args.warmup_steps
                else:
                    lr_scale = get_lr(global_step - args.warmup_steps,
                                      total_steps - args.warmup_steps, 1.0)
                for pg in optim.param_groups:
                    pg["lr"] = args.lr * lr_scale
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optim.step()
                optim.zero_grad()
                global_step += 1

                if global_step % args.log_steps == 0:
                    avg = running_loss / (args.log_steps * args.accumulation_steps)
                    cur_lr = optim.param_groups[0]["lr"]
                    print(f"epoch {epoch} step {global_step}/{total_steps} "
                          f"loss {avg:.4f} lr {cur_lr:.2e}")
                    if wandb_run is not None:
                        import wandb
                        wandb.log({
                            "loss": avg,
                            "lr": cur_lr,
                            "epoch": epoch,
                            "step": global_step,
                            "progress": global_step / total_steps,
                        }, step=global_step)
                    running_loss = 0.0
                if global_step % args.save_steps == 0:
                    save_checkpoint(model, args, config, global_step, tokenizer)

    save_checkpoint(model, args, config, global_step, tokenizer)
    print(f"== 训练完成,共 {global_step} 步 ==")
    if wandb_run is not None:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    main()
