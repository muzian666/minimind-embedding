"""
minimind_rerank/train.py — Rerank 模型训练脚本

pointwise yes/no 路线(对齐 Qwen3-Reranker):
  把"判断 query-doc 是否相关"建模为下一个 token 预测——
  模型在 [Query]\n[Document] 后输出 <yes> 或 <no>。
  复用预训练 lm_head,不引入新参数。

单阶段监督微调(Qwen3-Reranker 也跳过弱监督预训练)。

用法(smoke test):
  python -m minimind_rerank.train --data_type demo --batch_size 8 --epochs 3 --device cuda

真实训练:
  python -m minimind_rerank.train --data_type hf --hf_dataset t2reranking \
      --from_weight pretrain --batch_size 16 --epochs 1 --device cuda
"""
import os
import sys
import argparse
import math

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from shared import (
    load_tokenizer, load_pretrained_backbone, compute_num_params,
    get_lr, YES_TOKEN_ID, NO_TOKEN_ID,
)
from shared.model import MiniMindForRerank
from minimind_rerank.dataset import (
    RerankJsonlDataset, HFRerankDataset, RerankCollator, write_demo_rerank_jsonl,
)


def parse_args():
    p = argparse.ArgumentParser(description="MiniMind-Rerank 训练")
    p.add_argument("--config", default="rerank_dense_64m",
                   choices=["rerank_dense_64m", "rerank_moe_198m"])
    p.add_argument("--from_weight", default="pretrain",
                   help="底座来源:pretrain/full_sft/none")
    p.add_argument("--backbone_dir", default=None)
    # 数据
    p.add_argument("--data_type", default="demo", choices=["demo", "jsonl", "hf"])
    p.add_argument("--jsonl_path", default="data/rerank_train.jsonl")
    p.add_argument("--hf_dataset", default="t2reranking")
    p.add_argument("--max_train_samples", type=int, default=0)
    # 训练
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--accumulation_steps", type=int, default=1)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--warmup_steps", type=int, default=0)
    p.add_argument("--save_dir", default="checkpoints/rerank")
    p.add_argument("--save_steps", type=int, default=2000)
    p.add_argument("--log_steps", type=int, default=50)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    # wandb
    p.add_argument("--wandb_project", default="minimind-embedding")
    p.add_argument("--wandb_run_name", default=None)
    p.add_argument("--no_wandb", action="store_true")
    return p.parse_args()


def init_wandb(args, config):
    if args.no_wandb:
        return None
    try:
        import wandb
        run_name = args.wandb_run_name or f"{args.config}_rerank"
        run = wandb.init(
            project=args.wandb_project, name=run_name,
            config={
                "config": args.config, "task": "rerank",
                "use_moe": config.use_moe,
                "batch_size": args.batch_size, "lr": args.lr,
                "max_length": args.max_length, "epochs": args.epochs,
                "yes_token_id": YES_TOKEN_ID, "no_token_id": NO_TOKEN_ID,
            },
        )
        print(f"[wandb] 已启用,run={run_name}")
        return run
    except Exception as e:
        print(f"[wandb] 初始化失败,降级为纯 stdout: {e}")
        return None


def build_dataset(args):
    if args.data_type == "demo":
        args.jsonl_path = os.path.join(_ROOT, args.jsonl_path)
        write_demo_rerank_jsonl(args.jsonl_path, n=128)
        ds = RerankJsonlDataset(args.jsonl_path)
    elif args.data_type == "jsonl":
        ds = RerankJsonlDataset(args.jsonl_path)
    elif args.data_type == "hf":
        ds = HFRerankDataset(args.hf_dataset)
    else:
        raise ValueError(args.data_type)
    if args.max_train_samples > 0:
        n = min(args.max_train_samples, len(ds))
        ds = Subset(ds, list(range(n)))
        print(f"[build_dataset] 子集采样:只用前 {n} 条")
    return ds


def save_checkpoint(model, args, config, step):
    os.makedirs(args.save_dir, exist_ok=True)
    moe = "_moe" if config.use_moe else ""
    name = f"rerank_{config.hidden_size}{moe}.pth"
    path = os.path.join(args.save_dir, name)
    raw = model.module if hasattr(model, "module") else model
    state = {k: v.half().cpu() for k, v in raw.state_dict().items()}
    torch.save(state, path)
    print(f"[save] {path} (step {step})")


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    print(f"== MiniMind-Rerank 训练 ({args.config}) ==")

    config = get_rerank_config(args.config)
    tokenizer = load_tokenizer(padding_side="left")
    assert config.vocab_size == len(tokenizer), \
        f"vocab_size 不匹配: config={config.vocab_size} vs tokenizer len={len(tokenizer)}"

    # 模型
    model = MiniMindForRerank(config)
    if args.from_weight != "none":
        load_pretrained_backbone(
            model, hidden_size=config.hidden_size, use_moe=config.use_moe,
            from_weight=args.from_weight, save_dir=args.backbone_dir, device=args.device,
        )
    model.to(device)
    print(f"参数量: {compute_num_params(model, config)}")
    print(f"yes_token_id={config.yes_token_id}, no_token_id={config.no_token_id}")

    # 数据
    ds = build_dataset(args)
    collator = RerankCollator(tokenizer, max_length=args.max_length)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                        collate_fn=collator, num_workers=0, drop_last=True)

    # 优化器
    total_steps = math.ceil(len(loader) / args.accumulation_steps) * args.epochs
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)
    print(f"总步数: {total_steps}, batch={args.batch_size}, accum={args.accumulation_steps}")
    wandb_run = init_wandb(args, config)

    # 训练循环
    model.train()
    global_step = 0
    accum_step = 0
    optim.zero_grad()
    running_loss = 0.0
    running_acc = 0.0

    for epoch in range(args.epochs):
        for batch in loader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # 前向:返回 [B, 2] 的 (yes_logit, no_logit)
            yn_logits = model(ids, attention_mask=mask)
            # pointwise 交叉熵:label=1 → yes, label=0 → no
            loss = F.cross_entropy(yn_logits, labels)

            (loss / args.accumulation_steps).backward()
            running_loss += loss.item()
            # 准确率(诊断用)
            with torch.no_grad():
                pred = yn_logits.argmax(dim=-1)
                running_acc += (pred == labels).float().mean().item()
            accum_step += 1

            if accum_step % args.accumulation_steps == 0:
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
                    n = args.log_steps * args.accumulation_steps
                    avg_loss = running_loss / n
                    avg_acc = running_acc / n
                    cur_lr = optim.param_groups[0]["lr"]
                    print(f"epoch {epoch} step {global_step}/{total_steps} "
                          f"loss {avg_loss:.4f} acc {avg_acc:.3f} lr {cur_lr:.2e}")
                    if wandb_run is not None:
                        import wandb
                        wandb.log({
                            "loss": avg_loss, "accuracy": avg_acc, "lr": cur_lr,
                            "epoch": epoch, "step": global_step,
                            "progress": global_step / total_steps,
                        }, step=global_step)
                    running_loss = 0.0
                    running_acc = 0.0
                if global_step % args.save_steps == 0:
                    save_checkpoint(model, args, config, global_step)

    save_checkpoint(model, args, config, global_step)
    print(f"== 训练完成,共 {global_step} 步 ==")
    if wandb_run is not None:
        import wandb
        wandb.finish()


def get_rerank_config(name):
    from shared.configs import rerank_dense_64m, rerank_moe_198m
    return {"rerank_dense_64m": rerank_dense_64m, "rerank_moe_198m": rerank_moe_198m}[name]()


if __name__ == "__main__":
    main()
