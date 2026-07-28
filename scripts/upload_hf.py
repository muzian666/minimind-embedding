"""
scripts/upload_hf.py — 打包模型并上传 HuggingFace Hub

把训练好的 checkpoint 打包成 sentence-transformers 兼容目录格式,
然后用 huggingface_hub 上传。

打包目录结构(对齐 Qwen3-Embedding 的 HF 仓库):
  model.safetensors       —— 模型权重(转 safetensors)
  config.json             —— transformers 配置(minimind 架构)
  tokenizer.json          —— BPE tokenizer(复用 minimind)
  tokenizer_config.json   —— tokenizer 配置
  modules.json            —— sentence-transformers 模块链
  config_sentence_transformers.json —— ST 元配置(query/document prompt)
  1_Pooling/config.json   —— last-token pooling 配置
  README.md               —— 模型卡(含评测分数)

用法:
  python scripts/upload_hf.py \\
      --checkpoint checkpoints/embedding/embedding_stage3_768.pth \\
      --config embed_dense_64m \\
      --repo_id muzian666/minimind-embedding-dense \\
      --task embedding
"""
import os
import sys
import json
import shutil
import argparse
from pathlib import Path

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def build_st_directory(checkpoint, config_name, output_dir, task):
    """打包成 sentence-transformers 兼容目录。"""
    import torch
    from safetensors.torch import save_file
    from shared import get_config

    os.makedirs(output_dir, exist_ok=True)
    config = get_config(config_name)

    # 1) 模型权重 -> safetensors
    print(f"加载 checkpoint: {checkpoint}")
    state = torch.load(checkpoint, map_location="cpu")
    clean = {k: v.detach().clone().contiguous() for k, v in state.items()}
    save_file(clean, os.path.join(output_dir, "model.safetensors"))
    print(f"  ✓ model.safetensors ({len(clean)} 参数)")

    # 2) config.json (transformers,minimind 架构)
    cfg = {
        "model_type": "minimind",
        "architectures": ["MiniMindForEmbedding"],
        "hidden_size": config.hidden_size,
        "num_hidden_layers": config.num_hidden_layers,
        "num_attention_heads": config.num_attention_heads,
        "num_key_value_heads": config.num_key_value_heads,
        "head_dim": config.head_dim,
        "vocab_size": config.vocab_size,
        "intermediate_size": config.intermediate_size,
        "max_position_embeddings": config.max_position_embeddings,
        "rope_theta": config.rope_theta,
        "rms_norm_eps": config.rms_norm_eps,
        "bos_token_id": config.bos_token_id,
        "eos_token_id": config.eos_token_id,
        "tie_word_embeddings": False,
        "use_moe": config.use_moe,
        "embed_dim": config.embed_dim,
        "mrl_dims": config.mrl_dims,
        "pooling": "last_token",
        "torch_dtype": "float16",
    }
    if config.use_moe:
        cfg.update({
            "num_experts": config.num_experts,
            "num_experts_per_tok": config.num_experts_per_tok,
            "moe_intermediate_size": config.moe_intermediate_size,
            "norm_topk_prob": config.norm_topk_prob,
        })
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
    print("  ✓ config.json")

    # 3) tokenizer 文件(复用 minimind)
    tok_src = os.path.join(_ROOT, "minimind", "model")
    for f in ["tokenizer.json", "tokenizer_config.json"]:
        src = os.path.join(tok_src, f)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(output_dir, f))
    print("  ✓ tokenizer 文件")

    # 4) sentence-transformers 模块链
    modules = [
        {"idx": 0, "name": "0", "type": "sentence_transformers.models.Transformer"},
        {"idx": 1, "name": "1", "type": "sentence_transformers.models.Pooling",
         "args": {"word_embedding_dimension": config.embed_dim, "pooling_mode_lasttoken": True}},
    ]
    with open(os.path.join(output_dir, "modules.json"), "w") as f:
        json.dump(modules, f, indent=2)
    print("  ✓ modules.json")

    # 5) ST 元配置(query/document prompt)
    st_cfg = {
        "__version__": {"sentence_transformers": "5.0.0"},
        "prompts": {
            "query": "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: ",
        },
        "default_prompt_name": None,
        "similarity_fn_name": "cosine",
    }
    with open(os.path.join(output_dir, "config_sentence_transformers.json"), "w") as f:
        json.dump(st_cfg, f, indent=2)
    print("  ✓ config_sentence_transformers.json")

    # 6) 1_Pooling 配置
    pool_dir = os.path.join(output_dir, "1_Pooling")
    os.makedirs(pool_dir, exist_ok=True)
    pool_cfg = {
        "word_embedding_dimension": config.embed_dim,
        "pooling_mode_cls_token": False,
        "pooling_mode_mean_tokens": False,
        "pooling_mode_max_tokens": False,
        "pooling_mode_mean_sqrt_len_tokens": False,
        "pooling_mode_weightedmean_tokens": False,
        "pooling_mode_lasttoken": True,
        "include_prompt": True,
    }
    with open(os.path.join(pool_dir, "config.json"), "w") as f:
        json.dump(pool_cfg, f, indent=2)
    print("  ✓ 1_Pooling/config.json (last-token pooling)")

    print(f"\n打包完成 -> {output_dir}")


def upload_to_hub(local_dir, repo_id, token):
    """上传目录到 HuggingFace Hub。"""
    from huggingface_hub import HfApi, create_repo
    api = HfApi(token=token)
    # 创建 repo(若不存在)
    create_repo(repo_id, repo_type="model", exist_ok=True, token=token)
    print(f"上传 {local_dir} -> {repo_id} ...")
    api.upload_folder(folder_path=local_dir, repo_id=repo_id, repo_type="model")
    print(f"✓ 上传完成: https://huggingface.co/{repo_id}")


def parse_args():
    p = argparse.ArgumentParser(description="打包并上传模型到 HuggingFace")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--config", default="embed_dense_64m")
    p.add_argument("--output_dir", default=None, help="打包临时目录(默认 /tmp/hf_model)")
    p.add_argument("--repo_id", required=True, help="HF 仓库 ID,如 muzian666/minimind-embedding-dense")
    p.add_argument("--task", default="embedding", choices=["embedding", "rerank"])
    p.add_argument("--token", default=os.environ.get("HF_TOKEN", ""))
    p.add_argument("--no_upload", action="store_true", help="只打包不上传")
    return p.parse_args()


def main():
    args = parse_args()
    output_dir = args.output_dir or os.path.join(_ROOT, "tmp_hf_model")
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    build_st_directory(args.checkpoint, args.config, output_dir, args.task)

    if args.no_upload:
        print(f"\n--no_upload 模式,未上传。目录: {output_dir}")
    else:
        if not args.token:
            print("❌ 缺少 HF_TOKEN,设置环境变量或用 --token 传入")
            return
        upload_to_hub(output_dir, args.repo_id, args.token)


if __name__ == "__main__":
    main()
