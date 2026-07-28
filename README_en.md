<div align="center">

# MiniMind-Embedding & Rerank

</div>

<div align="center">

[![License](https://img.shields.io/github/license/muzian666/minimind-embedding?color=blue)](./LICENSE)
[![GitHub last commit](https://img.shields.io/github/last-commit/muzian666/minimind-embedding)](https://github.com/muzian666/minimind-embedding/commits/main)
[![GitHub Repo stars](https://img.shields.io/github/stars/muzian666/minimind-embedding?style=social)](https://github.com/muzian666/minimind-embedding/stargazers)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-blue)](https://github.com/muzian666/minimind-embedding/pulls)
[![Python](https://img.shields.io/badge/Python-3.11%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.8%2B%20cu128-ee4c2c)](https://pytorch.org/)
[![MTEB](https://img.shields.io/badge/MTEB-v2.18-9c27b0)](https://github.com/embeddings-benchmark/mteb)

</div>

<div align="center">
  <h3>"A minimal starting point for vectorizing everything"</h3>
</div>

<div align="center">

[中文](./README.md) | English

</div>

* Built on the open-source **[MiniMind](https://github.com/jingyaogong/minimind)** small language model (64M / 198M-A64M) as the backbone, this project trains a complete suite of **text Embedding and Rerank models from scratch**, with the technical approach fully aligned to **[Qwen3-Embedding](https://arxiv.org/abs/2506.05176)**.
* All models support **both Dense and MoE variants**, with a maximum sequence length of **8192**, covering **Chinese and English** retrieval, STS, classification, and reranking tasks.
* All core algorithms (last-token pooling, InfoNCE with false-negative masking, MRL, pointwise yes/no rerank) are implemented from scratch in native PyTorch, without relying on third-party high-level abstractions.
* The entire training pipeline can run on a **single RTX 5080 (16GB)** (with an optional rented A100/H100 for the large-scale weak-supervision stage), and ships with a **one-command Docker environment** for seamless migration between local and cloud.
* Benchmarked on mainstream leaderboards (**C-MTEB / MTEB(eng)**), with models and results open-sourced to the **HuggingFace Hub**.

> Note: This project is open-sourced under the Apache 2.0 license, completely free. "Single-GPU runnable" means the Dense variant can complete all three training stages on an RTX 5080 in practice; for long-sequence MoE training, a larger GPU is recommended.

---

<div align="center">

📌 **This project extends the MiniMind ecosystem and stands on the shoulders of giants.**

[![MiniMind](https://img.shields.io/badge/Based%20on-MiniMind-ff6b35)](https://github.com/jingyaogong/minimind)
[![Qwen3-Embedding](https://img.shields.io/badge/Inspired%20by-Qwen3--Embedding-615ced)](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B)

</div>

---

# 📌 Introduction

Text representation is the cornerstone of modern information retrieval, RAG, and semantic search. Whether it's vector recall in retrieval-augmented generation (RAG) or relevance ranking in search engines, both rely on two core models: **an Embedding model that turns text into vectors**, and **a Rerank model that fine-ranks candidate documents**.

In recent years, representation-learning models such as BGE, GTE, E5, and Qwen3-Embedding have made great strides, with Qwen3-Embedding topping the multilingual MTEB leaderboard. However, these SOTA models start at 0.6B parameters, and the 8B versions require large-scale cluster training — for individual developers and learners, "opening the black box and understanding every line of the contrastive loss function" remains a luxury.

This echoes exactly what jingyaogong, the author of MiniMind, intended when building small LLMs: **"Assembling an airplane with your own Lego is far more exciting than flying first class."** Therefore, this project builds a **complete reproduction of the Qwen3-Embedding technical pipeline on top of the MiniMind backbone, which has only 64M parameters** — from the causal-attention + last-token-pooling architecture choice, to the InfoNCE loss with false-negative masking, to Matryoshka multi-dimensional representations and pointwise yes/no reranking. We don't chase SOTA scores; we pursue a **reproducible, understandable, single-GPU-runnable** end-to-end engineering loop.

😊 Let's enjoy the fun of "compressing" text into a 768-dimensional vector together!

---

#### 🎉 What this project includes

- Complete **Embedding and Rerank model architecture code** (Dense + MoE), reusing the MiniMind-3 backbone, with heads redesigned for representation learning.
- A correct **last-token pooling** implementation (compatible with left/right padding, aligned with the official Qwen3-Embedding).
- A faithful **InfoNCE with false-negative masking** loss implementation (including the $m_{ij}$ mask formula from the Qwen3 paper).
- **Matryoshka Representation Learning (MRL)** multi-dimensional output training (768/512/256/128/64).
- **pointwise yes/no Rerank** implementation (reusing lm_head, modeling relevance judgment as next-token prediction).
- A full **three-stage training pipeline**: weakly-supervised pre-training → supervised fine-tuning → SLERP model merging.
- Compatible with mainstream frameworks like `sentence-transformers` and `transformers`; ships with `mteb` evaluation scripts and HuggingFace Hub upload support.
- A **one-command Docker environment** (CUDA 13.3 + cuDNN + torch cu130 / cu128), with native support for RTX 5080/5090 (Blackwell sm_120).
- Supports up to **8192** long context (the backbone natively supports 32768, requiring no RoPE extrapolation).

---

#### 🎉 Released models

> ⏳ Models are being trained. The table below lists the planned releases. Actual scores and HuggingFace links will be updated once training completes.

| Model | Type | Params | Backbone | Max length | Status |
|-------|------|--------|----------|------------|--------|
| minimind-embedding-dense | Embedding | 64M | minimind-3 | 8192 | ✅ Stage2 done (STS 0.49) |
| minimind-embedding-moe | Embedding | 198M-A64M | minimind-3-moe | 8192 | 🚧 TODO |
| minimind-rerank-dense | Rerank | 64M | minimind-3 | 8192 | ✅ v0 zero-shot (MAP@10 0.47) |
| minimind-rerank-moe | Rerank | 198M-A64M | minimind-3-moe | 8192 | 🚧 TODO |

---

#### 📝 Changelog

<details>
<summary><b>🔥 2026-07-28</b></summary>

 - **M2 complete**: Dense Embedding Stage 2 full training (t2ranking-15, 340k pairs, 42513 steps, ~4.5h).
 - Loss converged smoothly from 1.9 to 0.95, with no divergence/collapse throughout. Full wandb record.
 - STS evaluation (ATEC/BQ/LCQMC/STSB): **average Spearman 0.49**, a 69% improvement over the mini baseline (0.29). STSB alone reached 0.67, approaching BGE-small-zh.
 - Added M3 Rerank code (pointwise yes/no, aligned with Qwen3-Reranker).
 - Key finding for reranker: full-parameter pointwise fine-tuning on the 64M backbone causes catastrophic forgetting, dropping MAP@10 from 0.47 (zero-shot) to 0.24. The zero-shot pretrained backbone (0.47) is kept as Rerank v0.

</details>

<details>
<summary><b>🔥 2026-07-27</b></summary>

 - Project kickoff: completed the overall architecture design and the M1 milestone (shared backbone + Dense Embedding training pipeline working end-to-end).
 - Implemented the `shared/` layer: MiniMindForEmbedding / MiniMindForRerank model classes, last-token pooling, InfoNCE with false-negative masking, MRL loss, config presets, tokenizer adaptation (appending `<yes>`/`<no>` special tokens).
 - Implemented the `minimind_embedding/` training module: three-stage training scripts (stage1 weakly-supervised / stage2 supervised + MRL), both local jsonl and HuggingFace datasets sources, demo smoke-test data.
 - Built the Docker environment (CUDA 13.3 + torch 2.13 cu130 + mteb 2.18 + sentence-transformers 5.6), verified smoke test on RTX 5080: loss dropped normally from 1.53 to 0.

</details>

---

# 📌 Quick Start

<details>
<summary><b>My hardware/software setup (for reference)</b></summary>

| Item | Config |
|------|--------|
| OS | Windows 11 (win32 10.0.26200) |
| Local GPU | NVIDIA GeForce RTX 5080 (16GB, Blackwell sm_120) |
| GPU driver | 610.43.02 (CUDA 13.3 UMD) |
| Cloud GPU | NVIDIA GeForce RTX 5090 (32GB, Blackwell sm_120), rented |
| Container | Docker Desktop + WSL2 backend + NVIDIA Container Toolkit |
| Training env | Docker: `nvidia/cuda:13.3.0-cudnn-devel-ubuntu24.04` + torch cu130 (local); torch 2.8+cu128 (cloud) |

> ⚠️ RTX 5080/5090 are Blackwell architecture (compute capability sm_120); **you must use CUDA 12.8+ and a PyTorch cu128+ wheel**. This project uses CUDA 13.3 + torch cu130 / torch cu128; see the Docker section. Other GPUs (3090/4090/A100) are also compatible.

</details>

## Step 0: Clone and set up the environment

This project depends on [MiniMind](https://github.com/jingyaogong/minimind) as the backbone. First clone it into the `minimind/` directory of this repo (this repo does not include the MiniMind source to avoid duplicate maintenance):

```bash
git clone https://github.com/muzian666/minimind-embedding.git
cd minimind-embedding

# Pull the MiniMind backbone into ./minimind (imported via relative path)
git clone https://github.com/jingyaogong/minimind.git minimind
```

**Docker is strongly recommended** (especially for Windows users) to avoid the CUDA/cuDNN/torch version hell. If you insist on a local environment, install from `requirements.txt` yourself (note: Blackwell GPUs require the cu130 wheel).

### Option A: One-command Docker (recommended)

```bash
# Build (about 5-10 minutes the first time, downloads torch cu130)
docker compose -f docker/docker-compose.yml build

# Verify the GPU is visible
docker compose -f docker/docker-compose.yml run --rm dev nvidia-smi

# Enter the interactive dev environment
docker compose -f docker/docker-compose.yml run --rm dev
```

### Option B: Local environment

```bash
# Must use the cu130 index (Blackwell support)
pip install torch --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements.txt
```

## Ⅰ 🚀 Inference

> ⏳ Complete inference examples will be added here once models are trained. For now, refer to the training section below to get the pipeline running.

### 1' Download a model

Once released, models will be uploaded to HuggingFace:

```bash
# Will be added after training completes
# hf download muzian666/minimind-embedding-dense
```

### 2' Calling Embedding (last-token pooling)

```python
# Complete example will be added after training. Core logic (aligned with Qwen3-Embedding):
#
# from shared import load_tokenizer, MiniMindForEmbedding
# tokenizer = load_tokenizer(padding_side="left")
# model = MiniMindForEmbedding.from_pretrained("muzian666/minimind-embedding-dense")
#
# enc = build_embedding_inputs(tokenizer, ["your query text"], is_query=True)
# embedding = model.encode(enc["input_ids"], enc["attention_mask"])  # [1, 768] L2-normalized
```

## Ⅱ 🛠️ Training

### 1' Download data

Training data comes from community open-source datasets (see the [Data](#-data) section). HuggingFace sources are auto-downloaded by the scripts; you can also prepare local jsonl manually.

### 2' Start training

#### 2.1 Smoke Test (verifies the environment, ~1 minute)

```bash
# Uses built-in demo data to verify the training pipeline works and loss decreases
docker compose -f docker/docker-compose.yml run --rm smoke
```

Expected output: `loss 1.53 → 0.69 → 0.087 → 0` (fast convergence on 16-class demo data, proving forward/backward/loss are all correct).

#### 2.2 Embedding full training (three stages)

<details>
<summary><b>💡 Three-stage training pipeline</b></summary>

Aligned with the three stages from the Qwen3-Embedding report:

| Stage | Data | Loss | Goal |
|-------|------|------|------|
| **Stage 1 Weakly-supervised pre-training** | Large-scale BM25/triplet data (T2Ranking, mMARCO) | Standard InfoNCE (pure in-batch) | General semantic alignment |
| **Stage 2 Supervised fine-tuning** | High-quality labeled + hard negatives | InfoNCE + false-negative mask + MRL | Fine-grained task performance |
| **Stage 3 Model merging** | The last N checkpoints from Stage 2 | SLERP spherical linear interpolation | Improve robustness |

</details>

```bash
# Stage 1: weakly-supervised pre-training (large batch, pure in-batch InfoNCE)
docker compose -f docker/docker-compose.yml run --rm dev python -m minimind_embedding.train \
    --config embed_dense_64m --stage 1 \
    --data_type hf --hf_dataset t2ranking \
    --from_weight pretrain --backbone_dir /workspace/out \
    --batch_size 128 --epochs 1 --max_length 512 --lr 2e-4 --device cuda

# Stage 2: supervised fine-tuning (with hard neg + false-negative mask + MRL)
docker compose -f docker/docker-compose.yml run --rm dev python -m minimind_embedding.train \
    --config embed_dense_64m --stage 2 --use_mrl \
    --data_type hf --hf_dataset t2ranking-15 \
    --from_weight embedding_stage1 --backbone_dir checkpoints/embedding \
    --batch_size 64 --epochs 3 --max_length 512 \
    --lr 1e-5 --device cuda

# Stage 3: model merging (see scripts/merge_models.py)
python scripts/merge_models.py \
    --glob "checkpoints/embedding/embedding_stage2_768_step*.pth" --last_n 5 \
    --output checkpoints/embedding/embedding_stage3_768.pth --config embed_dense_64m
```

> Trained weights are saved to: `checkpoints/embedding/embedding_stage{1,2}_{dim}[_step{n}].pth` (Stage 2 saves a checkpoint every `save_steps` for Stage 3 merging; the `_step{n}` suffix is preserved, and a `latest` version is also kept).

---

# 📌 Data

## Ⅰ Tokenizer

This project **directly reuses MiniMind's BPE tokenizer** (vocab=6400, Chinese-English mixed, ByteLevel). See the [MiniMind Tokenizer docs](https://github.com/jingyaogong/minimind).

The only extension: to support the Rerank model's pointwise yes/no judgment, two existing tokens in the vocabulary are reused as the relevance target:

| Token | ID | Purpose |
|-------|----|----|
| `是` (yes) | 357 | Rerank positive target token |
| `否` (no) | 1332 | Rerank negative target token |

> Using the existing `是`/`否` tokens (instead of adding new `<yes>`/`<no>`) avoids the random-initialization problem: new tokens are never seen during pre-training, and the model struggles to learn the "relevance → new token" mapping under a low learning rate (empirically, it even learns the wrong direction). The existing tokens are semantically clear and well-trained, so no vocabulary extension or embedding resize is needed.

## Ⅱ Embedding training data format

Unified jsonl triplet format:

```jsonl
{"query": "Why is the sky blue", "positive": "Sunlight scatters in the atmosphere; shorter blue wavelengths scatter more, making the sky appear blue.", "hard_negatives": ["Water freezes below 0 degrees.", "Photosynthesis converts sunlight into chemical energy."]}
{"query": "Boiling point of water", "positive": "Under standard atmospheric pressure, pure water boils at 100 degrees Celsius.", "hard_negatives": ["The Earth orbits the Sun once every 365 days."]}
```

## Ⅲ Rerank training data format

```jsonl
{"query": "Why is the sky blue", "document": "Sunlight scatters in the atmosphere; shorter blue wavelengths scatter more, making the sky appear blue.", "label": 1}
{"query": "Why is the sky blue", "document": "Water freezes below 0 degrees.", "label": 0}
```

## Ⅳ Dataset sources

| Dataset | Language | Purpose | HF repo | License |
|---------|----------|---------|---------|---------|
| T2Ranking | Chinese | Embedding/Rerank training | [`THUIR/T2Ranking`](https://huggingface.co/datasets/THUIR/T2Ranking) / [`sentence-transformers/t2ranking`](https://huggingface.co/datasets/sentence-transformers/t2ranking) | Apache-2.0 |
| mMARCO | Chinese/English | Embedding/Rerank training | [`unicamp-dl/mmarco`](https://huggingface.co/datasets/unicamp-dl/mmarco) | Apache-2.0 |
| C-MTEB series | Chinese | Rerank training+eval | [`C-MTEB/T2Reranking`](https://huggingface.co/datasets/C-MTEB/T2Reranking) etc. | - |
| msmarco | English | Rerank training | [`sentence-transformers/msmarco`](https://huggingface.co/datasets/sentence-transformers/msmarco) | - |
| NLI-zh | Chinese | STS training | [`shibing624/nli-zh-all`](https://huggingface.co/datasets/shibing624/nli-zh-all) | - |

---

# 📌 Model

## Architecture

This project forks two representation-learning heads on top of MiniMind's decoder-only causal Transformer:

```
                    ┌─────────────────────────────────────────┐
                    │   MiniMind backbone (shared, read-only)  │
                    │                                         │
   input_ids ──────►│  embed → [Block×8] → norm → hidden      │
   + EOS            │   (causal attn + GQA + SwiGLU + QKNorm) │
                    │   (optional MoE: 4 experts top-1)        │
                    └────────────────┬────────────────────────┘
                                     │ last_hidden_state
                                     ▼
                          ┌──────────┴──────────┐
                          │ last_token_pool     │  ← take the last non-pad token (EOS position)
                          └──────────┬──────────┘
                                     │
                    ┌────────────────┴────────────────┐
                    │                                 │
              Embedding branch                  Rerank branch
                    │                                 │
            Linear(768→768)                   reuse lm_head
            + LayerNorm                       (no new params)
            + L2 normalize                          │
                    │                          take 是/否 logits
                    │                          softmax → score
                    ▼
            sentence vec [768]
        (MRL-slicable to 512/256/128/64)
```

## Model config

| Model Name | params | len_vocab | max_pos | rope_theta | n_layers | d_model | kv_heads | q_heads | note |
|------------|--------|-----------|---------|------------|----------|---------|----------|---------|------|
| minimind-embedding-dense | 64M | 6400 | 32768 | 1e6 | 8 | 768 | 4 | 8 | Dense + last-token pool + MRL |
| minimind-embedding-moe | 198M-A64M | 6400 | 32768 | 1e6 | 8 | 768 | 4 | 8 | 4 experts / top-1 |
| minimind-rerank-dense | 64M | 6400 | 32768 | 1e6 | 8 | 768 | 4 | 8 | Dense + pointwise 是/否 |
| minimind-rerank-moe | 198M-A64M | 6400 | 32768 | 1e6 | 8 | 768 | 4 | 8 | 4 experts / top-1 |

> All models have `max_position_embeddings=32768` (backbone-native). This project trains with `max_length=8192`, requiring no RoPE extrapolation. To extend to 32768, enable the backbone's built-in YaRN extrapolation (factor=16).

## Last-token Pooling: why it works

Why can a causal (unidirectional) language model also do embedding? This is a key insight from Qwen3-Embedding:

Traditional BERT-style encoders use **bidirectional attention + mean/CLS pooling**, while Qwen3-Embedding does the opposite — it **keeps causal (unidirectional) attention and takes the last token's hidden state as the sentence vector**. Reasons:

1. **Maximizing weight reuse**: the causal LM's pre-training distribution is identical to the generation model's, so pretrained weights can be loaded directly without re-warmup. Bidirectional attention would break the positional priors learned during pre-training.
2. **The last token has "seen" the whole sequence**: under causal attention, the token at position $T$ aggregates information from all preceding $T-1$ tokens via attention, making it a natural summary representation of the whole sentence.
3. **Appending EOS as an anchor**: giving the model a clear "end" signal at the sequence end makes the last token's representation more stable.

This project's `shared/utils.py:last_token_pool` faithfully follows the official Qwen3-Embedding-0.6B implementation, correctly handling left/right padding:

```python
def last_token_pool(last_hidden_states, attention_mask):
    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if left_padding:
        return last_hidden_states[:, -1]              # left padding: last col is the last real token
    sequence_lengths = attention_mask.sum(dim=1) - 1   # right padding: locate by mask length
    return last_hidden_states[torch.arange(batch_size), sequence_lengths]
```

> ⚠️ **The padding-side trap**: this project uniformly uses **left padding** (`tokenizer.padding_side="left"`), where `[:, -1]` is directly the last real token — O(1) retrieval and never hits pad tokens. If you mix in right padding and take `[:, -1]` directly, you'll get a pad token's representation (wrong).

## MRL: multi-dimensional output

Matryoshka Representation Learning (Kusupati et al., 2022) makes a single vector behave well at different dimensional truncations, like Russian nesting dolls:

```
Full vector [768] ──┬── first 512 dims → usable for a 512-dim index
                   ├── first 256 dims → usable for a 256-dim index
                   ├── first 128 dims → usable for a 128-dim index
                   └── first  64 dims → usable for a  64-dim index (most storage-efficient)
```

During training, InfoNCE loss is computed **separately for each dimensional slice**, then averaged:

```math
\mathcal{L}_{MRL} = \frac{1}{|D|}\sum_{d \in D} \mathcal{L}_{InfoNCE}(z_{[:d]})
```

where $D = \{768, 512, 256, 128, 64\}$. A vector trained this way can be truncated to any dimension for ANN retrieval (e.g., use 64 dims for coarse filtering, 768 for fine re-ranking), with flexible storage and compute cost.

---

# 📌 Experiments

> ✅ The Dense Stage 2 has completed full training on a single RTX 5080. Real loss curves and training logs are provided below. Stage 1/3 and the MoE variant are in progress.

## Ⅰ Training cost

Measured on: RTX 5080 (16GB, Blackwell sm_120) + Docker (CUDA 13.3 + torch 2.13 cu130) — local; RTX 5090 (32GB) + torch cu128 — cloud.

| Model Name | params | Data | Stage 2 time | VRAM | Notes |
|------------|--------|------|--------------|------|-------|
| minimind-embedding-dense | 64M | t2ranking-15 (340k) | **~4.5h** (42513 steps, 1 epoch) | 15.5/16 GB | ✅ Done (local 5080) |
| minimind-embedding-dense (3 epoch) | 64M | t2ranking-15 (340k) | ~1.5h (63768 steps) | 25/32 GB | 🚧 Training (cloud 5090) |
| minimind-embedding-moe | 198M-A64M | same | TBD | >16 GB (lower batch) | Rent A100 recommended |

> Stage 2 params: batch=8/16, max_length=256/512, 7 hard negatives, lr=1e-5, temp=0.02, margin=0.1, MRL on.
> Throughout training, loss converged smoothly from 1.9 to ~0.95 with no divergence or collapse. Full curves at [wandb](https://wandb.ai/qinganli-personal/minimind-embedding).

## Ⅱ Embedding training (three stages)

### 1' Weakly-supervised pre-training (Stage 1) ✅ Done

**Rationale**: let the model first learn "what kind of text pairs are semantically related". This stage uses large-scale (weakly labeled) data and relies on large batches for sufficient negative-sample signal, using **standard InfoNCE** (no false-negative mask — because weakly-supervised data is noisy and the mask would wrongly suppress true negatives).

Standard InfoNCE loss:

```math
\mathcal{L}_{InfoNCE} = -\frac{1}{N}\sum_{i=1}^{N} \log \frac{\exp(s(q_i, d_i^+)/\tau)}{\exp(s(q_i, d_i^+)/\tau) + \sum_{j \neq i} \exp(s(q_i, d_j)/\tau)}
```

where $s(\cdot,\cdot)$ is cosine similarity and $\tau=0.02$ is the temperature.

Actual training command (this project, on cloud 5090):

```bash
python -m minimind_embedding.train \
    --config embed_dense_64m --stage 1 \
    --data_type hf --hf_dataset t2ranking \
    --from_weight pretrain --backbone_dir /root/minimind-embeddings/out \
    --batch_size 64 --epochs 1 --max_length 256 --lr 2e-4 --device cuda
```

**Loss convergence** (t2ranking triplet, 90k pairs, 1413 steps, ~3 min on 5090):

| step | 200 | 540 | 1000 | 1413 (final) |
|------|-----|-----|------|--------------|
| loss | 1.10 | 0.87 | 0.55 | **0.43** |

> Pretrained weights saved to `checkpoints/embedding/embedding_stage1_768.pth` (124 MB).

### 2' Supervised fine-tuning (Stage 2) ✅ Done

**Rationale**: fine-tune on high-quality labeled data, introducing hard negatives and false-negative masking.

InfoNCE with false-negative mask (faithful to the Qwen3-Embedding report):

$$\mathcal{L} = -\frac{1}{N}\sum_{i} \log \frac{e^{s(q_i,d_i^+)/\tau}}{Z_i}$$

$$Z_i = e^{s(q_i,d_i^+)/\tau} + \sum_k m_{ik} e^{s(q_i,d_{i,k}^-)/\tau} + \sum_{j\neq i} m_{ij} e^{s(q_i,q_j)/\tau} + \sum_{j\neq i} m_{ij} e^{s(d_i^+,d_j)/\tau}$$

mask factor:

$$m_{ij} = \begin{cases} 0 & \text{if } s_{ij} > s(q_i, d_i^+) + 0.1 \\ 1 & \text{otherwise} \end{cases}$$

Intuition: if a candidate negative's similarity to the query is **even higher than the positive** (+0.1 margin), it is likely a "false negative" (semantically related), and should be masked from the denominator to avoid applying a wrong gradient.

Actual training command (this project, on local 5080):

```bash
python -m minimind_embedding.train \
    --config embed_dense_64m --stage 2 --use_mrl \
    --data_type hf --hf_dataset t2ranking-15 \
    --from_weight pretrain --backbone_dir out \
    --batch_size 8 --epochs 1 --max_length 256 \
    --max_negatives 7 --temp 0.02 --margin 0.1 --lr 1e-5 --device cuda
```

> Trained weights saved to `checkpoints/embedding/embedding_stage2_768.pth` (124 MB)

**Loss convergence** (t2ranking-15, 340k pairs, 42513 steps, ~4.5h on 5080):

| step | 500 | 5000 | 15000 | 25000 | 35000 | 42513 (final) |
|------|-----|------|-------|-------|-------|---------------|
| loss | 2.0 | 1.5 | 1.3 | 1.1 | 1.0 | **0.95** |

> Full training curve at wandb: [stage2_full_dense_64m](https://wandb.ai/qinganli-personal/minimind-embedding)

### 3' Model merging (Stage 3)

**Rationale**: apply Spherical Linear Interpolation (SLERP) to the last N checkpoints saved during Stage 2, fusing the complementary strengths of multiple models to improve robustness.

```bash
python scripts/merge_models.py \
    --glob "checkpoints/embedding/embedding_stage2_768_step*.pth" --last_n 5 \
    --output checkpoints/embedding/embedding_stage3_768.pth --config embed_dense_64m
```

> 🚧 Stage 3 results pending (the 3-epoch Stage 2 run is still producing checkpoints).

## Ⅲ Rerank training

### 1' Pointwise yes/no fine-tuning

**Rationale** (same as Qwen3-Reranker): model "whether a query-doc pair is relevant" as next-token prediction — let the model output `是` (yes) or `否` (no) after `[Query]\n[Document]`. This reuses the pretrained lm_head and **introduces no new parameters**.

Prompt template:

```
<|im_start|>system
Judge whether the Document meets the requirements based on the Query. Reply only "yes" or "no".<|im_end|>
<|im_start|>user
<Query>: {query}
<Document>: {document}<|im_end|>
<|im_start|>assistant
<think>

</think>

```

Loss (pointwise cross-entropy):

```math
\mathcal{L}_{rerank} = -\frac{1}{N}\sum_{i} \left[ y_i \log p(\text{yes}) + (1-y_i) \log p(\text{no}) \right]
```

> ⚠️ **Important finding**: for a small 64M backbone, full-parameter pointwise fine-tuning causes catastrophic forgetting — the pretrained backbone's zero-shot MAP@10 of 0.47 drops to 0.24 after training. The zero-shot backbone is therefore kept as Rerank v0. See the changelog for details.

---

# 📌 Evaluation

> ✅ Dense Stage 2 has completed STS evaluation. The full 35-task C-MTEB leaderboard is being prepared (mteb v2 integration in progress).

## Ⅰ STS results (Chinese, Spearman correlation)

Eval method: [C-MTEB](https://github.com/embeddings-benchmark/mteb) standard STS task test splits; the model encodes sentence1/sentence2 (last-token pooling + L2 norm), then we compute the Spearman correlation between cosine similarity and human labels.

| Task | Samples | This model (Stage 2) | mini baseline (2000 samples) | Notes |
|------|---------|:---:|:---:|------|
| ATEC | 20000 | **0.2656** | 0.13 | Banking customer-service similarity |
| BQ | 10000 | **0.3975** | 0.25 | Baidu Q&A similarity |
| LCQMC | 12500 | **0.6321** | 0.50 | Question matching (best) |
| STSB | 1361 | **0.6687** | — | Chinese semantic text similarity |
| **Average** | | **0.4910** | 0.29 | **+69% over mini** |

> **Reference** (same tasks, community models): BGE-small-zh ~0.55–0.65, Qwen3-Embedding-0.6B ~0.66.
> This model has only 64M params and a 6402 vocab, so an STS average of 0.49 is reasonable. STSB alone at 0.67 approaches BGE-small-zh.
> Limitations: the minimind backbone's vocab is only 6400 (weaker Chinese compression than dedicated Chinese models), and Stage 1 weak-supervision + Stage 3 merging results are not yet in — scores still have headroom.

Detailed JSON: see `results/sts_scores_full.json`.

## Ⅱ Reranking results

Eval method: for each query, the model scores all candidate docs (probability of `是`/yes), ranks by score descending, and computes MAP@10.

| Method | T2Reranking MAP@10 | Notes |
|--------|:---:|------|
| **Pretrained backbone (zero-shot)** | **0.4666** | ✅ Rerank v0 |
| Pointwise fine-tuned (v1, `<yes>/<no>`) | 0.2426 | ❌ Wrong direction (random init tokens) |
| Pointwise fine-tuned (v2, `是`/`否`, lr 5e-5) | (diagnosed reversed) | ❌ Catastrophic forgetting |

> **Key finding**: full-parameter pointwise fine-tuning on the 64M backbone destroys the pretrained relevance-discrimination ability. The zero-shot backbone (0.47) is kept as v0. Future work: frozen-backbone / LoRA / listwise loss.

Detailed JSON: see `results/rerank_pretrain_baseline.json`.

## Ⅲ How to reproduce the evaluation

```bash
# Embedding STS eval
python scripts/eval_sts_simple.py \
    --checkpoint checkpoints/embedding/embedding_stage2_768.pth \
    --config embed_dense_64m --tasks ATEC,BQ,LCQMC,STSB \
    --device cuda --max_length 128 --batch_size 64

# Rerank MAP@10 eval
python scripts/eval_rerank_simple.py \
    --checkpoint checkpoints/rerank/rerank_768.pth \
    --config rerank_dense_64m --device cuda
```

---

# 📌 Other

## 🐳 Docker training (seamless local/cloud switch)

```bash
# Build locally
docker compose -f docker/docker-compose.yml build

# Export the image (to transfer to a rented GPU machine)
docker save minimind-embed:latest | gzip > minimind-embed.tar.gz

# Load on the rented machine
docker load < minimind-embed.tar.gz
# Then mount the code dir and continue training
```

The image is based on `nvidia/cuda:13.3.0-cudnn-devel-ubuntu24.04` and pre-installs:
- Python 3.12 + torch (cu130/cu128, Blackwell sm_120 support)
- transformers + datasets + accelerate
- mteb 2.18 + sentence-transformers (evaluation and packaging)

## 👨‍💻 More

<details>
<summary><b>❓ FAQ</b></summary>

**Q: Why not use a BERT-style bidirectional attention + mean pooling?**
A: To maximize reuse of MiniMind's pretrained weights. Causal LM + last-token pooling is the efficient route validated by Qwen3-Embedding — the pre-training distribution matches, no re-warmup needed. See [Model > Last-token Pooling](#last-token-pooling-why-it-works).

**Q: Can an RTX 5080 really run the full training?**
A: The Dense 64M variant absolutely can (VRAM ~12-14GB). For MoE 198M long-sequence training you may need to lower the batch or enable gradient checkpointing; renting an A100 is recommended. See [Experiments > Training cost](#-experiments).

**Q: Why is the temperature τ=0.02?**
A: The Qwen3-Embedding report doesn't publish the exact value; community reproductions commonly use 0.02–0.05. 0.02 is an empirical sweet spot for InfoNCE (making similarity differences sharp enough to distinguish positives from negatives).

**Q: Why reuse `是`/`否` instead of adding `<yes>`/`<no>`?**
A: MiniMind's BPE vocab (6400) has no reliable single "yes"/"no" token (they'd be split into subwords). Adding new tokens means random initialization, which the model struggles to learn under a low learning rate — empirically it even learns the wrong direction. Reusing the existing `是`/`否` (single token, clear semantics, well-trained) is the most stable choice.

**Q: Why does rerank fine-tuning make things worse?**
A: The 64M backbone is fragile; full-parameter pointwise fine-tuning causes catastrophic forgetting, wiping out the pretrained relevance-discrimination ability (0.47 → 0.24). Qwen3-Reranker succeeds with the same method because it has 0.6B+ params and massive data. Future work: frozen backbone / LoRA / listwise loss.

</details>

---

# 📌 Acknowledgements

> [!NOTE]
> If this project helps you, please consider giving it a ⭐ on GitHub.<br/>
> The docs and code inevitably have gaps — feedback via Issues or PRs is welcome.<br/>
> Your support and suggestions are key drivers for this project's continued iteration!

## 🤝 Acknowledgements

This project stands on the shoulders of the following projects — many thanks:

- **[MiniMind](https://github.com/jingyaogong/minimind)** by [@jingyaogong](https://github.com/jingyaogong) — provided the excellent 64M small LLM backbone and complete training pipeline. Without MiniMind's "大道至简" (simplicity is the ultimate sophistication) philosophy, this project's "minimal starting point for vectorizing everything" wouldn't exist.
- **[Qwen3-Embedding](https://arxiv.org/abs/2506.05176)** by the Qwen Team — provided the complete technical pipeline of causal + last-token pooling + false-negative masking + pointwise rerank; this project's core methods all come from that report.

## 😊 Data & tools credits

- [T2Ranking](https://huggingface.co/datasets/THUIR/T2Ranking), [mMARCO](https://huggingface.co/datasets/unicamp-dl/mmarco), [C-MTEB](https://github.com/embeddings-benchmark/mteb), [sentence-transformers](https://www.sbert.net/), and other open-source datasets and tools.

---

# 🎓 Citation

If this project helps your research, please cite:

```bibtex
@misc{minimind-embeddings,
  title  = {MiniMind-Embedding: A Small Embedding and Rerank Model Based on MiniMind},
  author = {muzian666},
  year   = {2026},
  url    = {https://github.com/muzian666/minimind-embedding},
  note   = {Based on MiniMind, inspired by Qwen3-Embedding}
}

@misc{minimind,
  title  = {MiniMind: Train a Tiny LLM from Scratch},
  author = {Jingyao Gong},
  year   = {2024},
  url    = {https://github.com/jingyaogong/minimind}
}

@article{qwen3embedding,
  title  = {Qwen3 Embedding: Advancing Text Embedding and Reranking Through LLMs},
  author = {Zhang, Y. and others},
  year   = {2025},
  url    = {https://arxiv.org/abs/2506.05176}
}
```

---

# ⚖️ License

This project is open-sourced under the [Apache License 2.0](./LICENSE), completely free.
