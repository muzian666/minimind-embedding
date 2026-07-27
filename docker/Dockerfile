# syntax=docker/dockerfile:1
# MiniMind-Embedding & Rerank 训练环境(Blackwell / RTX 5080 适配)
#
# 关键版本选择(针对 RTX 5080,Blackwell sm_120 架构):
#   * 基础镜像:nvidia/cuda:13.3.0-cudnn-devel-ubuntu24.04
#       - CUDA 13.3 + cuDNN + devel(含 nvcc,便于按需编译扩展)
#       - Ubuntu 24.04(默认 Python 3.12)
#   * PyTorch: cu130 wheel(torch 2.13.0+cu130)
#       - 官方 wheel 最高到 cu130,在 13.3 驱动上向后兼容运行
#       - 自 Blackwell (sm_120) 内核,2.6+ 原生支持
#   * transformers 5.x:与 torch 2.13 兼容(DTensor 已稳定)
#   * mteb 2.18.x / sentence-transformers 5.x:最新评测栈
#
# 构建:docker compose -f docker/docker-compose.yml build
# 运行:docker compose -f docker/docker-compose.yml run --rm dev
#      docker compose -f docker/docker-compose.yml run --rm smoke

FROM nvidia/cuda:13.3.0-cudnn-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    HF_HOME=/workspace/hf_cache \
    TRANSFORMERS_CACHE=/workspace/hf_cache

# 系统依赖(Ubuntu 24.04 自带 Python 3.12)
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-venv \
        git curl wget vim build-essential \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3 /usr/bin/python \
    && ln -sf /usr/bin/pip3 /usr/bin/pip

# Python 依赖:torch(cu130,Blackwell 支持)
# 分层安装以利用缓存:torch 单独一层(体积大、变更少)
# 不升级 pip:Ubuntu 24.04 的 pip 由 apt 管理,自升级会失败(PEP 668);
#             系统 pip 24.0 已足够新,可直接装 torch 2.13。
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu130

# 训练 + 数据栈
RUN pip install \
        "transformers>=5.0.0" \
        "tokenizers>=0.20.0" \
        "datasets>=3.0.0" \
        "accelerate>=1.0.0" \
        "numpy<2.3.0" \
        "scipy>=1.13.0" \
        "scikit-learn>=1.5.0" \
        "tqdm>=4.66.0" \
        "swanlab>=0.4.0" \
        "wandb>=0.18.0"

# 评测框架(MTEB v2 + sentence-transformers)
RUN pip install \
        "mteb>=2.18.0" \
        "sentence-transformers>=5.0.0"

WORKDIR /workspace

CMD ["/bin/bash"]
