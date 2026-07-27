"""
shared/tokenizer.py — Tokenizer 适配

复用 minimind 的 BPE tokenizer(vocab=6400),做两处扩展:
  1. 追加 <yes> / <no> 两个特殊 token(用于 rerank),vocab 6400 → 6402。
  2. padding_side 统一设为 'left'(last-token pooling 的正确性依赖左 padding)。

并提供 embedding / rerank 的输入模板构造函数。
"""
import os
from typing import List, Optional

from transformers import AutoTokenizer

_MINIMIND_TOKENIZER_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, "minimind", "model")
)

# 追加的特殊 token(只对 rerank 有意义,embedding 也统一加载以便模型共用底座)
YES_TOKEN = "<yes>"
NO_TOKEN = "<no>"
YES_TOKEN_ID = 6400
NO_TOKEN_ID = 6401
EXTRA_SPECIAL_TOKENS = [YES_TOKEN, NO_TOKEN]

# Embedding 模型的 query 指令前缀(对齐 Qwen3-Embedding 的 Instruct 机制)
# 不同任务类型可挂不同前缀,这里给一个通用检索指令。
DEFAULT_QUERY_INSTRUCTION = (
    "Given a web search query, retrieve relevant passages that answer the query"
)


def load_tokenizer(padding_side: str = "left"):
    """加载 minimind tokenizer 并追加 <yes>/<no>。

    参数:
        padding_side: 'left'(默认,推荐)或 'right'。last-token pooling
                      在左 padding 下 O(1) 取值,正确且高效。
    """
    tokenizer = AutoTokenizer.from_pretrained(
        _MINIMIND_TOKENIZER_PATH,
        trust_remote_code=True,
    )
    # 追加 <yes>/<no>
    num_added = tokenizer.add_special_tokens(
        {"additional_special_tokens": EXTRA_SPECIAL_TOKENS}
    )
    if num_added > 0:
        # 确认 id 符合预期
        assert tokenizer.convert_tokens_to_ids(YES_TOKEN) == YES_TOKEN_ID
        assert tokenizer.convert_tokens_to_ids(NO_TOKEN) == NO_TOKEN_ID
    tokenizer.padding_side = padding_side
    return tokenizer


# ---------------------------------------------------------------------------
# Embedding 输入构造
# ---------------------------------------------------------------------------
def build_embedding_inputs(
    tokenizer,
    texts: List[str],
    is_query: bool = False,
    instruction: Optional[str] = None,
    max_length: int = 8192,
    add_eos: bool = True,
    return_tensors: str = "pt",
):
    """构造 embedding 模型的输入。

    * 末尾固定追加 EOS(<|im_end|>, id=2),作为 last-token pooling 的锚点。
    * 若 is_query,可在文本前 prepend instruction(对齐 Qwen3 query 指令)。

    返回 transformers BatchEncoding(input_ids, attention_mask)。
    """
    processed = []
    for t in texts:
        if is_query and (instruction is not None):
            t = f"Instruct: {instruction}\nQuery: {t}"
        processed.append(t)

    enc = tokenizer(
        processed,
        truncation=True,
        max_length=max_length,
        padding=True,                # 动态 padding 到 batch 内最长
        return_tensors=return_tensors,
        add_special_tokens=False,    # 我们手动控制 EOS
    )
    # 末尾追加 EOS(在 padding 之前;tokenizer 已 padding,这里把 EOS 插到真实 token 末尾)
    # 实现:先不带 padding tokenize,追加 EOS,再统一 padding。
    if add_eos:
        return _append_eos_and_pad(tokenizer, processed, max_length, return_tensors)
    return enc


def _append_eos_and_pad(tokenizer, texts, max_length, return_tensors):
    """对每条文本:tokenize(截断)→ 末尾加 EOS → 左 padding 到 batch 最长。"""
    eos_id = tokenizer.eos_token_id  # <|im_end|> = 2
    all_ids = []
    for t in texts:
        ids = tokenizer(t, truncation=True, max_length=max_length - 1,
                        add_special_tokens=False)["input_ids"]
        ids = ids + [eos_id]
        all_ids.append(ids)
    max_len = max(len(x) for x in all_ids)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    # 左 padding:[pad, pad, ..., t1, t2, ..., eos]
    input_ids, attention_mask = [], []
    for ids in all_ids:
        pad_len = max_len - len(ids)
        input_ids.append([pad_id] * pad_len + ids)
        attention_mask.append([0] * pad_len + [1] * len(ids))

    import torch
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
    }


# ---------------------------------------------------------------------------
# Rerank 输入构造(对齐 Qwen3-Reranker prompt)
# ---------------------------------------------------------------------------
RERANK_PROMPT_TEMPLATE = (
    "<|im_start|>system\n"
    "Judge whether the Document meets the requirements based on the Query. "
    'Only output "yes" or "no".'
    "<|im_end|>\n"
    "<|im_start|>user\n"
    "<Query>: {query}\n"
    "<Document>: {document}<|im_end|>\n"
    "<|im_start|>assistant\n"
    "<think>\n\n</think>\n\n"
)
# 注意:末尾 assistant 开头后,模型应输出 <yes> 或 <no> 作为下一个 token。
# minimind tokenizer_config 的 chat_template 已支持 <think> 标签。


def build_rerank_inputs(
    tokenizer,
    queries: List[str],
    documents: List[str],
    max_length: int = 8192,
    return_tensors: str = "pt",
):
    """构造 rerank 模型的输入。

    对每个 (query, document) 对,套用 RERANK_PROMPT_TEMPLATE,
    末尾让模型预测 <yes>/<no>。左 padding。
    """
    texts = [
        RERANK_PROMPT_TEMPLATE.format(query=q, document=d)
        for q, d in zip(queries, documents)
    ]
    enc = tokenizer(
        texts,
        truncation=True,
        max_length=max_length,
        padding=True,
        return_tensors=return_tensors,
        add_special_tokens=False,
    )
    return enc
