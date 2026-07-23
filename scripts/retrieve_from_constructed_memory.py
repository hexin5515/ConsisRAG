#!/usr/bin/env python3
"""
基于构图阶段输出的 memory / vdb 文件做一个简单、可运行的检索流程。

当前检索链路分三步：
1. 用 query 和 schema / schema_relation 向量做第一跳召回
2. 根据召回到的 schema / schema_relation，展开到候选 passages
3. 再用 query 和 chunk 向量对候选 passages 做一次精排

输入文件默认来自同一个 PDF 输出目录，例如：
    output/<run_name>/<pdf_name>/
        memory.json
        memory_embeddings.json
        chunk_embeddings/vdb_chunk.parquet
        schema_embeddings/vdb_schema.parquet
        schema_relation_embeddings/vdb_schema_relation.parquet
"""

from __future__ import annotations

import argparse
import base64
from dataclasses import asdict, dataclass
import html
import json
import mimetypes
import os
import re
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import httpx
import numpy as np
import pandas as pd
from openai import OpenAI


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from memory import ThreeLayerMemory
from evaluation.mmlong_bookrag_eval import BookRAGStyleAnswerExtractor, evaluate_mmlong_results
from prompts.linking import get_query_instruction

# Keep retrieval defaults aligned with the construction-stage embedding config.
DEFAULT_EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"
DEFAULT_EMBEDDING_BASE_URL = "http://localhost:8083/v1/embeddings"
DEFAULT_EMBEDDING_API_KEY = "openai"
DEFAULT_ANSWER_MODEL = "Qwen/Qwen3-8B-AWQ"
DEFAULT_ANSWER_BASE_URL = "http://localhost:8080/v1"
DEFAULT_ANSWER_API_KEY = "openai"
DEFAULT_VLM_MODEL = "Qwen/Qwen2-5-VL"
DEFAULT_VLM_BASE_URL = "http://localhost:8081/v1"
DEFAULT_VLM_API_KEY = "openai"
DEFAULT_JUDGE_MODEL = DEFAULT_ANSWER_MODEL
DEFAULT_JUDGE_BASE_URL = DEFAULT_ANSWER_BASE_URL
DEFAULT_JUDGE_API_KEY = DEFAULT_ANSWER_API_KEY
DEFAULT_PLANNER_MODEL = DEFAULT_ANSWER_MODEL
DEFAULT_PLANNER_BASE_URL = DEFAULT_ANSWER_BASE_URL
DEFAULT_PLANNER_API_KEY = DEFAULT_ANSWER_API_KEY
DEFAULT_MINERU_ROOT = "data/datasets/MMLongBench-Doc/process_documents"
GRAPH_TEXT_TOP_K = 5    #MMLong 5 3 2 Qasper 5 3 2 M3DocVAQ 10 3 2 
GRAPH_TABLE_TOP_K = 3
GRAPH_IMAGE_TOP_K = 2

BOOKRAG_GBC_ITER_GENERATION_SYS_PROMPT = """
Please refer to the following background information to answer the question."""

BOOKRAG_GBC_ITER_GENERATION_USER_PROMPT = """
--- User Question ---
{user_question}

--- Retrieved Documents ---
## Document Data
{retrieved_content}

"""

BOOKRAG_GBC_ITER_GENERATION_GRAPH = """
## Knowledge Graph Data
{knowledge_graph_subgraph}

"""

LEGACY_MMLONG_ANSWER_PROMPT_TEMPLATE = """
User Question: {query}

You are a helpful AI assistant. Please provide a comprehensive answer to the user's question based on the following context, which may include text, tables, and images.
First, provide a step-by-step explanation of your reasoning.
Then, provide a clear, final answer prefixed with "Final Answer:".

--- START OF CONTEXT ---
{context_str}
--- END OF CONTEXT ---

Explanation:
"""

BOOKRAG_SYNTHESIS_SYSTEM_PROMPT = """
You are an expert AI assistant specializing in synthesizing information.
Given the original user question, the decomposition plan, and the answers to
the retrieval sub-questions, produce one final answer.
"""

BOOKRAG_SYNTHESIS_USER_PROMPT_TEMPLATE = """
--- Original User Question ---
{original_query}

--- Decomposition Plan ---
{plan_str}

--- Retrieved Sub-question Answers ---
{sub_answers_str}

Please combine the sub-question answers into a final answer to the original
question. First provide a short reasoning trace. Then provide a final answer
prefixed with "Final Answer:".
"""

BOOKRAG_GLOBAL_NODE_ANALYSIS_PROMPT_TEMPLATE = """
You are working on a larger global document question. Your job right now is to
analyze exactly one filtered document item so it can later be combined with
other item-level analyses.

--- ORIGINAL GLOBAL QUERY ---
{original_query}

--- CURRENT TASK ---
{operation_instruction}

--- FILTERED ITEM ---
Chunk ID: {chunk_id}
Modality: {modality}
Page: {page_label}
Content:
{content}

Provide a concise analysis of this single filtered item only.
"""

BOOKRAG_GLOBAL_SYNTHESIS_SYSTEM_PROMPT = """
You are an expert AI assistant specializing in synthesizing information across
multiple filtered document items.
"""

BOOKRAG_GLOBAL_SYNTHESIS_USER_PROMPT_TEMPLATE = """
--- ORIGINAL GLOBAL QUERY ---
{original_query}

--- AGGREGATION OPERATION ---
{operation}

--- ITEM-LEVEL ANALYSES ---
{partial_answers_str}

Combine the item-level analyses into one final answer to the original global
question. First provide a short reasoning trace. Then provide a final answer
prefixed with "Final Answer:".
"""

BOOKRAG_NODE_SUMMARY_PROMPT = """Condense the content below into a self-contained, informative summary. The summary must capture the key points and be easily understandable on its own.

The content may be text, an image, or a table. If a title is provided, use it to understand the core subject.

---
**Content to Summarize:**
{node_text}
---

Directly output the summary. Your response must begin with the core conclusion, strictly avoiding any introductory phrases that describe the input, such as "The input is a table that shows...".
"""


def _schema_relation_text(content: Sequence[str]) -> str:
    head_type, relation, tail_type = content
    return f"{head_type} {relation} {tail_type}"


def _l2_normalize(vecs: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return vecs / norms


def _cosine_scores(query_vec: np.ndarray, doc_vecs: np.ndarray) -> np.ndarray:
    query_vec = query_vec.astype(np.float32)
    doc_vecs = doc_vecs.astype(np.float32)
    query_vec = query_vec / max(np.linalg.norm(query_vec), 1e-12)
    doc_vecs = _l2_normalize(doc_vecs)
    return np.dot(doc_vecs, query_vec)


def _resolve_vdb_paths(memory_path: Path, args: argparse.Namespace) -> Dict[str, Path]:
    pdf_dir = memory_path.parent
    return {
        "schema": Path(args.schema_vdb_path) if args.schema_vdb_path else pdf_dir / "schema_embeddings" / "vdb_schema.parquet",
        "schema_relation": Path(args.schema_relation_vdb_path) if args.schema_relation_vdb_path else pdf_dir / "schema_relation_embeddings" / "vdb_schema_relation.parquet",
        "chunk": Path(args.chunk_vdb_path) if args.chunk_vdb_path else pdf_dir / "chunk_embeddings" / "vdb_chunk.parquet",
    }


def _infer_doc_id_from_memory_path(memory_path: Path) -> str:
    """
    从当前输出目录推断 MMLongBench 的 doc_id。

    例如：
        output/.../tacl_a_00660/memory.json
    ->  tacl_a_00660.pdf
    """
    return f"{memory_path.parent.name}.pdf"


def _empty_token_usage() -> Dict[str, int]:
    return {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }


def _usage_to_dict(usage: Any) -> Dict[str, int]:
    """
    统一把 OpenAI SDK 返回的 usage 对象转成普通 dict。

    不同 provider / SDK 版本上，usage 可能是：
    - 具备属性的对象
    - dict
    - None

    这里统一落成三个最稳定的字段，方便后面做总量累计。
    """
    if usage is None:
        return _empty_token_usage()

    if isinstance(usage, dict):
        return {
            "prompt_tokens": int(usage.get("prompt_tokens", 0) or 0),
            "completion_tokens": int(usage.get("completion_tokens", 0) or 0),
            "total_tokens": int(usage.get("total_tokens", 0) or 0),
        }

    return {
        "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
        "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
        "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
    }


def _accumulate_token_usage(total: Dict[str, int], delta: Dict[str, int]) -> None:
    total["prompt_tokens"] += int(delta.get("prompt_tokens", 0) or 0)
    total["completion_tokens"] += int(delta.get("completion_tokens", 0) or 0)
    total["total_tokens"] += int(delta.get("total_tokens", 0) or 0)


def _empty_timing_breakdown() -> Dict[str, float]:
    return {
        "graph_guided": 0.0,
        "dense_baseline": 0.0,
        "hybrid": 0.0,
        "selected_strategy": 0.0,
        "total": 0.0,
    }


def _load_samples_for_doc(samples_json_path: str, doc_id: str) -> List[Dict[str, Any]]:
    samples_path = Path(samples_json_path).resolve()
    with open(samples_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list in samples file: {samples_path}")
    return [item for item in data if _infer_doc_id_from_sample_item(item) == doc_id]


def _infer_doc_id_from_sample_item(item: Dict[str, Any]) -> str:
    """
    Infer the benchmark doc_id from one sample item.

    The original HVM-GraphRAG MMLongBench path expects:
        {"doc_id": "tacl_a_00660.pdf", ...}

    Some Office-style exports instead omit `doc_id` and only keep:
        {"doc_path": "PATH_TO_DIR/tacl_a_00660.pdf", ...}

    To stay compatible with both, we prefer `doc_id` when present and fall back
    to the basename of `doc_path`.
    """
    doc_id = str(item.get("doc_id", "")).strip()
    if doc_id:
        return doc_id

    doc_path = str(item.get("doc_path", "")).strip()
    if doc_path:
        return Path(doc_path).name.strip()
    return ""


def _load_samples_grouped_by_doc(samples_json_path: str) -> Dict[str, List[Dict[str, Any]]]:
    samples_path = Path(samples_json_path).resolve()
    with open(samples_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list in samples file: {samples_path}")
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for item in data:
        doc_id = _infer_doc_id_from_sample_item(item)
        if not doc_id:
            continue
        grouped.setdefault(doc_id, []).append(item)
    return grouped


def _discover_memory_paths(memory_path_arg: str) -> List[Path]:
    target = Path(memory_path_arg).resolve()
    if target.is_file():
        return [target]
    if target.is_dir():
        return sorted(p for p in target.rglob("memory.json") if p.is_file())
    raise FileNotFoundError(f"Memory path not found: {target}")


def _progress_bar(current: int, total: int, width: int = 28) -> str:
    if total <= 0:
        return "[" + ("-" * width) + "]"
    filled = int(width * current / total)
    filled = min(max(filled, 0), width)
    return "[" + ("#" * filled) + ("-" * (width - filled)) + "]"


def _safe_output_name(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return "unknown"
    text = re.sub(r"[^\w.\-]+", "_", text)
    return text.strip("._") or "unknown"


def _max_top_schema_score(top_schemas: Sequence[Dict[str, Any]]) -> float:
    scores = [float(item.get("score", float("-inf"))) for item in top_schemas or []]
    if not scores:
        return float("-inf")
    return max(scores)


def _normalize_chat_base_url(base_url: str) -> str:
    value = (base_url or "").strip().rstrip("/")
    if value.endswith("/chat/completions"):
        value = value[: -len("/chat/completions")]
    return value


def _normalize_embed_base_url(base_url: str) -> str:
    value = (base_url or "").strip().rstrip("/")
    if value.endswith("/embeddings"):
        value = value[: -len("/embeddings")]
    return value


def _load_vdb(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"VDB file not found: {path}")
    df = pd.read_parquet(path)
    required_cols = {"content", "embedding"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"VDB file missing columns {sorted(missing)}: {path}")
    return df


def _load_optional_vdb(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["content", "embedding"])
    df = pd.read_parquet(path)
    required_cols = {"content", "embedding"}
    missing = required_cols - set(df.columns)
    if missing:
        return pd.DataFrame(columns=["content", "embedding"])
    return df


def _truncate_text(text: str, max_chars: int) -> str:
    value = str(text or "")
    if len(value) <= max_chars:
        return value
    return value[:max_chars] + "..."


def _encode_image_as_data_url(image_path: str) -> str | None:
    path = Path(str(image_path or ""))
    if not path.exists() or not path.is_file():
        return None

    mime_type, _ = mimetypes.guess_type(str(path))
    mime_type = mime_type or "image/jpeg"
    try:
        encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
    except Exception:
        return None
    return f"data:{mime_type};base64,{encoded}"


def _html_table_to_text(table_html: str) -> str:
    text = str(table_html or "").strip()
    if not text:
        return ""
    text = re.sub(r"</tr\s*>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"</t[dh]\s*>", " | ", text, flags=re.IGNORECASE)
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", "", text)
    text = html.unescape(text)
    lines = []
    for raw_line in text.splitlines():
        line = " ".join(raw_line.split()).strip(" |")
        if line:
            line = re.sub(r"\s*\|\s*\|\s*", " | ", line)
            lines.append(line)
    return "\n".join(lines)


def _infer_embedding_model_name(sample_dim: int) -> str:
    """
    根据向量维度做一个保守推断。

    当前优先从环境变量读取真实 embedding 模型名；如果未设置，
    再根据向量维度做保守推断。
    """
    env_model = os.environ.get("MEMGRAPHRAG_EMBED_MODEL", "").strip()
    if env_model:
        return env_model
    if DEFAULT_EMBEDDING_MODEL:
        return DEFAULT_EMBEDDING_MODEL
    if sample_dim == 4096:
        return "nvidia/NV-Embed-v2"
    if sample_dim in {1024, 1536, 2048, 3072}:
        return "openai-embedding"
    raise ValueError(
        "Unexpected embedding dimension "
        f"{sample_dim}. Set MEMGRAPHRAG_EMBED_MODEL explicitly, "
        "or pass --embedding-model. "
        "Known defaults: 4096 -> nvidia/NV-Embed-v2; "
        "1024/1536/2048/3072 -> openai-compatible embedding API."
    )


def _build_embedding_model(embedding_model_name: str) -> Any:
    from src.utils.config_utils import BaseConfig
    from src.embedding_model import _get_embedding_model_class

    config = BaseConfig()
    config.embedding_model_name = embedding_model_name
    config.embedding_base_url = _normalize_embed_base_url(
        os.environ.get("MEMGRAPHRAG_EMBED_BASE_URL", DEFAULT_EMBEDDING_BASE_URL)
    )
    config.embedding_api_key = os.environ.get("MEMGRAPHRAG_EMBED_API_KEY", DEFAULT_EMBEDDING_API_KEY)
    config.embedding_batch_size = int(os.environ.get("MEMGRAPHRAG_EMBED_BATCH_SIZE", "8"))
    model_cls = _get_embedding_model_class(embedding_model_name)
    return model_cls(global_config=config)


class MinerUMetadataLookup:
    """用 chunk_id 回查原始 MinerU tree.json，补回 table/image 的细粒度信息。"""

    def __init__(
        self,
        *,
        memory_path: str,
        mineru_root: str = "",
        tree_json_path: str = "",
        vlm_model: str = "",
        vlm_base_url: str = "",
        vlm_api_key: str = "",
    ) -> None:
        self.memory_path = Path(memory_path).resolve()
        self.doc_id = _infer_doc_id_from_memory_path(self.memory_path)
        self.tree_json_path = self._resolve_tree_json(tree_json_path=tree_json_path, mineru_root=mineru_root)
        self.chunk_map: Dict[str, Dict[str, Any]] = {}
        self.vlm_model = (vlm_model or os.environ.get("MEMGRAPHRAG_VLM_MODEL", DEFAULT_VLM_MODEL)).strip()
        self.vlm_base_url = _normalize_chat_base_url(
            vlm_base_url or os.environ.get("MEMGRAPHRAG_VLM_BASE_URL", DEFAULT_VLM_BASE_URL)
        )
        self.vlm_api_key = (vlm_api_key or os.environ.get("MEMGRAPHRAG_VLM_API_KEY", DEFAULT_VLM_API_KEY)).strip()
        self._image_description_cache: Dict[str, str] = {}
        self._vlm_client: OpenAI | None = None
        self._tree_passages_raw: List[Dict[str, Any]] = []
        if self.tree_json_path is not None:
            self.chunk_map = self._load_chunk_map(self.tree_json_path)
            self._tree_passages_raw = self._load_tree_passages_raw(self.tree_json_path)

    def _resolve_tree_json(self, *, tree_json_path: str, mineru_root: str) -> Path | None:
        if tree_json_path:
            path = Path(tree_json_path).resolve()
            return path if path.exists() else None

        root_value = mineru_root or os.environ.get("MEMGRAPHRAG_MINERU_ROOT", DEFAULT_MINERU_ROOT)
        root = Path(root_value).resolve()
        if not root.exists():
            return None

        # BookRAG preprocesses each document into a deterministic UUID-named folder:
        #     doc_uuid = str(uuid.uuid5(uuid.NAMESPACE_DNS, doc_id))
        # where doc_id for MMLongBench is the visible file name like `tacl_a_00660.pdf`.
        # We reuse the same mapping here instead of scanning every tree.json.
        doc_uuid = str(uuid.uuid5(uuid.NAMESPACE_DNS, self.doc_id))
        candidate = root / doc_uuid / "tree.json"
        if candidate.exists():
            return candidate

        # Keep one narrow compatibility fallback for runs whose output folder name
        # already matches the process_documents child directory name.
        stem_candidate = root / self.memory_path.parent.name / "tree.json"
        if stem_candidate.exists():
            return stem_candidate
        return None

    @staticmethod
    def _load_chunk_map(tree_json_path: Path) -> Dict[str, Dict[str, Any]]:
        from utils.mineru_loader import load_mineru_json_chunks

        chunk_map: Dict[str, Dict[str, Any]] = {}
        chunks = load_mineru_json_chunks(
            str(tree_json_path),
            include_title_chunks=False,
            include_image_chunks=True,
            include_empty_image_chunks=True,
        )
        for chunk in chunks:
            metadata = dict(chunk.metadata or {})
            current_node = metadata.get("current_node") or {}
            chunk_map[str(chunk.chunk_id)] = {
                "block_type": metadata.get("block_type"),
                "img_path": metadata.get("img_path"),
                "table_body": metadata.get("table_body") or current_node.get("table_body"),
                "summary": metadata.get("summary") or current_node.get("summary"),
                "caption": current_node.get("caption"),
                "footnote": current_node.get("footnote"),
                "current_content": current_node.get("content"),
            }
        return chunk_map

    @staticmethod
    def _modality_from_block_type(block_type: str) -> str:
        normalized = str(block_type or "").strip().lower()
        if normalized == "table":
            return "table"
        if normalized == "image":
            return "image"
        return "text"

    @classmethod
    def _load_tree_passages_raw(cls, tree_json_path: Path) -> List[Dict[str, Any]]:
        from utils.mineru_loader import load_mineru_json_chunks

        passages: List[Dict[str, Any]] = []
        chunks = load_mineru_json_chunks(
            str(tree_json_path),
            include_title_chunks=False,
            include_image_chunks=True,
            include_empty_image_chunks=True,
        )
        for chunk in chunks:
            metadata = dict(chunk.metadata or {})
            full_content = str(chunk.content or "")
            passages.append(
                {
                    "chunk_id": str(chunk.chunk_id),
                    "score": 0.0,
                    "modality": cls._modality_from_block_type(str(metadata.get("block_type", ""))),
                    "entity_mentions": [],
                    "content_preview": full_content,
                    "full_content": full_content,
                    "source_file": metadata.get("source_file"),
                    "source_path": metadata.get("source_path"),
                    "index_id": metadata.get("index_id"),
                    "parent_id": metadata.get("parent_id"),
                    "child_ids": metadata.get("child_ids", []),
                    "sibling_ids": metadata.get("sibling_ids", []),
                    "ancestor_ids": metadata.get("ancestor_ids", []),
                    "page_idx": metadata.get("page_idx"),
                    "node_type": metadata.get("node_type"),
                    "block_type": metadata.get("block_type"),
                    "title_path": metadata.get("title_path", []),
                    "img_path": metadata.get("img_path"),
                    "current_node": metadata.get("current_node"),
                    "parent_node": metadata.get("parent_node"),
                    "child_nodes": metadata.get("child_nodes", []),
                    "sibling_nodes": metadata.get("sibling_nodes", []),
                    "ancestor_nodes": metadata.get("ancestor_nodes", []),
                }
            )
        return passages

    def close(self) -> None:
        if self._vlm_client is None:
            return
        close_fn = getattr(self._vlm_client, "close", None)
        if callable(close_fn):
            close_fn()

    def _get_vlm_client(self) -> OpenAI | None:
        if not self.vlm_model or not self.vlm_base_url:
            return None
        if self._vlm_client is None:
            http_client = httpx.Client(timeout=180.0, trust_env=False)
            self._vlm_client = OpenAI(
                api_key=self.vlm_api_key or DEFAULT_VLM_API_KEY,
                base_url=self.vlm_base_url,
                http_client=http_client,
            )
        return self._vlm_client

    @staticmethod
    def _bookrag_image_summary_input(
        *,
        current_content: str,
        caption: str,
        footnote: str,
    ) -> str:
        caption_like = current_content or caption
        content = "This is an image. Here is the caption: "
        content += str(caption_like or "").strip()
        if footnote:
            content += f"\nFootnote: {str(footnote).strip()}"
        return BOOKRAG_NODE_SUMMARY_PROMPT.format(node_text=content.strip())

    def _describe_image(
        self,
        image_path: str,
        *,
        current_content: str = "",
        caption: str = "",
        footnote: str = "",
    ) -> str:
        path = str(image_path or "").strip()
        if not path:
            return ""
        if path in self._image_description_cache:
            return self._image_description_cache[path]

        data_url = _encode_image_as_data_url(path)
        client = self._get_vlm_client()
        if not data_url or client is None:
            self._image_description_cache[path] = ""
            return ""

        prompt = self._bookrag_image_summary_input(
            current_content=current_content,
            caption=caption,
            footnote=footnote,
        )
        try:
            response = client.chat.completions.create(
                model=self.vlm_model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": data_url}},
                        ],
                    }
                ],
                temperature=0.0,
                max_tokens=256,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            description = (response.choices[0].message.content or "").strip()
        except Exception:
            description = ""

        self._image_description_cache[path] = description
        return description

    def enrich_passage(
        self,
        record: Dict[str, Any],
        *,
        enable_image_vlm: bool = True,
    ) -> Dict[str, Any]:
        enriched = dict(record)
        chunk_id = str(record.get("chunk_id", ""))
        metadata = self.chunk_map.get(chunk_id)
        if not metadata:
            return enriched

        modality = str(record.get("modality", "")).lower()
        base_content = str(record.get("full_content", ""))
        qa_content = base_content

        if modality == "table":
            table_body = str(metadata.get("table_body") or "").strip()
            if table_body:
                table_text = _html_table_to_text(table_body)
                enriched["table_body"] = table_body
                enriched["table_text"] = table_text
                if table_text:
                    qa_content = f"{base_content}\n\n[Recovered Table Content]\n{table_text}"
        elif modality == "image":
            image_path = str(metadata.get("img_path") or "").strip()
            current_content = str(metadata.get("current_content") or "").strip()
            caption = str(metadata.get("caption") or "").strip()
            footnote = str(metadata.get("footnote") or "").strip()
            if enable_image_vlm:
                image_description = self._describe_image(
                    image_path,
                    current_content=current_content,
                    caption=caption,
                    footnote=footnote,
                )
                if image_description:
                    enriched["image_description"] = image_description
                    qa_content = f"{base_content}\n\n[Recovered Image Summary]\n{image_description}"

        enriched["qa_content"] = qa_content
        if self.tree_json_path is not None:
            enriched["tree_json_path"] = str(self.tree_json_path)
        return enriched

    def enrich_passages(
        self,
        passages: List[Dict[str, Any]],
        *,
        enable_image_vlm: bool = True,
    ) -> List[Dict[str, Any]]:
        return [
            self.enrich_passage(item, enable_image_vlm=enable_image_vlm)
            for item in passages
        ]

    def get_all_tree_passages(
        self,
        *,
        preview_chars: int,
        enable_image_vlm: bool = True,
    ) -> List[Dict[str, Any]]:
        passages: List[Dict[str, Any]] = []
        for item in self._tree_passages_raw:
            record = dict(item)
            record["content_preview"] = _truncate_text(str(record.get("full_content", "")), preview_chars)
            passages.append(record)
        return self.enrich_passages(passages, enable_image_vlm=enable_image_vlm)


def _encode_query(model: Any, query: str, instruction_name: str) -> np.ndarray:
    instruction = get_query_instruction(instruction_name)
    vec = model.batch_encode([query], instruction=instruction)[0]
    return np.asarray(vec, dtype=np.float32)


def _rank_vdb_rows(
    df: pd.DataFrame,
    query_vec: np.ndarray,
    *,
    top_k: int,
) -> List[Dict[str, Any]]:
    embeddings = np.vstack(df["embedding"].apply(lambda x: np.asarray(x, dtype=np.float32)).tolist())
    scores = _cosine_scores(query_vec, embeddings)
    order = np.argsort(-scores)[:top_k]
    results: List[Dict[str, Any]] = []
    for idx in order:
        row = df.iloc[int(idx)]
        results.append(
            {
                "content": row["content"],
                "score": float(scores[int(idx)]),
            }
        )
    return results


def _rank_chunk_candidates(
    chunk_df: pd.DataFrame,
    query_vec: np.ndarray,
    *,
    top_k: int,
    allowed_modalities: Iterable[str] | None = None,
) -> List[Dict[str, Any]]:
    """
    纯 dense baseline：
    直接对 chunk / sub-chunk 向量做检索，再聚合回原始 passage。
    """
    modality_filter = set(m.strip().lower() for m in allowed_modalities or [] if m.strip())
    embeddings = np.vstack(chunk_df["embedding"].apply(lambda x: np.asarray(x, dtype=np.float32)).tolist())
    scores = _cosine_scores(query_vec, embeddings)

    grouped: Dict[str, Dict[str, Any]] = {}
    has_parent_mapping = "parent_chunk_id" in chunk_df.columns
    has_parent_passage_idx = "parent_passage_idx" in chunk_df.columns

    for i, row in chunk_df.iterrows():
        dense_score = float(scores[int(i)])
        content = str(row["content"])
        modality = str(row["modality"]).lower() if "modality" in row and pd.notna(row["modality"]) else ""
        if modality_filter and modality and modality not in modality_filter:
            continue

        if has_parent_mapping:
            parent_chunk_id = str(row["parent_chunk_id"])
            parent_passage_idx = int(row["parent_passage_idx"]) if has_parent_passage_idx else -1
            key = parent_chunk_id
        else:
            parent_chunk_id = str(row.get("chunk_id", content))
            parent_passage_idx = int(row.get("passage_idx", -1))
            key = parent_chunk_id

        bucket = grouped.get(key)
        if bucket is None or dense_score > float(bucket["dense_score"]):
            grouped[key] = {
                "chunk_id": parent_chunk_id,
                "passage_idx": parent_passage_idx,
                "dense_score": dense_score,
                "best_segment_preview": content,
            }

    results = sorted(grouped.values(), key=lambda item: item["dense_score"], reverse=True)
    return results[:top_k]


def _collect_candidate_passages(
    memory: ThreeLayerMemory,
    schema_hits: List[Dict[str, Any]],
    relation_hits: List[Dict[str, Any]],
) -> Dict[int, Dict[str, Any]]:
    """
    将结构层召回结果展开成候选 passage 集。

    这里我们保留两类结构信号：
    - schema 命中
    - schema relation 命中

    passage 的初始分数先用结构层的最大分数，后面再做 chunk-level rerank。
    """
    candidates: Dict[int, Dict[str, Any]] = {}

    schema_score_map = {item["content"]: item["score"] for item in schema_hits}
    relation_score_map = {item["content"]: item["score"] for item in relation_hits}

    for schema in memory.schema_layer:
        score = schema_score_map.get(schema.content)
        if score is None:
            continue
        for passage_idx in schema.passage_indices:
            bucket = candidates.setdefault(
                passage_idx,
                {
                    "passage_idx": passage_idx,
                    "structural_score": float("-inf"),
                    "schema_hits": [],
                    "schema_relation_hits": [],
                },
            )
            bucket["structural_score"] = max(bucket["structural_score"], score)
            if schema.content not in bucket["schema_hits"]:
                bucket["schema_hits"].append(schema.content)

    for relation_node in memory.schema_relation_layer:
        relation_text = _schema_relation_text(relation_node.content)
        score = relation_score_map.get(relation_text)
        if score is None:
            continue
        for passage_idx in relation_node.passage_indices:
            bucket = candidates.setdefault(
                passage_idx,
                {
                    "passage_idx": passage_idx,
                    "structural_score": float("-inf"),
                    "schema_hits": [],
                    "schema_relation_hits": [],
                },
            )
            bucket["structural_score"] = max(bucket["structural_score"], score)
            if relation_text not in bucket["schema_relation_hits"]:
                bucket["schema_relation_hits"].append(relation_text)

    return candidates


def _expand_schema_hits_with_connected_relations(
    memory: ThreeLayerMemory,
    schema_hits: List[Dict[str, Any]],
    relation_hits: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    从已召回 schema 出发，保留“至少一端连接到已召回 schema”的 relation，
    并把 relation 的另一端 schema 一并补进 schema 命中集合。

    这样得到的是一个以 top-k schema 为种子、经一跳 relation 扩展得到的
    连通 schema 子图，而不是彼此孤立的 schema / relation 命中结果。
    """
    if not schema_hits or not relation_hits:
        return schema_hits, []

    relation_hit_map = {
        str(item.get("content", "")): dict(item)
        for item in relation_hits
    }
    schema_hit_map: Dict[str, Dict[str, Any]] = {
        str(item.get("content", "")): dict(item)
        for item in schema_hits
    }
    recalled_schema_contents = set(schema_hit_map.keys())
    kept_relation_hits: List[Dict[str, Any]] = []
    kept_relation_texts: set[str] = set()

    for relation_node in memory.schema_relation_layer:
        relation_text = _schema_relation_text(relation_node.content)
        relation_hit = relation_hit_map.get(relation_text)
        if relation_hit is None or relation_text in kept_relation_texts:
            continue

        source_schema = memory.get_schema_by_idx(relation_node.source_schema_idx)
        target_schema = memory.get_schema_by_idx(relation_node.target_schema_idx)
        if source_schema is None or target_schema is None:
            continue

        source_content = str(source_schema.content)
        target_content = str(target_schema.content)
        source_hit = source_content in recalled_schema_contents
        target_hit = target_content in recalled_schema_contents

        if not (source_hit or target_hit):
            continue

        kept_relation_hits.append(relation_hit)
        kept_relation_texts.add(relation_text)

        relation_score = float(relation_hit.get("score", 0.0))
        if not source_hit:
            schema_hit_map[source_content] = {
                "content": source_content,
                "score": relation_score,
                "expanded_from_relation": relation_text,
            }
            recalled_schema_contents.add(source_content)
        if not target_hit:
            schema_hit_map[target_content] = {
                "content": target_content,
                "score": relation_score,
                "expanded_from_relation": relation_text,
            }
            recalled_schema_contents.add(target_content)

    expanded_schema_hits = sorted(
        schema_hit_map.values(),
        key=lambda item: float(item.get("score", float("-inf"))),
        reverse=True,
    )
    kept_relation_hits.sort(
        key=lambda item: float(item.get("score", float("-inf"))),
        reverse=True,
    )
    return expanded_schema_hits, kept_relation_hits


def _rerank_passages(
    memory: ThreeLayerMemory,
    chunk_df: pd.DataFrame,
    query_vec: np.ndarray,
    candidates: Dict[int, Dict[str, Any]],
    *,
    top_k: int,
    allowed_modalities: Iterable[str] | None = None,
) -> List[Dict[str, Any]]:
    """
    用 chunk 向量对候选 passages 精排。

    最终分数做一个简单融合：
        final_score = 0.7 * dense_score + 0.3 * structural_score

    这里先用固定权重，目的是把流程跑通，后面如果我们要认真调 retrieval，
    再把融合策略做成可配置就行。
    """
    modality_filter = set(m.strip().lower() for m in allowed_modalities or [] if m.strip())
    chunk_embeddings = np.vstack(chunk_df["embedding"].apply(lambda x: np.asarray(x, dtype=np.float32)).tolist())
    dense_scores = _cosine_scores(query_vec, chunk_embeddings)
    parent_score_map: Dict[str, float] = {}
    parent_best_segment_map: Dict[str, str] = {}

    has_parent_mapping = "parent_chunk_id" in chunk_df.columns
    for i, row in chunk_df.iterrows():
        dense_score = float(dense_scores[int(i)])
        if has_parent_mapping:
            parent_chunk_id = str(row["parent_chunk_id"])
            if dense_score > parent_score_map.get(parent_chunk_id, float("-inf")):
                parent_score_map[parent_chunk_id] = dense_score
                parent_best_segment_map[parent_chunk_id] = str(row["content"])
        else:
            parent_key = str(row["content"])
            parent_score_map[parent_key] = dense_score
            parent_best_segment_map[parent_key] = str(row["content"])

    results: List[Dict[str, Any]] = []
    for passage_idx, meta in candidates.items():
        passage = memory.get_passage_by_idx(passage_idx)
        if passage is None:
            continue
        if modality_filter and passage.modality.lower() not in modality_filter:
            continue

        dense_lookup_key = str(passage.chunk_id) if has_parent_mapping else str(passage.content)
        dense_score = parent_score_map.get(dense_lookup_key)
        if dense_score is None:
            continue

        structural_score = float(meta["structural_score"])
        final_score = 1 * dense_score + 0.0 * structural_score

        results.append(
            {
                "passage_idx": passage.idx,
                "chunk_id": passage.chunk_id,
                "modality": passage.modality,
                "final_score": float(final_score),
                "dense_score": float(dense_score),
                "structural_score": structural_score,
                "schema_hits": meta["schema_hits"],
                "schema_relation_hits": meta["schema_relation_hits"],
                "entity_mentions": [
                    memory.get_entity_by_idx(entity_idx).content
                    for entity_idx in passage.entity_indices
                    if memory.get_entity_by_idx(entity_idx) is not None
                ],
                "best_segment_preview": parent_best_segment_map.get(dense_lookup_key, passage.content),
                "content": passage.content,
            }
        )

    results.sort(key=lambda item: item["final_score"], reverse=True)
    return results[:top_k]


def _format_passage_hits(
    passage_hits: List[Dict[str, Any]],
    *,
    preview_chars: int,
) -> List[Dict[str, Any]]:
    formatted: List[Dict[str, Any]] = []
    for item in passage_hits:
        record = dict(item)
        full_content = record.pop("content")
        preview = full_content[:preview_chars]
        if len(full_content) > preview_chars:
            preview += "..."
        record["content_preview"] = preview
        record["full_content"] = full_content
        formatted.append(record)
    return formatted


def _graph_modality_bucket(modality: str) -> str:
    value = str(modality or "").strip().lower()
    if value == "table":
        return "table"
    if value == "image":
        return "image"
    return "text"


def _normalize_query_text(query: str) -> str:
    return " ".join(str(query or "").strip().lower().split())


@dataclass(frozen=True)
class BookRAGStyleFilter:
    """
    Lightweight local replica of BookRAG's filter schema.

    BookRAG's GBC planner emits filters with the following shape:
      - filter_type in {"section", "image", "table", "page"}
      - filter_value optionally used for {"section", "page"}

    We keep the same structure here so the retrieval-time metadata is easy to
    compare against BookRAG's original behavior, even though the current
    HVM-GraphRAG retrieval runtime only applies the modality-relevant subset.
    """

    filter_type: str
    filter_value: str | None = None


@dataclass(frozen=True)
class BookRAGStyleFilterPlan:
    """
    Query analysis result inspired by BookRAG's GBC planning stage.

    We intentionally keep the public fields close to BookRAG:
      - query_type: simple / complex / global
      - filters: a list of structural filters
      - operation: optional aggregation intent for global-style queries

    Extra audit fields (`matched_keywords`, `reason`) are included so we can
    inspect later why a query was filtered a certain way.
    """

    query_type: str
    filters: List[BookRAGStyleFilter]
    operation: str | None
    matched_keywords: List[str]
    reason: str


@dataclass(frozen=True)
class BookRAGStyleSubQuestion:
    question: str
    type: str


@dataclass(frozen=True)
class BookRAGStylePlanResult:
    query_type: str
    original_query: str
    sub_questions: List[BookRAGStyleSubQuestion]
    filters: List[BookRAGStyleFilter]
    operation: str | None = None


def _dedupe_preserve_order(values: Iterable[str]) -> List[str]:
    seen: set[str] = set()
    deduped: List[str] = []
    for value in values:
        normalized = str(value or "").strip().lower()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(normalized)
    return deduped


def _build_bookrag_style_query_filter_plan(query: str) -> BookRAGStyleFilterPlan:
    """
    Build a BookRAG-style filter plan from the raw query text.

    This intentionally mirrors the *shape* of BookRAG's GBC planner:
      1. infer a coarse query type (`simple` / `complex` / `global`)
      2. construct a list of structural filters

    The implementation here is rule-based rather than LLM-based. That is a
    deliberate choice:
      - no extra model latency/cost during retrieval
      - deterministic and auditable
      - safe to add as an optional layer without perturbing the existing code

    Current runtime support:
      - `image` / `table` filters are translated into modality restrictions
      - `section` / `page` filters are preserved in metadata for future use
        but are not yet applied to candidate generation
    """
    normalized = _normalize_query_text(query)
    matched_keywords: List[str] = []
    filters: List[BookRAGStyleFilter] = []

    table_keywords = [
        "table",
        "tab.",
        "row",
        "column",
        "cell",
    ]
    image_keywords = [
        "figure",
        "fig.",
        "chart",
        "plot",
        "diagram",
        "image",
        "photo",
        "legend",
        "axis",
        "x-axis",
        "y-axis",
        "color",
        "red box",
        "blue box",
        "green box",
    ]
    section_patterns = [
        r"\bin the ([a-z0-9 \-]+?) section\b",
        r"\bfrom the ([a-z0-9 \-]+?) section\b",
        r"\baccording to the ([a-z0-9 \-]+?) section\b",
    ]

    page_range_match = re.search(r"\bpages?\s+(\d+)\s*-\s*(\d+)\b", normalized)
    page_single_match = re.search(r"\bpage\s+(\d+)\b", normalized)

    table_hits = [kw for kw in table_keywords if kw in normalized]
    image_hits = [kw for kw in image_keywords if kw in normalized]
    matched_keywords.extend(table_hits)
    matched_keywords.extend(image_hits)

    if table_hits:
        filters.append(BookRAGStyleFilter(filter_type="table"))
    if image_hits:
        filters.append(BookRAGStyleFilter(filter_type="image"))

    if page_range_match:
        page_value = f"{page_range_match.group(1)}-{page_range_match.group(2)}"
        filters.append(BookRAGStyleFilter(filter_type="page", filter_value=page_value))
        matched_keywords.append(f"pages {page_value}")
    elif page_single_match:
        page_value = page_single_match.group(1)
        filters.append(BookRAGStyleFilter(filter_type="page", filter_value=page_value))
        matched_keywords.append(f"page {page_value}")

    for pattern in section_patterns:
        section_match = re.search(pattern, normalized)
        if section_match:
            section_name = " ".join(section_match.group(1).split()).strip()
            if section_name:
                filters.append(
                    BookRAGStyleFilter(filter_type="section", filter_value=section_name)
                )
                matched_keywords.append(f"section:{section_name}")
            break

    operation: str | None = None
    if re.search(r"\bhow many\b|\bcount\b|\bnumber of\b", normalized):
        operation = "COUNT"
    elif re.search(r"\blist\b|\bwhich\b.*\ball\b|\bwhat are all\b", normalized):
        operation = "LIST"
    elif re.search(r"\bsummarize\b|\bsummary\b", normalized):
        operation = "SUMMARIZE"
    elif re.search(r"\banalyze\b|\bcompare\b|\btrend\b|\bdistribution\b", normalized):
        operation = "ANALYZE"

    if operation and filters:
        query_type = "global"
        reason = "query looks like a global aggregation over structurally filtered items"
    elif len(filters) > 1 and any(f.filter_type in {"image", "table"} for f in filters):
        query_type = "complex"
        reason = "query references multiple structural cues and may need cross-source grounding"
    else:
        query_type = "simple"
        reason = "query does not require BookRAG-style global aggregation planning"

    return BookRAGStyleFilterPlan(
        query_type=query_type,
        filters=filters,
        operation=operation,
        matched_keywords=_dedupe_preserve_order(matched_keywords),
        reason=reason,
    )


def _modalities_from_bookrag_style_filters(
    plan: BookRAGStyleFilterPlan,
) -> List[str]:
    """
    Convert BookRAG-style filters into the modality restrictions supported by
    the current retrieval runtime.

    The mapping is intentionally conservative:
      - image/table cues always keep `text` available for supporting context
        unless the query resembles a BookRAG-style global aggregation, where
        the original system would filter directly over that modality.
      - section/page filters are preserved in metadata only for now.
    """
    filter_types = {item.filter_type for item in plan.filters}
    has_image = "image" in filter_types
    has_table = "table" in filter_types

    if not has_image and not has_table:
        return []

    if plan.query_type == "global":
        if has_image and has_table:
            return ["image", "table"]
        if has_image:
            return ["image"]
        if has_table:
            return ["table"]

    recommended: List[str] = []
    if has_image:
        recommended.append("image")
    if has_table:
        recommended.append("table")
    recommended.append("text")
    return _dedupe_preserve_order(recommended)


def _resolve_query_modalities_with_bookrag_filter(
    *,
    query: str,
    requested_modalities: Iterable[str] | None,
    enable_bookrag_query_filter: bool,
) -> Tuple[List[str], Dict[str, Any]]:
    """
    Resolve the final modality filter for one query.

    Behavior:
      - if query filtering is disabled, preserve the original user-provided
        modalities unchanged
      - if enabled, build a BookRAG-style filter plan from the query
      - translate supported filters into modalities
      - merge them conservatively with explicit user constraints

    Merge policy:
      - if the user did not pass `--modalities`, use the query-derived set
      - if the user did pass `--modalities`, intersect the two
      - if the intersection is empty, keep the user-provided modalities to
        avoid a surprising hard failure caused by an over-aggressive heuristic
    """
    requested = _dedupe_preserve_order(requested_modalities or [])
    if not enable_bookrag_query_filter:
        return requested, {
            "enabled": False,
            "requested_modalities": requested,
            "effective_modalities": requested,
            "applied_modalities": [],
            "merge_strategy": "disabled",
            "plan": None,
        }

    plan = _build_bookrag_style_query_filter_plan(query)
    derived_modalities = _modalities_from_bookrag_style_filters(plan)

    if not derived_modalities:
        effective = requested
        merge_strategy = "no_supported_filter"
    elif not requested:
        effective = derived_modalities
        merge_strategy = "query_filter_only"
    else:
        intersection = [m for m in requested if m in set(derived_modalities)]
        if intersection:
            effective = intersection
            merge_strategy = "intersection"
        else:
            effective = requested
            merge_strategy = "user_modalities_override_empty_intersection"

    return effective, {
        "enabled": True,
        "requested_modalities": requested,
        "effective_modalities": effective,
        "applied_modalities": derived_modalities,
        "merge_strategy": merge_strategy,
        "plan": {
            "query_type": plan.query_type,
            "filters": [asdict(item) for item in plan.filters],
            "operation": plan.operation,
            "matched_keywords": plan.matched_keywords,
            "reason": plan.reason,
        },
    }


def _extract_json_object(text: str) -> Dict[str, Any]:
    cleaned = str(text or "").strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start < 0 or end < start:
        raise ValueError(f"Planner did not return a JSON object: {cleaned[:200]}")
    return json.loads(cleaned[start : end + 1])


def _normalize_bookrag_query_type(payload: Dict[str, Any]) -> str:
    """
    Normalize planner classification payloads into one of:
      - simple
      - complex
      - global

    We keep this tolerant because some models return slight schema drift such as:
      - {"type": "global"}
      - {"category": "Complex"}
      - {"query_type": "This is a global question"}
    """
    candidate_keys = [
        "query_type",
        "type",
        "category",
        "label",
        "classification",
    ]
    raw_value = ""
    for key in candidate_keys:
        if key in payload:
            raw_value = str(payload.get(key, "") or "").strip().lower()
            if raw_value:
                break

    if not raw_value:
        return "simple"
    if raw_value in {"simple", "complex", "global"}:
        return raw_value
    if "global" in raw_value:
        return "global"
    if "complex" in raw_value:
        return "complex"
    if "simple" in raw_value:
        return "simple"
    return "simple"


class BookRAGStyleTaskPlanner:
    """
    Local planner that mirrors BookRAG GBC planning:
      - classify into simple / complex / global
      - decompose complex questions into retrieval + synthesis steps
      - build filters + operation for global questions

    Unlike the lightweight query filter above, this planner is LLM-driven.
    """

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model_name: str,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> None:
        self.base_url = _normalize_chat_base_url(base_url)
        self.api_key = api_key
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        http_client = httpx.Client(timeout=300.0, trust_env=False)
        self.client = OpenAI(api_key=api_key, base_url=self.base_url, http_client=http_client)

    def close(self) -> None:
        close_fn = getattr(self.client, "close", None)
        if callable(close_fn):
            close_fn()

    def _json_completion(self, prompt: str) -> Tuple[Dict[str, Any], Dict[str, int]]:
        params: Dict[str, Any] = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        if self.base_url:
            params["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
        response = self.client.chat.completions.create(**params)
        content = str(response.choices[0].message.content or "").strip()
        return _extract_json_object(content), _usage_to_dict(getattr(response, "usage", None))

    @staticmethod
    def _classification_prompt(query: str) -> str:
        return f"""
You are an expert query analyzer. Your *only* task is to classify the user's question into one of three categories: "simple", "complex", or "global". Respond only with the specified JSON object.

Category Definitions:
1.  **simple**: The question can be fully answered by retrieving information from a **SINGLE, contiguous location** in the document (e.g., one specific paragraph, one complete table, or one figure).
    - This includes questions that require reasoning or comparison, **as long as all the necessary data is present within that single retrieved location.**
    - Example: "What is the title of Figure 2?"
    - Example: "How do 5% of the Latinos see economic upward mobility for their children?" -> This is SIMPLE because the answer can be found by looking at a **single chart or paragraph**. The system just needs to find that one location, find the '5%' data point, and extract the corresponding label.

2.  **complex**: The question requires decomposition into multiple simple sub-questions, where **each sub-question must be answered by a separate retrieval action**.
    - It often contains a **nested or indirect constraint** that requires a preliminary step to resolve before the main question can be answered.
    - The key is that **no single location contains all the necessary information** to answer the original query in one go.
    - Example: "What is the color of the personality vector in the soft-labled personality embedding matrix that with the highest Receptiviti score for User A2GBIFL43U1LKJ?" -> This is COMPLEX because it requires two **separate retrieval actions**: 1) A retrieval to find the vector with the 'highest score', and 2) a separate retrieval to find the 'color' associated with that vector.

3.  **global**: The question requires an **aggregation operation** (e.g., counting, listing, summarizing) over a set of items that are identified by a **clear structural filter**. The available filters are by layout type (such as `table`, `image`, `section`) or by a document range (such as `page` or `section`). This operation must be performed across all items that match the specified filter(s).
    - Example: "How many tables are in the document?" -> This is GLOBAL because the process is to **filter** for all items of type 'table' across the document, and then **aggregate** by counting them.
    - Example: "Summarize the discussion about 'data augmentation' in the 'Methodology' section." -> This is GLOBAL because the process is to first apply a **structural filter** (retrieve all content from the 'Methodology' section), and then perform an **aggregation** (summarize that content) with a focus on 'data augmentation'.

User Query: "{query}"
"""

    @staticmethod
    def _complex_prompt(query: str) -> str:
        return f"""
You are a query decomposition expert. You have been given a complex question.
Break it into simple sub-questions and assign each one a type.

Rules:
1. Each retrieval sub-question must be independently answerable from the document.
2. Retrieval sub-questions must not depend on each other's answers.
3. The final synthesis sub-question combines the earlier retrieved answers.

Return one JSON object with this shape:
{{
  "sub_questions": [
    {{"question": "<text>", "type": "retrieval"}},
    {{"question": "<text>", "type": "retrieval"}},
    {{"question": "<text>", "type": "synthesis"}}
  ]
}}

Complex Query: "{query}"
"""

    @staticmethod
    def _global_prompt(query: str) -> str:
        return f"""
You are a planning assistant for global document questions. Analyze the query
and produce both structural filters and an aggregation operation.

Allowed filter_type values:
- "section"
- "image"
- "table"
- "page"

filter_value may only be set for "section" and "page".

Allowed operation values:
- "COUNT"
- "LIST"
- "SUMMARIZE"
- "ANALYZE"

Return one JSON object with this shape:
{{
  "filters": [
    {{"filter_type": "image", "filter_value": null}}
  ],
  "operation": "COUNT"
}}

User Query: "{query}"
"""

    @staticmethod
    def _coerce_plan(query: str, payload: Dict[str, Any], query_type: str) -> BookRAGStylePlanResult:
        if query_type == "simple":
            return BookRAGStylePlanResult(
                query_type="simple",
                original_query=query,
                sub_questions=[],
                filters=[],
                operation=None,
            )
        if query_type == "complex":
            raw_sub_questions = payload.get("sub_questions") or []
            sub_questions: List[BookRAGStyleSubQuestion] = []
            for item in raw_sub_questions:
                if not isinstance(item, dict):
                    continue
                question_text = str(item.get("question", "")).strip()
                sub_type = str(item.get("type", "")).strip().lower()
                if not question_text or sub_type not in {"retrieval", "synthesis"}:
                    continue
                sub_questions.append(BookRAGStyleSubQuestion(question=question_text, type=sub_type))
            if not sub_questions:
                raise ValueError("Complex planner returned no valid sub-questions.")
            return BookRAGStylePlanResult(
                query_type="complex",
                original_query=query,
                sub_questions=sub_questions,
                filters=[],
                operation=None,
            )

        raw_filters = payload.get("filters") or []
        filters: List[BookRAGStyleFilter] = []
        for item in raw_filters:
            if not isinstance(item, dict):
                continue
            filter_type = str(item.get("filter_type", "")).strip().lower()
            filter_value = item.get("filter_value")
            if filter_type not in {"section", "image", "table", "page"}:
                continue
            filters.append(
                BookRAGStyleFilter(
                    filter_type=filter_type,
                    filter_value=(str(filter_value).strip() if filter_value is not None else None),
                )
            )
        operation = str(payload.get("operation", "")).strip().upper() or None
        return BookRAGStylePlanResult(
            query_type="global",
            original_query=query,
            sub_questions=[],
            filters=filters,
            operation=operation,
        )

    def analyze_with_usage(self, query: str) -> Tuple[BookRAGStylePlanResult, Dict[str, int]]:
        classification_payload, total_usage = self._json_completion(self._classification_prompt(query))
        query_type = _normalize_bookrag_query_type(classification_payload)

        if query_type == "simple":
            return self._coerce_plan(query, {}, query_type), total_usage
        if query_type == "complex":
            complex_payload, usage = self._json_completion(self._complex_prompt(query))
            _accumulate_token_usage(total_usage, usage)
            return self._coerce_plan(query, complex_payload, query_type), total_usage

        global_payload, usage = self._json_completion(self._global_prompt(query))
        _accumulate_token_usage(total_usage, usage)
        return self._coerce_plan(query, global_payload, query_type), total_usage

    def classify_query_type_with_usage(self, query: str) -> Tuple[str, Dict[str, int]]:
        classification_payload, usage = self._json_completion(self._classification_prompt(query))
        query_type = _normalize_bookrag_query_type(classification_payload)
        return query_type, usage

    def classify_query_type_debug_with_usage(
        self, query: str
    ) -> Tuple[str, Dict[str, int], Dict[str, Any]]:
        classification_payload, usage = self._json_completion(self._classification_prompt(query))
        query_type = _normalize_bookrag_query_type(classification_payload)
        return query_type, usage, classification_payload

    def build_global_plan_with_usage(
        self, query: str
    ) -> Tuple[BookRAGStylePlanResult, Dict[str, int]]:
        global_payload, usage = self._json_completion(self._global_prompt(query))
        return self._coerce_plan(query, global_payload, "global"), usage


def _analyze_query_evidence_needs(query: str) -> Dict[str, Any]:
    """
    Lightweight query analysis for evidence fusion.

    This helper does *not* decide the final answer route. Instead, it estimates
    what kinds of evidence are likely to be useful when we merge graph and dense
    retrieval results.

    The return value is intentionally simple and auditable:
    - needs_table / needs_image:
        Whether the query explicitly references table / figure / chart style
        objects, so we should preserve those modalities when possible.
    - prefers_direct_text:
        Whether the query looks like local text grounding is important. Dense
        retrieval is usually stronger on these direct-answer text spans.
    - query_type:
        A coarse label for later debugging and analysis.
    - matched_keywords:
        Which lexical cues fired.

    We keep this rule-based on purpose:
    - deterministic
    - cheap
    - easy to inspect when a fused context behaves unexpectedly
    """
    normalized = _normalize_query_text(query)

    table_keywords = [
        "table",
        "row",
        "column",
        "tab.",
    ]
    image_keywords = [
        "figure",
        "fig.",
        "fig ",
        "chart",
        "graph",
        "plot",
        "image",
        "photo",
        "diagram",
        "legend",
        "axis",
        "x-axis",
        "y-axis",
        "red box",
        "blue box",
        "green box",
        "color",
    ]
    text_localization_keywords = [
        "according to the passage",
        "according to the text",
        "according to the paragraph",
        "according to the section",
        "in the paragraph",
        "in the passage",
        "in the text",
        "from the paragraph",
        "from the text",
        "what does the text say",
        "what does the passage say",
    ]

    table_hits = [kw for kw in table_keywords if kw in normalized]
    image_hits = [kw for kw in image_keywords if kw in normalized]
    text_hits = [kw for kw in text_localization_keywords if kw in normalized]

    if table_hits and image_hits:
        query_type = "table_and_figure_question"
    elif table_hits:
        query_type = "table_question"
    elif image_hits:
        query_type = "figure_chart_question"
    elif text_hits:
        query_type = "text_localization_question"
    else:
        query_type = "generic_question"

    return {
        "query_type": query_type,
        "needs_table": bool(table_hits),
        "needs_image": bool(image_hits),
        "prefers_direct_text": bool(text_hits) or not (table_hits or image_hits),
        "matched_keywords": table_hits + image_hits + text_hits,
    }


def _route_query_to_strategy_family(query: str) -> Dict[str, Any]:
    """
    Coarse query router used before the fused strategy.

    This helper implements the exact idea we discussed:
    - queries with strong graph-style cues -> `graph`
    - queries that look like single-modality / local text grounding -> `dense`
    - everything else -> `fused`

    Important design choice:
    We keep this as a *high-confidence* router, not a hard full classifier.
    If the query is ambiguous, we intentionally fall back to `fused` instead of
    over-committing to graph or dense.
    """
    normalized = _normalize_query_text(query)

    graph_patterns = [
        r"\bfigure\b",
        r"\bfig\.\b",
        r"\bdiagram\b",
        r"\bimage\b",
        r"\bphoto\b",
        r"\btable\b",
        r"\bchart\b",
        r"\bgraph\b",
        r"\bplot\b",
        r"legend",
        r"x-axis",
        r"y-axis",
        r"axis",
        r"red box",
        r"blue box",
        r"green box",
        r"pipeline",
        r"branch",
        r"color",
    ]

    dense_patterns = [
        r"according to the passage",
        r"according to the text",
        r"according to the paragraph",
        r"according to the section",
        r"in the passage",
        r"in the text",
        r"in the paragraph",
        r"from the text",
        r"from the paragraph",
        r"what does the text say",
        r"what does the passage say",
    ]

    single_modality_dense_patterns = [
        r"how many",
        r"what percentage",
        r"percentage gap",
        r"what is the gap",
        r"average",
        r"total",
        r"highest",
        r"lowest",
        r"most",
        r"least",
        r"number of",
        r"count",
    ]

    matched_graph = [p for p in graph_patterns if re.search(p, normalized)]
    matched_dense = [p for p in dense_patterns if re.search(p, normalized)]
    matched_single_dense = [
        p for p in single_modality_dense_patterns if re.search(p, normalized)
    ]

    if matched_graph:
        return {
            "route_family": "graph",
            "reason": "query contains explicit structure/multimodal anchors",
            "matched_keywords": matched_graph,
        }
    if matched_dense or matched_single_dense:
        return {
            "route_family": "dense",
            "reason": "query looks like local text grounding or direct single-span lookup",
            "matched_keywords": matched_dense + matched_single_dense,
        }
    return {
        "route_family": "fused",
        "reason": "query is ambiguous, so use evidence-level fusion",
        "matched_keywords": [],
    }


def _take_unique_by_bucket(
    passages: Sequence[Dict[str, Any]],
    *,
    used_chunk_ids: set[str],
    bucket_name: str | None = None,
    top_k: int = 1,
) -> List[Dict[str, Any]]:
    """
    Pick up to `top_k` unique passages, optionally constrained to one modality
    bucket.

    This is the core primitive used by the fused strategy. We keep it tiny and
    explicit so the final context assembly is easy to reason about:
    - no hidden reranking
    - no hidden side effects besides marking used chunk ids
    """
    selected: List[Dict[str, Any]] = []
    for item in passages:
        chunk_id = str(item.get("chunk_id", ""))
        if chunk_id in used_chunk_ids:
            continue
        if bucket_name is not None and _graph_modality_bucket(item.get("modality", "")) != bucket_name:
            continue
        selected.append(dict(item))
        used_chunk_ids.add(chunk_id)
        if len(selected) >= top_k:
            break
    return selected


def _merge_unique_passages(
    primary_passages: Sequence[Dict[str, Any]],
    supplemental_passages: Sequence[Dict[str, Any]],
    *,
    supplemental_top_k: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    merged: List[Dict[str, Any]] = [dict(item) for item in primary_passages]
    used_chunk_ids = {str(item.get("chunk_id", "")) for item in merged}
    added: List[Dict[str, Any]] = []
    for item in supplemental_passages:
        chunk_id = str(item.get("chunk_id", ""))
        if chunk_id in used_chunk_ids:
            continue
        added.append(dict(item))
        used_chunk_ids.add(chunk_id)
        if len(added) >= supplemental_top_k:
            break
    merged.extend(added)
    return merged, added


def _select_graph_passages_by_bucket(
    passage_hits: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    graph 检索按三类模态分别截断：
    - text: 普通文本 + list 等非 table/image 模态
    - table
    - image

    当前固定配额：
    - text: top 10
    - table: top 2
    - image: top 3

    再合并后按最终分数排序，避免某一类模态把其它模态完全挤掉。
    """
    buckets: Dict[str, List[Dict[str, Any]]] = {"text": [], "table": [], "image": []}
    for item in passage_hits:
        bucket = _graph_modality_bucket(item.get("modality", ""))
        buckets[bucket].append(item)

    merged: List[Dict[str, Any]] = []
    limits = {
        "text": GRAPH_TEXT_TOP_K,
        "table": GRAPH_TABLE_TOP_K,
        "image": GRAPH_IMAGE_TOP_K,
    }
    for bucket_name in ("text", "table", "image"):
        bucket_items = sorted(
            buckets[bucket_name],
            key=lambda row: float(row.get("final_score", float("-inf"))),
            reverse=True,
        )
        merged.extend(bucket_items[: limits[bucket_name]])

    merged.sort(key=lambda row: float(row.get("final_score", float("-inf"))), reverse=True)
    return merged


def _filter_schema_hits_by_threshold(
    schema_hits: Sequence[Dict[str, Any]],
    *,
    score_threshold: float,
) -> List[Dict[str, Any]]:
    return [
        dict(item)
        for item in schema_hits
        if float(item.get("score", float("-inf"))) >= score_threshold
    ]


def _attach_passage_metadata(
    memory: ThreeLayerMemory,
    dense_hits: List[Dict[str, Any]],
    *,
    preview_chars: int,
) -> List[Dict[str, Any]]:
    enriched: List[Dict[str, Any]] = []
    for item in dense_hits:
        record = dict(item)
        passage = None
        if int(record.get("passage_idx", -1)) >= 0:
            passage = memory.get_passage_by_idx(int(record["passage_idx"]))
        if passage is None:
            for candidate in memory.passage_layer:
                if str(candidate.chunk_id) == str(record["chunk_id"]):
                    passage = candidate
                    break
        if passage is None:
            continue

        full_content = passage.content
        record.update(
            {
                "passage_idx": passage.idx,
                "chunk_id": passage.chunk_id,
                "modality": passage.modality,
                "entity_mentions": [
                    memory.get_entity_by_idx(entity_idx).content
                    for entity_idx in passage.entity_indices
                    if memory.get_entity_by_idx(entity_idx) is not None
                ],
                "content_preview": _truncate_text(full_content, preview_chars),
                "full_content": full_content,
            }
        )
        enriched.append(record)
    return enriched


class RetrievalRuntime:
    """
    复用 memory / vdb / embedding model，避免批量问题时每题重复加载。
    """

    def __init__(
        self,
        *,
        memory_path: str,
        schema_vdb_path: str = "",
        schema_relation_vdb_path: str = "",
        chunk_vdb_path: str = "",
        embedding_model_name: str = "",
        mineru_root: str = "",
        tree_json_path: str = "",
    ) -> None:
        self.memory_path = Path(memory_path).resolve()
        self.memory = ThreeLayerMemory.load(str(self.memory_path))
        stub_args = argparse.Namespace(
            schema_vdb_path=schema_vdb_path,
            schema_relation_vdb_path=schema_relation_vdb_path,
            chunk_vdb_path=chunk_vdb_path,
        )
        self.vdb_paths = _resolve_vdb_paths(self.memory_path, stub_args)
        self.chunk_df = _load_vdb(self.vdb_paths["chunk"])
        self.schema_df = _load_optional_vdb(self.vdb_paths["schema"])
        self.relation_df = _load_optional_vdb(self.vdb_paths["schema_relation"])
        self.graph_structure_available = not self.schema_df.empty
        if self.graph_structure_available:
            sample_embedding = np.asarray(self.schema_df.iloc[0]["embedding"], dtype=np.float32)
        else:
            sample_embedding = np.asarray(self.chunk_df.iloc[0]["embedding"], dtype=np.float32)
        self.embedding_model_name = embedding_model_name or _infer_embedding_model_name(sample_embedding.shape[0])
        self.embedding_model = _build_embedding_model(self.embedding_model_name)
        self.mineru_lookup = MinerUMetadataLookup(
            memory_path=str(self.memory_path),
            mineru_root=mineru_root,
            tree_json_path=tree_json_path,
        )

    def _dense_fallback_for_missing_graph(
        self,
        *,
        query: str,
        strategy_name: str,
        passage_top_k: int,
        modalities: Iterable[str] | None,
        preview_chars: int,
        bookrag_query_filtering: bool,
    ) -> Dict[str, Any]:
        dense_result = self.retrieve_dense_baseline(
            query,
            passage_top_k=passage_top_k,
            modalities=modalities,
            preview_chars=preview_chars,
            bookrag_query_filtering=bookrag_query_filtering,
        )
        payload = dict(dense_result)
        stats = dict(payload.get("stats", {}))
        stats["graph_structure_available"] = False
        stats["fallback_reason"] = "missing_or_empty_schema_vdb"
        payload.update(
            {
                "mode": strategy_name,
                "selected_mode": "dense_baseline",
                "schema_vdb_path": str(self.vdb_paths["schema"]),
                "schema_relation_vdb_path": str(self.vdb_paths["schema_relation"]),
                "chunk_vdb_path": str(self.vdb_paths["chunk"]),
                "graph_structure_available": False,
                "fallback_reason": "missing_or_empty_schema_vdb",
                "stats": stats,
            }
        )
        return payload

    def close(self) -> None:
        close_fn = getattr(self.embedding_model, "close", None)
        if callable(close_fn):
            close_fn()
        self.mineru_lookup.close()

    def retrieve_graph_guided(
        self,
        query: str,
        *,
        schema_top_k: int,
        schema_relation_top_k: int,
        passage_top_k: int,
        modalities: Iterable[str] | None,
        preview_chars: int,
        bookrag_query_filtering: bool = False,
    ) -> Dict[str, Any]:
        if not self.graph_structure_available:
            return self._dense_fallback_for_missing_graph(
                query=query,
                strategy_name="graph_guided",
                passage_top_k=passage_top_k,
                modalities=modalities,
                preview_chars=preview_chars,
                bookrag_query_filtering=bookrag_query_filtering,
            )
        effective_modalities, query_filter_info = _resolve_query_modalities_with_bookrag_filter(
            query=query,
            requested_modalities=modalities,
            enable_bookrag_query_filter=bookrag_query_filtering,
        )
        structure_query_vec = _encode_query(self.embedding_model, query, "query_to_fact")
        passage_query_vec = _encode_query(self.embedding_model, query, "query_to_passage")
        raw_schema_hits = _rank_vdb_rows(self.schema_df, structure_query_vec, top_k=schema_top_k)
        raw_relation_hits = _rank_vdb_rows(self.relation_df, structure_query_vec, top_k=schema_relation_top_k)
        schema_hits, relation_hits = _expand_schema_hits_with_connected_relations(
            self.memory,
            raw_schema_hits,
            raw_relation_hits,
        )
        candidates = _collect_candidate_passages(self.memory, schema_hits, relation_hits)
        reranked_passages = _rerank_passages(
            self.memory,
            self.chunk_df,
            passage_query_vec,
            candidates,
            top_k=max(
                passage_top_k,
                GRAPH_TEXT_TOP_K + GRAPH_TABLE_TOP_K + GRAPH_IMAGE_TOP_K,
            ),
            allowed_modalities=effective_modalities,
        )
        passage_hits = _select_graph_passages_by_bucket(reranked_passages)
        return {
            "query": query,
            "mode": "graph_guided",
            "embedding_model": self.embedding_model_name,
            "memory_path": str(self.memory_path),
            "schema_vdb_path": str(self.vdb_paths["schema"]),
            "schema_relation_vdb_path": str(self.vdb_paths["schema_relation"]),
            "chunk_vdb_path": str(self.vdb_paths["chunk"]),
            "query_filter_info": query_filter_info,
            "top_schemas": schema_hits,
            "top_schema_relations": relation_hits,
            "top_passages": self.mineru_lookup.enrich_passages(
                _format_passage_hits(passage_hits, preview_chars=preview_chars)
            ),
            "stats": {
                "num_schema_hits": len(schema_hits),
                "num_raw_schema_hits": len(raw_schema_hits),
                "num_schema_relation_hits": len(relation_hits),
                "num_raw_schema_relation_hits": len(raw_relation_hits),
                "num_candidate_passages": len(candidates),
                "num_reranked_passages": len(reranked_passages),
                "num_final_passages": len(passage_hits),
                "num_text_passages": sum(1 for item in passage_hits if _graph_modality_bucket(item.get("modality", "")) == "text"),
                "num_table_passages": sum(1 for item in passage_hits if _graph_modality_bucket(item.get("modality", "")) == "table"),
                "num_image_passages": sum(1 for item in passage_hits if _graph_modality_bucket(item.get("modality", "")) == "image"),
            },
        }

    def retrieve_dense_baseline(
        self,
        query: str,
        *,
        passage_top_k: int,
        modalities: Iterable[str] | None,
        preview_chars: int,
        bookrag_query_filtering: bool = False,
    ) -> Dict[str, Any]:
        effective_modalities, query_filter_info = _resolve_query_modalities_with_bookrag_filter(
            query=query,
            requested_modalities=modalities,
            enable_bookrag_query_filter=bookrag_query_filtering,
        )
        passage_query_vec = _encode_query(self.embedding_model, query, "query_to_passage")
        dense_hits = _rank_chunk_candidates(
            self.chunk_df,
            passage_query_vec,
            top_k=passage_top_k,
            allowed_modalities=effective_modalities,
        )
        top_passages = self.mineru_lookup.enrich_passages(
            _attach_passage_metadata(self.memory, dense_hits, preview_chars=preview_chars)
        )
        return {
            "query": query,
            "mode": "dense_baseline",
            "embedding_model": self.embedding_model_name,
            "memory_path": str(self.memory_path),
            "chunk_vdb_path": str(self.vdb_paths["chunk"]),
            "query_filter_info": query_filter_info,
            "top_passages": top_passages,
            "stats": {
                "num_final_passages": len(top_passages),
            },
        }

    def retrieve_dense_bucketed(
        self,
        query: str,
        *,
        passage_top_k: int,
        modalities: Iterable[str] | None,
        preview_chars: int,
        bookrag_query_filtering: bool = False,
    ) -> Dict[str, Any]:
        """
        Dense-only ablation with graph-style modality bucketing.

        This keeps the dense retrieval stage unchanged, but replaces the final
        flat top-k cutoff with the same text/table/image bucket caps used by
        the graph-guided branch.
        """
        effective_modalities, query_filter_info = _resolve_query_modalities_with_bookrag_filter(
            query=query,
            requested_modalities=modalities,
            enable_bookrag_query_filter=bookrag_query_filtering,
        )
        passage_query_vec = _encode_query(self.embedding_model, query, "query_to_passage")
        dense_hits = _rank_chunk_candidates(
            self.chunk_df,
            passage_query_vec,
            top_k=max(
                passage_top_k,
                GRAPH_TEXT_TOP_K + GRAPH_TABLE_TOP_K + GRAPH_IMAGE_TOP_K,
            ),
            allowed_modalities=effective_modalities,
        )
        dense_passages = _attach_passage_metadata(
            self.memory,
            dense_hits,
            preview_chars=preview_chars,
        )
        dense_bucket_inputs: List[Dict[str, Any]] = []
        for item in dense_passages:
            record = dict(item)
            record["final_score"] = float(record.get("dense_score", float("-inf")))
            dense_bucket_inputs.append(record)
        bucketed_hits = _select_graph_passages_by_bucket(dense_bucket_inputs)
        top_passages = self.mineru_lookup.enrich_passages(bucketed_hits)
        return {
            "query": query,
            "mode": "dense_bucketed",
            "selected_mode": "dense_bucketed",
            "embedding_model": self.embedding_model_name,
            "memory_path": str(self.memory_path),
            "chunk_vdb_path": str(self.vdb_paths["chunk"]),
            "query_filter_info": query_filter_info,
            "top_passages": top_passages,
            "stats": {
                "num_dense_hits": len(dense_hits),
                "num_dense_passages": len(dense_passages),
                "num_final_passages": len(top_passages),
                "num_text_passages": sum(1 for item in top_passages if _graph_modality_bucket(item.get("modality", "")) == "text"),
                "num_table_passages": sum(1 for item in top_passages if _graph_modality_bucket(item.get("modality", "")) == "table"),
                "num_image_passages": sum(1 for item in top_passages if _graph_modality_bucket(item.get("modality", "")) == "image"),
            },
        }

    def retrieve_bookrag_global(
        self,
        query: str,
        *,
        filter_plan: BookRAGStyleFilterPlan,
        passage_top_k: int,
        preview_chars: int,
    ) -> Dict[str, Any]:
        all_tree_passages = self.mineru_lookup.get_all_tree_passages(
            preview_chars=preview_chars,
            enable_image_vlm=False,
        )
        filtered_passages = _filter_tree_passages_with_bookrag_filters(
            all_tree_passages,
            filter_plan.filters,
        )
        operation = _normalize_bookrag_operation(filter_plan.operation)
        top_passages = filtered_passages[: max(passage_top_k, 0)] if passage_top_k > 0 else []
        return {
            "query": query,
            "mode": "bookrag_global",
            "selected_mode": "bookrag_global",
            "operation": operation,
            "embedding_model": self.embedding_model_name,
            "memory_path": str(self.memory_path),
            "tree_json_path": str(self.mineru_lookup.tree_json_path) if self.mineru_lookup.tree_json_path else "",
            "global_filter_plan": _serialize_bookrag_filter_plan(filter_plan),
            "top_passages": top_passages,
            "filtered_passages": filtered_passages,
            "stats": {
                "num_tree_passages": len(all_tree_passages),
                "num_filtered_passages": len(filtered_passages),
                "num_top_passages": len(top_passages),
                "num_image_passages": sum(
                    1 for item in filtered_passages if str(item.get("modality", "")).lower() == "image"
                ),
                "num_table_passages": sum(
                    1 for item in filtered_passages if str(item.get("modality", "")).lower() == "table"
                ),
                "num_text_passages": sum(
                    1 for item in filtered_passages if str(item.get("modality", "")).lower() == "text"
                ),
            },
        }

    def retrieve_graph_plus_dense(
        self,
        query: str,
        *,
        schema_top_k: int,
        schema_relation_top_k: int,
        passage_top_k: int,
        modalities: Iterable[str] | None,
        preview_chars: int,
        schema_score_threshold: float,
        dense_supplement_top_k: int,
        bookrag_query_filtering: bool = False,
    ) -> Dict[str, Any]:
        """
        Seed-schema graph_plus_dense behavior:

        1. Retrieve raw top-k schemas / schema relations
        2. Keep only schemas whose score >= threshold as seed schemas
        3. Use connected schema relations to perform one-hop expansion
        4. Collect graph candidate passages and rerank them
        5. Apply graph-side modality bucketing
        6. Append a few unique dense passages as direct-answer supplements

        If no seed schema survives the threshold, the method falls back to the
        dense baseline and uses dense passages only.
        """
        if not self.graph_structure_available:
            return self._dense_fallback_for_missing_graph(
                query=query,
                strategy_name="graph_plus_dense",
                passage_top_k=passage_top_k,
                modalities=modalities,
                preview_chars=preview_chars,
                bookrag_query_filtering=bookrag_query_filtering,
            )
        effective_modalities, query_filter_info = _resolve_query_modalities_with_bookrag_filter(
            query=query,
            requested_modalities=modalities,
            enable_bookrag_query_filter=bookrag_query_filtering,
        )
        structure_query_vec = _encode_query(self.embedding_model, query, "query_to_fact")
        passage_query_vec = _encode_query(self.embedding_model, query, "query_to_passage")
        raw_schema_hits = _rank_vdb_rows(self.schema_df, structure_query_vec, top_k=schema_top_k)
        raw_relation_hits = _rank_vdb_rows(self.relation_df, structure_query_vec, top_k=schema_relation_top_k)
        seed_schema_hits = _filter_schema_hits_by_threshold(
            raw_schema_hits,
            score_threshold=schema_score_threshold,
        )
        if seed_schema_hits:
            schema_hits, relation_hits = _expand_schema_hits_with_connected_relations(
                self.memory,
                seed_schema_hits,
                raw_relation_hits,
            )
            candidates = _collect_candidate_passages(self.memory, schema_hits, relation_hits)
            reranked_passages = _rerank_passages(
                self.memory,
                self.chunk_df,
                passage_query_vec,
                candidates,
                top_k=max(
                passage_top_k,
                GRAPH_TEXT_TOP_K + GRAPH_TABLE_TOP_K + GRAPH_IMAGE_TOP_K,
            ),
                allowed_modalities=effective_modalities,
            )
            passage_hits = _select_graph_passages_by_bucket(reranked_passages)
            graph_top_passages = self.mineru_lookup.enrich_passages(
                _format_passage_hits(passage_hits, preview_chars=preview_chars)
            )
        else:
            schema_hits = []
            relation_hits = []
            candidates = {}
            reranked_passages = []
            graph_top_passages = []

        dense_result = self.retrieve_dense_baseline(
            query,
            passage_top_k=passage_top_k,
            modalities=effective_modalities,
            preview_chars=preview_chars,
            bookrag_query_filtering=False,
        )
        max_schema_score = _max_top_schema_score(raw_schema_hits)

        if graph_top_passages:
            merged_passages, supplemental_passages = _merge_unique_passages(
                graph_top_passages,
                dense_result.get("top_passages", []),
                supplemental_top_k=dense_supplement_top_k,
            )
            final_passages = merged_passages
            selected_mode = "graph_plus_dense"
        else:
            supplemental_passages = []
            final_passages = list(dense_result.get("top_passages", []))
            selected_mode = "dense_baseline"

        return {
            "query": query,
            "mode": "graph_plus_dense",
            "selected_mode": selected_mode,
            "schema_score_threshold": schema_score_threshold,
            "max_top_schema_score": max_schema_score,
            "embedding_model": self.embedding_model_name,
            "memory_path": str(self.memory_path),
            "schema_vdb_path": str(self.vdb_paths["schema"]),
            "schema_relation_vdb_path": str(self.vdb_paths["schema_relation"]),
            "chunk_vdb_path": str(self.vdb_paths["chunk"]),
            "query_filter_info": query_filter_info,
            "seed_top_schemas": seed_schema_hits,
            "top_schemas": schema_hits,
            "top_schema_relations": relation_hits,
            "top_passages": final_passages,
            "supplemental_dense_passages": supplemental_passages,
            "stats": {
                "num_raw_schema_hits": len(raw_schema_hits),
                "num_seed_schema_hits": len(seed_schema_hits),
                "num_schema_hits": len(schema_hits),
                "num_raw_schema_relation_hits": len(raw_relation_hits),
                "num_schema_relation_hits": len(relation_hits),
                "num_candidate_passages": len(candidates),
                "num_reranked_passages": len(reranked_passages),
                "num_graph_passages": len(graph_top_passages),
                "num_dense_passages": len(dense_result.get("top_passages", [])),
                "num_supplemental_dense_passages": len(supplemental_passages),
                "num_final_passages": len(final_passages),
                "num_text_passages": sum(1 for item in final_passages if _graph_modality_bucket(item.get("modality", "")) == "text"),
                "num_table_passages": sum(1 for item in final_passages if _graph_modality_bucket(item.get("modality", "")) == "table"),
                "num_image_passages": sum(1 for item in final_passages if _graph_modality_bucket(item.get("modality", "")) == "image"),
            },
        }

    def retrieve_graph_plus_dense_no_bucket(
        self,
        query: str,
        *,
        schema_top_k: int,
        schema_relation_top_k: int,
        passage_top_k: int,
        modalities: Iterable[str] | None,
        preview_chars: int,
        schema_score_threshold: float,
        dense_supplement_top_k: int,
        bookrag_query_filtering: bool = False,
    ) -> Dict[str, Any]:
        if not self.graph_structure_available:
            return self._dense_fallback_for_missing_graph(
                query=query,
                strategy_name="graph_plus_dense_no_bucket",
                passage_top_k=passage_top_k,
                modalities=modalities,
                preview_chars=preview_chars,
                bookrag_query_filtering=bookrag_query_filtering,
            )
        effective_modalities, query_filter_info = _resolve_query_modalities_with_bookrag_filter(
            query=query,
            requested_modalities=modalities,
            enable_bookrag_query_filter=bookrag_query_filtering,
        )
        structure_query_vec = _encode_query(self.embedding_model, query, "query_to_fact")
        passage_query_vec = _encode_query(self.embedding_model, query, "query_to_passage")
        raw_schema_hits = _rank_vdb_rows(self.schema_df, structure_query_vec, top_k=schema_top_k)
        raw_relation_hits = _rank_vdb_rows(self.relation_df, structure_query_vec, top_k=schema_relation_top_k)
        seed_schema_hits = _filter_schema_hits_by_threshold(
            raw_schema_hits,
            score_threshold=schema_score_threshold,
        )
        if seed_schema_hits:
            schema_hits, relation_hits = _expand_schema_hits_with_connected_relations(
                self.memory,
                seed_schema_hits,
                raw_relation_hits,
            )
            candidates = _collect_candidate_passages(self.memory, schema_hits, relation_hits)
            reranked_passages = _rerank_passages(
                self.memory,
                self.chunk_df,
                passage_query_vec,
                candidates,
                top_k=passage_top_k,
                allowed_modalities=effective_modalities,
            )
            graph_top_passages = self.mineru_lookup.enrich_passages(
                _format_passage_hits(reranked_passages[:passage_top_k], preview_chars=preview_chars)
            )
        else:
            schema_hits = []
            relation_hits = []
            candidates = {}
            reranked_passages = []
            graph_top_passages = []

        dense_result = self.retrieve_dense_baseline(
            query,
            passage_top_k=passage_top_k,
            modalities=effective_modalities,
            preview_chars=preview_chars,
            bookrag_query_filtering=False,
        )
        merged_passages, supplemental_passages = _merge_unique_passages(
            graph_top_passages,
            dense_result.get("top_passages", []),
            supplemental_top_k=dense_supplement_top_k if graph_top_passages else passage_top_k,
        )
        final_passages = merged_passages
        return {
            "query": query,
            "mode": "graph_plus_dense_no_bucket",
            "selected_mode": "graph_plus_dense_no_bucket" if graph_top_passages else "dense_baseline",
            "schema_score_threshold": schema_score_threshold,
            "max_top_schema_score": _max_top_schema_score(raw_schema_hits),
            "embedding_model": self.embedding_model_name,
            "memory_path": str(self.memory_path),
            "schema_vdb_path": str(self.vdb_paths["schema"]),
            "schema_relation_vdb_path": str(self.vdb_paths["schema_relation"]),
            "chunk_vdb_path": str(self.vdb_paths["chunk"]),
            "query_filter_info": query_filter_info,
            "seed_top_schemas": seed_schema_hits,
            "top_schemas": schema_hits,
            "top_schema_relations": relation_hits,
            "top_passages": final_passages,
            "supplemental_dense_passages": supplemental_passages,
            "stats": {
                "num_raw_schema_hits": len(raw_schema_hits),
                "num_seed_schema_hits": len(seed_schema_hits),
                "num_schema_hits": len(schema_hits),
                "num_raw_schema_relation_hits": len(raw_relation_hits),
                "num_schema_relation_hits": len(relation_hits),
                "num_candidate_passages": len(candidates),
                "num_reranked_passages": len(reranked_passages),
                "num_graph_passages": len(graph_top_passages),
                "num_dense_passages": len(dense_result.get("top_passages", [])),
                "num_supplemental_dense_passages": len(supplemental_passages),
                "num_final_passages": len(final_passages),
                "num_text_passages": sum(1 for item in final_passages if _graph_modality_bucket(item.get("modality", "")) == "text"),
                "num_table_passages": sum(1 for item in final_passages if _graph_modality_bucket(item.get("modality", "")) == "table"),
                "num_image_passages": sum(1 for item in final_passages if _graph_modality_bucket(item.get("modality", "")) == "image"),
            },
        }

    def retrieve_graph_dense_fused(
        self,
        query: str,
        *,
        schema_top_k: int,
        schema_relation_top_k: int,
        passage_top_k: int,
        modalities: Iterable[str] | None,
        preview_chars: int,
        schema_score_threshold: float,
        dense_supplement_top_k: int,
        bookrag_query_filtering: bool = False,
    ) -> Dict[str, Any]:
        """
        Fused retrieval strategy designed to preserve complementary strengths:

        - Dense is usually better at direct local text grounding, especially for
          short factual spans and numeric answers written explicitly in prose.
        - Graph retrieval is usually better at preserving structure-aware
          evidence, especially when tables/figures/images are explicitly
          referenced by the question.

        Instead of routing to one side, we build the final context in slots:
        1. Dense direct-text evidence
        2. Graph structural text evidence
        3. Graph table/image evidence when the query suggests they matter
        4. Dense modality backfill if graph misses a required modality

        This keeps the implementation explicit and easy to ablate later.
        """
        if not self.graph_structure_available:
            return self._dense_fallback_for_missing_graph(
                query=query,
                strategy_name="graph_dense_fused",
                passage_top_k=passage_top_k,
                modalities=modalities,
                preview_chars=preview_chars,
                bookrag_query_filtering=bookrag_query_filtering,
            )
        effective_modalities, query_filter_info = _resolve_query_modalities_with_bookrag_filter(
            query=query,
            requested_modalities=modalities,
            enable_bookrag_query_filter=bookrag_query_filtering,
        )
        query_profile = _analyze_query_evidence_needs(query)
        structure_query_vec = _encode_query(self.embedding_model, query, "query_to_fact")
        passage_query_vec = _encode_query(self.embedding_model, query, "query_to_passage")

        raw_schema_hits = _rank_vdb_rows(self.schema_df, structure_query_vec, top_k=schema_top_k)
        raw_relation_hits = _rank_vdb_rows(self.relation_df, structure_query_vec, top_k=schema_relation_top_k)
        seed_schema_hits = _filter_schema_hits_by_threshold(
            raw_schema_hits,
            score_threshold=schema_score_threshold,
        )

        if seed_schema_hits:
            schema_hits, relation_hits = _expand_schema_hits_with_connected_relations(
                self.memory,
                seed_schema_hits,
                raw_relation_hits,
            )
            candidates = _collect_candidate_passages(self.memory, schema_hits, relation_hits)
            reranked_passages = _rerank_passages(
                self.memory,
                self.chunk_df,
                passage_query_vec,
                candidates,
                top_k=max(
                    passage_top_k,
                    GRAPH_TEXT_TOP_K + GRAPH_TABLE_TOP_K + GRAPH_IMAGE_TOP_K,
                ),
                allowed_modalities=effective_modalities,
            )
            graph_passages = _select_graph_passages_by_bucket(reranked_passages)
            graph_top_passages = self.mineru_lookup.enrich_passages(
                _format_passage_hits(graph_passages, preview_chars=preview_chars)
            )
        else:
            schema_hits = []
            relation_hits = []
            candidates = {}
            reranked_passages = []
            graph_top_passages = []

        dense_result = self.retrieve_dense_baseline(
            query,
            passage_top_k=passage_top_k,
            modalities=effective_modalities,
            preview_chars=preview_chars,
            bookrag_query_filtering=False,
        )
        dense_passages = list(dense_result.get("top_passages", []))

        # Slot-based fusion. We explicitly reserve some capacity for dense text
        # and some capacity for graph structure-aware evidence instead of letting
        # one side fully dominate the final context.
        used_chunk_ids: set[str] = set()
        final_passages: List[Dict[str, Any]] = []

        dense_text_budget = 2 if query_profile["prefers_direct_text"] else 1
        graph_text_budget = 1 if query_profile["prefers_direct_text"] else 2
        graph_table_budget = 1 if query_profile["needs_table"] else 0
        graph_image_budget = 1 if query_profile["needs_image"] else 0

        # 1) Dense direct-text evidence: keep a couple of high-probability text
        # spans that are often decisive for exact short answers.
        final_passages.extend(
            _take_unique_by_bucket(
                dense_passages,
                used_chunk_ids=used_chunk_ids,
                bucket_name="text",
                top_k=dense_text_budget,
            )
        )

        # 2) Graph structural text evidence: keep graph text chunks because they
        # often carry section-level or structure-guided context that dense misses.
        final_passages.extend(
            _take_unique_by_bucket(
                graph_top_passages,
                used_chunk_ids=used_chunk_ids,
                bucket_name="text",
                top_k=graph_text_budget,
            )
        )

        # 3) Graph modality-specific evidence: when the query explicitly asks
        # about a table/figure/chart, preserve that modality from the graph side.
        if graph_table_budget > 0:
            final_passages.extend(
                _take_unique_by_bucket(
                    graph_top_passages,
                    used_chunk_ids=used_chunk_ids,
                    bucket_name="table",
                    top_k=graph_table_budget,
                )
            )
        if graph_image_budget > 0:
            final_passages.extend(
                _take_unique_by_bucket(
                    graph_top_passages,
                    used_chunk_ids=used_chunk_ids,
                    bucket_name="image",
                    top_k=graph_image_budget,
                )
            )

        # 4) Dense modality backfill: if the query clearly needs table/image but
        # graph failed to keep that modality, try to backfill from dense.
        if query_profile["needs_table"] and not any(
            _graph_modality_bucket(item.get("modality", "")) == "table" for item in final_passages
        ):
            final_passages.extend(
                _take_unique_by_bucket(
                    dense_passages,
                    used_chunk_ids=used_chunk_ids,
                    bucket_name="table",
                    top_k=1,
                )
            )
        if query_profile["needs_image"] and not any(
            _graph_modality_bucket(item.get("modality", "")) == "image" for item in final_passages
        ):
            final_passages.extend(
                _take_unique_by_bucket(
                    dense_passages,
                    used_chunk_ids=used_chunk_ids,
                    bucket_name="image",
                    top_k=1,
                )
            )

        # 5) Keep a small extra dense supplement as a safety net for direct
        # answer spans that graph may still miss.
        final_passages.extend(
            _take_unique_by_bucket(
                dense_passages,
                used_chunk_ids=used_chunk_ids,
                bucket_name=None,
                top_k=dense_supplement_top_k,
            )
        )

        return {
            "query": query,
            "mode": "graph_dense_fused",
            "selected_mode": "graph_dense_fused",
            "schema_score_threshold": schema_score_threshold,
            "max_top_schema_score": _max_top_schema_score(raw_schema_hits),
            "embedding_model": self.embedding_model_name,
            "memory_path": str(self.memory_path),
            "schema_vdb_path": str(self.vdb_paths["schema"]),
            "schema_relation_vdb_path": str(self.vdb_paths["schema_relation"]),
            "chunk_vdb_path": str(self.vdb_paths["chunk"]),
            "query_filter_info": query_filter_info,
            "query_profile": query_profile,
            "seed_top_schemas": seed_schema_hits,
            "top_schemas": schema_hits,
            "top_schema_relations": relation_hits,
            "top_passages": final_passages,
            "graph_candidate_passages": graph_top_passages,
            "dense_candidate_passages": dense_passages,
            "stats": {
                "num_raw_schema_hits": len(raw_schema_hits),
                "num_seed_schema_hits": len(seed_schema_hits),
                "num_schema_hits": len(schema_hits),
                "num_raw_schema_relation_hits": len(raw_relation_hits),
                "num_schema_relation_hits": len(relation_hits),
                "num_candidate_passages": len(candidates),
                "num_reranked_passages": len(reranked_passages),
                "num_graph_passages": len(graph_top_passages),
                "num_dense_passages": len(dense_passages),
                "num_final_passages": len(final_passages),
                "num_text_passages": sum(1 for item in final_passages if _graph_modality_bucket(item.get("modality", "")) == "text"),
                "num_table_passages": sum(1 for item in final_passages if _graph_modality_bucket(item.get("modality", "")) == "table"),
                "num_image_passages": sum(1 for item in final_passages if _graph_modality_bucket(item.get("modality", "")) == "image"),
            },
        }

    def retrieve_graph_dense_fused_routed(
        self,
        query: str,
        *,
        schema_top_k: int,
        schema_relation_top_k: int,
        passage_top_k: int,
        modalities: Iterable[str] | None,
        preview_chars: int,
        schema_score_threshold: float,
        dense_supplement_top_k: int,
        bookrag_query_filtering: bool = False,
    ) -> Dict[str, Any]:
        """
        Route to graph / dense / fused using a coarse, high-confidence query
        classifier.

        This is intentionally conservative:
        - confident graph cue   -> graph_guided
        - confident dense cue   -> dense_baseline
        - ambiguous query       -> graph_dense_fused
        """
        if not self.graph_structure_available:
            return self._dense_fallback_for_missing_graph(
                query=query,
                strategy_name="graph_dense_fused_routed",
                passage_top_k=passage_top_k,
                modalities=modalities,
                preview_chars=preview_chars,
                bookrag_query_filtering=bookrag_query_filtering,
            )
        route_decision = _route_query_to_strategy_family(query)
        route_family = str(route_decision.get("route_family", "fused"))

        if route_family == "graph":
            selected_result = self.retrieve_graph_guided(
                query,
                schema_top_k=schema_top_k,
                schema_relation_top_k=schema_relation_top_k,
                passage_top_k=passage_top_k,
                modalities=modalities,
                preview_chars=preview_chars,
                bookrag_query_filtering=bookrag_query_filtering,
            )
            selected_mode = "graph_guided"
        elif route_family == "dense":
            selected_result = self.retrieve_dense_baseline(
                query,
                passage_top_k=passage_top_k,
                modalities=modalities,
                preview_chars=preview_chars,
                bookrag_query_filtering=bookrag_query_filtering,
            )
            selected_mode = "dense_baseline"
        else:
            selected_result = self.retrieve_graph_dense_fused(
                query,
                schema_top_k=schema_top_k,
                schema_relation_top_k=schema_relation_top_k,
                passage_top_k=passage_top_k,
                modalities=modalities,
                preview_chars=preview_chars,
                schema_score_threshold=schema_score_threshold,
                dense_supplement_top_k=dense_supplement_top_k,
                bookrag_query_filtering=bookrag_query_filtering,
            )
            selected_mode = "graph_dense_fused"

        payload = dict(selected_result)
        payload.update(
            {
                "mode": "graph_dense_fused_routed",
                "selected_mode": selected_mode,
                "route_family": route_family,
                "route_reason": route_decision.get("reason", ""),
                "route_matched_keywords": route_decision.get("matched_keywords", []),
            }
        )
        return payload

class AnswerGenerator:
    """
    用同一个回答模型分别读取 graph-guided / dense baseline 检索结果。
    """

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model_name: str,
        max_tokens: int,
        temperature: float,
        prompt_style: str = "gbc",
    ) -> None:
        self.base_url = _normalize_chat_base_url(base_url)
        self.api_key = api_key
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.prompt_style = str(prompt_style or "gbc").strip().lower()
        http_client = httpx.Client(timeout=300.0, trust_env=False)
        self.client = OpenAI(api_key=api_key, base_url=self.base_url, http_client=http_client)

    def close(self) -> None:
        close_fn = getattr(self.client, "close", None)
        if callable(close_fn):
            close_fn()

    @staticmethod
    def _build_context(passages: Sequence[Dict[str, Any]]) -> str:
        blocks: List[str] = []
        for idx, passage in enumerate(passages, start=1):
            modality = str(passage.get("modality", "text"))
            chunk_id = str(passage.get("chunk_id", ""))
            content = str(passage.get("qa_content") or passage.get("full_content", ""))
            blocks.append(
                f"[Passage {idx}]\n"
                f"Chunk ID: {chunk_id}\n"
                f"Modality: {modality}\n"
                f"Content:\n{content}"
            )
        return "\n\n".join(blocks)

    @staticmethod
    def _extract_final_answer(response_text: str) -> str:
        text = (response_text or "").strip()
        if not text:
            return text
        marker = "Final Answer:"
        if marker in text:
            final = text.split(marker, 1)[1].strip()
            return final or text
        return text

    def answer_with_usage(self, question: str, passages: Sequence[Dict[str, Any]]) -> Tuple[str, Dict[str, int]]:
        context_str = self._build_context(passages)
        if self.prompt_style == "legacy_mmlong":
            prompt_text = LEGACY_MMLONG_ANSWER_PROMPT_TEMPLATE.format(
                query=question,
                context_str=context_str,
            )
            messages = [{"role": "user", "content": prompt_text}]
        else:
            user_prompt = BOOKRAG_GBC_ITER_GENERATION_USER_PROMPT.format(
                user_question=question,
                retrieved_content=context_str,
            )
            messages = [
                {"role": "system", "content": BOOKRAG_GBC_ITER_GENERATION_SYS_PROMPT},
                {"role": "user", "content": user_prompt},
            ]
        params: Dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        if self.base_url:
            params["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
        response = self.client.chat.completions.create(**params)
        raw_text = (response.choices[0].message.content or "").strip()
        return self._extract_final_answer(raw_text), _usage_to_dict(getattr(response, "usage", None))

    def answer(self, question: str, passages: Sequence[Dict[str, Any]]) -> str:
        answer_text, _ = self.answer_with_usage(question, passages)
        return answer_text

    def _complete_with_usage(
        self,
        *,
        messages: Sequence[Dict[str, str]],
    ) -> Tuple[str, Dict[str, int]]:
        params: Dict[str, Any] = {
            "model": self.model_name,
            "messages": list(messages),
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        if self.base_url:
            params["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
        response = self.client.chat.completions.create(**params)
        raw_text = (response.choices[0].message.content or "").strip()
        return raw_text, _usage_to_dict(getattr(response, "usage", None))

    def _analyze_single_global_passage_with_usage(
        self,
        *,
        original_query: str,
        operation: str,
        passage: Dict[str, Any],
    ) -> Tuple[str, Dict[str, int]]:
        operation_instruction = {
            "LIST": "Extract and list the key information from this filtered item that is relevant to the original question.",
            "SUMMARIZE": "Summarize the key points from this filtered item as they relate to the original question.",
            "ANALYZE": "Analyze this filtered item and state its contribution to answering the original question.",
        }.get(operation, "Analyze this filtered item in the context of the original global question.")
        page_idx = passage.get("page_idx")
        try:
            page_label = str(int(page_idx) + 1) if page_idx is not None else "unknown"
        except Exception:
            page_label = "unknown"
        content = str(passage.get("qa_content") or passage.get("full_content", "")).strip()
        user_prompt = BOOKRAG_GLOBAL_NODE_ANALYSIS_PROMPT_TEMPLATE.format(
            original_query=original_query,
            operation_instruction=operation_instruction,
            chunk_id=str(passage.get("chunk_id", "")),
            modality=str(passage.get("modality", "text")),
            page_label=page_label,
            content=content or "[Empty content]",
        )
        raw_text, usage = self._complete_with_usage(messages=[{"role": "user", "content": user_prompt}])
        return raw_text.strip(), usage

    def answer_global_question_with_usage(
        self,
        *,
        original_query: str,
        operation: str,
        filtered_passages: Sequence[Dict[str, Any]],
    ) -> Tuple[str, Dict[str, int], List[Dict[str, Any]]]:
        if not filtered_passages:
            return "No matching items found.", _empty_token_usage(), []

        total_usage = _empty_token_usage()
        partial_analyses: List[Dict[str, Any]] = []
        for idx, passage in enumerate(filtered_passages, start=1):
            analysis_text, usage = self._analyze_single_global_passage_with_usage(
                original_query=original_query,
                operation=operation,
                passage=passage,
            )
            _accumulate_token_usage(total_usage, usage)
            page_idx = passage.get("page_idx")
            try:
                page_label = str(int(page_idx) + 1) if page_idx is not None else "unknown"
            except Exception:
                page_label = "unknown"
            partial_analyses.append(
                {
                    "chunk_id": str(passage.get("chunk_id", "")),
                    "modality": str(passage.get("modality", "text")),
                    "page": page_label,
                    "analysis": analysis_text,
                }
            )

        partial_answers_str = "\n\n".join(
            (
                f"### Item {idx}\n"
                f"Chunk ID: {item['chunk_id']}\n"
                f"Modality: {item['modality']}\n"
                f"Page: {item['page']}\n"
                f"Analysis: {item['analysis']}"
            )
            for idx, item in enumerate(partial_analyses, start=1)
        )
        user_prompt = BOOKRAG_GLOBAL_SYNTHESIS_USER_PROMPT_TEMPLATE.format(
            original_query=original_query,
            operation=operation,
            partial_answers_str=partial_answers_str or "(no item-level analyses)",
        )
        raw_text, usage = self._complete_with_usage(
            messages=[
                {"role": "system", "content": BOOKRAG_GLOBAL_SYNTHESIS_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ]
        )
        _accumulate_token_usage(total_usage, usage)
        return self._extract_final_answer(raw_text), total_usage, partial_analyses

    def synthesize_complex_answer_with_usage(
        self,
        *,
        original_query: str,
        sub_question_plan: Sequence[BookRAGStyleSubQuestion],
        sub_question_results: Sequence[Dict[str, str]],
    ) -> Tuple[str, Dict[str, int]]:
        plan_str = "\n".join(
            f"{idx}. [{item.type}] {item.question}"
            for idx, item in enumerate(sub_question_plan, start=1)
        )
        sub_answers_str = "\n".join(
            f"### Sub-question {idx}\nQuestion: {item.get('question', '')}\nAnswer: {item.get('answer', '')}"
            for idx, item in enumerate(sub_question_results, start=1)
        )
        user_prompt = BOOKRAG_SYNTHESIS_USER_PROMPT_TEMPLATE.format(
            original_query=original_query,
            plan_str=plan_str or "(empty plan)",
            sub_answers_str=sub_answers_str or "(no sub-question answers)",
        )
        params: Dict[str, Any] = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": BOOKRAG_SYNTHESIS_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        if self.base_url:
            params["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
        response = self.client.chat.completions.create(**params)
        raw_text = (response.choices[0].message.content or "").strip()
        return self._extract_final_answer(raw_text), _usage_to_dict(getattr(response, "usage", None))


def retrieve_with_runtime_strategy(
    runtime: RetrievalRuntime,
    *,
    strategy: str,
    query: str,
    schema_top_k: int,
    schema_relation_top_k: int,
    passage_top_k: int,
    modalities: Iterable[str] | None,
    preview_chars: int,
    schema_score_threshold: float,
    dense_supplement_top_k: int,
    bookrag_query_filtering: bool = False,
) -> Dict[str, Any]:
    if strategy == "dense":
        return runtime.retrieve_dense_baseline(
            query,
            passage_top_k=passage_top_k,
            modalities=modalities,
            preview_chars=preview_chars,
            bookrag_query_filtering=bookrag_query_filtering,
        )
    if strategy == "dense_bucketed":
        return runtime.retrieve_dense_bucketed(
            query,
            passage_top_k=passage_top_k,
            modalities=modalities,
            preview_chars=preview_chars,
            bookrag_query_filtering=bookrag_query_filtering,
        )
    if strategy == "graph_plus_dense":
        return runtime.retrieve_graph_plus_dense(
            query,
            schema_top_k=schema_top_k,
            schema_relation_top_k=schema_relation_top_k,
            passage_top_k=passage_top_k,
            modalities=modalities,
            preview_chars=preview_chars,
            schema_score_threshold=schema_score_threshold,
            dense_supplement_top_k=dense_supplement_top_k,
            bookrag_query_filtering=bookrag_query_filtering,
        )
    if strategy == "graph_plus_dense_no_bucket":
        return runtime.retrieve_graph_plus_dense_no_bucket(
            query,
            schema_top_k=schema_top_k,
            schema_relation_top_k=schema_relation_top_k,
            passage_top_k=passage_top_k,
            modalities=modalities,
            preview_chars=preview_chars,
            schema_score_threshold=schema_score_threshold,
            dense_supplement_top_k=dense_supplement_top_k,
            bookrag_query_filtering=bookrag_query_filtering,
        )
    if strategy == "graph_dense_fused":
        return runtime.retrieve_graph_dense_fused(
            query,
            schema_top_k=schema_top_k,
            schema_relation_top_k=schema_relation_top_k,
            passage_top_k=passage_top_k,
            modalities=modalities,
            preview_chars=preview_chars,
            schema_score_threshold=schema_score_threshold,
            dense_supplement_top_k=dense_supplement_top_k,
            bookrag_query_filtering=bookrag_query_filtering,
        )
    if strategy == "graph_dense_fused_routed":
        return runtime.retrieve_graph_dense_fused_routed(
            query,
            schema_top_k=schema_top_k,
            schema_relation_top_k=schema_relation_top_k,
            passage_top_k=passage_top_k,
            modalities=modalities,
            preview_chars=preview_chars,
            schema_score_threshold=schema_score_threshold,
            dense_supplement_top_k=dense_supplement_top_k,
            bookrag_query_filtering=bookrag_query_filtering,
        )
    return runtime.retrieve_graph_guided(
        query,
        schema_top_k=schema_top_k,
        schema_relation_top_k=schema_relation_top_k,
        passage_top_k=passage_top_k,
        modalities=modalities,
        preview_chars=preview_chars,
        bookrag_query_filtering=bookrag_query_filtering,
    )


def _serialize_bookrag_filter_plan(plan: BookRAGStyleFilterPlan) -> Dict[str, Any]:
    return {
        "query_type": plan.query_type,
        "filters": [asdict(item) for item in plan.filters],
        "operation": plan.operation,
        "matched_keywords": plan.matched_keywords,
        "reason": plan.reason,
    }


def _bookrag_filter_plan_from_global_plan_result(
    plan: BookRAGStylePlanResult,
) -> BookRAGStyleFilterPlan:
    return BookRAGStyleFilterPlan(
        query_type=plan.query_type,
        filters=list(plan.filters),
        operation=plan.operation,
        matched_keywords=[],
        reason="llm_global_prompt",
    )


def _normalize_bookrag_operation(operation: str | None) -> str:
    value = str(operation or "").strip().upper()
    if value in {"COUNT", "LIST", "SUMMARIZE", "ANALYZE"}:
        return value
    return "ANALYZE"


def _infer_count_target_modality(query: str) -> str | None:
    normalized = _normalize_query_text(query)
    if re.search(r"\b(figures?|images?|charts?|plots?|diagrams?)\b", normalized):
        return "image"
    if re.search(r"\b(tables?)\b", normalized):
        return "table"
    return None


def _ensure_count_target_filter(
    *,
    query: str,
    filter_plan: BookRAGStyleFilterPlan,
) -> Tuple[BookRAGStyleFilterPlan, str | None]:
    operation = _normalize_bookrag_operation(filter_plan.operation)
    target_modality = _infer_count_target_modality(query)
    if operation != "COUNT" or target_modality is None:
        return filter_plan, target_modality

    existing_types = {
        str(item.filter_type or "").strip().lower()
        for item in filter_plan.filters
    }
    if target_modality in existing_types:
        return filter_plan, target_modality

    augmented_filters = list(filter_plan.filters) + [
        BookRAGStyleFilter(filter_type=target_modality, filter_value=None)
    ]
    augmented_plan = BookRAGStyleFilterPlan(
        query_type=filter_plan.query_type,
        filters=augmented_filters,
        operation=filter_plan.operation,
        matched_keywords=list(filter_plan.matched_keywords) + [f"auto_{target_modality}_count_filter"],
        reason=f"{filter_plan.reason}; auto-added {target_modality} filter for COUNT query",
    )
    return augmented_plan, target_modality


def _page_filter_matches(page_idx: Any, filter_value: str | None) -> bool:
    if page_idx is None:
        return False
    try:
        page_number = int(page_idx) + 1
    except Exception:
        return False

    text = str(filter_value or "").strip().lower()
    if not text:
        return True

    numbers = [int(item) for item in re.findall(r"\d+", text)]
    if not numbers:
        return False

    if len(numbers) >= 2 and any(token in text for token in ["-", "to", "through"]):
        start, end = numbers[0], numbers[1]
        lo, hi = sorted((start, end))
        return lo <= page_number <= hi

    return page_number in numbers


def _normalize_section_text(value: Any) -> str:
    text = str(value or "").strip().lower()
    if not text:
        return ""
    text = re.sub(r"\[[^\]]*\]", " ", text)
    text = re.sub(r"[_/|]+", " ", text)
    text = re.sub(r"[^a-z0-9\s>&-]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _section_filter_matches(title_path: Sequence[Any], filter_value: str | None) -> bool:
    target = _normalize_section_text(filter_value)
    if not target:
        return True
    normalized_titles = [_normalize_section_text(item) for item in title_path]
    joined = " > ".join(item for item in normalized_titles if item)
    if joined and target in joined:
        return True

    target_tokens = set(target.split())
    if joined and target_tokens and target_tokens.issubset(set(joined.split())):
        return True
    return False


def _section_candidate_matches(candidate: Any, normalized_target: str, target_tokens: set[str]) -> bool:
    normalized_candidate = _normalize_section_text(candidate)
    if not normalized_candidate:
        return False
    if normalized_target in normalized_candidate:
        return True
    if target_tokens and target_tokens.issubset(set(normalized_candidate.split())):
        return True
    return False


def _structured_section_candidates(passage: Dict[str, Any]) -> List[str]:
    candidates: List[str] = []

    def _append_candidate(value: Any) -> None:
        text = str(value or "").strip()
        if text:
            candidates.append(text)

    for item in passage.get("title_path", []) or []:
        _append_candidate(item)

    parent_node = passage.get("parent_node") or {}
    if isinstance(parent_node, dict):
        title_level = parent_node.get("title_level")
        if title_level not in (None, -1) or str(parent_node.get("block_type", "")).lower() == "title":
            _append_candidate(parent_node.get("content"))

    for node in passage.get("ancestor_nodes", []) or []:
        if not isinstance(node, dict):
            continue
        title_level = node.get("title_level")
        block_type = str(node.get("block_type", "")).lower()
        if title_level not in (None, -1) or block_type == "title":
            _append_candidate(node.get("content"))

    # Deduplicate while preserving order.
    deduped: List[str] = []
    seen: set[str] = set()
    for item in candidates:
        normalized = _normalize_section_text(item)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(item)
    return deduped


def _passage_matches_bookrag_filter(passage: Dict[str, Any], filter_item: BookRAGStyleFilter) -> bool:
    filter_type = str(filter_item.filter_type or "").strip().lower()
    if filter_type == "image":
        return str(passage.get("modality", "")).lower() == "image"
    if filter_type == "table":
        return str(passage.get("modality", "")).lower() == "table"
    if filter_type == "page":
        return _page_filter_matches(passage.get("page_idx"), filter_item.filter_value)
    if filter_type == "section":
        normalized_target = _normalize_section_text(filter_item.filter_value)
        if not normalized_target:
            return True
        target_tokens = set(normalized_target.split())
        if _section_filter_matches(passage.get("title_path", []), filter_item.filter_value):
            return True

        # Besides the flattened title_path, keep the raw structural ancestors in
        # play. Some MinerU trees keep the most useful section cue in parent /
        # ancestor title nodes rather than in a perfectly aligned title_path.
        for candidate in _structured_section_candidates(passage):
            if _section_candidate_matches(candidate, normalized_target, target_tokens):
                return True

        # Some MinerU trees store section cues in image captions / block content
        # rather than in the structural title path. We check a few lightweight
        # text fields so global section-count questions can still anchor to the
        # right passages.
        content_candidates = [
            passage.get("current_content"),
            passage.get("caption"),
            passage.get("footnote"),
            passage.get("content_preview"),
            passage.get("full_content"),
            passage.get("qa_content"),
        ]
        for candidate in content_candidates:
            if _section_candidate_matches(candidate, normalized_target, target_tokens):
                return True
        return False
    return True


def _filter_tree_passages_with_bookrag_filters(
    passages: Sequence[Dict[str, Any]],
    filters: Sequence[BookRAGStyleFilter],
) -> List[Dict[str, Any]]:
    if not filters:
        return [dict(item) for item in passages]

    # Match BookRAG-style semantics more closely:
    # - different filter types are combined with AND
    # - multiple values of the same filter type are combined with OR
    grouped_filters: Dict[str, List[BookRAGStyleFilter]] = {}
    for filter_item in filters:
        filter_type = str(filter_item.filter_type or "").strip().lower()
        if not filter_type:
            continue
        grouped_filters.setdefault(filter_type, []).append(filter_item)

    filtered: List[Dict[str, Any]] = []
    for passage in passages:
        if all(
            any(_passage_matches_bookrag_filter(passage, filter_item) for filter_item in same_type_filters)
            for same_type_filters in grouped_filters.values()
        ):
            filtered.append(dict(passage))
    return filtered


def retrieve_with_bookrag_query_routing(
    runtime: RetrievalRuntime,
    *,
    planner: BookRAGStyleTaskPlanner,
    query: str,
    strategy: str,
    schema_top_k: int,
    schema_relation_top_k: int,
    passage_top_k: int,
    modalities: Iterable[str] | None,
    preview_chars: int,
    schema_score_threshold: float,
    dense_supplement_top_k: int,
) -> Tuple[Dict[str, Any], float, Dict[str, int]]:
    """
    Query-routing layer for `--bookrag-query-filtering`.

    Behavior requested by the user:
    - classify the query with BookRAG's classification prompt
    - if simple/complex: keep using the selected retrieval strategy
    - if global: divert into a dedicated global retrieval path

    Only the global branch should invoke BookRAG's global-planning prompt.
    Simple and complex queries stay on the normal HVM-GraphRAG retrieval path.
    """
    classify_start = time.perf_counter()
    query_type, planning_usage = planner.classify_query_type_with_usage(query)
    planning_time_sec = time.perf_counter() - classify_start

    if query_type in {"simple", "complex"}:
        print(
            "[bookrag-query-filtering] "
            f"query_type={query_type} | route=selected_strategy | "
            f"question={_truncate_text(query, 160)}"
        )
        retrieval_result = retrieve_with_runtime_strategy(
            runtime,
            strategy=strategy,
            query=query,
            schema_top_k=schema_top_k,
            schema_relation_top_k=schema_relation_top_k,
            passage_top_k=passage_top_k,
            modalities=modalities,
            preview_chars=preview_chars,
            schema_score_threshold=schema_score_threshold,
            dense_supplement_top_k=dense_supplement_top_k,
            bookrag_query_filtering=False,
        )
        retrieval_result = dict(retrieval_result)
        retrieval_result["bookrag_query_route"] = {
            "enabled": True,
            "query_type": query_type,
            "route": "selected_strategy",
            "global_flow_used": False,
        }
        return retrieval_result, planning_time_sec, planning_usage

    global_plan_start = time.perf_counter()
    global_plan_result, global_plan_usage = planner.build_global_plan_with_usage(query)
    planning_time_sec += time.perf_counter() - global_plan_start
    _accumulate_token_usage(planning_usage, global_plan_usage)
    print(
        "[bookrag-query-filtering] "
        f"query_type={query_type} | route=global_retrieval_flow | "
        f"question={_truncate_text(query, 160)}"
    )

    raw_filter_plan = _bookrag_filter_plan_from_global_plan_result(global_plan_result)
    filter_plan, count_target_modality = _ensure_count_target_filter(
        query=query,
        filter_plan=raw_filter_plan,
    )
    requested_modalities = _dedupe_preserve_order(modalities) if modalities is not None else []
    derived_modalities = _modalities_from_bookrag_style_filters(filter_plan)
    query_filter_info = {
        "enabled": True,
        "query_text": query,
        "requested_modalities": requested_modalities,
        "effective_modalities": list(derived_modalities),
        "applied_modalities": list(derived_modalities),
        "merge_strategy": "bookrag_global_tree_filter",
        "count_target_modality": count_target_modality,
        "plan": _serialize_bookrag_filter_plan(filter_plan),
    }
    retrieval_result = runtime.retrieve_bookrag_global(
        query,
        filter_plan=filter_plan,
        passage_top_k=passage_top_k,
        preview_chars=preview_chars,
    )
    retrieval_result = dict(retrieval_result)
    retrieval_result["query_filter_info"] = query_filter_info
    retrieval_result["selected_mode"] = (
        f"bookrag_global::{retrieval_result.get('selected_mode', retrieval_result.get('mode', strategy))}"
    )
    retrieval_result["bookrag_query_route"] = {
        "enabled": True,
        "query_type": query_type,
        "route": "global_retrieval_flow",
        "global_flow_used": True,
        "effective_modalities": derived_modalities,
        "count_target_modality": count_target_modality,
        "global_filter_plan": _serialize_bookrag_filter_plan(filter_plan),
        "query_filter_info": query_filter_info,
        "status": "bookrag_global_tree_filter_and_operation",
    }
    return retrieval_result, planning_time_sec, planning_usage


def _run_bookrag_global_answering(
    *,
    answer_generator: AnswerGenerator,
    question: str,
    retrieval_result: Dict[str, Any],
) -> Tuple[str, Dict[str, int], List[Dict[str, Any]]]:
    operation = _normalize_bookrag_operation(
        retrieval_result.get("operation")
        or ((retrieval_result.get("global_filter_plan") or {}).get("operation"))
    )
    filtered_passages = retrieval_result.get("filtered_passages", []) or []
    if operation == "COUNT":
        count_target_modality = (
            ((retrieval_result.get("bookrag_query_route") or {}).get("count_target_modality"))
            or ((retrieval_result.get("query_filter_info") or {}).get("count_target_modality"))
        )
        memory_path_text = str(retrieval_result.get("memory_path", "")).lower()
        tree_json_path_text = str(retrieval_result.get("tree_json_path", "")).lower()
        is_m3doc_global = (
            "m3docvqa" in memory_path_text
            or "m3docvqa" in tree_json_path_text
            or "m3docrag" in memory_path_text
            or "m3docrag" in tree_json_path_text
        )

        if count_target_modality:
            target = str(count_target_modality).strip().lower()
            target_passages = [
                item for item in filtered_passages
                if str(item.get("modality", "")).strip().lower() == target
            ]
            if is_m3doc_global:
                if target == "image":
                    def _joined_title_path(item: Dict[str, Any]) -> str:
                        return " > ".join(str(x or "") for x in (item.get("title_path") or [])).lower()

                    def _preview_text(item: Dict[str, Any]) -> str:
                        return str(
                            item.get("content_preview")
                            or item.get("full_content")
                            or item.get("qa_content")
                            or ""
                        ).lower()

                    def _is_reference_like(item: Dict[str, Any]) -> bool:
                        joined = _joined_title_path(item)
                        return any(
                            token in joined
                            for token in [
                                "references",
                                "reference",
                                "bibliography",
                                "citations",
                                "notes [edit]",
                                "notes",
                            ]
                        )

                    def _has_meaningful_caption(item: Dict[str, Any]) -> bool:
                        preview = _preview_text(item)
                        return bool(preview) and "[image caption missing]" not in preview

                    non_reference = [item for item in target_passages if not _is_reference_like(item)]
                    captioned_non_reference = [
                        item for item in non_reference if _has_meaningful_caption(item)
                    ]
                    if captioned_non_reference:
                        target_passages = captioned_non_reference
                    elif non_reference:
                        target_passages = non_reference

                seen_keys: set[str] = set()
                unique_count = 0
                for item in target_passages:
                    unique_key = ""
                    if target == "image":
                        unique_key = str(item.get("img_path") or "").strip()
                    elif target == "table":
                        unique_key = str(item.get("table_body") or "").strip()
                    if not unique_key:
                        unique_key = str(item.get("index_id") or "").strip()
                    if not unique_key:
                        unique_key = str(item.get("chunk_id") or "").strip()
                    if not unique_key:
                        unique_key = json.dumps(
                            {
                                "page_idx": item.get("page_idx"),
                                "title_path": item.get("title_path"),
                                "content_preview": item.get("content_preview"),
                            },
                            ensure_ascii=False,
                            sort_keys=True,
                        )
                    if unique_key in seen_keys:
                        continue
                    seen_keys.add(unique_key)
                    unique_count += 1
                count_value = unique_count
            else:
                count_value = len(target_passages)
        else:
            count_value = len(filtered_passages)
        return str(count_value), _empty_token_usage(), []
    return answer_generator.answer_global_question_with_usage(
        original_query=question,
        operation=operation,
        filtered_passages=filtered_passages,
    )


def _collect_unique_top_passages(
    retrieval_results: Sequence[Dict[str, Any]],
    *,
    passage_top_k: int,
) -> List[Dict[str, Any]]:
    merged: List[Dict[str, Any]] = []
    used_chunk_ids: set[str] = set()
    for result in retrieval_results:
        for item in result.get("top_passages", []):
            chunk_id = str(item.get("chunk_id", ""))
            if chunk_id in used_chunk_ids:
                continue
            used_chunk_ids.add(chunk_id)
            merged.append(dict(item))
            if len(merged) >= passage_top_k:
                return merged
    return merged


def run_selected_strategy_with_optional_bookrag_planning(
    *,
    runtime: RetrievalRuntime,
    answer_generator: AnswerGenerator,
    planner: BookRAGStyleTaskPlanner | None,
    question: str,
    strategy: str,
    schema_top_k: int,
    schema_relation_top_k: int,
    passage_top_k: int,
    modalities: Iterable[str] | None,
    preview_chars: int,
    schema_score_threshold: float,
    dense_supplement_top_k: int,
    bookrag_query_filtering: bool = False,
) -> Dict[str, Any]:
    planning_time_sec = 0.0
    planning_usage = _empty_token_usage()
    tree_json_path = (
        str(runtime.mineru_lookup.tree_json_path)
        if runtime.mineru_lookup.tree_json_path is not None
        else "(missing tree.json)"
    )
    print(
        "[question-context] "
        f"doc_id={runtime.mineru_lookup.doc_id} | "
        f"tree_json_path={tree_json_path} | "
        f"question={_truncate_text(question, 160)}"
    )
    if bookrag_query_filtering:
        if planner is None:
            raise ValueError("bookrag_query_filtering=True requires a planner instance.")
        retrieval_start = time.perf_counter()
        retrieval_result, planning_time_sec, planning_usage = retrieve_with_bookrag_query_routing(
            runtime,
            planner=planner,
            query=question,
            strategy=strategy,
            schema_top_k=schema_top_k,
            schema_relation_top_k=schema_relation_top_k,
            passage_top_k=passage_top_k,
            modalities=modalities,
            preview_chars=preview_chars,
            schema_score_threshold=schema_score_threshold,
            dense_supplement_top_k=dense_supplement_top_k,
        )
        retrieval_time_sec = time.perf_counter() - retrieval_start - planning_time_sec
        answer_start = time.perf_counter()
        partial_global_analyses: List[Dict[str, Any]] = []
        route_info = retrieval_result.get("bookrag_query_route") or {}
        if route_info.get("global_flow_used"):
            output, answer_usage, partial_global_analyses = _run_bookrag_global_answering(
                answer_generator=answer_generator,
                question=question,
                retrieval_result=retrieval_result,
            )
        else:
            output, answer_usage = answer_generator.answer_with_usage(
                question,
                retrieval_result.get("top_passages", []),
            )
        answer_time_sec = time.perf_counter() - answer_start
        payload = {
            "plan": retrieval_result.get("bookrag_query_route"),
            "retrieval": retrieval_result,
            "output": output,
            "retrieval_time_sec": max(retrieval_time_sec, 0.0),
            "answer_time_sec": answer_time_sec,
            "answer_usage": answer_usage,
            "planning_time_sec": planning_time_sec,
            "planning_usage": planning_usage,
        }
        if partial_global_analyses:
            payload["global_partial_analyses"] = partial_global_analyses
        return payload

    retrieval_start = time.perf_counter()
    retrieval_result = retrieve_with_runtime_strategy(
        runtime,
        strategy=strategy,
        query=question,
        schema_top_k=schema_top_k,
        schema_relation_top_k=schema_relation_top_k,
        passage_top_k=passage_top_k,
        modalities=modalities,
        preview_chars=preview_chars,
        schema_score_threshold=schema_score_threshold,
        dense_supplement_top_k=dense_supplement_top_k,
        bookrag_query_filtering=bookrag_query_filtering,
    )
    retrieval_time_sec = time.perf_counter() - retrieval_start
    answer_start = time.perf_counter()
    output, answer_usage = answer_generator.answer_with_usage(
        question,
        retrieval_result.get("top_passages", []),
    )
    answer_time_sec = time.perf_counter() - answer_start
    return {
        "plan": None,
        "retrieval": retrieval_result,
        "output": output,
        "retrieval_time_sec": retrieval_time_sec,
        "answer_time_sec": answer_time_sec,
        "answer_usage": answer_usage,
        "planning_time_sec": planning_time_sec,
        "planning_usage": planning_usage,
    }


def build_retrieval_result(args: argparse.Namespace) -> Dict[str, Any]:
    runtime = RetrievalRuntime(
        memory_path=args.memory_path,
        schema_vdb_path=args.schema_vdb_path,
        schema_relation_vdb_path=args.schema_relation_vdb_path,
        chunk_vdb_path=args.chunk_vdb_path,
        embedding_model_name=args.embedding_model,
        mineru_root=getattr(args, "mineru_root", ""),
        tree_json_path=getattr(args, "tree_json_path", ""),
    )
    planner = None
    try:
        strategy = getattr(args, "retrieval_strategy", "graph")
        if getattr(args, "bookrag_query_filtering", False):
            planner = BookRAGStyleTaskPlanner(
                base_url=getattr(args, "planner_base_url", DEFAULT_PLANNER_BASE_URL),
                api_key=getattr(args, "planner_api_key", DEFAULT_PLANNER_API_KEY),
                model_name=getattr(args, "planner_model", DEFAULT_PLANNER_MODEL),
                max_tokens=getattr(args, "planner_max_tokens", 1024),
                temperature=getattr(args, "planner_temperature", 0.0),
            )
            routed_result, planning_time_sec, planning_usage = retrieve_with_bookrag_query_routing(
                runtime,
                planner=planner,
                query=args.query,
                strategy=strategy,
                schema_top_k=args.schema_top_k,
                schema_relation_top_k=args.schema_relation_top_k,
                passage_top_k=args.passage_top_k,
                modalities=args.modalities,
                preview_chars=args.preview_chars,
                schema_score_threshold=args.graph_plus_dense_schema_score_threshold,
                dense_supplement_top_k=args.graph_plus_dense_supplement_top_k,
            )
            routed_result = dict(routed_result)
            routed_result["bookrag_query_filtering"] = {
                "planning_time_sec": planning_time_sec,
                "planning_usage": planning_usage,
            }
            return routed_result
        if strategy == "dense":
            return runtime.retrieve_dense_baseline(
                args.query,
                passage_top_k=args.passage_top_k,
                modalities=args.modalities,
                preview_chars=args.preview_chars,
                bookrag_query_filtering=getattr(args, "bookrag_query_filtering", False),
            )
        if strategy == "graph_plus_dense":
            return runtime.retrieve_graph_plus_dense(
                args.query,
                schema_top_k=args.schema_top_k,
                schema_relation_top_k=args.schema_relation_top_k,
                passage_top_k=args.passage_top_k,
                modalities=args.modalities,
                preview_chars=args.preview_chars,
                schema_score_threshold=args.graph_plus_dense_schema_score_threshold,
                dense_supplement_top_k=args.graph_plus_dense_supplement_top_k,
                bookrag_query_filtering=getattr(args, "bookrag_query_filtering", False),
            )
        if strategy == "graph_plus_dense_no_bucket":
            return runtime.retrieve_graph_plus_dense_no_bucket(
                args.query,
                schema_top_k=args.schema_top_k,
                schema_relation_top_k=args.schema_relation_top_k,
                passage_top_k=args.passage_top_k,
                modalities=args.modalities,
                preview_chars=args.preview_chars,
                schema_score_threshold=args.graph_plus_dense_schema_score_threshold,
                dense_supplement_top_k=args.graph_plus_dense_supplement_top_k,
                bookrag_query_filtering=getattr(args, "bookrag_query_filtering", False),
            )
        if strategy == "graph_dense_fused":
            return runtime.retrieve_graph_dense_fused(
                args.query,
                schema_top_k=args.schema_top_k,
                schema_relation_top_k=args.schema_relation_top_k,
                passage_top_k=args.passage_top_k,
                modalities=args.modalities,
                preview_chars=args.preview_chars,
                schema_score_threshold=args.graph_plus_dense_schema_score_threshold,
                dense_supplement_top_k=args.graph_plus_dense_supplement_top_k,
                bookrag_query_filtering=getattr(args, "bookrag_query_filtering", False),
            )
        if strategy == "graph_dense_fused_routed":
            return runtime.retrieve_graph_dense_fused_routed(
                args.query,
                schema_top_k=args.schema_top_k,
                schema_relation_top_k=args.schema_relation_top_k,
                passage_top_k=args.passage_top_k,
                modalities=args.modalities,
                preview_chars=args.preview_chars,
                schema_score_threshold=args.graph_plus_dense_schema_score_threshold,
                dense_supplement_top_k=args.graph_plus_dense_supplement_top_k,
                bookrag_query_filtering=getattr(args, "bookrag_query_filtering", False),
            )
        return runtime.retrieve_graph_guided(
            args.query,
            schema_top_k=args.schema_top_k,
            schema_relation_top_k=args.schema_relation_top_k,
            passage_top_k=args.passage_top_k,
            modalities=args.modalities,
            preview_chars=args.preview_chars,
            bookrag_query_filtering=getattr(args, "bookrag_query_filtering", False),
        )
    finally:
        runtime.close()
        if planner is not None:
            planner.close()


def build_retrieval_result_for_query(
    *,
    memory_path: str,
    query: str,
    schema_vdb_path: str = "",
    schema_relation_vdb_path: str = "",
    chunk_vdb_path: str = "",
    embedding_model: str = "",
    schema_top_k: int = 5,
    schema_relation_top_k: int = 5,
    passage_top_k: int = 10,
    modalities: Iterable[str] | None = None,
    preview_chars: int = 240,
    mineru_root: str = "",
    tree_json_path: str = "",
    bookrag_query_filtering: bool = False,
) -> Dict[str, Any]:
    """
    方便批量模式复用单 query 检索逻辑。
    """
    class _Args:
        pass

    args = _Args()
    args.memory_path = memory_path
    args.query = query
    args.schema_vdb_path = schema_vdb_path
    args.schema_relation_vdb_path = schema_relation_vdb_path
    args.chunk_vdb_path = chunk_vdb_path
    args.embedding_model = embedding_model
    args.schema_top_k = schema_top_k
    args.schema_relation_top_k = schema_relation_top_k
    args.passage_top_k = passage_top_k
    args.retrieval_strategy = "graph"
    args.graph_plus_dense_schema_score_threshold = 0.615
    args.graph_plus_dense_supplement_top_k = 2
    args.modalities = list(modalities or [])
    args.preview_chars = preview_chars
    args.mineru_root = mineru_root
    args.tree_json_path = tree_json_path
    args.bookrag_query_filtering = bool(bookrag_query_filtering)
    args.output_path = ""
    return build_retrieval_result(args)


def _evaluate_answer_sets(
    records: List[Dict[str, Any]],
    *,
    judge_base_url: str,
    judge_api_key: str,
    judge_model: str,
) -> Dict[str, Any]:
    extractor = BookRAGStyleAnswerExtractor(
        base_url=judge_base_url,
        api_key=judge_api_key,
        model_name=judge_model,
    )
    graph_records = [
        {
            "question": item["question"],
            "answer": item["answer"],
            "answer_format": item["answer_format"],
            "output": item["graph_guided_output"],
        }
        for item in records
    ]
    dense_records = [
        {
            "question": item["question"],
            "answer": item["answer"],
            "answer_format": item["answer_format"],
            "output": item["dense_baseline_output"],
        }
        for item in records
    ]
    graph_evaluated, graph_summary = evaluate_mmlong_results(graph_records, extractor=extractor)
    dense_evaluated, dense_summary = evaluate_mmlong_results(dense_records, extractor=extractor)
    return {
        "graph_guided": {
            "summary": graph_summary,
            "details": graph_evaluated,
        },
        "dense_baseline": {
            "summary": dense_summary,
            "details": dense_evaluated,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retrieve passages from constructed MinerU graph-memory outputs.")
    parser.add_argument("--memory-path", required=True, help="Path to memory.json produced by the construction stage.")
    parser.add_argument("--query", default="", help="User query for retrieval.")
    parser.add_argument(
        "--samples-json",
        default="",
        help="Optional MMLongBench samples.json path. If provided, auto-select all questions matching this memory's PDF.",
    )
    parser.add_argument(
        "--doc-id",
        default="",
        help="Optional explicit doc_id override when using --samples-json. Default: infer from memory directory name.",
    )

    parser.add_argument("--schema-vdb-path", default="", help="Optional override for schema_embeddings/vdb_schema.parquet")
    parser.add_argument("--schema-relation-vdb-path", default="", help="Optional override for schema_relation_embeddings/vdb_schema_relation.parquet")
    parser.add_argument("--chunk-vdb-path", default="", help="Optional override for chunk_embeddings/vdb_chunk.parquet")
    parser.add_argument(
        "--embedding-model",
        default="",
        help=(
            "Embedding model used to encode the query. "
            "If empty, infer from MEMGRAPHRAG_EMBED_MODEL first, then fall back to vector dimension."
        ),
    )

    parser.add_argument("--schema-top-k", type=int, default=5)
    parser.add_argument("--schema-relation-top-k", type=int, default=5)
    parser.add_argument("--passage-top-k", type=int, default=10)
    parser.add_argument(
        "--retrieval-strategy",
        choices=["graph", "dense", "dense_bucketed", "graph_plus_dense", "graph_plus_dense_no_bucket", "graph_dense_fused", "graph_dense_fused_routed"],
        default="graph",
        help=(
            "Which retrieval strategy to use when returning a single retrieval result. "
            "`graph` keeps the current graph-guided pipeline, "
            "`dense` uses the dense baseline, "
            "`dense_bucketed` is the dense-only ablation that keeps the graph-style modality buckets, and "
            "`graph_plus_dense` uses graph when schema confidence is high and supplements it with a few dense chunks, while "
            "`graph_plus_dense_no_bucket` is the ablation variant that removes graph-side modality bucketing, and "
            "`graph_dense_fused` explicitly fuses dense direct-text evidence with graph structure-aware evidence, and "
            "`graph_dense_fused_routed` first does coarse query routing, then falls back to the fused strategy when needed."
        ),
    )
    parser.add_argument(
        "--graph-plus-dense-schema-score-threshold",
        type=float,
        default=0.615,
        help="Schema score threshold for the graph_plus_dense retrieval strategy.",
    )
    parser.add_argument(
        "--graph-plus-dense-supplement-top-k",
        type=int,
        default=2,
        help="How many unique dense chunks to append when graph_plus_dense chooses the graph route.",
    )
    parser.add_argument(
        "--graph-schema-score-threshold",
        type=float,
        default=-1.0,
        help=(
            "Optional gating threshold for a hybrid route. "
            "If max(top schema score) >= threshold, use graph answer; otherwise use dense answer. "
            "Negative value disables hybrid gating."
        ),
    )
    parser.add_argument(
        "--modalities",
        nargs="*",
        default=[],
        help="Optional modality filter, e.g. --modalities text table image",
    )
    parser.add_argument(
        "--bookrag-query-filtering",
        action="store_true",
        help=(
            "Enable a BookRAG-style query-routing layer for the selected strategy. "
            "The system first uses BookRAG's query classification prompt to label "
            "the question as simple / complex / global. Simple and complex "
            "questions continue to use `--retrieval-strategy`, while global "
            "questions are diverted into a dedicated global retrieval flow. "
            "Disabled by default so existing experiments remain unchanged."
        ),
    )
    parser.add_argument(
        "--planner-base-url",
        default=os.environ.get("MEMGRAPHRAG_PLANNER_BASE_URL", DEFAULT_PLANNER_BASE_URL),
        help="OpenAI-compatible base URL for the BookRAG-style planner model.",
    )
    parser.add_argument(
        "--planner-api-key",
        default=os.environ.get("MEMGRAPHRAG_PLANNER_API_KEY", DEFAULT_PLANNER_API_KEY),
        help="API key for the planner model.",
    )
    parser.add_argument(
        "--planner-model",
        default=os.environ.get("MEMGRAPHRAG_PLANNER_MODEL", DEFAULT_PLANNER_MODEL),
        help="Planner model used for BookRAG-style query planning.",
    )
    parser.add_argument("--planner-max-tokens", type=int, default=1024)
    parser.add_argument("--planner-temperature", type=float, default=0.0)
    parser.add_argument("--preview-chars", type=int, default=240)
    parser.add_argument("--output-path", default="", help="Optional path to save retrieval result JSON.")
    parser.add_argument(
        "--per-doc-output-dir",
        default="",
        help=(
            "Optional directory to save one JSON per processed document. "
            "Each file will include doc name, queries, model answers, gold answers, and correctness."
        ),
    )
    parser.add_argument(
        "--mineru-root",
        default=os.environ.get("MEMGRAPHRAG_MINERU_ROOT", DEFAULT_MINERU_ROOT),
        help="Root directory containing MinerU process_documents subfolders with tree.json files.",
    )
    parser.add_argument(
        "--tree-json-path",
        default="",
        help="Optional explicit tree.json path override for this PDF.",
    )
    parser.add_argument(
        "--compare-answering",
        action="store_true",
        help="For matched samples, compare graph-guided retrieval against a dense-only baseline by generating answers and evaluating them.",
    )
    parser.add_argument(
        "--answer-base-url",
        default=os.environ.get("MEMGRAPHRAG_ANSWER_BASE_URL", DEFAULT_ANSWER_BASE_URL),
        help="OpenAI-compatible base URL for the answer generation model.",
    )
    parser.add_argument(
        "--answer-api-key",
        default=os.environ.get("MEMGRAPHRAG_ANSWER_API_KEY", DEFAULT_ANSWER_API_KEY),
        help="API key for the answer generation model.",
    )
    parser.add_argument(
        "--answer-model",
        default=os.environ.get("MEMGRAPHRAG_ANSWER_MODEL", DEFAULT_ANSWER_MODEL),
        help="Model name used to answer questions from retrieved passages.",
    )
    parser.add_argument("--answer-max-tokens", type=int, default=256)
    parser.add_argument("--answer-temperature", type=float, default=0.0)
    parser.add_argument(
        "--judge-base-url",
        default=os.environ.get("MEMGRAPHRAG_JUDGE_BASE_URL", DEFAULT_JUDGE_BASE_URL),
        help="OpenAI-compatible base URL for the BookRAG-style judge model.",
    )
    parser.add_argument(
        "--judge-api-key",
        default=os.environ.get("MEMGRAPHRAG_JUDGE_API_KEY", DEFAULT_JUDGE_API_KEY),
        help="API key for the judge model.",
    )
    parser.add_argument(
        "--judge-model",
        default=os.environ.get("MEMGRAPHRAG_JUDGE_MODEL", DEFAULT_JUDGE_MODEL),
        help="Judge model used in BookRAG-style MMLong evaluation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.samples_json:
        memory_paths = _discover_memory_paths(args.memory_path)
        if not memory_paths:
            raise ValueError(f"No memory.json files found from {args.memory_path}")

        grouped_samples = _load_samples_grouped_by_doc(args.samples_json)
        doc_jobs: List[Tuple[Path, str, List[Dict[str, Any]]]] = []
        if Path(args.memory_path).resolve().is_file():
            memory_path = memory_paths[0]
            doc_id = args.doc_id or _infer_doc_id_from_memory_path(memory_path)
            matched_samples = grouped_samples.get(doc_id, [])
            if not matched_samples:
                raise ValueError(f"No samples matched doc_id={doc_id} from {args.samples_json}")
            doc_jobs.append((memory_path, doc_id, matched_samples))
        else:
            for memory_path in memory_paths:
                doc_id = _infer_doc_id_from_memory_path(memory_path)
                matched_samples = grouped_samples.get(doc_id, [])
                if not matched_samples:
                    continue
                doc_jobs.append((memory_path, doc_id, matched_samples))
            if not doc_jobs:
                raise ValueError(
                    f"No matched samples found for any memory under {args.memory_path} from {args.samples_json}"
                )

        total_questions = sum(len(samples) for _, _, samples in doc_jobs)
        answer_generator = None
        judge_extractor = None
        planner = None
        all_doc_payloads: List[Dict[str, Any]] = []
        graph_correct_total = 0.0
        dense_correct_total = 0.0
        hybrid_correct_total = 0.0
        selected_strategy_correct_total = 0.0
        graph_f1_total = 0.0
        dense_f1_total = 0.0
        hybrid_f1_total = 0.0
        selected_strategy_f1_total = 0.0
        completed_questions = 0
        retrieval_time_totals = _empty_timing_breakdown()
        planning_time_totals = _empty_timing_breakdown()
        answer_time_totals = _empty_timing_breakdown()
        judge_time_totals = _empty_timing_breakdown()
        answer_token_totals = _empty_token_usage()
        planning_token_totals = _empty_token_usage()
        planning_token_breakdown: Dict[str, Dict[str, int]] = {
            "selected_strategy": _empty_token_usage(),
        }
        answer_token_breakdown: Dict[str, Dict[str, int]] = {
            "graph_guided": _empty_token_usage(),
            "dense_baseline": _empty_token_usage(),
            "selected_strategy": _empty_token_usage(),
        }
        do_selected_strategy_answering = not args.compare_answering
        try:
            if args.compare_answering or do_selected_strategy_answering:
                answer_generator = AnswerGenerator(
                    base_url=args.answer_base_url,
                    api_key=args.answer_api_key,
                    model_name=args.answer_model,
                    max_tokens=args.answer_max_tokens,
                    temperature=args.answer_temperature,
                    prompt_style="legacy_mmlong",
                )
                judge_extractor = BookRAGStyleAnswerExtractor(
                    base_url=args.judge_base_url,
                    api_key=args.judge_api_key,
                    model_name=args.judge_model,
                )
            if args.bookrag_query_filtering:
                planner = BookRAGStyleTaskPlanner(
                    base_url=args.planner_base_url,
                    api_key=args.planner_api_key,
                    model_name=args.planner_model,
                    max_tokens=args.planner_max_tokens,
                    temperature=args.planner_temperature,
                )

            for doc_idx, (memory_path, doc_id, matched_samples) in enumerate(doc_jobs, 1):
                runtime = RetrievalRuntime(
                    memory_path=str(memory_path),
                    schema_vdb_path=args.schema_vdb_path,
                    schema_relation_vdb_path=args.schema_relation_vdb_path,
                    chunk_vdb_path=args.chunk_vdb_path,
                    embedding_model_name=args.embedding_model,
                    mineru_root=args.mineru_root,
                    tree_json_path=args.tree_json_path,
                )
                try:
                    results = []
                    print(
                        f"\n[doc {doc_idx}/{len(doc_jobs)}] {doc_id} | "
                        f"{len(matched_samples)} question(s)"
                    )
                    for item in matched_samples:
                        question = item["question"]
                        graph_retrieval_start = time.perf_counter()
                        graph_result = runtime.retrieve_graph_guided(
                            question,
                            schema_top_k=args.schema_top_k,
                            schema_relation_top_k=args.schema_relation_top_k,
                            passage_top_k=args.passage_top_k,
                            modalities=args.modalities,
                            preview_chars=args.preview_chars,
                            bookrag_query_filtering=False,
                        )
                        graph_retrieval_time = time.perf_counter() - graph_retrieval_start
                        retrieval_time_totals["graph_guided"] += graph_retrieval_time
                        retrieval_time_totals["total"] += graph_retrieval_time

                        if args.compare_answering:
                            dense_retrieval_start = time.perf_counter()
                            dense_result = runtime.retrieve_dense_baseline(
                                question,
                                passage_top_k=args.passage_top_k,
                                modalities=args.modalities,
                                preview_chars=args.preview_chars,
                                bookrag_query_filtering=False,
                            )
                            dense_retrieval_time = time.perf_counter() - dense_retrieval_start
                            retrieval_time_totals["dense_baseline"] += dense_retrieval_time
                            retrieval_time_totals["total"] += dense_retrieval_time
                            graph_answer_start = time.perf_counter()
                            graph_output, graph_answer_usage = answer_generator.answer_with_usage(
                                question,
                                graph_result["top_passages"],
                            )
                            graph_answer_time = time.perf_counter() - graph_answer_start
                            answer_time_totals["graph_guided"] += graph_answer_time
                            answer_time_totals["total"] += graph_answer_time

                            dense_answer_start = time.perf_counter()
                            dense_output, dense_answer_usage = answer_generator.answer_with_usage(
                                question,
                                dense_result["top_passages"],
                            )
                            dense_answer_time = time.perf_counter() - dense_answer_start
                            answer_time_totals["dense_baseline"] += dense_answer_time
                            answer_time_totals["total"] += dense_answer_time
                            _accumulate_token_usage(answer_token_totals, graph_answer_usage)
                            _accumulate_token_usage(answer_token_totals, dense_answer_usage)
                            _accumulate_token_usage(answer_token_breakdown["graph_guided"], graph_answer_usage)
                            _accumulate_token_usage(answer_token_breakdown["dense_baseline"], dense_answer_usage)
                            selected_strategy_name = args.retrieval_strategy
                            selected_strategy_result = None
                            selected_strategy_output = None
                            selected_strategy_retrieval_time = 0.0
                            selected_strategy_answer_usage = _empty_token_usage()
                            selected_strategy_answer_time = 0.0
                            selected_strategy_planning_time = 0.0
                            selected_strategy_planning_usage = _empty_token_usage()
                            if selected_strategy_name == "dense":
                                selected_strategy_result = dense_result
                                selected_strategy_output = dense_output
                            elif selected_strategy_name == "graph":
                                selected_strategy_result = graph_result
                                selected_strategy_output = graph_output
                            else:
                                planned_bundle = run_selected_strategy_with_optional_bookrag_planning(
                                    runtime=runtime,
                                    answer_generator=answer_generator,
                                    planner=planner,
                                    question=question,
                                    strategy=selected_strategy_name,
                                    schema_top_k=args.schema_top_k,
                                    schema_relation_top_k=args.schema_relation_top_k,
                                    passage_top_k=args.passage_top_k,
                                    modalities=args.modalities,
                                    preview_chars=args.preview_chars,
                                    schema_score_threshold=args.graph_plus_dense_schema_score_threshold,
                                    dense_supplement_top_k=args.graph_plus_dense_supplement_top_k,
                                    bookrag_query_filtering=args.bookrag_query_filtering,
                                )
                                selected_strategy_result = planned_bundle["retrieval"]
                                selected_strategy_output = planned_bundle["output"]
                                selected_strategy_retrieval_time = float(planned_bundle["retrieval_time_sec"])
                                selected_strategy_answer_time = float(planned_bundle["answer_time_sec"])
                                selected_strategy_answer_usage = planned_bundle["answer_usage"]
                                selected_strategy_planning_time = float(planned_bundle["planning_time_sec"])
                                selected_strategy_planning_usage = planned_bundle["planning_usage"]
                                retrieval_time_totals["selected_strategy"] += selected_strategy_retrieval_time
                                retrieval_time_totals["total"] += selected_strategy_retrieval_time
                            _accumulate_token_usage(answer_token_totals, selected_strategy_answer_usage)
                            _accumulate_token_usage(
                                answer_token_breakdown["selected_strategy"],
                                selected_strategy_answer_usage,
                            )
                            answer_time_totals["selected_strategy"] += selected_strategy_answer_time
                            answer_time_totals["total"] += selected_strategy_answer_time
                            planning_time_totals["selected_strategy"] += selected_strategy_planning_time
                            planning_time_totals["total"] += selected_strategy_planning_time
                            _accumulate_token_usage(planning_token_totals, selected_strategy_planning_usage)
                            _accumulate_token_usage(
                                planning_token_breakdown["selected_strategy"],
                                selected_strategy_planning_usage,
                            )
                            max_schema_score = _max_top_schema_score(graph_result.get("top_schemas", []))
                            use_hybrid_graph = (
                                args.graph_schema_score_threshold >= 0.0
                                and max_schema_score >= args.graph_schema_score_threshold
                            )
                            hybrid_output = graph_output if use_hybrid_graph else dense_output
                            results.append(
                                {
                                    "doc_id": doc_id,
                                    "question": question,
                                    "answer": item.get("answer"),
                                    "answer_format": item.get("answer_format"),
                                    "graph_guided_retrieval": graph_result,
                                    "dense_baseline_retrieval": dense_result,
                                    "graph_guided_output": graph_output,
                                    "dense_baseline_output": dense_output,
                                    "selected_strategy_name": selected_strategy_name,
                                    "selected_strategy_retrieval": selected_strategy_result,
                                    "selected_strategy_output": selected_strategy_output,
                                    "graph_guided_retrieval_time_sec": graph_retrieval_time,
                                    "dense_baseline_retrieval_time_sec": dense_retrieval_time,
                                    "selected_strategy_retrieval_time_sec": selected_strategy_retrieval_time,
                                    "graph_guided_answer_time_sec": graph_answer_time,
                                    "dense_baseline_answer_time_sec": dense_answer_time,
                                    "selected_strategy_answer_time_sec": selected_strategy_answer_time,
                                    "selected_strategy_planning_time_sec": selected_strategy_planning_time,
                                    "graph_guided_answer_usage": graph_answer_usage,
                                    "dense_baseline_answer_usage": dense_answer_usage,
                                    "selected_strategy_answer_usage": selected_strategy_answer_usage,
                                    "selected_strategy_planning_usage": selected_strategy_planning_usage,
                                    "max_top_schema_score": max_schema_score,
                                    "hybrid_selected_mode": "graph_guided" if use_hybrid_graph else "dense_baseline",
                                    "hybrid_output": hybrid_output,
                                }
                            )
                        elif do_selected_strategy_answering:
                            selected_strategy_name = args.retrieval_strategy
                            selected_strategy_result = None
                            selected_strategy_output = None
                            selected_strategy_retrieval_time = 0.0
                            selected_strategy_answer_usage = _empty_token_usage()
                            selected_strategy_answer_time = 0.0
                            selected_strategy_planning_time = 0.0
                            selected_strategy_planning_usage = _empty_token_usage()
                            if selected_strategy_name == "graph":
                                selected_strategy_result = graph_result
                                selected_strategy_answer_start = time.perf_counter()
                                selected_strategy_output, selected_strategy_answer_usage = answer_generator.answer_with_usage(
                                    question,
                                    selected_strategy_result["top_passages"],
                                )
                                selected_strategy_retrieval_time = graph_retrieval_time
                                selected_strategy_answer_time = time.perf_counter() - selected_strategy_answer_start
                            else:
                                planned_bundle = run_selected_strategy_with_optional_bookrag_planning(
                                    runtime=runtime,
                                    answer_generator=answer_generator,
                                    planner=planner,
                                    question=question,
                                    strategy=selected_strategy_name,
                                    schema_top_k=args.schema_top_k,
                                    schema_relation_top_k=args.schema_relation_top_k,
                                    passage_top_k=args.passage_top_k,
                                    modalities=args.modalities,
                                    preview_chars=args.preview_chars,
                                    schema_score_threshold=args.graph_plus_dense_schema_score_threshold,
                                    dense_supplement_top_k=args.graph_plus_dense_supplement_top_k,
                                    bookrag_query_filtering=args.bookrag_query_filtering,
                                )
                                selected_strategy_result = planned_bundle["retrieval"]
                                selected_strategy_output = planned_bundle["output"]
                                selected_strategy_retrieval_time = float(planned_bundle["retrieval_time_sec"])
                                selected_strategy_answer_time = float(planned_bundle["answer_time_sec"])
                                selected_strategy_answer_usage = planned_bundle["answer_usage"]
                                selected_strategy_planning_time = float(planned_bundle["planning_time_sec"])
                                selected_strategy_planning_usage = planned_bundle["planning_usage"]
                            retrieval_time_totals["selected_strategy"] += selected_strategy_retrieval_time
                            if selected_strategy_name != "graph":
                                retrieval_time_totals["total"] += selected_strategy_retrieval_time
                            answer_time_totals["selected_strategy"] += selected_strategy_answer_time
                            answer_time_totals["total"] += selected_strategy_answer_time
                            planning_time_totals["selected_strategy"] += selected_strategy_planning_time
                            planning_time_totals["total"] += selected_strategy_planning_time
                            _accumulate_token_usage(answer_token_totals, selected_strategy_answer_usage)
                            _accumulate_token_usage(
                                answer_token_breakdown["selected_strategy"],
                                selected_strategy_answer_usage,
                            )
                            _accumulate_token_usage(planning_token_totals, selected_strategy_planning_usage)
                            _accumulate_token_usage(
                                planning_token_breakdown["selected_strategy"],
                                selected_strategy_planning_usage,
                            )
                            results.append(
                                {
                                    "doc_id": doc_id,
                                    "question": question,
                                    "answer": item.get("answer"),
                                    "answer_format": item.get("answer_format"),
                                    "selected_strategy_name": selected_strategy_name,
                                    "selected_strategy_retrieval": selected_strategy_result,
                                    "selected_strategy_output": selected_strategy_output,
                                    "selected_strategy_retrieval_time_sec": selected_strategy_retrieval_time,
                                    "selected_strategy_answer_time_sec": selected_strategy_answer_time,
                                    "selected_strategy_planning_time_sec": selected_strategy_planning_time,
                                    "selected_strategy_answer_usage": selected_strategy_answer_usage,
                                    "selected_strategy_planning_usage": selected_strategy_planning_usage,
                                    "max_top_schema_score": _max_top_schema_score(
                                        (graph_result.get("top_schemas", []) if isinstance(graph_result, dict) else [])
                                    ),
                                }
                            )
                        else:
                            results.append(
                                {
                                    "doc_id": doc_id,
                                    "question": question,
                                    "answer": item.get("answer"),
                                    "answer_format": item.get("answer_format"),
                                    "retrieval": graph_result,
                                    "graph_guided_retrieval_time_sec": graph_retrieval_time,
                                }
                            )

                        completed_questions += 1
                        print(
                            f"\rProgress {_progress_bar(completed_questions, total_questions)} "
                            f"{completed_questions}/{total_questions} questions",
                            end="",
                            flush=True,
                        )

                    doc_payload: Dict[str, Any] = {
                        "doc_id": doc_id,
                        "memory_path": str(memory_path),
                        "num_questions": len(results),
                        "results": results,
                    }
                    if args.compare_answering:
                        graph_records = [
                            {
                                "question": item["question"],
                                "answer": item["answer"],
                                "answer_format": item["answer_format"],
                                "output": item["graph_guided_output"],
                            }
                            for item in results
                        ]
                        dense_records = [
                            {
                                "question": item["question"],
                                "answer": item["answer"],
                                "answer_format": item["answer_format"],
                                "output": item["dense_baseline_output"],
                            }
                            for item in results
                        ]
                        hybrid_records = [
                            {
                                "question": item["question"],
                                "answer": item["answer"],
                                "answer_format": item["answer_format"],
                                "output": item["hybrid_output"],
                            }
                            for item in results
                        ] if args.graph_schema_score_threshold >= 0.0 else []
                        selected_strategy_records = [
                            {
                                "question": item["question"],
                                "answer": item["answer"],
                                "answer_format": item["answer_format"],
                                "output": item["selected_strategy_output"],
                            }
                            for item in results
                        ] if args.retrieval_strategy in {"dense_bucketed", "graph_plus_dense", "graph_plus_dense_no_bucket", "graph_dense_fused", "graph_dense_fused_routed"} else []
                        graph_judge_start = time.perf_counter()
                        graph_evaluated, graph_summary = evaluate_mmlong_results(graph_records, extractor=judge_extractor)
                        graph_judge_time = time.perf_counter() - graph_judge_start
                        judge_time_totals["graph_guided"] += graph_judge_time
                        judge_time_totals["total"] += graph_judge_time

                        dense_judge_start = time.perf_counter()
                        dense_evaluated, dense_summary = evaluate_mmlong_results(dense_records, extractor=judge_extractor)
                        dense_judge_time = time.perf_counter() - dense_judge_start
                        judge_time_totals["dense_baseline"] += dense_judge_time
                        judge_time_totals["total"] += dense_judge_time
                        if args.graph_schema_score_threshold >= 0.0:
                            hybrid_judge_start = time.perf_counter()
                            hybrid_evaluated, hybrid_summary = evaluate_mmlong_results(hybrid_records, extractor=judge_extractor)
                            hybrid_judge_time = time.perf_counter() - hybrid_judge_start
                            judge_time_totals["hybrid"] += hybrid_judge_time
                            judge_time_totals["total"] += hybrid_judge_time
                        else:
                            hybrid_evaluated, hybrid_summary = [], None
                            hybrid_judge_time = 0.0
                        if args.retrieval_strategy in {"dense_bucketed", "graph_plus_dense", "graph_plus_dense_no_bucket", "graph_dense_fused", "graph_dense_fused_routed"}:
                            selected_strategy_judge_start = time.perf_counter()
                            selected_strategy_evaluated, selected_strategy_summary = evaluate_mmlong_results(
                                selected_strategy_records,
                                extractor=judge_extractor,
                            )
                            selected_strategy_judge_time = time.perf_counter() - selected_strategy_judge_start
                            judge_time_totals["selected_strategy"] += selected_strategy_judge_time
                            judge_time_totals["total"] += selected_strategy_judge_time
                        else:
                            selected_strategy_evaluated, selected_strategy_summary = [], None
                            selected_strategy_judge_time = 0.0
                        doc_payload["evaluation"] = {
                            "graph_guided": {
                                "summary": graph_summary,
                                "details": graph_evaluated,
                            },
                            "dense_baseline": {
                                "summary": dense_summary,
                                "details": dense_evaluated,
                            },
                        }
                        if hybrid_summary is not None:
                            doc_payload["evaluation"]["hybrid"] = {
                                "summary": hybrid_summary,
                                "details": hybrid_evaluated,
                            }
                        if selected_strategy_summary is not None:
                            doc_payload["evaluation"]["selected_strategy"] = {
                                "name": args.retrieval_strategy,
                                "summary": selected_strategy_summary,
                                "details": selected_strategy_evaluated,
                            }
                        graph_correct_total += sum(float(item.get("score", 0.0)) for item in graph_evaluated)
                        dense_correct_total += sum(float(item.get("score", 0.0)) for item in dense_evaluated)
                        graph_f1_total += float(graph_summary.get("Avg f1", 0.0)) * len(results)
                        dense_f1_total += float(dense_summary.get("Avg f1", 0.0)) * len(results)
                        if hybrid_summary is not None:
                            hybrid_correct_total += sum(float(item.get("score", 0.0)) for item in hybrid_evaluated)
                            hybrid_f1_total += float(hybrid_summary.get("Avg f1", 0.0)) * len(results)
                        if selected_strategy_summary is not None:
                            selected_strategy_correct_total += sum(float(item.get("score", 0.0)) for item in selected_strategy_evaluated)
                            selected_strategy_f1_total += float(selected_strategy_summary.get("Avg f1", 0.0)) * len(results)
                        graph_acc = graph_correct_total / completed_questions if completed_questions else 0.0
                        dense_acc = dense_correct_total / completed_questions if completed_questions else 0.0
                        hybrid_acc = hybrid_correct_total / completed_questions if completed_questions else 0.0
                        selected_strategy_acc = selected_strategy_correct_total / completed_questions if completed_questions else 0.0
                        graph_f1 = graph_f1_total / completed_questions if completed_questions else 0.0
                        dense_f1 = dense_f1_total / completed_questions if completed_questions else 0.0
                        hybrid_f1 = hybrid_f1_total / completed_questions if completed_questions else 0.0
                        selected_strategy_f1 = selected_strategy_f1_total / completed_questions if completed_questions else 0.0

                        if args.per_doc_output_dir:
                            graph_by_question = {str(item.get("question", "")): item for item in graph_evaluated}
                            dense_by_question = {str(item.get("question", "")): item for item in dense_evaluated}
                            hybrid_by_question = {str(item.get("question", "")): item for item in hybrid_evaluated} if hybrid_summary is not None else {}
                            selected_strategy_by_question = {
                                str(item.get("question", "")): item for item in selected_strategy_evaluated
                            } if selected_strategy_summary is not None else {}
                            per_doc_questions = []
                            for raw_item in results:
                                question = str(raw_item.get("question", ""))
                                graph_eval = graph_by_question.get(question, {})
                                dense_eval = dense_by_question.get(question, {})
                                hybrid_eval = hybrid_by_question.get(question, {})
                                selected_strategy_eval = selected_strategy_by_question.get(question, {})
                                per_doc_questions.append(
                                    {
                                        "doc_id": doc_id,
                                        "question": question,
                                        "gold_answer": raw_item.get("answer"),
                                        "answer_format": raw_item.get("answer_format"),
                                        "max_top_schema_score": raw_item.get("max_top_schema_score"),
                                        "graph_guided_retrieval_time_sec": raw_item.get("graph_guided_retrieval_time_sec", 0.0),
                                        "dense_baseline_retrieval_time_sec": raw_item.get("dense_baseline_retrieval_time_sec", 0.0),
                                        "selected_strategy_retrieval_time_sec": raw_item.get("selected_strategy_retrieval_time_sec", 0.0),
                                        "graph_guided_answer_time_sec": raw_item.get("graph_guided_answer_time_sec", 0.0),
                                        "dense_baseline_answer_time_sec": raw_item.get("dense_baseline_answer_time_sec", 0.0),
                                        "selected_strategy_answer_time_sec": raw_item.get("selected_strategy_answer_time_sec", 0.0),
                                        "graph_guided": {
                                            "output": raw_item.get("graph_guided_output"),
                                            "answer_usage": raw_item.get("graph_guided_answer_usage", _empty_token_usage()),
                                            "extracted_result": graph_eval.get("extracted_res"),
                                            "pred": graph_eval.get("pred"),
                                            "pred_format": graph_eval.get("pred_format"),
                                            "match_result": bool(float(graph_eval.get("score", 0.0)) >= 1.0),
                                            "correct": bool(float(graph_eval.get("score", 0.0)) >= 1.0),
                                            "score": graph_eval.get("score", 0.0),
                                            "f1_token": graph_eval.get("f1_token", 0.0),
                                            "llm_score": graph_eval.get("llm_score", 0.0),
                                        },
                                        "dense_baseline": {
                                            "output": raw_item.get("dense_baseline_output"),
                                            "answer_usage": raw_item.get("dense_baseline_answer_usage", _empty_token_usage()),
                                            "extracted_result": dense_eval.get("extracted_res"),
                                            "pred": dense_eval.get("pred"),
                                            "pred_format": dense_eval.get("pred_format"),
                                            "match_result": bool(float(dense_eval.get("score", 0.0)) >= 1.0),
                                            "correct": bool(float(dense_eval.get("score", 0.0)) >= 1.0),
                                            "score": dense_eval.get("score", 0.0),
                                            "f1_token": dense_eval.get("f1_token", 0.0),
                                            "llm_score": dense_eval.get("llm_score", 0.0),
                                        },
                                    }
                                )
                                if hybrid_summary is not None:
                                    per_doc_questions[-1]["hybrid"] = {
                                        "selected_mode": raw_item.get("hybrid_selected_mode"),
                                        "output": raw_item.get("hybrid_output"),
                                        "extracted_result": hybrid_eval.get("extracted_res"),
                                        "pred": hybrid_eval.get("pred"),
                                        "pred_format": hybrid_eval.get("pred_format"),
                                        "match_result": bool(float(hybrid_eval.get("score", 0.0)) >= 1.0),
                                        "correct": bool(float(hybrid_eval.get("score", 0.0)) >= 1.0),
                                        "score": hybrid_eval.get("score", 0.0),
                                        "f1_token": hybrid_eval.get("f1_token", 0.0),
                                        "llm_score": hybrid_eval.get("llm_score", 0.0),
                                    }
                                if selected_strategy_summary is not None:
                                    per_doc_questions[-1]["selected_strategy"] = {
                                        "name": args.retrieval_strategy,
                                        "output": raw_item.get("selected_strategy_output"),
                                        "selected_mode": (raw_item.get("selected_strategy_retrieval") or {}).get("selected_mode"),
                                        "answer_usage": raw_item.get("selected_strategy_answer_usage", _empty_token_usage()),
                                        "extracted_result": selected_strategy_eval.get("extracted_res"),
                                        "pred": selected_strategy_eval.get("pred"),
                                        "pred_format": selected_strategy_eval.get("pred_format"),
                                        "match_result": bool(float(selected_strategy_eval.get("score", 0.0)) >= 1.0),
                                        "correct": bool(float(selected_strategy_eval.get("score", 0.0)) >= 1.0),
                                        "score": selected_strategy_eval.get("score", 0.0),
                                        "f1_token": selected_strategy_eval.get("f1_token", 0.0),
                                        "llm_score": selected_strategy_eval.get("llm_score", 0.0),
                                    }

                            per_doc_payload = {
                                "doc_id": doc_id,
                                "memory_path": str(memory_path),
                                "num_questions": len(per_doc_questions),
                                "graph_summary": graph_summary,
                                "dense_summary": dense_summary,
                                "runtime_stats": {
                                    "planning_time_sec": {
                                        "selected_strategy": sum(
                                            float(item.get("selected_strategy_planning_time_sec", 0.0))
                                            for item in results
                                        ),
                                        "total": sum(
                                            float(item.get("selected_strategy_planning_time_sec", 0.0))
                                            for item in results
                                        ),
                                    },
                                    "judge_time_sec": {
                                        "graph_guided": graph_judge_time,
                                        "dense_baseline": dense_judge_time,
                                        "hybrid": hybrid_judge_time,
                                        "selected_strategy": selected_strategy_judge_time,
                                        "total": graph_judge_time + dense_judge_time + hybrid_judge_time + selected_strategy_judge_time,
                                    }
                                },
                                "questions": per_doc_questions,
                            }
                            if hybrid_summary is not None:
                                per_doc_payload["hybrid_summary"] = hybrid_summary
                            if selected_strategy_summary is not None:
                                per_doc_payload["selected_strategy_summary"] = {
                                    "name": args.retrieval_strategy,
                                    "summary": selected_strategy_summary,
                                }
                            per_doc_output_dir = Path(args.per_doc_output_dir).resolve()
                            per_doc_output_dir.mkdir(parents=True, exist_ok=True)
                            per_doc_output_path = per_doc_output_dir / f"{_safe_output_name(doc_id)}.json"
                            with open(per_doc_output_path, "w", encoding="utf-8") as f:
                                json.dump(per_doc_payload, f, ensure_ascii=False, indent=2)

                        message = (
                            f"\n[doc {doc_idx}/{len(doc_jobs)} done] "
                            f"cumulative graph acc: {graph_correct_total:.0f}/{completed_questions} ({graph_acc:.4f}), "
                            f"f1: {graph_f1:.4f} | "
                            f"dense acc: {dense_correct_total:.0f}/{completed_questions} ({dense_acc:.4f}), "
                            f"f1: {dense_f1:.4f}"
                        )
                        if hybrid_summary is not None:
                            message += (
                                f" | hybrid acc: {hybrid_correct_total:.0f}/{completed_questions} ({hybrid_acc:.4f}), "
                                f"f1: {hybrid_f1:.4f}"
                            )
                        if selected_strategy_summary is not None:
                            message += (
                                f" | {args.retrieval_strategy} acc: "
                                f"{selected_strategy_correct_total:.0f}/{completed_questions} ({selected_strategy_acc:.4f}), "
                                f"f1: {selected_strategy_f1:.4f}"
                            )
                        print(message)
                    elif do_selected_strategy_answering:
                        selected_strategy_records = [
                            {
                                "question": item["question"],
                                "answer": item["answer"],
                                "answer_format": item["answer_format"],
                                "output": item["selected_strategy_output"],
                            }
                            for item in results
                        ]
                        selected_strategy_judge_start = time.perf_counter()
                        selected_strategy_evaluated, selected_strategy_summary = evaluate_mmlong_results(
                            selected_strategy_records,
                            extractor=judge_extractor,
                        )
                        selected_strategy_judge_time = time.perf_counter() - selected_strategy_judge_start
                        judge_time_totals["selected_strategy"] += selected_strategy_judge_time
                        judge_time_totals["total"] += selected_strategy_judge_time
                        doc_payload["evaluation"] = {
                            "selected_strategy": {
                                "name": args.retrieval_strategy,
                                "summary": selected_strategy_summary,
                                "details": selected_strategy_evaluated,
                            }
                        }
                        selected_strategy_correct_total += sum(
                            float(item.get("score", 0.0)) for item in selected_strategy_evaluated
                        )
                        selected_strategy_f1_total += float(selected_strategy_summary.get("Avg f1", 0.0)) * len(results)
                        selected_strategy_acc = (
                            selected_strategy_correct_total / completed_questions if completed_questions else 0.0
                        )
                        selected_strategy_f1 = (
                            selected_strategy_f1_total / completed_questions if completed_questions else 0.0
                        )

                        if args.per_doc_output_dir:
                            selected_strategy_by_question = {
                                str(item.get("question", "")): item for item in selected_strategy_evaluated
                            }
                            per_doc_questions = []
                            for raw_item in results:
                                question = str(raw_item.get("question", ""))
                                selected_strategy_eval = selected_strategy_by_question.get(question, {})
                                per_doc_questions.append(
                                    {
                                        "doc_id": doc_id,
                                        "question": question,
                                        "gold_answer": raw_item.get("answer"),
                                        "answer_format": raw_item.get("answer_format"),
                                        "max_top_schema_score": raw_item.get("max_top_schema_score"),
                                        "selected_strategy_retrieval_time_sec": raw_item.get("selected_strategy_retrieval_time_sec", 0.0),
                                        "selected_strategy_answer_time_sec": raw_item.get("selected_strategy_answer_time_sec", 0.0),
                                        "selected_strategy_planning_time_sec": raw_item.get("selected_strategy_planning_time_sec", 0.0),
                                        "selected_strategy": {
                                            "name": args.retrieval_strategy,
                                            "output": raw_item.get("selected_strategy_output"),
                                            "selected_mode": (raw_item.get("selected_strategy_retrieval") or {}).get("selected_mode"),
                                            "answer_usage": raw_item.get("selected_strategy_answer_usage", _empty_token_usage()),
                                            "planning_usage": raw_item.get("selected_strategy_planning_usage", _empty_token_usage()),
                                            "extracted_result": selected_strategy_eval.get("extracted_res"),
                                            "pred": selected_strategy_eval.get("pred"),
                                            "pred_format": selected_strategy_eval.get("pred_format"),
                                            "match_result": bool(float(selected_strategy_eval.get("score", 0.0)) >= 1.0),
                                            "correct": bool(float(selected_strategy_eval.get("score", 0.0)) >= 1.0),
                                            "score": selected_strategy_eval.get("score", 0.0),
                                            "f1_token": selected_strategy_eval.get("f1_token", 0.0),
                                            "llm_score": selected_strategy_eval.get("llm_score", 0.0),
                                        },
                                    }
                                )
                            per_doc_payload = {
                                "doc_id": doc_id,
                                "memory_path": str(memory_path),
                                "num_questions": len(per_doc_questions),
                                "runtime_stats": {
                                    "planning_time_sec": {
                                        "selected_strategy": sum(
                                            float(item.get("selected_strategy_planning_time_sec", 0.0))
                                            for item in results
                                        ),
                                        "total": sum(
                                            float(item.get("selected_strategy_planning_time_sec", 0.0))
                                            for item in results
                                        ),
                                    },
                                    "judge_time_sec": {
                                        "selected_strategy": selected_strategy_judge_time,
                                        "total": selected_strategy_judge_time,
                                    }
                                },
                                "selected_strategy_summary": {
                                    "name": args.retrieval_strategy,
                                    "summary": selected_strategy_summary,
                                },
                                "questions": per_doc_questions,
                            }
                            per_doc_output_dir = Path(args.per_doc_output_dir).resolve()
                            per_doc_output_dir.mkdir(parents=True, exist_ok=True)
                            per_doc_output_path = per_doc_output_dir / f"{_safe_output_name(doc_id)}.json"
                            with open(per_doc_output_path, "w", encoding="utf-8") as f:
                                json.dump(per_doc_payload, f, ensure_ascii=False, indent=2)

                        print(
                            f"\n[doc {doc_idx}/{len(doc_jobs)} done] "
                            f"cumulative {args.retrieval_strategy} acc: "
                            f"{selected_strategy_correct_total:.0f}/{completed_questions} ({selected_strategy_acc:.4f}), "
                            f"f1: {selected_strategy_f1:.4f}"
                        )
                    else:
                        print()
                    all_doc_payloads.append(doc_payload)
                finally:
                    runtime.close()
        finally:
            if answer_generator is not None:
                answer_generator.close()
            if planner is not None:
                planner.close()

        payload: Dict[str, Any] = {
            "num_docs": len(all_doc_payloads),
            "num_questions": total_questions,
            "documents": all_doc_payloads,
            "runtime_stats": {
                "retrieval_time_sec": retrieval_time_totals,
                "planning_time_sec": planning_time_totals,
                "answer_time_sec": answer_time_totals,
                "judge_time_sec": judge_time_totals,
                "planning_token_usage": {
                    "total": planning_token_totals,
                    "breakdown": planning_token_breakdown,
                },
                "answer_token_usage": {
                    "total": answer_token_totals,
                    "breakdown": answer_token_breakdown,
                },
            },
        }
        if args.compare_answering:
            payload["overall_accuracy"] = {
                "graph_guided": {
                    "correct": graph_correct_total,
                    "total": completed_questions,
                    "accuracy": (graph_correct_total / completed_questions) if completed_questions else 0.0,
                },
                "dense_baseline": {
                    "correct": dense_correct_total,
                    "total": completed_questions,
                    "accuracy": (dense_correct_total / completed_questions) if completed_questions else 0.0,
                },
            }
            if args.graph_schema_score_threshold >= 0.0:
                payload["overall_accuracy"]["hybrid"] = {
                    "correct": hybrid_correct_total,
                    "total": completed_questions,
                    "accuracy": (hybrid_correct_total / completed_questions) if completed_questions else 0.0,
                    "schema_score_threshold": args.graph_schema_score_threshold,
                }
            if args.retrieval_strategy in {"dense_bucketed", "graph_plus_dense", "graph_plus_dense_no_bucket", "graph_dense_fused", "graph_dense_fused_routed"}:
                payload["overall_accuracy"]["selected_strategy"] = {
                    "name": args.retrieval_strategy,
                    "correct": selected_strategy_correct_total,
                    "total": completed_questions,
                    "accuracy": (selected_strategy_correct_total / completed_questions) if completed_questions else 0.0,
                }
                if args.retrieval_strategy in {"graph_plus_dense", "graph_plus_dense_no_bucket", "graph_dense_fused", "graph_dense_fused_routed"}:
                    payload["overall_accuracy"]["selected_strategy"]["schema_score_threshold"] = args.graph_plus_dense_schema_score_threshold
                    payload["overall_accuracy"]["selected_strategy"]["dense_supplement_top_k"] = args.graph_plus_dense_supplement_top_k
        elif do_selected_strategy_answering:
            payload["overall_accuracy"] = {
                "selected_strategy": {
                    "name": args.retrieval_strategy,
                    "correct": selected_strategy_correct_total,
                    "total": completed_questions,
                    "accuracy": (selected_strategy_correct_total / completed_questions) if completed_questions else 0.0,
                    "f1": (selected_strategy_f1_total / completed_questions) if completed_questions else 0.0,
                }
            }
        if len(all_doc_payloads) == 1:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            print(
                f"\nFinished {len(all_doc_payloads)} document(s), "
                f"{completed_questions}/{total_questions} question(s)."
            )
            if args.compare_answering:
                graph_summary = payload["overall_accuracy"]["graph_guided"]
                dense_summary = payload["overall_accuracy"]["dense_baseline"]
                summary_message = (
                    "Overall accuracy | "
                    f"graph: {graph_summary['correct']:.0f}/{graph_summary['total']} "
                    f"({graph_summary['accuracy']:.4f}), f1: {(graph_f1_total / completed_questions) if completed_questions else 0.0:.4f} | "
                    f"dense: {dense_summary['correct']:.0f}/{dense_summary['total']} "
                    f"({dense_summary['accuracy']:.4f}), f1: {(dense_f1_total / completed_questions) if completed_questions else 0.0:.4f}"
                )
                if args.graph_schema_score_threshold >= 0.0:
                    hybrid_summary = payload["overall_accuracy"]["hybrid"]
                    summary_message += (
                        f" | hybrid: {hybrid_summary['correct']:.0f}/{hybrid_summary['total']} "
                        f"({hybrid_summary['accuracy']:.4f}), "
                        f"f1: {(hybrid_f1_total / completed_questions) if completed_questions else 0.0:.4f} "
                        f"@ threshold={hybrid_summary['schema_score_threshold']}"
                    )
                if args.retrieval_strategy in {"dense_bucketed", "graph_plus_dense", "graph_plus_dense_no_bucket", "graph_dense_fused", "graph_dense_fused_routed"}:
                    selected_summary = payload["overall_accuracy"]["selected_strategy"]
                    summary_message += (
                        f" | {selected_summary['name']}: "
                        f"{selected_summary['correct']:.0f}/{selected_summary['total']} "
                        f"({selected_summary['accuracy']:.4f}), "
                        f"f1: {(selected_strategy_f1_total / completed_questions) if completed_questions else 0.0:.4f}"
                    )
                    if args.retrieval_strategy in {"graph_plus_dense", "graph_plus_dense_no_bucket", "graph_dense_fused", "graph_dense_fused_routed"}:
                        summary_message += (
                            f" @ threshold={selected_summary['schema_score_threshold']}, "
                            f"dense+={selected_summary['dense_supplement_top_k']}"
                        )
                print(summary_message)
                print(
                    "Runtime stats | "
                    f"retrieval total: {retrieval_time_totals['total']:.2f}s "
                    f"(graph={retrieval_time_totals['graph_guided']:.2f}s, "
                    f"dense={retrieval_time_totals['dense_baseline']:.2f}s, "
                    f"selected={retrieval_time_totals['selected_strategy']:.2f}s) | "
                    f"planning total: {planning_time_totals['total']:.2f}s | "
                    f"answer total: {answer_time_totals['total']:.2f}s | "
                    f"judge total: {judge_time_totals['total']:.2f}s | "
                    f"planning tokens: {planning_token_totals['total_tokens']} total | "
                    f"answer tokens: {answer_token_totals['total_tokens']} total "
                    f"(prompt={answer_token_totals['prompt_tokens']}, "
                    f"completion={answer_token_totals['completion_tokens']})"
                )
            elif do_selected_strategy_answering:
                selected_summary = payload["overall_accuracy"]["selected_strategy"]
                print(
                    "Overall accuracy | "
                    f"{selected_summary['name']}: "
                    f"{selected_summary['correct']:.0f}/{selected_summary['total']} "
                    f"({selected_summary['accuracy']:.4f}), "
                    f"f1: {selected_summary['f1']:.4f}"
                )
                print(
                    "Runtime stats | "
                    f"retrieval total: {retrieval_time_totals['total']:.2f}s "
                    f"(graph={retrieval_time_totals['graph_guided']:.2f}s, "
                    f"selected={retrieval_time_totals['selected_strategy']:.2f}s) | "
                    f"planning total: {planning_time_totals['total']:.2f}s | "
                    f"answer total: {answer_time_totals['total']:.2f}s | "
                    f"judge total: {judge_time_totals['total']:.2f}s | "
                    f"planning tokens: {planning_token_totals['total_tokens']} total | "
                    f"answer tokens: {answer_token_totals['total_tokens']} total "
                    f"(prompt={answer_token_totals['prompt_tokens']}, "
                    f"completion={answer_token_totals['completion_tokens']})"
                )

        if args.output_path:
            output_path = Path(args.output_path).resolve()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            print(f"\nSaved retrieval result to: {output_path}")
        return

    if not args.query:
        raise ValueError("Please provide --query, or use --samples-json for batch retrieval.")

    result = build_retrieval_result(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))

    if args.output_path:
        output_path = Path(args.output_path).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"\nSaved retrieval result to: {output_path}")


if __name__ == "__main__":
    main()
