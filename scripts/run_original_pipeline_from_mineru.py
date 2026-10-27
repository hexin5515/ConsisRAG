#!/usr/bin/env python3


from __future__ import annotations

import argparse
import ast
import base64
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import html
import json
import mimetypes
import os
import re
import sqlite3
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DEFAULT_LLM_MODEL = "Qwen/Qwen3-8B-AWQ"
DEFAULT_LLM_BASE_URL = "http://localhost:8080/v1"
DEFAULT_ENTITY_LLM_MODEL = DEFAULT_LLM_MODEL
DEFAULT_ENTITY_LLM_BASE_URL = DEFAULT_LLM_BASE_URL
DEFAULT_VLM_MODEL = "Qwen/Qwen2-5-VL"
DEFAULT_VLM_BASE_URL = "http://localhost:8081/v1"
DEFAULT_EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"
DEFAULT_EMBEDDING_BASE_URL = "http://localhost:8083/v1/embeddings"
DEFAULT_EMBEDDING_API_KEY = "openai"
SKIP_TREE_DIR_NAMES = {
    "9b1b9703-915c-55ea-9631-6923bd75ad3c",
}


def discover_mineru_json_files(input_path: str) -> List[Path]:


    path = Path(input_path)
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(f"MinerU input path does not exist: {input_path}")
    tree_files = sorted(p for p in path.rglob("tree.json") if p.is_file())
    if not tree_files:
        raise FileNotFoundError(
            f"No recursive 'tree.json' files found under MinerU directory: {input_path}"
        )
    return tree_files


def load_mineru_chunks(
    input_path: str,
    *,
    include_title_chunks: bool = False,
    include_image_chunks: bool = True,
    include_empty_image_chunks: bool = False,
) -> List[Any]:


    from utils.mineru_loader import load_mineru_json_chunks

    chunks: List[Any] = []
    for json_path in discover_mineru_json_files(input_path):
        try:
            chunks.extend(
                load_mineru_json_chunks(
                    str(json_path),
                    include_title_chunks=include_title_chunks,
                    include_image_chunks=include_image_chunks,
                    include_empty_image_chunks=include_empty_image_chunks,
                )
            )
        except Exception as exc:
            print(f"Warning: skip {json_path}: {exc}")
    return chunks


def _load_one_mineru_tree(
    json_path: Path,
    *,
    include_title_chunks: bool = False,
    include_image_chunks: bool = True,
    include_empty_image_chunks: bool = False,
) -> List[Any]:


    from utils.mineru_loader import load_mineru_json_chunks

    return load_mineru_json_chunks(
        str(json_path),
        include_title_chunks=include_title_chunks,
        include_image_chunks=include_image_chunks,
        include_empty_image_chunks=include_empty_image_chunks,
    )


def group_chunks_by_pdf(chunks: Iterable[Any]) -> Dict[str, List[Any]]:


    grouped: Dict[str, List[Any]] = {}
    for chunk in chunks:
        source_file = chunk.metadata.get("source_file") or "unknown_pdf"
        grouped.setdefault(str(source_file), []).append(chunk)
    return grouped


def safe_stem(name: str) -> str:

    stem = Path(name).stem or "unknown_pdf"
    return "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in stem)


def _schema_relation_text(schema_relation: Tuple[str, str, str]) -> str:

    head_type, relation, tail_type = schema_relation
    return f"{head_type} {relation} {tail_type}"


def _normalize_section_text(text: str) -> str:

    return " ".join(str(text).strip().lower().split())


def is_reference_chunk(chunk: Any) -> bool:


    metadata = dict(getattr(chunk, "metadata", {}) or {})
    title_path = metadata.get("title_path") or []
    section_titles = [_normalize_section_text(item) for item in title_path if item]

    current_node = metadata.get("current_node") or {}
    current_content = _normalize_section_text(current_node.get("content", ""))
    current_block_type = str(current_node.get("block_type", "")).lower()
    if current_block_type == "title" and current_content:
        section_titles.append(current_content)

    reference_markers = {"references", "reference", "bibliography"}
    return any(title in reference_markers for title in section_titles)


def filter_reference_chunks(chunks: List[Any]) -> List[Any]:

    return [chunk for chunk in chunks if not is_reference_chunk(chunk)]


def _entity_system_prompt(*, include_image_summary: bool = False) -> str:

    prompt = (
        "You are an expert entity extraction assistant. "
        "Follow the task instructions exactly, but your final output MUST be a single valid JSON object "
        'with root key "entities". Each entity must contain keys "entity_name", "entity_type", and "description". '
        "Do not return any extra explanation."
    )
    if include_image_summary:
        prompt += (
            ' Include an additional root key "image_summary" containing a grounded '
            "summary of the image's visible content, relationships, labels, values, and units."
        )
    return prompt


def _entity_messages_from_prompt(
    prompt_body: str,
    *,
    include_image_summary: bool = False,
) -> List[Dict[str, Any]]:

    return [
        {"role": "system", "content": _entity_system_prompt(include_image_summary=include_image_summary)},
        {"role": "user", "content": prompt_body},
    ]


def _encode_image_as_data_url(image_path: str) -> str | None:

    path = Path(str(image_path or ""))
    if not path.exists() or not path.is_file():
        return None

    mime_type, _ = mimetypes.guess_type(str(path))
    mime_type = mime_type or "image/jpeg"
    try:
        with path.open("rb") as f:
            encoded = base64.b64encode(f.read()).decode("utf-8")
        return f"data:{mime_type};base64,{encoded}"
    except Exception:
        return None


def _image_entity_messages(prompt_body: str, image_path: str | None) -> List[Dict[str, Any]]:


    data_url = _encode_image_as_data_url(image_path or "")
    if not data_url:
        return _entity_messages_from_prompt(prompt_body, include_image_summary=True)

    return [
        {"role": "system", "content": _entity_system_prompt(include_image_summary=True)},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt_body},
                {"type": "image_url", "image_url": {"url": data_url}},
            ],
        },
    ]


def _strip_section_prefix(text: str) -> str:

    stripped = str(text or "").strip()
    if not stripped.startswith("Section:"):
        return stripped
    parts = stripped.split("\n\n", 1)
    return parts[1].strip() if len(parts) == 2 else stripped


def _table_description_text(metadata: Dict[str, Any], chunk_context: str) -> str:

    parts = [
        str(metadata.get("caption") or "").strip(),
        str(metadata.get("footnote") or "").strip(),
    ]
    description = " ".join(part for part in parts if part).strip()
    return description or _strip_section_prefix(chunk_context)


def _table_body_html(metadata: Dict[str, Any]) -> str:

    current_node = metadata.get("current_node") or {}
    return str(metadata.get("table_body") or current_node.get("table_body") or "").strip()


def parse_html_table_to_grid(html_string: str) -> List[List[str]]:

    from bs4 import BeautifulSoup

    if not html_string:
        return []

    soup = BeautifulSoup(html_string, "html.parser")
    rows = soup.find_all("tr")
    if not rows:
        return []

    grid: List[List[str]] = []
    for row_idx, row_elem in enumerate(rows):
        while len(grid) <= row_idx:
            grid.append([])

        cells = row_elem.find_all(["td", "th"])
        for cell_elem in cells:
            col_idx = 0
            while len(grid[row_idx]) > col_idx and grid[row_idx][col_idx] is not None:
                col_idx += 1

            colspan = int(cell_elem.get("colspan", 1))
            rowspan = int(cell_elem.get("rowspan", 1))
            cell_text = html.unescape(cell_elem.get_text(strip=True))

            for i in range(rowspan):
                for j in range(colspan):
                    target_row = row_idx + i
                    target_col = col_idx + j
                    while len(grid) <= target_row:
                        grid.append([])
                    while len(grid[target_row]) <= target_col:
                        grid[target_row].append(None)
                    if grid[target_row][target_col] is None:
                        grid[target_row][target_col] = cell_text

    return grid


def is_numeric(value: str) -> bool:

    if not isinstance(value, str):
        return False
    text = value.strip().replace("%", "").replace("(", "").replace(")", "").replace(",", "")
    if not text:
        return False
    try:
        float(text)
        return True
    except (TypeError, ValueError):
        return False


def identify_header_rows(grid: List[List[str]]) -> int:

    if not grid or len(grid) < 2:
        return 0 if not grid else 1

    header_row_count = 0
    max_cols = max(len(row) for row in grid)
    check_depth = min(3, len(grid) - 1)

    for i in range(check_depth):
        current_row = grid[i]
        next_row = grid[i + 1]

        has_horizontal_span = False
        for j in range(len(current_row) - 1):
            if current_row[j] and current_row[j] == current_row[j + 1]:
                has_horizontal_span = True
                break

        type_mismatch_count = 0
        text_to_numeric_count = 0
        for j in range(min(len(current_row), len(next_row))):
            current_numeric = is_numeric(current_row[j])
            next_numeric = is_numeric(next_row[j])
            if current_numeric != next_numeric:
                type_mismatch_count += 1
                if not current_numeric and next_numeric:
                    text_to_numeric_count += 1

        is_strong_header_signal = text_to_numeric_count > (max_cols / 3)
        is_structural_header = has_horizontal_span
        is_type_mismatch_header = type_mismatch_count > (max_cols / 2)

        if is_strong_header_signal or is_structural_header or is_type_mismatch_header:
            header_row_count += 1
            continue

        if i == 0:
            current_all_text = all(not is_numeric(cell) for cell in current_row if cell)
            next_all_text = all(not is_numeric(cell) for cell in next_row if cell)
            if current_all_text and next_all_text:
                dissimilar_cells = 0
                for j in range(min(len(current_row), len(next_row))):
                    if current_row[j] != next_row[j]:
                        dissimilar_cells += 1
                if dissimilar_cells > max_cols / 2:
                    header_row_count += 1
                    continue

        break

    return header_row_count


def create_hierarchical_headers(grid: List[List[str]], num_header_rows: int) -> List[str]:

    column_headers = []
    if num_header_rows > 0 and grid and len(grid) >= num_header_rows:
        header_grid = grid[:num_header_rows]
        num_cols = len(header_grid[0]) if header_grid else 0
        for col_idx in range(num_cols):
            path_parts = [str(header_grid[row_idx][col_idx] or "").strip() for row_idx in range(num_header_rows)]
            unique_ordered_parts = list(dict.fromkeys(part for part in path_parts if part))
            if unique_ordered_parts:
                column_headers.append(" > ".join(unique_ordered_parts))
    return sorted(set(column_headers))


def _count_tokens(text: str) -> int:

    try:
        from utils.llm_utils import num_tokens_by_tiktoken

        return int(num_tokens_by_tiktoken(text or ""))
    except Exception:
        return max(1, len(str(text or "").split()))


def _split_table_row_batches(
    grid: List[List[str]],
    num_header_rows: int,
    *,
    max_tokens_for_rows: int,
) -> List[List[str]]:

    body_rows = grid[num_header_rows:]
    row_strings = [" | ".join((cell or "").strip() for cell in row) for row in body_rows]

    batches: List[List[str]] = []
    current_batch: List[str] = []
    current_tokens = 0

    for row_str in row_strings:
        row_tokens = _count_tokens(row_str)
        if row_tokens > max_tokens_for_rows:
            if current_batch:
                batches.append(current_batch)
            batches.append([row_str])
            current_batch = []
            current_tokens = 0
            continue

        if current_batch and current_tokens + row_tokens > max_tokens_for_rows:
            batches.append(current_batch)
            current_batch = [row_str]
            current_tokens = row_tokens
        else:
            current_batch.append(row_str)
            current_tokens += row_tokens

    if current_batch:
        batches.append(current_batch)

    return batches


def _build_table_entity_extraction_message_batches(chunk: Any) -> List[List[Dict[str, str]]]:


    from prompts.kg_prompt import (
        DEFAULT_ENTITY_TYPES,
        TABLE_BODY_EXTRACTION,
        TABLE_DESCRIPTION_EXTRACTION,
    )

    metadata = dict(chunk.metadata)
    chunk_context = str(chunk.content)
    description = _table_description_text(metadata, chunk_context)
    table_body = _table_body_html(metadata)

    if not table_body:
        return []

    grid = parse_html_table_to_grid(table_body)
    if not grid:
        return []

    entity_types = ",".join(DEFAULT_ENTITY_TYPES)
    num_header_rows = identify_header_rows(grid)
    column_headers = create_hierarchical_headers(grid, num_header_rows)

    desc_input_json = json.dumps(
        {
            "description": description,
            "column_headers": column_headers,
        },
        ensure_ascii=False,
        indent=2,
    )
    prompts: List[List[Dict[str, str]]] = [
        _entity_messages_from_prompt(
            TABLE_DESCRIPTION_EXTRACTION.format(
                entity_types=entity_types,
                input_json=desc_input_json,
            )
        )
    ]

    prompt_overhead = _count_tokens(description) + _count_tokens(json.dumps(column_headers, ensure_ascii=False))
    max_tokens_for_rows = max(400, 3200 - prompt_overhead - _count_tokens(TABLE_BODY_EXTRACTION))
    row_batches = _split_table_row_batches(
        grid,
        num_header_rows,
        max_tokens_for_rows=max_tokens_for_rows,
    )

    for batch in row_batches:
        body_input_json = json.dumps(
            {
                "description": description,
                "column_headers": column_headers,
                "rows_batch": batch,
            },
            ensure_ascii=False,
            indent=2,
        )
        prompts.append(
            _entity_messages_from_prompt(
                TABLE_BODY_EXTRACTION.format(
                    entity_types=entity_types,
                    input_json=body_input_json,
                )
            )
        )

    return prompts


def _build_entity_extraction_messages(chunk: Any) -> List[Dict[str, Any]]:


    from prompts.kg_prompt import (
        DEFAULT_ENTITY_TYPES,
        DEFAULT_TUPLE_DELIMITER,
        DEFAULT_RECORD_DELIMITER,
        DEFAULT_COMPLETION_DELIMITER,
        ENTITY_EXTRACTION,
        EQUATION_ENTITY_EXTRACTION,
        IMAGE_ENTITY_EXTRACTION,
        SECTION_ENTITY_EXTRACTION,
    )

    metadata = dict(chunk.metadata)
    block_type = str(metadata.get("block_type", "text")).lower()
    entity_types = ", ".join(DEFAULT_ENTITY_TYPES)
    chunk_context = str(chunk.content)
    current_node = metadata.get("current_node") or {}
    section_path = metadata.get("title_path", [])

    if block_type == "table":
        table_batches = _build_table_entity_extraction_message_batches(chunk)
        if table_batches:
            return table_batches[0]
        prompt_body = ENTITY_EXTRACTION.format(
            entity_types=entity_types,
            tuple_delimiter=DEFAULT_TUPLE_DELIMITER,
            record_delimiter=DEFAULT_RECORD_DELIMITER,
            completion_delimiter=DEFAULT_COMPLETION_DELIMITER,
            input_text=chunk_context,
        )
    elif block_type == "image":
        prompt_body = IMAGE_ENTITY_EXTRACTION.format(
            entity_types=entity_types,
            image_description=json.dumps(
                {
                    "description": chunk_context,
                },
                ensure_ascii=False,
            ),
        )
        return _image_entity_messages(prompt_body, getattr(chunk, "image_path", None))
    elif block_type == "equation":
        equation_text = str(current_node.get("content") or "").strip() or chunk_context
        prompt_body = EQUATION_ENTITY_EXTRACTION.format(
            formula_latex_code=equation_text,
        )
    elif block_type == "title":
        prompt_body = SECTION_ENTITY_EXTRACTION.format(
            entity_types=entity_types,
            input_json=json.dumps(
                {
                    "context": {
                        "current_section_path": section_path,
                    },
                    "title_to_process": chunk_context,
                },
                ensure_ascii=False,
            ),
        )
    else:
        prompt_body = ENTITY_EXTRACTION.format(
            entity_types=entity_types,
            tuple_delimiter=DEFAULT_TUPLE_DELIMITER,
            record_delimiter=DEFAULT_RECORD_DELIMITER,
            completion_delimiter=DEFAULT_COMPLETION_DELIMITER,
            input_text=chunk_context,
        )

    return _entity_messages_from_prompt(prompt_body)


def _build_image_text_only_entity_messages(chunk: Any) -> List[Dict[str, Any]]:

    from prompts.kg_prompt import DEFAULT_ENTITY_TYPES, IMAGE_ENTITY_EXTRACTION

    prompt_body = IMAGE_ENTITY_EXTRACTION.format(
        entity_types=", ".join(DEFAULT_ENTITY_TYPES),
        image_description=json.dumps(
            {
                "description": str(chunk.content),
            },
            ensure_ascii=False,
        ),
    )
    return _entity_messages_from_prompt(prompt_body, include_image_summary=True)


def _entity_position(text: str, entity_text: str, used_spans: List[Tuple[int, int]]) -> Tuple[int, int]:


    if not entity_text:
        return -1, -1

    start = text.find(entity_text)
    while start != -1:
        end = start + len(entity_text)
        overlaps = any(not (end <= s or start >= e) for s, e in used_spans)
        if not overlaps:
            used_spans.append((start, end))
            return start, end
        start = text.find(entity_text, start + 1)
    return -1, -1


def _default_equation_entity(chunk_text: str) -> Dict[str, str]:


    equation_text = " ".join(str(chunk_text).split()[:8]).strip()
    entity_name = f"Equation: {equation_text}" if equation_text else "Equation"
    return {
        "entity_name": entity_name,
        "entity_type": "EQUATION_OR_FORMULA",
        "description": f"A formula represented by LaTeX code: {chunk_text}",
    }


def extract_entities_from_one_mineru_chunk_llm(
    chunk: Any,
    *,
    output_dir: Path,
    model_name: str,
    base_url: str | None,
    vlm_model_name: str | None = None,
    vlm_base_url: str | None = None,
    temperature: float,
    max_tokens: int,
) -> Dict[str, Any]:


    cache_path = output_dir / "cache" / f"{model_name.replace('/', '_')}_entity_cache.sqlite"
    metadata = dict(getattr(chunk, "metadata", {}) or {})
    block_type = str(metadata.get("block_type", "text")).lower()
    effective_model_name = (vlm_model_name or model_name) if block_type == "image" else model_name
    effective_base_url = vlm_base_url if (block_type == "image" and vlm_base_url is not None) else base_url

    if block_type == "table":
        message_batches = _build_table_entity_extraction_message_batches(chunk)
        if not message_batches:
            message_batches = [_build_entity_extraction_messages(chunk)]
    else:
        message_batches = [_build_entity_extraction_messages(chunk)]

    responses: List[Any] = []
    for messages in message_batches:
        try:
            responses.append(
                _chat_json(
                    messages=messages,
                    model_name=effective_model_name,
                    base_url=effective_base_url,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    cache_path=cache_path,
                )
            )
        except Exception as exc:
            if block_type == "image":
                try:
                    fallback_messages = _build_image_text_only_entity_messages(chunk)
                    responses.append(
                        _chat_json(
                            messages=fallback_messages,
                            model_name=effective_model_name,
                            base_url=effective_base_url,
                            temperature=temperature,
                            max_tokens=max_tokens,
                            cache_path=cache_path,
                        )
                    )
                    continue
                except Exception:
                    pass
            print(f"Warning: entity extraction failed for chunk {chunk.chunk_id}: {exc}")

    entities = []
    used_spans: List[Tuple[int, int]] = []
    seen = set()

    response_entities: List[Any] = []
    for response in responses:
        if isinstance(response, dict):
            response_entities.extend(response.get("entities", []))
        elif isinstance(response, list):
            response_entities.extend(response)

    if block_type == "equation" and not response_entities:
        equation_text = str((metadata.get("current_node") or {}).get("content") or "").strip() or chunk.content
        response_entities.append(_default_equation_entity(equation_text))

    for item in response_entities:
        if not isinstance(item, dict):
            continue
        entity_text = str(item.get("entity_name") or item.get("text") or "").strip()
        if block_type == "equation":
            entity_label = "EQUATION_OR_FORMULA"
        else:
            entity_label = str(item.get("entity_type") or item.get("label") or "").strip().upper()
        entity_description = str(item.get("description") or item.get("label_description") or "").strip()
        if not entity_text or not entity_label:
            continue
        dedup_key = (entity_text, entity_label)
        if dedup_key in seen:
            continue
        seen.add(dedup_key)
        start_char, end_char = _entity_position(chunk.content, entity_text, used_spans)
        entities.append(
            {
                "text": entity_text,
                "label": entity_label,
                "start_char": start_char,
                "end_char": end_char,
                "label_description": entity_description,
            }
        )

    result = {
        "chunk_id": chunk.chunk_id,
        "chunk_text": chunk.content,
        "entities": entities,
    }
    if block_type == "image":
        image_summary = ""
        for response in responses:
            candidate = response.get("image_summary") if isinstance(response, dict) else None
            if isinstance(candidate, str) and candidate.strip():
                image_summary = candidate.strip()
                break
        if not image_summary:
            image_summary = next(
                (
                    entity["label_description"]
                    for entity in entities
                    if entity["label"] == "IMAGE" and entity["label_description"]
                ),
                "",
            )
        result["image_summary"] = image_summary
        result["original_content"] = str(chunk.content)
        if image_summary and image_summary not in str(chunk.content):
            result["chunk_text"] = f"{chunk.content}\n\n[Image Summary]\n{image_summary}"
    return result


def schema_key_from_parts(head_type: str, relation: str, tail_type: str) -> str:

    return f"({head_type}, {relation}, {tail_type})"


def make_schema_id(schema: List[str]) -> str:

    payload = "|".join(str(x) for x in schema)
    return f"schema_{hashlib.sha1(payload.encode('utf-8')).hexdigest()[:12]}"


def make_type_schema_id(entity_type: str) -> str:

    payload = str(entity_type)
    return f"type_schema_{hashlib.sha1(payload.encode('utf-8')).hexdigest()[:12]}"


def make_triple_id(chunk_id: str, triple: List[str]) -> str:

    payload = f"{chunk_id}|{'|'.join(str(x) for x in triple)}"
    return f"triple_{hashlib.sha1(payload.encode('utf-8')).hexdigest()[:12]}"


def make_entity_id(entity_name: str, entity_type: str) -> str:

    payload = f"{str(entity_type).strip().lower()}|{str(entity_name).strip().lower()}"
    return f"entity_{hashlib.sha1(payload.encode('utf-8')).hexdigest()[:12]}"


def summarize_chunk_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:

    return {
        "source_file": metadata.get("source_file"),
        "page_idx": metadata.get("page_idx"),
        "block_type": metadata.get("block_type"),
        "node_type": metadata.get("node_type"),
        "title_path": metadata.get("title_path"),
        "parent_id": metadata.get("parent_id"),
        "child_ids": metadata.get("child_ids"),
        "img_path": metadata.get("img_path"),
    }


def entity_chunk_for_output(entity_chunk: Dict[str, Any]) -> Dict[str, Any]:


    result = {
        "chunk_id": entity_chunk.get("chunk_id"),
        "context": entity_chunk.get("chunk_text", ""),
        "entities": entity_chunk.get("entities", []),
    }
    for key in ("image_summary", "original_content"):
        if key in entity_chunk:
            result[key] = entity_chunk[key]
    return result


def write_json(data: Dict[str, Any], path: Path) -> None:

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _build_embedding_runtime_config() -> Any:
    from src.utils.config_utils import BaseConfig

    config = BaseConfig()
    config.embedding_model_name = DEFAULT_EMBEDDING_MODEL
    config.embedding_base_url = DEFAULT_EMBEDDING_BASE_URL
    config.embedding_api_key = DEFAULT_EMBEDDING_API_KEY
    config.embedding_batch_size = int(os.environ.get("MEMGRAPHRAG_EMBED_BATCH_SIZE", "8"))
    config.embedding_max_seq_len = int(
        os.environ.get("MEMGRAPHRAG_EMBED_MAX_SEQ_LEN", "512")
    )
    return config


def create_shared_embedding_model() -> tuple[Any, Any]:
    from src.embedding_model import _get_embedding_model_class

    config = _build_embedding_runtime_config()
    embedding_model = _get_embedding_model_class(config.embedding_model_name)(global_config=config)
    return config, embedding_model


def encode_memory_embeddings(
    memory: Any,
    output_dir: Path,
    *,
    embedding_model: Any | None = None,
    embedding_config: Any | None = None,
) -> None:


    from src.embedding_store import EmbeddingStore
    from src.utils.misc_utils import compute_mdhash_id
    import numpy as np
    import pandas as pd
    import torch

    config = embedding_config or _build_embedding_runtime_config()
    local_embedding_model = embedding_model
    owns_embedding_model = local_embedding_model is None

    try:
        if local_embedding_model is None:
            _, local_embedding_model = create_shared_embedding_model()
        schema_texts = [node.content for node in memory.schema_layer]
        schema_relation_texts = [_schema_relation_text(node.content) for node in memory.schema_relation_layer]
        schema_store = EmbeddingStore(
            local_embedding_model,
            os.path.join(str(output_dir), "schema_embeddings"),
            config.embedding_batch_size,
            "schema",
        )
        schema_relation_store = EmbeddingStore(
            local_embedding_model,
            os.path.join(str(output_dir), "schema_relation_embeddings"),
            config.embedding_batch_size,
            "schema_relation",
        )

        schema_store.insert_strings(schema_texts)
        schema_relation_store.insert_strings(schema_relation_texts)


        chunk_embeddings_dir = output_dir / "chunk_embeddings"
        chunk_embeddings_dir.mkdir(parents=True, exist_ok=True)
        chunk_vdb_path = chunk_embeddings_dir / "vdb_chunk.parquet"

        segment_rows: List[Dict[str, Any]] = []
        passage_embeddings: Dict[str, List[float]] = {}
        split_fn = getattr(local_embedding_model, "_split_long_text", None)
        max_length = int(config.embedding_max_seq_len)

        for passage in memory.passage_layer:
            if callable(split_fn):
                segments = split_fn(passage.content, max_length=max_length)
            else:
                segments = [passage.content]

            segment_vectors = local_embedding_model.batch_encode(segments)
            segment_vectors = np.asarray(segment_vectors, dtype=np.float32)
            if segment_vectors.ndim == 1:
                segment_vectors = segment_vectors.reshape(1, -1)

            for seg_idx, (segment_text, segment_vec) in enumerate(zip(segments, segment_vectors)):
                hash_source = f"{passage.chunk_id}::{seg_idx}::{segment_text}"
                segment_rows.append(
                    {
                        "hash_id": compute_mdhash_id(hash_source, prefix="chunk-"),
                        "content": segment_text,
                        "embedding": segment_vec,
                        "parent_chunk_id": passage.chunk_id,
                        "parent_passage_idx": passage.idx,
                        "segment_idx": seg_idx,
                    }
                )

            merged_vec = segment_vectors.mean(axis=0)
            norm = float(np.linalg.norm(merged_vec))
            if norm > 0:
                merged_vec = merged_vec / norm
            passage_embeddings[passage.chunk_id] = merged_vec.astype(np.float32).tolist()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        pd.DataFrame(segment_rows).to_parquet(chunk_vdb_path, index=False)

        schema_embeddings = {
            node.content: schema_store.get_embedding(schema_store.text_to_hash_id[node.content]).tolist()
            for node in memory.schema_layer
            if node.content in schema_store.text_to_hash_id
        }
        schema_relation_embeddings = {
            node.content: schema_relation_store.get_embedding(
                schema_relation_store.text_to_hash_id[_schema_relation_text(node.content)]
            ).tolist()
            for node in memory.schema_relation_layer
            if _schema_relation_text(node.content) in schema_relation_store.text_to_hash_id
        }

        memory.attach_embeddings(
            schema_embeddings=schema_embeddings,
            passage_embeddings=passage_embeddings,
            schema_relation_embeddings=schema_relation_embeddings,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Embedding model '{config.embedding_model_name}' failed during memory encoding. "
            "The pipeline no longer falls back to 'hash-test'. "
            f"Original error: {type(exc).__name__}: {exc}. "
            "Please fix the configured embedding endpoint / credentials / model name, "
            "or switch the embedding model explicitly."
        ) from exc
    finally:
        close_fn = getattr(local_embedding_model, "close", None)
        if owns_embedding_model and callable(close_fn):
            close_fn()


def _cache_key(messages: List[Dict[str, Any]], model_name: str, temperature: float) -> str:

    payload = {
        "messages": messages,
        "model": model_name,
        "temperature": temperature,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _init_cache(cache_path: Path) -> None:

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(cache_path))
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS cache (
                key TEXT PRIMARY KEY,
                response TEXT NOT NULL
            )
            """
        )
        conn.commit()
    finally:
        conn.close()


def _read_cache(cache_path: Path, key: str) -> str | None:

    conn = sqlite3.connect(str(cache_path))
    try:
        row = conn.execute("SELECT response FROM cache WHERE key = ?", (key,)).fetchone()
    finally:
        conn.close()
    if row is None:
        return None
    return str(row[0])


def _write_cache(cache_path: Path, key: str, response: str) -> None:

    conn = sqlite3.connect(str(cache_path))
    try:
        conn.execute(
            "INSERT OR REPLACE INTO cache (key, response) VALUES (?, ?)",
            (key, response),
        )
        conn.commit()
    finally:
        conn.close()


def _parse_json_response(response_text: str) -> Any:

    text = response_text.strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    stripped = text.strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        tuple_delimiter = "<|>"
        record_delimiter = "##"
        completion_delimiter = "<|COMPLETE|>"

        def strip_wrapping_quotes(value: str) -> str:
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
                return value[1:-1]
            return value

        tuple_text = stripped.replace(completion_delimiter, "")
        entities: List[Dict[str, Any]] = []
        relationships: List[Dict[str, Any]] = []

        for record in [item.strip() for item in tuple_text.split(record_delimiter) if item.strip()]:
            if not (record.startswith("(") and record.endswith(")")):
                continue
            inner = record[1:-1].strip()
            parts = [part.strip() for part in inner.split(tuple_delimiter)]
            if not parts:
                continue
            record_type = strip_wrapping_quotes(parts[0]).lower()
            if record_type == "entity" and len(parts) >= 4:
                entities.append(
                    {
                        "entity_name": strip_wrapping_quotes(parts[1]),
                        "entity_type": strip_wrapping_quotes(parts[2]),
                        "description": strip_wrapping_quotes(tuple_delimiter.join(parts[3:])),
                    }
                )
            elif record_type == "relationship" and len(parts) >= 5:
                weight_raw = strip_wrapping_quotes(parts[-1])
                try:
                    weight = float(weight_raw)
                except Exception:
                    weight = 1.0
                relationships.append(
                    {
                        "src_entity_name": strip_wrapping_quotes(parts[1]),
                        "tgt_entity_name": strip_wrapping_quotes(parts[2]),
                        "description": strip_wrapping_quotes(tuple_delimiter.join(parts[3:-1])),
                        "weight": weight,
                    }
                )

        if entities or relationships:
            return {"entities": entities, "relationships": relationships}
        raise


def _chat_json(
    *,
    messages: List[Dict[str, Any]],
    model_name: str,
    base_url: str | None,
    temperature: float,
    max_tokens: int,
    cache_path: Path,
) -> Any:


    from llm_client import LLMClient, LLMConfig

    effective_model_name = str(model_name or "").strip()
    effective_base_url = str(base_url or "").strip()
    if not effective_model_name:
        raise ValueError("Local model name is empty. Please set DEFAULT_LLM_MODEL / DEFAULT_ENTITY_LLM_MODEL / DEFAULT_VLM_MODEL.")
    if not effective_base_url:
        raise ValueError("Local base_url is empty. Please set DEFAULT_LLM_BASE_URL / DEFAULT_ENTITY_LLM_BASE_URL / DEFAULT_VLM_BASE_URL.")

    config = LLMConfig(
        backend="api",
        model_name=effective_model_name,
        api_key="openai",
        base_url=effective_base_url,
        max_tokens=max_tokens,
        temperature=temperature,
        cache_dir=str(cache_path.parent),
    )
    client = LLMClient(config)
    parsed, _, _ = client.chat_json(messages, use_cache=True)
    return parsed


def _filter_entities(entities: List[Dict[str, Any]], max_entities: int) -> List[Dict[str, Any]]:

    excluded_labels = {"CARDINAL", "ORDINAL", "MONEY", "PERCENT", "QUANTITY"}
    filtered = [ent for ent in entities if ent.get("label") not in excluded_labels]

    seen = set()
    unique_entities = []
    for ent in filtered:
        key = (ent.get("text"), ent.get("label"))
        if key in seen:
            continue
        seen.add(key)
        unique_entities.append(ent)

    return unique_entities[:max_entities]


def _valid_relation(item: Dict[str, Any]) -> bool:

    required = ["head", "relation", "tail", "head_type", "tail_type"]
    return all(item.get(field) for field in required)


def extract_entities_from_one_mineru_chunk(
    chunk: Any,
    *,
    output_dir: Path,
    model_name: str,
    base_url: str | None,
    vlm_model_name: str | None = None,
    vlm_base_url: str | None = None,
    temperature: float,
    max_tokens: int,
    extraction_mode: str = "llm",
    nlp: Any = None,
    spacy_module: Any = None,
) -> Dict[str, Any]:


    if extraction_mode == "llm":
        return extract_entities_from_one_mineru_chunk_llm(
            chunk,
            output_dir=output_dir,
            model_name=model_name,
            base_url=base_url,
            vlm_model_name=vlm_model_name,
            vlm_base_url=vlm_base_url,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    doc = nlp(chunk.content)
    entities = []
    for ent in doc.ents:
        entities.append(
            {
                "text": ent.text,
                "label": ent.label_,
                "start_char": ent.start_char,
                "end_char": ent.end_char,
                "label_description": spacy_module.explain(ent.label_),
            }
        )

    return {
        "chunk_id": chunk.chunk_id,
        "chunk_text": chunk.content,
        "entities": entities,
    }


def extract_relations_from_one_entity_chunk(
    entity_chunk: Dict[str, Any],
    *,
    output_dir: Path,
    model_name: str,
    base_url: str | None,
    temperature: float,
    max_tokens: int,
    max_entities_per_chunk: int,
) -> List[Dict[str, Any]]:


    from prompt_builder import build_relation_extraction_messages

    chunk_id = entity_chunk.get("chunk_id")
    if any(
        str(entity.get("label") or "").strip().upper() == "EQUATION_OR_FORMULA"
        for entity in entity_chunk.get("entities", [])
    ):
        return []
    filtered_entities = _filter_entities(
        entity_chunk.get("entities", []),
        max_entities=max_entities_per_chunk,
    )
    if len(filtered_entities) < 2:
        return []

    messages = build_relation_extraction_messages(
        entity_chunk.get("chunk_text", ""),
        filtered_entities,
    )
    cache_path = output_dir / "cache" / f"{model_name.replace('/', '_')}_relation_cache.sqlite"

    try:
        response = _chat_json(
            messages=messages,
            model_name=model_name,
            base_url=base_url,
            temperature=temperature,
            max_tokens=max_tokens,
            cache_path=cache_path,
        )
    except Exception as exc:
        print(f"Warning: relation extraction failed for chunk {chunk_id}: {exc}")
        return []

    relation_items: List[Dict[str, Any]] = []
    if isinstance(response, dict):
        if "relations" in response and isinstance(response.get("relations"), list):
            relation_items = [item for item in response["relations"] if isinstance(item, dict)]
        elif _valid_relation(response):
            relation_items = [response]
    elif isinstance(response, list):
        relation_items = [item for item in response if isinstance(item, dict)]

    triples = []
    for item in relation_items:
        if _valid_relation(item):
            triples.append(
                {
                    "head": str(item["head"]),
                    "relation": str(item["relation"]),
                    "tail": str(item["tail"]),
                    "head_type": str(item["head_type"]),
                    "tail_type": str(item["tail_type"]),
                    "chunk_id": chunk_id,
                }
            )
    return triples


def relation_chunk_record(
    entity_chunk: Dict[str, Any],
    triples: List[Dict[str, Any]],
    *,
    chunk_metadata: Dict[str, Any],
) -> Dict[str, Any]:


    triple_to_schema = {}
    for triple in triples:
        triple_key = f"({triple['head']}, {triple['relation']}, {triple['tail']})"
        schema_value = f"({triple['head_type']}, {triple['relation']}, {triple['tail_type']})"
        triple_to_schema[triple_key] = schema_value

    result = {
        "chunk_id": entity_chunk.get("chunk_id"),
        "chunk_text": entity_chunk.get("chunk_text", ""),
        "entities": entity_chunk.get("entities", []),
        "triples": triples,
        "num_triples": len(triples),
        "triple_to_schema": triple_to_schema,
        "chunk_metadata": chunk_metadata,
    }
    for key in ("image_summary", "original_content"):
        if key in entity_chunk:
            result[key] = entity_chunk[key]
    return result


def relation_records_to_output(
    relation_chunks: List[Dict[str, Any]],
    *,
    source_metadata: Dict[str, Any],
    model_name: str,
) -> Dict[str, Any]:


    relation_counts: Dict[str, int] = {}
    schema_counts: Dict[str, int] = {}
    schema_nodes_by_type: Dict[str, Dict[str, Any]] = {}
    schema_relations_by_key: Dict[str, Dict[str, Any]] = {}
    entity_nodes_by_id: Dict[str, Dict[str, Any]] = {}
    triples_output: List[Dict[str, Any]] = []
    chunks_output: List[Dict[str, Any]] = []

    for chunk in relation_chunks:
        chunk_id = str(chunk.get("chunk_id"))
        chunk_triple_ids: List[str] = []
        chunk_entity_ids: set = set()
        chunk_text = str(chunk.get("chunk_text", ""))
        chunk_metadata = summarize_chunk_metadata(chunk.get("chunk_metadata", {}))

        for entity in chunk.get("entities", []):
            entity_name = str(entity.get("text") or "").strip()
            entity_type = str(entity.get("label") or "").strip().upper()
            entity_description = str(entity.get("label_description") or "").strip()
            if not entity_name or not entity_type:
                continue

            schema_id = make_type_schema_id(entity_type)
            if entity_type not in schema_nodes_by_type:
                schema_nodes_by_type[entity_type] = {
                    "schema_id": schema_id,
                    "schema_type": entity_type,
                    "in_degree": 0,
                    "out_degree": 0,
                    "entity_ids": [],
                    "chunk_ids": [],
                }

            entity_id = make_entity_id(entity_name, entity_type)
            chunk_entity_ids.add(entity_id)
            node = entity_nodes_by_id.setdefault(
                entity_id,
                {
                    "entity_id": entity_id,
                    "entity_name": entity_name,
                    "entity_type": entity_type,
                    "schema_id": schema_id,
                    "description": entity_description,
                    "chunk_ids": [],
                    "mention_count": 0,
                },
            )
            if entity_description and not node.get("description"):
                node["description"] = entity_description
            node["mention_count"] += 1
            if chunk_id not in node["chunk_ids"]:
                node["chunk_ids"].append(chunk_id)

        for triple in chunk.get("triples", []):
            schema = [
                str(triple["head_type"]),
                str(triple["relation"]),
                str(triple["tail_type"]),
            ]
            schema_key = schema_key_from_parts(schema[0], schema[1], schema[2])
            schema_relation_id = make_schema_id(schema)
            head_schema_id = make_type_schema_id(schema[0])
            tail_schema_id = make_type_schema_id(schema[2])
            head_entity_id = make_entity_id(str(triple["head"]), schema[0])
            tail_entity_id = make_entity_id(str(triple["tail"]), schema[2])
            relation_counts[schema[1]] = relation_counts.get(schema[1], 0) + 1
            schema_counts[schema_key] = schema_counts.get(schema_key, 0) + 1

            if schema[0] not in schema_nodes_by_type:
                schema_nodes_by_type[schema[0]] = {
                    "schema_id": head_schema_id,
                    "schema_type": schema[0],
                    "in_degree": 0,
                    "out_degree": 0,
                    "entity_ids": [],
                    "chunk_ids": [],
                }
            if schema[2] not in schema_nodes_by_type:
                schema_nodes_by_type[schema[2]] = {
                    "schema_id": tail_schema_id,
                    "schema_type": schema[2],
                    "in_degree": 0,
                    "out_degree": 0,
                    "entity_ids": [],
                    "chunk_ids": [],
                }

            if schema_key not in schema_relations_by_key:
                schema_relations_by_key[schema_key] = {
                    "schema_relation_id": schema_relation_id,
                    "schema_key": schema_key,
                    "source_schema_id": head_schema_id,
                    "target_schema_id": tail_schema_id,
                    "head_type": schema[0],
                    "relation": schema[1],
                    "tail_type": schema[2],
                    "count": 0,
                }
                schema_nodes_by_type[schema[0]]["out_degree"] += 1
                schema_nodes_by_type[schema[2]]["in_degree"] += 1
            schema_relations_by_key[schema_key]["count"] += 1

            triple_list = [str(triple["head"]), str(triple["relation"]), str(triple["tail"])]
            triple_id = make_triple_id(chunk_id, triple_list)
            chunk_triple_ids.append(triple_id)
            chunk_entity_ids.add(head_entity_id)
            chunk_entity_ids.add(tail_entity_id)

            head_node = entity_nodes_by_id.setdefault(
                head_entity_id,
                {
                    "entity_id": head_entity_id,
                    "entity_name": triple_list[0],
                    "entity_type": schema[0],
                    "schema_id": head_schema_id,
                    "description": "",
                    "chunk_ids": [],
                    "mention_count": 0,
                },
            )
            tail_node = entity_nodes_by_id.setdefault(
                tail_entity_id,
                {
                    "entity_id": tail_entity_id,
                    "entity_name": triple_list[2],
                    "entity_type": schema[2],
                    "schema_id": tail_schema_id,
                    "description": "",
                    "chunk_ids": [],
                    "mention_count": 0,
                },
            )
            if chunk_id not in head_node["chunk_ids"]:
                head_node["chunk_ids"].append(chunk_id)
            if chunk_id not in tail_node["chunk_ids"]:
                tail_node["chunk_ids"].append(chunk_id)

            triples_output.append(
                {
                    "triple_id": triple_id,
                    "chunk_id": chunk_id,
                    "schema_relation_id": schema_relation_id,
                    "head_schema_id": head_schema_id,
                    "tail_schema_id": tail_schema_id,
                    "head_entity_id": head_entity_id,
                    "tail_entity_id": tail_entity_id,
                    "head": triple_list[0],
                    "relation": triple_list[1],
                    "tail": triple_list[2],
                    "head_type": schema[0],
                    "tail_type": schema[2],
                }
            )

        chunks_output.append(
            {
                "chunk_id": chunk_id,
                "num_entities": len(chunk_entity_ids),
                "num_triples": len(chunk.get("triples", [])),
                "entity_ids": sorted(chunk_entity_ids),
                "triple_ids": chunk_triple_ids,
                "chunk_text_preview": chunk_text[:200],
                "chunk_metadata": chunk_metadata,
            }
        )

    for entity_node in entity_nodes_by_id.values():
        entity_node["chunk_ids"] = sorted(set(entity_node.get("chunk_ids", [])))
        entity_node["num_chunks"] = len(entity_node["chunk_ids"])

        schema_type = str(entity_node["entity_type"])
        schema_node = schema_nodes_by_type.setdefault(
            schema_type,
            {
                "schema_id": make_type_schema_id(schema_type),
                "schema_type": schema_type,
                "in_degree": 0,
                "out_degree": 0,
                "entity_ids": [],
                "chunk_ids": [],
            },
        )
        if entity_node["entity_id"] not in schema_node["entity_ids"]:
            schema_node["entity_ids"].append(entity_node["entity_id"])
        for chunk_id in entity_node["chunk_ids"]:
            if chunk_id not in schema_node["chunk_ids"]:
                schema_node["chunk_ids"].append(chunk_id)

    schema_nodes_output: List[Dict[str, Any]] = []
    for schema_node in schema_nodes_by_type.values():
        schema_node["entity_ids"] = sorted(schema_node.get("entity_ids", []))
        schema_node["chunk_ids"] = sorted(schema_node.get("chunk_ids", []))
        schema_node["num_entities"] = len(schema_node["entity_ids"])
        schema_node["num_chunks"] = len(schema_node["chunk_ids"])
        schema_nodes_output.append(schema_node)

    entity_nodes_output = sorted(entity_nodes_by_id.values(), key=lambda item: item["entity_id"])
    schema_nodes_output = sorted(schema_nodes_output, key=lambda item: item["schema_id"])
    schema_edges_output = sorted(
        schema_relations_by_key.values(),
        key=lambda item: item["schema_relation_id"],
    )

    return {
        "metadata": {
            "source_file": source_metadata,
            "total_chunks": len(chunks_output),
            "total_entities": len(entity_nodes_output),
            "total_relations": len(triples_output),
            "unique_schema_nodes": len(schema_nodes_by_type),
            "unique_schema_relations": len(schema_relations_by_key),
            "relation_types_count": relation_counts,
            "schema_count": schema_counts,
            "model": model_name,
            "mode": "incremental",
        },
        "chunks": chunks_output,
        "entities": entity_nodes_output,
        "triples": triples_output,
        "schemas": schema_nodes_output,
        "schema_graph": {
            "nodes": schema_nodes_output,
            "edges": schema_edges_output,
        },
    }


def triple_key_from_list(triple: List[str]) -> str:

    return str(tuple(str(x) for x in triple))


def triple_list_from_relation(item: Dict[str, Any]) -> List[str]:

    return [str(item["head"]), str(item["relation"]), str(item["tail"])]


def ontology_from_relation(item: Dict[str, Any]) -> List[str]:

    return [str(item["head_type"]), str(item["relation"]), str(item["tail_type"])]


def openie_doc_from_relation_chunk(relation_chunk: Dict[str, Any]) -> Dict[str, Any]:


    triples: List[List[str]] = []
    triple_ontology: Dict[str, List[str]] = {}
    for item in relation_chunk.get("triples", []):
        triple = triple_list_from_relation(item)
        ontology = ontology_from_relation(item)
        triples.append(triple)
        triple_ontology[triple_key_from_list(triple)] = ontology

    result = {
        "idx": str(relation_chunk.get("chunk_id")),
        "passage": relation_chunk.get("chunk_text", ""),
        "extracted_triples": triples,
        "extracted_triple_ontology": triple_ontology,
        "chunk_metadata": summarize_chunk_metadata(relation_chunk.get("chunk_metadata", {})),
    }
    for key in ("image_summary", "original_content"):
        if key in relation_chunk:
            result[key] = relation_chunk[key]
    return result


def iter_doc_triples(docs: List[Dict[str, Any]]) -> Iterable[Tuple[Dict[str, Any], List[str], List[str]]]:


    for doc in docs:
        triple_ontology = doc.get("extracted_triple_ontology") or {}
        for triple in doc.get("extracted_triples", []):
            if not (isinstance(triple, list) and len(triple) == 3):
                continue
            ontology = triple_ontology.get(triple_key_from_list(triple))
            if not (isinstance(ontology, list) and len(ontology) == 3):
                ontology = ["UNKNOWN", str(triple[1]), "UNKNOWN"]
            yield doc, [str(x) for x in triple], [str(x) for x in ontology]


def empty_conflict_index() -> Dict[str, Any]:

    return {
        "records": {},
        "by_head_relation": {},
        "by_entity": {},
        "by_schema": {},
    }


def _index_add(mapping: Dict[Any, set], key: Any, triple_id: str) -> None:

    mapping.setdefault(key, set()).add(triple_id)


def build_conflict_index(docs: List[Dict[str, Any]]) -> Dict[str, Any]:


    index = empty_conflict_index()
    for doc, triple, ontology in iter_doc_triples(docs):
        add_triple_to_conflict_index(index, doc, triple, ontology)

    return index


def add_triple_to_conflict_index(
    index: Dict[str, Any],
    doc: Dict[str, Any],
    triple: List[str],
    ontology: List[str],
) -> None:

    triple_id = triple_key_from_list(triple)
    previous = index["records"].get(triple_id)
    sources = list(conflict_record_sources(previous)) if previous else []
    source = {
        "doc_idx": doc.get("idx"),
        "passage": doc.get("passage", ""),
        "modality": (doc.get("chunk_metadata") or {}).get("block_type", "text"),
    }
    if source not in sources:
        sources.append(source)
    triple_record = {
        "id": triple_id,
        "triple": triple,
        "ontology": ontology,
        "passage": doc.get("passage", ""),
        "doc_idx": doc.get("idx"),
        "sources": sources,
    }
    index["records"][triple_id] = triple_record

    head = str(triple[0]).lower()
    relation = str(triple[1]).lower()
    tail = str(triple[2]).lower()
    schema_key = tuple(str(x).lower() for x in ontology)

    _index_add(index["by_head_relation"], (head, relation), triple_id)
    _index_add(index["by_entity"], head, triple_id)
    _index_add(index["by_entity"], tail, triple_id)
    _index_add(index["by_schema"], schema_key, triple_id)


def conflict_record_sources(record: Dict[str, Any]) -> List[Dict[str, Any]]:

    sources = record.get("sources")
    if isinstance(sources, list) and sources:
        return sources
    return [{
        "doc_idx": record.get("doc_idx"),
        "passage": record.get("passage", ""),
        "modality": record.get("modality", "text"),
    }]


def collect_conflict_supporting_chunks(
    index: Dict[str, Any],
    triple_records: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:

    entity_names = {
        str(triple[position]).lower()
        for record in triple_records
        for triple in [record.get("triple", [])]
        if isinstance(triple, list) and len(triple) == 3
        for position in (0, 2)
    }
    supporting_fact_ids: set[str] = set()
    for entity_name in entity_names:
        supporting_fact_ids.update(index["by_entity"].get(entity_name, set()))

    sources: List[Dict[str, Any]] = []
    seen: set[Tuple[str, str]] = set()
    for fact_id in sorted(supporting_fact_ids):
        for source in conflict_record_sources(index["records"][fact_id]):
            key = (str(source.get("doc_idx")), str(source.get("passage", "")))
            if key not in seen:
                seen.add(key)
                sources.append(dict(source))
    return sources


def add_doc_to_conflict_index(index: Dict[str, Any], doc: Dict[str, Any]) -> None:

    for _, triple, ontology in iter_doc_triples([doc]):
        add_triple_to_conflict_index(index, doc, triple, ontology)


def conflict_index_for_output(index: Dict[str, Any]) -> Dict[str, Any]:


    return {
        "metadata": {
            "num_records": len(index.get("records", {})),
            "num_head_relation_keys": len(index.get("by_head_relation", {})),
            "num_entity_keys": len(index.get("by_entity", {})),
            "num_schema_keys": len(index.get("by_schema", {})),
        },
        "records": list(index.get("records", {}).values()),
        "by_head_relation": {
            f"{key[0]}|||{key[1]}": sorted(list(value))
            for key, value in index.get("by_head_relation", {}).items()
        },
        "by_entity": {
            str(key): sorted(list(value))
            for key, value in index.get("by_entity", {}).items()
        },
        "by_schema": {
            "|||".join(key): sorted(list(value))
            for key, value in index.get("by_schema", {}).items()
        },
    }


def find_related_existing_triples(
    new_triple: List[str],
    conflict_index: Dict[str, Any],
    *,
    new_ontology: List[str] | None,
    max_related: int,
) -> List[Dict[str, Any]]:


    new_s, new_p, new_o = [x.lower() for x in new_triple]
    new_triple_id = triple_key_from_list(new_triple)
    related = []
    seen = set()
    candidate_ids: List[Tuple[str, str]] = []

    for triple_id in conflict_index["by_head_relation"].get((new_s, new_p), set()):
        candidate_ids.append((triple_id, "same_subject_predicate"))

    for entity in {new_s, new_o}:
        for triple_id in conflict_index["by_entity"].get(entity, set()):
            candidate_ids.append((triple_id, "shared_entity"))

    if new_ontology and len(new_ontology) == 3:
        schema_key = tuple(str(x).lower() for x in new_ontology)
        for triple_id in conflict_index["by_schema"].get(schema_key, set()):
            candidate_ids.append((triple_id, "same_schema"))

    for triple_id, source in candidate_ids:
        if triple_id == new_triple_id:
            continue
        if triple_id in seen:
            continue
        seen.add(triple_id)
        record = conflict_index["records"].get(triple_id)
        if not record:
            continue
        related.append(
            {
                "id": record["id"],
                "triple": record["triple"],
                "source": source,
                "passage": record.get("passage", ""),
                "doc_idx": record.get("doc_idx"),
                "sources": conflict_record_sources(record),
            }
        )
        if len(related) >= max_related:
            break

    return related


def filter_conflict_candidates_for_llm(
    new_triple: List[str],
    related_triples: List[Dict[str, Any]],
    *,
    max_related: int,
) -> List[Dict[str, Any]]:


    new_s = str(new_triple[0]).strip().lower()
    new_p = str(new_triple[1]).strip().lower()
    new_o = str(new_triple[2]).strip().lower()

    strongest: List[Dict[str, Any]] = []
    fallback: List[Dict[str, Any]] = []
    seen_ids: set = set()

    for item in related_triples:
        triple = item.get("triple")
        if not (isinstance(triple, list) and len(triple) == 3):
            continue
        triple_id = str(item.get("id"))
        if triple_id in seen_ids:
            continue

        old_s = str(triple[0]).strip().lower()
        old_p = str(triple[1]).strip().lower()
        old_o = str(triple[2]).strip().lower()

        if old_s == new_s and old_p == new_p and old_o != new_o:
            strongest.append(item)
            seen_ids.add(triple_id)
            continue

        source = str(item.get("source", "")).lower()
        if source == "same_schema":
            fallback.append(item)
            seen_ids.add(triple_id)

    narrowed = strongest if strongest else fallback
    return narrowed[:max_related]


def find_related_triples_within_doc(
    new_triple: List[str],
    current_doc: Dict[str, Any],
    *,
    max_related: int,
) -> List[Dict[str, Any]]:


    new_s, new_p, new_o = [x.lower() for x in new_triple]
    related = []
    seen = set()
    new_key = triple_key_from_list(new_triple)

    for old_triple in current_doc.get("extracted_triples", []):
        if not (isinstance(old_triple, list) and len(old_triple) == 3):
            continue

        old_key = triple_key_from_list(old_triple)
        if old_key == new_key:
            continue

        old_s, old_p, old_o = [str(x).lower() for x in old_triple]
        same_subject_predicate = old_s == new_s and old_p == new_p
        shared_entity = old_s in {new_s, new_o} or old_o in {new_s, new_o}
        if not (same_subject_predicate or shared_entity):
            continue

        if old_key in seen:
            continue
        seen.add(old_key)
        related.append(
            {
                "id": old_key,
                "triple": [str(x) for x in old_triple],
                "source": "same_subject_predicate_within_doc" if same_subject_predicate else "shared_entity_within_doc",
                "passage": current_doc.get("passage", ""),
                "doc_idx": current_doc.get("idx"),
            }
        )
        if len(related) >= max_related:
            break

    return related


def merge_related_triple_candidates(*candidate_groups: List[Dict[str, Any]], max_related: int) -> List[Dict[str, Any]]:


    merged: List[Dict[str, Any]] = []
    seen = set()
    for group in candidate_groups:
        for item in group:
            triple_id = item.get("id")
            if triple_id in seen:
                continue
            seen.add(triple_id)
            merged.append(item)
            if len(merged) >= max_related:
                return merged
    return merged


def check_conflict_for_new_triple(
    new_triple: List[str],
    related_triples: List[Dict[str, Any]],
    *,
    output_dir: Path,
    args: argparse.Namespace,
) -> Dict[str, Any]:


    if not related_triples:
        return {"has_conflict": False, "conflicts": [], "conflicting_triple_ids": []}

    from prompt import PROMPT

    prompt_template = PROMPT["conflict_detection"]
    related_text = ""
    for i, item in enumerate(related_triples, 1):

        related_text += (
            f"Triple {i} (ID: {item['id']}, Source: {item.get('source', 'unknown')}): "
            f"{item['triple']}\n"
        )

    user_prompt = prompt_template["user"].replace("${target_triple}", str(new_triple))
    user_prompt = user_prompt.replace("${related_triples}", related_text.strip())
    messages = [
        {"role": "system", "content": prompt_template["system"]},
        {"role": "user", "content": user_prompt},
    ]
    cache_path = output_dir / "cache" / f"{args.llm_model.replace('/', '_')}_conflict_detection_cache.sqlite"

    try:
        result = _chat_json(
            messages=messages,
            model_name=args.llm_model,
            base_url=args.llm_base_url,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            cache_path=cache_path,
        )
    except Exception as exc:
        print(f"Warning: conflict detection failed for {new_triple}: {exc}")
        return {"has_conflict": False, "conflicts": [], "conflicting_triple_ids": []}

    if not isinstance(result, dict):
        return {"has_conflict": False, "conflicts": [], "conflicting_triple_ids": []}
    result.setdefault("has_conflict", False)
    result.setdefault("conflicts", [])
    result.setdefault("conflicting_triple_ids", [])
    return result


def build_conflict_groups(index: Dict[str, Any]) -> List[Dict[str, Any]]:


    records = index.get("records", {})
    groups: List[Dict[str, Any]] = []

    def build_group(
        *,
        group_type: str,
        raw_key: Tuple[str, ...],
        triple_ids: Iterable[str],
        possible_conflict_reason: str,
    ) -> None:


        triple_ids = list(dict.fromkeys(str(item) for item in triple_ids))
        if len(triple_ids) < 2:

            return


        group_records = [records[item] for item in triple_ids if item in records]
        if len(group_records) < 2:
            return


        involved_chunk_ids = sorted(
            {
                str(source.get("doc_idx"))
                for item in group_records
                for source in conflict_record_sources(item)
                if source.get("doc_idx") is not None
            }
        )


        primary_chunk_id = involved_chunk_ids[0] if involved_chunk_ids else None
        groups.append(
            {
                "group_id": f"{group_type}::{'|||'.join(raw_key)}",
                "group_type": group_type,
                "group_key": list(raw_key),
                "possible_conflict_reason": possible_conflict_reason,
                "primary_chunk_id": primary_chunk_id,
                "involved_chunk_ids": involved_chunk_ids,
                "triples": group_records,
            }
        )


    for key, triple_ids in index.get("by_head_relation", {}).items():


        unique_tails = {
            str(records[item]["triple"][2]).strip().lower()
            for item in triple_ids
            if item in records
        }
        if len(unique_tails) < 2:
            continue
        build_group(
            group_type="same_head_relation",
            raw_key=(str(key[0]), str(key[1])),
            triple_ids=triple_ids,
            possible_conflict_reason=(
                "These triples share the same head entity and relation but may point to different tails. "
                "This often indicates mutual, temporal, or granularity conflict."
            ),
        )


    head_tail_groups: Dict[Tuple[str, str], List[str]] = {}
    head_only_groups: Dict[str, List[str]] = {}
    for triple_id, record in records.items():
        triple = record.get("triple")
        if not (isinstance(triple, list) and len(triple) == 3):
            continue
        head = str(triple[0]).strip().lower()
        relation = str(triple[1]).strip().lower()
        tail = str(triple[2]).strip().lower()
        head_tail_groups.setdefault((head, tail), []).append(triple_id)
        head_only_groups.setdefault(head, []).append(triple_id)


    for key, triple_ids in head_tail_groups.items():
        unique_relations = {
            str(records[item]["triple"][1]).strip().lower()
            for item in triple_ids
            if item in records
        }
        if len(unique_relations) < 2:
            continue
        build_group(
            group_type="same_head_tail_different_relation",
            raw_key=(str(key[0]), str(key[1])),
            triple_ids=triple_ids,
            possible_conflict_reason=(
                "These triples share the same head and tail entities but use different relations. "
                "The relation semantics may be inconsistent, contradictory, or redundantly mislabeled."
            ),
        )


    for head, triple_ids in head_only_groups.items():
        if len(triple_ids) < 2:
            continue
        build_group(
            group_type="same_head",
            raw_key=(str(head),),
            triple_ids=triple_ids,
            possible_conflict_reason=(
                "These triples share the same head entity. Some relations may be mutually exclusive, "
                "time-sensitive, or inconsistent when considered together."
            ),
        )


    groups.sort(key=lambda item: (item["group_type"], item["group_id"]))
    return groups


def classify_conflict_group_scope(
    group: Dict[str, Any],
    *,
    new_doc_idx: str,
) -> str | None:


    doc_ids = {
        str(source.get("doc_idx"))
        for item in group.get("triples", [])
        for source in conflict_record_sources(item)
        if source.get("doc_idx") is not None
    }
    if not doc_ids:
        return None

    if doc_ids == {new_doc_idx}:
        return "new_doc_internal"
    if new_doc_idx in doc_ids:
        return "new_doc_vs_accepted"
    return None


def check_conflict_for_group(
    group: Dict[str, Any],
    *,
    output_dir: Path,
    args: argparse.Namespace,
) -> Dict[str, Any]:


    from prompt import PROMPT

    triples_text_lines: List[str] = []
    for idx, item in enumerate(group.get("triples", []), 1):
        source_ids = [source.get("doc_idx") for source in conflict_record_sources(item)]
        triples_text_lines.append(
            f"Triple {idx} (ID: {item.get('id')}, Chunks: {source_ids}): "
            f"{item.get('triple')}"
        )

    prompt_template = PROMPT["conflict_detection"]
    user_prompt = prompt_template["user"]
    user_prompt = user_prompt.replace("${group_id}", str(group.get("group_id", "")))
    user_prompt = user_prompt.replace("${group_type}", str(group.get("group_type", "")))
    user_prompt = user_prompt.replace("${group_scope}", str(group.get("group_scope", "")))
    user_prompt = user_prompt.replace("${group_reason}", str(group.get("possible_conflict_reason", "")))
    user_prompt = user_prompt.replace("${group_triples}", "\n".join(triples_text_lines))

    messages = [
        {"role": "system", "content": prompt_template["system"]},
        {"role": "user", "content": user_prompt},
    ]
    cache_path = output_dir / "cache" / f"{args.llm_model.replace('/', '_')}_conflict_detection_cache.sqlite"

    try:
        result = _chat_json(
            messages=messages,
            model_name=args.llm_model,
            base_url=args.llm_base_url,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            cache_path=cache_path,
        )
    except Exception as exc:
        print(f"Warning: conflict detection failed for group {group.get('group_id')}: {exc}")
        return {"has_conflict": False, "conflicts": [], "conflicting_triple_ids": []}

    if not isinstance(result, dict):
        return {"has_conflict": False, "conflicts": [], "conflicting_triple_ids": []}
    result.setdefault("has_conflict", False)
    result.setdefault("conflicts", [])
    result.setdefault("conflicting_triple_ids", [])
    return result


def detect_conflicts_for_docs_batch(
    docs: List[Dict[str, Any]],
    *,
    output_dir: Path,
    args: argparse.Namespace,
) -> Dict[str, Any]:


    full_index = build_conflict_index(docs)
    candidate_groups = build_conflict_groups(full_index)

    all_conflicts: List[Dict[str, Any]] = []
    conflicting_triple_ids: set = set()
    conflicts_by_chunk: Dict[str, List[Dict[str, Any]]] = {}

    for group in candidate_groups:
        detection = check_conflict_for_group(
            group,
            output_dir=output_dir,
            args=args,
        )
        if not detection.get("has_conflict", False):
            continue

        conflicting_ids = list(detection.get("conflicting_triple_ids", []))
        conflicting_triple_ids.update(conflicting_ids)
        triple_records = group.get("triples", [])
        filtered_triples = [
            item for item in triple_records
            if str(item.get("id")) in set(conflicting_ids)
        ] or triple_records

        conflict_item = {
            "group_id": group.get("group_id"),
            "group_type": group.get("group_type"),
            "group_key": group.get("group_key", []),
            "possible_conflict_reason": group.get("possible_conflict_reason", ""),
            "chunk_id": group.get("primary_chunk_id"),
            "involved_chunk_ids": group.get("involved_chunk_ids", []),
            "conflicts": detection.get("conflicts", []),
            "conflicting_triple_ids": conflicting_ids,
            "triples": filtered_triples,
            "supporting_chunks": collect_conflict_supporting_chunks(full_index, filtered_triples),
        }
        all_conflicts.append(conflict_item)
        primary_chunk_id = str(group.get("primary_chunk_id"))
        conflicts_by_chunk.setdefault(primary_chunk_id, []).append(conflict_item)

    return {
        "mode": "posthoc_batch",
        "grouping_mode": "grouped_batch",
        "total_groups": len(candidate_groups),
        "total_triples": sum(len(doc.get("extracted_triples", [])) for doc in docs),
        "conflicts_detected": len(all_conflicts),
        "conflicting_triple_ids": sorted(conflicting_triple_ids),
        "conflicts_by_chunk": [
            {
                "chunk_id": chunk_id,
                "conflicts": items,
                "num_conflicts": len(items),
            }
            for chunk_id, items in conflicts_by_chunk.items()
        ],
        "conflicts": all_conflicts,
    }


def detect_conflicts_for_new_doc_groups(
    *,
    new_doc: Dict[str, Any],
    accepted_docs: List[Dict[str, Any]],
    output_dir: Path,
    args: argparse.Namespace,
) -> Dict[str, Any]:


    current_docs = accepted_docs + [new_doc]
    full_index = build_conflict_index(current_docs)
    new_doc_idx = str(new_doc.get("idx"))
    candidate_groups: List[Dict[str, Any]] = []
    for group in build_conflict_groups(full_index):
        group_scope = classify_conflict_group_scope(group, new_doc_idx=new_doc_idx)
        if group_scope is None:
            continue
        enriched_group = dict(group)
        enriched_group["group_scope"] = group_scope
        candidate_groups.append(enriched_group)

    all_conflicts: List[Dict[str, Any]] = []
    conflicting_triple_ids: set = set()

    for group in candidate_groups:
        detection = check_conflict_for_group(
            group,
            output_dir=output_dir,
            args=args,
        )
        if not detection.get("has_conflict", False):
            continue

        conflicting_ids = list(detection.get("conflicting_triple_ids", []))
        conflicting_triple_ids.update(conflicting_ids)

        triple_records = group.get("triples", [])
        filtered_triples = [
            item for item in triple_records
            if str(item.get("id")) in set(conflicting_ids)
        ] or triple_records

        all_conflicts.append(
            {
                "group_id": group.get("group_id"),
                "group_type": group.get("group_type"),
                "group_scope": group.get("group_scope"),
                "group_key": group.get("group_key", []),
                "possible_conflict_reason": group.get("possible_conflict_reason", ""),
                "chunk_id": new_doc_idx,
                "involved_chunk_ids": group.get("involved_chunk_ids", []),
                "conflicts": detection.get("conflicts", []),
                "conflicting_triple_ids": conflicting_ids,
                "triples": filtered_triples,
                "supporting_chunks": collect_conflict_supporting_chunks(full_index, filtered_triples),
            }
        )

    return {
        "mode": "incremental_grouped",
        "chunk_id": new_doc_idx,
        "grouping_mode": "grouped_incremental",
        "total_groups": len(candidate_groups),
        "total_triples": len(new_doc.get("extracted_triples", [])),
        "conflicts_detected": len(all_conflicts),
        "conflicting_triple_ids": sorted(conflicting_triple_ids),
        "conflicts_by_chunk": [
            {
                "chunk_id": new_doc_idx,
                "conflicts": all_conflicts,
                "num_conflicts": len(all_conflicts),
            }
        ] if all_conflicts else [],
        "conflicts": all_conflicts,
    }


def build_resolution_source_text(
    *,
    new_doc: Dict[str, Any],
    conflict_items: List[Dict[str, Any]],
) -> str:

    lines = []
    lines.append("New chunk source:")
    lines.append(f"Chunk ID: {new_doc.get('idx')}")
    lines.append(f"Passage: {new_doc.get('passage', '')}")
    lines.append("")
    lines.append("Conflicting triples:")

    seen = set()
    seen_sources = {(str(new_doc.get("idx")), str(new_doc.get("passage", "")))}

    def append_source(source: Dict[str, Any]) -> None:
        key = (str(source.get("doc_idx")), str(source.get("passage", "")))
        if key in seen_sources:
            return
        seen_sources.add(key)
        lines.append(f"  Source chunk: {source.get('doc_idx')}")
        lines.append(f"  Modality: {source.get('modality', 'text')}")
        lines.append(f"  Source passage: {source.get('passage', '')}")

    def append_fact(record: Dict[str, Any]) -> None:
        triple = record.get("triple")
        fact_id = record.get("id") or triple_key_from_list(triple)
        if fact_id in seen:
            return
        seen.add(fact_id)
        sources = conflict_record_sources(record)
        lines.append(f"- ID: {fact_id}")
        lines.append(f"  Triple: {triple}")
        lines.append(f"  Supporting chunk IDs: {[source.get('doc_idx') for source in sources]}")
        for source in sources:
            append_source(source)

    for item in conflict_items:
        grouped_triples = item.get("triples")
        if isinstance(grouped_triples, list) and grouped_triples:
            lines.append(f"Conflict group: {item.get('group_id')}")
            lines.append(f"Group type: {item.get('group_type')}")
            lines.append(f"Possible conflict reason: {item.get('possible_conflict_reason', '')}")
            for grouped_item in grouped_triples:
                append_fact(grouped_item)
        else:
            target = item.get("target_triple")
            if target:
                append_fact({
                    "id": triple_key_from_list(target),
                    "triple": target,
                    "doc_idx": new_doc.get("idx"),
                    "passage": new_doc.get("passage", ""),
                })
            for related in item.get("related_triples", []):
                append_fact(related)

        supporting_chunks = item.get("supporting_chunks", [])
        if supporting_chunks:
            lines.append("Additional entity-level supporting evidence (context, not extra facts to resolve):")
            for source in supporting_chunks:
                append_source(source)

    return "\n".join(lines)


def resolve_conflicts_for_chunk(
    *,
    new_doc: Dict[str, Any],
    conflict_items: List[Dict[str, Any]],
    output_dir: Path,
    args: argparse.Namespace,
) -> Dict[str, Any]:


    from prompt import PROMPT

    prompt_template = PROMPT["conflict_resolution"]
    source_text = build_resolution_source_text(new_doc=new_doc, conflict_items=conflict_items)
    user_prompt = prompt_template["user"].replace("${conflicting_triples_with_sources}", source_text)
    messages = [
        {"role": "system", "content": prompt_template["system"]},
        {"role": "user", "content": user_prompt},
    ]
    cache_path = output_dir / "cache" / f"{args.llm_model.replace('/', '_')}_conflict_resolution_cache.sqlite"

    try:
        result = _chat_json(
            messages=messages,
            model_name=args.llm_model,
            base_url=args.llm_base_url,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            cache_path=cache_path,
        )
    except Exception as exc:
        print(f"Warning: conflict resolution failed for chunk {new_doc.get('idx')}: {exc}")
        return {"resolved_triples": [], "unresolved_conflicts": [], "summary": str(exc)}

    if not isinstance(result, dict):
        return {"resolved_triples": [], "unresolved_conflicts": [], "summary": "LLM did not return a JSON object"}

    resolved_items = result.get("resolved_triples", [])
    if not isinstance(resolved_items, list):
        resolved_items = []
    normalized_resolved_items = []
    for idx, item in enumerate(resolved_items):
        if not isinstance(item, dict):
            print(
                "Warning: skipping malformed conflict resolution item "
                f"for chunk {new_doc.get('idx')}: resolved_triples[{idx}] is "
                f"{type(item).__name__}, expected dict."
            )
            continue
        normalized_resolved_items.append(item)

    unresolved_items = result.get("unresolved_conflicts", [])
    if not isinstance(unresolved_items, list):
        unresolved_items = []

    summary = result.get("summary", "")
    if not isinstance(summary, str):
        summary = str(summary)

    return {
        "resolved_triples": normalized_resolved_items,
        "unresolved_conflicts": unresolved_items,
        "summary": summary,
    }


def remove_triple_from_doc(doc: Dict[str, Any], triple: List[str]) -> None:


    key = triple_key_from_list(triple)
    doc["extracted_triples"] = [
        item for item in doc.get("extracted_triples", []) if triple_key_from_list(item) != key
    ]
    triple_ontology = doc.get("extracted_triple_ontology") or {}
    if key in triple_ontology:
        del triple_ontology[key]
    doc["extracted_triple_ontology"] = triple_ontology


def add_or_replace_triple_in_doc(
    doc: Dict[str, Any],
    *,
    original_triple: List[str],
    resolved_triple: List[str],
    original_ontology: List[str],
) -> None:


    remove_triple_from_doc(doc, original_triple)
    add_resolved_triple_to_doc(doc, resolved_triple, original_ontology)


def add_resolved_triple_to_doc(
    doc: Dict[str, Any],
    resolved_triple: List[str],
    original_ontology: List[str],
) -> None:

    existing_keys = {triple_key_from_list(item) for item in doc.get("extracted_triples", [])}
    resolved_key = triple_key_from_list(resolved_triple)
    existing_ontology = (doc.get("extracted_triple_ontology") or {}).get(resolved_key)
    if resolved_key in existing_keys and isinstance(existing_ontology, list) and len(existing_ontology) == 3:

        return
    if resolved_key not in existing_keys:
        doc.setdefault("extracted_triples", []).append(list(resolved_triple))

    ontology = list(original_ontology)
    if len(ontology) == 3:
        ontology[1] = str(resolved_triple[1])
    else:
        ontology = ["UNKNOWN", str(resolved_triple[1]), "UNKNOWN"]
    doc.setdefault("extracted_triple_ontology", {})[resolved_key] = ontology


def find_doc_containing_triple(
    docs: List[Dict[str, Any]],
    triple: List[str],
) -> Tuple[Dict[str, Any] | None, List[str] | None]:


    key = triple_key_from_list(triple)
    for doc in docs:
        for existing in doc.get("extracted_triples", []):
            if triple_key_from_list(existing) == key:
                ontology = (doc.get("extracted_triple_ontology") or {}).get(key)
                if not (isinstance(ontology, list) and len(ontology) == 3):
                    ontology = ["UNKNOWN", str(triple[1]), "UNKNOWN"]
                return doc, ontology
    return None, None


def find_docs_containing_triple(
    docs: List[Dict[str, Any]],
    triple: List[str],
) -> List[Tuple[Dict[str, Any], List[str]]]:

    key = triple_key_from_list(triple)
    matches = []
    for doc in docs:
        if not any(triple_key_from_list(existing) == key for existing in doc.get("extracted_triples", [])):
            continue
        ontology = (doc.get("extracted_triple_ontology") or {}).get(key)
        if not (isinstance(ontology, list) and len(ontology) == 3):
            ontology = ["UNKNOWN", str(triple[1]), "UNKNOWN"]
        matches.append((doc, list(ontology)))
    return matches


def apply_resolution_to_docs(
    *,
    accepted_docs: List[Dict[str, Any]],
    new_doc: Dict[str, Any],
    resolution: Dict[str, Any],
) -> None:


    all_docs = accepted_docs + [new_doc]
    resolved_items = resolution.get("resolved_triples", [])
    if not isinstance(resolved_items, list):
        print(
            "Warning: conflict resolution payload has non-list resolved_triples "
            f"for chunk {new_doc.get('idx')}: {type(resolved_items).__name__}"
        )
        return

    plans: Dict[str, Dict[str, Any]] = {}
    for idx, item in enumerate(resolved_items):
        if not isinstance(item, dict):
            print(
                "Warning: skipping malformed resolved_triples item "
                f"for chunk {new_doc.get('idx')}: index={idx}, "
                f"type={type(item).__name__}"
            )
            continue
        original = item.get("original_triple")
        if not (isinstance(original, list) and len(original) == 3):
            continue

        original = [str(x) for x in original]
        decision = str(item.get("resolution", "")).lower()
        if decision not in {"kept", "discarded", "modified"}:
            continue
        resolved = None
        if decision == "modified":
            resolved = item.get("resolved_triple")
            if not (isinstance(resolved, list) and len(resolved) == 3):
                continue
            resolved = [str(x) for x in resolved]
        original_key = triple_key_from_list(original)
        if original_key in plans:
            print(f"Warning: repeated resolution for {original_key}; using the last valid decision.")
        plans[original_key] = {
            "original": original,
            "decision": decision,
            "resolved": resolved,
            "occurrences": find_docs_containing_triple(all_docs, original),
        }

    for plan in plans.values():
        if plan["decision"] in {"discarded", "modified"}:
            for doc, _ in plan["occurrences"]:
                remove_triple_from_doc(doc, plan["original"])
    for plan in plans.values():
        if plan["decision"] == "modified":
            for doc, ontology in plan["occurrences"]:
                add_resolved_triple_to_doc(doc, plan["resolved"], ontology)


def rebuild_relation_chunks_from_openie_docs(
    relation_chunks: List[Dict[str, Any]],
    accepted_docs: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:


    doc_by_idx = {str(doc.get("idx")): doc for doc in accepted_docs}
    rebuilt = []

    for chunk in relation_chunks:
        chunk_id = str(chunk.get("chunk_id"))
        doc = doc_by_idx.get(chunk_id)
        if doc is None:
            rebuilt.append(chunk)
            continue

        new_triples = []
        ontology_map = doc.get("extracted_triple_ontology") or {}
        for triple in doc.get("extracted_triples", []):
            ontology = ontology_map.get(triple_key_from_list(triple), ["UNKNOWN", str(triple[1]), "UNKNOWN"])
            new_triples.append(
                {
                    "head": str(triple[0]),
                    "relation": str(triple[1]),
                    "tail": str(triple[2]),
                    "head_type": str(ontology[0]),
                    "tail_type": str(ontology[2]),
                    "chunk_id": chunk.get("chunk_id"),
                }
            )

        updated = dict(chunk)
        updated["triples"] = new_triples
        updated["num_triples"] = len(new_triples)
        updated["triple_to_schema"] = {
            f"({t['head']}, {t['relation']}, {t['tail']})": f"({t['head_type']}, {t['relation']}, {t['tail_type']})"
            for t in new_triples
        }
        rebuilt.append(updated)

    return rebuilt


def run_one_pdf_incremental_pipeline(
    *,
    pdf_name: str,
    chunks: List[Any],
    output_dir: Path,
    args: argparse.Namespace,
    embedding_model: Any | None = None,
    embedding_config: Any | None = None,
) -> None:


    from memory import ThreeLayerMemory

    output_dir.mkdir(parents=True, exist_ok=True)
    entities_path = output_dir / "output_entities.json"
    memory_path = output_dir / "memory.json"

    print(f"\n=== PDF: {pdf_name} ===")
    print(f"Chunks: {len(chunks)}")

    if args.dry_run:
        for chunk in chunks[: args.preview_limit]:
            print(f"- {chunk.chunk_id} | {chunk.metadata.get('block_type')} | page={chunk.metadata.get('page_idx')}")
        return

    nlp = None
    spacy_module = None
    if args.entity_extraction_mode == "spacy":
        import spacy

        print(f"Loading spaCy model: {args.spacy_model}")
        nlp = spacy.load(args.spacy_model)
        spacy_module = spacy

    entity_chunks: List[Dict[str, Any]] = []
    relation_chunks: List[Dict[str, Any]] = []
    accepted_docs: List[Dict[str, Any]] = []
    conflict_index = empty_conflict_index()
    events: List[Dict[str, Any]] = []
    conflict_logs: List[Dict[str, Any]] = []

    for idx, chunk in enumerate(chunks, 1):
        print(f"[chunk {idx}/{len(chunks)}] {chunk.chunk_id}")

        entity_chunk = extract_entities_from_one_mineru_chunk(
            chunk,
            output_dir=output_dir,
            model_name=args.entity_llm_model or args.llm_model,
            base_url=args.entity_llm_base_url if args.entity_llm_base_url is not None else args.llm_base_url,
            vlm_model_name=args.vlm_model,
            vlm_base_url=args.vlm_base_url,
            temperature=args.entity_temperature,
            max_tokens=args.entity_max_tokens,
            extraction_mode=args.entity_extraction_mode,
            nlp=nlp,
            spacy_module=spacy_module,
        )
        entity_chunks.append(entity_chunk)

        if args.stop_after_entities:
            events.append(
                {
                    "chunk_id": chunk.chunk_id,
                    "num_entities": len(entity_chunk.get("entities", [])),
                    "num_triples": 0,
                    "status": "entities_only",
                }
            )
            continue


        triples = extract_relations_from_one_entity_chunk(
            entity_chunk,
            output_dir=output_dir,
            model_name=args.llm_model,
            base_url=args.llm_base_url,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            max_entities_per_chunk=args.max_entities_per_chunk,
        )
        relation_chunk = relation_chunk_record(
            entity_chunk,
            triples,
            chunk_metadata=summarize_chunk_metadata(dict(chunk.metadata)),
        )
        new_doc = openie_doc_from_relation_chunk(relation_chunk)

        if args.disable_conflict_resolution:
            conflict_detection_result = {
                "mode": "incremental_grouped",
                "status": "disabled",
                "chunk_id": str(new_doc.get("idx")),
                "total_groups": 0,
                "conflicts_by_chunk": [],
                "conflicts": [],
                "conflicting_triple_ids": [],
                "conflicts_detected": 0,
                "total_triples": len(new_doc.get("extracted_triples", [])),
            }
        else:
            conflict_detection_result = detect_conflicts_for_new_doc_groups(
                new_doc=new_doc,
                accepted_docs=accepted_docs,
                output_dir=output_dir,
                args=args,
            )

        chunk_conflict_items = conflict_detection_result.get("conflicts", [])
        if chunk_conflict_items:
            resolution = resolve_conflicts_for_chunk(
                new_doc=new_doc,
                conflict_items=chunk_conflict_items,
                output_dir=output_dir,
                args=args,
            )
            apply_resolution_to_docs(
                accepted_docs=accepted_docs,
                new_doc=new_doc,
                resolution=resolution,
            )
            conflict_logs.append(
                {
                    "chunk_id": str(new_doc.get("idx")),
                    "detection": conflict_detection_result,
                    "resolution": resolution,
                }
            )

        accepted_docs.append(new_doc)
        relation_chunks.append(relation_chunk)
        conflict_index = build_conflict_index(accepted_docs)

        events.append(
            {
                "chunk_id": chunk.chunk_id,
                "num_entities": len(entity_chunk.get("entities", [])),
                "num_extracted_triples": len(triples),
                "num_accepted_triples_after_resolution": len(new_doc.get("extracted_triples", [])),
                "num_conflict_groups": int(conflict_detection_result.get("conflicts_detected", 0)),
                "status": "processed",
            }
        )

    if args.stop_after_entities:
        entity_data = {
            "metadata": {
                "total_chunks": len(entity_chunks),
                "total_entities": sum(len(item.get("entities", [])) for item in entity_chunks),
                "model": args.entity_llm_model or args.llm_model if args.entity_extraction_mode == "llm" else args.spacy_model,
                "entity_extraction_mode": args.entity_extraction_mode,
            },
            "chunks": [entity_chunk_for_output(item) for item in entity_chunks],
        }
        write_json(entity_data, entities_path)
        print(f"Stopped after entity extraction: {entities_path}")
        return

    relation_chunks = rebuild_relation_chunks_from_openie_docs(relation_chunks, accepted_docs)
    openie_data = {
        "metadata": {
            "source": "mineru_incremental_pipeline",
            "total_docs": len(accepted_docs),
            "total_triples": sum(len(doc.get("extracted_triples", [])) for doc in accepted_docs),
        },
        "docs": accepted_docs,
    }

    print(f"[final] Build three-layer memory: {memory_path}")
    memory = ThreeLayerMemory()
    memory.build_from_openie_results(openie_data)
    encode_memory_embeddings(
        memory,
        output_dir,
        embedding_model=embedding_model,
        embedding_config=embedding_config,
    )
    memory.save(str(memory_path))


def _process_one_tree_file(
    json_path: Path,
    root_output_dir: Path,
    args: argparse.Namespace,
) -> Dict[str, Any]:


    result: Dict[str, Any] = {
        "json_path": str(json_path),
        "status": "ok",
        "processed_pdfs": [],
        "skipped_pdfs": [],
        "warning": None,
        "error": None,
    }

    if json_path.parent.name in SKIP_TREE_DIR_NAMES:
        result["status"] = "skipped"
        result["warning"] = (
            f"tree dir {json_path.parent.name} is in the skip list"
        )
        return result

    try:
        pdf_chunks = _load_one_mineru_tree(
            json_path,
            include_title_chunks=args.include_title_chunks,
            include_image_chunks=not args.exclude_image_chunks,
            include_empty_image_chunks=args.include_empty_image_chunks,
        )
    except Exception as exc:
        result["status"] = "warning"
        result["warning"] = f"skip {json_path}: {exc}"
        return result

    if not args.include_reference_chunks:
        pdf_chunks = filter_reference_chunks(pdf_chunks)
    if not pdf_chunks:
        result["status"] = "warning"
        result["warning"] = f"no usable chunks after filtering for {json_path}"
        return result

    grouped = group_chunks_by_pdf(pdf_chunks)
    items = list(grouped.items())
    if not items:
        result["status"] = "warning"
        result["warning"] = f"unable to group chunks for {json_path}"
        return result

    for pdf_name, one_pdf_chunks in items:
        if args.limit_chunks > 0:
            one_pdf_chunks = one_pdf_chunks[: args.limit_chunks]
        pdf_output_dir = root_output_dir / safe_stem(pdf_name)
        memory_path = pdf_output_dir / "memory.json"
        if memory_path.exists():
            result["skipped_pdfs"].append(
                {
                    "pdf_name": pdf_name,
                    "reason": f"found existing memory at {memory_path}",
                }
            )
            continue
        run_one_pdf_incremental_pipeline(
            pdf_name=pdf_name,
            chunks=one_pdf_chunks,
            output_dir=pdf_output_dir,
            args=args,
            embedding_model=None,
            embedding_config=None,
        )
        result["processed_pdfs"].append(
            {
                "pdf_name": pdf_name,
                "num_chunks": len(one_pdf_chunks),
                "output_dir": str(pdf_output_dir),
            }
        )

    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the original README pipeline from MinerU JSON chunks."
    )
    parser.add_argument(
        "mineru_input",
        help="One MinerU tree.json file, or a root directory whose subdirectories each contain a tree.json.",
    )
    parser.add_argument("--output-dir", default="output/mineru_original_pipeline")
    parser.add_argument("--single-memory", action="store_true", help="Put all MinerU chunks into one memory instead of grouping by PDF.")
    parser.add_argument("--include-title-chunks", action="store_true")
    parser.add_argument("--exclude-image-chunks", action="store_true")
    parser.add_argument("--include-empty-image-chunks", action="store_true")
    parser.add_argument(
        "--include-reference-chunks",
        action="store_true",
        help="Keep chunks under References/Bibliography sections. Default excludes them from extraction.",
    )
    parser.add_argument("--entity-extraction-mode", choices=["llm", "spacy"], default="llm", help="How to extract entities. Default uses an LLM.")
    parser.add_argument("--spacy-model", default="en_core_web_trf")
    parser.add_argument("--entity-llm-model", default=DEFAULT_ENTITY_LLM_MODEL, help="Model override for entity extraction. Default is configured in this script.")
    parser.add_argument("--entity-llm-base-url", default=DEFAULT_ENTITY_LLM_BASE_URL, help="Base URL override for entity extraction. Default is configured in this script.")
    parser.add_argument("--vlm-model", default=DEFAULT_VLM_MODEL, help="Model override for image VLM entity extraction. Default is configured in this script.")
    parser.add_argument("--vlm-base-url", default=DEFAULT_VLM_BASE_URL, help="Base URL override for image VLM entity extraction. Default is configured in this script.")
    parser.add_argument("--entity-temperature", type=float, default=0.0)
    parser.add_argument("--entity-max-tokens", type=int, default=4096)
    parser.add_argument("--llm-model", default=DEFAULT_LLM_MODEL)
    parser.add_argument("--llm-base-url", default=DEFAULT_LLM_BASE_URL)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--max-entities-per-chunk", type=int, default=100)
    parser.add_argument("--disable-conflict-resolution", action="store_true", help="Do not call conflict detection/resolution while processing chunks.")
    parser.add_argument("--max-conflict-candidates", type=int, default=20, help="Maximum existing triples sent as candidates for each new triple.")
    parser.add_argument("--save-every", type=int, default=10, help="Save intermediate incremental outputs every N chunks.")
    parser.add_argument("--stop-after-entities", action="store_true", help="Only run MinerU chunk loading and entity extraction.")
    parser.add_argument("--limit-pdfs", type=int, default=0, help="0 means no limit.")
    parser.add_argument("--limit-chunks", type=int, default=0, help="Limit chunks per PDF for small relation-extraction tests. 0 means no limit.")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help=(
            "Number of PDFs/tree files to process in parallel. "
            "1 keeps the original serial behavior."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--preview-limit", type=int, default=5)
    args = parser.parse_args()

    tree_files = discover_mineru_json_files(args.mineru_input)
    if args.limit_pdfs > 0:
        tree_files = tree_files[: args.limit_pdfs]

    root_output_dir = Path(args.output_dir)
    print(f"Discovered {len(tree_files)} tree.json file(s) from: {args.mineru_input}")
    print(f"Output root: {root_output_dir}")

    shared_embedding_config = None
    shared_embedding_model = None
    use_parallel = (
        args.num_workers > 1
        and not args.single_memory
        and not args.dry_run
    )

    if not use_parallel and not args.dry_run and not args.stop_after_entities:
        shared_embedding_config, shared_embedding_model = create_shared_embedding_model()

    try:
        if args.single_memory:
            chunks = load_mineru_chunks(
                args.mineru_input,
                include_title_chunks=args.include_title_chunks,
                include_image_chunks=not args.exclude_image_chunks,
                include_empty_image_chunks=args.include_empty_image_chunks,
            )
            if not args.include_reference_chunks:
                chunks = filter_reference_chunks(chunks)
            if not chunks:
                raise ValueError(f"No MinerU chunks found from: {args.mineru_input}")

            print(f"Loaded {len(chunks)} chunk(s) into one shared memory.")
            if args.limit_chunks > 0:
                chunks = chunks[: args.limit_chunks]
            run_one_pdf_incremental_pipeline(
                pdf_name="all_mineru_documents",
                chunks=chunks,
                output_dir=root_output_dir / "all_mineru_documents",
                args=args,
                embedding_model=shared_embedding_model,
                embedding_config=shared_embedding_config,
            )
            print("\nMinerU original pipeline finished.")
            return

        if use_parallel:
            print(f"Parallel mode enabled: {args.num_workers} worker(s)")
            with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
                future_to_meta = {
                    executor.submit(
                        _process_one_tree_file,
                        json_path,
                        root_output_dir,
                        args,
                    ): (idx, json_path)
                    for idx, json_path in enumerate(tree_files, 1)
                }

                completed = 0
                for future in as_completed(future_to_meta):
                    idx, json_path = future_to_meta[future]
                    completed += 1
                    try:
                        result = future.result()
                    except Exception as exc:
                        print(
                            f"[parallel {completed}/{len(tree_files)}] "
                            f"{json_path} failed: {type(exc).__name__}: {exc}"
                        )
                        continue

                    prefix = f"[parallel {completed}/{len(tree_files)}]"
                    status = result.get("status", "ok")
                    if status == "skipped":
                        print(f"{prefix} skip {json_path}: {result.get('warning')}")
                        continue
                    if status == "warning":
                        print(f"{prefix} warning {json_path}: {result.get('warning')}")
                        continue

                    processed = result.get("processed_pdfs", [])
                    skipped = result.get("skipped_pdfs", [])
                    print(
                        f"{prefix} finished {json_path}: "
                        f"{len(processed)} processed, {len(skipped)} skipped"
                    )
                    for item in processed:
                        print(
                            f"  processed {item['pdf_name']} "
                            f"({item['num_chunks']} chunk(s)) -> {item['output_dir']}"
                        )
                    for item in skipped:
                        print(f"  skipped {item['pdf_name']}: {item['reason']}")
            print("\nMinerU original pipeline finished.")
            return

        for idx, json_path in enumerate(tree_files, 1):
            print(f"\n[pdf {idx}/{len(tree_files)}] loading {json_path}")
            if json_path.parent.name in SKIP_TREE_DIR_NAMES:
                print(
                    f"[pdf {idx}/{len(tree_files)}] skip {json_path}: "
                    f"tree dir {json_path.parent.name} is in the skip list"
                )
                continue
            try:
                pdf_chunks = _load_one_mineru_tree(
                    json_path,
                    include_title_chunks=args.include_title_chunks,
                    include_image_chunks=not args.exclude_image_chunks,
                    include_empty_image_chunks=args.include_empty_image_chunks,
                )
            except Exception as exc:
                print(f"Warning: skip {json_path}: {exc}")
                continue

            if not args.include_reference_chunks:
                pdf_chunks = filter_reference_chunks(pdf_chunks)
            if not pdf_chunks:
                print(f"Warning: no usable chunks after filtering for {json_path}")
                continue

            grouped = group_chunks_by_pdf(pdf_chunks)
            items = list(grouped.items())
            if not items:
                print(f"Warning: unable to group chunks for {json_path}")
                continue

            for pdf_name, one_pdf_chunks in items:
                if args.limit_chunks > 0:
                    one_pdf_chunks = one_pdf_chunks[: args.limit_chunks]
                print(f"[pdf {idx}/{len(tree_files)}] {pdf_name}: {len(one_pdf_chunks)} chunk(s)")
                pdf_output_dir = root_output_dir / safe_stem(pdf_name)
                memory_path = pdf_output_dir / "memory.json"
                if memory_path.exists():
                    print(
                        f"[pdf {idx}/{len(tree_files)}] skip {pdf_name}: "
                        f"found existing memory at {memory_path}"
                    )
                    continue
                run_one_pdf_incremental_pipeline(
                    pdf_name=pdf_name,
                    chunks=one_pdf_chunks,
                    output_dir=pdf_output_dir,
                    args=args,
                    embedding_model=shared_embedding_model,
                    embedding_config=shared_embedding_config,
                )
    finally:
        close_fn = getattr(shared_embedding_model, "close", None)
        if callable(close_fn):
            close_fn()

    print("\nMinerU original pipeline finished.")


if __name__ == "__main__":
    main()
