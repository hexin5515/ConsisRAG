import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


@dataclass
class MinerUChunk:
    chunk_id: str
    content: str
    metadata: Dict[str, Any]
    image_path: Optional[str] = None


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).replace("\r\n", "\n").replace("\r", "\n")
    lines = [" ".join(line.split()) for line in text.split("\n")]
    return "\n".join(line for line in lines if line).strip()


def _node_kind(node: Dict[str, Any]) -> str:
    node_type = str(node.get("type") or "")
    return node_type.split(".")[-1].upper()


def _block_type(node: Dict[str, Any]) -> str:


    meta = node.get("meta_info") or {}
    pdf_para_block = meta.get("pdf_para_block") or {}
    block_type = pdf_para_block.get("type")
    if isinstance(block_type, str) and block_type.strip():
        return block_type.strip().upper()
    return _node_kind(node)


def _collect_span_text(obj: Any) -> List[str]:


    texts: List[str] = []

    if isinstance(obj, dict):
        content = obj.get("content")
        span_type = obj.get("type")
        if isinstance(content, str) and content.strip():
            if span_type in {None, "text", "inline_equation", "interline_equation"}:
                texts.append(content)

        for key in ("blocks", "lines", "spans"):
            value = obj.get(key)
            if isinstance(value, list):
                for item in value:
                    texts.extend(_collect_span_text(item))

    elif isinstance(obj, list):
        for item in obj:
            texts.extend(_collect_span_text(item))

    return texts


def _pdf_para_block_text(meta: Dict[str, Any]) -> str:

    pdf_para_block = meta.get("pdf_para_block") or {}
    texts = _collect_span_text(pdf_para_block)
    return _clean_text("\n".join(texts))


def _root_meta(nodes: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    for node in nodes:
        if _node_kind(node) == "ROOT":
            return node.get("meta_info") or {}
    return {}


def _children_by_parent(nodes: Iterable[Dict[str, Any]]) -> Dict[Any, List[Any]]:
    children: Dict[Any, List[Any]] = {}
    for node in nodes:
        parent_id = node.get("parent_id")
        node_id = node.get("index_id")
        children.setdefault(parent_id, []).append(node_id)
    return children


def _node_text(node: Dict[str, Any]) -> str:
    meta = node.get("meta_info") or {}
    kind = _block_type(node)
    primary_text = _clean_text(meta.get("content"))
    fallback_text = ""

    if kind == "TABLE":
        parts = [
            primary_text,
            _clean_text(meta.get("caption")),
            _clean_text(meta.get("footnote")),
        ]
        if not primary_text:
            fallback_text = _pdf_para_block_text(meta)
            parts.append(fallback_text)
    elif kind == "IMAGE":
        parts = [
            primary_text,
            _clean_text(meta.get("caption")),
            _clean_text(meta.get("footnote")),
        ]
        if not primary_text:
            fallback_text = _pdf_para_block_text(meta)
            parts.append(fallback_text)
    else:
        parts = [primary_text]
        if not primary_text:
            fallback_text = _pdf_para_block_text(meta)
            parts.append(fallback_text)

    deduped_parts: List[str] = []
    seen = set()
    for part in parts:
        if not part or part in seen:
            continue
        seen.add(part)
        deduped_parts.append(part)

    return "\n\n".join(deduped_parts)


def _title_path(node: Dict[str, Any], nodes_by_id: Dict[Any, Dict[str, Any]]) -> List[str]:
    titles = []
    parent_id = node.get("parent_id")

    while parent_id is not None and parent_id in nodes_by_id:
        parent = nodes_by_id[parent_id]
        if _node_kind(parent) == "TITLE":
            title_text = _node_text(parent)
            if title_text:
                titles.append(title_text)
        parent_id = parent.get("parent_id")

    titles.reverse()
    return titles


def _node_ref(
    node: Optional[Dict[str, Any]],
    children_by_parent: Dict[Any, List[Any]],
) -> Optional[Dict[str, Any]]:
    if node is None:
        return None

    meta = node.get("meta_info") or {}
    node_id = node.get("index_id")
    kind = _block_type(node)

    return {
        "index_id": node_id,
        "parent_id": node.get("parent_id"),
        "child_ids": children_by_parent.get(node_id, []),
        "node_type": node.get("type"),
        "block_type": kind.lower(),
        "page_idx": meta.get("page_idx"),
        "content": _node_text(node),
        "title_level": meta.get("title_level"),
        "img_path": meta.get("img_path"),
        "caption": meta.get("caption"),
        "footnote": meta.get("footnote"),
        "table_body": meta.get("table_body"),
    }


def _ancestor_nodes(
    node: Dict[str, Any],
    nodes_by_id: Dict[Any, Dict[str, Any]],
    children_by_parent: Dict[Any, List[Any]],
) -> List[Dict[str, Any]]:
    ancestors = []
    parent_id = node.get("parent_id")

    while parent_id is not None and parent_id in nodes_by_id:
        parent = nodes_by_id[parent_id]
        parent_ref = _node_ref(parent, children_by_parent)
        if parent_ref is not None:
            ancestors.append(parent_ref)
        parent_id = parent.get("parent_id")

    ancestors.reverse()
    return ancestors


def _format_chunk_text(
    *,
    text: str,
    title_path: List[str],
) -> str:


    if title_path:
        return "Section: " + " > ".join(title_path) + "\n\n" + text
    return text


def load_mineru_json_chunks(
    json_path: str,
    *,
    include_title_chunks: bool = False,
    include_image_chunks: bool = True,
    include_empty_image_chunks: bool = False,
) -> List[MinerUChunk]:


    path = Path(json_path)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    nodes = data.get("nodes", [])
    nodes_by_id = {node.get("index_id"): node for node in nodes}
    children_by_parent = _children_by_parent(nodes)
    root = _root_meta(nodes)
    source_file = root.get("file_name") or path.name
    source_path = root.get("file_path")

    chunks: List[MinerUChunk] = []
    allowed_types = {"TEXT", "TABLE", "LIST", "EQUATION"}
    if include_title_chunks:
        allowed_types.add("TITLE")
    if include_image_chunks:
        allowed_types.add("IMAGE")

    for node in nodes:
        kind = _block_type(node)
        if kind not in allowed_types:
            continue

        meta = node.get("meta_info") or {}
        text = _node_text(node)
        if not text and not (kind == "IMAGE" and (include_empty_image_chunks or meta.get("img_path"))):
            continue

        title_path = _title_path(node, nodes_by_id)
        if kind == "TITLE":
            title_text = _node_text(node)
            if title_text:
                title_path = title_path + [title_text]

        index_id = node.get("index_id")
        page_idx = meta.get("page_idx")
        image_path = meta.get("img_path")
        parent_id = node.get("parent_id")
        child_ids = children_by_parent.get(index_id, [])
        sibling_ids = [
            child_id
            for child_id in children_by_parent.get(parent_id, [])
            if child_id != index_id
        ]
        current_node = _node_ref(node, children_by_parent)
        parent_node = _node_ref(nodes_by_id.get(parent_id), children_by_parent)
        child_nodes = [
            _node_ref(nodes_by_id.get(child_id), children_by_parent)
            for child_id in child_ids
        ]
        sibling_nodes = [
            _node_ref(nodes_by_id.get(sibling_id), children_by_parent)
            for sibling_id in sibling_ids
        ]
        ancestor_nodes = _ancestor_nodes(node, nodes_by_id, children_by_parent)
        chunk_id = f"{Path(source_file).stem}::node_{index_id}"

        if kind == "IMAGE":
            image_caption = _clean_text(meta.get("caption"))
            image_footnote = _clean_text(meta.get("footnote"))
            image_content = _clean_text(meta.get("content"))
            image_parts = [part for part in [image_caption, image_footnote] if part]
            chunk_body = "\n\n".join(image_parts).strip() or image_content or "[Image caption missing]"
        else:
            chunk_body = text or "[Empty chunk text]"

        chunk_text = _format_chunk_text(
            text=chunk_body,
            title_path=title_path,
        )

        chunks.append(
            MinerUChunk(
                chunk_id=chunk_id,
                content=chunk_text,
                metadata={
                    "source_file": source_file,
                    "source_path": source_path,
                    "index_id": index_id,
                    "parent_id": parent_id,
                    "child_ids": child_ids,
                    "sibling_ids": sibling_ids,
                    "ancestor_ids": [item["index_id"] for item in ancestor_nodes],
                    "page_idx": page_idx,
                    "node_type": node.get("type"),
                    "block_type": kind.lower(),
                    "title_path": title_path,
                    "img_path": image_path,
                    "current_node": current_node,
                    "parent_node": parent_node,
                    "child_nodes": [item for item in child_nodes if item is not None],
                    "sibling_nodes": [item for item in sibling_nodes if item is not None],
                    "ancestor_nodes": ancestor_nodes,
                },
                image_path=image_path if kind == "IMAGE" else None,
            )
        )

    return chunks


def load_mineru_json_dir_chunks(json_dir: str, **kwargs: Any) -> List[MinerUChunk]:
    chunks: List[MinerUChunk] = []
    for path in sorted(Path(json_dir).glob("*.json")):
        chunks.extend(load_mineru_json_chunks(str(path), **kwargs))
    return chunks


def load_mineru_json_dir_as_docs(json_dir: str, **kwargs: Any) -> List[str]:
    return [chunk.content for chunk in load_mineru_json_dir_chunks(json_dir, **kwargs)]
