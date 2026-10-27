from __future__ import annotations

import ast
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


DEFAULT_OPENIE_RESULTS_PATH = ""
DEFAULT_MEMORY_OUTPUT_PATH = ""


@dataclass
class SchemaNode:


    idx: int
    content: str
    frequency: int = 0
    embedding: Optional[List[float]] = None
    entity_indices: List[int] = field(default_factory=list)
    passage_indices: List[int] = field(default_factory=list)
    passage_indices_by_modality: Dict[str, List[int]] = field(default_factory=dict)
    outgoing_relation_indices: List[int] = field(default_factory=list)
    incoming_relation_indices: List[int] = field(default_factory=list)


@dataclass
class EntityNode:


    idx: int
    content: str
    entity_type: str
    schema_idx: int
    embedding: Optional[List[float]] = None
    passage_indices: List[int] = field(default_factory=list)
    passage_indices_by_modality: Dict[str, List[int]] = field(default_factory=dict)
    mention_count: int = 0


@dataclass
class PassageNode:


    idx: int
    chunk_id: str
    content: str
    modality: str
    embedding: Optional[List[float]] = None
    entity_indices: List[int] = field(default_factory=list)
    schema_relation_indices: List[int] = field(default_factory=list)
    image_summary: str = ""
    original_content: str = ""


@dataclass
class SchemaRelationNode:


    idx: int
    content: Tuple[str, str, str]
    source_schema_idx: int
    target_schema_idx: int
    relation: str
    embedding: Optional[List[float]] = None
    passage_indices: List[int] = field(default_factory=list)
    passage_indices_by_modality: Dict[str, List[int]] = field(default_factory=dict)
    entity_pair_indices: List[List[int]] = field(default_factory=list)
    support_count: int = 0


class ThreeLayerMemory:


    def __init__(self):
        self.schema_layer: List[SchemaNode] = []
        self.entity_layer: List[EntityNode] = []
        self.passage_layer: List[PassageNode] = []
        self.schema_relation_layer: List[SchemaRelationNode] = []

        self._schema_to_idx: Dict[str, int] = {}
        self._entity_to_idx: Dict[Tuple[str, str], int] = {}
        self._chunk_id_to_idx: Dict[str, int] = {}
        self._schema_relation_to_idx: Dict[Tuple[str, str, str], int] = {}

    def _get_or_create_schema(self, entity_type: str) -> int:
        entity_type = str(entity_type)
        if entity_type in self._schema_to_idx:
            return self._schema_to_idx[entity_type]

        idx = len(self.schema_layer)
        node = SchemaNode(idx=idx, content=entity_type)
        self.schema_layer.append(node)
        self._schema_to_idx[entity_type] = idx
        return idx

    def _get_or_create_entity(self, entity_name: str, entity_type: str) -> int:
        key = (str(entity_name), str(entity_type))
        if key in self._entity_to_idx:
            return self._entity_to_idx[key]

        schema_idx = self._get_or_create_schema(entity_type)
        idx = len(self.entity_layer)
        node = EntityNode(
            idx=idx,
            content=str(entity_name),
            entity_type=str(entity_type),
            schema_idx=schema_idx,
        )
        self.entity_layer.append(node)
        self._entity_to_idx[key] = idx
        return idx

    def _get_or_create_passage(self, chunk_id: str, passage_text: str, modality: str) -> int:
        chunk_id = str(chunk_id)
        if chunk_id in self._chunk_id_to_idx:
            return self._chunk_id_to_idx[chunk_id]

        idx = len(self.passage_layer)
        node = PassageNode(idx=idx, chunk_id=chunk_id, content=str(passage_text), modality=str(modality))
        self.passage_layer.append(node)
        self._chunk_id_to_idx[chunk_id] = idx
        return idx

    @staticmethod
    def _passage_modality(doc: Dict[str, Any]) -> str:
        chunk_metadata = doc.get("chunk_metadata") or {}
        modality = str(chunk_metadata.get("block_type") or "").strip().lower()
        return modality or "text"

    @staticmethod
    def _append_unique(mapping: Dict[str, List[int]], key: str, value: int) -> None:
        bucket = mapping.setdefault(key, [])
        if value not in bucket:
            bucket.append(value)

    def _get_or_create_schema_relation(
        self,
        head_type: str,
        relation: str,
        tail_type: str,
    ) -> int:
        key = (str(head_type), str(relation), str(tail_type))
        if key in self._schema_relation_to_idx:
            return self._schema_relation_to_idx[key]

        source_schema_idx = self._get_or_create_schema(head_type)
        target_schema_idx = self._get_or_create_schema(tail_type)
        idx = len(self.schema_relation_layer)
        node = SchemaRelationNode(
            idx=idx,
            content=key,
            source_schema_idx=source_schema_idx,
            target_schema_idx=target_schema_idx,
            relation=str(relation),
        )
        self.schema_relation_layer.append(node)
        self._schema_relation_to_idx[key] = idx

        if idx not in self.schema_layer[source_schema_idx].outgoing_relation_indices:
            self.schema_layer[source_schema_idx].outgoing_relation_indices.append(idx)
        if idx not in self.schema_layer[target_schema_idx].incoming_relation_indices:
            self.schema_layer[target_schema_idx].incoming_relation_indices.append(idx)
        return idx

    def build_from_openie_results(self, data: Dict[str, Any]) -> None:


        docs = data.get("docs", [])
        print(f"Building memory from {len(docs)} docs...")

        for doc in docs:
            chunk_id = str(doc.get("idx", ""))
            passage_text = str(doc.get("passage", ""))
            triple_ont_map = doc.get("extracted_triple_ontology") or {}

            if not chunk_id:
                continue

            modality = self._passage_modality(doc)
            passage_idx = self._get_or_create_passage(chunk_id, passage_text, modality)
            self.passage_layer[passage_idx].image_summary = str(doc.get("image_summary") or "")
            self.passage_layer[passage_idx].original_content = str(doc.get("original_content") or "")

            if not isinstance(triple_ont_map, dict) or not triple_ont_map:
                continue

            for triple_key, ontology in triple_ont_map.items():
                try:
                    triple_tuple = ast.literal_eval(triple_key)
                    if not (isinstance(triple_tuple, tuple) and len(triple_tuple) == 3):
                        continue
                except Exception:
                    continue

                if not (isinstance(ontology, list) and len(ontology) == 3):
                    continue

                head, relation, tail = (str(x) for x in triple_tuple)
                head_type, _, tail_type = (str(x) for x in ontology)

                head_entity_idx = self._get_or_create_entity(head, head_type)
                tail_entity_idx = self._get_or_create_entity(tail, tail_type)
                head_schema_idx = self.entity_layer[head_entity_idx].schema_idx
                tail_schema_idx = self.entity_layer[tail_entity_idx].schema_idx

                schema_relation_idx = self._get_or_create_schema_relation(
                    head_type,
                    relation,
                    tail_type,
                )


                for entity_idx in (head_entity_idx, tail_entity_idx):
                    entity_node = self.entity_layer[entity_idx]
                    if passage_idx not in entity_node.passage_indices:
                        entity_node.passage_indices.append(passage_idx)
                    self._append_unique(entity_node.passage_indices_by_modality, modality, passage_idx)
                    entity_node.mention_count += 1

                    if entity_idx not in self.passage_layer[passage_idx].entity_indices:
                        self.passage_layer[passage_idx].entity_indices.append(entity_idx)


                if head_entity_idx not in self.schema_layer[head_schema_idx].entity_indices:
                    self.schema_layer[head_schema_idx].entity_indices.append(head_entity_idx)
                if tail_entity_idx not in self.schema_layer[tail_schema_idx].entity_indices:
                    self.schema_layer[tail_schema_idx].entity_indices.append(tail_entity_idx)


                if passage_idx not in self.schema_layer[head_schema_idx].passage_indices:
                    self.schema_layer[head_schema_idx].passage_indices.append(passage_idx)
                self._append_unique(self.schema_layer[head_schema_idx].passage_indices_by_modality, modality, passage_idx)
                if passage_idx not in self.schema_layer[tail_schema_idx].passage_indices:
                    self.schema_layer[tail_schema_idx].passage_indices.append(passage_idx)
                self._append_unique(self.schema_layer[tail_schema_idx].passage_indices_by_modality, modality, passage_idx)


                relation_node = self.schema_relation_layer[schema_relation_idx]
                if passage_idx not in relation_node.passage_indices:
                    relation_node.passage_indices.append(passage_idx)
                self._append_unique(relation_node.passage_indices_by_modality, modality, passage_idx)
                if schema_relation_idx not in self.passage_layer[passage_idx].schema_relation_indices:
                    self.passage_layer[passage_idx].schema_relation_indices.append(schema_relation_idx)

                pair = [head_entity_idx, tail_entity_idx]
                if pair not in relation_node.entity_pair_indices:
                    relation_node.entity_pair_indices.append(pair)
                relation_node.support_count += 1

        for schema_node in self.schema_layer:
            schema_node.entity_indices = sorted(set(schema_node.entity_indices))
            schema_node.passage_indices = sorted(set(schema_node.passage_indices))
            schema_node.passage_indices_by_modality = {
                key: sorted(set(value))
                for key, value in schema_node.passage_indices_by_modality.items()
            }
            schema_node.outgoing_relation_indices = sorted(set(schema_node.outgoing_relation_indices))
            schema_node.incoming_relation_indices = sorted(set(schema_node.incoming_relation_indices))
            schema_node.frequency = len(schema_node.entity_indices)

        for entity_node in self.entity_layer:
            entity_node.passage_indices = sorted(set(entity_node.passage_indices))
            entity_node.passage_indices_by_modality = {
                key: sorted(set(value))
                for key, value in entity_node.passage_indices_by_modality.items()
            }

        for passage_node in self.passage_layer:
            passage_node.entity_indices = sorted(set(passage_node.entity_indices))
            passage_node.schema_relation_indices = sorted(set(passage_node.schema_relation_indices))

        for relation_node in self.schema_relation_layer:
            relation_node.passage_indices = sorted(set(relation_node.passage_indices))
            relation_node.passage_indices_by_modality = {
                key: sorted(set(value))
                for key, value in relation_node.passage_indices_by_modality.items()
            }

        print("Built memory structure:")
        print(f"  Schema layer:          {len(self.schema_layer)} unique schema nodes")
        print(f"  Entity layer:          {len(self.entity_layer)} unique entity nodes")
        print(f"  Passage layer:         {len(self.passage_layer)} unique chunks")
        print(f"  Schema relation layer: {len(self.schema_relation_layer)} unique schema relations")

    def attach_embeddings(
        self,
        *,
        schema_embeddings: Dict[str, List[float]] | None = None,
        passage_embeddings: Dict[str, List[float]] | None = None,
        schema_relation_embeddings: Dict[Tuple[str, str, str], List[float]] | None = None,
    ) -> None:


        schema_embeddings = schema_embeddings or {}
        passage_embeddings = passage_embeddings or {}
        schema_relation_embeddings = schema_relation_embeddings or {}

        for node in self.schema_layer:
            embedding = schema_embeddings.get(node.content)
            if embedding is not None:
                node.embedding = embedding

        for node in self.passage_layer:
            embedding = passage_embeddings.get(node.chunk_id)
            if embedding is not None:
                node.embedding = embedding

        for node in self.schema_relation_layer:
            embedding = schema_relation_embeddings.get(node.content)
            if embedding is not None:
                node.embedding = embedding

    def get_schema_by_idx(self, idx: int) -> Optional[SchemaNode]:
        if 0 <= idx < len(self.schema_layer):
            return self.schema_layer[idx]
        return None

    def get_entity_by_idx(self, idx: int) -> Optional[EntityNode]:
        if 0 <= idx < len(self.entity_layer):
            return self.entity_layer[idx]
        return None

    def get_passage_by_idx(self, idx: int) -> Optional[PassageNode]:
        if 0 <= idx < len(self.passage_layer):
            return self.passage_layer[idx]
        return None

    def get_schema_relation_by_idx(self, idx: int) -> Optional[SchemaRelationNode]:
        if 0 <= idx < len(self.schema_relation_layer):
            return self.schema_relation_layer[idx]
        return None

    def get_entities_by_schema(self, schema_idx: int) -> List[EntityNode]:
        schema = self.get_schema_by_idx(schema_idx)
        if schema is None:
            return []
        return [self.entity_layer[i] for i in schema.entity_indices]

    def get_passages_by_entity(self, entity_idx: int) -> List[PassageNode]:
        entity = self.get_entity_by_idx(entity_idx)
        if entity is None:
            return []
        return [self.passage_layer[i] for i in entity.passage_indices]

    def get_entities_by_passage(self, passage_idx: int) -> List[EntityNode]:
        passage = self.get_passage_by_idx(passage_idx)
        if passage is None:
            return []
        return [self.entity_layer[i] for i in passage.entity_indices]

    def get_schema_relations_by_schema(self, schema_idx: int) -> List[SchemaRelationNode]:
        schema = self.get_schema_by_idx(schema_idx)
        if schema is None:
            return []
        indices = schema.outgoing_relation_indices + schema.incoming_relation_indices
        return [self.schema_relation_layer[i] for i in sorted(set(indices))]

    def get_passages_by_schema_relation(self, relation_idx: int) -> List[PassageNode]:
        relation = self.get_schema_relation_by_idx(relation_idx)
        if relation is None:
            return []
        return [self.passage_layer[i] for i in relation.passage_indices]

    def to_dict(self, *, include_embeddings: bool = False) -> Dict[str, Any]:
        return {
            "schema_layer": [
                {
                    "idx": node.idx,
                    "content": node.content,
                    "frequency": node.frequency,
                    **({"embedding": node.embedding} if include_embeddings else {}),
                    "entity_indices": node.entity_indices,
                    "passage_indices": node.passage_indices,
                    "passage_indices_by_modality": node.passage_indices_by_modality,
                    "outgoing_relation_indices": node.outgoing_relation_indices,
                    "incoming_relation_indices": node.incoming_relation_indices,
                }
                for node in self.schema_layer
            ],
            "entity_layer": [
                {
                    "idx": node.idx,
                    "content": node.content,
                    "entity_type": node.entity_type,
                    "schema_idx": node.schema_idx,
                    **({"embedding": node.embedding} if include_embeddings else {}),
                    "passage_indices": node.passage_indices,
                    "passage_indices_by_modality": node.passage_indices_by_modality,
                    "mention_count": node.mention_count,
                }
                for node in self.entity_layer
            ],
            "passage_layer": [
                {
                    "idx": node.idx,
                    "chunk_id": node.chunk_id,
                    "content": node.content,
                    "modality": node.modality,
                    **({"image_summary": node.image_summary} if node.image_summary else {}),
                    **({"original_content": node.original_content} if node.original_content else {}),
                    **({"embedding": node.embedding} if include_embeddings else {}),
                    "entity_indices": node.entity_indices,
                    "schema_relation_indices": node.schema_relation_indices,
                }
                for node in self.passage_layer
            ],
            "schema_relation_layer": [
                {
                    "idx": node.idx,
                    "content": list(node.content),
                    "source_schema_idx": node.source_schema_idx,
                    "target_schema_idx": node.target_schema_idx,
                    "relation": node.relation,
                    **({"embedding": node.embedding} if include_embeddings else {}),
                    "passage_indices": node.passage_indices,
                    "passage_indices_by_modality": node.passage_indices_by_modality,
                    "entity_pair_indices": node.entity_pair_indices,
                    "support_count": node.support_count,
                }
                for node in self.schema_relation_layer
            ],
            "stats": {
                "num_schemas": len(self.schema_layer),
                "num_entities": len(self.entity_layer),
                "num_passages": len(self.passage_layer),
                "num_schema_relations": len(self.schema_relation_layer),
            },
        }

    def embeddings_to_dict(self) -> Dict[str, Any]:
        return {
            "schema_layer": [
                {"idx": node.idx, "content": node.content, "embedding": node.embedding}
                for node in self.schema_layer
                if node.embedding is not None
            ],
            "entity_layer": [
                {
                    "idx": node.idx,
                    "content": node.content,
                    "entity_type": node.entity_type,
                    "embedding": node.embedding,
                }
                for node in self.entity_layer
                if node.embedding is not None
            ],
            "passage_layer": [
                {"idx": node.idx, "chunk_id": node.chunk_id, "embedding": node.embedding}
                for node in self.passage_layer
                if node.embedding is not None
            ],
            "schema_relation_layer": [
                {"idx": node.idx, "content": list(node.content), "embedding": node.embedding}
                for node in self.schema_relation_layer
                if node.embedding is not None
            ],
        }

    def _load_embeddings_from_dict(self, data: Dict[str, Any]) -> None:
        for item in data.get("schema_layer", []):
            idx = item.get("idx")
            if isinstance(idx, int) and 0 <= idx < len(self.schema_layer):
                self.schema_layer[idx].embedding = item.get("embedding")

        for item in data.get("entity_layer", []):
            idx = item.get("idx")
            if isinstance(idx, int) and 0 <= idx < len(self.entity_layer):
                self.entity_layer[idx].embedding = item.get("embedding")

        for item in data.get("passage_layer", []):
            idx = item.get("idx")
            if isinstance(idx, int) and 0 <= idx < len(self.passage_layer):
                self.passage_layer[idx].embedding = item.get("embedding")

        for item in data.get("schema_relation_layer", []):
            idx = item.get("idx")
            if isinstance(idx, int) and 0 <= idx < len(self.schema_relation_layer):
                self.schema_relation_layer[idx].embedding = item.get("embedding")

    def _has_any_embeddings(self) -> bool:
        return any(node.embedding is not None for node in self.schema_layer) or \
            any(node.embedding is not None for node in self.entity_layer) or \
            any(node.embedding is not None for node in self.passage_layer) or \
            any(node.embedding is not None for node in self.schema_relation_layer)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ThreeLayerMemory":
        memory = cls()

        for item in data.get("schema_layer", []):
            node = SchemaNode(
                idx=item["idx"],
                content=item["content"],
                frequency=item.get("frequency", 0),
                embedding=item.get("embedding"),
                entity_indices=item.get("entity_indices", []),
                passage_indices=item.get("passage_indices", []),
                passage_indices_by_modality=item.get("passage_indices_by_modality", {}),
                outgoing_relation_indices=item.get("outgoing_relation_indices", []),
                incoming_relation_indices=item.get("incoming_relation_indices", []),
            )
            memory.schema_layer.append(node)
            memory._schema_to_idx[node.content] = node.idx

        for item in data.get("entity_layer", []):
            node = EntityNode(
                idx=item["idx"],
                content=item["content"],
                entity_type=item["entity_type"],
                schema_idx=item["schema_idx"],
                embedding=item.get("embedding"),
                passage_indices=item.get("passage_indices", []),
                passage_indices_by_modality=item.get("passage_indices_by_modality", {}),
                mention_count=item.get("mention_count", 0),
            )
            memory.entity_layer.append(node)
            memory._entity_to_idx[(node.content, node.entity_type)] = node.idx

        for item in data.get("passage_layer", []):
            node = PassageNode(
                idx=item["idx"],
                chunk_id=item["chunk_id"],
                content=item["content"],
                modality=item.get("modality", "text"),
                embedding=item.get("embedding"),
                entity_indices=item.get("entity_indices", []),
                schema_relation_indices=item.get("schema_relation_indices", []),
                image_summary=item.get("image_summary", ""),
                original_content=item.get("original_content", ""),
            )
            memory.passage_layer.append(node)
            memory._chunk_id_to_idx[node.chunk_id] = node.idx

        for item in data.get("schema_relation_layer", []):
            node = SchemaRelationNode(
                idx=item["idx"],
                content=tuple(item["content"]),
                source_schema_idx=item["source_schema_idx"],
                target_schema_idx=item["target_schema_idx"],
                relation=item["relation"],
                embedding=item.get("embedding"),
                passage_indices=item.get("passage_indices", []),
                passage_indices_by_modality=item.get("passage_indices_by_modality", {}),
                entity_pair_indices=item.get("entity_pair_indices", []),
                support_count=item.get("support_count", 0),
            )
            memory.schema_relation_layer.append(node)
            memory._schema_relation_to_idx[node.content] = node.idx

        return memory

    def save(self, path: str, *, include_embeddings: bool = False, embeddings_path: Optional[str] = None) -> None:
        print(f"Saving memory structure to: {path}")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(include_embeddings=include_embeddings), f, ensure_ascii=False, indent=2)

        if not include_embeddings and self._has_any_embeddings():
            if embeddings_path is None:
                if path.endswith(".json"):
                    embeddings_path = path[:-5] + "_embeddings.json"
                else:
                    embeddings_path = path + "_embeddings.json"
            print(f"Saving memory embeddings to: {embeddings_path}")
            with open(embeddings_path, "w", encoding="utf-8") as f:
                json.dump(self.embeddings_to_dict(), f, ensure_ascii=False, indent=2)
        print("Saved.")

    @classmethod
    def load(cls, path: str, *, embeddings_path: Optional[str] = None) -> "ThreeLayerMemory":
        print(f"Loading memory structure from: {path}")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        memory = cls.from_dict(data)
        if embeddings_path is None:
            if path.endswith(".json"):
                embeddings_path = path[:-5] + "_embeddings.json"
            else:
                embeddings_path = path + "_embeddings.json"
        try:
            with open(embeddings_path, "r", encoding="utf-8") as f:
                embedding_data = json.load(f)
            memory._load_embeddings_from_dict(embedding_data)
            print(f"Loaded embeddings from: {embeddings_path}")
        except FileNotFoundError:
            pass
        print(
            f"Loaded: {len(memory.schema_layer)} schemas, "
            f"{len(memory.entity_layer)} entities, "
            f"{len(memory.passage_layer)} passages, "
            f"{len(memory.schema_relation_layer)} schema relations"
        )
        return memory

    def print_summary(self) -> None:
        print("\n===== Three-Layer Memory Summary =====")
        print(f"Schema layer:          {len(self.schema_layer)}")
        print(f"Entity layer:          {len(self.entity_layer)}")
        print(f"Passage layer:         {len(self.passage_layer)}")
        print(f"Schema relation layer: {len(self.schema_relation_layer)}")

        if self.schema_layer:
            freqs = [s.frequency for s in self.schema_layer]
            print("\nSchema -> Entity count stats:")
            print(f"  Min: {min(freqs)}, Max: {max(freqs)}, Avg: {sum(freqs)/len(freqs):.2f}")

        if self.entity_layer:
            passage_counts = [len(e.passage_indices) for e in self.entity_layer]
            print("\nEntity -> Passage count stats:")
            print(
                f"  Min: {min(passage_counts)}, Max: {max(passage_counts)}, "
                f"Avg: {sum(passage_counts)/len(passage_counts):.2f}"
            )

        if self.schema_relation_layer:
            support_counts = [r.support_count for r in self.schema_relation_layer]
            print("\nSchemaRelation support stats:")
            print(
                f"  Min: {min(support_counts)}, Max: {max(support_counts)}, "
                f"Avg: {sum(support_counts)/len(support_counts):.2f}"
            )

    def print_sample(self, n: int = 3) -> None:
        print(f"\n===== Sample Data (top {n}) =====")

        print("\n--- Schema Layer ---")
        for schema in self.schema_layer[:n]:
            print(
                f"  [{schema.idx}] {schema.content}, "
                f"entities={schema.entity_indices[:5]}, "
                f"passages={schema.passage_indices[:5]}, "
                f"by_modality={schema.passage_indices_by_modality}"
            )

        print("\n--- Entity Layer ---")
        for entity in self.entity_layer[:n]:
            print(
                f"  [{entity.idx}] {entity.content} ({entity.entity_type}), "
                f"schema={entity.schema_idx}, passages={entity.passage_indices[:5]}, "
                f"by_modality={entity.passage_indices_by_modality}"
            )

        print("\n--- Passage Layer ---")
        for passage in self.passage_layer[:n]:
            content_preview = passage.content[:80] + "..." if len(passage.content) > 80 else passage.content
            print(f"  [{passage.idx}] {passage.chunk_id}")
            print(f"       content: {content_preview}")
            print(f"       modality: {passage.modality}")
            print(f"       entities={passage.entity_indices[:5]}")
            print(f"       schema_relations={passage.schema_relation_indices[:5]}")

        print("\n--- Schema Relation Layer ---")
        for relation in self.schema_relation_layer[:n]:
            print(
                f"  [{relation.idx}] {relation.content}, "
                f"passages={relation.passage_indices[:5]}, "
                f"by_modality={relation.passage_indices_by_modality}, "
                f"entity_pairs={relation.entity_pair_indices[:3]}"
            )


def load_openie_results(path: str) -> Dict[str, Any]:
    print(f"Loading OpenIE results from: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"Loaded {len(data.get('docs', []))} docs")
    return data


def main():
    input_path = DEFAULT_OPENIE_RESULTS_PATH
    output_path = DEFAULT_MEMORY_OUTPUT_PATH

    data = load_openie_results(input_path)

    memory = ThreeLayerMemory()
    memory.build_from_openie_results(data)
    memory.print_summary()
    memory.print_sample(n=3)
    memory.save(output_path)

    print("\n--- Verification: reload and check ---")
    loaded_memory = ThreeLayerMemory.load(output_path)
    loaded_memory.print_summary()


if __name__ == "__main__":
    main()
