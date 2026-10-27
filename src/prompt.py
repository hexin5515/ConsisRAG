#!/usr/bin/env python3

PROMPT={}

RELATION_EXTRACTION_SYSTEM_PROMPT = """You are an expert in knowledge graph construction and relation extraction.
Your task is to extract relations between given entities from the provided text.

## Instructions:
1. Analyze the text carefully and identify meaningful relations between the provided entities.
2. Only extract relations that are explicitly stated or strongly implied in the text.
3. Use lowercase and underscores for relation names (e.g., "works_for", "located_in").
4. Ensure relations are directional: (subject_entity, relation, object_entity).
5. Do not invent relations that are not supported by the text.
6. Return only high-confidence relations. Skip weak, generic, repetitive, or speculative relations.
7. Do not repeat the same relation triple. If the same (head, relation, tail) appears multiple times, return it only once.
8. Prefer specific factual relations such as affiliation, citation, authorship, part_of, type_of, evaluates, uses, improves, inspired_by, or related_to only when the text strongly supports them.
9. Do not create placeholder entities, inferred aliases, or broad summary relations that are not directly grounded in the text.
10. Return at most 20 distinct relations, ordered from most explicit/high-confidence to least.

## Output Format:
Return exactly one valid JSON object with this structure:
{
  "relations": [
    {
      "head": "the subject entity (exactly as provided)",
      "relation": "the relation type",
      "tail": "the object entity (exactly as provided)",
      "head_type": "the entity type/label of the head entity (exactly as provided)",
      "tail_type": "the entity type/label of the tail entity (exactly as provided)"
    }
  ]
}

The "relations" value must always be a JSON array.
If multiple valid relations exist, include all of them in the array.
If no valid relations can be extracted, return {"relations": []}.
Before returning, deduplicate the array so that each relation triple appears only once.

Example input:
## Text:
Apple Inc. was founded by Steve Jobs in 1976.

## Entities to consider:
- Apple Inc. (ORG)
- Steve Jobs (PERSON)
- 1976 (DATE)

Example output:
```json
{
  "relations": [
    {
      "head": "Apple Inc.",
      "relation": "founded_in",
      "tail": "1976",
      "head_type": "ORG",
      "tail_type": "DATE"
    },
    {
      "head": "Steve Jobs",
      "relation": "founded",
      "tail": "Apple Inc.",
      "head_type": "PERSON",
      "tail_type": "ORG"
    }
  ]
}
```
Do not return markdown, explanations, or any keys other than "relations".
"""


RELATION_EXTRACTION_USER_PROMPT_TEMPLATE = """## Text:
{text}

## Entities to consider:
{entities}

Please extract all meaningful relations between these entities based on the text above.
Only include high-confidence, non-duplicate relations.
Return at most 20 distinct relations.
Return your answer as a JSON object with a "relations" array following the specified format."""


PROMPT['conflict_detection'] = {
    "system": """You are an expert fact checker. Given one candidate conflict group of triples, determine whether any triples in the group truly conflict with each other.

Your task:
Analyze the triples within the group together, compare them pairwise where useful, and detect real conflicts. Classify conflicts into three types:
(1) mutual conflict (mutual exclusivity / one-to-one relations)
(2) Temporal conflict (time-dependent facts; conflicts arise when time scopes overlap or are missing)
(3) Granularity conflict (different levels of specificity; may be compatible via containment/hypernymy)

Definitions and rules:

1) mutual conflict (type = "mutual")
A mutual conflict happens when:
- Same subject and predicate, but different objects, AND the predicate is one-to-one / mutually exclusive.
  Example: (X, birthplace, Shanghai) vs (X, birthplace, Beijing)
- Or cyclic/contradictory relational structure that cannot both be true under common-sense constraints.
  Example: (A, father, B) vs (B, father, A)

2) Temporal conflict (type = "temporal")
A temporal conflict happens when:
- The predicate describes a role/state that can change over time and is typically unique at a given moment
  (e.g., president/CEO/champion/current location).
- If both triples claim different objects for the same subject-predicate:
  - If explicit time scopes exist and overlap → hard temporal conflict.
  - If time scopes exist and do NOT overlap → not a conflict .
  - If time scopes are missing but the predicate is time-variant and moment-unique → suspected temporal conflict
    (ask for time ranges; do NOT assert a hard conflict without time info).

3) Granularity conflict (type = "granularity")    
A granularity conflict happens when:
- Triples differ due to specificity/abstraction level.
  Example: (X, birthplace, Shanghai) vs (X, birthplace, China)
- If one object is a parent/superset/contains the other (hypernym/meronym/administrative containment),
  then it is usually compatible → classify as "granularity".
- If objects are incompatible (cannot contain each other and cannot both be true) → Logical conflict.

Important instructions:
- Only judge conflicts among the triples inside the provided group.
- Use the group type and possible conflict reason as hints, not as proof of conflict.
- Do not mark a conflict unless it is clearly supported by the triples.
- If there is no real conflict in the group, return has_conflict=false.

Output MUST be a valid JSON object following the required schema.""",

    "user": """Analyze the following candidate conflict group.

Group ID:
${group_id}

Group Type:
${group_type}

Group Scope:
${group_scope}

Possible Conflict Reason:
${group_reason}

Triples in Group:
${group_triples}

Output a JSON object with the following structure:
{
  "has_conflict": true/false,
  "conflicts": [
    {
      "triple1_id": "id1",
      "triple1": ["head", "relation", "tail"],
      "triple2_id": "id2",
      "triple2": ["head", "relation", "tail"],
      "conflict_type": "mutual|temporal|granularity",
      "conflict_reason": "brief explanation of why these triples conflict"
    }
  ],
  "conflicting_triple_ids": ["id1", "id2", ...]
}

Analyze the whole group. If conflicts exist, list all conflicting triple pairs.
If has_conflict is false, return empty arrays for conflicts and conflicting_triple_ids.

JSON payload:
"""
}


PROMPT['conflict_resolution'] = {
    "system": """You are an expert knowledge graph curator. Given a set of conflicting triples and their source passages, your task is to resolve the conflicts and produce corrected triples.

A triple ID identifies one shared fact that may have multiple supporting chunks.
Return one decision per original conflicting fact; the decision applies to all of
that fact's occurrences, not just one source chunk. Consider all supplied sources,
including additional entity-level evidence. Additional evidence is context only;
do not resolve unrelated facts merely because they appear in a supporting chunk.

Conflict Resolution Strategies:

1. Mutual Conflict (type = "mutual"):
   - These are contradictory claims about the same entity (e.g., same subject-predicate but different objects)
   - Resolution: Analyze the source passages to determine which triple is more accurate
   - Keep only the CORRECT triple, discard the incorrect one(s)
   - If both seem equally valid based on context, prefer the one with more specific/credible source

2. Temporal Conflict (type = "temporal"):
   - These are time-dependent facts where time scopes overlap or are missing
   - Resolution: Add time information to the relation to distinguish the facts
   - Modify the predicate to include time context (e.g., "was president of [2000-2005]" vs "was president of [2005-2010]")
   - If time info is not in sources, note it as "temporal_conflict_unresolved"

3. Granularity Conflict (type = "granularity"):
   - These are facts at different levels of specificity (e.g., "born in Shanghai" vs "born in China")
   - Resolution: Add granularity description to the relation to clarify the scope
   - Modify the predicate to include granularity context (e.g., "was born in [city: Shanghai]" vs "was born in [country: China]")
   - Both can be kept if they are compatible (containment relationship)

Output MUST be a valid JSON object following the required schema.""",

    "user": """Resolve the following conflicting triples using their source passages.

Conflicting Triples and Their Sources:
${conflicting_triples_with_sources}

Output a JSON object with the following structure:
{
  "resolved_triples": [
    {
      "original_triple": ["head", "relation", "tail"],
      "triple_id": "fact_id",
      "conflict_type": "mutual|temporal|granularity",
      "resolution": "kept|discarded|modified",
      "resolved_triple": ["head", "modified_relation", "tail"] or null if discarded,
      "reason": "explanation of why this resolution was chosen"
    }
  ],
  "unresolved_conflicts": [
    {
      "triple_ids": ["id1", "id2"],
      "reason": "reason why conflict could not be resolved"
    }
  ],
  "summary": "brief summary of how conflicts were resolved"
}

For each conflicting triple:
- If resolution is "kept": Keep the triple as is (it's correct)
- If resolution is "discarded": The triple is incorrect, set resolved_triple to null
- If resolution is "modified": Provide the modified triple with time/granularity info in the relation

JSON payload:
"""
}
