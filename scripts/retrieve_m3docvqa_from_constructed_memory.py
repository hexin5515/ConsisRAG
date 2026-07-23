#!/usr/bin/env python3
"""
Run retrieval + answer generation on M3DocVQA/M3DocRAG memories, and evaluate
with BookRAG-compatible EM/F1 logic.

Design goals:
1. Reuse the current RetrievalRuntime / AnswerGenerator so retrieval behavior
   stays aligned with the latest HVM-GraphRAG pipeline.
2. Mirror BookRAG's M3DocRAG evaluation:
   - use a judge LLM to extract a normalized final answer
   - compute list-style EM/F1 the same way as BookRAG
3. Keep the output structure, progress display, and runtime/token accounting
   close to the existing Qasper/MMLong scripts so later analysis feels uniform.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from retrieve_from_constructed_memory import (  # type: ignore
    DEFAULT_ANSWER_API_KEY,
    DEFAULT_ANSWER_BASE_URL,
    DEFAULT_ANSWER_MODEL,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_JUDGE_API_KEY,
    DEFAULT_JUDGE_BASE_URL,
    DEFAULT_JUDGE_MODEL,
    DEFAULT_PLANNER_API_KEY,
    DEFAULT_PLANNER_BASE_URL,
    DEFAULT_PLANNER_MODEL,
    AnswerGenerator,
    BookRAGStyleTaskPlanner,
    RetrievalRuntime,
    _accumulate_token_usage,
    _discover_memory_paths,
    _empty_timing_breakdown,
    _empty_token_usage,
    run_selected_strategy_with_optional_bookrag_planning,
)
from evaluation.m3doc_bookrag_eval import (
    BookRAGM3DocAnswerExtractor,
    list_em,
    list_f1,
)


DEFAULT_M3DOC_JSON = "data/datasets/m3docvqa/m3docrag_office.json"
DEFAULT_M3DOC_MINERU_ROOT = "data/datasets/m3docvqa/process_documents"


def _load_m3doc_samples(m3doc_json_path: str) -> List[Dict[str, Any]]:
    with Path(m3doc_json_path).open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list in M3DocVQA file: {m3doc_json_path}")
    return data


def _safe_name(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return "unknown"
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in text)


def _progress_bar(current: int, total: int, width: int = 28) -> str:
    if total <= 0:
        return "[" + ("-" * width) + "]"
    filled = int(width * current / total)
    filled = min(max(filled, 0), width)
    return "[" + ("#" * filled) + ("-" * (width - filled)) + "]"


def _group_m3doc_by_doc(samples: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    grouped: Dict[str, Dict[str, Any]] = {}
    for item in samples:
        doc_id = str(item.get("doc_id", "")).strip()
        if not doc_id:
            doc_id = Path(str(item.get("doc_path", "")).strip()).stem
        if not doc_id:
            continue
        slot = grouped.setdefault(
            doc_id,
            {
                "doc_id": doc_id,
                "doc_uuid": str(item.get("doc_uuid", "")).strip(),
                "doc_path": str(item.get("doc_path", "")).strip(),
                "questions": [],
            },
        )
        slot["questions"].append(item)
    return grouped


def _retrieve_with_runtime(
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
) -> Dict[str, Any]:
    if strategy == "dense":
        return runtime.retrieve_dense_baseline(
            query,
            passage_top_k=passage_top_k,
            modalities=modalities,
            preview_chars=preview_chars,
        )
    if strategy == "dense_bucketed":
        return runtime.retrieve_dense_bucketed(
            query,
            passage_top_k=passage_top_k,
            modalities=modalities,
            preview_chars=preview_chars,
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
        )
    return runtime.retrieve_graph_guided(
        query,
        schema_top_k=schema_top_k,
        schema_relation_top_k=schema_relation_top_k,
        passage_top_k=passage_top_k,
        modalities=modalities,
        preview_chars=preview_chars,
    )


def _mean_or_zero(values: Sequence[float]) -> float:
    return float(mean(values)) if values else 0.0


def _build_summary(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    em_values = [float(item.get("em", 0.0)) for item in records]
    f1_values = [float(item.get("f1", 0.0)) for item in records]
    llm_values = [float(item.get("llm_score", 0.0)) for item in records]
    return {
        "total_samples": len(records),
        "avg_em": round(_mean_or_zero(em_values), 6),
        "avg_f1": round(_mean_or_zero(f1_values), 6),
        "avg_llm_score": round(_mean_or_zero(llm_values), 6),
    }


def _judge_one_output(
    *,
    judge_extractor: BookRAGM3DocAnswerExtractor,
    question: str,
    output: str,
    gold_answer: str,
) -> Dict[str, Any]:
    judge_start = time.perf_counter()
    extracted_res, pred_ans, pred_format, llm_score = judge_extractor.extract(
        question,
        output,
        gold_answer,
    )
    em = list_em(pred_ans, gold_answer)
    f1 = list_f1(pred_ans, gold_answer)
    judge_time = time.perf_counter() - judge_start
    return {
        "output": output,
        "extracted_res": extracted_res,
        "pred": pred_ans,
        "pred_format": pred_format,
        "llm_score": llm_score,
        "em": em,
        "f1": f1,
        "judge_time_sec": judge_time,
    }


def _mode_records(records: Sequence[Dict[str, Any]], mode_key: str) -> List[Dict[str, Any]]:
    extracted: List[Dict[str, Any]] = []
    for item in records:
        mode_payload = dict(item.get(mode_key, {}))
        mode_payload["gold_answer"] = item.get("gold_answer", "")
        extracted.append(mode_payload)
    return extracted


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate HVM-GraphRAG retrieval pipelines on M3DocVQA with BookRAG-compatible EM/F1.",
    )
    parser.add_argument("memory_path", help="One memory.json file or a root directory containing many memory.json files.")
    parser.add_argument("--m3doc-json", default=DEFAULT_M3DOC_JSON)
    parser.add_argument("--output-path", default="output/m3docvqa_eval_results.json")
    parser.add_argument("--per-doc-output-dir", default="output/per_doc_m3docvqa_eval")
    parser.add_argument(
        "--retrieval-strategy",
        choices=[
            "graph",
            "dense",
            "dense_bucketed",
            "graph_plus_dense",
            "graph_plus_dense_no_bucket",
            "graph_dense_fused",
            "graph_dense_fused_routed",
        ],
        default="graph_plus_dense",
    )
    parser.add_argument("--schema-top-k", type=int, default=5)
    parser.add_argument("--schema-relation-top-k", type=int, default=5)
    parser.add_argument("--passage-top-k", type=int, default=10)
    parser.add_argument("--graph-plus-dense-schema-score-threshold", type=float, default=0.615)
    parser.add_argument("--graph-plus-dense-supplement-top-k", type=int, default=2)
    parser.add_argument("--modalities", nargs="*", default=[])
    parser.add_argument("--bookrag-query-filtering", action="store_true")
    parser.add_argument("--planner-model", default=DEFAULT_PLANNER_MODEL)
    parser.add_argument("--planner-base-url", default=DEFAULT_PLANNER_BASE_URL)
    parser.add_argument("--planner-api-key", default=DEFAULT_PLANNER_API_KEY)
    parser.add_argument("--planner-max-tokens", type=int, default=1024)
    parser.add_argument("--planner-temperature", type=float, default=0.0)
    parser.add_argument("--preview-chars", type=int, default=240)
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--mineru-root", default=DEFAULT_M3DOC_MINERU_ROOT)
    parser.add_argument("--answer-model", default=DEFAULT_ANSWER_MODEL)
    parser.add_argument("--answer-base-url", default=DEFAULT_ANSWER_BASE_URL)
    parser.add_argument("--answer-api-key", default=DEFAULT_ANSWER_API_KEY)
    parser.add_argument("--answer-max-tokens", type=int, default=4096)
    parser.add_argument("--answer-temperature", type=float, default=0.0)
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    parser.add_argument("--judge-base-url", default=DEFAULT_JUDGE_BASE_URL)
    parser.add_argument("--judge-api-key", default=DEFAULT_JUDGE_API_KEY)
    parser.add_argument(
        "--compare-answering",
        action="store_true",
        help=(
            "Run graph_guided, dense_baseline, and the selected retrieval strategy "
            "for every M3DocVQA question, and save retrieval/output/judging details "
            "for all three versions."
        ),
    )
    parser.add_argument("--max-docs", type=int, default=0)
    parser.add_argument("--max-questions", type=int, default=0)
    args = parser.parse_args()

    samples = _load_m3doc_samples(args.m3doc_json)
    grouped_by_doc = _group_m3doc_by_doc(samples)
    memory_paths = _discover_memory_paths(args.memory_path)

    doc_to_memory: Dict[str, Path] = {}
    for memory_path in memory_paths:
        doc_to_memory[memory_path.parent.name] = memory_path

    matched_docs: List[Dict[str, Any]] = []
    missing_docs: List[str] = []
    for doc_id, info in grouped_by_doc.items():
        memory_path = doc_to_memory.get(doc_id)
        if memory_path is None:
            missing_docs.append(doc_id)
            continue
        tree_json_path = Path(args.mineru_root) / info["doc_uuid"] / "tree.json"
        matched_docs.append(
            {
                "doc_id": doc_id,
                "doc_uuid": info["doc_uuid"],
                "doc_path": info["doc_path"],
                "tree_json_path": str(tree_json_path) if tree_json_path.exists() else "",
                "memory_path": memory_path,
                "questions": info["questions"],
            }
        )

    if args.max_docs > 0:
        matched_docs = matched_docs[: args.max_docs]

    total_questions_planned = 0
    for doc in matched_docs:
        question_count = len(doc["questions"])
        if args.max_questions > 0:
            question_count = min(question_count, args.max_questions)
        total_questions_planned += question_count

    per_doc_output_dir = Path(args.per_doc_output_dir)
    per_doc_output_dir.mkdir(parents=True, exist_ok=True)

    answer_generator = AnswerGenerator(
        base_url=args.answer_base_url,
        api_key=args.answer_api_key,
        model_name=args.answer_model,
        max_tokens=args.answer_max_tokens,
        temperature=args.answer_temperature,
    )
    planner = None
    if args.bookrag_query_filtering:
        planner = BookRAGStyleTaskPlanner(
            base_url=args.planner_base_url,
            api_key=args.planner_api_key,
            model_name=args.planner_model,
            max_tokens=args.planner_max_tokens,
            temperature=args.planner_temperature,
        )
    judge_extractor = BookRAGM3DocAnswerExtractor(
        base_url=args.judge_base_url,
        api_key=args.judge_api_key,
        model_name=args.judge_model,
    )

    all_records: List[Dict[str, Any]] = []
    per_doc_summaries: List[Dict[str, Any]] = []
    processed_questions = 0
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

    try:
        total_docs = len(matched_docs)
        for doc_idx, doc in enumerate(matched_docs, start=1):
            runtime = RetrievalRuntime(
                memory_path=str(doc["memory_path"]),
                embedding_model_name=args.embedding_model,
                mineru_root=args.mineru_root,
                tree_json_path=doc["tree_json_path"],
            )
            doc_records: List[Dict[str, Any]] = []
            try:
                questions = list(doc["questions"])
                if args.max_questions > 0:
                    questions = questions[: args.max_questions]

                for sample in questions:
                    question = str(sample.get("question", "")).strip()
                    gold_answer = str(sample.get("answer", "")).strip()
                    base_record = {
                        "doc_id": doc["doc_id"],
                        "doc_uuid": doc["doc_uuid"],
                        "doc_path": doc["doc_path"],
                        "question_id": sample.get("qid", ""),
                        "question": question,
                        "gold_answer": gold_answer,
                        "answers": sample.get("answers", []),
                        "metadata": sample.get("metadata", {}),
                        "supporting_context": sample.get("supporting_context", []),
                    }

                    if args.compare_answering:
                        graph_retrieval_start = time.perf_counter()
                        graph_retrieval = runtime.retrieve_graph_guided(
                            question,
                            schema_top_k=args.schema_top_k,
                            schema_relation_top_k=args.schema_relation_top_k,
                            passage_top_k=args.passage_top_k,
                            modalities=args.modalities,
                            preview_chars=args.preview_chars,
                        )
                        graph_retrieval_time = time.perf_counter() - graph_retrieval_start
                        retrieval_time_totals["graph_guided"] += graph_retrieval_time
                        retrieval_time_totals["total"] += graph_retrieval_time

                        dense_retrieval_start = time.perf_counter()
                        dense_retrieval = runtime.retrieve_dense_baseline(
                            question,
                            passage_top_k=args.passage_top_k,
                            modalities=args.modalities,
                            preview_chars=args.preview_chars,
                        )
                        dense_retrieval_time = time.perf_counter() - dense_retrieval_start
                        retrieval_time_totals["dense_baseline"] += dense_retrieval_time
                        retrieval_time_totals["total"] += dense_retrieval_time

                        selected_bundle = run_selected_strategy_with_optional_bookrag_planning(
                            runtime=runtime,
                            answer_generator=answer_generator,
                            planner=planner,
                            question=question,
                            strategy=args.retrieval_strategy,
                            schema_top_k=args.schema_top_k,
                            schema_relation_top_k=args.schema_relation_top_k,
                            passage_top_k=args.passage_top_k,
                            modalities=args.modalities,
                            preview_chars=args.preview_chars,
                            schema_score_threshold=args.graph_plus_dense_schema_score_threshold,
                            dense_supplement_top_k=args.graph_plus_dense_supplement_top_k,
                            bookrag_query_filtering=args.bookrag_query_filtering,
                        )
                        selected_retrieval = selected_bundle["retrieval"]
                        selected_output = selected_bundle["output"]
                        selected_retrieval_time = float(selected_bundle["retrieval_time_sec"])
                        selected_answer_time = float(selected_bundle["answer_time_sec"])
                        selected_answer_usage = selected_bundle["answer_usage"]
                        selected_planning_time = float(selected_bundle["planning_time_sec"])
                        selected_planning_usage = selected_bundle["planning_usage"]
                        retrieval_time_totals["selected_strategy"] += selected_retrieval_time
                        retrieval_time_totals["total"] += selected_retrieval_time
                        planning_time_totals["selected_strategy"] += selected_planning_time
                        planning_time_totals["total"] += selected_planning_time
                        _accumulate_token_usage(planning_token_totals, selected_planning_usage)
                        _accumulate_token_usage(planning_token_breakdown["selected_strategy"], selected_planning_usage)

                        graph_answer_start = time.perf_counter()
                        graph_output, graph_answer_usage = answer_generator.answer_with_usage(
                            question,
                            graph_retrieval.get("top_passages", []),
                        )
                        graph_answer_time = time.perf_counter() - graph_answer_start
                        answer_time_totals["graph_guided"] += graph_answer_time
                        answer_time_totals["total"] += graph_answer_time

                        dense_answer_start = time.perf_counter()
                        dense_output, dense_answer_usage = answer_generator.answer_with_usage(
                            question,
                            dense_retrieval.get("top_passages", []),
                        )
                        dense_answer_time = time.perf_counter() - dense_answer_start
                        answer_time_totals["dense_baseline"] += dense_answer_time
                        answer_time_totals["total"] += dense_answer_time

                        answer_time_totals["selected_strategy"] += selected_answer_time
                        answer_time_totals["total"] += selected_answer_time

                        _accumulate_token_usage(answer_token_totals, graph_answer_usage)
                        _accumulate_token_usage(answer_token_totals, dense_answer_usage)
                        _accumulate_token_usage(answer_token_totals, selected_answer_usage)
                        _accumulate_token_usage(answer_token_breakdown["graph_guided"], graph_answer_usage)
                        _accumulate_token_usage(answer_token_breakdown["dense_baseline"], dense_answer_usage)
                        _accumulate_token_usage(answer_token_breakdown["selected_strategy"], selected_answer_usage)

                        record = dict(base_record)
                        record.update(
                            {
                                "graph_guided_retrieval": graph_retrieval,
                                "dense_baseline_retrieval": dense_retrieval,
                                "selected_strategy_name": args.retrieval_strategy,
                                "selected_strategy_retrieval": selected_retrieval,
                                "graph_guided_retrieval_time_sec": graph_retrieval_time,
                                "dense_baseline_retrieval_time_sec": dense_retrieval_time,
                                "selected_strategy_retrieval_time_sec": selected_retrieval_time,
                                "graph_guided_answer_time_sec": graph_answer_time,
                                "dense_baseline_answer_time_sec": dense_answer_time,
                                "selected_strategy_answer_time_sec": selected_answer_time,
                                "selected_strategy_planning_time_sec": selected_planning_time,
                                "graph_guided": _judge_one_output(
                                    judge_extractor=judge_extractor,
                                    question=question,
                                    output=graph_output,
                                    gold_answer=gold_answer,
                                ),
                                "dense_baseline": _judge_one_output(
                                    judge_extractor=judge_extractor,
                                    question=question,
                                    output=dense_output,
                                    gold_answer=gold_answer,
                                ),
                                "selected_strategy": _judge_one_output(
                                    judge_extractor=judge_extractor,
                                    question=question,
                                    output=selected_output,
                                    gold_answer=gold_answer,
                                ),
                            }
                        )
                        record["graph_guided"]["answer_usage"] = graph_answer_usage
                        record["dense_baseline"]["answer_usage"] = dense_answer_usage
                        record["selected_strategy"]["answer_usage"] = selected_answer_usage
                        record["selected_strategy"]["planning_usage"] = selected_planning_usage

                        judge_time_totals["graph_guided"] += float(record["graph_guided"].get("judge_time_sec", 0.0))
                        judge_time_totals["dense_baseline"] += float(record["dense_baseline"].get("judge_time_sec", 0.0))
                        judge_time_totals["selected_strategy"] += float(record["selected_strategy"].get("judge_time_sec", 0.0))
                        judge_time_totals["total"] += (
                            float(record["graph_guided"].get("judge_time_sec", 0.0))
                            + float(record["dense_baseline"].get("judge_time_sec", 0.0))
                            + float(record["selected_strategy"].get("judge_time_sec", 0.0))
                        )
                    else:
                        selected_bundle = run_selected_strategy_with_optional_bookrag_planning(
                            runtime=runtime,
                            answer_generator=answer_generator,
                            planner=planner,
                            question=question,
                            strategy=args.retrieval_strategy,
                            schema_top_k=args.schema_top_k,
                            schema_relation_top_k=args.schema_relation_top_k,
                            passage_top_k=args.passage_top_k,
                            modalities=args.modalities,
                            preview_chars=args.preview_chars,
                            schema_score_threshold=args.graph_plus_dense_schema_score_threshold,
                            dense_supplement_top_k=args.graph_plus_dense_supplement_top_k,
                            bookrag_query_filtering=args.bookrag_query_filtering,
                        )
                        retrieval_result = selected_bundle["retrieval"]
                        retrieval_time = float(selected_bundle["retrieval_time_sec"])
                        answer_time = float(selected_bundle["answer_time_sec"])
                        answer_usage = selected_bundle["answer_usage"]
                        planning_time = float(selected_bundle["planning_time_sec"])
                        planning_usage = selected_bundle["planning_usage"]
                        output = selected_bundle["output"]
                        retrieval_time_totals["selected_strategy"] += retrieval_time
                        retrieval_time_totals["total"] += retrieval_time
                        answer_time_totals["selected_strategy"] += answer_time
                        answer_time_totals["total"] += answer_time
                        planning_time_totals["selected_strategy"] += planning_time
                        planning_time_totals["total"] += planning_time

                        _accumulate_token_usage(answer_token_totals, answer_usage)
                        _accumulate_token_usage(answer_token_breakdown["selected_strategy"], answer_usage)
                        _accumulate_token_usage(planning_token_totals, planning_usage)
                        _accumulate_token_usage(planning_token_breakdown["selected_strategy"], planning_usage)

                        judged = _judge_one_output(
                            judge_extractor=judge_extractor,
                            question=question,
                            output=output,
                            gold_answer=gold_answer,
                        )
                        record = dict(base_record)
                        record.update(
                            {
                                "retrieval_strategy": args.retrieval_strategy,
                                "retrieval": retrieval_result,
                                "retrieval_time_sec": retrieval_time,
                                "planning_time_sec": planning_time,
                                "answer_time_sec": answer_time,
                                "planning_usage": planning_usage,
                                "answer_usage": answer_usage,
                                **judged,
                            }
                        )
                        judge_time_totals["selected_strategy"] += float(record.get("judge_time_sec", 0.0))
                        judge_time_totals["total"] += float(record.get("judge_time_sec", 0.0))

                    doc_records.append(record)
                    all_records.append(record)
                    processed_questions += 1

                if args.compare_answering:
                    doc_summary = {
                        "doc_id": doc["doc_id"],
                        "doc_uuid": doc["doc_uuid"],
                        "doc_path": doc["doc_path"],
                        "memory_path": str(doc["memory_path"]),
                        "graph_guided": _build_summary(_mode_records(doc_records, "graph_guided")),
                        "dense_baseline": _build_summary(_mode_records(doc_records, "dense_baseline")),
                        "selected_strategy": _build_summary(_mode_records(doc_records, "selected_strategy")),
                    }
                else:
                    doc_summary = _build_summary(doc_records)
                    doc_summary.update(
                        {
                            "doc_id": doc["doc_id"],
                            "doc_uuid": doc["doc_uuid"],
                            "doc_path": doc["doc_path"],
                            "memory_path": str(doc["memory_path"]),
                        }
                    )
                per_doc_summaries.append(doc_summary)

                doc_save_path = per_doc_output_dir / f"{_safe_name(doc['doc_id'])}.json"
                with doc_save_path.open("w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "doc_summary": doc_summary,
                            "records": doc_records,
                        },
                        f,
                        ensure_ascii=False,
                        indent=2,
                    )

                progress = _progress_bar(processed_questions, total_questions_planned)
                if args.compare_answering:
                    graph_so_far = _build_summary(_mode_records(all_records, "graph_guided"))
                    dense_so_far = _build_summary(_mode_records(all_records, "dense_baseline"))
                    selected_so_far = _build_summary(_mode_records(all_records, "selected_strategy"))
                    graph_correct = sum(int(float(item["graph_guided"].get("em", 0.0)) > 0.0) for item in all_records)
                    dense_correct = sum(int(float(item["dense_baseline"].get("em", 0.0)) > 0.0) for item in all_records)
                    selected_correct = sum(int(float(item["selected_strategy"].get("em", 0.0)) > 0.0) for item in all_records)
                    print(
                        f"{progress} [questions {processed_questions}/{total_questions_planned}] "
                        f"[doc {doc_idx}/{total_docs}] {doc['doc_id']} | "
                        f"graph correct={graph_correct}/{processed_questions}, em={graph_so_far['avg_em']:.4f}, f1={graph_so_far['avg_f1']:.4f} | "
                        f"dense correct={dense_correct}/{processed_questions}, em={dense_so_far['avg_em']:.4f}, f1={dense_so_far['avg_f1']:.4f} | "
                        f"{args.retrieval_strategy} correct={selected_correct}/{processed_questions}, em={selected_so_far['avg_em']:.4f}, f1={selected_so_far['avg_f1']:.4f}"
                    )
                else:
                    overall_so_far = _build_summary(all_records)
                    cumulative_correct = sum(int(float(item.get("em", 0.0)) > 0.0) for item in all_records)
                    print(
                        f"{progress} [questions {processed_questions}/{total_questions_planned}] "
                        f"[doc {doc_idx}/{total_docs}] {doc['doc_id']} | "
                        f"doc em={doc_summary['avg_em']:.4f}, doc f1={doc_summary['avg_f1']:.4f} | "
                        f"cumulative correct={cumulative_correct}/{processed_questions} | "
                        f"cumulative question em={overall_so_far['avg_em']:.4f}, "
                        f"cumulative question f1={overall_so_far['avg_f1']:.4f}"
                    )
            finally:
                runtime.close()
    finally:
        answer_generator.close()
        if planner is not None:
            planner.close()

    if args.compare_answering:
        graph_summary = _build_summary(_mode_records(all_records, "graph_guided"))
        dense_summary = _build_summary(_mode_records(all_records, "dense_baseline"))
        selected_summary = _build_summary(_mode_records(all_records, "selected_strategy"))
        graph_correct = sum(int(float(item["graph_guided"].get("em", 0.0)) > 0.0) for item in all_records)
        dense_correct = sum(int(float(item["dense_baseline"].get("em", 0.0)) > 0.0) for item in all_records)
        selected_correct = sum(int(float(item["selected_strategy"].get("em", 0.0)) > 0.0) for item in all_records)
        payload = {
            "config": {
                "memory_path": str(Path(args.memory_path).resolve()),
                "m3doc_json": str(Path(args.m3doc_json).resolve()),
                "retrieval_strategy": args.retrieval_strategy,
                "compare_answering": True,
                "schema_top_k": args.schema_top_k,
                "schema_relation_top_k": args.schema_relation_top_k,
                "passage_top_k": args.passage_top_k,
                "graph_plus_dense_schema_score_threshold": args.graph_plus_dense_schema_score_threshold,
                "graph_plus_dense_supplement_top_k": args.graph_plus_dense_supplement_top_k,
                "modalities": list(args.modalities or []),
                "bookrag_query_filtering": bool(args.bookrag_query_filtering),
                "embedding_model": args.embedding_model,
                "mineru_root": args.mineru_root,
                "answer_model": args.answer_model,
                "planner_model": args.planner_model,
                "judge_model": args.judge_model,
            },
            "overall_summaries": {
                "graph_guided": graph_summary,
                "dense_baseline": dense_summary,
                "selected_strategy": selected_summary,
            },
            "correct_counts": {
                "graph_guided": f"{graph_correct}/{processed_questions}",
                "dense_baseline": f"{dense_correct}/{processed_questions}",
                "selected_strategy": f"{selected_correct}/{processed_questions}",
            },
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
            "num_total_m3doc_samples": len(samples),
            "num_matched_docs": len(matched_docs),
            "num_missing_docs": len(missing_docs),
            "missing_docs": missing_docs,
            "processed_questions": processed_questions,
            "per_doc_summaries": per_doc_summaries,
            "records": all_records,
        }
    else:
        overall_summary = _build_summary(all_records)
        total_correct = sum(int(float(item.get("em", 0.0)) > 0.0) for item in all_records)
        payload = {
            "config": {
                "memory_path": str(Path(args.memory_path).resolve()),
                "m3doc_json": str(Path(args.m3doc_json).resolve()),
                "retrieval_strategy": args.retrieval_strategy,
                "compare_answering": False,
                "schema_top_k": args.schema_top_k,
                "schema_relation_top_k": args.schema_relation_top_k,
                "passage_top_k": args.passage_top_k,
                "graph_plus_dense_schema_score_threshold": args.graph_plus_dense_schema_score_threshold,
                "graph_plus_dense_supplement_top_k": args.graph_plus_dense_supplement_top_k,
                "modalities": list(args.modalities or []),
                "bookrag_query_filtering": bool(args.bookrag_query_filtering),
                "embedding_model": args.embedding_model,
                "mineru_root": args.mineru_root,
                "answer_model": args.answer_model,
                "planner_model": args.planner_model,
                "judge_model": args.judge_model,
            },
            "overall_summary": overall_summary,
            "correct_questions": f"{total_correct}/{processed_questions}",
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
            "num_total_m3doc_samples": len(samples),
            "num_matched_docs": len(matched_docs),
            "num_missing_docs": len(missing_docs),
            "missing_docs": missing_docs,
            "processed_questions": processed_questions,
            "per_doc_summaries": per_doc_summaries,
            "records": all_records,
        }

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print("--------------------------------------------------")
    print(f"Processed questions: {processed_questions}/{total_questions_planned}")
    print(f"Matched docs: {len(matched_docs)}")
    print(f"Missing docs: {len(missing_docs)}")
    if args.compare_answering:
        print(
            f"Graph correct: {graph_correct}/{processed_questions} | "
            f"Avg em: {graph_summary['avg_em']:.6f} | Avg f1: {graph_summary['avg_f1']:.6f}"
        )
        print(
            f"Dense correct: {dense_correct}/{processed_questions} | "
            f"Avg em: {dense_summary['avg_em']:.6f} | Avg f1: {dense_summary['avg_f1']:.6f}"
        )
        print(
            f"{args.retrieval_strategy} correct: {selected_correct}/{processed_questions} | "
            f"Avg em: {selected_summary['avg_em']:.6f} | Avg f1: {selected_summary['avg_f1']:.6f}"
        )
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
    else:
        print(f"Correct questions: {total_correct}/{processed_questions}")
        print(f"Avg em: {overall_summary['avg_em']:.6f}")
        print(f"Avg f1: {overall_summary['avg_f1']:.6f}")
        print(
            "Runtime stats | "
            f"retrieval total: {retrieval_time_totals['total']:.2f}s | "
            f"planning total: {planning_time_totals['total']:.2f}s | "
            f"answer total: {answer_time_totals['total']:.2f}s | "
            f"judge total: {judge_time_totals['total']:.2f}s | "
            f"planning tokens: {planning_token_totals['total_tokens']} total | "
            f"answer tokens: {answer_token_totals['total_tokens']} total "
            f"(prompt={answer_token_totals['prompt_tokens']}, "
            f"completion={answer_token_totals['completion_tokens']})"
        )
    print(f"Saved detailed results to {output_path}")


if __name__ == "__main__":
    main()
