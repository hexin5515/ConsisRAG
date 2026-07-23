#!/usr/bin/env python3
"""
Run a lightweight joint hyperparameter sweep for graph_plus_dense.

The sweep varies:
1. graph_plus_dense_schema_score_threshold
2. graph_plus_dense_supplement_top_k

Unlike the full evaluation scripts, this script only keeps compact aggregate
metrics for each parameter combination. It does not write per-question answers
or retrieval payloads unless you explicitly pass --output-path, in which case
it only saves the compact sweep summary JSON.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Sequence, Tuple


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
    DEFAULT_MINERU_ROOT,
    DEFAULT_PLANNER_API_KEY,
    DEFAULT_PLANNER_BASE_URL,
    DEFAULT_PLANNER_MODEL,
    AnswerGenerator,
    BookRAGStyleTaskPlanner,
    RetrievalRuntime,
    _discover_memory_paths,
    _infer_doc_id_from_memory_path,
    _load_samples_grouped_by_doc,
    run_selected_strategy_with_optional_bookrag_planning,
)
from retrieve_qasper_from_constructed_memory import (  # type: ignore
    DEFAULT_QASPER_JSON,
    DEFAULT_QASPER_MINERU_ROOT,
    _group_qasper_by_doc,
    _load_qasper_samples,
    _resolve_qasper_tree_json_path,
)
from retrieve_m3docvqa_from_constructed_memory import (  # type: ignore
    DEFAULT_M3DOC_JSON,
    DEFAULT_M3DOC_MINERU_ROOT,
    _group_m3doc_by_doc,
    _load_m3doc_samples,
)
from evaluation.mmlong_bookrag_eval import (  # type: ignore
    BookRAGStyleAnswerExtractor,
    eval_score,
)
from evaluation.qasper_bookrag_eval import (  # type: ignore
    BookRAGQasperAnswerExtractor,
    eval_single_res,
    get_answers_and_evidence,
)
from evaluation.m3doc_bookrag_eval import (  # type: ignore
    BookRAGM3DocAnswerExtractor,
    list_em,
    list_f1,
)


DEFAULT_MMLONG_JSON = "data/datasets/MMLongBench-Doc/dataset/MMLongBench-Doc_office.json"


@dataclass
class SweepDocJob:
    doc_id: str
    memory_path: Path
    tree_json_path: str
    questions: List[Dict[str, Any]]


def _parse_float_list(values: Sequence[str]) -> List[float]:
    parsed: List[float] = []
    for raw in values:
        text = str(raw).strip()
        if not text:
            continue
        parsed.append(float(text))
    return parsed


def _parse_int_list(values: Sequence[str]) -> List[int]:
    parsed: List[int] = []
    for raw in values:
        text = str(raw).strip()
        if not text:
            continue
        parsed.append(int(text))
    return parsed


def _mean_or_zero(values: Sequence[float]) -> float:
    return float(mean(values)) if values else 0.0


def _take_doc_jobs(dataset: str, args: argparse.Namespace) -> List[SweepDocJob]:
    memory_paths = _discover_memory_paths(args.memory_path)

    if dataset == "mmlong":
        grouped_samples = _load_samples_grouped_by_doc(args.samples_json)
        jobs: List[SweepDocJob] = []
        for memory_path in memory_paths:
            doc_id = _infer_doc_id_from_memory_path(memory_path)
            matched = grouped_samples.get(doc_id, [])
            if not matched:
                continue
            jobs.append(
                SweepDocJob(
                    doc_id=doc_id,
                    memory_path=memory_path,
                    tree_json_path="",
                    questions=matched,
                )
            )
        return jobs

    if dataset == "qasper":
        samples = _load_qasper_samples(args.qasper_json)
        grouped = _group_qasper_by_doc(samples)
        paper_to_memory: Dict[str, Path] = {p.parent.name: p for p in memory_paths}
        jobs = []
        for paper_id, info in grouped.items():
            memory_path = paper_to_memory.get(paper_id)
            if memory_path is None:
                continue
            jobs.append(
                SweepDocJob(
                    doc_id=paper_id,
                    memory_path=memory_path,
                    tree_json_path=_resolve_qasper_tree_json_path(
                        mineru_root=args.mineru_root,
                        doc_uuid=str(info.get("doc_uuid", "")),
                    ),
                    questions=list(info.get("questions", [])),
                )
            )
        return jobs

    if dataset == "m3doc":
        samples = _load_m3doc_samples(args.m3doc_json)
        grouped = _group_m3doc_by_doc(samples)
        doc_to_memory: Dict[str, Path] = {p.parent.name: p for p in memory_paths}
        jobs = []
        for doc_id, info in grouped.items():
            memory_path = doc_to_memory.get(doc_id)
            if memory_path is None:
                continue
            doc_uuid = str(info.get("doc_uuid", "")).strip()
            tree_json_path = ""
            if doc_uuid:
                candidate = Path(args.mineru_root).resolve() / doc_uuid / "tree.json"
                if candidate.exists():
                    tree_json_path = str(candidate)
            jobs.append(
                SweepDocJob(
                    doc_id=doc_id,
                    memory_path=memory_path,
                    tree_json_path=tree_json_path,
                    questions=list(info.get("questions", [])),
                )
            )
        return jobs

    raise ValueError(f"Unsupported dataset: {dataset}")


def _judge_mmlong_one(
    *,
    extractor: BookRAGStyleAnswerExtractor,
    question: str,
    output: str,
    gold_answer: Any,
    answer_format: str,
) -> Dict[str, float]:
    _, pred_answer, _, score_value = extractor.extract(question, output, str(gold_answer))
    return {
        "primary": float(score_value),
        "f1": float(eval_score(gold_answer, pred_answer, answer_format)),
    }


def _judge_qasper_one(
    *,
    extractor: BookRAGQasperAnswerExtractor,
    question: str,
    output: str,
    gold_answers: Sequence[Dict[str, Any]],
) -> Dict[str, float]:
    correct_answer = str(gold_answers[0].get("answer", "")) if gold_answers else ""
    _, pred_ans, _, llm_score = extractor.extract(question, output, correct_answer)
    acc, f1 = eval_single_res(pred_ans, gold_answers)
    return {
        "primary": float(acc),
        "f1": float(f1),
        "llm_score": float(llm_score),
    }


def _judge_m3doc_one(
    *,
    extractor: BookRAGM3DocAnswerExtractor,
    question: str,
    output: str,
    gold_answer: str,
) -> Dict[str, float]:
    _, pred_ans, _, llm_score = extractor.extract(question, output, gold_answer)
    em = list_em(pred_ans, gold_answer)
    f1 = list_f1(pred_ans, gold_answer)
    return {
        "primary": float(em),
        "f1": float(f1),
        "llm_score": float(llm_score),
    }


def _progress(current: int, total: int) -> str:
    if total <= 0:
        return "[----------------------------]"
    width = 28
    filled = min(width, max(0, int(width * current / total)))
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep graph_plus_dense threshold x supplement_top_k with compact aggregate outputs."
    )
    parser.add_argument("--dataset", choices=["mmlong", "qasper", "m3doc"], required=True)
    parser.add_argument("memory_path", help="One memory.json file or a root directory containing many memory.json files.")

    parser.add_argument("--samples-json", default=DEFAULT_MMLONG_JSON)
    parser.add_argument("--qasper-json", default=DEFAULT_QASPER_JSON)
    parser.add_argument("--m3doc-json", default=DEFAULT_M3DOC_JSON)
    parser.add_argument("--mineru-root", default="")

    parser.add_argument("--retrieval-strategy", default="graph_plus_dense", choices=["graph_plus_dense"])
    parser.add_argument("--schema-top-k", type=int, default=5)
    parser.add_argument("--schema-relation-top-k", type=int, default=5)
    parser.add_argument("--passage-top-k", type=int, default=10)
    parser.add_argument("--modalities", nargs="*", default=[])
    parser.add_argument("--bookrag-query-filtering", action="store_true")
    parser.add_argument("--preview-chars", type=int, default=240)

    parser.add_argument("--thresholds", nargs="+", default=["0.25", "0.35", "0.45", "0.55", "0.65"])
    parser.add_argument("--supplements", nargs="+", default=["0", "1", "2", "3"])

    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--answer-model", default=DEFAULT_ANSWER_MODEL)
    parser.add_argument("--answer-base-url", default=DEFAULT_ANSWER_BASE_URL)
    parser.add_argument("--answer-api-key", default=DEFAULT_ANSWER_API_KEY)
    parser.add_argument("--answer-max-tokens", type=int, default=0)
    parser.add_argument("--answer-temperature", type=float, default=0.0)
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    parser.add_argument("--judge-base-url", default=DEFAULT_JUDGE_BASE_URL)
    parser.add_argument("--judge-api-key", default=DEFAULT_JUDGE_API_KEY)
    parser.add_argument("--planner-model", default=DEFAULT_PLANNER_MODEL)
    parser.add_argument("--planner-base-url", default=DEFAULT_PLANNER_BASE_URL)
    parser.add_argument("--planner-api-key", default=DEFAULT_PLANNER_API_KEY)
    parser.add_argument("--planner-max-tokens", type=int, default=1024)
    parser.add_argument("--planner-temperature", type=float, default=0.0)

    parser.add_argument("--max-docs", type=int, default=0)
    parser.add_argument("--max-questions", type=int, default=0)
    parser.add_argument("--output-path", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.mineru_root = {
        "mmlong": args.mineru_root or DEFAULT_MINERU_ROOT,
        "qasper": args.mineru_root or DEFAULT_QASPER_MINERU_ROOT,
        "m3doc": args.mineru_root or DEFAULT_M3DOC_MINERU_ROOT,
    }[args.dataset]
    if args.answer_max_tokens <= 0:
        args.answer_max_tokens = 256 if args.dataset == "mmlong" else 4096

    thresholds = _parse_float_list(args.thresholds)
    supplements = _parse_int_list(args.supplements)
    combos: List[Tuple[float, int]] = [(thr, sup) for thr in thresholds for sup in supplements]
    if not combos:
        raise ValueError("No valid threshold/supplement combinations were parsed.")

    jobs = _take_doc_jobs(args.dataset, args)
    if args.max_docs > 0:
        jobs = jobs[: args.max_docs]
    if not jobs:
        raise ValueError("No matched document jobs found.")

    total_questions = sum(len(job.questions) for job in jobs)
    if args.max_questions > 0:
        capped_jobs: List[SweepDocJob] = []
        remaining = args.max_questions
        for job in jobs:
            if remaining <= 0:
                break
            taken = list(job.questions[:remaining])
            if taken:
                capped_jobs.append(
                    SweepDocJob(
                        doc_id=job.doc_id,
                        memory_path=job.memory_path,
                        tree_json_path=job.tree_json_path,
                        questions=taken,
                    )
                )
                remaining -= len(taken)
        jobs = capped_jobs
        total_questions = sum(len(job.questions) for job in jobs)

    prompt_style = "legacy_mmlong" if args.dataset == "mmlong" else "gbc"
    answer_generator = AnswerGenerator(
        base_url=args.answer_base_url,
        api_key=args.answer_api_key,
        model_name=args.answer_model,
        max_tokens=args.answer_max_tokens,
        temperature=args.answer_temperature,
        prompt_style=prompt_style,
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

    if args.dataset == "mmlong":
        judge_extractor: Any = BookRAGStyleAnswerExtractor(
            base_url=args.judge_base_url,
            api_key=args.judge_api_key,
            model_name=args.judge_model,
        )
        primary_metric_name = "accuracy"
    elif args.dataset == "qasper":
        judge_extractor = BookRAGQasperAnswerExtractor(
            base_url=args.judge_base_url,
            api_key=args.judge_api_key,
            model_name=args.judge_model,
        )
        primary_metric_name = "accuracy"
    else:
        judge_extractor = BookRAGM3DocAnswerExtractor(
            base_url=args.judge_base_url,
            api_key=args.judge_api_key,
            model_name=args.judge_model,
        )
        primary_metric_name = "em"

    results_by_combo: Dict[Tuple[float, int], Dict[str, Any]] = {
        combo: {
            "threshold": combo[0],
            "supplement_top_k": combo[1],
            "primary_values": [],
            "f1_values": [],
            "llm_values": [],
            "question_count": 0,
            "elapsed_sec": 0.0,
        }
        for combo in combos
    }

    processed_questions = 0
    sweep_start = time.perf_counter()
    try:
        for doc_idx, job in enumerate(jobs, 1):
            runtime = RetrievalRuntime(
                memory_path=str(job.memory_path),
                embedding_model_name=args.embedding_model,
                mineru_root=args.mineru_root,
                tree_json_path=job.tree_json_path,
            )
            try:
                print(
                    f"\n[doc {doc_idx}/{len(jobs)}] {job.doc_id} | "
                    f"{len(job.questions)} question(s) | memory={job.memory_path.parent.name}"
                )
                for q_idx, item in enumerate(job.questions, 1):
                    question = str(item.get("question", "")).strip()
                    for combo in combos:
                        thr, sup = combo
                        combo_start = time.perf_counter()
                        bundle = run_selected_strategy_with_optional_bookrag_planning(
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
                            schema_score_threshold=thr,
                            dense_supplement_top_k=sup,
                            bookrag_query_filtering=args.bookrag_query_filtering,
                        )
                        output = str(bundle.get("output", ""))
                        if args.dataset == "mmlong":
                            judged = _judge_mmlong_one(
                                extractor=judge_extractor,
                                question=question,
                                output=output,
                                gold_answer=item.get("answer", ""),
                                answer_format=str(item.get("answer_format", "Str")),
                            )
                        elif args.dataset == "qasper":
                            gold_answers, _ = get_answers_and_evidence(item)
                            judged = _judge_qasper_one(
                                extractor=judge_extractor,
                                question=question,
                                output=output,
                                gold_answers=gold_answers,
                            )
                        else:
                            judged = _judge_m3doc_one(
                                extractor=judge_extractor,
                                question=question,
                                output=output,
                                gold_answer=str(item.get("answer", "")),
                            )
                        bucket = results_by_combo[combo]
                        bucket["primary_values"].append(float(judged["primary"]))
                        bucket["f1_values"].append(float(judged["f1"]))
                        if "llm_score" in judged:
                            bucket["llm_values"].append(float(judged["llm_score"]))
                        bucket["question_count"] += 1
                        bucket["elapsed_sec"] += time.perf_counter() - combo_start
                    processed_questions += 1
                    print(
                        f"{_progress(processed_questions, total_questions)} "
                        f"[questions {processed_questions}/{total_questions}] "
                        f"{question[:120]}"
                    )
            finally:
                runtime.close()
    finally:
        answer_generator.close()
        if planner is not None:
            planner.close()

    elapsed_total = time.perf_counter() - sweep_start
    summary_rows: List[Dict[str, Any]] = []
    for combo in combos:
        row = results_by_combo[combo]
        summary_rows.append(
            {
                "threshold": row["threshold"],
                "supplement_top_k": row["supplement_top_k"],
                primary_metric_name: round(_mean_or_zero(row["primary_values"]), 6),
                "f1": round(_mean_or_zero(row["f1_values"]), 6),
                "avg_llm_score": round(_mean_or_zero(row["llm_values"]), 6) if row["llm_values"] else None,
                "question_count": int(row["question_count"]),
                "elapsed_sec": round(float(row["elapsed_sec"]), 4),
            }
        )
    summary_rows.sort(key=lambda item: (-float(item[primary_metric_name]), -float(item["f1"]), item["threshold"], item["supplement_top_k"]))

    print("\n=== Joint Sweep Summary ===")
    print(
        f"dataset={args.dataset} | strategy={args.retrieval_strategy} | "
        f"questions={total_questions} | total_elapsed={elapsed_total:.2f}s"
    )
    header = f"{'threshold':>10}  {'supplement':>10}  {primary_metric_name:>10}  {'f1':>10}  {'count':>8}"
    print(header)
    print("-" * len(header))
    for row in summary_rows:
        print(
            f"{row['threshold']:>10.3f}  "
            f"{row['supplement_top_k']:>10d}  "
            f"{row[primary_metric_name]:>10.4f}  "
            f"{row['f1']:>10.4f}  "
            f"{row['question_count']:>8d}"
        )

    payload = {
        "dataset": args.dataset,
        "retrieval_strategy": args.retrieval_strategy,
        "schema_top_k": args.schema_top_k,
        "schema_relation_top_k": args.schema_relation_top_k,
        "passage_top_k": args.passage_top_k,
        "modalities": list(args.modalities),
        "bookrag_query_filtering": bool(args.bookrag_query_filtering),
        "thresholds": thresholds,
        "supplements": supplements,
        "primary_metric": primary_metric_name,
        "num_docs": len(jobs),
        "num_questions": total_questions,
        "elapsed_total_sec": round(elapsed_total, 4),
        "results": summary_rows,
    }
    if args.output_path:
        output_path = Path(args.output_path).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"\nSaved compact sweep summary to: {output_path}")


if __name__ == "__main__":
    main()
