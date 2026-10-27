#!/usr/bin/env bash
set -euo pipefail


DATASET="mmlong"

SCHEMA_TOP_K=(5 10 15 20 25 30 35 40 45 50)

STAMP="$(date +%Y%m%d_%H%M%S)"
SUMMARY_CSV="output/sweep_${DATASET}_schema_${STAMP}.csv"


BASE_CMD=(
  python scripts/retrieve_from_constructed_memory.py
  --memory-path output/mineru_relation_full_folder_test_reencoded_0_6b
  --samples-json data/datasets/MMLongBench-Doc/dataset/MMLongBench-Doc_office.json
  --retrieval-strategy graph_plus_dense
  --graph-plus-dense-schema-score-threshold 0.25
  --graph-plus-dense-supplement-top-k 2
)


mkdir -p output
printf "schema_top_k,primary_metric,f1_metric\n" > "${SUMMARY_CSV}"

extract_metrics() {
  local dataset="$1"
  local output_json="$2"
  python - "$dataset" "$output_json" <<'PY'
import json
import sys

dataset = sys.argv[1]
path = sys.argv[2]

with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)

if dataset == "mmlong":
    overall = data.get("overall_accuracy", {})
    if isinstance(overall, dict):
        selected = overall.get("selected_strategy", overall)
        if isinstance(selected, dict):
            primary = float(selected.get("accuracy", 0.0))
            f1 = float(selected.get("f1", primary))
        else:
            primary = float(selected)
            f1 = primary
    else:
        primary = float(overall)
        f1 = primary
elif dataset == "qasper":
    summary = data.get("overall_summary")
    if summary is None:
        summary = data.get("overall_summaries", {}).get("selected_strategy", {})
    primary = float(summary.get("avg_acc", 0.0))
    f1 = float(summary.get("avg_f1", 0.0))
elif dataset == "m3doc":
    summary = data.get("overall_summary")
    if summary is None:
        summary = data.get("overall_summaries", {}).get("selected_strategy", {})
    primary = float(summary.get("avg_em", 0.0))
    f1 = float(summary.get("avg_f1", 0.0))
else:
    raise SystemExit(f"Unsupported dataset: {dataset}")

print(f"{primary:.6f},{f1:.6f}")
PY
}

run_one() {
  local stop_k="$1"

  local temp_json
  local temp_dir
  temp_json="$(mktemp "/tmp/${DATASET}_sweep_${stop_k}_XXXX.json")"
  temp_dir="$(mktemp -d "/tmp/${DATASET}_sweep_${stop_k}_XXXX")"

  echo
  echo "==> schema_top_k=${stop_k}"

  "${BASE_CMD[@]}" \
    --schema-top-k "${stop_k}" \
    --output-path "${temp_json}" \
    --per-doc-output-dir "${temp_dir}"

  local metrics
  local primary
  local f1
  metrics="$(extract_metrics "${DATASET}" "${temp_json}")"
  primary="${metrics%%,*}"
  f1="${metrics##*,}"

  printf "%s,%s,%s\n" "${stop_k}" "${primary}" "${f1}" >> "${SUMMARY_CSV}"
  echo "    primary=${primary} f1=${f1}"

  rm -f "${temp_json}"
  rm -rf "${temp_dir}"
}

for stop_k in "${SCHEMA_TOP_K[@]}"; do
  run_one "${stop_k}"
done

echo
echo "Sweep finished."
echo "Summary CSV: ${SUMMARY_CSV}"
