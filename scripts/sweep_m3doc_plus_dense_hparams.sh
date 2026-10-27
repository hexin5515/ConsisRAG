#!/usr/bin/env bash
set -euo pipefail


DATASET="m3doc"

SCHEMA_TOP_K=(5 10 15 20 25 30 35 40 45 50)

STAMP="$(date +%Y%m%d_%H%M%S)"
SUMMARY_CSV="output/sweep_${DATASET}_schema_${STAMP}.csv"

BASE_CMD=(
  python scripts/retrieve_m3docvqa_from_constructed_memory.py
  output/m3docvqa_constructed_memory
  --m3doc-json data/datasets/m3docvqa/m3docrag_office.json
  --mineru-root data/datasets/m3docvqa/process_documents
  --retrieval-strategy graph_plus_dense
  --schema-relation-top-k 15
  --passage-top-k 10
  --graph-plus-dense-schema-score-threshold 0.5
  --graph-plus-dense-supplement-top-k 4
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
    primary = float(data.get("overall_accuracy", 0.0))
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
