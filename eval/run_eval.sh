#!/usr/bin/env bash
# LLM-as-a-Judge evaluation: Table 1 (main), Table 2 (ablation), Table 3 (errors).
#
# Sequential on purpose:
#   - Tables 1 and 2 share one judge cache
#     (outputs/main_exp/eval/judge_cache_shared.jsonl), so overlapping calls
#     (e.g., exp4 / exp5-k5-det faithfulness) are billed/computed once.
#   - Table 3 runs in DERIVED mode: it aggregates Table 1's per-note verdicts
#     with ZERO extra judge calls, so it MUST run after Table 1.
# Re-running this script is safe and cheap: cached calls are never recomputed.
#
# Two judge backends:
#   BACKEND=openai (default) — OpenAI API. Needs OPENAI_API_KEY in the env
#                              (do NOT hardcode the key in this file).
#   BACKEND=ollama           — local model, free, no API key. Slower.
#
# Usage:
#   # OpenAI judge:
#   export OPENAI_API_KEY=sk-...
#   JUDGE_MODEL=gpt-4.1-mini bash run_eval.sh          # 10 records/professor (210/method)
#   SAMPLE=0 JUDGE_MODEL=gpt-4.1-mini bash run_eval.sh # full 1050 records
#
#   # Local Ollama judge (free):
#   BACKEND=ollama JUDGE_MODEL=gpt-oss:120b bash run_eval.sh
#   BACKEND=ollama JUDGE_MODEL=qwen3.5:122b SAMPLE=0 bash run_eval.sh
#
#   DRYRUN=1 bash run_eval.sh                          # plumbing test, no judge calls
#
# Results: outputs/main_exp/eval/table{1_main,2_ablation,3_error_analysis}.csv

set -euo pipefail
cd "$(dirname "$0")"

BACKEND="${BACKEND:-openai}"
SAMPLE="${SAMPLE:-10}"
WORKERS="${WORKERS:-8}"
OLLAMA_HOST="${OLLAMA_HOST:-127.0.0.1:11434}"

# A different judge family than the generator (qwen3.6:35b) reduces
# self-preference bias; gpt-oss:120b is the recommended local default.
if [[ "$BACKEND" == "ollama" ]]; then
    JUDGE_MODEL="${JUDGE_MODEL:-gpt-oss:120b}"
else
    JUDGE_MODEL="${JUDGE_MODEL:-gpt-4o}"
fi

EXTRA_ARGS=()
if [[ "${DRYRUN:-0}" == "1" ]]; then
    EXTRA_ARGS+=(--dry_run)
elif [[ "$BACKEND" == "ollama" ]]; then
    EXTRA_ARGS+=(--backend ollama --ollama_host "$OLLAMA_HOST")
    # Ollama serves requests serially per loaded model; high worker counts just queue.
    WORKERS="${WORKERS_OVERRIDE:-2}"
elif [[ -z "${OPENAI_API_KEY:-}" ]]; then
    echo "ERROR: OPENAI_API_KEY is not set. Export it, set BACKEND=ollama, or DRYRUN=1." >&2
    exit 1
fi

echo "[eval] backend=$BACKEND judge=$JUDGE_MODEL sample_per_professor=$SAMPLE workers=$WORKERS dryrun=${DRYRUN:-0}"

echo
echo "=== [1/3] Table 1: Main comparison ==="
python eval_table1_main.py \
  --judge_model "$JUDGE_MODEL" \
  --sample_per_professor "$SAMPLE" \
  --workers "$WORKERS" \
  "${EXTRA_ARGS[@]}"

echo
echo "=== [2/3] Table 2: Few-shot ablation ==="
python eval_table2_ablation.py \
  --judge_model "$JUDGE_MODEL" \
  --sample_per_professor "$SAMPLE" \
  --workers "$WORKERS" \
  "${EXTRA_ARGS[@]}"

echo
echo "=== [3/3] Table 3: Error analysis (derived from Table 1, no judge calls) ==="
# Derived mode: aggregates Table 1's per-note verdicts (unsupported claims,
# patient-mixing, absent critical facts). Reuses exactly the records Table 1
# evaluated. Free regardless of backend.
python eval_table3_error_analysis.py

echo
echo "All three tables finished. Results in outputs/main_exp/eval/:"
ls -la outputs/main_exp/eval/table*.csv 2>/dev/null || true
