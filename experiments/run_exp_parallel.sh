#!/usr/bin/env bash
# Parallel experiment runner: distributes all experiment runs across 3 GPUs.
#
#   Method                                Fact Extraction  Iterative Agent  Few-shot Style
#   exp1  Raw-to-Note                          x                 x                x
#   exp2  Raw-to-Fact-to-Note                  o                 x                x
#   exp3  Chunk-to-Fact-to-Note                o                 x                x
#   exp4  Iterative Multi-Agent Fact-to-Note   o                 o                x
#   exp5  Ours (main + ablation sweep)         o                 o                o
#
# How it works:
#   - One Ollama instance per GPU (qwen3.6:35b fits in a single A100 40GB).
#       GPU 0 -> 127.0.0.1:11444  (pinned with CUDA_VISIBLE_DEVICES=0)
#       GPU 1 -> 127.0.0.1:11445  (pinned with CUDA_VISIBLE_DEVICES=1)
#       GPU 2 -> 127.0.0.1:11446  (pinned with CUDA_VISIBLE_DEVICES=2)
#     The default snap server (11434) and 11435 (another project) are left
#     alone; the default server is only asked to unload the model so GPU 0
#     has room for our pinned instance.
#   - 3 workers (one per GPU) pull jobs from a shared queue; whichever GPU
#     finishes first picks up the next job, so load balances automatically.
#   - Every job runs with --resume: finished jobs are near-instant no-ops, and
#     re-running this script after an interruption continues where it stopped.
#   - Determinism: each instance serves one synchronous request at a time, so
#     with temperature 0 + fixed seed the results are identical to a
#     sequential run (no cross-request batching).
#
# Usage:
#   bash run_exp_parallel.sh                 # real run
#   RUN_EXP_DRYRUN=1 bash run_exp_parallel.sh  # plumbing test: no GPU/Ollama,
#                                              # 5 rows/job, outputs to a temp dir
#
# Monitor:
#   tail -f outputs/logs/<job>.log
#   watch -n5 nvidia-smi

set -uo pipefail   # no -e: one failed job must not kill the queue; failures are collected.
cd "$(dirname "$0")"

MODEL="qwen3.6:35b"
# IMPORTANT: do NOT use /snap/bin/ollama (the snap wrapper). Its launcher runs
# `OLLAMA_HOST=$(snapctl get host)` which overwrites our per-GPU OLLAMA_HOST
# with 127.0.0.1:11434 and the spawned server dies with "address already in
# use". The raw binary inside the snap respects the environment.
if [[ -z "${OLLAMA_BIN:-}" ]]; then
    if [[ -x /snap/ollama/current/bin/ollama ]]; then
        OLLAMA_BIN=/snap/ollama/current/bin/ollama
    else
        OLLAMA_BIN="$(command -v ollama)"
    fi
fi
OLLAMA_MODELS_DIR="${OLLAMA_MODELS_DIR:-/var/snap/ollama/common/models}"

GPU_IDS=(0 1 2)
GPU_HOSTS=("127.0.0.1:11444" "127.0.0.1:11445" "127.0.0.1:11446")

OUT_DIR="outputs"
EXTRA_ARGS=()
DRYRUN="${RUN_EXP_DRYRUN:-0}"
if [[ "$DRYRUN" == "1" ]]; then
    OUT_DIR="$(mktemp -d /tmp/run_exp_parallel_dry.XXXXXX)"
    EXTRA_ARGS=(--dry_run --max_rows 5 --no_progress)
    echo "[dry-run] outputs go to $OUT_DIR; no Ollama servers are started."
fi

LOG_DIR="$OUT_DIR/logs"
mkdir -p "$LOG_DIR"

RUN_DIR="$(mktemp -d /tmp/run_exp_parallel.XXXXXX)"
COUNTER="$RUN_DIR/next_job"
LOCK="$RUN_DIR/lock"
FAILED="$RUN_DIR/failed"
echo 0 > "$COUNTER"
: > "$FAILED"

export PYTHONUNBUFFERED=1

# ---------------------------------------------------------------------------
# Job table. Workers pull these in order; long and short jobs mix freely.
# ---------------------------------------------------------------------------
JOB_NAMES=(
    # exp1-3 already completed; uncomment to re-run (resume makes it a no-op).
    # exp1_raw_to_note
    # exp2_raw_fact_to_note
    # exp3_chunk_fact_to_note
    exp4_iterative_fact_to_note
    exp5_ours_k5_deterministic
    exp5_ours_k3_deterministic
    exp5_ours_k5_random_seed1225
    exp5_ours_k3_random_seed1225
)

run_job() {
    local name="$1" host="$2"
    case "$name" in
        exp1_raw_to_note)
            python exp1_raw_to_note.py \
                --model "$MODEL" --ollama_host "http://$host" \
                --output_csv "$OUT_DIR/exp1_raw_to_note.csv" \
                --audit_jsonl "$OUT_DIR/exp1_raw_to_note_audit.jsonl" \
                --resume "${EXTRA_ARGS[@]}" ;;
        exp2_raw_fact_to_note)
            python exp2_raw_fact_to_note.py \
                --model "$MODEL" --ollama_host "http://$host" \
                --output_csv "$OUT_DIR/exp2_raw_fact_to_note.csv" \
                --audit_jsonl "$OUT_DIR/exp2_raw_fact_to_note_audit.jsonl" \
                --resume "${EXTRA_ARGS[@]}" ;;
        exp3_chunk_fact_to_note)
            python exp3_chunk_fact_to_note.py \
                --model "$MODEL" --ollama_host "http://$host" \
                --output_csv "$OUT_DIR/exp3_chunk_fact_to_note.csv" \
                --audit_jsonl "$OUT_DIR/exp3_chunk_fact_to_note_audit.jsonl" \
                --resume "${EXTRA_ARGS[@]}" ;;
        exp4_iterative_fact_to_note)
            python exp4_iterative_fact_to_note.py \
                --model "$MODEL" --ollama_host "http://$host" \
                --output_csv "$OUT_DIR/exp4_iterative_fact_to_note.csv" \
                --audit_jsonl "$OUT_DIR/exp4_iterative_fact_to_note_audit.jsonl" \
                --resume "${EXTRA_ARGS[@]}" ;;
        exp5_ours_k5_deterministic)
            run_exp5 "$host" 5 deterministic "" ;;
        exp5_ours_k3_deterministic)
            run_exp5 "$host" 3 deterministic "" ;;
        exp5_ours_k5_random_seed1225)
            run_exp5 "$host" 5 random 1225 ;;
        exp5_ours_k3_random_seed1225)
            run_exp5 "$host" 3 random 1225 ;;
        *)
            echo "unknown job: $name" >&2
            return 1 ;;
    esac
}

run_exp5() {
    local host="$1" k="$2" selection="$3" seed="$4"
    local tag="k${k}_${selection}"
    local seed_args=()
    if [[ "$selection" == "random" ]]; then
        tag="${tag}_seed${seed}"
        seed_args=(--reference_seed "$seed")
    fi
    python exp5_ours_fewshot_style.py \
        --model "$MODEL" --ollama_host "http://$host" \
        --sample_count "$k" --reference_selection "$selection" "${seed_args[@]}" \
        --output_csv "$OUT_DIR/exp5_ours_${tag}.csv" \
        --audit_jsonl "$OUT_DIR/exp5_ours_${tag}_audit.jsonl" \
        --style_cache_jsonl "$OUT_DIR/exp5_ours_${tag}_style_prompts.jsonl" \
        --resume "${EXTRA_ARGS[@]}"
}

# ---------------------------------------------------------------------------
# Ollama server management (skipped entirely in dry-run mode)
# ---------------------------------------------------------------------------
STARTED_PIDS=()

server_alive() {
    curl -sf "http://$1/api/version" >/dev/null 2>&1
}

# The default snap server (11434) may still hold the model on GPU 0 from
# earlier sequential runs. Two 27GB copies do not fit in 40GB, so ask it to
# unload before our pinned GPU-0 instance loads. It reloads on next use.
evict_default_model() {
    if server_alive "127.0.0.1:11434"; then
        curl -s "http://127.0.0.1:11434/api/generate" \
            -d "{\"model\": \"$MODEL\", \"keep_alive\": 0}" >/dev/null 2>&1 || true
        echo "[setup] asked default server (11434) to unload $MODEL to free GPU 0 VRAM"
        sleep 3
    fi
}

start_server() {
    local gpu="$1" host="$2"
    local port="${host##*:}"
    if server_alive "$host"; then
        echo "[setup] reusing Ollama already serving on $host (GPU $gpu job slot)"
        return 0
    fi
    echo "[setup] starting Ollama pinned to GPU $gpu on $host"
    CUDA_VISIBLE_DEVICES="$gpu" OLLAMA_HOST="$host" OLLAMA_MODELS="$OLLAMA_MODELS_DIR" \
        "$OLLAMA_BIN" serve >"$LOG_DIR/ollama_gpu${gpu}_${port}.log" 2>&1 &
    STARTED_PIDS+=($!)
    local i
    for i in $(seq 1 60); do
        if server_alive "$host"; then
            return 0
        fi
        sleep 1
    done
    echo "ERROR: Ollama on $host did not become ready within 60s (see $LOG_DIR/ollama_gpu${gpu}_${port}.log)" >&2
    return 1
}

cleanup() {
    # Stop only the servers this script started; pre-existing servers are untouched.
    local pid
    for pid in "${STARTED_PIDS[@]:-}"; do
        [[ -n "$pid" ]] && kill "$pid" 2>/dev/null
    done
    rm -rf "$RUN_DIR"
}
trap cleanup EXIT
trap 'echo "interrupted"; exit 130' INT TERM

# ---------------------------------------------------------------------------
# Worker pool: one worker per GPU, dynamic job pulling via a locked counter
# ---------------------------------------------------------------------------
next_index() {
    (
        flock -x 9
        local n
        n=$(<"$COUNTER")
        echo $((n + 1)) >"$COUNTER"
        echo "$n"
    ) 9>>"$LOCK"
}

worker() {
    local gpu="$1" host="$2"
    while :; do
        local idx
        idx=$(next_index)
        if (( idx >= ${#JOB_NAMES[@]} )); then
            break
        fi
        local name="${JOB_NAMES[$idx]}"
        local log="$LOG_DIR/${name}.log"
        echo "[gpu$gpu] start: $name  (log: $log)"
        local t0=$SECONDS
        if run_job "$name" "$host" >"$log" 2>&1; then
            echo "[gpu$gpu] done : $name  ($((SECONDS - t0))s)"
        else
            echo "[gpu$gpu] FAIL : $name  ($((SECONDS - t0))s)  -- see $log"
            echo "$name" >>"$FAILED"
        fi
    done
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if [[ "$DRYRUN" != "1" ]]; then
    evict_default_model
    for i in "${!GPU_IDS[@]}"; do
        start_server "${GPU_IDS[$i]}" "${GPU_HOSTS[$i]}" || exit 1
    done
fi

echo "[run] ${#JOB_NAMES[@]} jobs on ${#GPU_IDS[@]} GPUs"
WORKER_PIDS=()
for i in "${!GPU_IDS[@]}"; do
    worker "${GPU_IDS[$i]}" "${GPU_HOSTS[$i]}" &
    WORKER_PIDS+=($!)
done
wait "${WORKER_PIDS[@]}"

if [[ -s "$FAILED" ]]; then
    echo "FAILED jobs:"
    sed 's/^/  - /' "$FAILED"
    exit 1
fi
echo "All ${#JOB_NAMES[@]} experiment runs finished successfully."
if [[ "$DRYRUN" == "1" ]]; then
    echo "[dry-run] inspect results under $OUT_DIR, then delete it."
fi
