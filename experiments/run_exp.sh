#!/usr/bin/env bash
# Paper comparison experiments: 5 methods + exp5 few-shot reference ablation.
#
#   Method                                Fact Extraction  Iterative Agent  Few-shot Style
#   exp1  Raw-to-Note                          x                 x                x
#   exp2  Raw-to-Fact-to-Note                  o                 x                x
#   exp3  Chunk-to-Fact-to-Note                o                 x                x
#   exp4  Iterative Multi-Agent Fact-to-Note   o                 o                x
#   exp5  Ours                                 o                 o                o
#
# All runs use --resume: safe to re-run this script after an interruption;
# already-processed rows are skipped and generation continues.

cd "$(dirname "$0")"

# ---------------------------------------------------------------------------
# Baselines (style-free single agent)
# ---------------------------------------------------------------------------

# python exp1_raw_to_note.py \
#   --model qwen3.6:35b \
#   --output_csv outputs/exp1_raw_to_note.csv \
#   --audit_jsonl outputs/exp1_raw_to_note_audit.jsonl \
#   --resume

# python exp2_raw_fact_to_note.py \
#   --model qwen3.6:35b \
#   --output_csv outputs/exp2_raw_fact_to_note.csv \
#   --audit_jsonl outputs/exp2_raw_fact_to_note_audit.jsonl \
#   --resume

# python exp3_chunk_fact_to_note.py \
#   --model qwen3.6:35b \
#   --output_csv outputs/exp3_chunk_fact_to_note.csv \
#   --audit_jsonl outputs/exp3_chunk_fact_to_note_audit.jsonl \
#   --resume

python exp4_iterative_fact_to_note.py \
  --model qwen3.6:35b \
  --output_csv outputs/exp4_iterative_fact_to_note.csv \
  --audit_jsonl outputs/exp4_iterative_fact_to_note_audit.jsonl \
  --resume

# ---------------------------------------------------------------------------
# Ours (exp5): main configuration
#   k=5, deterministic (length-stratified) reference selection
#   Output paths are auto-suffixed by config:
#   -> outputs/exp5_ours_k5_deterministic.csv
# ---------------------------------------------------------------------------

python exp5_ours_fewshot_style.py \
  --model qwen3.6:35b \
  --sample_count 5 --reference_selection deterministic \
  --resume

# ---------------------------------------------------------------------------
# Ours (exp5): few-shot reference ablation
#   Axis 1 - k:         3 vs 5
#   Axis 2 - selection: deterministic vs random (seeded, per-professor keyed)
#   -> outputs/exp5_ours_k3_deterministic.csv
#   -> outputs/exp5_ours_k5_random_seed1225.csv
#   -> outputs/exp5_ours_k3_random_seed1225.csv
# ---------------------------------------------------------------------------

python exp5_ours_fewshot_style.py \
  --model qwen3.6:35b \
  --sample_count 3 --reference_selection deterministic \
  --resume

python exp5_ours_fewshot_style.py \
  --model qwen3.6:35b \
  --sample_count 5 --reference_selection random --reference_seed 1225 \
  --resume

python exp5_ours_fewshot_style.py \
  --model qwen3.6:35b \
  --sample_count 3 --reference_selection random --reference_seed 1225 \
  --resume

# Optional: extra random seeds to report variance over reference draws.
# python exp5_ours_fewshot_style.py --model qwen3.6:35b --sample_count 5 --reference_selection random --reference_seed 7  --resume
# python exp5_ours_fewshot_style.py --model qwen3.6:35b --sample_count 5 --reference_selection random --reference_seed 42 --resume
# python exp5_ours_fewshot_style.py --model qwen3.6:35b --sample_count 3 --reference_selection random --reference_seed 7  --resume
# python exp5_ours_fewshot_style.py --model qwen3.6:35b --sample_count 3 --reference_selection random --reference_seed 42 --resume

echo "All experiments finished."
