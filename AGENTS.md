# Repository Guidelines

## Project Structure & Module Organization

This repository contains a Python clinical document pipeline plus supporting documentation. Primary scripts live at the repository root:

- `stage1_merge_chatml_all.py`: merges professor-level XLSX files into one patient CSV.
- `stage1_temporal_document_sort.py`: splits and orders source documents into `Sorted_Timeline`.
- `stage2_core_fact_extraction_verification.py`: runs Ollama-based extraction and verification.
- `docs/`: pipeline, command, and data-safety reference material.
- `outputs/`, `data/`, `raw/`, generated CSV/JSON/Markdown reports, and model artifacts are local-only and ignored.

Superseded exploratory scripts are intentionally ignored by `.gitignore`; prefer updating the stage-specific scripts above.

## Build, Test, and Development Commands

Install dependencies from the repository root:

```bash
python -m pip install -r requirements.txt
ollama pull qwen3.5:9b
```

Run Stage 1A merge:

```bash
python stage1_merge_chatml_all.py --input-dir /path/to/chatml_All --output-csv outputs/chatml_All_grouped_professor_patient.csv
```

Run a Stage 1B smoke test:

```bash
python stage1_temporal_document_sort.py --input-csv outputs/chatml_All_grouped_professor_patient.csv --output-csv outputs/stage1_first_row_temporal_sort.csv --max-patients 1
```

Run a Stage 2 smoke test:

```bash
python stage2_core_fact_extraction_verification.py --input-csv outputs/chatml_All_document_temporal_sorted.csv --output-csv outputs/stage2_10rows_fact_extraction_qwen35_9b.csv --extractor-model qwen3.5:9b --verifier-model qwen3.5:9b --max-patients 10 --max-iterations 2
```

See `docs/commands.md` for full-run options.

## Coding Style & Naming Conventions

Use Python 3 with 4-space indentation, `from __future__ import annotations`, `pathlib.Path` for filesystem paths, and `argparse` CLIs. Keep functions small and deterministic where possible. Use `snake_case` for functions, variables, CLI flags, and filenames; use `PascalCase` for dataclasses. Preserve UTF-8 handling for Korean column names and text.

## Testing Guidelines

There is no formal test suite yet. Validate changes with smoke runs before full processing: Stage 1B with `--max-patients 1`, then Stage 2 with `--max-patients 1` or `10`. Inspect output columns such as `Sorted_Timeline`, `Stage2_Status`, `Stage2_Approved`, and unresolved issues. Do not use real patient text in test fixtures unless it is approved and de-identified.

## Commit & Pull Request Guidelines

Existing commits use short imperative summaries, for example `Document medical report summarization pipeline`. Keep commits focused and avoid staging generated data. Pull requests should describe the affected stage, include the exact smoke command used, summarize output/schema changes, and link related issues. Add screenshots only for documentation/UI artifacts.

## Security & Configuration Tips

Treat raw records, generated outputs, and LLM logs as sensitive clinical data. Before pushing, run `git status --short` and confirm no `.xlsx`, `.csv`, `.jsonl`, or patient-level generated reports are staged. Follow `docs/data_policy.md`.
