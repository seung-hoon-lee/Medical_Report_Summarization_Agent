# Commands

All commands assume the repository root as the working directory.

```bash
cd /path/to/Medical_Report_Summarization_Agent
```

## Install

```bash
python -m pip install -r requirements.txt
ollama pull qwen3.5:9b
```

## Stage 1A: Merge XLSX Files

```bash
python stage1_merge_chatml_all.py \
  --input-dir /path/to/chatml_All \
  --output-csv outputs/chatml_All_grouped_professor_patient.csv
```

Arguments:

| Argument | Default | Description |
| --- | --- | --- |
| `--input-dir` | `/root/JS/training_dataset/chatml_All` | Directory containing professor-level `.xlsx` files |
| `--output-csv` | `outputs/chatml_All_grouped_professor_patient.csv` | Merged patient-level CSV |
| `--encoding` | `utf-8-sig` | CSV encoding |
| `--keep-empty-output` / `--no-keep-empty-output` | enabled | Keep rows with empty `Output` |
| `--sort` / `--no-sort` | enabled | Sort by `Professor_ID`, `수술ID` |

## Stage 1B: Document Temporal Sorting

First row smoke test:

```bash
python stage1_temporal_document_sort.py \
  --input-csv outputs/chatml_All_grouped_professor_patient.csv \
  --output-csv outputs/stage1_first_row_temporal_sort.csv \
  --output-json outputs/stage1_first_row_temporal_sort.json \
  --max-patients 1
```

Full dataset:

```bash
python stage1_temporal_document_sort.py \
  --input-csv outputs/chatml_All_grouped_professor_patient.csv \
  --output-csv outputs/chatml_All_document_temporal_sorted.csv \
  --output-json outputs/stage1_temporal_sort_metadata.json \
  --skip-json \
  --max-patients 0
```

Arguments:

| Argument | Default | Description |
| --- | --- | --- |
| `--input-csv` | `outputs/chatml_All_grouped_professor_patient.csv` | Stage 1A merged CSV |
| `--output-csv` | `outputs/stage1_first_row_temporal_sort.csv` | Compact CSV with `Sorted_Timeline` |
| `--output-json` | `outputs/stage1_first_row_temporal_sort.json` | Optional metadata-rich JSON |
| `--skip-json` | false | Skip JSON output for full-dataset runs |
| `--start-index` | `0` | First row index |
| `--max-patients` | `1` | Number of rows. Use `0` for all rows |

## Stage 2: Core Fact Extraction and Verification

First row:

```bash
python stage2_core_fact_extraction_verification.py \
  --input-csv outputs/chatml_All_document_temporal_sorted.csv \
  --output-csv outputs/stage2_first_row_fact_extraction_qwen35_9b.csv \
  --extractor-model qwen3.5:9b \
  --verifier-model qwen3.5:9b \
  --max-patients 1 \
  --max-iterations 2 \
  --num-ctx 12000 \
  --num-predict 4096 \
  --save-every 1
```

10-row smoke test:

```bash
python stage2_core_fact_extraction_verification.py \
  --input-csv outputs/chatml_All_document_temporal_sorted.csv \
  --output-csv outputs/stage2_10rows_fact_extraction_qwen35_9b.csv \
  --extractor-model qwen3.5:9b \
  --verifier-model qwen3.5:9b \
  --max-patients 10 \
  --max-iterations 2 \
  --num-ctx 12000 \
  --num-predict 4096 \
  --save-every 1
```

Full dataset:

```bash
python stage2_core_fact_extraction_verification.py \
  --input-csv outputs/chatml_All_document_temporal_sorted.csv \
  --output-csv outputs/stage2_all_fact_extraction_qwen35_9b.csv \
  --extractor-model qwen3.5:9b \
  --verifier-model qwen3.5:9b \
  --max-patients 0 \
  --max-iterations 2 \
  --coverage-threshold 0.85 \
  --evidence-threshold 0.95 \
  --num-ctx 12000 \
  --num-predict 4096 \
  --save-every 10 \
  --skip-readable-report
```

Arguments:

| Argument | Default | Description |
| --- | --- | --- |
| `--input-csv` | `outputs/chatml_All_document_temporal_sorted.csv` | Stage 1B output |
| `--output-csv` | `outputs/stage2_first_row_fact_extraction.csv` | Stage 2 output CSV |
| `--extractor-model` | `qwen3.5:9b` | Agent 1 Ollama model |
| `--verifier-model` | `qwen3.5:9b` | Agent 2 Ollama model |
| `--temperature` | `0.0` | Deterministic generation |
| `--num-ctx` | `16384` | Context window requested from Ollama |
| `--num-predict` | `4096` | Max generated tokens per LLM call |
| `--timeout` | `900` | Ollama request timeout in seconds |
| `--retries` | `2` | Retries per LLM JSON call |
| `--retry-sleep` | `5` | Backoff base seconds |
| `--max-iterations` | `2` | Max extraction-verification loops per chunk |
| `--coverage-threshold` | `0.85` | Minimum coverage score for PASS |
| `--evidence-threshold` | `0.95` | Minimum evidence support score for PASS |
| `--start-index` | `0` | First row index |
| `--max-patients` | `1` | Number of rows. Use `0` for all rows |
| `--save-every` | `10` | Partial CSV save interval. Use `0` to disable |
| `--skip-readable-report` | false | Skip full markdown report generation |

## Recommended Batch Strategy

1. Run Stage 1A once.
2. Run Stage 1B on the full dataset with `--skip-json`.
3. Run Stage 2 on 10 rows.
4. Inspect `Stage2_Status`, `Stage2_Approved`, and unresolved issues.
5. Run Stage 2 full dataset with `--skip-readable-report`.

