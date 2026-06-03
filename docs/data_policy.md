# Data Policy

This repository is intended to store source code and documentation only.

## Do Not Commit

Do not commit:

- raw clinical records
- `.xlsx`, `.xls`, `.csv`, `.json`, `.jsonl` patient-level data
- generated Stage 1 or Stage 2 outputs
- model outputs containing patient facts
- local environment files
- cache directories

The `.gitignore` is intentionally broad:

```text
outputs/
data/
datasets/
raw/
*.xlsx
*.xls
*.csv
*.jsonl
```

## Local Data Layout

Use local-only directories for data:

```text
outputs/
data/
raw/
```

These paths are ignored by git.

## Public Documentation

When documenting examples:

- Use synthetic or path-only examples.
- Do not paste patient-level text.
- Do not include real `Professor_ID` + `수술ID` combinations.
- Do not include generated facts from real records unless explicitly de-identified
  and approved for release.

## Model and Log Outputs

LLM responses may contain protected clinical details. Treat all generated
outputs as sensitive derived data unless they have been reviewed and
de-identified.

## Pre-Push Checklist

Before pushing:

```bash
git status --short
git check-ignore -v outputs/* 2>/dev/null | head
find . -maxdepth 3 -type f \( -name "*.csv" -o -name "*.xlsx" -o -name "*.jsonl" \)
```

Only code and documentation should be staged.

