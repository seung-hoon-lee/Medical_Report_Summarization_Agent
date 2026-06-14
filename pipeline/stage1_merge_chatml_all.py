#!/usr/bin/env python3
"""
Merge professor-specific chatml_All XLSX files into one patient-level CSV.

Input files:
  /root/JS/training_dataset/chatml_All/*_All.xlsx

Each XLSX file is assumed to correspond to one professor and must contain:
  수술ID, Input, Output

Output CSV columns:
  Professor_ID, 수술ID, Input, Output
"""

from __future__ import annotations

import argparse
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


DEFAULT_INPUT_DIR = Path("/root/JS/training_dataset/chatml_All")
DEFAULT_OUTPUT_CSV = Path("/root/seunghoon/project/outputs/chatml_All_grouped_professor_patient.csv")
REQUIRED_COLUMNS = ["수술ID", "Input", "Output"]
OUTPUT_COLUMNS = ["Professor_ID", "수술ID", "Input", "Output"]


@dataclass
class FileMergeSummary:
    """Simple per-file merge summary for logging."""

    source_file: str
    professor_id: str
    input_rows: int
    output_rows: int
    skipped_rows: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge chatml_All professor XLSX files into one CSV."
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument(
        "--encoding",
        default="utf-8-sig",
        help="CSV encoding. utf-8-sig is convenient for Excel.",
    )
    parser.add_argument(
        "--keep-empty-output",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep rows even when Output is empty; empty Output becomes ''.",
    )
    parser.add_argument(
        "--sort",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Sort final CSV by Professor_ID and 수술ID.",
    )
    return parser.parse_args()


def normalize_text(value: object) -> str:
    """Normalize a value into a clean string."""

    text = "" if value is None else str(value).strip()
    return unicodedata.normalize("NFC", text)


def professor_id_from_filename(path: Path) -> str:
    """Infer Professor_ID from filename, e.g. 강창현_All.xlsx -> 강창현."""

    stem = unicodedata.normalize("NFC", path.stem)
    professor_id = re.sub(r"_All$", "", stem)
    return normalize_text(professor_id)


def read_and_normalize_xlsx(
    path: Path,
    keep_empty_output: bool,
) -> tuple[pd.DataFrame | None, FileMergeSummary | None]:
    """Read one XLSX and normalize it to OUTPUT_COLUMNS."""

    try:
        dataframe = pd.read_excel(path)
    except Exception as exc:
        print(f"[WARN] Skipping unreadable file: {path} ({exc})")
        return None, None

    missing_columns = [column for column in REQUIRED_COLUMNS if column not in dataframe.columns]
    if missing_columns:
        print(f"[WARN] Skipping {path.name}: missing columns {missing_columns}")
        return None, None

    professor_id = professor_id_from_filename(path)
    input_rows = len(dataframe)

    merged = dataframe[REQUIRED_COLUMNS].copy()
    merged.insert(0, "Professor_ID", professor_id)

    merged["Professor_ID"] = merged["Professor_ID"].map(normalize_text)
    merged["수술ID"] = merged["수술ID"].map(normalize_text)
    merged["Input"] = merged["Input"].map(normalize_text)

    if keep_empty_output:
        merged["Output"] = merged["Output"].fillna("").map(normalize_text)
    else:
        merged["Output"] = merged["Output"].map(normalize_text)
        merged = merged[merged["Output"].str.len() > 0]

    merged = merged[OUTPUT_COLUMNS]
    merged = merged[
        (merged["Professor_ID"].str.len() > 0)
        & (merged["수술ID"].str.len() > 0)
        & (merged["Input"].str.len() > 0)
    ].copy()

    summary = FileMergeSummary(
        source_file=str(path),
        professor_id=professor_id,
        input_rows=input_rows,
        output_rows=len(merged),
        skipped_rows=input_rows - len(merged),
    )
    return merged, summary


def merge_chatml_all(
    input_dir: Path,
    keep_empty_output: bool = True,
    sort_output: bool = True,
) -> tuple[pd.DataFrame, list[FileMergeSummary]]:
    """Merge every XLSX file in input_dir into one DataFrame."""

    xlsx_paths = sorted(input_dir.glob("*.xlsx"))
    if not xlsx_paths:
        raise FileNotFoundError(f"No .xlsx files found in {input_dir}")

    frames: list[pd.DataFrame] = []
    summaries: list[FileMergeSummary] = []

    for path in xlsx_paths:
        frame, summary = read_and_normalize_xlsx(
            path=path,
            keep_empty_output=keep_empty_output,
        )
        if frame is None or summary is None:
            continue
        frames.append(frame)
        summaries.append(summary)

    if not frames:
        raise RuntimeError(f"No valid XLSX files were merged from {input_dir}")

    combined = pd.concat(frames, ignore_index=True)
    if sort_output:
        combined = combined.sort_values(["Professor_ID", "수술ID"], kind="stable")
    combined = combined.reset_index(drop=True)
    return combined, summaries


def validate_merged_dataframe(dataframe: pd.DataFrame) -> dict[str, int]:
    """Return validation metrics for the merged CSV."""

    return {
        "num_rows": int(len(dataframe)),
        "num_professors": int(dataframe["Professor_ID"].nunique()),
        "num_unique_professor_patient": int(
            dataframe[["Professor_ID", "수술ID"]].drop_duplicates().shape[0]
        ),
        "duplicate_professor_patient": int(
            dataframe.duplicated(["Professor_ID", "수술ID"]).sum()
        ),
        "null_values": int(dataframe[OUTPUT_COLUMNS].isna().sum().sum()),
    }


def write_csv(dataframe: pd.DataFrame, output_csv: Path, encoding: str) -> None:
    """Write the merged DataFrame to CSV."""

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(output_csv, index=False, encoding=encoding)


def main() -> None:
    args = parse_args()

    combined, summaries = merge_chatml_all(
        input_dir=args.input_dir,
        keep_empty_output=args.keep_empty_output,
        sort_output=args.sort,
    )
    metrics = validate_merged_dataframe(combined)
    write_csv(combined, args.output_csv, args.encoding)

    print(f"[INFO] Input directory: {args.input_dir}")
    print(f"[INFO] Output CSV: {args.output_csv}")
    print(f"[INFO] Files merged: {len(summaries)}")
    print(f"[INFO] Rows: {metrics['num_rows']}")
    print(f"[INFO] Professors: {metrics['num_professors']}")
    print(f"[INFO] Unique Professor_ID + 수술ID: {metrics['num_unique_professor_patient']}")
    print(f"[INFO] Duplicate Professor_ID + 수술ID: {metrics['duplicate_professor_patient']}")
    print(f"[INFO] Null values in output columns: {metrics['null_values']}")


if __name__ == "__main__":
    main()
