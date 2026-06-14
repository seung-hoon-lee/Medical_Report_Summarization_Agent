#!/usr/bin/env python3
"""
Stage 1: deterministic temporal document sorting.

This version of Stage 1 does not summarize records and does not extract facts.
It splits each patient's Input into source documents, assigns each document a
representative document date when one is explicit, preserves internal reference
dates separately, sorts documents with a conservative clinical fallback, and
writes the raw document text back out.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_INPUT_CSV = Path("/root/seunghoon/project/outputs/chatml_All_grouped_professor_patient.csv")
DEFAULT_OUTPUT_JSON = Path("/root/seunghoon/project/outputs/stage1_first_row_temporal_sort.json")
DEFAULT_OUTPUT_CSV = Path("/root/seunghoon/project/outputs/stage1_first_row_temporal_sort.csv")
REQUIRED_COLUMNS = ["Professor_ID", "수술ID", "Input", "Output"]


@dataclass
class DateMention:
    """One date-like mention found inside a source document."""

    normalized_date: str
    raw_text: str
    role: str
    context: str


@dataclass
class SortedDocument:
    """One source document after temporal metadata assignment."""

    source_index: int | None
    original_order: int
    document_type: str
    document_date: str | None
    date_confidence: str
    date_source: str
    sort_group: str
    sort_key: str | None
    internal_dates: list[DateMention] = field(default_factory=list)
    raw_text: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sort patient source documents without LLM calls.")
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT_CSV)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument(
        "--skip-json",
        action="store_true",
        help="Write only the CSV output. Useful for full-dataset runs because JSON preserves extra raw metadata.",
    )
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-patients", type=int, default=1, help="Number of patients to process. Use 0 for all rows.")
    return parser.parse_args()


def load_input_csv(path: Path) -> pd.DataFrame:
    """Load the merged professor/patient CSV."""

    if not path.exists():
        raise FileNotFoundError(f"Input CSV does not exist: {path}")

    dataframe = pd.read_csv(path)
    missing = [column for column in REQUIRED_COLUMNS if column not in dataframe.columns]
    if missing:
        raise ValueError(f"Input CSV is missing required columns: {missing}")

    dataframe = dataframe[REQUIRED_COLUMNS].copy()
    dataframe["Professor_ID"] = dataframe["Professor_ID"].fillna("").astype(str)
    dataframe["수술ID"] = dataframe["수술ID"].fillna("").astype(str)
    dataframe["Input"] = dataframe["Input"].fillna("").astype(str)
    dataframe["Output"] = dataframe["Output"].fillna("").astype(str)
    return dataframe


def split_source_documents(text: str) -> list[tuple[int | None, str, str]]:
    """Split one patient Input into raw source documents."""

    header_pattern = re.compile(r"(?:^|\n)\[(\d+)\]\s+([^:\n]+):\s*\{\{")
    matches = list(header_pattern.finditer(text))
    if not matches:
        return [(None, "Unknown Document", text.strip())] if text.strip() else []

    documents: list[tuple[int | None, str, str]] = []
    for position, match in enumerate(matches):
        start = match.start()
        if text[start : start + 1] == "\n":
            start += 1
        end = matches[position + 1].start() if position + 1 < len(matches) else len(text)
        raw_text = text[start:end].strip()
        source_index = int(match.group(1))
        document_type = normalize_document_type(match.group(2))
        if is_empty_source_document(raw_text):
            continue
        documents.append((source_index, document_type, raw_text))
    return documents


def source_document_body(raw_text: str) -> str:
    """Return the text inside a source document's double-brace body."""

    marker_index = raw_text.find("{{")
    if marker_index < 0:
        return raw_text.strip()

    body = raw_text[marker_index + 2 :]
    closing_index = body.rfind("}}")
    if closing_index >= 0 and not body[closing_index + 2 :].strip():
        body = body[:closing_index]
    return body.strip()


def is_empty_source_document(raw_text: str) -> bool:
    """Return whether a source document has no meaningful body text."""

    return not source_document_body(raw_text)


def normalize_document_type(value: str) -> str:
    """Normalize document type whitespace while preserving the source label."""

    return re.sub(r"\s+", " ", value).strip() or "Unknown Document"


def normalize_year_month_day(year: int, month: int, day: int) -> str | None:
    """Return an ISO date if components form a plausible calendar date."""

    if not (1900 <= year <= 2099):
        return None
    try:
        return date(year, month, day).isoformat()
    except ValueError:
        return None


def nearby_context(text: str, start: int, end: int, window: int = 90) -> str:
    """Return compact text around one date mention."""

    snippet = text[max(0, start - window) : min(len(text), end + window)]
    return re.sub(r"\s+", " ", snippet).strip()


def nearest_year(text: str, position: int, window: int = 500) -> int | None:
    """Infer a nearby year for parenthesized month.day mentions."""

    context = text[max(0, position - window) : min(len(text), position + window)]
    years = [int(match.group(0)) for match in re.finditer(r"(?<!\d)(20\d{2}|19\d{2})(?!\d)", context)]
    if years:
        return years[-1]

    yymmdd_years = [
        2000 + int(match.group(1))
        for match in re.finditer(r"(?<!\d)(\d{2})(\d{2})(\d{2})(?!\d)", context)
        if normalize_year_month_day(2000 + int(match.group(1)), int(match.group(2)), int(match.group(3)))
    ]
    return yymmdd_years[-1] if yymmdd_years else None


def classify_date_role(document_type: str, context: str, raw_text: str) -> str:
    """Assign a conservative role to a date mention."""

    lowered = context.lower()
    raw_pattern = re.escape(raw_text.split()[0])
    if re.search(rf"compared\s+with\s+{raw_pattern}", lowered) or re.search(
        rf"comparison\s+.*{raw_pattern}", lowered
    ):
        return "comparison_date"
    if "퇴원일" in context:
        return "discharge_date"
    if "수술일" in context or "surg start" in lowered or "operation date" in lowered:
        return "operation_date"
    if "일자:" in context and "operative" in document_type.lower():
        return "operation_date"
    if "pft" in lowered or "pulmonary function" in lowered or "fvc" in lowered or "fev1" in lowered:
        return "study_date"
    if "wt:" in lowered or "bmi:" in lowered or "weight" in lowered:
        return "prior_measurement_date"
    if "chest ct" in lowered or "c.i.>" in lowered or "[finding]" in lowered:
        return "study_date"
    if "상담기록" in context or "통화" in context:
        return "note_date"
    return "date_mention"


def extract_date_mentions(text: str, document_type: str) -> list[DateMention]:
    """Extract normalized date mentions while preserving role/context metadata."""

    mentions: list[tuple[int, DateMention]] = []
    occupied: list[tuple[int, int]] = []

    def add_mention(start: int, end: int, normalized: str | None, raw: str) -> None:
        if not normalized:
            return
        context = nearby_context(text, start, end)
        role = classify_date_role(document_type, context, raw)
        mentions.append(
            (
                start,
                DateMention(
                    normalized_date=normalized,
                    raw_text=raw,
                    role=role,
                    context=context,
                ),
            )
        )
        occupied.append((start, end))

    def overlaps(start: int, end: int) -> bool:
        return any(start < span_end and end > span_start for span_start, span_end in occupied)

    for match in re.finditer(r"(?<!\d)(20\d{2}|19\d{2})[-/.](\d{1,2})[-/.](\d{1,2})(?!\d)", text):
        add_mention(
            match.start(),
            match.end(),
            normalize_year_month_day(int(match.group(1)), int(match.group(2)), int(match.group(3))),
            match.group(0),
        )

    for match in re.finditer(r"(?<!\d)(20\d{2}|19\d{2})\s+(\d{1,2})\s+(\d{1,2})(?!\d)", text):
        if overlaps(match.start(), match.end()):
            continue
        add_mention(
            match.start(),
            match.end(),
            normalize_year_month_day(int(match.group(1)), int(match.group(2)), int(match.group(3))),
            match.group(0),
        )

    for match in re.finditer(r"(?<!\d)(\d{2})(\d{2})(\d{2})(?!\d)", text):
        if overlaps(match.start(), match.end()):
            continue
        year = 2000 + int(match.group(1))
        add_mention(
            match.start(),
            match.end(),
            normalize_year_month_day(year, int(match.group(2)), int(match.group(3))),
            match.group(0),
        )

    for match in re.finditer(r"(?<![\d.])(\d{1,2})[.](\d{1,2})(?![\d.])", text):
        if overlaps(match.start(), match.end()):
            continue
        if match.start() == 0 or text[match.start() - 1] != "(":
            continue
        if match.end() >= len(text) or text[match.end()] != ")":
            continue
        year = nearest_year(text, match.start())
        normalized = (
            normalize_year_month_day(year, int(match.group(1)), int(match.group(2)))
            if year is not None
            else None
        )
        add_mention(match.start(), match.end(), normalized, f"{match.group(0)} (year inferred)")

    deduped: dict[tuple[str, str, str], DateMention] = {}
    for _, mention in sorted(mentions, key=lambda item: item[0]):
        key = (mention.normalized_date, mention.raw_text, mention.role)
        deduped.setdefault(key, mention)
    return list(deduped.values())


def select_document_date(document_type: str, mentions: list[DateMention]) -> tuple[str | None, str, str]:
    """Pick the representative document date, not an internal reference date."""

    lowered_type = document_type.lower()
    if "operative" in lowered_type or "operation" in lowered_type:
        for mention in mentions:
            if mention.role == "operation_date":
                return mention.normalized_date, "explicit_operation_date", mention.context

    if "discharge" in lowered_type:
        for mention in mentions:
            if mention.role == "discharge_date":
                return mention.normalized_date, "explicit_discharge_date", mention.context

    if "outpatient" in lowered_type or "consult" in lowered_type or "clinic" in lowered_type:
        for mention in mentions:
            if mention.role == "note_date":
                return mention.normalized_date, "explicit_note_date", mention.context

    excluded_roles = {"comparison_date", "prior_measurement_date"}
    for mention in mentions:
        if mention.role not in excluded_roles:
            return mention.normalized_date, f"fallback_{mention.role}", mention.context

    return None, "missing", "No explicit representative document date found."


def clinical_phase_rank(document_type: str) -> int:
    """Return a fallback clinical-order rank for documents with weak dates."""

    lowered = document_type.lower()
    if "initial" in lowered or "first" in lowered:
        return 10
    if "preoperative" in lowered or "pre-op" in lowered or "outpatient" in lowered:
        return 20
    if "admission" in lowered:
        return 30
    if "operative" in lowered or "operation" in lowered:
        return 40
    if "discharge" in lowered:
        return 50
    if "follow" in lowered:
        return 60
    return 90


def sort_tuple(document: SortedDocument) -> tuple[int, int, str, int]:
    """Sort by clinical document phase, then explicit date, then source order."""

    missing_date = 1 if document.document_date is None else 0
    return (
        clinical_phase_rank(document.document_type),
        missing_date,
        document.document_date or "9999-12-31",
        document.original_order,
    )


def sort_patient_documents(row: pd.Series, source_row_index: int) -> dict[str, Any]:
    """Build one patient-level Stage 1 result."""

    documents: list[SortedDocument] = []
    for original_order, (source_index, document_type, raw_text) in enumerate(
        split_source_documents(str(row["Input"])),
        start=1,
    ):
        mentions = extract_date_mentions(raw_text, document_type)
        document_date, confidence, date_source = select_document_date(document_type, mentions)
        sort_group = "explicit_document_date" if document_date else "undated_clinical_fallback"
        documents.append(
            SortedDocument(
                source_index=source_index,
                original_order=original_order,
                document_type=document_type,
                document_date=document_date,
                date_confidence=confidence,
                date_source=date_source,
                sort_group=sort_group,
                sort_key=document_date,
                internal_dates=mentions,
                raw_text=raw_text,
            )
        )

    sorted_documents = sorted(documents, key=sort_tuple)
    sorted_input = "\n\n".join(document.raw_text for document in sorted_documents)
    return {
        "Professor_ID": str(row["Professor_ID"]),
        "Patient_ID": str(row["수술ID"]),
        "source_row_index": source_row_index,
        "num_documents": len(sorted_documents),
        "original_input": str(row["Input"]),
        "sorting_strategy": (
            "Sort source documents by conservative clinical phase and explicit representative "
            "document_date. Preserve all raw document text; keep internal/reference dates as metadata."
        ),
        "sorted_documents": [
            {
                **asdict(document),
                "internal_dates": [asdict(mention) for mention in document.internal_dates],
            }
            for document in sorted_documents
        ],
        "sorted_input": sorted_input,
    }


def write_outputs(
    results: list[dict[str, Any]],
    output_json: Path,
    output_csv: Path,
    input_csv: Path,
    skip_json: bool = False,
) -> None:
    """Write JSON plus a compact CSV view."""

    if not skip_json:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "metadata": {
                "stage": "stage2_temporal_document_sort",
                "input_csv": str(input_csv),
                "num_patients": len(results),
                "llm_used": False,
            },
            "patients": results,
        }
        output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    rows = []
    for result in results:
        rows.append(
            {
                "Professor_ID": result["Professor_ID"],
                "수술ID": result["Patient_ID"],
                "Input": result["original_input"],
                "Sorted_Timeline": result["sorted_input"],
            }
        )
    pd.DataFrame(rows).to_csv(output_csv, index=False, encoding="utf-8-sig")


def main() -> None:
    args = parse_args()
    dataframe = load_input_csv(args.input_csv)
    end_index = len(dataframe) if args.max_patients <= 0 else min(len(dataframe), args.start_index + args.max_patients)
    indices = list(range(args.start_index, end_index))
    results = [sort_patient_documents(dataframe.loc[index], index) for index in indices]
    write_outputs(results, args.output_json, args.output_csv, args.input_csv, args.skip_json)

    print(f"[INFO] Input CSV: {args.input_csv}")
    if not args.skip_json:
        print(f"[INFO] Output JSON: {args.output_json}")
    print(f"[INFO] Output CSV: {args.output_csv}")
    print(f"[INFO] Patients written: {len(results)}")


if __name__ == "__main__":
    main()
