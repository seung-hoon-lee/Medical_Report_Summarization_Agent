#!/usr/bin/env python3
"""Streamlit-facing adapters for the medical report pipeline.

The existing project is intentionally CLI-first. This module keeps the web
layer thin: normalize uploads into the schemas that the stage scripts already
accept, create timestamped run directories, and call the scripts as subprocesses.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
import time
import unicodedata
import uuid
import zipfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any, Iterable, Sequence
from xml.etree import ElementTree

import pandas as pd


WEB_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = WEB_DIR.parent
WEB_RUNS_ROOT = PROJECT_ROOT / "outputs" / "web_runs"
INPUT_CSV_NAME = "uploaded_patient_inputs.csv"
STYLE_SAMPLE_SUFFIX = "_Samples.csv"

REQUIRED_INPUT_COLUMNS = ("Professor_ID", "수술ID", "Input", "Output")
OUTPUT_COLUMN_CANDIDATES = (
    "reference",
    "Reference",
    "references",
    "References",
    "output",
    "Output",
    "OUTPUT",
    "actual_output",
    "Actual_Output",
    "reference_output",
    "Reference_Output",
    "ground_truth",
    "Ground_Truth",
    "GT",
    "note",
    "Note",
    "outpatient_note",
    "Outpatient_Note",
    "외래기록지",
    "퇴원기록지",
    "실제외래기록지",
)
INPUT_COLUMN_CANDIDATES = (
    "input",
    "Input",
    "INPUT",
    "source",
    "Source",
    "original_record",
    "Original_Record",
    "medical_record",
    "Medical_Record",
    "의료기록지",
    "원본의료기록지",
)
RECORD_ID_COLUMN_CANDIDATES = (
    "수술ID",
    "환자 ID",
    "환자ID",
    "Patient_ID",
    "patient_id",
    "record_id",
    "Record_ID",
    "No.",
    "No",
    "no",
)
SUPPORTED_TEXT_EXTENSIONS = {".txt", ".md", ".json", ".jsonl", ".xml", ".html"}
SUPPORTED_TABLE_EXTENSIONS = {".csv", ".xlsx", ".xls"}


@dataclass(frozen=True)
class UploadedPayload:
    """Small serializable representation of a Streamlit upload."""

    name: str
    data: bytes


@dataclass(frozen=True)
class RunPaths:
    """All file-system locations for one web run."""

    run_id: str
    root: Path
    uploads_dir: Path
    input_uploads_dir: Path
    style_uploads_dir: Path
    prof_samples_dir: Path
    outputs_dir: Path
    logs_dir: Path
    manifest_path: Path
    input_csv: Path
    stage1_csv: Path
    stage1_json: Path
    stage2_csv: Path
    reference_csv: Path
    style_cache_jsonl: Path
    stage4_csv: Path
    stage4_audit_jsonl: Path


@dataclass(frozen=True)
class CommandResult:
    """Captured subprocess result for display and audit."""

    command: list[str]
    cwd: str
    returncode: int
    stdout: str
    stderr: str
    duration_seconds: float

    @property
    def ok(self) -> bool:
        return self.returncode == 0

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["ok"] = self.ok
        return payload


@dataclass(frozen=True)
class PreparedRun:
    """Result of normalizing uploads into a run workspace."""

    paths: RunPaths
    input_rows: int
    style_sample_rows: int
    warnings: list[str]
    saved_files: list[dict[str, Any]]

    def to_manifest(self) -> dict[str, Any]:
        return {
            "run_id": self.paths.run_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "paths": {key: str(value) for key, value in asdict(self.paths).items()},
            "input_rows": self.input_rows,
            "style_sample_rows": self.style_sample_rows,
            "warnings": self.warnings,
            "saved_files": self.saved_files,
        }


def safe_slug(value: str, default: str = "run", max_length: int = 64) -> str:
    """Return a filesystem-safe slug while preserving Korean labels."""

    normalized = unicodedata.normalize("NFC", value or "").strip()
    normalized = re.sub(r"\s+", "_", normalized)
    normalized = re.sub(r"[^\w가-힣.-]+", "_", normalized, flags=re.UNICODE)
    normalized = normalized.strip("._-")
    if not normalized:
        normalized = default
    return normalized[:max_length]


def short_hash(data: bytes | str, length: int = 10) -> str:
    material = data.encode("utf-8-sig") if isinstance(data, str) else data
    return hashlib.sha256(material).hexdigest()[:length]


def create_run_paths(label: str) -> RunPaths:
    WEB_RUNS_ROOT.mkdir(parents=True, exist_ok=True)
    root: Path | None = None
    run_id = ""
    for _ in range(8):
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        run_id = f"{timestamp}_{safe_slug(label, default='web_run', max_length=36)}_{uuid.uuid4().hex[:8]}"
        candidate = WEB_RUNS_ROOT / run_id
        try:
            candidate.mkdir(parents=False, exist_ok=False)
        except FileExistsError:
            continue
        root = candidate
        break
    if root is None:
        raise RuntimeError("Could not create a unique run workspace")
    paths = RunPaths(
        run_id=run_id,
        root=root,
        uploads_dir=root / "uploads",
        input_uploads_dir=root / "uploads" / "input_documents",
        style_uploads_dir=root / "uploads" / "style_samples",
        prof_samples_dir=root / "prof_samples",
        outputs_dir=root / "outputs",
        logs_dir=root / "logs",
        manifest_path=root / "manifest.json",
        input_csv=root / INPUT_CSV_NAME,
        stage1_csv=root / "outputs" / "stage1_sorted_timeline.csv",
        stage1_json=root / "outputs" / "stage1_sort_metadata.json",
        stage2_csv=root / "outputs" / "stage2_verified_facts.csv",
        reference_csv=root / "reference_examples.csv",
        style_cache_jsonl=root / "outputs" / "fewshot_professor_style_prompts.jsonl",
        stage4_csv=root / "outputs" / "generated_notes.csv",
        stage4_audit_jsonl=root / "outputs" / "generated_notes_audit.jsonl",
    )
    for directory in (
        paths.input_uploads_dir,
        paths.style_uploads_dir,
        paths.prof_samples_dir,
        paths.outputs_dir,
        paths.logs_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)
    return paths


def decode_text(data: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-8", "cp949", "euc-kr", "latin-1"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def clean_uploaded_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() in {"nan", "none", "nat"}:
        return ""
    return text.replace("\\r\\n", "\n").replace("\\n", "\n").replace("\\t", "\t")


def dataframe_to_text(dataframe: pd.DataFrame, title: str) -> str:
    rows = [f"[Source table: {title}]", ""]
    preview = dataframe.fillna("").astype(str)
    for index, row in preview.iterrows():
        rows.append(f"Row {index + 1}")
        for column, value in row.items():
            value_text = clean_uploaded_text(value)
            if value_text:
                rows.append(f"- {column}: {value_text}")
        rows.append("")
    return "\n".join(rows).strip()


def read_docx_text(data: bytes) -> str:
    """Extract text from a .docx file without adding a dependency."""

    with zipfile.ZipFile(BytesIO(data)) as archive:
        xml = archive.read("word/document.xml")
    root = ElementTree.fromstring(xml)
    namespace = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    paragraphs: list[str] = []
    for paragraph in root.findall(".//w:p", namespace):
        texts = [node.text or "" for node in paragraph.findall(".//w:t", namespace)]
        line = "".join(texts).strip()
        if line:
            paragraphs.append(line)
    return "\n".join(paragraphs)


def choose_column(dataframe: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    columns = [str(column) for column in dataframe.columns]
    for candidate in candidates:
        if candidate in columns:
            return candidate
    normalized = {
        re.sub(r"[\s_]+", "", column).lower(): column
        for column in columns
    }
    for candidate in candidates:
        key = re.sub(r"[\s_]+", "", candidate).lower()
        if key in normalized:
            return normalized[key]
    return None


def choose_record_id(row: pd.Series, fallback: str) -> str:
    for column in RECORD_ID_COLUMN_CANDIDATES:
        if column not in row.index:
            continue
        value = str(row.get(column, "")).strip()
        if value:
            return value
    return fallback


def rows_from_dataframe(
    dataframe: pd.DataFrame,
    source_name: str,
    default_professor: str,
    output_type: str,
) -> list[dict[str, str]]:
    dataframe = dataframe.fillna("").astype(str)
    required_present = all(column in dataframe.columns for column in REQUIRED_INPUT_COLUMNS[:3])
    if required_present:
        rows: list[dict[str, str]] = []
        for index, row in dataframe.iterrows():
            input_text = clean_uploaded_text(row.get("Input", ""))
            if not input_text:
                continue
            rows.append(
                {
                    "Professor_ID": str(row.get("Professor_ID", default_professor)).strip() or default_professor,
                    "수술ID": choose_record_id(row, f"{safe_slug(Path(source_name).stem)}_{index + 1}"),
                    "Input": input_text,
                    "Output": clean_uploaded_text(row.get("Output", "")),
                    "Output_Type": output_type,
                    "Source_File": source_name,
                }
            )
        return rows

    input_col = choose_column(dataframe, INPUT_COLUMN_CANDIDATES)
    output_col = choose_column(dataframe, OUTPUT_COLUMN_CANDIDATES)
    if input_col:
        rows = []
        for index, row in dataframe.iterrows():
            input_text = clean_uploaded_text(row.get(input_col, ""))
            if not input_text:
                continue
            rows.append(
                {
                    "Professor_ID": default_professor,
                    "수술ID": choose_record_id(row, f"{safe_slug(Path(source_name).stem)}_{index + 1}"),
                    "Input": input_text,
                    "Output": clean_uploaded_text(row.get(output_col, "")) if output_col else "",
                    "Output_Type": output_type,
                    "Source_File": source_name,
                }
            )
        return rows

    rendered = dataframe_to_text(dataframe, source_name)
    if not rendered:
        return []
    return [
        {
            "Professor_ID": default_professor,
            "수술ID": f"{safe_slug(Path(source_name).stem)}_{short_hash(rendered, 6)}",
            "Input": rendered,
            "Output": "",
            "Output_Type": output_type,
            "Source_File": source_name,
        }
    ]


def rows_from_json(
    data: bytes,
    source_name: str,
    default_professor: str,
    output_type: str,
) -> list[dict[str, str]]:
    text = decode_text(data)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return single_text_row(text, source_name, default_professor, output_type)

    if isinstance(payload, dict):
        payload_rows = payload.get("rows") or payload.get("records") or payload.get("patients")
        if isinstance(payload_rows, list):
            payload = payload_rows

    if isinstance(payload, list) and all(isinstance(item, dict) for item in payload):
        frame = pd.DataFrame(payload)
        return rows_from_dataframe(frame, source_name, default_professor, output_type)

    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    return single_text_row(rendered, source_name, default_professor, output_type)


def single_text_row(
    text: str,
    source_name: str,
    default_professor: str,
    output_type: str,
) -> list[dict[str, str]]:
    cleaned = clean_uploaded_text(text)
    if not cleaned:
        return []
    return [
        {
            "Professor_ID": default_professor,
            "수술ID": f"{safe_slug(Path(source_name).stem)}_{short_hash(cleaned, 6)}",
            "Input": cleaned,
            "Output": "",
            "Output_Type": output_type,
            "Source_File": source_name,
        }
    ]


def payload_to_input_rows(
    payload: UploadedPayload,
    default_professor: str,
    output_type: str,
) -> tuple[list[dict[str, str]], list[str]]:
    suffix = Path(payload.name).suffix.lower()
    warnings: list[str] = []
    try:
        if suffix == ".csv":
            frame = pd.read_csv(BytesIO(payload.data), dtype=str, keep_default_na=False)
            return rows_from_dataframe(frame, payload.name, default_professor, output_type), warnings
        if suffix in {".xlsx", ".xls"}:
            sheets = pd.read_excel(BytesIO(payload.data), sheet_name=None, dtype=str).items()
            rows: list[dict[str, str]] = []
            for sheet_name, frame in sheets:
                sheet_rows = rows_from_dataframe(
                    frame,
                    f"{payload.name}#{sheet_name}",
                    default_professor,
                    output_type,
                )
                rows.extend(sheet_rows)
            return rows, warnings
        if suffix == ".json":
            return rows_from_json(payload.data, payload.name, default_professor, output_type), warnings
        if suffix == ".docx":
            text = read_docx_text(payload.data)
            return single_text_row(text, payload.name, default_professor, output_type), warnings
        if suffix in SUPPORTED_TEXT_EXTENSIONS or not suffix:
            return single_text_row(decode_text(payload.data), payload.name, default_professor, output_type), warnings
    except Exception as exc:  # noqa: BLE001 - uploaded file formats vary.
        warnings.append(f"{payload.name}: could not parse upload ({exc})")
        return [], warnings

    warnings.append(f"{payload.name}: saved, but text extraction is not supported for {suffix or 'this file type'}")
    return [], warnings


def extract_style_rows(payload: UploadedPayload) -> tuple[list[dict[str, str]], list[str]]:
    suffix = Path(payload.name).suffix.lower()
    warnings: list[str] = []

    def from_frame(frame: pd.DataFrame, source: str) -> list[dict[str, str]]:
        frame = frame.fillna("").astype(str)
        output_col = choose_column(frame, OUTPUT_COLUMN_CANDIDATES)
        input_col = choose_column(frame, INPUT_COLUMN_CANDIDATES)
        if not output_col:
            warnings.append(f"{source}: no output/reference-note column detected")
            return []
        rows = []
        for index, row in frame.iterrows():
            output_text = clean_uploaded_text(row.get(output_col, ""))
            if not output_text:
                continue
            rows.append(
                {
                    "Input": clean_uploaded_text(row.get(input_col, "")) if input_col else "",
                    "Output": output_text,
                    "reference_id": choose_record_id(row, f"{safe_slug(Path(source).stem)}_{index + 1}"),
                    "source_file": source,
                    "source_row": str(index + 1),
                }
            )
        return rows

    try:
        if suffix == ".csv":
            return from_frame(pd.read_csv(BytesIO(payload.data), dtype=str, keep_default_na=False), payload.name), warnings
        if suffix in {".xlsx", ".xls"}:
            rows: list[dict[str, str]] = []
            for sheet_name, frame in pd.read_excel(BytesIO(payload.data), sheet_name=None, dtype=str).items():
                rows.extend(from_frame(frame, f"{payload.name}#{sheet_name}"))
            return rows, warnings
        if suffix == ".docx":
            text = clean_uploaded_text(read_docx_text(payload.data))
            return [{"Input": "", "Output": text, "reference_id": f"{safe_slug(Path(payload.name).stem)}_1", "source_file": payload.name, "source_row": "1"}], warnings
        if suffix in SUPPORTED_TEXT_EXTENSIONS or suffix == ".json" or not suffix:
            text = clean_uploaded_text(decode_text(payload.data))
            if suffix == ".json":
                try:
                    text = json.dumps(json.loads(text), ensure_ascii=False, indent=2)
                except json.JSONDecodeError:
                    pass
            return [{"Input": "", "Output": text, "reference_id": f"{safe_slug(Path(payload.name).stem)}_1", "source_file": payload.name, "source_row": "1"}], warnings
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"{payload.name}: could not parse style sample ({exc})")
        return [], warnings

    warnings.append(f"{payload.name}: style text extraction is not supported for {suffix or 'this file type'}")
    return [], warnings


def save_payloads(payloads: Sequence[UploadedPayload], directory: Path) -> list[dict[str, Any]]:
    directory.mkdir(parents=True, exist_ok=True)
    saved: list[dict[str, Any]] = []
    for payload in payloads:
        safe_name = safe_slug(Path(payload.name).stem, default="upload")
        suffix = Path(payload.name).suffix.lower()
        destination = directory / f"{safe_name}_{short_hash(payload.data, 8)}{suffix}"
        destination.write_bytes(payload.data)
        saved.append(
            {
                "original_name": payload.name,
                "saved_path": str(destination),
                "bytes": len(payload.data),
                "sha256": hashlib.sha256(payload.data).hexdigest(),
            }
        )
    return saved


def prepare_run_workspace(
    *,
    label: str,
    input_payloads: Sequence[UploadedPayload],
    style_payloads: Sequence[UploadedPayload],
    professor_id: str,
    output_type: str,
    max_style_samples: int = 5,
) -> PreparedRun:
    paths = create_run_paths(label)
    warnings: list[str] = []
    saved_files = []
    saved_files.extend(save_payloads(input_payloads, paths.input_uploads_dir))
    saved_files.extend(save_payloads(style_payloads, paths.style_uploads_dir))

    input_rows: list[dict[str, str]] = []
    for payload in input_payloads:
        rows, parse_warnings = payload_to_input_rows(payload, professor_id, output_type)
        input_rows.extend(rows)
        warnings.extend(parse_warnings)

    if input_rows:
        input_frame = pd.DataFrame(input_rows)
    else:
        input_frame = pd.DataFrame(columns=[*REQUIRED_INPUT_COLUMNS, "Output_Type", "Source_File"])
        warnings.append("No usable input rows were extracted from uploaded input documents.")
    input_frame.to_csv(paths.input_csv, index=False, encoding="utf-8-sig")

    style_rows: list[dict[str, str]] = []
    for payload in style_payloads[:max_style_samples]:
        rows, parse_warnings = extract_style_rows(payload)
        style_rows.extend(rows)
        warnings.extend(parse_warnings)
    if len(style_payloads) > max_style_samples:
        warnings.append(f"Only the first {max_style_samples} style sample uploads were used.")

    style_frame = pd.DataFrame(style_rows[:max_style_samples], columns=["Input", "Output", "reference_id", "source_file", "source_row"])
    if not style_frame.empty:
        reference_frame = pd.DataFrame(
            {
                "Professor_ID": professor_id,
                "수술ID": [
                    str(value).strip() or f"REF_{index + 1}"
                    for index, value in enumerate(style_frame["reference_id"].tolist())
                ],
                "Input": style_frame["Input"].fillna("").astype(str),
                "Output": style_frame["Output"].fillna("").astype(str),
            }
        )
        reference_frame.to_csv(paths.reference_csv, index=False, encoding="utf-8-sig")

        style_csv = paths.prof_samples_dir / f"{safe_slug(professor_id)}{STYLE_SAMPLE_SUFFIX}"
        style_frame[["Input", "Output", "source_file", "source_row"]].to_csv(
            style_csv,
            index=False,
            encoding="utf-8-sig",
        )
    elif style_payloads:
        warnings.append("No usable reference output text was extracted from style sample uploads.")

    prepared = PreparedRun(
        paths=paths,
        input_rows=int(len(input_frame)),
        style_sample_rows=int(len(style_frame)),
        warnings=warnings,
        saved_files=saved_files,
    )
    paths.manifest_path.write_text(
        json.dumps(prepared.to_manifest(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return prepared


def run_subprocess(command: Sequence[str], cwd: Path = PROJECT_ROOT, timeout: float | None = None) -> CommandResult:
    start = time.monotonic()
    completed = subprocess.run(
        list(command),
        cwd=str(cwd),
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )
    return CommandResult(
        command=list(command),
        cwd=str(cwd),
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
        duration_seconds=round(time.monotonic() - start, 3),
    )


def python_command(script_name: str, *args: str) -> list[str]:
    return [sys.executable, str(PROJECT_ROOT / script_name), *map(str, args)]


def run_stage1(paths: RunPaths, *, max_patients: int = 0) -> CommandResult:
    command = python_command(
        "pipeline/stage2_temporal_document_sort.py",
        "--input-csv",
        str(paths.input_csv),
        "--output-csv",
        str(paths.stage1_csv),
        "--output-json",
        str(paths.stage1_json),
        "--max-patients",
        str(max_patients),
    )
    return run_subprocess(command)


def run_stage2(
    paths: RunPaths,
    *,
    extractor_model: str,
    verifier_model: str,
    max_patients: int,
    max_iterations: int,
    coverage_threshold: float,
    evidence_threshold: float,
    ollama_host: str | None = None,
) -> CommandResult:
    command = python_command(
        "pipeline/stage3_core_fact_extraction_verification.py",
        "--input-csv",
        str(paths.stage1_csv),
        "--output-csv",
        str(paths.stage2_csv),
        "--extractor-model",
        extractor_model,
        "--verifier-model",
        verifier_model,
        "--max-patients",
        str(max_patients),
        "--max-iterations",
        str(max_iterations),
        "--coverage-threshold",
        str(coverage_threshold),
        "--evidence-threshold",
        str(evidence_threshold),
        "--save-every",
        "1",
        "--skip-readable-report",
    )
    if ollama_host:
        command.extend(["--ollama-host", ollama_host])
    return run_subprocess(command)


def run_stage34(
    paths: RunPaths,
    *,
    facts_csv: Path,
    model: str,
    style_model: str | None,
    generator_model: str | None,
    dry_run: bool,
    sample_count: int,
    max_rows: int | None,
    strict_validation: bool,
    skip_unmatched: bool,
    save_prompts: bool,
    ollama_host: str | None = None,
) -> CommandResult:
    command = python_command(
        "pipeline/stage4_5_fewshot_professor_style_agents.py",
        "--facts_csv",
        str(facts_csv),
        "--reference_csv",
        str(paths.reference_csv),
        "--output_csv",
        str(paths.stage4_csv),
        "--audit_jsonl",
        str(paths.stage4_audit_jsonl),
        "--style_cache_jsonl",
        str(paths.style_cache_jsonl),
        "--model",
        model,
        "--sample_count",
        str(sample_count),
        "--no_progress",
    )
    if style_model:
        command.extend(["--style_model", style_model])
    if generator_model:
        command.extend(["--generator_model", generator_model])
    if dry_run:
        command.append("--dry_run")
    if max_rows is not None:
        command.extend(["--max_rows", str(max_rows)])
    if strict_validation:
        command.append("--strict_validation")
    if skip_unmatched:
        command.append("--skip_unmatched")
    if save_prompts:
        command.append("--save_prompts")
    if ollama_host:
        command.extend(["--ollama_host", ollama_host])
    return run_subprocess(command)


def append_command_log(paths: RunPaths, stage_name: str, result: CommandResult) -> Path:
    destination = paths.logs_dir / f"{safe_slug(stage_name)}.json"
    destination.write_text(json.dumps(result.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    return destination


def existing_artifacts(paths: RunPaths) -> list[Path]:
    candidates = [
        paths.input_csv,
        paths.stage1_csv,
        paths.stage1_json,
        paths.stage2_csv,
        paths.reference_csv,
        paths.style_cache_jsonl,
        paths.stage4_csv,
        paths.stage4_audit_jsonl,
        paths.manifest_path,
    ]
    return [path for path in candidates if path.exists()]


def csv_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False}
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    return {
        "exists": True,
        "rows": int(len(frame)),
        "columns": list(map(str, frame.columns)),
    }


def read_table_preview(path: Path, max_rows: int = 20) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path, dtype=str, keep_default_na=False).head(max_rows)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path, dtype=str).fillna("").head(max_rows)
    return pd.DataFrame()
