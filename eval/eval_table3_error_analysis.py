#!/usr/bin/env python3
"""
Table 3 — Error analysis: LLM-as-a-Judge error taxonomy over the 5 pipelines.

    Method                                 Error columns
    1. Raw-to-Note                         Unsupported / Missing / Patient-mixing
    2. Raw-to-Fact-to-Note
    3. Chunk-to-Fact-to-Note
    4. Iterative Multi-Agent Fact-to-Note
    5. Ours (k5 deterministic)

By DEFAULT this script runs in DERIVED mode: it aggregates Table 1's per-note
results (outputs/main_exp/eval/table1_main_per_note.jsonl) into the error table
with ZERO additional API calls. Table 1's faithfulness pass already enumerates
every unsupported claim with its error_type (fabricated / contradicted /
patient_mixing) and its completeness pass already enumerates absent critical
facts, so the three error columns are exact aggregations of those verdicts:
  Unsupported    = faithfulness unsupported claims
  Patient-mixing = faithfulness claims with error_type == "patient_mixing"
  Missing        = completeness absent critical facts
Run eval_table1_main.py first. Pass --standalone for an independent judge pass
(the taxonomy below), e.g. as a cross-check.

Standalone taxonomy (one combined judge call per note; the judge sees the raw
SOURCE_RECORD, the professor's real note (GT), and the generated note):

  * Unsupported     - a clinical claim in the generated note that is not
                      grounded in SOURCE_RECORD. Subtypes:
                        fabricated      (no basis anywhere in the source)
                        contradicted    (conflicts with the source: wrong date,
                                         laterality, value, name, status)
  * Patient-mixing  - a special, most-severe subtype of unsupported content:
                      clinical material that plausibly belongs to a DIFFERENT
                      patient or encounter (foreign organ system, operation, or
                      demographic detail).
  * Missing         - a critical clinical fact present in the GT note (main
                      diagnosis, main operation+date, key pathology/treatment,
                      follow-up) that is absent from the generated note.

Reported per method (lower is better for all):
  unsupported_per_note, notes_with_unsupported(%),
  missing_per_note,     notes_with_missing(%),
  patient_mixing_per_note, notes_with_mixing(%)

Fairness controls (identical to Tables 1 and 2):
  * The SAME seeded per-professor record sample is used for every method.
  * Judge temperature 0 + fixed seed; judge IO cached in a JSONL shared with
    the other eval scripts, so reruns and overlapping calls are free.
  * Empty generated notes count every GT critical fact as missing, without an
    API call for the unsupported side (an empty note asserts nothing).

Usage:
  python eval_table3_error_analysis.py                       # derived; no API, runs after Table 1
  python eval_table3_error_analysis.py --standalone          # independent judge pass (costs API)
  export OPENAI_API_KEY=sk-... ; python eval_table3_error_analysis.py --standalone --sample_per_professor 10
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm.auto import tqdm


TABLE_NAME = "table3_error_analysis"

DEFAULT_CSV_DIR = Path("/root/DY/Agents/outputs/main_exp/csvs")
DEFAULT_OUT_DIR = Path("/root/DY/Agents/outputs/main_exp/eval")

METHODS: dict[str, tuple[str, str]] = {
    "raw_to_note": ("Raw-to-Note", "exp1_raw_to_note_w_GT.csv"),
    "raw_fact_to_note": ("Raw-to-Fact-to-Note", "exp2_raw_fact_to_note_w_GT.csv"),
    "chunk_fact_to_note": ("Chunk-to-Fact-to-Note", "exp3_chunk_fact_to_note_w_GT.csv"),
    "iterative_fact_to_note": ("Iterative Multi-Agent Fact-to-Note", "exp4_iterative_fact_to_note_w_GT.csv"),
    "ours": ("Ours", "exp5_ours_k3_deterministic_w_GT.csv"),  # main config: 3-shot deterministic
}

RECORD_COLUMN = "record_id"
PROFESSOR_COLUMN = "professor"
NOTE_COLUMN = "generated_note"
SOURCE_COLUMN = "Input"
GT_COLUMN = "Output"

ERROR_SYSTEM = """
You are a meticulous clinical documentation auditor performing an ERROR
ANALYSIS of a machine-generated outpatient note.

Evidence rules:
- SOURCE_RECORD is the only ground truth for whether a claim is supported.
- GROUND_TRUTH_NOTE (the professor's real note for this encounter) defines
  which facts are critical and should have been included.
- Formatting tokens such as <|section_start|> ... <|section_end|> and pure
  section headers are structure, not claims.
- Do not use outside medical knowledge to justify patient-specific claims.
Return valid JSON only.
""".strip()

ERROR_USER_TEMPLATE = """
<SOURCE_RECORD>
{source}
</SOURCE_RECORD>

<GROUND_TRUTH_NOTE>
{gt}
</GROUND_TRUTH_NOTE>

<GENERATED_NOTE>
{note}
</GENERATED_NOTE>

Audit GENERATED_NOTE and report three error families:

1. "unsupported_claims": every atomic clinical claim in GENERATED_NOTE that is
   NOT grounded in SOURCE_RECORD (abbreviation/translation equivalence counts
   as grounded). For each give:
     "claim": the offending text (short),
     "error_type": "fabricated"     (no basis anywhere in the source)
                 | "contradicted"   (conflicts with the source: wrong date,
                                     laterality, value, name, status)
                 | "patient_mixing" (clinical content that plausibly belongs to
                                     a DIFFERENT patient or encounter, e.g., an
                                     organ system, operation, or demographic
                                     detail foreign to this record),
     "severity": "minor" | "major"  (major = could change clinical handover).

2. "missing_critical_facts": every critical clinical fact in GROUND_TRUTH_NOTE
   (main diagnosis or R/O diagnosis, main operation/procedure with date, key
   pathology/treatment, follow-up/status) that is ABSENT from GENERATED_NOTE.
   Facts that are present but with a wrong qualifier belong to
   unsupported_claims (contradicted), not here. For each give:
     "fact": short description,
     "category": "diagnosis|operation|date|pathology|treatment|follow_up|other".

3. "patient_mixing_suspected": true if any content suggests another patient's
   record was mixed in, else false.

Be precise: report an empty list when a family has no errors.

Return JSON exactly:
{{
  "unsupported_claims": [{{"claim": "...", "error_type": "...", "severity": "..."}}],
  "missing_critical_facts": [{{"fact": "...", "category": "..."}}],
  "patient_mixing_suspected": false,
  "rationale": "one sentence"
}}
""".strip()

GT_FACTS_SYSTEM = """
You are a clinical documentation auditor. Extract the critical clinical facts
from the professor's outpatient note. Ignore formatting tokens, headers, and
stylistic fragments. Return valid JSON only.
""".strip()

GT_FACTS_USER_TEMPLATE = """
<GROUND_TRUTH_NOTE>
{gt}
</GROUND_TRUTH_NOTE>

List the critical clinical facts (main diagnosis or R/O diagnosis, main
operation/procedure with date, key pathology/treatment, follow-up/status).

Return JSON exactly:
{{"critical_facts": [{{"fact": "...", "category": "diagnosis|operation|date|pathology|treatment|follow_up|other"}}]}}
""".strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Table 3 (error analysis) LLM-as-a-Judge evaluation.")
    parser.add_argument("--csv_dir", type=Path, default=DEFAULT_CSV_DIR)
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--judge_model", default="gpt-4o")
    parser.add_argument(
        "--backend",
        choices=("openai", "ollama"),
        default="openai",
        help="Judge backend. 'ollama' runs a local model (free, no API key, slower).",
    )
    parser.add_argument("--ollama_host", default=os.environ.get("OLLAMA_HOST"), help="e.g. 127.0.0.1:11434")
    parser.add_argument("--ollama_num_ctx", type=int, default=32768)
    parser.add_argument(
        "--sample_per_professor",
        type=int,
        default=10,
        help="Records per professor; the SAME sample is used for every method. 0 = all records.",
    )
    parser.add_argument("--sample_seed", type=int, default=1225)
    parser.add_argument(
        "--methods",
        nargs="*",
        choices=list(METHODS),
        default=list(METHODS),
        help="Subset of methods to evaluate.",
    )
    parser.add_argument(
        "--standalone",
        action="store_true",
        help="Run an independent judge pass (one combined error call per note). "
        "Default is DERIVED mode: aggregate Table 1's per-note results with zero API calls.",
    )
    parser.add_argument(
        "--source_per_note",
        type=Path,
        default=None,
        help="Table 1 per-note JSONL consumed by derived mode "
        "(default: <out_dir>/table1_main_per_note.jsonl).",
    )
    parser.add_argument("--max_source_chars", type=int, default=30000)
    parser.add_argument("--max_note_chars", type=int, default=8000)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--request_retries", type=int, default=4)
    parser.add_argument("--retry_sleep", type=float, default=3.0)
    parser.add_argument("--dry_run", action="store_true", help="Deterministic placeholder judge; no API calls.")
    parser.add_argument("--no_progress", action="store_true")
    return parser.parse_args()


def clean_scalar(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    text = str(value).strip()
    if text.lower() in {"nan", "none", "nat"}:
        return ""
    return text


def stable_json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def truncate_middle(text: str, max_chars: int) -> str:
    text = clean_scalar(text)
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    left = max_chars // 2
    right = max(0, max_chars - left - 32)
    return text[:left].rstrip() + "\n...[TRUNCATED]...\n" + text[-right:].lstrip()


def extract_json_object(text: str) -> dict[str, Any]:
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", clean_scalar(text), flags=re.I)
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass
    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end > start:
        data = json.loads(text[start : end + 1])
        if isinstance(data, dict):
            return data
    raise ValueError(f"Judge did not return a JSON object. Head: {text[:300]}")


class JudgeClient:
    """OpenAI chat judge with caching, retries, and a deterministic dry-run mode."""

    def __init__(self, args: argparse.Namespace, cache_path: Path) -> None:
        self.model = args.judge_model
        self.dry_run = args.dry_run
        self.retries = max(0, args.request_retries)
        self.retry_sleep = max(0.0, args.retry_sleep)
        self.cache_path = cache_path
        self.cache: dict[str, dict[str, Any]] = {}
        self.cache_lock = threading.Lock()
        self.supports_sampling_params = True
        self._client = None
        if cache_path.exists():
            with cache_path.open(encoding="utf-8") as f:
                for line in f:
                    try:
                        row = json.loads(line)
                        self.cache[row["key"]] = row["response"]
                    except (json.JSONDecodeError, KeyError):
                        continue
        self.backend = getattr(args, "backend", "openai")
        self.ollama_host = getattr(args, "ollama_host", None)
        self.ollama_num_ctx = getattr(args, "ollama_num_ctx", 32768)
        if not self.dry_run:
            if self.backend == "ollama":
                import ollama

                self._client = ollama.Client(host=self.ollama_host) if self.ollama_host else ollama.Client()
            else:
                if not os.environ.get("OPENAI_API_KEY"):
                    raise RuntimeError(
                        "OPENAI_API_KEY is not set. Export it, run with --backend ollama, or --dry_run."
                    )
                from openai import OpenAI

                self._client = OpenAI()

    def judge(self, task: str, system: str, user: str) -> dict[str, Any]:
        key = sha256_text(stable_json({"model": self.model, "task": task, "system": system, "user": user}))
        with self.cache_lock:
            if key in self.cache:
                return self.cache[key]
        response = self._placeholder(task, user) if self.dry_run else self._call(system, user)
        with self.cache_lock:
            self.cache[key] = response
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            with self.cache_path.open("a", encoding="utf-8") as f:
                f.write(stable_json({"key": key, "task": task, "model": self.model, "response": response}) + "\n")
        return response

    def _call(self, system: str, user: str) -> dict[str, Any]:
        if self.backend == "ollama":
            return self._call_ollama(system, user)
        return self._call_openai(system, user)

    def _call_ollama(self, system: str, user: str) -> dict[str, Any]:
        """Local Ollama judge. format=json forces structured output; <think> blocks stripped."""
        last_exc: Exception | None = None
        options = {"temperature": 0.0, "top_p": 1.0, "num_ctx": self.ollama_num_ctx, "seed": 1225}
        for attempt in range(self.retries + 1):
            try:
                kwargs: dict[str, Any] = {
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    "format": "json",
                    "options": options,
                    "stream": False,
                    "keep_alive": "30m",
                }
                try:
                    resp = self._client.chat(**kwargs, think=False)
                except TypeError:
                    resp = self._client.chat(**kwargs)
                message = resp.get("message") if isinstance(resp, dict) else getattr(resp, "message", None)
                if isinstance(message, dict):
                    content = message.get("content", "")
                else:
                    content = getattr(message, "content", "") if message is not None else ""
                content = re.sub(r"<think>.*?(?:</think>|$)", "", content or "", flags=re.I | re.S)
                return extract_json_object(content)
            except Exception as exc:  # noqa: BLE001 - local model / JSON parse errors.
                last_exc = exc
                if attempt < self.retries:
                    time.sleep(self.retry_sleep * (attempt + 1))
        raise RuntimeError(f"Ollama judge call failed after {self.retries + 1} attempt(s): {last_exc}")

    def _call_openai(self, system: str, user: str) -> dict[str, Any]:
        last_exc: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                kwargs: dict[str, Any] = {
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    "response_format": {"type": "json_object"},
                    "timeout": 180,
                }
                if self.supports_sampling_params:
                    kwargs["temperature"] = 0.0
                    kwargs["seed"] = 1225
                try:
                    resp = self._client.chat.completions.create(**kwargs)
                except Exception as exc:  # noqa: BLE001 - reasoning models reject temperature/seed.
                    message = str(exc).lower()
                    if self.supports_sampling_params and ("temperature" in message or "seed" in message):
                        self.supports_sampling_params = False
                        kwargs.pop("temperature", None)
                        kwargs.pop("seed", None)
                        resp = self._client.chat.completions.create(**kwargs)
                    else:
                        raise
                return extract_json_object(resp.choices[0].message.content or "")
            except Exception as exc:  # noqa: BLE001 - rate limits / transient API errors.
                last_exc = exc
                if attempt < self.retries:
                    time.sleep(self.retry_sleep * (attempt + 1))
        raise RuntimeError(f"Judge call failed after {self.retries + 1} attempt(s): {last_exc}")

    @staticmethod
    def _placeholder(task: str, user: str) -> dict[str, Any]:
        digest = int(sha256_text(task + user)[:8], 16)
        if task == "error_analysis":
            unsupported = []
            if digest % 4 == 0:
                unsupported.append({"claim": "placeholder unsupported", "error_type": "fabricated", "severity": "minor"})
            if digest % 11 == 0:
                unsupported.append({"claim": "placeholder mixing", "error_type": "patient_mixing", "severity": "major"})
            missing = [] if digest % 3 else [{"fact": "placeholder missing op", "category": "operation"}]
            return {
                "unsupported_claims": unsupported,
                "missing_critical_facts": missing,
                "patient_mixing_suspected": any(c["error_type"] == "patient_mixing" for c in unsupported),
                "rationale": "dry run",
            }
        if task == "gt_facts":
            return {
                "critical_facts": [
                    {"fact": "placeholder dx", "category": "diagnosis"},
                    {"fact": "placeholder op", "category": "operation"},
                ]
            }
        return {"rationale": "dry run"}


def load_method_frame(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Result CSV not found: {path}")
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    required = {RECORD_COLUMN, PROFESSOR_COLUMN, NOTE_COLUMN, SOURCE_COLUMN, GT_COLUMN}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path.name} missing columns: {sorted(missing)}")
    before = len(df)
    df = df.drop_duplicates(subset=[RECORD_COLUMN], keep="first").reset_index(drop=True)
    if len(df) != before:
        print(f"WARNING: {path.name}: dropped {before - len(df)} duplicate record_id rows (GT merge artifact).", file=sys.stderr)
    return df


def build_sample(reference_frame: pd.DataFrame, per_professor: int, seed: int) -> set[str]:
    """Seeded per-professor sample of record_ids; identical for every method and every eval script."""
    if per_professor <= 0:
        return set(reference_frame[RECORD_COLUMN])
    selected: set[str] = set()
    for professor, group in reference_frame.groupby(PROFESSOR_COLUMN):
        ids = sorted(group[RECORD_COLUMN].unique())
        if len(ids) <= per_professor:
            selected.update(ids)
        else:
            rng = random.Random(f"{seed}:{professor}")
            selected.update(rng.sample(ids, per_professor))
    return selected


def count_gt_critical_facts(client: JudgeClient, gt: str, args: argparse.Namespace) -> int:
    """Used only for empty generated notes: every GT critical fact is missing."""
    user = GT_FACTS_USER_TEMPLATE.format(gt=truncate_middle(gt, args.max_note_chars))
    data = client.judge("gt_facts", GT_FACTS_SYSTEM, user)
    facts = data.get("critical_facts") if isinstance(data.get("critical_facts"), list) else []
    return len(facts)


def judge_errors(client: JudgeClient, source: str, gt: str, note: str, args: argparse.Namespace) -> dict[str, Any]:
    user = ERROR_USER_TEMPLATE.format(
        source=truncate_middle(source, args.max_source_chars),
        gt=truncate_middle(gt, args.max_note_chars),
        note=truncate_middle(note, args.max_note_chars),
    )
    data = client.judge("error_analysis", ERROR_SYSTEM, user)
    unsupported = data.get("unsupported_claims") if isinstance(data.get("unsupported_claims"), list) else []
    missing = data.get("missing_critical_facts") if isinstance(data.get("missing_critical_facts"), list) else []
    unsupported = [c for c in unsupported if isinstance(c, dict)]
    missing = [f for f in missing if isinstance(f, dict)]
    mixing = [c for c in unsupported if clean_scalar(c.get("error_type")) == "patient_mixing"]
    return {
        "n_unsupported": len(unsupported),
        "n_missing": len(missing),
        "n_mixing": len(mixing),
        "mixing_suspected": bool(data.get("patient_mixing_suspected")) or bool(mixing),
        "unsupported_claims": unsupported,
        "missing_critical_facts": missing,
        "raw": data,
    }


def patient_mixing_count(unsupported_claims: Any) -> int:
    if not isinstance(unsupported_claims, list):
        return 0
    return sum(
        1
        for c in unsupported_claims
        if isinstance(c, dict) and clean_scalar(c.get("error_type")).lower() == "patient_mixing"
    )


def evaluate_derived(args: argparse.Namespace) -> int:
    """Build Table 3 from Table 1's per-note results — no API calls.

    Mapping (identical taxonomy, same judge verdicts as Table 1):
      unsupported  <- faithfulness unsupported claims (n_unsupported)
      patient_mixing <- faithfulness claims with error_type == "patient_mixing"
      missing      <- completeness absent critical facts (n_absent)
    """
    src = args.source_per_note or (args.out_dir / "table1_main_per_note.jsonl")
    if not src.exists():
        raise FileNotFoundError(
            f"Derived mode requires Table 1 results first: {src} not found.\n"
            "Run `python eval_table1_main.py` with the same sample settings, "
            "or run this script with --standalone for an independent judge pass."
        )
    rows: list[dict[str, Any]] = []
    with src.open(encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("error"):
                continue
            rows.append(row)
    if not rows:
        raise ValueError(f"No usable rows in {src}.")
    df = pd.DataFrame(rows)

    required = {"method", "record_id", "n_unsupported", "n_absent", "unsupported_claims"}
    missing_cols = required - set(df.columns)
    if missing_cols:
        raise ValueError(
            f"{src} is missing columns {sorted(missing_cols)}. It was produced by an older "
            "eval_table1_main.py; re-run Table 1 so it records unsupported_claims/absent_facts, "
            "or use --standalone."
        )

    df = df[df["method"].isin(args.methods)].copy()
    if df.empty:
        raise ValueError(f"No rows for requested methods {args.methods} in {src}.")
    df["n_unsupported"] = pd.to_numeric(df["n_unsupported"], errors="coerce").fillna(0).astype(int)
    df["n_missing"] = pd.to_numeric(df["n_absent"], errors="coerce").fillna(0).astype(int)
    df["n_mixing"] = df["unsupported_claims"].map(patient_mixing_count)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    n_records = df["record_id"].nunique()
    print(
        f"Derived mode: aggregating {len(df)} per-note rows ({n_records} records) "
        f"from {src.name}; no API calls.",
        file=sys.stderr,
    )

    table_rows = []
    for method in args.methods:
        sub = df[df["method"] == method]
        if sub.empty:
            continue
        table_rows.append(
            {
                "method": METHODS[method][0],
                "n": len(sub),
                "unsupported_per_note": sub["n_unsupported"].mean(),
                "notes_with_unsupported_pct": 100 * (sub["n_unsupported"] > 0).mean(),
                "missing_per_note": sub["n_missing"].mean(),
                "notes_with_missing_pct": 100 * (sub["n_missing"] > 0).mean(),
                "patient_mixing_per_note": sub["n_mixing"].mean(),
                "notes_with_mixing_pct": 100 * (sub["n_mixing"] > 0).mean(),
            }
        )
    table = pd.DataFrame(table_rows)
    table_path = args.out_dir / f"{TABLE_NAME}.csv"
    table.to_csv(table_path, index=False, encoding="utf-8-sig", float_format="%.2f")

    examples_path = args.out_dir / f"{TABLE_NAME}_examples.jsonl"
    with examples_path.open("w", encoding="utf-8") as f:
        for method in args.methods:
            sub = df[df["method"] == method].copy()
            if sub.empty:
                continue
            sub["_sev"] = sub["n_mixing"] * 100 + sub["n_unsupported"] * 10 + sub["n_missing"]
            for _, row in sub.sort_values("_sev", ascending=False).head(5).iterrows():
                f.write(
                    stable_json(
                        {
                            "method": method,
                            "record_id": row["record_id"],
                            "n_unsupported": int(row["n_unsupported"]),
                            "n_missing": int(row["n_missing"]),
                            "n_mixing": int(row["n_mixing"]),
                            "unsupported_claims": row.get("unsupported_claims", []),
                            "missing_critical_facts": row.get("absent_facts", []),
                        }
                    )
                    + "\n"
                )

    print("\n=== Table 3: Error analysis [derived from Table 1] (all columns: lower is better) ===")
    with pd.option_context("display.max_columns", None, "display.width", 200, "display.float_format", "{:.2f}".format):
        print(table.to_string(index=False))
    print(f"\nSaved table   : {table_path}")
    print(f"Worst examples: {examples_path}")
    print(f"Source        : {src}")
    return 0


def evaluate_standalone(args: argparse.Namespace) -> int:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    # Shared with Tables 1 and 2: identical judge calls are paid for once.
    client = JudgeClient(args, args.out_dir / "judge_cache_shared.jsonl")

    frames = {method: load_method_frame(args.csv_dir / METHODS[method][1]) for method in args.methods}
    anchor = frames[args.methods[0]]
    sample = build_sample(anchor, args.sample_per_professor, args.sample_seed)
    print(f"Evaluating {len(sample)} records x {len(args.methods)} methods (judge={args.judge_model}, dry_run={args.dry_run})", file=sys.stderr)

    indexed: dict[str, dict[str, dict[str, str]]] = {}
    for method, frame in frames.items():
        indexed[method] = {
            clean_scalar(r[RECORD_COLUMN]): {
                "professor": clean_scalar(r[PROFESSOR_COLUMN]),
                "note": clean_scalar(r[NOTE_COLUMN]),
                "source": clean_scalar(r[SOURCE_COLUMN]),
                "gt": clean_scalar(r[GT_COLUMN]),
            }
            for _, r in frame.iterrows()
        }

    def evaluate_record(method: str, record_id: str) -> dict[str, Any]:
        row = indexed[method][record_id]
        result: dict[str, Any] = {"method": method, "record_id": record_id, "professor": row["professor"]}
        if not row["note"]:
            n_missing = count_gt_critical_facts(client, row["gt"], args)
            result.update(
                n_unsupported=0, n_missing=n_missing, n_mixing=0, mixing_suspected=False,
                unsupported_claims=[], missing_critical_facts=[{"fact": "ALL (empty note)", "category": "other"}] * 0,
                empty_note=True,
            )
            return result
        errors = judge_errors(client, row["source"], row["gt"], row["note"], args)
        result.update(
            n_unsupported=errors["n_unsupported"],
            n_missing=errors["n_missing"],
            n_mixing=errors["n_mixing"],
            mixing_suspected=errors["mixing_suspected"],
            unsupported_claims=errors["unsupported_claims"],
            missing_critical_facts=errors["missing_critical_facts"],
            empty_note=False,
        )
        return result

    jobs = [(m, rid) for m in args.methods for rid in sorted(sample) if rid in indexed[m]]
    rows: list[dict[str, Any]] = []
    detail_path = args.out_dir / f"{TABLE_NAME}_per_note.jsonl"
    with detail_path.open("w", encoding="utf-8") as detail, ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(evaluate_record, m, rid): (m, rid) for m, rid in jobs}
        progress = tqdm(total=len(futures), desc="Auditing", unit="note", disable=args.no_progress, dynamic_ncols=True)
        for future in as_completed(futures):
            method, record_id = futures[future]
            try:
                row = future.result()
            except Exception as exc:  # noqa: BLE001 - keep judging the rest; report at the end.
                row = {"method": method, "record_id": record_id, "error": str(exc)}
            rows.append(row)
            detail.write(stable_json(row) + "\n")
            detail.flush()
            progress.update(1)
        progress.close()

    df = pd.DataFrame(rows)
    if "error" in df.columns and df["error"].notna().any():
        n_err = int(df["error"].notna().sum())
        print(f"WARNING: {n_err} judge failures (excluded from aggregation); see {detail_path}", file=sys.stderr)
        df = df[df["error"].isna()]

    table_rows = []
    for method in args.methods:
        sub = df[df["method"] == method]
        if sub.empty:
            continue
        table_rows.append(
            {
                "method": METHODS[method][0],
                "n": len(sub),
                "unsupported_per_note": sub["n_unsupported"].mean(),
                "notes_with_unsupported_pct": 100 * (sub["n_unsupported"] > 0).mean(),
                "missing_per_note": sub["n_missing"].mean(),
                "notes_with_missing_pct": 100 * (sub["n_missing"] > 0).mean(),
                "patient_mixing_per_note": sub["n_mixing"].mean(),
                "notes_with_mixing_pct": 100 * (sub["mixing_suspected"]).mean(),
            }
        )
    table = pd.DataFrame(table_rows)
    table_path = args.out_dir / f"{TABLE_NAME}.csv"
    table.to_csv(table_path, index=False, encoding="utf-8-sig", float_format="%.2f")

    # Qualitative appendix: the worst offending examples per method.
    examples_path = args.out_dir / f"{TABLE_NAME}_examples.jsonl"
    with examples_path.open("w", encoding="utf-8") as f:
        for method in args.methods:
            sub = df[df["method"] == method].copy()
            if sub.empty:
                continue
            sub["_sev"] = sub["n_mixing"] * 100 + sub["n_unsupported"] * 10 + sub["n_missing"]
            for _, row in sub.sort_values("_sev", ascending=False).head(5).iterrows():
                f.write(
                    stable_json(
                        {
                            "method": method,
                            "record_id": row["record_id"],
                            "n_unsupported": int(row["n_unsupported"]),
                            "n_missing": int(row["n_missing"]),
                            "n_mixing": int(row["n_mixing"]),
                            "unsupported_claims": row.get("unsupported_claims", []),
                            "missing_critical_facts": row.get("missing_critical_facts", []),
                        }
                    )
                    + "\n"
                )

    print("\n=== Table 3: Error analysis (all columns: lower is better) ===")
    with pd.option_context("display.max_columns", None, "display.width", 200, "display.float_format", "{:.2f}".format):
        print(table.to_string(index=False))
    print(f"\nSaved table   : {table_path}")
    print(f"Saved detail  : {detail_path}")
    print(f"Worst examples: {examples_path}")
    print(f"Judge cache   : {client.cache_path}")
    return 0


def main() -> None:
    args = parse_args()
    if args.standalone:
        raise SystemExit(evaluate_standalone(args))
    raise SystemExit(evaluate_derived(args))


if __name__ == "__main__":
    main()


"""
# Derived (default, free) — run after Table 1:
python eval_table3_error_analysis.py

# Standalone cross-check (independent judge, costs API):
export OPENAI_API_KEY=sk-...
python eval_table3_error_analysis.py --standalone --judge_model gpt-4o --sample_per_professor 10
python eval_table3_error_analysis.py --standalone --dry_run     # plumbing test, no API
"""
