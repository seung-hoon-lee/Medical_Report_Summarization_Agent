#!/usr/bin/env python3
"""
Table 2 — Few-shot style ablation: LLM-as-a-Judge evaluation.

    Few-shot setting        Source CSV
    No few-shot             exp4 (iterative facts, style-free single agent)
    3-shot Random           exp5 k3 random seed1225
    3-shot Deterministic    exp5 k3 deterministic
    5-shot Random           exp5 k5 random seed1225
    5-shot Deterministic    exp5 k5 deterministic (Ours main)

    All five settings consume IDENTICAL iterative-verified facts (e3), so this
    table isolates the contribution of the few-shot style stage and its (k,
    selection) configuration. Style Win is judged pairwise against the
    No few-shot baseline (its own row shows '-').

Methodology (grounded in established LLM-as-judge literature):
  * Faithfulness  - FActScore/RAGAS-style atomic-claim decomposition: the judge
    extracts atomic clinical claims from the generated note and verifies each
    against the raw SOURCE_RECORD (supported / borderline / unsupported, with
    error subtype for unsupported). It also assigns a G-Eval-style 5-point
    Likert with an explicit BORDERLINE band:
        5-4 = pass, 3 = borderline, 2-1 = fail.
      -> Faithful Pass      = % notes with Likert >= 4 (strict)
      -> Faithful Borderline= % notes with Likert == 3 (reported separately)
      -> Hallucination-free = % notes with zero unsupported claims
      -> Unsupported/Note   = mean count of unsupported claims per note
  * Completeness  - GT-anchored: the judge extracts the critical clinical facts
    from the professor's real note (GT) and checks each is present / partial /
    absent in the generated note.
      -> Critical Complete         = % notes with ALL critical facts present (strict)
      -> Critical Complete lenient = % notes with no absent fact (partial allowed)
  * Style Win     - MT-Bench-style pairwise comparison with POSITION-SWAP
    debiasing: for each record, the method's note battles a fixed opponent
    (default: ours) given 3 reference notes by the same professor (other
    records only -> no GT leakage of the evaluated record). Two judge calls
    with A/B order swapped; win=1, tie=0.5 per call, averaged.

Fairness controls:
  * The SAME record sample is used for every method (seeded per professor).
  * Judge temperature 0 + fixed seed; all judge IO cached to a JSONL so reruns
    are free and resumable.
  * Empty generated notes are auto-scored as failures without an API call.

Usage:
  export OPENAI_API_KEY=sk-...
  python eval_table1_main.py --sample_per_professor 10        # 210 records/method
  python eval_table1_main.py --sample_per_professor 0         # all 1050 (expensive)
  python eval_table1_main.py --dry_run                        # no API, plumbing test
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


TABLE_NAME = "table2_ablation"

DEFAULT_CSV_DIR = Path("/root/DY/Agents/outputs/main_exp/csvs")
DEFAULT_OUT_DIR = Path("/root/DY/Agents/outputs/main_exp/eval")

# Evaluation row order is preserved in the final table.
METHODS: dict[str, tuple[str, str]] = {
    "no_fewshot": ("No few-shot", "exp4_iterative_fact_to_note_w_GT.csv"),
    "k3_random": ("3-shot Random", "exp5_ours_k3_random_seed1225_w_GT.csv"),
    "k3_deterministic": ("3-shot Deterministic", "exp5_ours_k3_deterministic_w_GT.csv"),
    "k5_random": ("5-shot Random", "exp5_ours_k5_random_seed1225_w_GT.csv"),
    "k5_deterministic": ("5-shot Deterministic", "exp5_ours_k5_deterministic_w_GT.csv"),
}

RECORD_COLUMN = "record_id"
PROFESSOR_COLUMN = "professor"
NOTE_COLUMN = "generated_note"
SOURCE_COLUMN = "Input"
GT_COLUMN = "Output"

FAITHFULNESS_SYSTEM = """
You are a meticulous clinical documentation auditor evaluating machine-generated
outpatient notes for factual faithfulness.

Ground truth evidence is ONLY the SOURCE_RECORD given by the user. Reference
nothing else; do not use outside medical knowledge to justify patient-specific
claims.

NOT claims (never audit these — they are documentation scaffolding, not
patient-specific clinical assertions; ignore them entirely):
- Formatting tokens such as <|section_start|> ... <|section_end|> and section
  headers (e.g. "[Main Diagnosis]", "Description", "Assessment:", "Plan:", "소견").
- Visit-type / encounter-framing labels, e.g. "postop 1st visit",
  "post-operative first visit", "first postoperative follow-up visit",
  "this is a postoperative visit", "OPD follow-up", together with any visit date
  attached to such a label.
- Generic disposition boilerplate that asserts no specific finding, e.g.
  "OPD f/u", "regular follow-up", "follow up".
These are stylistic conventions of the note and cannot be hallucinations.

Return valid JSON only.
""".strip()

FAITHFULNESS_USER_TEMPLATE = """
<SOURCE_RECORD>
{source}
</SOURCE_RECORD>

<GENERATED_NOTE>
{note}
</GENERATED_NOTE>

Step 1 - Decompose GENERATED_NOTE into atomic PATIENT-SPECIFIC clinical claims
(diagnosis, procedure, date of a procedure/test, laterality, measurement,
pathology, medication, specific finding/status, specific follow-up instruction).
EXCLUDE documentation scaffolding that is not a patient-specific clinical
assertion: section headers; visit-type/encounter-framing labels ("postop 1st
visit", "first postoperative follow-up", "OPD follow-up", and any visit date
attached to them); and generic disposition boilerplate ("OPD f/u", "follow up").
Such scaffolding is style, not a verifiable clinical claim — do not list it.

Step 2 - For each claim assign:
  "verdict": "supported"   (explicitly stated in SOURCE_RECORD, allowing
                            obvious abbreviation/translation equivalence),
             "borderline"  (plausible paraphrase but not explicitly verifiable),
             "unsupported" (not in the source, or conflicts with it)
  "error_type" (only when unsupported):
             "fabricated"      - no basis anywhere in the source
             "contradicted"    - conflicts with the source (wrong date,
                                 laterality, value, name, status)
             "patient_mixing"  - clinical content that plausibly belongs to a
                                 DIFFERENT patient/encounter (e.g., an organ
                                 system, operation, or demographic detail
                                 foreign to this record)
  "evidence": short supporting quote from SOURCE_RECORD, or null.

Step 3 - Overall 5-point faithfulness Likert:
  5 = every claim explicitly supported
  4 = all claims supported; only trivial paraphrase ambiguity
  3 = BORDERLINE: no clear hallucination but >=1 claim only weakly supported
  2 = >=1 clearly unsupported or contradicted claim
  1 = multiple hallucinations or any patient-mixing content

Return JSON exactly:
{{
  "claims": [{{"claim": "...", "verdict": "...", "error_type": null, "evidence": null}}],
  "faithfulness_likert": 1,
  "rationale": "one sentence"
}}
""".strip()

COMPLETENESS_SYSTEM = """
You are a clinical documentation auditor evaluating whether a machine-generated
outpatient note covers the clinically critical content of the professor's real
note (the ground-truth note for the same encounter).

The ground-truth note defines what content selection is correct: facts the
professor chose to write are the critical facts. Ignore styling, headers,
formatting tokens, and greetings. Return valid JSON only.
""".strip()

COMPLETENESS_USER_TEMPLATE = """
<GROUND_TRUTH_NOTE>
{gt}
</GROUND_TRUTH_NOTE>

<GENERATED_NOTE>
{note}
</GENERATED_NOTE>

Step 1 - Extract the critical clinical facts from GROUND_TRUTH_NOTE: main
diagnosis or R/O diagnosis, main operation/procedure with its date, essential
pathology/treatment items, and follow-up/status statements. Exclude structural
lines and stylistic fragments.

Step 2 - For each critical fact decide its status in GENERATED_NOTE:
  "present" (stated, allowing abbreviation/paraphrase equivalence),
  "partial" (mentioned but with missing/incorrect qualifier such as date or laterality),
  "absent".

Step 3 - Overall 5-point completeness Likert (5 = everything covered,
3 = borderline with minor omissions, 1 = core content missing).

Return JSON exactly:
{{
  "critical_facts": [{{"fact": "...", "category": "diagnosis|operation|date|pathology|treatment|follow_up|other", "status": "present"}}],
  "completeness_likert": 1,
  "rationale": "one sentence"
}}
""".strip()

STYLE_SYSTEM = """
You are a clinical documentation STYLE judge. You compare two candidate
outpatient notes against reference notes written by the same professor and
decide which candidate better matches the professor's writing style.

Style means: format and layout, typical length and compactness, abbreviation
and shorthand habits, line ordering, section headers, and content-selection
behavior (what kinds of facts this professor typically includes or omits).

Do NOT judge factual correctness; the candidates describe a different patient
than the references. Do not reward a candidate merely for containing more
information. Return valid JSON only.
""".strip()

STYLE_USER_TEMPLATE = """
<PROFESSOR_REFERENCE_NOTES>
{references}
</PROFESSOR_REFERENCE_NOTES>

<CANDIDATE_A>
{note_a}
</CANDIDATE_A>

<CANDIDATE_B>
{note_b}
</CANDIDATE_B>

Which candidate matches the professor's style more closely?
Answer "tie" only when they are genuinely indistinguishable in style.

Return JSON exactly:
{{"winner": "A", "rationale": "one sentence"}}
(winner must be "A", "B", or "tie")
""".strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Table 2 (few-shot ablation) LLM-as-a-Judge evaluation.")
    parser.add_argument("--csv_dir", type=Path, default=DEFAULT_CSV_DIR)
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--judge_model", default="gpt-4o", help="OpenAI judge model (e.g., gpt-4o, gpt-4o-mini, gpt-5).")
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
        "--style_opponent",
        default="no_fewshot",
        choices=list(METHODS),
        help="Fixed opponent for pairwise Style Win; the opponent's own row shows '-'.",
    )
    parser.add_argument("--style_reference_count", type=int, default=3)
    parser.add_argument("--pass_threshold", type=int, default=4, help="Likert >= threshold counts as pass.")
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
        if task == "faithfulness":
            likert = 3 + (digest % 3)  # 3..5 deterministic spread
            claims = [
                {"claim": "placeholder claim", "verdict": "supported", "error_type": None, "evidence": "dry run"}
            ]
            if digest % 5 == 0:
                claims.append(
                    {"claim": "placeholder unsupported", "verdict": "unsupported", "error_type": "fabricated", "evidence": None}
                )
                likert = 2
            return {"claims": claims, "faithfulness_likert": likert, "rationale": "dry run"}
        if task == "completeness":
            status = "present" if digest % 4 else "absent"
            return {
                "critical_facts": [
                    {"fact": "placeholder dx", "category": "diagnosis", "status": "present"},
                    {"fact": "placeholder op", "category": "operation", "status": status},
                ],
                "completeness_likert": 5 if status == "present" else 3,
                "rationale": "dry run",
            }
        if task == "style":
            return {"winner": ["A", "B", "tie"][digest % 3], "rationale": "dry run"}
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


def build_style_references(frame: pd.DataFrame, count: int) -> dict[str, list[tuple[str, str]]]:
    """professor -> [(record_id, gt_note)] length-stratified (short/median/long) reference pool."""
    references: dict[str, list[tuple[str, str]]] = {}
    for professor, group in frame.groupby(PROFESSOR_COLUMN):
        rows = [(rid, clean_scalar(gt)) for rid, gt in zip(group[RECORD_COLUMN], group[GT_COLUMN]) if clean_scalar(gt)]
        rows.sort(key=lambda item: (len(item[1]), item[0]))
        if not rows:
            references[professor] = []
            continue
        if len(rows) <= count:
            references[professor] = rows
        else:
            picks = sorted({round(q * (len(rows) - 1)) for q in [i / max(1, count - 1) for i in range(count)]})
            references[professor] = [rows[i] for i in picks]
    return references


def style_reference_block(
    references: dict[str, list[tuple[str, str]]],
    professor: str,
    exclude_record: str,
    count: int,
    max_chars: int,
) -> str:
    pool = [item for item in references.get(professor, []) if item[0] != exclude_record][:count]
    blocks = [f"[Reference {i + 1}]\n{truncate_middle(note, max_chars)}" for i, (_, note) in enumerate(pool)]
    return "\n\n".join(blocks) if blocks else "(no reference available)"


def judge_faithfulness(client: JudgeClient, source: str, note: str, args: argparse.Namespace) -> dict[str, Any]:
    user = FAITHFULNESS_USER_TEMPLATE.format(
        source=truncate_middle(source, args.max_source_chars),
        note=truncate_middle(note, args.max_note_chars),
    )
    data = client.judge("faithfulness", FAITHFULNESS_SYSTEM, user)
    claims = data.get("claims") if isinstance(data.get("claims"), list) else []
    unsupported = [c for c in claims if isinstance(c, dict) and c.get("verdict") == "unsupported"]
    likert = data.get("faithfulness_likert")
    likert = int(likert) if isinstance(likert, (int, float)) and 1 <= int(likert) <= 5 else 1
    return {
        "likert": likert,
        "n_claims": len(claims),
        "n_unsupported": len(unsupported),
        "unsupported_claims": unsupported,
        "raw": data,
    }


def judge_completeness(client: JudgeClient, gt: str, note: str, args: argparse.Namespace) -> dict[str, Any]:
    user = COMPLETENESS_USER_TEMPLATE.format(
        gt=truncate_middle(gt, args.max_note_chars),
        note=truncate_middle(note, args.max_note_chars),
    )
    data = client.judge("completeness", COMPLETENESS_SYSTEM, user)
    facts = data.get("critical_facts") if isinstance(data.get("critical_facts"), list) else []
    statuses = [clean_scalar(f.get("status")).lower() for f in facts if isinstance(f, dict)]
    n_absent = sum(s == "absent" for s in statuses)
    n_partial = sum(s == "partial" for s in statuses)
    return {
        "n_critical": len(statuses),
        "n_absent": n_absent,
        "n_partial": n_partial,
        "complete_strict": bool(statuses) and n_absent == 0 and n_partial == 0,
        "complete_lenient": bool(statuses) and n_absent == 0,
        "raw": data,
    }


def judge_style_pair(
    client: JudgeClient,
    references_block: str,
    note_method: str,
    note_opponent: str,
    args: argparse.Namespace,
) -> float:
    """Position-swapped pairwise battle. Returns method score in [0, 1] (win=1, tie=0.5)."""
    note_m = truncate_middle(note_method, args.max_note_chars)
    note_o = truncate_middle(note_opponent, args.max_note_chars)
    if not note_m and not note_o:
        return 0.5
    if not note_m:
        return 0.0
    if not note_o:
        return 1.0
    score = 0.0
    for method_is_a in (True, False):
        user = STYLE_USER_TEMPLATE.format(
            references=references_block,
            note_a=note_m if method_is_a else note_o,
            note_b=note_o if method_is_a else note_m,
        )
        data = client.judge("style", STYLE_SYSTEM, user)
        winner = clean_scalar(data.get("winner")).upper()
        if winner == "TIE":
            score += 0.5
        elif (winner == "A") == method_is_a and winner in {"A", "B"}:
            score += 1.0
    return score / 2.0


def evaluate(args: argparse.Namespace) -> int:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    # Shared across all three eval scripts: cache keys are (model, task, prompt)
    # only, so identical judge calls (e.g., exp4 faithfulness in Tables 1 and 2)
    # are paid for once.
    client = JudgeClient(args, args.out_dir / "judge_cache_shared.jsonl")

    frames: dict[str, pd.DataFrame] = {}
    for method in dict.fromkeys(list(args.methods) + [args.style_opponent]):
        frames[method] = load_method_frame(args.csv_dir / METHODS[method][1])

    anchor = frames[args.methods[0]]
    sample = build_sample(anchor, args.sample_per_professor, args.sample_seed)
    references = build_style_references(anchor, max(args.style_reference_count * 4, 12))
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
            result.update(
                likert=1, n_claims=0, n_unsupported=0, hallucination_free=False, faithful_pass=False,
                faithful_borderline=False, n_critical=0, n_absent=0, n_partial=0,
                complete_strict=False, complete_lenient=False, style_score=None, empty_note=True,
            )
            if method != args.style_opponent and indexed[args.style_opponent][record_id]["note"]:
                result["style_score"] = 0.0
            return result
        faith = judge_faithfulness(client, row["source"], row["note"], args)
        comp = judge_completeness(client, row["gt"], row["note"], args)
        result.update(
            likert=faith["likert"],
            n_claims=faith["n_claims"],
            n_unsupported=faith["n_unsupported"],
            hallucination_free=faith["n_unsupported"] == 0,
            faithful_pass=faith["likert"] >= args.pass_threshold,
            faithful_borderline=faith["likert"] == 3,
            n_critical=comp["n_critical"],
            n_absent=comp["n_absent"],
            n_partial=comp["n_partial"],
            complete_strict=comp["complete_strict"],
            complete_lenient=comp["complete_lenient"],
            empty_note=False,
            style_score=None,
        )
        if method != args.style_opponent:
            ref_block = style_reference_block(
                references, row["professor"], record_id, args.style_reference_count, args.max_note_chars
            )
            opponent_note = indexed[args.style_opponent][record_id]["note"]
            result["style_score"] = judge_style_pair(client, ref_block, row["note"], opponent_note, args)
        return result

    jobs = [(m, rid) for m in args.methods for rid in sorted(sample) if rid in indexed[m]]
    rows: list[dict[str, Any]] = []
    detail_path = args.out_dir / f"{TABLE_NAME}_per_note.jsonl"
    with detail_path.open("w", encoding="utf-8") as detail, ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(evaluate_record, m, rid): (m, rid) for m, rid in jobs}
        progress = tqdm(total=len(futures), desc="Judging", unit="note", disable=args.no_progress, dynamic_ncols=True)
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
    errors = df[df.get("error").notna()] if "error" in df.columns else pd.DataFrame()
    if len(errors):
        print(f"WARNING: {len(errors)} judge failures (excluded from aggregation); see {detail_path}", file=sys.stderr)
        df = df[df["error"].isna()] if "error" in df.columns else df

    table_rows = []
    for method in args.methods:
        sub = df[df["method"] == method]
        if sub.empty:
            continue
        style_scores = sub["style_score"].dropna()
        table_rows.append(
            {
                "method": METHODS[method][0],
                "n": len(sub),
                "faithful_pass_pct": 100 * sub["faithful_pass"].mean(),
                "faithful_borderline_pct": 100 * sub["faithful_borderline"].mean(),
                "hallucination_free_pct": 100 * sub["hallucination_free"].mean(),
                "unsupported_per_note": sub["n_unsupported"].mean(),
                "critical_complete_pct": 100 * sub["complete_strict"].mean(),
                "critical_complete_lenient_pct": 100 * sub["complete_lenient"].mean(),
                "style_win_pct": 100 * style_scores.astype(float).mean() if len(style_scores) else float("nan"),
                "style_n": len(style_scores),
            }
        )
    table = pd.DataFrame(table_rows)
    table_path = args.out_dir / f"{TABLE_NAME}.csv"
    table.to_csv(table_path, index=False, encoding="utf-8-sig", float_format="%.2f")

    print(f"\n=== Table 2: Few-shot ablation (style opponent: {METHODS[args.style_opponent][0]}) ===")
    with pd.option_context("display.max_columns", None, "display.width", 200, "display.float_format", "{:.2f}".format):
        print(table.to_string(index=False))
    print(f"\nSaved table : {table_path}")
    print(f"Saved detail: {detail_path}")
    print(f"Judge cache : {client.cache_path}")
    return 0


def main() -> None:
    raise SystemExit(evaluate(parse_args()))


if __name__ == "__main__":
    main()


"""
export OPENAI_API_KEY=sk-...
python eval_table2_ablation.py --judge_model gpt-4o --sample_per_professor 10
python eval_table2_ablation.py --sample_per_professor 0   # full 1050 records (expensive)
python eval_table2_ablation.py --dry_run                  # plumbing test, no API
"""
