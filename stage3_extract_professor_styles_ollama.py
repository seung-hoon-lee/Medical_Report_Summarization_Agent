#!/usr/bin/env python3
"""
Extract compact professor-specific outpatient-note style prompts from CSV samples using local Ollama.

Expected input:
    prof_samples/
      강창현_Samples.csv
      한원식_Samples.csv
      ...

Each CSV should contain real outpatient-note examples. The script tries to detect columns such as:
    output, Output, actual_output, note, outpatient_note, 외래기록지
Optionally it also uses:
    input, Input, source, original_record, 의료기록지
    generated_note

Main output:
    Professor_Styles_extracted.xlsx
      - Sheet1: professor, style_prompt
      - Style_JSON: structured extracted style profile
      - Extraction_Audit: input/output stats and Ollama metadata
      - Common_Prompts: recommended global prompts for the generation agent

Design goal:
    The extracted prompt must not merely describe wording style.
    It must strongly capture:
      1) note compactness,
      2) content-selection priority,
      3) what this professor omits even when factual,
      4) abbreviation/date/header habits,
      5) how to avoid operative-summary over-generation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_MODEL = "qwen3.5:9b"
DEFAULT_INPUT_DIR = Path("prof_samples")
DEFAULT_OUTPUT_XLSX = Path("Professor_Styles_extracted.xlsx")
DEFAULT_AUDIT_JSONL = Path("Professor_Styles_extracted_audit.jsonl")

OUTPUT_COLUMN_CANDIDATES = (
    "output", "Output", "OUTPUT", "actual_output", "Actual_Output",
    "reference_output", "Reference_Output", "gt", "GT", "ground_truth",
    "Ground_Truth", "note", "Note", "outpatient_note", "Outpatient_Note",
    "외래기록지", "실제외래기록지",
)

INPUT_COLUMN_CANDIDATES = (
    "input", "Input", "INPUT", "source", "Source", "original_record",
    "Original_Record", "medical_record", "Medical_Record", "의료기록지", "원본의료기록지",
)

GENERATED_COLUMN_CANDIDATES = (
    "generated_note", "Generated_Note", "prediction", "Prediction", "pred", "Pred",
)

DATE_RE = re.compile(
    r"\b(?:\d{4}[./-]\d{1,2}(?:[./-]\d{1,2})?|\d{2}[./-]\d{1,2}[./-]\d{1,2}|'\d{2}[./-]\d{1,2}[./-]\d{1,2})\b"
)

ABBREVIATION_RE = re.compile(
    r"\b("
    r"s/p|d/t|R/O|r/o|postop|post-op|POD|f/u|F/U|"
    r"BCS|SLNB|ALND|MRM|TM|TE|ADM|NAC|RTx|CTx|HTx|"
    r"LAR|AR|APR|RHC|LHC|TC|IRA|TME|ISR|stoma|AVF|AVG|"
    r"VATS|RUL|RML|RLL|LUL|LLL|wedge|lobectomy|"
    r"GB|CBD|LN|LNx|Bx|op|OP|rec|fu"
    r")\b",
    flags=re.IGNORECASE,
)

TECH_DETAIL_TERMS = (
    "trocar", "port", "incision", "dissection", "ligation", "anastomosis",
    "stapler", "drain", "JP", "chest tube", "foley", "closure", "suture",
    "EBL", "blood loss", "anesthesia", "leakage test", "air leak", "repair",
    "mucosal injury", "vessel", "artery", "vein", "high ligation",
    "specimen", "frozen", "frozen biopsy", "gross", "operative finding",
    "수술소견", "절개", "박리", "결찰", "문합", "배액관", "봉합", "출혈량",
    "마취", "누출검사", "수리", "손상", "검체",
)

NEGATIVE_DETAIL_TERMS = (
    "no complication", "without complication", "no evidence", "negative",
    "정상", "특이소견 없음", "합병증 없음", "재발 없음", "전이 없음",
)

COMMON_PROMPTS = {
    "global_medical_safety_prompt": """You are a safety-critical medical documentation generator.

Hard rules:
1. Use only facts explicitly present in CURRENT_ROW_FACTS.
2. Never use information from another row.
3. Never use information from another professor's style prompt as medical content.
4. Professor style controls only formatting, phrase style, abbreviation habits, section structure, ordering, compactness, and omission policy.
5. Professor style is not a source of patient-specific medical facts.
6. Do not infer diagnosis, surgery history, medications, dates, pathology, treatment plan, recurrence, metastasis, or follow-up schedule unless explicitly present in CURRENT_ROW_FACTS.
7. Do not complete missing medical information based on common clinical patterns.
8. Preserve uncertainty exactly.
9. The task is not to summarize all available facts. The task is to write the final outpatient note in the target professor's style.
10. Include only facts that are both explicitly supported and stylistically likely to appear in the final outpatient note.
11. Omit factual but low-priority details when the professor's style is compact.""",
    "compact_specialty_rule_pack": """General compact outpatient-note rule:
- Match the professor's reference output compactness, not the input record richness.
- Do not summarize operative reports.
- Do not write discharge summaries.
- Do not expand abbreviations if the professor's real notes use abbreviations.
- Remove low-priority operative details first: approach, ports/trocars, dissection, ligation, anastomosis device, drain/chest tube, repair, EBL, anesthesia, closure, routine negative findings, discharge course, long PMHx, incidental comorbidities.
- Never omit core outpatient-note anchors if explicitly supported and typical for the professor: main diagnosis/R/O diagnosis, main operation/procedure, operation/procedure date, short postop/follow-up/status phrase.""",
    "recommended_runtime_prompt_layout": """Use the generated Sheet1 style_prompt with the global safety prompt and CURRENT_ROW_FACTS.
Do not put all professor prompts in the same generation call.
Use only the style prompt for the matched professor.""",
}


@dataclass
class NoteStats:
    n_rows: int
    n_valid_outputs: int
    mean_chars: float
    median_chars: float
    mean_lines: float
    median_lines: float
    p25_lines: float
    p75_lines: float
    short_note_rate_le_4_lines: float
    long_note_rate_ge_8_lines: float
    date_rate: float
    abbreviation_rate: float
    section_header_rate: float
    bullet_rate: float
    numbered_rate: float
    tech_detail_rate: float
    negative_detail_rate: float
    common_abbreviations: str
    common_first_lines: str


def clean_scalar(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    text = str(value).strip()
    if text.lower() in {"nan", "none", "nat"}:
        return ""
    return text


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8-sig")).hexdigest()


def stable_json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def professor_name_from_path(path: Path) -> str:
    # Normalize Korean filenames created on macOS/Linux so professor names match
    # the Professor_ID values used by Stage 4.
    name = unicodedata.normalize("NFC", path.stem).strip()
    for suffix in (
        "_Samples", "_samples", "-Samples", "-samples",
        "_Sample", "_sample", "-Sample", "-sample",
    ):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    return unicodedata.normalize("NFC", name.strip())


def choose_column(df: pd.DataFrame, candidates: tuple[str, ...], required: bool = False, role: str = "") -> str | None:
    columns = list(df.columns)
    for candidate in candidates:
        if candidate in columns:
            return candidate

    normalized = {str(c).replace(" ", "").replace("_", "").lower(): c for c in columns}
    for candidate in candidates:
        key = candidate.replace(" ", "").replace("_", "").lower()
        if key in normalized:
            return str(normalized[key])

    if required:
        raise ValueError(
            f"Could not find required {role or 'column'} column. "
            f"Available columns: {columns}. Candidates: {list(candidates)}"
        )
    return None


def non_empty_lines(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]


def looks_like_section_header(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if "<|section_start|>" in stripped or "<|section_end|>" in stripped:
        return True
    if re.search(r"(Description|소견|Assessment|Plan|진단|계획|수술|병리|처방)\s*[:：<-]", stripped):
        return True
    if stripped.endswith(":") and len(stripped) <= 40:
        return True
    return False


def count_term_presence(notes: list[str], terms: tuple[str, ...]) -> float:
    if not notes:
        return 0.0
    count = 0
    lowered_terms = tuple(t.lower() for t in terms)
    for note in notes:
        lowered = note.lower()
        if any(term in lowered for term in lowered_terms):
            count += 1
    return count / len(notes)


def compute_note_stats(outputs: list[str]) -> NoteStats:
    notes = [o for o in outputs if clean_scalar(o)]
    n = len(notes)
    if n == 0:
        return NoteStats(
            n_rows=len(outputs),
            n_valid_outputs=0,
            mean_chars=0,
            median_chars=0,
            mean_lines=0,
            median_lines=0,
            p25_lines=0,
            p75_lines=0,
            short_note_rate_le_4_lines=0,
            long_note_rate_ge_8_lines=0,
            date_rate=0,
            abbreviation_rate=0,
            section_header_rate=0,
            bullet_rate=0,
            numbered_rate=0,
            tech_detail_rate=0,
            negative_detail_rate=0,
            common_abbreviations="",
            common_first_lines="",
        )

    char_lens = [len(note) for note in notes]
    line_lens = [len(non_empty_lines(note)) for note in notes]
    first_lines = [non_empty_lines(note)[0] for note in notes if non_empty_lines(note)]

    abbreviations: list[str] = []
    for note in notes:
        abbreviations.extend([m.group(0) for m in ABBREVIATION_RE.finditer(note)])

    abbrev_counts = pd.Series(abbreviations).value_counts().head(20) if abbreviations else pd.Series(dtype=int)
    first_line_counts = pd.Series(first_lines).value_counts().head(10) if first_lines else pd.Series(dtype=int)

    return NoteStats(
        n_rows=len(outputs),
        n_valid_outputs=n,
        mean_chars=float(sum(char_lens) / n),
        median_chars=float(pd.Series(char_lens).median()),
        mean_lines=float(sum(line_lens) / n),
        median_lines=float(pd.Series(line_lens).median()),
        p25_lines=float(pd.Series(line_lens).quantile(0.25)),
        p75_lines=float(pd.Series(line_lens).quantile(0.75)),
        short_note_rate_le_4_lines=float(sum(x <= 4 for x in line_lens) / n),
        long_note_rate_ge_8_lines=float(sum(x >= 8 for x in line_lens) / n),
        date_rate=float(sum(bool(DATE_RE.search(note)) for note in notes) / n),
        abbreviation_rate=float(sum(bool(ABBREVIATION_RE.search(note)) for note in notes) / n),
        section_header_rate=float(
            sum(any(looks_like_section_header(line) for line in non_empty_lines(note)) for note in notes) / n
        ),
        bullet_rate=float(sum(any(line.startswith(("-", "*", "•")) for line in non_empty_lines(note)) for note in notes) / n),
        numbered_rate=float(sum(bool(re.search(r"(?m)^\s*\d+[.)]", note)) for note in notes) / n),
        tech_detail_rate=count_term_presence(notes, TECH_DETAIL_TERMS),
        negative_detail_rate=count_term_presence(notes, NEGATIVE_DETAIL_TERMS),
        common_abbreviations=", ".join([f"{k}:{v}" for k, v in abbrev_counts.items()]),
        common_first_lines=" | ".join([f"{k} ({v})" for k, v in first_line_counts.items()]),
    )


def format_percent(x: float) -> str:
    return f"{x * 100:.1f}%"


def build_reference_examples(
    df: pd.DataFrame,
    output_col: str,
    input_col: str | None,
    generated_col: str | None,
    max_examples: int,
    max_output_chars: int,
    max_input_chars: int,
) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    df_work = df.copy()
    df_work["_output_len"] = df_work[output_col].map(lambda x: len(clean_scalar(x)))

    if len(df_work) > max_examples:
        quantile_targets = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
        selected_indices: list[int] = []
        lengths = df_work["_output_len"]
        for q in quantile_targets:
            target = lengths.quantile(q)
            idx = (lengths - target).abs().sort_values().index[0]
            if idx not in selected_indices:
                selected_indices.append(idx)
        remaining = [idx for idx in df_work.sort_values("_output_len").index if idx not in selected_indices]
        selected_indices.extend(remaining[: max(0, max_examples - len(selected_indices))])
        selected_indices = selected_indices[:max_examples]
        df_work = df_work.loc[selected_indices]

    for i, row in df_work.iterrows():
        output = truncate_middle(clean_scalar(row.get(output_col)), max_output_chars)
        if not output:
            continue
        item = {
            "case_index": str(i),
            "reference_output": output,
        }
        if input_col:
            item["source_input_excerpt"] = truncate_middle(clean_scalar(row.get(input_col)), max_input_chars)
        if generated_col:
            item["previous_generated_note_excerpt"] = truncate_middle(clean_scalar(row.get(generated_col)), max_output_chars)
        records.append(item)
    return records


def truncate_middle(text: str, max_chars: int) -> str:
    text = clean_scalar(text)
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    keep_left = max_chars // 2
    keep_right = max(0, max_chars - keep_left - 40)
    return text[:keep_left].rstrip() + "\n...[TRUNCATED]...\n" + text[-keep_right:].lstrip()


def build_style_extraction_messages(
    professor: str,
    stats: NoteStats,
    examples: list[dict[str, str]],
    target_chars: int,
) -> list[dict[str, str]]:
    system_prompt = """You are an expert clinical documentation style analyst.

You extract reusable professor-specific outpatient-note style prompts from real reference outpatient notes.

Critical objective:
- Do NOT merely describe surface wording.
- The most important part is CONTENT SELECTION STYLE:
  what this professor keeps, what this professor omits, and how compactly they write.
- The resulting style_prompt will be used by another medical-note generation agent.
- The generation agent must avoid over-generating operative summaries.
- Therefore, your style prompt must strongly tell the generation agent NOT to include low-priority details even if those details are factual.

Safety:
- Never copy patient-specific facts from the examples into the final style_prompt.
- Examples are evidence for style only, not reusable clinical content.
- Do not include names, dates, diagnoses, operation dates, or patient-specific facts as fixed content.
- You may mention generic placeholders such as [diagnosis], [operation], [date], [status].
- Return valid JSON only. No markdown, no explanation, no <think> block."""

    user_prompt = f"""
Analyze the real outpatient-note samples for professor: {professor}

The notes are real reference OUTPUT notes, not source operative records.
Extract a compact but strict style prompt for generating future notes in this professor's style.

Observed note statistics:
- number of valid output notes: {stats.n_valid_outputs}
- mean characters: {stats.mean_chars:.1f}
- median characters: {stats.median_chars:.1f}
- mean non-empty lines: {stats.mean_lines:.1f}
- median non-empty lines: {stats.median_lines:.1f}
- 25th-75th percentile lines: {stats.p25_lines:.1f}-{stats.p75_lines:.1f}
- short note rate <=4 lines: {format_percent(stats.short_note_rate_le_4_lines)}
- long note rate >=8 lines: {format_percent(stats.long_note_rate_ge_8_lines)}
- date usage rate: {format_percent(stats.date_rate)}
- abbreviation usage rate: {format_percent(stats.abbreviation_rate)}
- section header usage rate: {format_percent(stats.section_header_rate)}
- bullet/list usage rate: {format_percent(stats.bullet_rate)}
- numbered-list usage rate: {format_percent(stats.numbered_rate)}
- operative technical detail rate: {format_percent(stats.tech_detail_rate)}
- routine negative/no-complication detail rate: {format_percent(stats.negative_detail_rate)}
- common abbreviations observed: {stats.common_abbreviations or "none"}
- common first lines observed: {stats.common_first_lines or "none"}

Reference examples:
{json.dumps(examples, ensure_ascii=False, indent=2)}

You must produce JSON with exactly these keys:
{{
  "professor": "{professor}",
  "style_summary": "short high-level style analysis",
  "length_policy": "how many lines/chars the note usually has and how aggressively to compress",
  "content_priority": ["ordered content priorities; these are priorities, not mandatory sections"],
  "strong_omit_rules": ["details to omit even when factual"],
  "format_rules": ["headers, bullet style, date format, spacing, line order"],
  "abbreviation_rules": ["abbreviation and shorthand habits; include do-not-expand rules"],
  "unknown_policy": "how to handle missing fields in this style",
  "style_prompt": "a final reusable prompt for a generation agent"
}}

Requirements for style_prompt:
- English only.
- Around {target_chars} characters if possible; do not exceed {target_chars + 400} characters.
- Keep every JSON list short: maximum 5 items per list.
- Keep style_summary, length_policy, and unknown_policy to one sentence each.
- Must be directly usable as the professor-specific style_prompt.
- Must include:
  1. Professor style target
  2. Style
  3. Length
  4. Content priority, not mandatory sections
  5. Strong omit rule
  6. Format / notation
  7. Mini example pattern with placeholders only
- Must explicitly say: do not summarize the operative report.
- Must explicitly say: omit factual but low-priority details if not typical of this professor.
- Must explicitly say: preserve core anchors if explicitly supported and typical for the professor:
  main diagnosis/R/O diagnosis, main operation/procedure, date, short postop/follow-up/status phrase.
- Must not include patient-specific diagnoses/dates/procedures copied from samples.
- Must not create a long multi-section template unless the professor's actual outputs are long.
- Do not use the word "unknown" everywhere; only say to write unknown when the style/schema requires a missing field, otherwise omit missing fields.
""".strip()

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def strip_model_thinking(text: str) -> str:
    cleaned = re.sub(r"<think>.*?(?:</think>|$)", "", text, flags=re.IGNORECASE | re.DOTALL).strip()
    cleaned = re.sub(r"^```(?:json|JSON)?\s*", "", cleaned).strip()
    cleaned = re.sub(r"\s*```$", "", cleaned).strip()
    return cleaned


def extract_json_object(text: str) -> dict[str, Any]:
    text = strip_model_thinking(text)
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        candidate = text[start : end + 1]
        try:
            data = json.loads(candidate)
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass

    raise ValueError(
        "Could not parse JSON object from model output. "
        "The model output is likely incomplete or malformed."
        f" Output head:\n{text[:1000]}"
    )


def build_json_repair_messages(
    professor: str,
    raw_output: str,
    stats: NoteStats,
    target_chars: int,
) -> list[dict[str, str]]:
    """Ask the model to convert a partial/malformed style extraction into short valid JSON."""
    system_prompt = (
        "You repair malformed JSON from a clinical style extraction model. "
        "Return valid JSON only. No markdown. No explanation. No <think> block. "
        "Keep the result short."
    )
    user_prompt = f"""
Repair or reconstruct the JSON for professor {professor}.

The previous model output may be truncated:
{truncate_middle(raw_output, 2500)}

Use these statistics if needed:
- median lines: {stats.median_lines:.1f}
- mean lines: {stats.mean_lines:.1f}
- median chars: {stats.median_chars:.1f}
- abbreviation rate: {format_percent(stats.abbreviation_rate)}
- section header rate: {format_percent(stats.section_header_rate)}
- bullet/list rate: {format_percent(stats.bullet_rate)}
- common abbreviations: {stats.common_abbreviations or "none"}

Return JSON with exactly these keys:
{{
  "professor": "{professor}",
  "style_summary": "one sentence",
  "length_policy": "one sentence",
  "content_priority": ["max 4 short items"],
  "strong_omit_rules": ["max 4 short items"],
  "format_rules": ["max 4 short items"],
  "abbreviation_rules": ["max 4 short items"],
  "unknown_policy": "one sentence",
  "style_prompt": "final reusable prompt, <= {target_chars} characters"
}}

The style_prompt must include:
- Professor style target
- Style
- Length
- Content priority, not mandatory sections
- Strong omit rule
- Format / notation
- Mini example pattern with placeholders only
- The sentence: Do not summarize the operative report.
- The sentence: Omit factual but low-priority details if not typical of this professor.
""".strip()
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def deterministic_style_json(professor: str, stats: NoteStats) -> dict[str, Any]:
    """Last-resort non-LLM style profile so one bad professor never blocks the batch."""
    if stats.median_lines <= 4:
        length = "Usually 2-4 non-empty lines; compress aggressively."
        mini = "[status or diagnosis]\n[date] [operation/procedure]"
    elif stats.median_lines <= 8:
        length = "Usually 4-8 non-empty lines; keep a compact timeline/list style."
        mini = "[status]\n[date] [operation/procedure]\n#1. [key diagnosis/history if typical]"
    else:
        length = "Often a longer compact timeline/problem-list note, but still avoid narrative expansion."
        mini = "[status]\n[date] [operation/procedure]\n#1. [core condition]\n- s/p [key prior treatment/date]"

    header_rule = (
        "Use the professor's observed section header style when present."
        if stats.section_header_rate >= 0.5
        else "Do not add section headers unless clearly required by the source style."
    )
    bullet_rule = (
        "Use short bullet/problem-list lines when the reference style uses them."
        if stats.bullet_rate >= 0.3 or stats.numbered_rate >= 0.3
        else "Avoid expanding into many bullets unless the reference style requires them."
    )
    abbrev_rule = (
        "Preserve abbreviations and do not expand them into long phrases."
        if stats.abbreviation_rate >= 0.4
        else "Use concise medical shorthand only when clearly supported."
    )

    style_prompt = f"""Professor style target: {professor}

Style:
- Compact outpatient-note style based on the professor's reference outputs.
- Prefer note-like fragments over explanatory narrative.
- Do not summarize the operative report.

Length:
- {length}
- Match the reference output compactness, not the source record richness.

Content priority, not mandatory sections:
1. Main diagnosis/R/O diagnosis or visit status if typical.
2. Main operation/procedure with date if supported.
3. Key timeline/problem-list items only if typical for this professor.
4. Short postop/follow-up/status phrase if supported.

Strong omit rule:
- Omit factual but low-priority details if not typical of this professor.
- Omit routine operative technical steps, anesthesia, EBL, drain/chest tube/closure/repair details, routine negative findings, discharge course, and incidental comorbidities unless the reference style consistently includes them.
- Do not fill every possible category.

Format / notation:
- {header_rule}
- {bullet_rule}
- {abbrev_rule}
- Preserve source dates and laterality exactly.

Mini example pattern:
{mini}"""
    return {
        "professor": professor,
        "style_summary": "Deterministic fallback style profile generated because Ollama did not return valid JSON.",
        "length_policy": length,
        "content_priority": [
            "main diagnosis/R/O diagnosis or visit status",
            "main operation/procedure with date",
            "key timeline/problem-list items only if typical",
            "short postop/follow-up/status phrase",
        ],
        "strong_omit_rules": [
            "do not summarize operative report",
            "omit factual but low-priority details",
            "omit routine technical/discharge/negative details",
        ],
        "format_rules": [header_rule, bullet_rule, "preserve dates and laterality exactly"],
        "abbreviation_rules": [abbrev_rule],
        "unknown_policy": "Omit missing fields unless the style/schema explicitly requires an unknown placeholder.",
        "style_prompt": style_prompt,
    }


def parse_or_repair_style_json(
    raw_output: str,
    professor: str,
    stats: NoteStats,
    extractor: "OllamaStyleExtractor",
    target_chars: int,
) -> tuple[dict[str, Any], str, dict[str, Any]]:
    """Parse JSON; if malformed, run one repair call; if still bad, use deterministic fallback."""
    try:
        return extract_json_object(raw_output), raw_output, {"parse_mode": "direct"}
    except Exception as direct_exc:  # noqa: BLE001
        repair_messages = build_json_repair_messages(
            professor=professor,
            raw_output=raw_output,
            stats=stats,
            target_chars=target_chars,
        )
        try:
            repaired_output, repair_metadata = extractor.chat(repair_messages)
            data = extract_json_object(repaired_output)
            repair_metadata = dict(repair_metadata)
            repair_metadata["parse_mode"] = "repair"
            repair_metadata["direct_parse_error"] = str(direct_exc)
            return data, repaired_output, repair_metadata
        except Exception as repair_exc:  # noqa: BLE001
            data = deterministic_style_json(professor, stats)
            return data, stable_json(data), {
                "parse_mode": "deterministic_fallback",
                "direct_parse_error": str(direct_exc),
                "repair_error": str(repair_exc),
            }


class OllamaStyleExtractor:
    def __init__(
        self,
        model: str,
        host: str | None,
        num_ctx: int | None,
        num_predict: int,
        seed: int | None,
        retries: int,
        retry_sleep: float,
    ) -> None:
        try:
            import ollama
        except ImportError as exc:
            raise RuntimeError("Install Ollama Python client first: pip install ollama") from exc

        self.client = ollama.Client(host=host) if host else ollama.Client()
        self.model = model
        self.num_ctx = num_ctx
        self.num_predict = num_predict
        self.seed = seed
        self.retries = max(0, retries)
        self.retry_sleep = max(0.0, retry_sleep)

    def chat(self, messages: list[dict[str, str]]) -> tuple[str, dict[str, Any]]:
        options: dict[str, Any] = {
            "temperature": 0,
            "top_p": 1,
            "num_predict": self.num_predict,
        }
        if self.num_ctx:
            options["num_ctx"] = self.num_ctx
        if self.seed is not None:
            options["seed"] = self.seed

        last_exc: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                kwargs = {
                    "model": self.model,
                    "messages": messages,
                    "options": options,
                    "stream": False,
                    "keep_alive": "30m",
                    "format": "json",
                }
                try:
                    response = self.client.chat(**kwargs, think=False)
                except TypeError:
                    response = self.client.chat(**kwargs)

                if isinstance(response, dict):
                    msg = response.get("message", {})
                    content = msg.get("content", "") if isinstance(msg, dict) else getattr(msg, "content", "")
                else:
                    msg = getattr(response, "message", {})
                    content = msg.get("content", "") if isinstance(msg, dict) else getattr(msg, "content", "")

                metadata = {
                    "model": response_get(response, "model"),
                    "done_reason": response_get(response, "done_reason"),
                    "total_duration": response_get(response, "total_duration"),
                    "prompt_eval_count": response_get(response, "prompt_eval_count"),
                    "eval_count": response_get(response, "eval_count"),
                    "attempt": attempt + 1,
                }
                return strip_model_thinking(clean_scalar(content)), metadata
            except Exception as exc:
                last_exc = exc
                if attempt < self.retries:
                    time.sleep(self.retry_sleep)
        raise RuntimeError(f"Ollama request failed after {self.retries + 1} attempts: {last_exc}")


def response_get(response: Any, key: str) -> Any:
    if isinstance(response, dict):
        return response.get(key)
    return getattr(response, key, None)


def validate_style_json(data: dict[str, Any], professor: str) -> list[str]:
    warnings: list[str] = []
    required = [
        "professor", "style_summary", "length_policy", "content_priority",
        "strong_omit_rules", "format_rules", "abbreviation_rules", "unknown_policy", "style_prompt",
    ]
    for key in required:
        if key not in data:
            warnings.append(f"missing key: {key}")
    style_prompt = clean_scalar(data.get("style_prompt"))
    if not style_prompt:
        warnings.append("empty style_prompt")
    if professor not in clean_scalar(data.get("professor", professor)):
        warnings.append("professor field may not match filename")
    lower = style_prompt.lower()
    must_include_phrases = [
        "do not summarize the operative report",
        "content priority",
        "omit",
        "mini example",
    ]
    for phrase in must_include_phrases:
        if phrase not in lower:
            warnings.append(f"style_prompt missing recommended phrase/concept: {phrase}")
    if len(style_prompt) > 3500:
        warnings.append(f"style_prompt may be too long: {len(style_prompt)} chars")
    if len(style_prompt) < 500:
        warnings.append(f"style_prompt may be too short: {len(style_prompt)} chars")
    return warnings


def repair_style_prompt_if_needed(data: dict[str, Any], professor: str, stats: NoteStats) -> dict[str, Any]:
    prompt = clean_scalar(data.get("style_prompt"))
    if not prompt:
        return data

    lower = prompt.lower()
    additions: list[str] = []

    if "do not summarize the operative report" not in lower:
        additions.append("- Do not summarize the operative report.")
    if "factual but low-priority" not in lower:
        additions.append("- Omit factual but low-priority details when they are not typical of this professor's final outpatient notes.")
    if "core outpatient-note anchors" not in lower and "core anchors" not in lower:
        additions.append(
            "- Preserve core outpatient-note anchors if explicitly supported and typical for this professor: main diagnosis/R/O diagnosis, main operation/procedure, date, and short postop/follow-up/status phrase."
        )
    if "do not expand abbreviations" not in lower and stats.abbreviation_rate >= 0.4:
        additions.append("- Do not expand abbreviations when this professor's real notes use abbreviations.")

    if additions:
        prompt = prompt.rstrip() + "\n\nCritical anti-overgeneration rules:\n" + "\n".join(additions)
        data["style_prompt"] = prompt
    return data


def process_one_file(path: Path, extractor: OllamaStyleExtractor, args: argparse.Namespace) -> dict[str, Any]:
    professor = professor_name_from_path(path)
    df = pd.read_csv(path, dtype=str, keep_default_na=False).fillna("")

    output_col = choose_column(df, OUTPUT_COLUMN_CANDIDATES, required=True, role="reference output")
    input_col = choose_column(df, INPUT_COLUMN_CANDIDATES, required=False, role="source input")
    generated_col = choose_column(df, GENERATED_COLUMN_CANDIDATES, required=False, role="previous generated note")

    assert output_col is not None
    outputs = [clean_scalar(x) for x in df[output_col].tolist()]
    stats = compute_note_stats(outputs)
    examples = build_reference_examples(
        df=df,
        output_col=output_col,
        input_col=input_col,
        generated_col=generated_col,
        max_examples=args.max_examples,
        max_output_chars=args.max_output_chars,
        max_input_chars=args.max_input_chars,
    )

    if not examples:
        raise ValueError(f"No valid output examples in {path}")

    messages = build_style_extraction_messages(
        professor=professor,
        stats=stats,
        examples=examples,
        target_chars=args.target_style_chars,
    )

    if args.dry_run:
        style_json = {
            "professor": professor,
            "style_summary": "DRY RUN ONLY",
            "length_policy": "DRY RUN ONLY",
            "content_priority": ["DRY RUN ONLY"],
            "strong_omit_rules": ["DRY RUN ONLY"],
            "format_rules": ["DRY RUN ONLY"],
            "abbreviation_rules": ["DRY RUN ONLY"],
            "unknown_policy": "DRY RUN ONLY",
            "style_prompt": f"""Professor style target: {professor}

Style:
- DRY RUN ONLY.

Length:
- Match reference note compactness.

Content priority, not mandatory sections:
1. Main diagnosis/R/O diagnosis if typical.
2. Main operation/procedure with date if supported.
3. Short postop/follow-up/status phrase if typical.

Strong omit rule:
- Do not summarize the operative report.
- Omit factual but low-priority details when not typical.

Format / notation:
- Preserve professor-specific abbreviations.

Mini example pattern:
[diagnosis or status]
- s/p [operation] ([date])""",
        }
        metadata = {"backend": "dry_run"}
        raw_output = stable_json(style_json)
    else:
        raw_output, metadata = extractor.chat(messages)
        style_json, parsed_output, parse_metadata = parse_or_repair_style_json(
            raw_output=raw_output,
            professor=professor,
            stats=stats,
            extractor=extractor,
            target_chars=args.target_style_chars,
        )
        style_json = repair_style_prompt_if_needed(style_json, professor, stats)
        if parsed_output != raw_output:
            raw_output = parsed_output
        metadata = {**metadata, **parse_metadata}

    warnings = validate_style_json(style_json, professor)

    return {
        "professor": professor,
        "source_file": str(path),
        "n_rows": len(df),
        "output_col": output_col,
        "input_col": input_col or "",
        "generated_col": generated_col or "",
        "stats": asdict(stats),
        "examples_used": examples,
        "messages_hash": sha256_text(stable_json(messages)),
        "raw_model_output": raw_output,
        "style_json": style_json,
        "style_prompt": clean_scalar(style_json.get("style_prompt")),
        "style_prompt_hash": sha256_text(clean_scalar(style_json.get("style_prompt"))),
        "warnings": warnings,
        "metadata": metadata,
    }


def write_outputs(results: list[dict[str, Any]], output_xlsx: Path, audit_jsonl: Path) -> None:
    output_xlsx.parent.mkdir(parents=True, exist_ok=True)
    audit_jsonl.parent.mkdir(parents=True, exist_ok=True)

    sheet1_rows = []
    style_json_rows = []
    audit_rows = []

    for result in results:
        professor = result["professor"]
        stats = result["stats"]
        style_json = result["style_json"]

        sheet1_rows.append(
            {
                "professor": professor,
                "style_prompt": result["style_prompt"],
                "style_prompt_hash": result["style_prompt_hash"],
                "source_file": result["source_file"],
                "n_examples": result["n_rows"],
                "mean_lines": stats["mean_lines"],
                "median_lines": stats["median_lines"],
                "mean_chars": stats["mean_chars"],
                "median_chars": stats["median_chars"],
                "short_note_rate_le_4_lines": stats["short_note_rate_le_4_lines"],
                "abbreviation_rate": stats["abbreviation_rate"],
                "warnings": stable_json(result["warnings"]),
            }
        )

        style_json_rows.append(
            {
                "professor": professor,
                "style_summary": clean_scalar(style_json.get("style_summary")),
                "length_policy": clean_scalar(style_json.get("length_policy")),
                "content_priority": stable_json(style_json.get("content_priority", [])),
                "strong_omit_rules": stable_json(style_json.get("strong_omit_rules", [])),
                "format_rules": stable_json(style_json.get("format_rules", [])),
                "abbreviation_rules": stable_json(style_json.get("abbreviation_rules", [])),
                "unknown_policy": clean_scalar(style_json.get("unknown_policy")),
                "style_prompt": result["style_prompt"],
            }
        )

        audit_rows.append(
            {
                "professor": professor,
                "source_file": result["source_file"],
                "output_col": result["output_col"],
                "input_col": result["input_col"],
                "generated_col": result["generated_col"],
                "stats_json": stable_json(result["stats"]),
                "examples_used_json": stable_json(result["examples_used"]),
                "messages_hash": result["messages_hash"],
                "style_prompt_hash": result["style_prompt_hash"],
                "warnings": stable_json(result["warnings"]),
                "metadata_json": stable_json(result["metadata"]),
            }
        )

    common_rows = [{"key": k, "prompt_text": v} for k, v in COMMON_PROMPTS.items()]

    with pd.ExcelWriter(output_xlsx, engine="openpyxl") as writer:
        pd.DataFrame(sheet1_rows).to_excel(writer, sheet_name="Sheet1", index=False)
        pd.DataFrame(style_json_rows).to_excel(writer, sheet_name="Style_JSON", index=False)
        pd.DataFrame(audit_rows).to_excel(writer, sheet_name="Extraction_Audit", index=False)
        pd.DataFrame(common_rows).to_excel(writer, sheet_name="Common_Prompts", index=False)

    with audit_jsonl.open("w", encoding="utf-8-sig") as f:
        for result in results:
            f.write(stable_json(result) + "\n")


def find_sample_files(input_dir: Path) -> list[Path]:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    patterns = ("*_Samples.csv", "*_samples.csv", "*_Sample.csv", "*_sample.csv")
    files: list[Path] = []
    seen: set[Path] = set()
    for pattern in patterns:
        for path in sorted(input_dir.glob(pattern)):
            if path not in seen:
                files.append(path)
                seen.add(path)
    if not files:
        files = sorted(input_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {input_dir}")
    return files


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract compact professor-specific outpatient-note style prompts from prof_samples CSV files using Ollama."
    )
    parser.add_argument("--input_dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output_xlsx", type=Path, default=DEFAULT_OUTPUT_XLSX)
    parser.add_argument("--audit_jsonl", type=Path, default=DEFAULT_AUDIT_JSONL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--ollama_host", default=os.environ.get("OLLAMA_HOST"))
    parser.add_argument("--ollama_num_ctx", type=int, default=16384)
    parser.add_argument("--num_predict", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=1225)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--retry_sleep", type=float, default=2.0)
    parser.add_argument("--max_examples", type=int, default=8)
    parser.add_argument("--max_output_chars", type=int, default=600)
    parser.add_argument("--max_input_chars", type=int, default=0)
    parser.add_argument("--target_style_chars", type=int, default=850)
    parser.add_argument("--professor", help="Optional exact professor name to process.")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--continue_on_error", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    files = find_sample_files(args.input_dir)

    if args.professor:
        files = [p for p in files if professor_name_from_path(p) == args.professor]
        if not files:
            raise SystemExit(f"No sample CSV matched professor: {args.professor}")

    extractor = OllamaStyleExtractor(
        model=args.model,
        host=args.ollama_host,
        num_ctx=args.ollama_num_ctx,
        num_predict=args.num_predict,
        seed=args.seed,
        retries=args.retries,
        retry_sleep=args.retry_sleep,
    )

    results: list[dict[str, Any]] = []
    errors: list[str] = []

    for i, path in enumerate(files, start=1):
        professor = professor_name_from_path(path)
        print(f"[{i}/{len(files)}] Extracting style: {professor} <- {path}", file=sys.stderr)
        try:
            result = process_one_file(path, extractor, args)
            results.append(result)
            if result["warnings"]:
                print(f"  warnings: {result['warnings']}", file=sys.stderr)
            print(
                f"  style_prompt chars={len(result['style_prompt'])}, hash={result['style_prompt_hash'][:10]}",
                file=sys.stderr,
            )
        except Exception as exc:
            message = f"{path}: {exc}"
            errors.append(message)
            print(f"  ERROR: {message}", file=sys.stderr)
            if not args.continue_on_error:
                raise

    if not results:
        raise SystemExit("No style prompts were extracted.")

    write_outputs(results, args.output_xlsx, args.audit_jsonl)
    print(f"\nSaved: {args.output_xlsx}", file=sys.stderr)
    print(f"Saved: {args.audit_jsonl}", file=sys.stderr)

    if errors:
        print("\nCompleted with errors:", file=sys.stderr)
        for err in errors:
            print(f"- {err}", file=sys.stderr)


if __name__ == "__main__":
    main()

"""
python extract_professor_styles_ollama.py \
  --input_dir prof_samples \
  --output_xlsx Professor_Styles_extracted.xlsx \
  --audit_jsonl Professor_Styles_extracted_audit.jsonl \
  --model qwen3.5:9b \
  --ollama_num_ctx 16384 \
  --num_predict 2600

"""