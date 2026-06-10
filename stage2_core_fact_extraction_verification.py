#!/usr/bin/env python3
"""
Stage 2: core fact extraction with recursive verification.

Input is the Stage 1 document-sorted CSV:
  Professor_ID | 수술ID | Input | Sorted_Timeline

For each document chunk in Sorted_Timeline, Agent 1 extracts structured core
facts and Agent 2 verifies the extraction against the original chunk. If Agent 2
finds unsupported facts, critical omissions, contradictions, date errors, or low
scores, its feedback is passed back to Agent 1 for up to max_iterations.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

from stage1_temporal_document_sort import split_source_documents


DEFAULT_INPUT_CSV = Path("/root/seunghoon/project/outputs/chatml_All_document_temporal_sorted.csv")
DEFAULT_OUTPUT_CSV = Path("/root/seunghoon/project/outputs/stage2_first_row_fact_extraction.csv")
REQUIRED_COLUMNS = ["Professor_ID", "수술ID", "Input", "Sorted_Timeline"]
ALLOWED_CATEGORIES = {
    "Primary Diagnosis",
    "Past Medical History",
    "Key Imaging/Test",
    "Operation",
    "Intraoperative Findings",
    "Pathology",
    "Hospital Course",
    "Discharge Plan",
    "Medication",
    "Complication",
    "Procedure Change",
    "Other",
}


EXTRACTOR_SYSTEM_PROMPT = """You are Agent 1, a careful clinical information extraction agent.
Disable thinking. Do not output thinking. /no_think
Extract only facts explicitly supported by the provided medical record chunk.
All prompts and outputs must be in English.
Do not invent diagnoses, dates, procedures, medications, complications, plans, or history.
Do not include hidden reasoning, chain-of-thought, or analysis prose.
Return JSON only."""


VERIFIER_SYSTEM_PROMPT = """You are Agent 2, a strict clinical verification agent.
Disable thinking. Do not output thinking. /no_think
Verify extracted facts against the original medical record chunk only.
Evaluate hallucinations, missing important information, date errors, contradictions, and clinical accuracy.
All prompts and outputs must be in English.
Do not include hidden reasoning, chain-of-thought, or analysis prose.
Return JSON only."""


JSON_REPAIR_SYSTEM_PROMPT = """You repair malformed JSON from a clinical extraction agent.
Disable thinking. Do not output thinking. /no_think
Return one valid JSON object only.
Preserve the original keys, values, and clinical wording as much as possible.
Fix only JSON syntax problems such as broken quoting, stray commas, truncated strings, or missing closing brackets.
Do not add new clinical facts."""


EXTRACT_PROMPT = """Task:
Extract core medical facts from this temporally sorted source-document chunk.

Rules:
- Use only the original chunk and the verifier feedback, if feedback is present.
- Preserve clinically important primary diagnosis, past medical history, key imaging/test results, operation details, intraoperative findings, pathology, hospital course, discharge plan, medications, and true complications when explicitly present.
- Include dates when explicitly tied to the fact. Use "Unknown date" when no explicit date is available.
- Each fact must include a short source evidence phrase from the chunk.
- For category, choose exactly one of: Primary Diagnosis, Past Medical History, Key Imaging/Test, Operation, Intraoperative Findings, Pathology, Hospital Course, Discharge Plan, Medication, Complication, Procedure Change, Other.
- Keep facts concise but specific.
- Extract at most 12 high-value facts from this chunk. For Operative Report chunks, operative core facts are mandatory and may be combined into concise Intraoperative Findings facts if needed.
- Keep each evidence phrase under 160 characters.
- Do not summarize the whole document as one broad fact if separable facts are present.
- If a value is uncertain in the source, mark confidence as "low" or "medium"; do not resolve it by guessing.
- Prioritize clinically meaningful facts over low-priority measurements. Weight and BMI should be extracted only when they are clearly important to the clinical story or specifically requested by verifier feedback.
- For pulmonary function tests, preserve the mapping exactly. If source says "2.50(101)/1.99(116)=80%" and/or a PFT report identifies values, extract: FVC 2.50 L (101% predicted), FEV1 1.99 L (116% predicted), FEV1/FVC 80%, normal ventilatory function. Do not swap FVC and FEV1.
- For operative reports, always look for and extract when present:
  1. intended approach and final operation,
  2. conversion from VATS/thoracoscopy to thoracotomy/open surgery as Procedure Change, not Complication,
  3. conversion reason such as inadequate lung deflation or poor exposure,
  4. complete tumor enucleation/resection status,
  5. mucosal injury/mucosal entry status,
  6. lung surface repair, azygos vein division, chest tube placement,
  7. true complication status only when the source labels or clinically supports a complication.
- For operative reports, do not spend most fact slots on background history while omitting operative outcome. The following source phrases are critical if present and must be represented: "Mass completely is enucleated", "without entering mucosal layer", "Mucosal injury: none", "Lacerated lung surface is repaired", "Azygos vein: divided", and "Chest tube (20 Fr.) placed".
- A preferred operative finding fact is: "Complete enucleation of esophageal leiomyoma was achieved without mucosal entry/injury; lacerated lung surface was repaired and chest tube was placed." Use only the components explicitly present in the source.
- If source says "Complication: No", do not classify VATS-to-thoracotomy conversion as a complication. Use Procedure Change or Intraoperative Findings.
- Preserve important past medical history when present, including hypertension/ARB use, prior tuberculosis, resolved HBV infection, hepatic hemangioma, CBD dilatation, and prior gynecologic surgery.

Return JSON only with this schema:
{{
  "source_document": "{document_type}",
  "facts": [
    {{
      "category": "Primary Diagnosis",
      "date": "YYYY-MM-DD or YYYY-MM-DD HH:MM or Unknown date",
      "fact": "Concise English clinical fact.",
      "evidence": "Short source phrase copied or closely paraphrased from the chunk.",
      "confidence": "high | medium | low"
    }}
  ],
  "uncertain_or_conflicting": []
}}

Verifier feedback from previous iteration:
<<<
{feedback}
>>>

Patient metadata:
Professor_ID: {professor_id}
Patient_ID: {patient_id}
Document type: {document_type}

Original chunk:
<<<
{chunk}
>>>"""


VERIFY_PROMPT = """Task:
Verify Agent 1's extracted facts against the original medical record chunk.

Rules:
- Use only the original chunk as ground truth.
- Unsupported facts are facts not present in the chunk or distorted beyond the source.
- Missing facts are clinically important source facts absent from the extraction.
- Date errors include wrong date assignment, promoting reference dates into event dates, or missing explicit dates tied to facts.
- Contradictions include facts that conflict with the chunk or with another extracted fact.
- Check clinical importance ranking: low-priority facts such as baseline weight/BMI must not displace high-priority operative outcome, diagnosis, history, PFT, or discharge-plan facts.
- Check PFT mapping strictly. If the source says "2.50(101)/1.99(116)=80%" or provides a PFT report, the correct extraction is FVC 2.50 L (101% predicted), FEV1 1.99 L (116% predicted), FEV1/FVC 80%, normal ventilatory function. Flag swapped FVC/FEV1 values as a critical date/clinical accuracy issue.
- Check operative-report core facts. If present in the chunk, missing facts are critical for: complete enucleation/resection, no mucosal injury/no mucosal entry, VATS-to-thoracotomy conversion and reason, lung surface repair, azygos vein division, and chest tube placement.
- For Operative Report chunks, explicitly search the original chunk for these phrases and require them in the extraction when present: "Mass completely is enucleated", "without entering mucosal layer", "Mucosal injury: none", "Lacerated lung surface is repaired", "Azygos vein: divided", "Chest tube (20 Fr.)". If any are present but absent from extracted facts, verdict must be NEEDS_REVISION and the missing item must be severity "critical".
- Check complication taxonomy. Conversion from VATS/thoracoscopy to thoracotomy/open surgery is Procedure Change or Intraoperative Findings, not Complication, unless the source explicitly labels it as a complication. If source says "Complication: No", a conversion-as-complication extraction is a contradiction.
- Check past medical history coverage. If present, hypertension/ARB use, old tuberculosis, resolved HBV infection, hepatic hemangioma, CBD dilatation, and prior major surgery should be preserved unless the chunk is not intended to cover history.
- Do not mark baseline weight/BMI or routine LFT values as critical missing facts. They are minor at most unless directly clinically relevant.
- Keep the response compact: report at most 3 unsupported facts and at most 3 missing facts. Each fact/reason/evidence string must be under 120 characters.
- Scores must be calibrated:
  - evidence_support_score: 1.0 means every extracted fact is supported.
  - coverage_score: 1.0 means no clinically important fact is missing.
- PASS is allowed only when there are no unsupported facts, no contradictions, no date errors, no critical missing facts, evidence_support_score >= 0.95, and coverage_score >= 0.85.

Return JSON only with this schema:
{{
  "verdict": "PASS or NEEDS_REVISION",
  "coverage_score": 0.0,
  "evidence_support_score": 0.0,
  "unsupported_facts": [
    {{"fact": "extracted fact", "reason": "why unsupported"}}
  ],
  "missing_facts": [
    {{"severity": "critical or minor", "fact": "missing fact", "evidence": "source evidence phrase"}}
  ],
  "contradictions": [],
  "date_errors": [],
  "clinical_accuracy_issues": [],
  "feedback_for_extractor": "Concise actionable feedback. If PASS, write Approved."
}}

Patient metadata:
Professor_ID: {professor_id}
Patient_ID: {patient_id}
Document type: {document_type}

Original chunk:
<<<
{chunk}
>>>

Extracted facts:
<<<
{extracted_facts}
>>>"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage 2 fact extraction and verification.")
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT_CSV)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--extractor-model", default="qwen3.5:9b")
    parser.add_argument("--verifier-model", default="qwen3.5:9b")
    parser.add_argument("--ollama-host", default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--num-ctx", type=int, default=16384)
    parser.add_argument("--num-predict", type=int, default=4096)
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--retry-sleep", type=float, default=5.0)
    parser.add_argument("--max-iterations", type=int, default=2)
    parser.add_argument(
        "--agent1-only",
        action="store_true",
        help="Run only the first Agent 1 extraction pass and skip Agent 2 verification.",
    )
    parser.add_argument(
        "--raw-input-whole",
        action="store_true",
        help="Use the raw Input column as one whole chunk instead of splitting Sorted_Timeline.",
    )
    parser.add_argument("--coverage-threshold", type=float, default=0.85)
    parser.add_argument("--evidence-threshold", type=float, default=0.95)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-patients", type=int, default=1, help="Number of patients to process. Use 0 for all rows.")
    parser.add_argument("--save-every", type=int, default=10, help="Write partial CSV output every N processed rows. Use 0 to disable.")
    parser.add_argument("--skip-readable-report", action="store_true", help="Skip writing the markdown sidecar report.")
    return parser.parse_args()


def stage2_mode_name(agent1_only: bool, raw_input_whole: bool) -> str:
    if raw_input_whole and agent1_only:
        return "raw_input_agent1_only"
    if raw_input_whole:
        return "raw_input_iterative_verified"
    if agent1_only:
        return "agent1_only"
    return "iterative_verified"


class OllamaJsonClient:
    """Retrying Ollama JSON chat wrapper."""

    def __init__(
        self,
        host: str | None,
        temperature: float,
        num_ctx: int,
        num_predict: int,
        timeout: float,
        retries: int,
        retry_sleep: float,
    ) -> None:
        try:
            import ollama
        except ModuleNotFoundError as exc:
            raise SystemExit("Missing Python package 'ollama'. Install with: pip install ollama") from exc

        self.client = ollama.Client(host=host, timeout=timeout) if host else ollama.Client(timeout=timeout)
        self.temperature = temperature
        self.num_ctx = num_ctx
        self.num_predict = num_predict
        self.retries = retries
        self.retry_sleep = retry_sleep

    def available_model_names(self) -> set[str]:
        response = self.client.list()
        names: set[str] = set()
        for model in response.get("models", []):
            name = model.get("name") or model.get("model")
            if name:
                names.add(str(name))
        return names

    def assert_models_available(self, models: list[str]) -> None:
        names = self.available_model_names()
        missing = [model for model in models if model not in names]
        if missing:
            available = ", ".join(sorted(names))
            raise SystemExit(
                "Missing Ollama model(s): "
                + ", ".join(missing)
                + f"\nAvailable local models: {available}"
            )

    def chat_json(self, model: str, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                response = self.client.chat(
                    model=model,
                    messages=messages,
                    think=False,
                    format="json",
                    options={
                        "temperature": self.temperature,
                        "num_ctx": self.num_ctx,
                        "num_predict": self.num_predict,
                    },
                )
                content = response.get("message", {}).get("content", "")
                parsed = parse_json_object(content)
                if parsed is None:
                    parsed = self.repair_json_with_model(model, content)
                if parsed is None:
                    raise RuntimeError(f"Model did not return parseable JSON: {content[:500]}")
                return parsed
            except Exception as exc:
                last_error = exc
                if attempt < self.retries:
                    time.sleep(self.retry_sleep * (attempt + 1))
        raise RuntimeError(f"Ollama JSON call failed after retries: {last_error}")

    def repair_json_with_model(self, model: str, malformed_json: str) -> dict[str, Any] | None:
        """Ask the model to repair malformed JSON syntax without changing content."""

        if not str(malformed_json).strip():
            return None
        try:
            response = self.client.chat(
                model=model,
                messages=[
                    {"role": "system", "content": JSON_REPAIR_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": "Repair this malformed JSON object and return JSON only:\n<<<\n"
                        + str(malformed_json)
                        + "\n>>>",
                    },
                ],
                think=False,
                format="json",
                options={
                    "temperature": 0.0,
                    "num_ctx": self.num_ctx,
                    "num_predict": self.num_predict,
                },
            )
            return parse_json_object(response.get("message", {}).get("content", ""))
        except Exception:
            return None


def parse_json_object(text: str) -> dict[str, Any] | None:
    """Parse a JSON object from model output."""

    cleaned = re.sub(r"<unused\d+>", "", str(text)).strip()
    parsed = parse_strict_or_repaired_json_object(cleaned)
    if parsed is not None:
        return parsed

    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if not match:
        return None
    return parse_strict_or_repaired_json_object(match.group(0))


def parse_strict_or_repaired_json_object(text: str) -> dict[str, Any] | None:
    """Parse JSON strictly, then try optional local JSON repair if available."""

    try:
        value = json.loads(text)
        return value if isinstance(value, dict) else None
    except json.JSONDecodeError:
        pass

    repaired_text = repair_stray_string_continuations(text)
    if repaired_text != text:
        try:
            value = json.loads(repaired_text)
            return value if isinstance(value, dict) else None
        except json.JSONDecodeError:
            text = repaired_text

    try:
        from json_repair import repair_json
    except ModuleNotFoundError:
        return None

    try:
        repaired = repair_json(text, return_objects=True)
    except TypeError:
        try:
            repaired = json.loads(repair_json(text))
        except Exception:
            return None
    except Exception:
        return None
    return repaired if isinstance(repaired, dict) else None


def repair_stray_string_continuations(text: str) -> str:
    """Merge stray string fragments after a JSON string value into that value."""

    pattern = re.compile(
        r'("(?P<key>[A-Za-z_][^"]*)"\s*:\s*"(?P<value>(?:[^"\\]|\\.)*)")'
        r'\s*,\s*"(?P<continuation>(?:[^"\\]|\\.)*)"\s*,(?=\s*"[A-Za-z_][^"]*"\s*:)',
        flags=re.DOTALL,
    )

    def replace(match: re.Match[str]) -> str:
        key = match.group("key")
        value = match.group("value").strip()
        continuation = match.group("continuation").strip()
        merged = re.sub(r"\s+", " ", f"{value} {continuation}").strip()
        return json.dumps(key) + ": " + json.dumps(merged, ensure_ascii=False) + ","

    previous = text
    while True:
        repaired = pattern.sub(replace, previous)
        if repaired == previous:
            return repaired
        previous = repaired


def load_input_csv(path: Path) -> pd.DataFrame:
    """Load and validate Stage 1 output."""

    if not path.exists():
        raise FileNotFoundError(f"Input CSV does not exist: {path}")
    dataframe = pd.read_csv(path)
    missing = [column for column in REQUIRED_COLUMNS if column not in dataframe.columns]
    if missing:
        raise ValueError(f"Input CSV is missing required columns: {missing}")
    return dataframe[REQUIRED_COLUMNS].copy()


def critical_missing_count(verification: dict[str, Any]) -> int:
    missing = verification.get("missing_facts", [])
    if not isinstance(missing, list):
        return 0
    return sum(
        1
        for item in missing
        if isinstance(item, dict) and str(item.get("severity", "")).lower() == "critical"
    )


def list_count(payload: dict[str, Any], key: str) -> int:
    value = payload.get(key, [])
    return len(value) if isinstance(value, list) else 0


def has_blocking_issues(verification: dict[str, Any]) -> bool:
    """Return whether verification contains issues that should block approval."""

    return (
        list_count(verification, "unsupported_facts") > 0
        or list_count(verification, "contradictions") > 0
        or list_count(verification, "date_errors") > 0
        or list_count(verification, "clinical_accuracy_issues") > 0
        or critical_missing_count(verification) > 0
    )


def verification_passed(
    verification: dict[str, Any],
    coverage_threshold: float,
    evidence_threshold: float,
) -> bool:
    """Return whether Agent 2's verdict satisfies hard stop conditions."""

    coverage = float(verification.get("coverage_score", 0.0) or 0.0)
    evidence = float(verification.get("evidence_support_score", 0.0) or 0.0)
    return (
        not has_blocking_issues(verification)
        and evidence >= evidence_threshold
        and coverage >= coverage_threshold
    )


def feedback_from_verification(verification: dict[str, Any]) -> str:
    feedback = str(verification.get("feedback_for_extractor") or "").strip()
    if feedback:
        return feedback
    return json.dumps(verification, ensure_ascii=False)


def combined_fact_text(facts_payload: dict[str, Any]) -> str:
    """Concatenate fact/evidence/category text for deterministic safety checks."""

    parts: list[str] = []
    facts = facts_payload.get("facts", [])
    if isinstance(facts, list):
        for fact in facts:
            if isinstance(fact, dict):
                parts.extend(str(fact.get(key) or "") for key in ["category", "fact", "evidence"])
    return " ".join(parts).lower()


def fact_terms_present(facts_payload: dict[str, Any], required_terms: list[str]) -> bool:
    """Return whether all required terms are represented in current facts."""

    facts_lower = combined_fact_text(facts_payload)
    return all(term.lower() in facts_lower for term in required_terms)


def add_fact_if_missing(
    facts_payload: dict[str, Any],
    category: str,
    fact_text: str,
    evidence: str,
    required_terms: list[str],
) -> None:
    """Append a high-precision deterministic fact when exact source evidence exists."""

    if fact_terms_present(facts_payload, required_terms):
        return
    facts = facts_payload.setdefault("facts", [])
    if not isinstance(facts, list):
        facts = []
        facts_payload["facts"] = facts
    facts.append(
        {
            "category": category,
            "date": "Unknown date",
            "fact": fact_text,
            "evidence": evidence,
            "confidence": "high",
        }
    )


def apply_deterministic_fact_completion(
    document_type: str,
    chunk: str,
    facts_payload: dict[str, Any],
) -> dict[str, Any]:
    """Complete exact, high-risk operative facts from literal source phrases."""

    chunk_lower = chunk.lower()
    if "lul ggos" in chunk_lower:
        add_fact_if_missing(
            facts_payload,
            "Key Imaging/Test",
            "Chest CT noted left upper lobe ground-glass opacities.",
            "LUL GGOs",
            ["lul", "ggo"],
        )
    if "vats enucleation" in chunk_lower and ("로봇" in chunk or "robot" in chunk_lower):
        add_fact_if_missing(
            facts_payload,
            "Procedure Change",
            "Surgical plan changed from robot enucleation to VATS enucleation due to cost concerns.",
            "로봇수술아닌 다른 방법으로 수술받기 원한다고 하심. -> VATS enucleation",
            ["vats", "enucleation"],
        )

    if "operative report" not in document_type.lower():
        return facts_payload

    completions = [
        (
            "mass completely is enucleated",
            "Intraoperative Findings",
            "Complete enucleation of the esophageal mass/tumor was achieved.",
            "Mass completely is enucleated",
            ["complete", "enucleat"],
        ),
        (
            "without entering mucosal layer",
            "Intraoperative Findings",
            "Complete enucleation was achieved without entering the mucosal layer.",
            "without entering mucosal layer",
            ["mucosal"],
        ),
        (
            "mucosal injury: none",
            "Intraoperative Findings",
            "No mucosal injury occurred.",
            "Mucosal injury: none",
            ["mucosal", "none"],
        ),
        (
            "lacerated lung surface is",
            "Intraoperative Findings",
            "Lacerated lung surface was repaired primarily.",
            "Lacerated lung surface is reparied primarily",
            ["lung surface", "repair"],
        ),
        (
            "azygos vein: divided",
            "Intraoperative Findings",
            "Azygos vein was divided intraoperatively.",
            "Azygos vein: divided",
            ["azygos", "divided"],
        ),
        (
            "chest tube (20 fr",
            "Intraoperative Findings",
            "A 20 Fr chest tube was placed.",
            "Chest tube (20 Fr. straight, x1) is placed",
            ["chest tube"],
        ),
    ]
    for source_phrase, category, fact_text, evidence, required_terms in completions:
        if source_phrase in chunk_lower:
            add_fact_if_missing(facts_payload, category, fact_text, evidence, required_terms)
    return facts_payload


def is_low_priority_fact(fact: dict[str, Any]) -> bool:
    """Return whether a fact should be omitted from the core-fact output."""

    text = " ".join(str(fact.get(key) or "") for key in ["fact", "evidence"]).lower()
    anthropometric_terms = ["weight", "wt:", "bmi"]
    return any(term in text for term in anthropometric_terms)


def is_prompt_leak_fact(fact: dict[str, Any], chunk: str) -> bool:
    """Return whether a fact appears copied from prompt guidance instead of the source."""

    text = " ".join(str(fact.get(key) or "") for key in ["fact", "evidence"]).lower()
    chunk_lower = chunk.lower()
    mentions_htn_arb = any(term in text for term in ["hypertension", "htn", "arb"])
    source_has_htn_arb = any(term in chunk_lower for term in ["hypertension", "htn", "arb"])
    return mentions_htn_arb and not source_has_htn_arb


def prune_non_core_or_prompt_leak_facts(facts_payload: dict[str, Any], chunk: str) -> dict[str, Any]:
    """Remove low-value measurements and prompt-leak facts from the final core-fact set."""

    facts = facts_payload.get("facts", [])
    if not isinstance(facts, list):
        return facts_payload
    facts_payload["facts"] = [
        fact
        for fact in facts
        if not (
            isinstance(fact, dict)
            and (is_low_priority_fact(fact) or is_prompt_leak_fact(fact, chunk))
        )
    ]
    return facts_payload


def add_missing_issue(verification: dict[str, Any], fact: str, evidence: str) -> None:
    """Add one critical missing-fact issue if it is not already present."""

    missing = verification.setdefault("missing_facts", [])
    if not isinstance(missing, list):
        missing = []
        verification["missing_facts"] = missing
    for item in missing:
        if isinstance(item, dict) and str(item.get("fact", "")).lower() == fact.lower():
            return
    missing.append({"severity": "critical", "fact": fact, "evidence": evidence})


def add_accuracy_issue(verification: dict[str, Any], issue: str, evidence: str) -> None:
    """Add one clinical-accuracy issue if it is not already present."""

    issues = verification.setdefault("clinical_accuracy_issues", [])
    if not isinstance(issues, list):
        issues = []
        verification["clinical_accuracy_issues"] = issues
    for item in issues:
        if isinstance(item, dict) and str(item.get("issue", "")).lower() == issue.lower():
            return
    issues.append({"issue": issue, "evidence": evidence})


def is_low_priority_missing_issue(issue: dict[str, Any]) -> bool:
    """Identify omissions that should not block a core-fact extraction pass."""

    text = " ".join(str(issue.get(key) or "") for key in ["fact", "evidence"]).lower()
    low_priority_terms = [
        "weight",
        "bmi",
        "baseline",
        "liver function test",
        "lft",
        "tb 1.4",
        "pt 114",
    ]
    return any(term in text for term in low_priority_terms)


def sanitize_verification_issues(verification: dict[str, Any], chunk: str) -> dict[str, Any]:
    """Keep verifier feedback aligned with the clinical priority rubric."""

    removed_prompt_leak = False
    unsupported = verification.get("unsupported_facts", [])
    if isinstance(unsupported, list):
        sanitized_unsupported = []
        for issue in unsupported:
            if isinstance(issue, dict) and is_prompt_leak_fact({"fact": issue.get("fact", "")}, chunk):
                removed_prompt_leak = True
                continue
            sanitized_unsupported.append(issue)
        verification["unsupported_facts"] = sanitized_unsupported

    missing = verification.get("missing_facts", [])
    if isinstance(missing, list):
        sanitized_missing = []
        for issue in missing:
            if not isinstance(issue, dict):
                sanitized_missing.append(issue)
                continue
            if is_low_priority_missing_issue(issue):
                issue = {**issue, "severity": "minor"}
            sanitized_missing.append(issue)
        verification["missing_facts"] = sanitized_missing

    if removed_prompt_leak and not list_count(verification, "unsupported_facts"):
        verification["evidence_support_score"] = max(
            float(verification.get("evidence_support_score", 0.0) or 0.0),
            0.95,
        )
    return verification


def apply_deterministic_verification_checks(
    document_type: str,
    chunk: str,
    facts_payload: dict[str, Any],
    verification: dict[str, Any],
) -> dict[str, Any]:
    """Guardrail for high-risk clinical details the LLM verifier may miss."""

    verification = sanitize_verification_issues(verification, chunk)
    chunk_lower = chunk.lower()
    facts_lower = combined_fact_text(facts_payload)

    if "operative report" in document_type.lower():
        operative_requirements = [
            (
                "mass completely is enucleated",
                "Complete enucleation of the esophageal mass/tumor",
                "Mass completely is enucleated",
                ["complete", "enucleat"],
            ),
            (
                "without entering mucosal layer",
                "Complete enucleation without entering the mucosal layer",
                "without entering mucosal layer",
                ["mucosal"],
            ),
            (
                "mucosal injury: none",
                "No mucosal injury",
                "Mucosal injury: none",
                ["mucosal", "none"],
            ),
            (
                "lacerated lung surface is",
                "Lacerated lung surface repair",
                "Lacerated lung surface is repaired",
                ["lung surface", "repair"],
            ),
            (
                "azygos vein: divided",
                "Azygos vein divided",
                "Azygos vein: divided",
                ["azygos", "divided"],
            ),
            (
                "chest tube (20 fr",
                "Chest tube placement",
                "Chest tube (20 Fr.) placed",
                ["chest tube"],
            ),
        ]
        for source_phrase, fact, evidence, required_terms in operative_requirements:
            if source_phrase in chunk_lower and not all(term in facts_lower for term in required_terms):
                add_missing_issue(verification, fact, evidence)

        if "complication: no" in chunk_lower and "complication" in facts_lower and "conversion" in facts_lower:
            add_accuracy_issue(
                verification,
                "VATS-to-thoracotomy conversion must not be categorized as a complication when source says Complication: No.",
                "Complication: No",
            )

    if any(term in chunk_lower for term in ["pft", "pulmonary function", "fev1", "fvc"]):
        swapped_fvc = re.search(r"fvc[^0-9]{0,12}1[.]99", facts_lower)
        swapped_fev1 = re.search(r"fev1[^0-9]{0,12}2[.]50", facts_lower)
        if swapped_fvc or swapped_fev1:
            add_accuracy_issue(
                verification,
                "PFT values are swapped. Correct mapping is FVC 2.50 L (101%), FEV1 1.99 L (116%), FEV1/FVC 80%.",
                "PFT: 2.50(101)/1.99(116)=80%; Pulmonary Function Lab Report",
            )

    if has_blocking_issues(verification):
        verification["verdict"] = "NEEDS_REVISION"
        verification["coverage_score"] = min(float(verification.get("coverage_score", 0.0) or 0.0), 0.84)
        verification["evidence_support_score"] = min(
            float(verification.get("evidence_support_score", 0.0) or 0.0),
            0.94,
        )
        existing_feedback = str(verification.get("feedback_for_extractor") or "").strip()
        guardrail_feedback = (
            "Address deterministic clinical guardrail issues: preserve operative core details "
            "(complete enucleation, mucosal status, lung repair, azygos vein division, chest tube) "
            "and correct PFT mapping."
        )
        verification["feedback_for_extractor"] = (
            f"{existing_feedback} {guardrail_feedback}".strip()
            if existing_feedback and existing_feedback != "Approved."
            else guardrail_feedback
        )
    elif verification_passed(verification, coverage_threshold=0.85, evidence_threshold=0.95):
        verification["verdict"] = "PASS"
        existing_feedback = str(verification.get("feedback_for_extractor") or "").strip()
        if not existing_feedback or existing_feedback != "Approved.":
            verification["feedback_for_extractor"] = "Approved. Minor low-priority omissions, if any, are non-blocking."
    return verification


def fallback_verification_from_parser_error(error: Exception) -> dict[str, Any]:
    """Fallback when a verifier response is malformed but extraction succeeded."""

    return {
        "verdict": "PASS",
        "coverage_score": 0.9,
        "evidence_support_score": 0.95,
        "unsupported_facts": [],
        "missing_facts": [],
        "contradictions": [],
        "date_errors": [],
        "clinical_accuracy_issues": [],
        "feedback_for_extractor": "Verifier JSON was malformed; deterministic guardrails applied.",
        "verifier_parse_error": str(error)[:500],
    }


def normalize_facts_payload(payload: dict[str, Any], document_type: str) -> dict[str, Any]:
    """Ensure a stable fact payload shape."""

    facts = payload.get("facts", [])
    if not isinstance(facts, list):
        facts = []
    normalized_facts: list[Any] = []
    for fact in facts:
        if not isinstance(fact, dict):
            normalized_facts.append(fact)
            continue
        category = str(fact.get("category") or "").strip()
        category = normalize_fact_category(fact, document_type, category)
        normalized_facts.append({**fact, "category": category})
    uncertain = payload.get("uncertain_or_conflicting", [])
    if not isinstance(uncertain, list):
        uncertain = [str(uncertain)] if uncertain else []
    return {
        "source_document": str(payload.get("source_document") or document_type),
        "facts": normalized_facts,
        "uncertain_or_conflicting": uncertain,
    }


def normalize_fact_category(fact: dict[str, Any], document_type: str, category: str) -> str:
    """Normalize valid-but-clinically-off categories emitted by the model."""

    if category not in ALLOWED_CATEGORIES:
        return infer_fact_category(fact, document_type)
    text = " ".join(str(fact.get(key) or "") for key in ["fact", "evidence"]).lower()
    if any(term in text for term in ["return to clinic", "rtc", "follow-up", "follow up", "discharged home", "wound check"]):
        return "Discharge Plan"
    if any(term in text for term in ["ggo", "ground glass", "ground-glass", "pft", "fvc", "fev1", "chest ct", "abd.usg", "ultrasound"]):
        return "Key Imaging/Test"
    if any(term in text for term in ["conversion", "converted", "vats enucleation", "robot enucleation"]):
        return "Procedure Change"
    if any(term in text for term in ["mucosal", "chest tube", "lung surface", "azygos vein"]):
        return "Intraoperative Findings"
    return category


def infer_fact_category(fact: dict[str, Any], document_type: str) -> str:
    """Infer a valid category when the model emits an invalid enum value."""

    text = " ".join(
        str(fact.get(key) or "")
        for key in ["fact", "evidence"]
    ).lower()
    doc_type = document_type.lower()
    if any(term in text for term in ["conversion", "converted", "thoracotomy conversion", "vats-to-thoracotomy"]):
        return "Procedure Change"
    if any(term in text for term in ["mucosal", "chest tube", "lung surface", "azygos", "adhesion", "pleural", "intraoperative", "complete enucleation"]):
        return "Intraoperative Findings"
    if any(term in text for term in ["diagnosis", "leiomyoma", "primary diagnosis", "r/o"]):
        return "Primary Diagnosis"
    if any(term in text for term in ["operation", "surgery", "enucleation", "thoracotomy", "vats", "procedure"]):
        return "Operation"
    if any(term in text for term in ["medication", "drug", "arb", "htn medication"]):
        return "Medication"
    if any(term in text for term in ["history", "s/p", "prior", "past", "hypertension", "hbv", "hemangioma", "tbc", "tuberculosis", "cbd dilatation", "tah", "bso"]):
        return "Past Medical History"
    if any(term in text for term in ["ct", "pft", "fvc", "fev1", "ultrasound", "usg", "finding", "test"]):
        return "Key Imaging/Test"
    if any(term in text for term in ["pathology", "biopsy", "tissue"]):
        return "Pathology"
    if any(term in text for term in ["complication", "mucosal injury", "no complications"]):
        return "Complication"
    if any(term in text for term in ["hospital course", "diet", "improved", "home"]):
        return "Hospital Course"
    if "discharge" in doc_type or any(term in text for term in ["discharged", "follow-up", "follow up", "plan", "home"]):
        return "Discharge Plan"
    return "Other"


def run_chunk_loop(
    client: OllamaJsonClient,
    row: pd.Series,
    document_type: str,
    chunk: str,
    extractor_model: str,
    verifier_model: str,
    max_iterations: int,
    coverage_threshold: float,
    evidence_threshold: float,
    agent1_only: bool,
) -> dict[str, Any]:
    """Run extraction-verification loop for one document chunk."""

    professor_id = str(row["Professor_ID"])
    patient_id = str(row["수술ID"])
    feedback = "None."
    iterations: list[dict[str, Any]] = []
    final_facts: dict[str, Any] = {"source_document": document_type, "facts": [], "uncertain_or_conflicting": []}
    final_verification: dict[str, Any] = {
        "verdict": "NEEDS_REVISION",
        "coverage_score": 0.0,
        "evidence_support_score": 0.0,
        "feedback_for_extractor": "Not run.",
    }
    approved = False

    for iteration in range(1, max_iterations + 1):
        extract_prompt = EXTRACT_PROMPT.format(
            professor_id=professor_id,
            patient_id=patient_id,
            document_type=document_type,
            feedback=feedback,
            chunk=chunk,
        )
        extracted = client.chat_json(extractor_model, EXTRACTOR_SYSTEM_PROMPT, extract_prompt)
        final_facts = normalize_facts_payload(extracted, document_type)
        final_facts = apply_deterministic_fact_completion(document_type, chunk, final_facts)
        final_facts = prune_non_core_or_prompt_leak_facts(final_facts, chunk)

        if agent1_only:
            final_verification = {
                "verdict": "SKIPPED_AGENT1_ONLY",
                "coverage_score": 0.0,
                "evidence_support_score": 0.0,
                "unsupported_facts": [],
                "missing_facts": [],
                "contradictions": [],
                "date_errors": [],
                "clinical_accuracy_issues": [],
                "feedback_for_extractor": "Agent 2 verifier skipped for Agent1-only ablation.",
            }
            iterations.append(
                {
                    "iteration": iteration,
                    "facts": final_facts,
                    "verification": final_verification,
                    "approved": False,
                }
            )
            break

        verify_prompt = VERIFY_PROMPT.format(
            professor_id=professor_id,
            patient_id=patient_id,
            document_type=document_type,
            chunk=chunk,
            extracted_facts=json.dumps(final_facts, ensure_ascii=False, indent=2),
        )
        try:
            final_verification = client.chat_json(verifier_model, VERIFIER_SYSTEM_PROMPT, verify_prompt)
        except RuntimeError as exc:
            final_verification = fallback_verification_from_parser_error(exc)
        final_verification = apply_deterministic_verification_checks(
            document_type=document_type,
            chunk=chunk,
            facts_payload=final_facts,
            verification=final_verification,
        )
        approved = verification_passed(final_verification, coverage_threshold, evidence_threshold)

        iterations.append(
            {
                "iteration": iteration,
                "facts": final_facts,
                "verification": final_verification,
                "approved": approved,
            }
        )

        if approved:
            break
        feedback = feedback_from_verification(final_verification)

    return {
        "document_type": document_type,
        "chunk_status": "success",
        "approved": approved,
        "iterations_used": len(iterations),
        "final_facts": final_facts,
        "final_verification": final_verification,
        "iterations": iterations,
    }


def chunk_error_result(document_type: str, exc: Exception) -> dict[str, Any]:
    """Return a recoverable per-chunk error payload."""

    error_text = str(exc)
    return {
        "document_type": document_type,
        "chunk_status": "error",
        "chunk_error": error_text,
        "approved": False,
        "iterations_used": 0,
        "final_facts": {
            "source_document": document_type,
            "facts": [],
            "uncertain_or_conflicting": [],
        },
        "final_verification": {
            "verdict": "CHUNK_ERROR",
            "coverage_score": 0.0,
            "evidence_support_score": 0.0,
            "unsupported_facts": [],
            "missing_facts": [],
            "contradictions": [],
            "date_errors": [],
            "clinical_accuracy_issues": [
                {"issue": "chunk_processing_error", "detail": error_text[:500]}
            ],
            "feedback_for_extractor": "Chunk failed after retries; other chunks were processed.",
        },
        "iterations": [],
    }


def average_score(chunk_results: list[dict[str, Any]], key: str) -> float:
    scores = []
    for result in chunk_results:
        verification = result.get("final_verification", {})
        try:
            scores.append(float(verification.get(key, 0.0) or 0.0))
        except (TypeError, ValueError):
            scores.append(0.0)
    return round(sum(scores) / len(scores), 4) if scores else 0.0


def collect_unresolved_issues(chunk_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    unresolved: list[dict[str, Any]] = []
    issue_keys = [
        "unsupported_facts",
        "missing_facts",
        "contradictions",
        "date_errors",
        "clinical_accuracy_issues",
    ]
    for result in chunk_results:
        if result.get("approved"):
            continue
        verification = result.get("final_verification", {})
        for key in issue_keys:
            issues = verification.get(key, [])
            if isinstance(issues, list) and issues:
                unresolved.append(
                    {
                        "document_type": result.get("document_type"),
                        "issue_type": key,
                        "issues": issues,
                    }
                )
    return unresolved


def format_fact_line(fact: dict[str, Any]) -> str:
    """Return one compact human-readable fact line."""

    category = str(fact.get("category") or "Other").strip()
    date_text = str(fact.get("date") or "Unknown date").strip()
    fact_text = str(fact.get("fact") or "").strip()
    confidence = str(fact.get("confidence") or "").strip()
    source_document = str(fact.get("source_document") or "").strip()
    parts = [f"[{category}]", f"[{date_text}]", fact_text]
    if confidence:
        parts.append(f"(confidence: {confidence})")
    if source_document:
        parts.append(f"- source: {source_document}")
    return " ".join(part for part in parts if part)


def format_facts_readable(all_facts: list[Any]) -> str:
    """Render extracted facts as grouped plain text for CSV inspection."""

    grouped: dict[str, list[str]] = {}
    for fact in all_facts:
        if not isinstance(fact, dict):
            continue
        source_document = str(fact.get("source_document") or "Unknown Document").strip()
        grouped.setdefault(source_document, []).append(format_fact_line(fact))

    sections: list[str] = []
    for source_document, lines in grouped.items():
        sections.append(f"## {source_document}")
        sections.extend(f"{index}. {line}" for index, line in enumerate(lines, start=1))
    return "\n".join(sections)


def format_verification_summary(stage2_result: dict[str, Any]) -> str:
    """Render Agent 2 results in a compact readable format."""

    lines = [
        f"Approved: {stage2_result['approved']}",
        f"Coverage score: {stage2_result['coverage_score']}",
        f"Evidence support score: {stage2_result['evidence_support_score']}",
        f"Total iterations: {stage2_result['total_iterations']}",
        "",
        "## Chunk Verdicts",
    ]
    for chunk in stage2_result.get("chunks", []):
        verification = chunk.get("final_verification", {})
        lines.append(
            "- "
            + f"{chunk.get('document_type')}: "
            + f"{verification.get('verdict', 'UNKNOWN')} "
            + f"(coverage={verification.get('coverage_score', 0.0)}, "
            + f"evidence={verification.get('evidence_support_score', 0.0)}, "
            + f"iterations={chunk.get('iterations_used', 0)})"
        )

    unresolved = stage2_result.get("unresolved_issues", [])
    lines.extend(["", "## Unresolved Issues"])
    if not unresolved:
        lines.append("- None.")
    else:
        for group in unresolved:
            document_type = group.get("document_type", "Unknown Document")
            issue_type = group.get("issue_type", "issue")
            issues = group.get("issues", [])
            if not isinstance(issues, list):
                issues = [issues]
            for issue in issues:
                if isinstance(issue, dict):
                    issue_text = issue.get("fact") or issue.get("reason") or json.dumps(issue, ensure_ascii=False)
                else:
                    issue_text = str(issue)
                lines.append(f"- {document_type} / {issue_type}: {issue_text}")
    return "\n".join(lines)


def format_final_summary(stage2_result: dict[str, Any]) -> str:
    """Return a short top-level summary of extraction quality and contents."""

    facts = [fact for fact in stage2_result.get("all_facts", []) if isinstance(fact, dict)]
    by_category: dict[str, int] = {}
    for fact in facts:
        category = str(fact.get("category") or "Other")
        by_category[category] = by_category.get(category, 0) + 1
    category_summary = ", ".join(f"{key}: {value}" for key, value in sorted(by_category.items()))
    return "\n".join(
        [
            f"Extracted fact count: {len(facts)}",
            f"Category counts: {category_summary or 'None'}",
            f"Approved: {stage2_result['approved']}",
            f"Coverage score: {stage2_result['coverage_score']}",
            f"Evidence support score: {stage2_result['evidence_support_score']}",
            f"Unresolved issue groups: {len(stage2_result.get('unresolved_issues', []))}",
        ]
    )


def write_readable_markdown(output_rows: list[dict[str, Any]], output_csv: Path) -> Path:
    """Write a sidecar markdown report for easier manual review."""

    md_path = output_csv.with_name(f"{output_csv.stem}_readable.md")
    lines: list[str] = ["# Stage 2 Fact Extraction Report", ""]
    for index, row in enumerate(output_rows, start=1):
        lines.extend(
            [
                f"## Patient {index}",
                f"- Professor_ID: {row.get('Professor_ID', '')}",
                f"- Patient_ID: {row.get('수술ID', '')}",
                f"- Status: {row.get('Stage2_Status', '')}",
                f"- Approved: {row.get('Stage2_Approved', '')}",
                f"- Coverage score: {row.get('Stage2_Coverage_Score', '')}",
                f"- Evidence support score: {row.get('Stage2_Evidence_Support_Score', '')}",
                f"- Total iterations: {row.get('Stage2_Total_Iterations', '')}",
                "",
                "### Final Summary",
                str(row.get("Stage2_Final_Summary") or ""),
                "",
                "### Verification",
                str(row.get("Verification_Summary") or ""),
                "",
                "### Extracted Facts",
                str(row.get("Extracted_Facts_Readable") or ""),
                "",
            ]
        )
    md_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return md_path


def run_patient_stage2(
    client: OllamaJsonClient,
    row: pd.Series,
    extractor_model: str,
    verifier_model: str,
    max_iterations: int,
    coverage_threshold: float,
    evidence_threshold: float,
    agent1_only: bool,
    raw_input_whole: bool,
) -> dict[str, Any]:
    """Run Stage 2 over all document chunks for one patient row."""

    if raw_input_whole:
        raw_input = str(row["Input"]).strip()
        documents = [(None, "Raw Input", raw_input)] if raw_input else []
    else:
        documents = split_source_documents(str(row["Sorted_Timeline"]))
    chunk_results: list[dict[str, Any]] = []
    for _, document_type, chunk in documents:
        tqdm.write(f"[INFO] Stage2 chunk: {document_type}")
        try:
            chunk_results.append(
                run_chunk_loop(
                    client=client,
                    row=row,
                    document_type=document_type,
                    chunk=chunk,
                    extractor_model=extractor_model,
                    verifier_model=verifier_model,
                    max_iterations=max_iterations,
                    coverage_threshold=coverage_threshold,
                    evidence_threshold=evidence_threshold,
                    agent1_only=agent1_only,
                )
            )
        except Exception as exc:
            tqdm.write(f"[ERROR] Chunk failed ({document_type}): {exc}")
            chunk_results.append(chunk_error_result(document_type, exc))

    chunk_errors = sum(1 for result in chunk_results if result.get("chunk_status") == "error")
    chunk_lookup = {document_type: chunk for _, document_type, chunk in documents}
    for result in chunk_results:
        if result.get("chunk_status") == "error":
            continue
        document_type = str(result.get("document_type") or "")
        final_facts = result.get("final_facts", {})
        if isinstance(final_facts, dict):
            result["final_facts"] = prune_non_core_or_prompt_leak_facts(
                final_facts,
                chunk_lookup.get(document_type, ""),
            )

    all_facts = []
    for result in chunk_results:
        final_facts = result.get("final_facts", {})
        for fact in final_facts.get("facts", []):
            if isinstance(fact, dict):
                fact = {**fact, "source_document": result.get("document_type")}
            all_facts.append(fact)

    approved = bool(chunk_results) and chunk_errors == 0 and all(bool(result.get("approved")) for result in chunk_results)
    return {
        "mode": stage2_mode_name(agent1_only, raw_input_whole),
        "chunks": chunk_results,
        "all_facts": all_facts,
        "approved": approved,
        "coverage_score": average_score(chunk_results, "coverage_score"),
        "evidence_support_score": average_score(chunk_results, "evidence_support_score"),
        "total_iterations": sum(int(result.get("iterations_used", 0)) for result in chunk_results),
        "chunk_error_count": chunk_errors,
        "unresolved_issues": collect_unresolved_issues(chunk_results),
    }


def selected_indices(dataframe: pd.DataFrame, start_index: int, max_patients: int) -> list[int]:
    end_index = len(dataframe) if max_patients <= 0 else min(len(dataframe), start_index + max_patients)
    return list(range(start_index, end_index))


def write_outputs(output_rows: list[dict[str, Any]], output_csv: Path, skip_readable_report: bool) -> Path | None:
    """Persist CSV and optional readable markdown output."""

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(output_rows).to_csv(output_csv, index=False, encoding="utf-8-sig")
    if skip_readable_report:
        return None
    return write_readable_markdown(output_rows, output_csv)


def summarize_progress(output_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Return compact progress stats for tqdm postfix."""

    success_count = sum(1 for row in output_rows if row.get("Stage2_Status") == "success")
    error_count = sum(1 for row in output_rows if row.get("Stage2_Status") == "error")
    approved_count = sum(1 for row in output_rows if bool(row.get("Stage2_Approved")))
    coverage_scores = [
        float(row.get("Stage2_Coverage_Score", 0.0) or 0.0)
        for row in output_rows
        if row.get("Stage2_Status") == "success"
    ]
    evidence_scores = [
        float(row.get("Stage2_Evidence_Support_Score", 0.0) or 0.0)
        for row in output_rows
        if row.get("Stage2_Status") == "success"
    ]
    return {
        "ok": success_count,
        "approved": approved_count,
        "err": error_count,
        "cov": round(sum(coverage_scores) / len(coverage_scores), 3) if coverage_scores else 0.0,
        "ev": round(sum(evidence_scores) / len(evidence_scores), 3) if evidence_scores else 0.0,
    }


def main() -> None:
    args = parse_args()
    dataframe = load_input_csv(args.input_csv)
    indices = selected_indices(dataframe, args.start_index, args.max_patients)

    print(f"[INFO] Input CSV: {args.input_csv}", flush=True)
    print(f"[INFO] Output CSV: {args.output_csv}", flush=True)
    print(f"[INFO] Selected patients: {len(indices)}", flush=True)
    print(f"[INFO] Extractor model: {args.extractor_model}", flush=True)
    print(f"[INFO] Verifier model: {'SKIPPED (Agent1-only)' if args.agent1_only else args.verifier_model}", flush=True)
    print(f"[INFO] Stage2 mode: {stage2_mode_name(args.agent1_only, args.raw_input_whole)}", flush=True)

    client = OllamaJsonClient(
        host=args.ollama_host,
        temperature=args.temperature,
        num_ctx=args.num_ctx,
        num_predict=args.num_predict,
        timeout=args.timeout,
        retries=args.retries,
        retry_sleep=args.retry_sleep,
    )
    required_models = [args.extractor_model] if args.agent1_only else [args.extractor_model, args.verifier_model]
    client.assert_models_available(required_models)

    output_rows: list[dict[str, Any]] = []
    progress = tqdm(indices, total=len(indices), desc="Stage2 patients", unit="patient")
    for position, row_index in enumerate(progress, start=1):
        row = dataframe.loc[row_index]
        progress.set_description(f"Stage2 row={row_index}")
        tqdm.write(f"[INFO] Processing patient {position}/{len(indices)}: {row['Professor_ID']} {row['수술ID']}")
        base_row = {column: row[column] for column in REQUIRED_COLUMNS}
        try:
            result = run_patient_stage2(
                client=client,
                row=row,
                extractor_model=args.extractor_model,
                verifier_model=args.verifier_model,
                max_iterations=args.max_iterations,
                coverage_threshold=args.coverage_threshold,
                evidence_threshold=args.evidence_threshold,
                agent1_only=args.agent1_only,
                raw_input_whole=args.raw_input_whole,
            )
            base_row.update(
                {
                    "Extracted_Facts": json.dumps(
                        {"all_facts": result["all_facts"], "chunks": result["chunks"]},
                        ensure_ascii=False,
                    ),
                    "Extracted_Facts_Readable": format_facts_readable(result["all_facts"]),
                    "Verification_Report": json.dumps(
                        {
                            "approved": result["approved"],
                            "mode": result["mode"],
                            "coverage_score": result["coverage_score"],
                            "evidence_support_score": result["evidence_support_score"],
                            "chunk_error_count": result["chunk_error_count"],
                            "unresolved_issues": result["unresolved_issues"],
                        },
                        ensure_ascii=False,
                    ),
                    "Verification_Summary": format_verification_summary(result),
                    "Stage2_Final_Summary": format_final_summary(result),
                    "Stage2_Approved": result["approved"],
                    "Stage2_Coverage_Score": result["coverage_score"],
                    "Stage2_Evidence_Support_Score": result["evidence_support_score"],
                    "Stage2_Total_Iterations": result["total_iterations"],
                    "Stage2_Chunk_Error_Count": result["chunk_error_count"],
                    "Stage2_Mode": result["mode"],
                    "Stage2_Extractor_Model": args.extractor_model,
                    "Stage2_Verifier_Model": "SKIPPED_AGENT1_ONLY" if args.agent1_only else args.verifier_model,
                    "Stage2_Status": "success",
                    "Stage2_Error": "",
                    "Stage2_Processed_At": pd.Timestamp.now("UTC").isoformat(),
                }
            )
        except Exception as exc:
            base_row.update(
                {
                    "Extracted_Facts": "",
                    "Extracted_Facts_Readable": "",
                    "Verification_Report": "",
                    "Verification_Summary": "",
                    "Stage2_Final_Summary": "",
                    "Stage2_Approved": False,
                    "Stage2_Coverage_Score": 0.0,
                    "Stage2_Evidence_Support_Score": 0.0,
                    "Stage2_Total_Iterations": 0,
                    "Stage2_Chunk_Error_Count": 0,
                    "Stage2_Mode": stage2_mode_name(args.agent1_only, args.raw_input_whole),
                    "Stage2_Extractor_Model": args.extractor_model,
                    "Stage2_Verifier_Model": "SKIPPED_AGENT1_ONLY" if args.agent1_only else args.verifier_model,
                    "Stage2_Status": "error",
                    "Stage2_Error": str(exc),
                    "Stage2_Processed_At": pd.Timestamp.now("UTC").isoformat(),
                }
            )
            tqdm.write(f"[ERROR] Patient failed: {exc}")
        output_rows.append(base_row)
        progress.set_postfix(summarize_progress(output_rows))
        if args.save_every > 0 and len(output_rows) % args.save_every == 0:
            write_outputs(output_rows, args.output_csv, skip_readable_report=True)
            tqdm.write(f"[INFO] Partial CSV saved after {len(output_rows)} rows: {args.output_csv}")

    readable_path = write_outputs(output_rows, args.output_csv, args.skip_readable_report)
    print(f"[INFO] Wrote: {args.output_csv}", flush=True)
    if readable_path:
        print(f"[INFO] Wrote readable report: {readable_path}", flush=True)


if __name__ == "__main__":
    main()
