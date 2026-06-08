#!/usr/bin/env python3
"""Streamlit web console for the medical report summarization pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import pandas as pd
import streamlit as st

from web_pipeline import (
    CommandResult,
    PreparedRun,
    RunPaths,
    UploadedPayload,
    append_command_log,
    csv_summary,
    existing_artifacts,
    prepare_run_workspace,
    read_table_preview,
    run_stage1,
    run_stage2,
    run_stage34,
)


OUTPUT_TYPES = [
    "외래기록지",
    "퇴원기록지",
    "입원기록지",
    "수술기록 요약",
    "협진의뢰서",
    "사용자 정의",
]

PIPELINE_TARGETS = [
    "최종 문서 생성",
    "검증된 Fact CSV",
    "스타일 워크북",
    "정렬된 타임라인",
    "감사 번들",
]

SENSITIVE_COLUMNS = {
    "Input",
    "Output",
    "Sorted_Timeline",
    "Extracted_Facts",
    "Extracted_Facts_Readable",
    "Verification_Report",
    "Stage2_Final_Summary",
    "style_prompt",
    "raw_model_output",
    "generated_note",
    "validation_warnings",
    "unsupported_terms_or_claims",
}


def main() -> None:
    st.set_page_config(
        page_title="Medical Report Studio",
        page_icon=None,
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_css()
    initialize_state()

    render_header()
    sidebar_settings = render_sidebar()

    setup_tab, upload_tab, run_tab, results_tab, audit_tab = st.tabs(
        ["작업 설정", "문서 업로드", "파이프라인 실행", "결과", "감사"]
    )

    with setup_tab:
        render_setup(sidebar_settings)
    with upload_tab:
        render_uploads(sidebar_settings)
    with run_tab:
        render_run_controls(sidebar_settings)
    with results_tab:
        render_results()
    with audit_tab:
        render_audit()


def inject_css() -> None:
    st.markdown(
        """
        <style>
        :root {
          --paper: #f6f2e9;
          --panel: #fffdf8;
          --ink: #23211d;
          --muted: #736b5f;
          --line: #d7cfbf;
          --teal: #0f766e;
          --amber: #b7791f;
          --plum: #7c3f58;
          --steel: #405161;
        }
        .stApp {
          background: linear-gradient(180deg, #f6f2e9 0%, #eee8dc 100%);
          color: var(--ink);
        }
        .block-container {
          max-width: 1320px;
          padding-top: 1.25rem;
          padding-bottom: 3rem;
        }
        [data-testid="stSidebar"] {
          background: #282620;
          border-right: 1px solid #3e392f;
        }
        [data-testid="stSidebar"] * {
          color: #f5efe3;
        }
        [data-testid="stSidebar"] .stSelectbox label,
        [data-testid="stSidebar"] .stNumberInput label,
        [data-testid="stSidebar"] .stTextInput label,
        [data-testid="stSidebar"] .stSlider label,
        [data-testid="stSidebar"] .stCheckbox label {
          color: #f5efe3;
        }
        .app-header {
          display: flex;
          align-items: flex-end;
          justify-content: space-between;
          gap: 1rem;
          padding: 1.1rem 0 1rem;
          border-bottom: 1px solid var(--line);
          margin-bottom: 1.25rem;
        }
        .eyebrow {
          margin: 0 0 .35rem;
          color: var(--teal);
          font-size: .78rem;
          font-weight: 800;
          letter-spacing: 0;
          text-transform: uppercase;
        }
        .app-title {
          margin: 0;
          color: var(--ink);
          font-size: 2.35rem;
          line-height: 1.05;
          letter-spacing: 0;
        }
        .run-pill {
          border: 1px solid var(--line);
          background: var(--panel);
          color: var(--steel);
          border-radius: 8px;
          padding: .55rem .7rem;
          font-size: .86rem;
          font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
          max-width: 34rem;
          overflow-wrap: anywhere;
        }
        .metric-grid {
          display: grid;
          grid-template-columns: repeat(4, minmax(0, 1fr));
          gap: .75rem;
          margin: .25rem 0 1rem;
        }
        .metric-card {
          background: var(--panel);
          border: 1px solid var(--line);
          border-radius: 8px;
          padding: .85rem .95rem;
          min-height: 84px;
        }
        .metric-label {
          margin: 0 0 .35rem;
          color: var(--muted);
          font-size: .76rem;
          font-weight: 700;
          letter-spacing: 0;
        }
        .metric-value {
          margin: 0;
          color: var(--ink);
          font-size: 1.22rem;
          font-weight: 850;
          line-height: 1.2;
          overflow-wrap: anywhere;
        }
        .section-band {
          border-top: 1px solid var(--line);
          padding-top: 1rem;
          margin-top: 1rem;
        }
        div.stButton > button,
        div.stDownloadButton > button {
          border-radius: 6px;
          border: 1px solid #8a7f6b;
          background: #2f3e46;
          color: #fffaf0;
          font-weight: 750;
          letter-spacing: 0;
        }
        div.stButton > button:hover,
        div.stDownloadButton > button:hover {
          border-color: var(--teal);
          background: #0f766e;
          color: white;
        }
        [data-testid="stDataFrame"] {
          border: 1px solid var(--line);
          border-radius: 8px;
          overflow: hidden;
        }
        .small-note {
          color: var(--muted);
          font-size: .86rem;
          line-height: 1.45;
        }
        @media (max-width: 900px) {
          .app-header { align-items: flex-start; flex-direction: column; }
          .app-title { font-size: 1.85rem; }
          .metric-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def initialize_state() -> None:
    st.session_state.setdefault("prepared_run", None)
    st.session_state.setdefault("command_results", {})


def render_header() -> None:
    prepared: PreparedRun | None = st.session_state.get("prepared_run")
    run_label = prepared.paths.run_id if prepared else "workspace not prepared"
    st.markdown(
        f"""
        <div class="app-header">
          <div>
            <p class="eyebrow">Clinical document pipeline</p>
            <h1 class="app-title">Medical Report Studio</h1>
          </div>
          <div class="run-pill">{run_label}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar() -> dict[str, Any]:
    st.sidebar.header("Run Settings")
    run_label = st.sidebar.text_input("Run label", value="clinical_note_run")
    professor_id = st.sidebar.text_input("Professor ID", value="web_professor")

    output_choice = st.sidebar.selectbox("Output document", OUTPUT_TYPES, index=0)
    custom_output = ""
    if output_choice == "사용자 정의":
        custom_output = st.sidebar.text_input("Custom output", value="맞춤 기록지")
    output_type = custom_output.strip() or output_choice

    pipeline_target = st.sidebar.selectbox("Pipeline target", PIPELINE_TARGETS, index=0)
    max_patients = st.sidebar.number_input("Rows to process, 0 for all", min_value=0, value=1, step=1)
    max_iterations = st.sidebar.number_input("Stage 2 max iterations", min_value=1, value=2, step=1)
    coverage_threshold = st.sidebar.slider("Coverage threshold", 0.0, 1.0, 0.85, 0.01)
    evidence_threshold = st.sidebar.slider("Evidence threshold", 0.0, 1.0, 0.95, 0.01)

    st.sidebar.divider()
    extractor_model = st.sidebar.text_input("Extractor model", value="qwen3.5:9b")
    verifier_model = st.sidebar.text_input("Verifier model", value="qwen3.5:9b")
    style_model = st.sidebar.text_input("Style model", value="qwen3.6:35b")
    generation_model = st.sidebar.text_input("Generation model", value="qwen3.6:35b")
    ollama_host = st.sidebar.text_input("Ollama host", value="")
    stage34_mode = st.sidebar.selectbox("Stage 3/4 backend", ["ollama", "dry_run"], index=0)
    if stage34_mode == "dry_run":
        st.sidebar.warning("dry_run creates placeholder rows only. Use ollama for real outpatient-note generation.")

    st.sidebar.divider()
    strict_validation = st.sidebar.checkbox("Strict validation", value=True)
    skip_unmatched = st.sidebar.checkbox("Skip unmatched styles", value=False)
    save_prompts = st.sidebar.checkbox("Save prompts in audit", value=False)

    return {
        "run_label": run_label,
        "professor_id": professor_id.strip() or "web_professor",
        "output_type": output_type,
        "pipeline_target": pipeline_target,
        "max_patients": int(max_patients),
        "max_iterations": int(max_iterations),
        "coverage_threshold": float(coverage_threshold),
        "evidence_threshold": float(evidence_threshold),
        "extractor_model": extractor_model.strip() or "qwen3.5:9b",
        "verifier_model": verifier_model.strip() or "qwen3.5:9b",
        "style_model": style_model.strip() or "qwen3.5:9b",
        "generation_model": generation_model.strip() or "qwen3.5:9b",
        "ollama_host": ollama_host.strip() or None,
        "stage34_mode": stage34_mode,
        "stage34_dry_run": stage34_mode == "dry_run",
        "strict_validation": bool(strict_validation),
        "skip_unmatched": bool(skip_unmatched),
        "save_prompts": bool(save_prompts),
    }


def render_setup(settings: dict[str, Any]) -> None:
    prepared: PreparedRun | None = st.session_state.get("prepared_run")
    metrics = [
        ("Target", settings["pipeline_target"]),
        ("Output", settings["output_type"]),
        ("Rows", str(settings["max_patients"] or "all")),
        ("Backend", settings["stage34_mode"]),
    ]
    render_metric_grid(metrics)

    if prepared:
        render_metric_grid(
            [
                ("Workspace", str(prepared.paths.root)),
                ("Input rows", str(prepared.input_rows)),
                ("Style samples", str(prepared.style_sample_rows)),
                ("Warnings", str(len(prepared.warnings))),
            ]
        )
        if prepared.warnings:
            with st.expander("Workspace warnings", expanded=False):
                for warning in prepared.warnings:
                    st.warning(warning)
    else:
        st.info("문서 업로드 탭에서 실행 workspace를 먼저 준비하세요.")


def render_uploads(settings: dict[str, Any]) -> None:
    col_a, col_b = st.columns([1.15, 1])
    with col_a:
        input_files = st.file_uploader(
            "Input documents",
            type=["xlsx", "xls", "csv", "txt", "md", "json", "jsonl", "docx"],
            accept_multiple_files=True,
            key="input_uploads",
        )
    with col_b:
        style_files = st.file_uploader(
            "Reference output samples, up to 5",
            type=["xlsx", "xls", "csv", "txt", "md", "json", "jsonl", "docx"],
            accept_multiple_files=True,
            key="style_uploads",
        )

    selected_style_files = list(style_files or [])[:5]
    if style_files and len(style_files) > 5:
        st.warning("Style samples are limited to the first 5 uploaded files.")

    st.divider()
    prepare_disabled = not input_files and not selected_style_files
    if st.button("Prepare workspace", disabled=prepare_disabled, use_container_width=True):
        input_payloads = uploads_to_payloads(input_files or [])
        style_payloads = uploads_to_payloads(selected_style_files)
        with st.spinner("Preparing isolated run workspace"):
            prepared = prepare_run_workspace(
                label=settings["run_label"],
                input_payloads=input_payloads,
                style_payloads=style_payloads,
                professor_id=settings["professor_id"],
                output_type=settings["output_type"],
                max_style_samples=5,
            )
        st.session_state["prepared_run"] = prepared
        st.session_state["command_results"] = {}
        st.success(f"Workspace prepared: {prepared.paths.run_id}")

    prepared: PreparedRun | None = st.session_state.get("prepared_run")
    if prepared:
        render_prepared_summary(prepared)


def render_run_controls(settings: dict[str, Any]) -> None:
    prepared: PreparedRun | None = st.session_state.get("prepared_run")
    if not prepared:
        st.info("No prepared workspace.")
        return

    paths = prepared.paths
    render_metric_grid(
        [
            ("Input CSV", status_text(paths.input_csv.exists())),
            ("Stage 1", status_text(paths.stage1_csv.exists())),
            ("Stage 2", status_text(paths.stage2_csv.exists())),
            ("References", status_text(paths.reference_csv.exists())),
            ("Stage 3/4", status_text(paths.stage4_csv.exists())),
            ("Output", settings["output_type"]),
            ("Rows", str(settings["max_patients"] or "all")),
            ("Run ID", paths.run_id),
        ]
    )

    st.subheader("Stage Controls")
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("Run Stage 1", disabled=prepared.input_rows == 0, use_container_width=True):
            execute_stage(
                "stage1",
                lambda: run_stage1(paths, max_patients=settings["max_patients"]),
                paths,
            )
    with c2:
        stage2_disabled = not paths.stage1_csv.exists()
        if st.button("Run Stage 2", disabled=stage2_disabled, use_container_width=True):
            execute_stage(
                "stage2",
                lambda: run_stage2(
                    paths,
                    extractor_model=settings["extractor_model"],
                    verifier_model=settings["verifier_model"],
                    max_patients=settings["max_patients"],
                    max_iterations=settings["max_iterations"],
                    coverage_threshold=settings["coverage_threshold"],
                    evidence_threshold=settings["evidence_threshold"],
                    ollama_host=settings["ollama_host"],
                ),
                paths,
            )
    with c3:
        stage34_disabled = not paths.stage2_csv.exists() or not paths.reference_csv.exists()
        if st.button("Run Stage 3/4", disabled=stage34_disabled, use_container_width=True):
            execute_stage(
                "stage3_4",
                lambda: run_stage34(
                    paths,
                    facts_csv=paths.stage2_csv,
                    model=settings["generation_model"],
                    style_model=settings["style_model"],
                    generator_model=settings["generation_model"],
                    dry_run=settings["stage34_dry_run"],
                    sample_count=5,
                    max_rows=settings["max_patients"] or None,
                    strict_validation=settings["strict_validation"],
                    skip_unmatched=settings["skip_unmatched"],
                    save_prompts=settings["save_prompts"],
                    ollama_host=settings["ollama_host"],
                ),
                paths,
            )

    st.divider()
    demo_disabled = not paths.stage2_csv.exists() or not paths.reference_csv.exists()
    if st.button("Run no-LLM Stage 3/4 contract test", disabled=demo_disabled, use_container_width=True):
        execute_demo_pipeline(paths, settings)

    render_command_results()


def render_results() -> None:
    prepared: PreparedRun | None = st.session_state.get("prepared_run")
    if not prepared:
        st.info("No prepared workspace.")
        return

    paths = prepared.paths
    artifacts = existing_artifacts(paths)
    if not artifacts:
        st.info("No artifacts yet.")
        return

    render_generated_notes(paths)

    allow_sensitive_downloads = st.checkbox("Enable sensitive artifact downloads", value=False)
    if allow_sensitive_downloads:
        render_artifact_downloads(artifacts)
    else:
        st.info("Raw artifact downloads are hidden until sensitive downloads are enabled.")
    previewable = [path for path in artifacts if path.suffix.lower() in {".csv", ".xlsx", ".xls"}]
    if not previewable:
        return

    st.divider()
    selected = st.selectbox("Preview artifact", previewable, format_func=lambda path: path.name)
    show_sensitive = st.toggle("Show sensitive text columns", value=False)
    try:
        preview = read_table_preview(Path(selected), max_rows=30)
        if not show_sensitive:
            preview = redact_dataframe(preview)
        st.dataframe(preview, use_container_width=True, hide_index=True)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not preview artifact: {exc}")


def render_generated_notes(paths: RunPaths) -> None:
    if not paths.stage4_csv.exists():
        return

    st.subheader("Generated Notes")
    st.caption("The outpatient note is stored in the generated_note column of outputs/generated_notes.csv.")
    try:
        frame = pd.read_csv(paths.stage4_csv, dtype=str, keep_default_na=False)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not read generated notes: {exc}")
        return

    if frame.empty:
        st.info("generated_notes.csv exists, but it has no rows.")
        return

    status_values = sorted(set(frame.get("validation_status", pd.Series(dtype=str)).astype(str)))
    if "dry_run" in status_values:
        st.warning("This result was created in dry_run mode, so generated_note contains a placeholder, not an LLM-generated note.")
    elif "needs_review" in status_values:
        st.warning("One or more generated notes need review. Check validation_warnings before using the note.")

    summary_columns = [
        column
        for column in ("record_id", "professor", "validation_status", "validation_warnings")
        if column in frame.columns
    ]
    if summary_columns:
        st.dataframe(frame[summary_columns].head(30), use_container_width=True, hide_index=True)

    show_notes = st.checkbox("Show generated note text", value=False)
    if not show_notes:
        st.info("Generated note text is hidden because it may include patient information.")
        return

    for index, row in frame.head(30).iterrows():
        record_id = row.get("record_id", f"row {index}")
        status = row.get("validation_status", "")
        title = f"{record_id} · {status}" if status else str(record_id)
        with st.expander(title, expanded=len(frame) == 1):
            st.text(row.get("generated_note", ""))
            warnings = row.get("validation_warnings", "")
            if warnings and warnings != "[]":
                st.caption(f"validation_warnings: {warnings}")


def render_audit() -> None:
    prepared: PreparedRun | None = st.session_state.get("prepared_run")
    if not prepared:
        st.info("No prepared workspace.")
        return

    paths = prepared.paths
    st.code(str(paths.root), language="text")
    if paths.manifest_path.exists():
        with st.expander("Run manifest", expanded=True):
            st.json(json.loads(paths.manifest_path.read_text(encoding="utf-8")))

    log_files = sorted(paths.logs_dir.glob("*.json"))
    if log_files:
        st.subheader("Command logs")
        show_command_logs = st.checkbox("Show sensitive command logs", value=False)
        if show_command_logs:
            for path in log_files:
                with st.expander(path.name, expanded=False):
                    st.json(json.loads(path.read_text(encoding="utf-8")))
        else:
            st.info("Command logs are hidden because they may include patient identifiers or model output.")


def execute_demo_pipeline(paths: RunPaths, settings: dict[str, Any]) -> None:
    execute_stage(
        "stage3_4_dry_run",
        lambda: run_stage34(
            paths,
            facts_csv=paths.stage2_csv,
            model=settings["generation_model"],
            style_model=settings["style_model"],
            generator_model=settings["generation_model"],
            dry_run=True,
            sample_count=5,
            max_rows=settings["max_patients"] or None,
            strict_validation=settings["strict_validation"],
            skip_unmatched=settings["skip_unmatched"],
            save_prompts=False,
            ollama_host=settings["ollama_host"],
        ),
        paths,
    )


def execute_stage(stage_name: str, runner: Callable[[], CommandResult], paths: RunPaths) -> None:
    with st.status(f"{stage_name} running", expanded=True) as status:
        result = runner()
        append_command_log(paths, stage_name, result)
        st.session_state["command_results"][stage_name] = result
        st.code(" ".join(result.command), language="bash")
        if result.stdout.strip() or result.stderr.strip():
            st.caption("Command output is hidden by default because it may include sensitive data. See the audit tab to reveal logs.")
        if result.ok:
            status.update(label=f"{stage_name} complete", state="complete")
        else:
            status.update(label=f"{stage_name} failed", state="error")


def render_command_results() -> None:
    results: dict[str, CommandResult] = st.session_state.get("command_results", {})
    if not results:
        return
    st.subheader("Recent Runs")
    for stage_name, result in results.items():
        state = "complete" if result.ok else "error"
        with st.expander(f"{stage_name}: {state} in {result.duration_seconds}s", expanded=False):
            st.code(" ".join(result.command), language="bash")
            if result.stdout.strip() or result.stderr.strip():
                st.caption("Command output hidden; use the audit tab's sensitive log switch if needed.")


def render_prepared_summary(prepared: PreparedRun) -> None:
    st.subheader("Prepared Workspace")
    render_metric_grid(
        [
            ("Run ID", prepared.paths.run_id),
            ("Input rows", str(prepared.input_rows)),
            ("Style samples", str(prepared.style_sample_rows)),
            ("Manifest", prepared.paths.manifest_path.name),
        ]
    )
    input_summary = csv_summary(prepared.paths.input_csv)
    if input_summary.get("exists"):
        st.caption(f"Input CSV columns: {', '.join(input_summary['columns'])}")
    with st.expander("Saved files", expanded=False):
        st.dataframe(pd.DataFrame(prepared.saved_files), use_container_width=True, hide_index=True)


def render_metric_grid(items: list[tuple[str, str]]) -> None:
    cards = []
    for label, value in items:
        cards.append(
            '<div class="metric-card">'
            f'<p class="metric-label">{html_escape(label)}</p>'
            f'<p class="metric-value">{html_escape(value)}</p>'
            "</div>"
        )
    st.markdown(
        f'<div class="metric-grid">{"".join(cards)}</div>',
        unsafe_allow_html=True,
    )


def render_artifact_downloads(artifacts: list[Path]) -> None:
    st.subheader("Artifacts")
    columns = st.columns(3)
    for index, path in enumerate(artifacts):
        with columns[index % 3]:
            st.download_button(
                label=path.name,
                data=path.read_bytes(),
                file_name=path.name,
                mime=mime_for_path(path),
                use_container_width=True,
            )


def uploads_to_payloads(files: list[Any]) -> list[UploadedPayload]:
    return [UploadedPayload(name=file.name, data=file.getvalue()) for file in files]


def choose_facts_csv(paths: RunPaths) -> Path:
    if paths.stage2_csv.exists():
        return paths.stage2_csv
    if paths.stage1_csv.exists():
        return paths.stage1_csv
    return paths.input_csv


def redact_dataframe(frame: pd.DataFrame) -> pd.DataFrame:
    redacted = frame.copy()
    for column in redacted.columns:
        if str(column) in SENSITIVE_COLUMNS:
            redacted[column] = redacted[column].map(lambda value: redaction_label(value))
    return redacted


def redaction_label(value: Any) -> str:
    text = "" if value is None else str(value)
    if not text:
        return ""
    return f"[redacted, {len(text)} chars]"


def status_text(value: bool) -> str:
    return "ready" if value else "pending"


def tail(text: str, max_chars: int = 6000) -> str:
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]


def mime_for_path(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return "text/csv"
    if suffix == ".xlsx":
        return "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    if suffix == ".json":
        return "application/json"
    if suffix == ".jsonl":
        return "application/x-ndjson"
    return "application/octet-stream"


def html_escape(value: Any) -> str:
    text = str(value)
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#x27;")
    )


if __name__ == "__main__":
    main()
