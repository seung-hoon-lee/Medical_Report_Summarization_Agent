#!/usr/bin/env python3
"""Streamlit web console for the medical report summarization pipeline."""

from __future__ import annotations

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

DEFAULT_SETTINGS: dict[str, Any] = {
    "run_label": "clinical_note_run",
    "professor_id": "web_professor",
    "output_type": "외래기록지",
    "pipeline_target": "최종 문서 생성",
    "max_patients": 0,
    "max_iterations": 3,
    "coverage_threshold": 0.85,
    "evidence_threshold": 0.95,
    "extractor_model": "qwen3.5:9b",
    "verifier_model": "qwen3.5:9b",
    "style_model": "qwen3.6:35b",
    "generation_model": "qwen3.6:35b",
    "ollama_host": None,
    "stage34_mode": "ollama",
    "stage34_dry_run": False,
    "strict_validation": True,
    "skip_unmatched": False,
    "save_prompts": False,
}

STAGE_LABELS = {
    "stage1": "문서 정리",
    "stage2": "핵심 정보 추출",
    "stage3_4": "외래기록지 생성",
    "stage3_4_dry_run": "생성 흐름 점검",
}

PIPELINE_STEPS = [
    ("stage1", "문서 정리", "업로드한 의료 문서를 시간 순서와 문서 종류에 맞게 정리합니다."),
    ("stage2", "핵심 정보 추출", "정리된 문서에서 진단, 검사, 수술, 경과 등 핵심 사실을 검증하며 추출합니다."),
    ("stage3_4", "외래기록지 생성", "참고 샘플의 문체를 반영해 최종 외래기록지를 작성합니다."),
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
        initial_sidebar_state="collapsed",
    )
    inject_css()
    initialize_state()

    render_header()
    settings = default_settings()

    setup_tab, upload_tab, run_tab, results_tab = st.tabs(
        ["시작", "문서 업로드", "문서 생성", "결과"]
    )

    with setup_tab:
        render_setup(settings)
    with upload_tab:
        render_uploads(settings)
    with run_tab:
        render_run_controls(settings)
    with results_tab:
        render_results()


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
        [data-testid="stSidebar"],
        [data-testid="stSidebarCollapsedControl"],
        [data-testid="collapsedControl"] {
          display: none;
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
        .step-grid {
          display: grid;
          grid-template-columns: repeat(3, minmax(0, 1fr));
          gap: .75rem;
          margin: .75rem 0 1.1rem;
        }
        .step-card {
          background: var(--panel);
          border: 1px solid var(--line);
          border-radius: 8px;
          padding: .95rem;
          min-height: 118px;
        }
        .step-index {
          width: 1.8rem;
          height: 1.8rem;
          display: inline-flex;
          align-items: center;
          justify-content: center;
          border-radius: 999px;
          background: #0f766e;
          color: white;
          font-weight: 800;
          margin-bottom: .65rem;
        }
        .step-title {
          margin: 0 0 .35rem;
          color: var(--ink);
          font-size: 1rem;
          font-weight: 850;
        }
        .step-copy {
          margin: 0;
          color: var(--muted);
          font-size: .9rem;
          line-height: 1.45;
        }
        .pipeline-grid {
          display: grid;
          grid-template-columns: repeat(3, minmax(0, 1fr));
          gap: .85rem;
          margin: .7rem 0 1rem;
        }
        .pipeline-card {
          position: relative;
          background: var(--panel);
          border: 1px solid var(--line);
          border-radius: 8px;
          padding: 1rem;
          min-height: 142px;
          box-shadow: 0 10px 24px rgba(35, 33, 29, .05);
          overflow: hidden;
        }
        .pipeline-card::before {
          content: "";
          position: absolute;
          inset: 0 0 auto 0;
          height: 4px;
          background: var(--line);
        }
        .pipeline-card.done::before { background: #0f766e; }
        .pipeline-card.active::before { background: #b7791f; }
        .pipeline-card.failed::before { background: #a33f3f; }
        .pipeline-card.active {
          border-color: #b7791f;
          background: #fffaf0;
        }
        .pipeline-card.done {
          border-color: rgba(15, 118, 110, .45);
        }
        .pipeline-card.failed {
          border-color: rgba(163, 63, 63, .55);
          background: #fff7f4;
        }
        .pipeline-topline {
          display: flex;
          align-items: center;
          justify-content: space-between;
          gap: .75rem;
          margin-bottom: .7rem;
        }
        .pipeline-index {
          width: 2rem;
          height: 2rem;
          display: inline-flex;
          align-items: center;
          justify-content: center;
          border-radius: 999px;
          background: #2f3e46;
          color: #fffaf0;
          font-weight: 850;
        }
        .pipeline-card.done .pipeline-index { background: #0f766e; }
        .pipeline-card.active .pipeline-index { background: #b7791f; }
        .pipeline-card.failed .pipeline-index { background: #a33f3f; }
        .pipeline-badge {
          border: 1px solid var(--line);
          border-radius: 999px;
          padding: .18rem .55rem;
          color: var(--steel);
          font-size: .76rem;
          font-weight: 800;
          white-space: nowrap;
        }
        .pipeline-card.done .pipeline-badge {
          border-color: rgba(15, 118, 110, .35);
          color: #0f766e;
        }
        .pipeline-card.active .pipeline-badge {
          border-color: rgba(183, 121, 31, .42);
          color: #8a5a11;
        }
        .pipeline-card.failed .pipeline-badge {
          border-color: rgba(163, 63, 63, .35);
          color: #8f3030;
        }
        .pipeline-title {
          margin: 0 0 .35rem;
          color: var(--ink);
          font-size: 1.05rem;
          font-weight: 850;
        }
        .pipeline-copy {
          margin: 0;
          color: var(--muted);
          font-size: .9rem;
          line-height: 1.45;
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
          .step-grid { grid-template-columns: 1fr; }
          .pipeline-grid { grid-template-columns: 1fr; }
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
    run_label = prepared.paths.run_id if prepared else "업로드 대기 중"
    st.markdown(
        f"""
        <div class="app-header">
          <div>
            <p class="eyebrow">의료 문서 자동 작성</p>
            <h1 class="app-title">Medical Report Studio</h1>
          </div>
          <div class="run-pill">{run_label}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def default_settings() -> dict[str, Any]:
    """Return fixed web defaults so patients do not see internal pipeline knobs."""

    return dict(DEFAULT_SETTINGS)


def render_setup(settings: dict[str, Any]) -> None:
    prepared: PreparedRun | None = st.session_state.get("prepared_run")
    metrics = [
        ("생성 문서", settings["output_type"]),
        ("처리 범위", "업로드 전체"),
        ("참고 양식", "최대 5개"),
        ("실행 방식", "자동"),
    ]
    render_metric_grid(metrics)
    render_step_cards(
        [
            ("문서 업로드", "환자의 입력 문서를 올리면 시스템이 필요한 형식으로 정리합니다."),
            ("참고 양식 선택", "비슷한 외래기록지 샘플을 최대 5개까지 올려 문체를 참고합니다."),
            ("결과 생성", "핵심 정보를 추출한 뒤 참고 양식의 스타일에 맞춰 외래기록지를 생성합니다."),
        ]
    )

    if prepared:
        render_metric_grid(
            [
                ("준비된 문서", str(prepared.input_rows)),
                ("참고 샘플", str(prepared.style_sample_rows)),
                ("작업 번호", prepared.paths.run_id),
                ("확인 필요", str(len(prepared.warnings))),
            ]
        )
        if prepared.warnings:
            with st.expander("확인할 내용", expanded=False):
                for warning in prepared.warnings:
                    st.warning(warning)
    else:
        st.info("문서 업로드 탭에서 입력 문서와 참고 양식을 먼저 올려주세요.")


def render_uploads(settings: dict[str, Any]) -> None:
    col_a, col_b = st.columns([1.15, 1])
    with col_a:
        input_files = st.file_uploader(
            "환자 입력 문서",
            help="진료 기록, 수술 기록, 퇴원 요약 등 생성에 필요한 원본 문서를 업로드하세요.",
            type=["xlsx", "xls", "csv", "txt", "md", "json", "jsonl", "docx"],
            accept_multiple_files=True,
            key="input_uploads",
        )
    with col_b:
        style_files = st.file_uploader(
            "참고 외래기록지 샘플",
            help="원하는 문체와 구성을 보여주는 외래기록지 예시를 최대 5개까지 업로드하세요.",
            type=["xlsx", "xls", "csv", "txt", "md", "json", "jsonl", "docx"],
            accept_multiple_files=True,
            key="style_uploads",
        )

    selected_style_files = list(style_files or [])[:5]
    if style_files and len(style_files) > 5:
        st.warning("참고 샘플은 처음 5개 파일만 사용합니다.")

    st.divider()
    prepare_disabled = not input_files and not selected_style_files
    if st.button("업로드 파일 확인하기", disabled=prepare_disabled, use_container_width=True):
        input_payloads = uploads_to_payloads(input_files or [])
        style_payloads = uploads_to_payloads(selected_style_files)
        with st.spinner("업로드 파일을 정리하는 중입니다."):
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
        st.success("문서 준비가 완료되었습니다.")

    prepared: PreparedRun | None = st.session_state.get("prepared_run")
    if prepared:
        render_prepared_summary(prepared)


def render_run_controls(settings: dict[str, Any]) -> None:
    prepared: PreparedRun | None = st.session_state.get("prepared_run")
    if not prepared:
        st.info("먼저 문서 업로드 탭에서 파일을 올리고 확인을 완료해주세요.")
        return

    paths = prepared.paths
    render_metric_grid(
        [
            ("입력 문서", status_text(paths.input_csv.exists())),
            ("문서 정리", status_text(paths.stage1_csv.exists())),
            ("핵심 정보", status_text(paths.stage2_csv.exists())),
            ("참고 양식", status_text(paths.reference_csv.exists())),
            ("생성 결과", status_text(paths.stage4_csv.exists())),
            ("작업 번호", paths.run_id),
        ]
    )

    st.subheader("문서 생성")
    st.caption("버튼 한 번으로 문서 정리, 핵심 정보 추출, 외래기록지 생성을 차례대로 실행합니다.")
    render_pipeline_flow(pipeline_states_from_paths(paths))

    missing_requirements = []
    if prepared.input_rows == 0:
        missing_requirements.append("환자 입력 문서")
    if prepared.style_sample_rows == 0 or not paths.reference_csv.exists():
        missing_requirements.append("참고 외래기록지 샘플")
    if missing_requirements:
        st.warning("자동 생성을 시작하려면 " + ", ".join(missing_requirements) + "이 필요합니다.")

    start_disabled = bool(missing_requirements)
    if st.button("외래기록지 자동 생성 시작", disabled=start_disabled, use_container_width=True):
        execute_full_pipeline(paths, settings)


def render_results() -> None:
    prepared: PreparedRun | None = st.session_state.get("prepared_run")
    if not prepared:
        st.info("먼저 문서 업로드 탭에서 파일을 올리고 확인을 완료해주세요.")
        return

    paths = prepared.paths
    downloadable_results = result_downloads(paths)
    if not downloadable_results:
        st.info("아직 생성된 결과가 없습니다.")
        return

    render_generated_notes(paths)

    allow_sensitive_downloads = st.checkbox("결과 파일 다운로드 허용", value=False)
    if allow_sensitive_downloads:
        render_result_downloads(downloadable_results)
    else:
        st.info("환자 정보 보호를 위해 다운로드는 직접 허용한 뒤 표시됩니다.")

    st.divider()
    selected_label = st.selectbox("미리 볼 결과", [label for label, _ in downloadable_results])
    selected = next(path for label, path in downloadable_results if label == selected_label)
    show_sensitive = st.toggle("민감한 텍스트 열 표시", value=False)
    try:
        preview = read_table_preview(selected, max_rows=30)
        if not show_sensitive:
            preview = redact_dataframe(preview)
        st.dataframe(preview, use_container_width=True, hide_index=True)
    except Exception as exc:  # noqa: BLE001
        st.error(f"결과 파일을 미리 볼 수 없습니다: {exc}")


def render_generated_notes(paths: RunPaths) -> None:
    if not paths.stage4_csv.exists():
        return

    st.subheader("생성된 외래기록지")
    st.caption("생성 결과는 generated_notes.csv 파일에도 저장됩니다.")
    try:
        frame = pd.read_csv(paths.stage4_csv, dtype=str, keep_default_na=False)
    except Exception as exc:  # noqa: BLE001
        st.error(f"생성 결과를 읽을 수 없습니다: {exc}")
        return

    if frame.empty:
        st.info("생성 결과 파일은 있지만 아직 표시할 행이 없습니다.")
        return

    status_values = sorted(set(frame.get("validation_status", pd.Series(dtype=str)).astype(str)))
    if "dry_run" in status_values:
        st.warning("이 결과는 점검 모드로 생성되어 실제 LLM 생성 문서가 아닙니다.")
    elif "needs_review" in status_values:
        st.warning("검토가 필요한 결과가 있습니다. 사용 전 경고 내용을 확인하세요.")

    summary_columns = [
        column
        for column in ("record_id", "professor", "validation_status", "validation_warnings")
        if column in frame.columns
    ]
    if summary_columns:
        st.dataframe(frame[summary_columns].head(30), use_container_width=True, hide_index=True)

    show_notes = st.checkbox("생성된 외래기록지 본문 보기", value=False)
    if not show_notes:
        st.info("환자 정보 보호를 위해 본문은 기본적으로 숨겨져 있습니다.")
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


def execute_full_pipeline(paths: RunPaths, settings: dict[str, Any]) -> None:
    steps: list[tuple[str, Callable[[], CommandResult]]] = [
        ("stage1", lambda: run_stage1(paths, max_patients=settings["max_patients"])),
        (
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
        ),
        (
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
        ),
    ]

    states = {stage_name: "pending" for stage_name, _, _ in PIPELINE_STEPS}
    flow_slot = st.empty()
    progress = st.progress(0, text="자동 생성을 준비하고 있습니다.")

    with st.status("외래기록지 자동 생성 진행 중", expanded=True) as status:
        for index, (stage_name, runner) in enumerate(steps, start=1):
            display_name = STAGE_LABELS.get(stage_name, stage_name)
            states[stage_name] = "active"
            with flow_slot.container():
                render_pipeline_flow(states)
            progress.progress((index - 1) / len(steps), text=f"{display_name} 진행 중")
            st.write(f"{display_name}을 실행하고 있습니다.")

            result = runner()
            append_command_log(paths, stage_name, result)
            st.session_state["command_results"][stage_name] = result

            if result.ok:
                states[stage_name] = "done"
                with flow_slot.container():
                    render_pipeline_flow(states)
                progress.progress(index / len(steps), text=f"{display_name} 완료")
                continue

            states[stage_name] = "failed"
            with flow_slot.container():
                render_pipeline_flow(states)
            progress.progress((index - 1) / len(steps), text=f"{display_name} 실패")
            if result.stdout.strip() or result.stderr.strip():
                st.caption("상세 실행 출력은 보안상 화면에 표시하지 않습니다. 관리자에게 작업 번호를 전달해주세요.")
            status.update(label=f"{display_name}에서 중단되었습니다.", state="error")
            return

        progress.progress(1.0, text="외래기록지 생성 완료")
        status.update(label="외래기록지 생성이 완료되었습니다.", state="complete")
    st.success("생성된 외래기록지는 결과 탭에서 확인할 수 있습니다.")


def render_prepared_summary(prepared: PreparedRun) -> None:
    st.subheader("준비된 문서")
    render_metric_grid(
        [
            ("작업 번호", prepared.paths.run_id),
            ("입력 문서", str(prepared.input_rows)),
            ("참고 샘플", str(prepared.style_sample_rows)),
            ("생성 문서", DEFAULT_SETTINGS["output_type"]),
        ]
    )
    input_summary = csv_summary(prepared.paths.input_csv)
    if input_summary.get("exists"):
        st.caption(f"정리된 입력 항목: {', '.join(input_summary['columns'])}")
    with st.expander("저장된 업로드 파일", expanded=False):
        st.dataframe(pd.DataFrame(prepared.saved_files), use_container_width=True, hide_index=True)


def render_step_cards(items: list[tuple[str, str]]) -> None:
    cards = []
    for index, (title, copy) in enumerate(items, start=1):
        cards.append(
            '<div class="step-card">'
            f'<div class="step-index">{index}</div>'
            f'<p class="step-title">{html_escape(title)}</p>'
            f'<p class="step-copy">{html_escape(copy)}</p>'
            "</div>"
        )
    st.markdown(
        f'<div class="step-grid">{"".join(cards)}</div>',
        unsafe_allow_html=True,
    )


def pipeline_states_from_paths(paths: RunPaths) -> dict[str, str]:
    return {
        "stage1": "done" if paths.stage1_csv.exists() else "pending",
        "stage2": "done" if paths.stage2_csv.exists() else "pending",
        "stage3_4": "done" if paths.stage4_csv.exists() else "pending",
    }


def render_pipeline_flow(states: dict[str, str]) -> None:
    cards = []
    for index, (stage_name, title, copy) in enumerate(PIPELINE_STEPS, start=1):
        state = states.get(stage_name, "pending")
        cards.append(
            f'<div class="pipeline-card {html_escape(state)}">'
            '<div class="pipeline-topline">'
            f'<div class="pipeline-index">{index}</div>'
            f'<div class="pipeline-badge">{pipeline_state_label(state)}</div>'
            "</div>"
            f'<p class="pipeline-title">{html_escape(title)}</p>'
            f'<p class="pipeline-copy">{html_escape(copy)}</p>'
            "</div>"
        )
    st.markdown(
        f'<div class="pipeline-grid">{"".join(cards)}</div>',
        unsafe_allow_html=True,
    )


def pipeline_state_label(state: str) -> str:
    labels = {
        "pending": "대기",
        "active": "진행 중",
        "done": "완료",
        "failed": "중단",
    }
    return labels.get(state, "대기")


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


def result_downloads(paths: RunPaths) -> list[tuple[str, Path]]:
    candidates = [
        ("추출된 Fact", paths.stage2_csv),
        ("생성된 외래기록지", paths.stage4_csv),
    ]
    return [(label, path) for label, path in candidates if path.exists()]


def render_result_downloads(results: list[tuple[str, Path]]) -> None:
    st.subheader("결과 파일")
    columns = st.columns(2)
    filenames = {
        "추출된 Fact": "extracted_facts.csv",
        "생성된 외래기록지": "generated_outpatient_notes.csv",
    }
    for index, (label, path) in enumerate(results):
        with columns[index % len(columns)]:
            st.download_button(
                label=label,
                data=path.read_bytes(),
                file_name=filenames.get(label, path.name),
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
    return "완료" if value else "대기"


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
