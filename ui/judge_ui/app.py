import os
import json
import uuid
from typing import List, Optional

import streamlit as st

from models import (
    PatientCase,
    SpecialistAssessment,
    WorkerEvidence,
    JudgeRequest,
    JudgeResponse,
)
from judge_client import DummyJudgeClient, APIJudgeClient
from utils import (
    parse_list_from_textarea,
    generate_markdown_report,
)

APP_TITLE = "BLUEmed Judge — Specialist Interface"
SESSION_KEYS = {
    "mode": "mode",
    "api_url": "api_url",
    "api_key": "api_key",
    "last_request": "last_request",
    "last_response": "last_response",
    "history": "history",
}


def init_session_state():
    st.session_state.setdefault(SESSION_KEYS["mode"], "Dummy (no backend)")
    st.session_state.setdefault(SESSION_KEYS["api_url"], os.getenv("JUDGE_API_URL", ""))
    st.session_state.setdefault(SESSION_KEYS["api_key"], os.getenv("JUDGE_API_KEY", ""))
    st.session_state.setdefault(SESSION_KEYS["last_request"], None)
    st.session_state.setdefault(SESSION_KEYS["last_response"], None)
    st.session_state.setdefault(SESSION_KEYS["history"], [])


def sidebar_controls():
    st.sidebar.header("Settings")

    mode = st.sidebar.radio(
        "Judge backend",
        options=["Dummy (no backend)", "HTTP API"],
        index=0 if st.session_state[SESSION_KEYS["mode"]] == "Dummy (no backend)" else 1,
    )
    st.session_state[SESSION_KEYS["mode"]] = mode

    api_url = st.sidebar.text_input(
        "JUDGE_API_URL",
        value=st.session_state[SESSION_KEYS["api_url"]],
        placeholder="https://your-judge-service/run",
        help="Only used in HTTP API mode",
    )
    st.session_state[SESSION_KEYS["api_url"]] = api_url

    api_key = st.sidebar.text_input(
        "JUDGE_API_KEY",
        value=st.session_state[SESSION_KEYS["api_key"]],
        type="password",
        help="Only used in HTTP API mode",
    )
    st.session_state[SESSION_KEYS["api_key"]] = api_key

    st.sidebar.markdown("---")
    if st.sidebar.button("Clear last run"):
        st.session_state[SESSION_KEYS["last_request"]] = None
        st.session_state[SESSION_KEYS["last_response"]] = None
        st.success("Cleared last run")

    if st.sidebar.button("Clear history"):
        st.session_state[SESSION_KEYS["history"]] = []
        st.success("Cleared history")

    st.sidebar.markdown("---")
    st.sidebar.caption("Tip: Switch to HTTP API mode once your judge service is live.")


def build_client():
    mode = st.session_state[SESSION_KEYS["mode"]]
    if mode.startswith("Dummy"):
        return DummyJudgeClient()
    else:
        return APIJudgeClient(
            base_url=st.session_state[SESSION_KEYS["api_url"]].strip(),
            api_key=st.session_state[SESSION_KEYS["api_key"]].strip(),
        )


def main():
    st.set_page_config(page_title="BLUEmed Judge", layout="wide")
    init_session_state()
    sidebar_controls()

    st.title(APP_TITLE)
    st.write(
        "Enter the patient case and your assessment. Optionally include worker bot evidence (WebMD/Mayo). "
        "Click “Run Judge” to see the adjudication."
    )

    with st.form("case_intake"):
        st.subheader("Patient Case")
        col_a, col_b, col_c = st.columns([1, 1, 2], gap="large")
        with col_a:
            case_id = st.text_input("Case ID", value=str(uuid.uuid4())[:8])
            age = st.number_input("Age", min_value=0, max_value=120, value=45, step=1)
            sex = st.selectbox("Sex", options=["Male", "Female", "Other", "Unknown"], index=0)
            duration = st.text_input("Symptom duration (e.g., 3 weeks)", value="")
        with col_b:
            symptoms_text = st.text_area(
                "Symptoms (comma-separated)",
                value="fever, cough, fatigue",
                height=100,
            )
            symptoms = parse_list_from_textarea(symptoms_text)
        with col_c:
            notes = st.text_area(
                "Clinical notes",
                value="Patient reports intermittent fever and dry cough. No chest pain. Recently returned from travel.",
                height=150,
            )

        st.subheader("Your Assessment (Specialist)")
        col_d, col_e = st.columns([1, 1], gap="large")
        with col_d:
            provisional_diagnosis = st.text_input("Provisional diagnosis", value="Viral upper respiratory infection")
            icd10 = st.text_input("ICD-10 (optional)", value="")
        with col_e:
            differential_text = st.text_area(
                "Differential diagnoses (comma-separated)",
                value="influenza, covid-19, bacterial pneumonia",
                height=100,
            )
            differential = parse_list_from_textarea(differential_text)
        rationale = st.text_area(
            "Rationale (why you think this is the correct diagnosis)",
            value="Symptoms and timing are consistent with viral etiology. No focal chest findings.",
            height=120,
        )

        st.subheader("Worker Evidence (optional)")
        col_left, col_right = st.columns(2, gap="large")

        with col_left:
            webmd_summary = st.text_area(
                "WebMD summary",
                value="WebMD indicates that viral URIs often present with cough and low-grade fever; supportive care is typical.",
                height=120,
            )
            webmd_citations_text = st.text_area(
                "WebMD citations (one URL per line)",
                value="https://www.webmd.com/cold-and-flu/common_cold_overview\n",
                height=80,
            )
            webmd_citations = parse_list_from_textarea(webmd_citations_text, sep="\n")

        with col_right:
            mayo_summary = st.text_area(
                "Mayo Clinic summary",
                value="Mayo Clinic notes influenza and COVID-19 share overlapping symptoms such as fever and cough.",
                height=120,
            )
            mayo_citations_text = st.text_area(
                "Mayo citations (one URL per line)",
                value="https://www.mayoclinic.org/diseases-conditions/flu/symptoms-causes/syc-20351719\n",
                height=80,
            )
            mayo_citations = parse_list_from_textarea(mayo_citations_text, sep="\n")

        run = st.form_submit_button("Run Judge", use_container_width=True)

    if run:
        case = PatientCase(
            case_id=case_id.strip() or str(uuid.uuid4())[:8],
            age=int(age),
            sex=sex,
            symptoms=symptoms,
            duration=duration.strip(),
            notes=notes.strip(),
        )
        specialist = SpecialistAssessment(
            provisional_diagnosis=provisional_diagnosis.strip(),
            differential=differential,
            rationale=rationale.strip(),
            icd10=icd10.strip() or None,
        )
        evidence: List[WorkerEvidence] = []
        if webmd_summary.strip() or webmd_citations:
            evidence.append(
                WorkerEvidence(
                    source="webmd",
                    summary=webmd_summary.strip(),
                    citations=webmd_citations,
                    quotes=[],
                )
            )
        if mayo_summary.strip() or mayo_citations:
            evidence.append(
                WorkerEvidence(
                    source="mayoclinic",
                    summary=mayo_summary.strip(),
                    citations=mayo_citations,
                    quotes=[],
                )
            )

        req = JudgeRequest(case=case, specialist=specialist, evidence=evidence)

        st.session_state[SESSION_KEYS["last_request"]] = json.loads(req.model_dump_json())

        client = build_client()
        with st.spinner("Running judge..."):
            try:
                resp: JudgeResponse = client.run(req)
            except Exception as e:
                st.error(f"Judge error: {e}")
                return

        st.session_state[SESSION_KEYS["last_response"]] = json.loads(resp.model_dump_json())
        st.session_state[SESSION_KEYS["history"]].append(st.session_state[SESSION_KEYS]["last_response"])

    # Show results
    last_resp = st.session_state[SESSION_KEYS["last_response"]]
    last_req = st.session_state[SESSION_KEYS["last_request"]]
    if last_resp and last_req:
        st.markdown("---")
        st.subheader("Judge Result")

        verdict = last_resp["verdict"]["verdict"]
        confidence = last_resp["verdict"]["confidence"]
        critique = last_resp["verdict"]["critique"]
        key_points = last_resp["verdict"]["key_points"]
        alignment_scores = last_resp["verdict"]["alignment_scores"]
        citations = last_resp["verdict"]["citations"]

        verdict_display = {
            "agree": ("✅ Agreement", "green"),
            "disagree": ("❌ Disagreement", "red"),
            "uncertain": ("⚠️ Uncertain", "orange"),
        }.get(verdict, ("ℹ️", "blue"))

        st.markdown(f"### {verdict_display[0]} (confidence: {confidence:.2f})")
        cols = st.columns(len(alignment_scores) or 1)
        for i, (src, score) in enumerate(alignment_scores.items()):
            with cols[i]:
                st.metric(label=f"{src} alignment", value=f"{score:.2f}")

        st.markdown("#### Rationale")
        st.write(critique)

        if key_points:
            st.markdown("#### Key Points")
            for kp in key_points:
                st.markdown(f"- {kp}")

        if citations:
            st.markdown("#### Citations")
            for c in citations:
                st.markdown(f"- {c}")

        # Downloads
        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                label="Download JSON",
                file_name=f"judge_{last_req['case']['case_id']}.json",
                mime="application/json",
                data=json.dumps({"request": last_req, "response": last_resp}, indent=2),
                use_container_width=True,
            )
        with col2:
            md = generate_markdown_report(last_req, last_resp)
            st.download_button(
                label="Download Markdown Report",
                file_name=f"judge_{last_req['case']['case_id']}.md",
                mime="text/markdown",
                data=md,
                use_container_width=True,
            )

    # History
    if st.session_state[SESSION_KEYS["history"]]:
        st.markdown("---")
        st.subheader("History (this session)")
        for i, item in enumerate(reversed(st.session_state[SESSION_KEYS["history"]])):
            with st.expander(f"Run {len(st.session_state[SESSION_KEYS['history']]) - i} — verdict: {item['verdict']['verdict']} (conf {item['verdict']['confidence']:.2f})"):
                st.json(item)


if __name__ == "__main__":
    main()