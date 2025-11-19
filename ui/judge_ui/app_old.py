import os
import json
import uuid
from typing import Optional

import streamlit as st

from models import DebateRequest, DebateResponse
from judge_client import DummyJudgeClient, APIJudgeClient
from utils import generate_markdown_report

APP_TITLE = "BLUEmed — Medical Note Analysis System"


def init_session_state():
    """Initialize session state variables."""
    st.session_state.setdefault("mode", "Dummy (no backend)")
    st.session_state.setdefault("api_url", os.getenv("JUDGE_API_URL", ""))
    st.session_state.setdefault("api_key", os.getenv("JUDGE_API_KEY", ""))
    st.session_state.setdefault("last_response", None)
    st.session_state.setdefault("history", [])


def sidebar_controls():
    """Render sidebar with configuration controls."""
    st.sidebar.header("⚙️ Settings")

    # Mode selector
    mode = st.sidebar.radio(
        "Backend Mode",
        options=["Dummy (no backend)", "HTTP API"],
        index=0 if st.session_state["mode"] == "Dummy (no backend)" else 1,
        help="Choose whether to use the dummy simulator or connect to the real backend API"
    )
    st.session_state["mode"] = mode

    # API configuration (only show if HTTP API mode)
    if mode == "HTTP API":
        st.sidebar.subheader("API Configuration")
        
        api_url = st.sidebar.text_input(
            "API URL",
            value=st.session_state["api_url"],
            placeholder="https://your-api.com/analyze",
            help="The endpoint URL for the debate system API"
        )
        st.session_state["api_url"] = api_url

        api_key = st.sidebar.text_input(
            "API Key",
            value=st.session_state["api_key"],
            type="password",
            placeholder="Enter your API key",
            help="Optional authentication key for the API"
        )
        st.session_state["api_key"] = api_key

    # Action buttons
    st.sidebar.markdown("---")
    st.sidebar.subheader("Actions")
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("🗑️ Clear Result", use_container_width=True):
            st.session_state["last_response"] = None
            st.rerun()
    
    with col2:
        if st.button("🔄 Clear All", use_container_width=True):
            st.session_state["last_response"] = None
            st.session_state["history"] = []
            st.rerun()

    # Show history count
    if st.session_state["history"]:
        st.sidebar.info(f"📊 History: {len(st.session_state['history'])} analysis(es)")


def build_client():
    """Build the appropriate judge client based on mode."""
    if st.session_state["mode"] == "Dummy (no backend)":
        return DummyJudgeClient()
    else:
        return APIJudgeClient(
            base_url=st.session_state["api_url"],
            api_key=st.session_state["api_key"]
        )


def render_expert_argument(expert_name: str, expert_label: str, argument, color: str):
    """Render a single expert's argument in a styled container."""
    with st.container():
        st.markdown(f"### {expert_label}")
        st.markdown(f"**Round {argument.round}**")
        
        # Display the argument content in a styled box
        st.markdown(
            f"""<div style="background-color: {color}; padding: 15px; border-radius: 8px; 
            border-left: 4px solid {'#1f77b4' if expert_name == 'A' else '#ff7f0e'};">
            {argument.content.replace(chr(10), '<br>')}
            </div>""",
            unsafe_allow_html=True
        )


def render_judge_decision(decision):
    """Render the judge's final decision."""
    st.markdown("## 🏛️ Judge's Final Decision")
    
    # Visual indicator for final answer
    answer_colors = {
        "CORRECT": ("✅", "#d4edda", "#155724"),
        "INCORRECT": ("❌", "#f8d7da", "#721c24"),
        "UNKNOWN": ("❓", "#fff3cd", "#856404")
    }
    
    icon, bg_color, text_color = answer_colors.get(decision.final_answer, ("ℹ️", "#d1ecf1", "#0c5460"))
    
    st.markdown(
        f"""<div style="background-color: {bg_color}; padding: 20px; border-radius: 10px; 
        border: 2px solid {text_color}; margin: 20px 0;">
        <h2 style="color: {text_color}; margin: 0;">{icon} Final Answer: {decision.final_answer}</h2>
        </div>""",
        unsafe_allow_html=True
    )
    
    # Show metrics
    col1, col2 = st.columns(2)
    with col1:
        if decision.confidence_score:
            st.metric("Confidence Score", f"{decision.confidence_score}/10")
    with col2:
        if decision.winner:
            st.metric("Winner", decision.winner)
    
    # Show reasoning
    st.markdown("### 📝 Reasoning")
    st.write(decision.reasoning)


def main():
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon="🏥",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    init_session_state()
    
    st.title(APP_TITLE)
    st.markdown("*AI-powered medical note analysis through expert debate*")
    st.markdown("---")

    sidebar_controls()

    # Main input form
    st.markdown("## 📋 Medical Note Input")
    
    with st.form("medical_note_form"):
        medical_note = st.text_area(
            "Enter the medical note to analyze:",
            height=200,
            placeholder="Example: 54-year-old woman with a painful, rapidly growing leg lesion for 1 month...",
            help="Paste the complete medical note here. The system will analyze it for potential errors."
        )
        
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            submit_button = st.form_submit_button("🔍 Analyze Medical Note", use_container_width=True, type="primary")
        with col2:
            use_example = st.form_submit_button("📄 Load Example", use_container_width=True)

    # Handle example loading
    if use_example:
        example_note = """54-year-old woman with a painful, rapidly growing leg lesion for 1 month.
History includes Crohn's disease, diabetes, hypertension, and previous anterior uveitis.
Examination revealed a 4-cm tender ulcerative lesion with necrotic base and purplish borders,
along with pitting edema and dilated veins. Diagnosed as a venous ulcer."""
        st.rerun()

    # Handle form submission
    if submit_button:
        if not medical_note.strip():
            st.error("⚠️ Please enter a medical note to analyze.")
        else:
            # Create request
            req = DebateRequest(medical_note=medical_note.strip(), max_rounds=2)
            
            # Run analysis
            client = build_client()
            
            with st.spinner("🔄 Running debate analysis... This may take a moment."):
                try:
                    resp: DebateResponse = client.run(req)
                    st.session_state["last_response"] = json.loads(resp.model_dump_json())
                    st.session_state["history"].append(st.session_state["last_response"])
                    st.success("✅ Analysis complete!")
                except Exception as e:
                    st.error(f"❌ Error during analysis: {str(e)}")
                    return

    # Display results
    if st.session_state["last_response"]:
        resp_data = st.session_state["last_response"]
        
        st.markdown("---")
        st.markdown("## 📊 Analysis Results")
        
        # Show medical note being analyzed
        with st.expander("📄 Medical Note (click to expand)", expanded=False):
            st.text(resp_data["medical_note"])
        
        st.markdown("---")
        
        # Round 1 Arguments
        st.markdown("## 🥊 Round 1: Initial Arguments")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if resp_data["expertA_arguments"]:
                arg_data = resp_data["expertA_arguments"][0]
                if isinstance(arg_data, dict):
                    from models import DebateArgument
                    arg = DebateArgument(**arg_data)
                else:
                    arg = arg_data
                render_expert_argument(
                    "A",
                    "👨‍⚕️ Expert A (Mayo Clinic)",
                    arg,
                    "#e3f2fd"
                )
                # Show retrieved docs if available
                if resp_data.get("expertA_retrieved_docs"):
                    with st.expander("📚 Mayo Clinic Sources", expanded=False):
                        for i, doc in enumerate(resp_data["expertA_retrieved_docs"][:3], 1):
                            st.markdown(f"**{i}.** {doc[:200]}...")
        
        with col2:
            if resp_data["expertB_arguments"]:
                arg_data = resp_data["expertB_arguments"][0]
                if isinstance(arg_data, dict):
                    from models import DebateArgument
                    arg = DebateArgument(**arg_data)
                else:
                    arg = arg_data
                render_expert_argument(
                    "B",
                    "👩‍⚕️ Expert B (WebMD)",
                    arg,
                    "#fff3e0"
                )
                # Show retrieved docs if available
                if resp_data.get("expertB_retrieved_docs"):
                    with st.expander("📚 WebMD Sources", expanded=False):
                        for i, doc in enumerate(resp_data["expertB_retrieved_docs"][:3], 1):
                            st.markdown(f"**{i}.** {doc[:200]}...")
        
        st.markdown("---")
        
        # Round 2 Arguments
        st.markdown("## 🥊 Round 2: Counter-Arguments")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if len(resp_data["expertA_arguments"]) > 1:
                arg_data = resp_data["expertA_arguments"][1]
                if isinstance(arg_data, dict):
                    from models import DebateArgument
                    arg = DebateArgument(**arg_data)
                else:
                    arg = arg_data
                render_expert_argument(
                    "A",
                    "👨‍⚕️ Expert A (Mayo Clinic)",
                    arg,
                    "#e3f2fd"
                )
        
        with col2:
            if len(resp_data["expertB_arguments"]) > 1:
                arg_data = resp_data["expertB_arguments"][1]
                if isinstance(arg_data, dict):
                    from models import DebateArgument
                    arg = DebateArgument(**arg_data)
                else:
                    arg = arg_data
                render_expert_argument(
                    "B",
                    "👩‍⚕️ Expert B (WebMD)",
                    arg,
                    "#fff3e0"
                )
        
        st.markdown("---")
        
        # Judge Decision
        if resp_data.get("judge_decision"):
            from models import JudgeDecision
            decision = JudgeDecision(**resp_data["judge_decision"]) if isinstance(resp_data["judge_decision"], dict) else resp_data["judge_decision"]
            render_judge_decision(decision)
        
        st.markdown("---")
        
        # Download options
        st.markdown("## 💾 Download Results")
        col1, col2 = st.columns(2)
        
        with col1:
            st.download_button(
                label="📥 Download JSON",
                file_name=f"debate_analysis_{resp_data['request_id'][:8]}.json",
                mime="application/json",
                data=json.dumps(resp_data, indent=2),
                use_container_width=True
            )
        
        with col2:
            md_report = generate_markdown_report(resp_data)
            st.download_button(
                label="📥 Download Report (MD)",
                file_name=f"debate_analysis_{resp_data['request_id'][:8]}.md",
                mime="text/markdown",
                data=md_report,
                use_container_width=True
            )


if __name__ == "__main__":
    main()
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