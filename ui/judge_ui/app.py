import os
import json
from typing import Optional

import streamlit as st

from models import DebateRequest, DebateResponse, DebateArgument, JudgeDecision
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
    st.sidebar.header("Settings")

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
        if st.button("Clear Result", use_container_width=True):
            st.session_state["last_response"] = None
            st.rerun()
    
    with col2:
        if st.button("Clear All", use_container_width=True):
            st.session_state["last_response"] = None
            st.session_state["history"] = []
            st.rerun()

    # Show history count
    if st.session_state["history"]:
        st.sidebar.info(f"History: {len(st.session_state['history'])} analysis(es)")


def build_client():
    """Build the appropriate judge client based on mode."""
    if st.session_state["mode"] == "Dummy (no backend)":
        return DummyJudgeClient()
    else:
        return APIJudgeClient(
            base_url=st.session_state["api_url"],
            api_key=st.session_state["api_key"]
        )


def render_expert_argument(expert_name: str, expert_label: str, argument: DebateArgument, color: str):
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


def render_judge_decision(decision: JudgeDecision):
    """Render the judge's final decision."""
    st.markdown("## Judge's Final Decision")
    
    # Visual indicator for final answer
    answer_colors = {
        "CORRECT": ("#d4edda", "#155724"),
        "INCORRECT": ("#f8d7da", "#721c24"),
        "UNKNOWN": ("#fff3cd", "#856404")
    }
    
    bg_color, text_color = answer_colors.get(decision.final_answer, ("#d1ecf1", "#0c5460"))
    
    st.markdown(
        f"""<div style="background-color: {bg_color}; padding: 20px; border-radius: 10px; 
        border: 2px solid {text_color}; margin: 20px 0;">
        <h2 style="color: {text_color}; margin: 0;">Final Answer: {decision.final_answer}</h2>
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
    st.markdown("### Reasoning")
    st.write(decision.reasoning)


def main():
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon="⚕",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    init_session_state()
    
    st.title(APP_TITLE)
    st.markdown("*AI-powered medical note analysis through expert debate*")
    st.markdown("---")

    sidebar_controls()

    # Main input form
    st.markdown("## Medical Note Input")
    
    with st.form("medical_note_form"):
        medical_note = st.text_area(
            "Enter the medical note to analyze:",
            height=200,
            placeholder="Example: 54-year-old woman with a painful, rapidly growing leg lesion for 1 month...",
            help="Paste the complete medical note here. The system will analyze it for potential errors."
        )
        
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            submit_button = st.form_submit_button("Analyze Medical Note", use_container_width=True, type="primary")
        with col2:
            use_example = st.form_submit_button("Load Example", use_container_width=True)

    # Handle example loading
    if use_example:
        st.info("Click 'Load Example' again after the page reloads, or paste your own medical note.")

    # Handle form submission
    if submit_button:
        if not medical_note.strip():
            st.error("Please enter a medical note to analyze.")
        else:
            # Create request
            req = DebateRequest(medical_note=medical_note.strip(), max_rounds=2)
            
            # Run analysis
            client = build_client()
            
            with st.spinner("Running debate analysis... This may take a moment."):
                try:
                    resp: DebateResponse = client.run(req)
                    st.session_state["last_response"] = json.loads(resp.model_dump_json())
                    st.session_state["history"].append(st.session_state["last_response"])
                    st.success("Analysis complete!")
                except Exception as e:
                    st.error(f"Error during analysis: {str(e)}")
                    return

    # Display results
    if st.session_state["last_response"]:
        resp_data = st.session_state["last_response"]
        
        st.markdown("---")
        st.markdown("## Analysis Results")
        
        # Show medical note being analyzed
        with st.expander("Medical Note (click to expand)", expanded=False):
            st.text(resp_data["medical_note"])
        
        st.markdown("---")
        
        # Judge Decision (shown first)
        if resp_data.get("judge_decision"):
            decision_data = resp_data["judge_decision"]
            decision = JudgeDecision(**decision_data) if isinstance(decision_data, dict) else decision_data
            render_judge_decision(decision)
        
        st.markdown("---")
        
        # Round 1 Arguments
        with st.expander("Round 1: Initial Arguments", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                if resp_data["expertA_arguments"]:
                    arg_data = resp_data["expertA_arguments"][0]
                    arg = DebateArgument(**arg_data) if isinstance(arg_data, dict) else arg_data
                    render_expert_argument("A", "Expert A (Mayo Clinic)", arg, "#e3f2fd")
                    
                    # Show retrieved docs if available
                    if resp_data.get("expertA_retrieved_docs"):
                        with st.expander("Mayo Clinic Sources", expanded=False):
                            for i, doc in enumerate(resp_data["expertA_retrieved_docs"][:3], 1):
                                st.markdown(f"**{i}.** {doc[:200]}...")
            
            with col2:
                if resp_data["expertB_arguments"]:
                    arg_data = resp_data["expertB_arguments"][0]
                    arg = DebateArgument(**arg_data) if isinstance(arg_data, dict) else arg_data
                    render_expert_argument("B", "Expert B (WebMD)", arg, "#fff3e0")
                    
                    # Show retrieved docs if available
                    if resp_data.get("expertB_retrieved_docs"):
                        with st.expander("WebMD Sources", expanded=False):
                            for i, doc in enumerate(resp_data["expertB_retrieved_docs"][:3], 1):
                                st.markdown(f"**{i}.** {doc[:200]}...")
        
        st.markdown("---")
        
        # Round 2 Arguments
        with st.expander("Round 2: Counter-Arguments", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                if len(resp_data["expertA_arguments"]) > 1:
                    arg_data = resp_data["expertA_arguments"][1]
                    arg = DebateArgument(**arg_data) if isinstance(arg_data, dict) else arg_data
                    render_expert_argument("A", "Expert A (Mayo Clinic)", arg, "#e3f2fd")
            
            with col2:
                if len(resp_data["expertB_arguments"]) > 1:
                    arg_data = resp_data["expertB_arguments"][1]
                    arg = DebateArgument(**arg_data) if isinstance(arg_data, dict) else arg_data
                    render_expert_argument("B", "Expert B (WebMD)", arg, "#fff3e0")
        
        st.markdown("---")
        
        # Download options
        st.markdown("## Download Results")
        col1, col2 = st.columns(2)
        
        with col1:
            st.download_button(
                label="Download JSON",
                file_name=f"debate_analysis_{resp_data['request_id'][:8]}.json",
                mime="application/json",
                data=json.dumps(resp_data, indent=2),
                use_container_width=True
            )
        
        with col2:
            md_report = generate_markdown_report(resp_data)
            st.download_button(
                label="Download Report (MD)",
                file_name=f"debate_analysis_{resp_data['request_id'][:8]}.md",
                mime="text/markdown",
                data=md_report,
                use_container_width=True
            )


if __name__ == "__main__":
    main()
