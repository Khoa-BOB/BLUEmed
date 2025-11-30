import os
import sys
import json
import uuid
from typing import Optional
from datetime import datetime
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import streamlit as st
from langchain_core.messages import HumanMessage

from models import DebateRequest, DebateResponse, DebateArgument, JudgeDecision
from utils import generate_markdown_report

# Import the same components as main.py
from config.settings import settings
from app.graph.graph import build_graph

APP_TITLE = "BLUEmed — Medical Note Analysis System"


def extract_judge_decision(judge_decision_raw: str) -> dict:
    """
    Extract structured information from judge's raw decision.
    (Same function as in main.py)

    Args:
        judge_decision_raw: Raw judge decision string

    Returns:
        Dictionary with extracted fields
    """
    import re

    extracted = {
        "final_answer": "UNKNOWN",
        "confidence_score": None,
        "winner": None,
        "reasoning": None
    }

    if not judge_decision_raw:
        return extracted

    # Try to parse JSON from judge decision
    decision_json = {}

    def fix_json_string(s: str) -> str:
        """Fix common JSON issues like unescaped newlines in string values."""
        import re
        # Find content between quotes and escape newlines
        def escape_newlines_in_strings(match):
            content = match.group(0)
            # Replace actual newlines with \n (but not already escaped ones)
            content = content.replace('\r\n', '\\n').replace('\r', '\\n').replace('\n', '\\n')
            return content
        # Match JSON string values (between quotes, handling escaped quotes)
        fixed = re.sub(r'"(?:[^"\\]|\\.)*"', escape_newlines_in_strings, s, flags=re.DOTALL)
        return fixed

    # Method 1: Try direct JSON parse first
    try:
        decision_json = json.loads(judge_decision_raw)
    except json.JSONDecodeError:
        # Try with fixed JSON
        try:
            fixed_json = fix_json_string(judge_decision_raw)
            decision_json = json.loads(fixed_json)
        except json.JSONDecodeError:
            pass

    # Method 2: Extract JSON from markdown code blocks (```json ... ```)
    if not decision_json:
        markdown_match = re.search(r'```(?:json)?\s*(\{.*\})\s*```', judge_decision_raw, re.DOTALL)
        if markdown_match:
            json_str = markdown_match.group(1)
            try:
                decision_json = json.loads(json_str)
            except json.JSONDecodeError:
                # Try with fixed JSON
                try:
                    fixed_json = fix_json_string(json_str)
                    decision_json = json.loads(fixed_json)
                except json.JSONDecodeError:
                    pass

    # Method 3: Find any JSON object containing "Final Answer"
    if not decision_json:
        json_match = re.search(r'(\{[^{}]*"Final Answer"[^{}]*\})', judge_decision_raw, re.DOTALL)
        if json_match:
            try:
                decision_json = json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass

    # Method 4: Greedy match for any JSON object
    if not decision_json:
        json_match = re.search(r'\{.*\}', judge_decision_raw, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            try:
                decision_json = json.loads(json_str)
            except json.JSONDecodeError:
                # Try with fixed JSON
                try:
                    fixed_json = fix_json_string(json_str)
                    decision_json = json.loads(fixed_json)
                except json.JSONDecodeError:
                    pass

    # Extract fields (handle different key formats)
    extracted["final_answer"] = (
        decision_json.get("Final Answer") or
        decision_json.get("final_answer") or
        decision_json.get("final answer") or
        "UNKNOWN"
    )

    # Extract confidence score
    conf_score = (
        decision_json.get("Confidence Score") or
        decision_json.get("confidence_score") or
        decision_json.get("confidence score") or
        decision_json.get("Confidence") or
        decision_json.get("confidence")
    )

    if conf_score is not None:
        try:
            extracted["confidence_score"] = int(conf_score)
        except (ValueError, TypeError):
            pass

    # Extract winner
    extracted["winner"] = (
        decision_json.get("Winner") or
        decision_json.get("winner")
    )

    # Extract reasoning
    extracted["reasoning"] = (
        decision_json.get("Reasoning") or
        decision_json.get("reasoning")
    )

    return extracted


def save_debate_log(medical_note: str, result: dict, output_dir: str = "logs/debates"):
    """
    Save debate results to a JSON file for analysis and auditing.
    (Same function as in main.py)

    Args:
        medical_note: The input medical note
        result: The graph execution result containing all arguments and decisions
        output_dir: Directory to save logs
    """
    # Create logs directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"debate_{timestamp}.json"
    filepath = Path(output_dir) / filename

    # Helper function to convert document objects to dictionaries
    def doc_to_dict(doc):
        """Convert a document object to a JSON-serializable dictionary."""
        return {
            "metadata": doc.metadata if hasattr(doc, 'metadata') else {},
            "content": doc.page_content if hasattr(doc, 'page_content') else str(doc)
        }

    # Extract judge decision into separate fields
    judge_decision_raw = result.get("judge_decision", "")
    judge_info = extract_judge_decision(judge_decision_raw)

    # Build JSON structure
    log_data = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "max_rounds": result.get("max_rounds", 2),
        "medical_note": medical_note,
        "expert_a": {
            "source": "Mayo Clinic",
            "arguments": result.get("expertA_arguments", []),
            "retrieved_docs": [doc_to_dict(doc) for doc in result.get("expertA_retrieved_docs", [])]
        },
        "expert_b": {
            "source": "WebMD",
            "arguments": result.get("expertB_arguments", []),
            "retrieved_docs": [doc_to_dict(doc) for doc in result.get("expertB_retrieved_docs", [])]
        },
        # Judge decision - separate key-value pairs
        "final_answer": judge_info["final_answer"],
        "confidence_score": judge_info["confidence_score"],
        "winner": judge_info["winner"],
        "reasoning": judge_info["reasoning"],
        "system_metadata": {
            "use_retriever": settings.USE_RETRIEVER,
            "embedding_model": settings.EMBEDDING_MODEL,
            "expert_model": settings.EXPERT_MODEL,
            "judge_model": settings.JUDGE_MODEL
        }
    }

    # Save to JSON file
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(log_data, f, indent=2, ensure_ascii=False)

    return filepath


def init_session_state():
    """Initialize session state variables."""
    st.session_state.setdefault("graph", None)
    st.session_state.setdefault("last_response", None)
    st.session_state.setdefault("history", [])


def sidebar_controls():
    """Render sidebar with configuration controls."""
    st.sidebar.header("System Information")

    # Show model configuration
    st.sidebar.markdown("### Models")
    st.sidebar.text(f"Expert: {settings.EXPERT_MODEL}")
    st.sidebar.text(f"Judge: {settings.JUDGE_MODEL}")
    st.sidebar.text(f"Retriever: {'Enabled' if settings.USE_RETRIEVER else 'Disabled'}")

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


@st.cache_resource
def get_debate_graph():
    """Build and cache the debate graph (same as main.py)."""
    return build_graph(settings)


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
            # Get the debate graph (same as main.py)
            graph = get_debate_graph()

            # Create initial state (same as main.py)
            initial_state = {
                "messages": [HumanMessage(content="Analyze this medical note for errors")],
                "medical_note": medical_note.strip(),
                "current_round": 1,
                "max_rounds": 2,
                "expertA_arguments": [],
                "expertA_retrieved_docs": [],
                "expertB_arguments": [],
                "expertB_retrieved_docs": [],
                "judge_decision": None,
                "final_answer": None
            }

            # Run the debate (same as main.py)
            with st.spinner("Running debate analysis... This may take a moment."):
                try:
                    # Run the graph
                    result = graph.invoke(initial_state)

                    # Extract judge decision (same as main.py)
                    judge_decision_raw = result.get("judge_decision", "")
                    judge_info = extract_judge_decision(judge_decision_raw)

                    # IMPORTANT: Use final_answer from result (which includes safety layer override)
                    # NOT from judge_info (which is just the raw judge response)
                    final_answer_with_safety = result.get("final_answer", judge_info["final_answer"])

                    # Format retrieved docs for display
                    def format_retrieved_docs(docs):
                        formatted = []
                        for doc in docs:
                            if hasattr(doc, 'page_content'):
                                content = doc.page_content
                            else:
                                content = str(doc)
                            formatted.append(content)
                        return formatted

                    # Build response (matching the expected structure)
                    response_data = {
                        "request_id": str(uuid.uuid4()),
                        "medical_note": medical_note.strip(),
                        "expertA_arguments": result.get("expertA_arguments", []),
                        "expertB_arguments": result.get("expertB_arguments", []),
                        "expertA_retrieved_docs": format_retrieved_docs(result.get("expertA_retrieved_docs", [])),
                        "expertB_retrieved_docs": format_retrieved_docs(result.get("expertB_retrieved_docs", [])),
                        "judge_decision": {
                            "final_answer": final_answer_with_safety,  # Use safety layer result!
                            "confidence_score": judge_info["confidence_score"],
                            "winner": judge_info["winner"],
                            "reasoning": judge_info["reasoning"] or judge_decision_raw
                        },
                        "model_info": {
                            "expert": settings.EXPERT_MODEL,
                            "judge": settings.JUDGE_MODEL,
                            "retriever_enabled": str(settings.USE_RETRIEVER)
                        }
                    }

                    # Save debate log (same as main.py)
                    log_path = save_debate_log(medical_note.strip(), result)

                    st.session_state["last_response"] = response_data
                    st.session_state["history"].append(response_data)
                    st.success(f"Analysis complete! Log saved to: {log_path}")
                except Exception as e:
                    st.error(f"Error during analysis: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
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
