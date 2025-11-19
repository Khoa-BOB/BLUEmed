from typing import Dict, Any


def generate_markdown_report(resp_data: Dict[str, Any]) -> str:
    """Generate a formatted markdown report from the debate response."""
    
    md = []
    
    # Header
    md.append(f"# BLUEmed Debate Analysis Report\n")
    md.append(f"**Request ID:** {resp_data['request_id']}\n")
    md.append("---\n\n")
    
    # Medical Note
    md.append("## 📋 Medical Note\n")
    md.append(f"{resp_data['medical_note']}\n\n")
    md.append("---\n\n")
    
    # Round 1
    md.append("## 🥊 Round 1: Initial Arguments\n\n")
    
    # Expert A Round 1
    if resp_data.get("expertA_arguments") and len(resp_data["expertA_arguments"]) > 0:
        arg = resp_data["expertA_arguments"][0]
        md.append("### 👨‍⚕️ Expert A (Mayo Clinic)\n")
        md.append(f"**Round {arg['round']}**\n\n")
        md.append(f"{arg['content']}\n\n")
        
        # Add sources if available
        if resp_data.get("expertA_retrieved_docs"):
            md.append("**Sources:**\n")
            for i, doc in enumerate(resp_data["expertA_retrieved_docs"], 1):
                md.append(f"{i}. {doc[:200]}...\n")
            md.append("\n")
    
    # Expert B Round 1
    if resp_data.get("expertB_arguments") and len(resp_data["expertB_arguments"]) > 0:
        arg = resp_data["expertB_arguments"][0]
        md.append("### 👩‍⚕️ Expert B (WebMD)\n")
        md.append(f"**Round {arg['round']}**\n\n")
        md.append(f"{arg['content']}\n\n")
        
        # Add sources if available
        if resp_data.get("expertB_retrieved_docs"):
            md.append("**Sources:**\n")
            for i, doc in enumerate(resp_data["expertB_retrieved_docs"], 1):
                md.append(f"{i}. {doc[:200]}...\n")
            md.append("\n")
    
    md.append("---\n\n")
    
    # Round 2
    md.append("## 🥊 Round 2: Counter-Arguments\n\n")
    
    # Expert A Round 2
    if resp_data.get("expertA_arguments") and len(resp_data["expertA_arguments"]) > 1:
        arg = resp_data["expertA_arguments"][1]
        md.append("### 👨‍⚕️ Expert A (Mayo Clinic)\n")
        md.append(f"**Round {arg['round']}**\n\n")
        md.append(f"{arg['content']}\n\n")
    
    # Expert B Round 2
    if resp_data.get("expertB_arguments") and len(resp_data["expertB_arguments"]) > 1:
        arg = resp_data["expertB_arguments"][1]
        md.append("### 👩‍⚕️ Expert B (WebMD)\n")
        md.append(f"**Round {arg['round']}**\n\n")
        md.append(f"{arg['content']}\n\n")
    
    md.append("---\n\n")
    
    # Judge Decision
    md.append("## 🏛️ Judge's Final Decision\n\n")
    
    if resp_data.get("judge_decision"):
        decision = resp_data["judge_decision"]
        
        final_answer = decision.get("final_answer", "UNKNOWN")
        icon = "✅" if final_answer == "CORRECT" else "❌" if final_answer == "INCORRECT" else "❓"
        
        md.append(f"### {icon} Final Answer: **{final_answer}**\n\n")
        
        if decision.get("confidence_score"):
            md.append(f"**Confidence Score:** {decision['confidence_score']}/10\n\n")
        
        if decision.get("winner"):
            md.append(f"**Winner:** {decision['winner']}\n\n")
        
        md.append("### 📝 Reasoning\n\n")
        md.append(f"{decision.get('reasoning', 'No reasoning provided.')}\n\n")
    
    md.append("---\n\n")
    
    # Model Info
    if resp_data.get("model_info"):
        md.append("## ℹ️ System Information\n\n")
        for key, value in resp_data["model_info"].items():
            md.append(f"- **{key.title()}:** {value}\n")
        md.append("\n")
    
    return "".join(md)