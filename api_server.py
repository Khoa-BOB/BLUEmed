"""
FastAPI server for BLUEmed debate system.
Provides HTTP API for the Streamlit UI to communicate with the backend.

Usage:
    uvicorn api_server:app --host 0.0.0.0 --port 8000 --reload

Environment variables:
    API_KEY: Optional authentication key for API security
"""

import os
import uuid
from typing import Optional
from datetime import datetime

from fastapi import FastAPI, HTTPException, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from config.settings import settings
from app.graph.graph import build_graph
from langchain_core.messages import HumanMessage


# =====================================================================
# MODELS (matching UI expectations)
# =====================================================================

class DebateArgument(BaseModel):
    """Represents a single argument in a debate round."""
    round: int = Field(..., description="Round number (1 or 2)")
    content: str = Field(..., description="The argument text")


class DebateRequest(BaseModel):
    """Request to run the debate system on a medical note."""
    medical_note: str = Field(..., description="The medical note to analyze")
    max_rounds: int = Field(default=2, description="Maximum number of debate rounds")


class JudgeDecision(BaseModel):
    """Judge's final decision after evaluating the debate."""
    final_answer: str = Field(..., description="Final classification (CORRECT/INCORRECT/UNKNOWN)")
    confidence_score: Optional[int] = Field(None, ge=1, le=10, description="Confidence level 1-10")
    winner: Optional[str] = Field(None, description="Which expert made the better case")
    reasoning: str = Field(..., description="Detailed explanation of the decision")


class DebateResponse(BaseModel):
    """Complete response from the debate system."""
    request_id: str = Field(..., description="Unique identifier for this debate")
    medical_note: str = Field(..., description="The medical note that was analyzed")
    
    # Expert arguments
    expertA_arguments: list[DebateArgument] = Field(default_factory=list)
    expertB_arguments: list[DebateArgument] = Field(default_factory=list)
    
    # Retrieved documents (optional)
    expertA_retrieved_docs: list[str] = Field(default_factory=list)
    expertB_retrieved_docs: list[str] = Field(default_factory=list)
    
    # Judge decision
    judge_decision: JudgeDecision
    
    # Metadata
    model_info: dict[str, str] = Field(default_factory=dict)


# =====================================================================
# FASTAPI APP
# =====================================================================

app = FastAPI(
    title="BLUEmed API",
    description="Multi-agent medical note analysis system",
    version="2.0.0"
)

# CORS middleware for Streamlit
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Optional API key authentication
API_KEY = os.getenv("API_KEY", None)


def verify_api_key(authorization: Optional[str] = Header(None)) -> bool:
    """Verify API key if configured."""
    if API_KEY is None:
        return True  # No authentication required
    
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing authorization header")
    
    # Expected format: "Bearer <key>"
    try:
        scheme, token = authorization.split()
        if scheme.lower() != "bearer":
            raise HTTPException(status_code=401, detail="Invalid authentication scheme")
        if token != API_KEY:
            raise HTTPException(status_code=401, detail="Invalid API key")
    except ValueError:
        raise HTTPException(status_code=401, detail="Invalid authorization header format")
    
    return True


# Initialize graph once at startup
print("\n" + "="*80)
print("INITIALIZING BLUEMED API SERVER")
print("="*80)
print("\nBuilding debate graph...")
debate_graph = build_graph(settings)
print("✓ Graph built successfully!")
print("\n" + "="*80 + "\n")


# =====================================================================
# HELPER FUNCTIONS
# =====================================================================

def extract_judge_decision(judge_decision_raw: str) -> dict:
    """
    Extract structured information from judge's raw decision.
    Handles JSON and plain text formats with robust parsing.
    """
    import json
    import re
    
    extracted = {
        "final_answer": "UNKNOWN",
        "confidence_score": None,
        "winner": None,
        "reasoning": judge_decision_raw or "No reasoning provided."
    }
    
    if not judge_decision_raw:
        return extracted
    
    # Helper to fix common JSON issues
    def fix_json_string(s: str) -> str:
        """Fix common JSON formatting issues."""
        # Remove markdown code blocks if present
        s = re.sub(r'```json\s*', '', s)
        s = re.sub(r'```\s*$', '', s)
        # Replace literal newlines in strings
        lines = s.split('\n')
        fixed_lines = []
        in_string = False
        for line in lines:
            # Count unescaped quotes to detect string context
            quote_count = len(re.findall(r'(?<!\\)"', line))
            if quote_count % 2 == 1:
                in_string = not in_string
            if in_string and fixed_lines:
                # Append to previous line with escaped newline
                fixed_lines[-1] = fixed_lines[-1] + '\\n' + line.strip()
            else:
                fixed_lines.append(line)
        return '\n'.join(fixed_lines)
    
    # Try to parse JSON with multiple attempts
    decision_json = None
    
    # Method 1: Direct JSON parse
    try:
        decision_json = json.loads(judge_decision_raw)
    except json.JSONDecodeError:
        pass
    
    # Method 2: Try fixing common issues and parse again
    if decision_json is None:
        try:
            fixed = fix_json_string(judge_decision_raw)
            decision_json = json.loads(fixed)
        except json.JSONDecodeError:
            pass
    
    # Method 3: Extract JSON from text (e.g., if wrapped in markdown)
    if decision_json is None:
        json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', judge_decision_raw, re.DOTALL)
        if json_match:
            try:
                decision_json = json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass
    
    # If we successfully parsed JSON, extract fields
    if decision_json:
        # Extract final answer
        final_answer_raw = (
            decision_json.get("Final Answer") or
            decision_json.get("final_answer") or
            decision_json.get("final answer") or
            "UNKNOWN"
        )
        # Clean up and validate
        if isinstance(final_answer_raw, str):
            final_answer_clean = final_answer_raw.strip().upper()
            # Only accept valid values
            if final_answer_clean in ["CORRECT", "INCORRECT", "UNKNOWN"]:
                extracted["final_answer"] = final_answer_clean
        
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
                conf_int = int(conf_score)
                if 1 <= conf_int <= 10:
                    extracted["confidence_score"] = conf_int
            except (ValueError, TypeError):
                pass
        
        # Extract winner
        winner_raw = (
            decision_json.get("Winner") or
            decision_json.get("winner")
        )
        # Clean up and validate
        if isinstance(winner_raw, str):
            winner_clean = winner_raw.strip()
            # Only accept valid values
            if winner_clean in ["Expert A", "Expert B", "Tie"]:
                extracted["winner"] = winner_clean
        
        # Extract reasoning
        extracted["reasoning"] = (
            decision_json.get("Reasoning") or
            decision_json.get("reasoning") or
            judge_decision_raw
        )
        
        return extracted
    
    # Fallback: extract from plain text using regex
    print("⚠️  JSON parsing failed, using regex fallback")
    
    # Extract final answer
    final_answer_match = re.search(r'Final Answer[:\s]*([A-Z]+)', judge_decision_raw, re.IGNORECASE)
    if final_answer_match:
        answer = final_answer_match.group(1).upper()
        if answer in ["CORRECT", "INCORRECT", "UNKNOWN"]:
            extracted["final_answer"] = answer
    
    # Extract confidence score
    conf_match = re.search(r'Confidence[:\s]*(\d+)', judge_decision_raw, re.IGNORECASE)
    if conf_match:
        try:
            conf = int(conf_match.group(1))
            if 1 <= conf <= 10:
                extracted["confidence_score"] = conf
        except ValueError:
            pass
    
    # Extract winner
    winner_match = re.search(r'Winner[:\s]*([^.\n]+)', judge_decision_raw, re.IGNORECASE)
    if winner_match:
        winner = winner_match.group(1).strip()
        if winner in ["Expert A", "Expert B", "Tie"]:
            extracted["winner"] = winner
    
    return extracted


def format_retrieved_docs(docs: list) -> list[str]:
    """Format retrieved documents for UI display."""
    formatted = []
    for doc in docs:
        if hasattr(doc, 'page_content'):
            content = doc.page_content
        else:
            content = str(doc)
        
        # Truncate long documents
        if len(content) > 500:
            content = content[:500] + "..."
        
        formatted.append(content)
    
    return formatted


# =====================================================================
# API ENDPOINTS
# =====================================================================

@app.get("/")
def root():
    """Root endpoint with API information."""
    return {
        "service": "BLUEmed API",
        "version": "2.0.0",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "analyze": "/analyze (POST)"
        }
    }


@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "models": {
            "expert": settings.EXPERT_MODEL,
            "judge": settings.JUDGE_MODEL
        },
        "retriever_enabled": settings.USE_RETRIEVER
    }


@app.post("/analyze", response_model=DebateResponse)
def analyze_medical_note(
    request: DebateRequest,
    _auth: bool = Depends(verify_api_key)
) -> DebateResponse:
    """
    Analyze a medical note using the multi-agent debate system.
    
    Args:
        request: Medical note and configuration
        
    Returns:
        Complete debate results with expert arguments and judge decision
    """
    
    print(f"\n{'='*80}")
    print(f"NEW ANALYSIS REQUEST")
    print(f"{'='*80}")
    print(f"Medical note length: {len(request.medical_note)} characters")
    print(f"Max rounds: {request.max_rounds}")
    
    try:
        # Create initial state
        initial_state = {
            "messages": [HumanMessage(content="Analyze this medical note for errors")],
            "medical_note": request.medical_note,
            "current_round": 1,
            "max_rounds": request.max_rounds,
            "expertA_arguments": [],
            "expertA_retrieved_docs": [],
            "expertB_arguments": [],
            "expertB_retrieved_docs": [],
            "judge_decision": None,
            "final_answer": None
        }
        
        # Run the debate graph
        print("\nRunning debate...")
        result = debate_graph.invoke(initial_state)
        print("✓ Debate complete")
        
        # Extract judge decision
        judge_decision_raw = result.get("judge_decision", "")
        print(f"\n📋 Raw judge decision (first 500 chars):")
        print(f"{judge_decision_raw[:500]}...")
        
        judge_info = extract_judge_decision(judge_decision_raw)
        
        print(f"\n✓ Extracted judge info:")
        print(f"  Final Answer: {judge_info['final_answer']}")
        print(f"  Confidence: {judge_info['confidence_score']}")
        print(f"  Winner: {judge_info['winner']}")
        
        # Build response
        response = DebateResponse(
            request_id=str(uuid.uuid4()),
            medical_note=request.medical_note,
            expertA_arguments=[
                DebateArgument(round=arg["round"], content=arg["content"])
                for arg in result.get("expertA_arguments", [])
            ],
            expertB_arguments=[
                DebateArgument(round=arg["round"], content=arg["content"])
                for arg in result.get("expertB_arguments", [])
            ],
            expertA_retrieved_docs=format_retrieved_docs(result.get("expertA_retrieved_docs", [])),
            expertB_retrieved_docs=format_retrieved_docs(result.get("expertB_retrieved_docs", [])),
            judge_decision=JudgeDecision(
                final_answer=judge_info["final_answer"],
                confidence_score=judge_info["confidence_score"],
                winner=judge_info["winner"],
                reasoning=judge_info["reasoning"]
            ),
            model_info={
                "expert": settings.EXPERT_MODEL,
                "judge": settings.JUDGE_MODEL,
                "retriever_enabled": str(settings.USE_RETRIEVER)
            }
        )
        
        print(f"✓ Response prepared: {response.judge_decision.final_answer}")
        print(f"{'='*80}\n")
        
        return response
        
    except Exception as e:
        print(f"\n❌ Error during analysis: {str(e)}")
        print(f"{'='*80}\n")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("PORT", 8000))
    print(f"\n🚀 Starting BLUEmed API server on port {port}...")
    
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=port,
        reload=True,
        log_level="info"
    )
