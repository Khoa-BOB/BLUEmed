from typing import List, Optional, Dict, Literal
from pydantic import BaseModel, Field


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
    final_answer: Literal["CORRECT", "INCORRECT", "UNKNOWN"] = Field(..., description="Final classification")
    confidence_score: Optional[int] = Field(None, ge=1, le=10, description="Confidence level 1-10")
    winner: Optional[Literal["Expert A", "Expert B", "Tie"]] = Field(None, description="Which expert made the better case")
    reasoning: str = Field(..., description="Detailed explanation of the decision")


class DebateResponse(BaseModel):
    """Complete response from the debate system."""
    request_id: str = Field(..., description="Unique identifier for this debate")
    medical_note: str = Field(..., description="The medical note that was analyzed")
    
    # Expert arguments
    expertA_arguments: List[DebateArgument] = Field(default_factory=list, description="Mayo Clinic expert arguments")
    expertB_arguments: List[DebateArgument] = Field(default_factory=list, description="WebMD expert arguments")
    
    # Retrieved documents (optional)
    expertA_retrieved_docs: List[str] = Field(default_factory=list, description="Mayo Clinic sources used")
    expertB_retrieved_docs: List[str] = Field(default_factory=list, description="WebMD sources used")
    
    # Judge decision
    judge_decision: JudgeDecision
    
    # Metadata
    model_info: Dict[str, str] = Field(default_factory=dict, description="Model versions used")
    raw: Optional[dict] = Field(None, description="Raw backend response")