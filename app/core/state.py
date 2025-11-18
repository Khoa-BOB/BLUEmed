from typing import TypedDict, List, Optional, Annotated
from langgraph.graph.message import AnyMessage, add_messages

class DebateArgument(TypedDict):
    """Represents a single argument in the debate."""
    round: int
    content: str

class MedState(TypedDict):
    """State for multi-agent debate system."""
    messages: Annotated[List[AnyMessage], add_messages]
    medical_note: str  # The medical note to analyze

    # Debate tracking
    current_round: int
    max_rounds: int

    # Expert A (Mayo Clinic) arguments
    expertA_arguments: List[DebateArgument]
    expertA_retrieved_docs: List[str]

    # Expert B (WebMD) arguments
    expertB_arguments: List[DebateArgument]
    expertB_retrieved_docs: List[str]

    # Judge decision
    judge_decision: Optional[str]
    final_answer: Optional[str]  # CORRECT or INCORRECT