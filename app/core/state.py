from typing import TypedDict, List, Optional
from langgraph.graph.message import AnyMessage

class MedState(TypedDict):
    messages: List[AnyMessage]
    retrieved_context: str
    expert1_analysis: Optional[str]
    expert2_analysis: Optional[str]
    judge_decision: Optional[str]