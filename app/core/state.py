from typing import TypedDict, List, Optional, Annotated
from langgraph.graph.message import AnyMessage, add_messages

class MedState(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]
    retrieved_context: str
    expert1_analysis: Optional[str]
    expert2_analysis: Optional[str]
    judge_decision: Optional[str]