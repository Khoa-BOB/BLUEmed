from app.core.state import MedState
from app.core.prompts import JUDGE_SYSTEM
from langchain_core.messages import SystemMessage, HumanMessage

def Judge_node(state: MedState, llm) -> MedState:
    user_text = state["messages"][-1].content
    context = state["retrieved_context"]

    messages = [
        SystemMessage(content=JUDGE_SYSTEM.format(context=context)),
        HumanMessage(content=user_text),
    ]
    resp = llm.invoke(messages)
    state["judge_decision"] = resp.content
    return state
