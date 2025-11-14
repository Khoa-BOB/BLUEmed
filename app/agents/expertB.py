from app.core.state import MedState
from app.core.prompts import EXPERT1_SYSTEM
from langchain_core.messages import SystemMessage, HumanMessage

def expertB_node(state: MedState, llm) -> MedState:
    user_text = state["messages"][-1].content
    context = state["retrieved_context"]

    messages = [
        SystemMessage(content=EXPERT1_SYSTEM.format(context=context)),
        HumanMessage(content=user_text),
    ]
    resp = llm.invoke(messages)
    state["expert1_analysis"] = resp.content
    return state
