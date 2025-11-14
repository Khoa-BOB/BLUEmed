from app.core.state import MedState
from app.core.prompts import EXPERT_A_SYSTEM
from langchain_core.messages import SystemMessage, HumanMessage

def expertA_node(state: MedState, llm) -> dict:
    user_text = state["messages"][-1].content
    context = state["retrieved_context"]

    messages = [
        SystemMessage(content=EXPERT_A_SYSTEM.format(context=context)),
        HumanMessage(content=user_text),
    ]
    resp = llm.invoke(messages)
    return {"expert1_analysis": resp.content}
