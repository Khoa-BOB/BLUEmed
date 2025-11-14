from langgraph.graph import StateGraph, END
from app.core.state import MedState
# from app.rag.retrieve_node import retrieve_node, init_retriever
from app.agents.expertA import expertA_node
from app.agents.expertB import expertB_node
from app.agents.judge import Judge_node
from llm.factory import build_llm # your helper

def build_graph(config) -> StateGraph:
    llm_expert = build_llm(model_name=config.EXPERT_MODEL)
    llm_judge = build_llm(model_name=config.JUDGE_MODEL)

    # init_retriever(config.VECTORDIR)

    builder = StateGraph(MedState)

    # Wrap nodes with LLMs injected
    from functools import partial
    # builder.add_node("retrieve", retrieve_node)
    builder.add_node("expertA", partial(expertA_node, llm=llm_expert))
    builder.add_node("expertB", partial(expertB_node, llm=llm_expert))
    builder.add_node("judge", partial(Judge_node, llm=llm_judge))

    builder.set_entry_point("retrieve")
    builder.add_edge("retrieve", "expert1")
    builder.add_edge("retrieve", "expert2")
    builder.add_edge("expert1", "judge")
    builder.add_edge("expert2", "judge")
    builder.add_edge("judge", END)

    return builder.compile()