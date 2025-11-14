# graph/chain.py
from functools import partial
from langgraph.graph import StateGraph, START, END

from app.core.state import MedState
from app.agents.expertA import expertA_node
from app.agents.judge import Judge_node
from llm.factory import build_llm


def build_chain(llm=None):
    if llm is None:
        llm = build_llm()  # only real LLM when running the app

    graph = StateGraph(MedState)

    # add nodes, injecting llm where needed
    graph.add_node("expert1", partial(expertA_node, llm=llm))
    graph.add_node("judge", partial(Judge_node, llm=llm))

    # add edges
    graph.add_edge(START, "retriever")
    graph.add_edge("retriever", "expert1")
    graph.add_edge("expert1", "judge")
    graph.add_edge("judge", END)

    return graph.compile()
