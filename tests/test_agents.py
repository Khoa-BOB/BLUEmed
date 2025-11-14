import types
from langchain_core.messages import HumanMessage
from app.agents.expertA import expertA_node
from app.graph.chain import build_chain
from app.core.state import MedState


class FakeLLM:
    def __init__(self, replies):
        self.replies = replies
        self.i = 0

    def invoke(self, messages):
        # simulate a LangChain LLM result with .content
        content = self.replies[self.i]
        self.i += 1
        resp = types.SimpleNamespace(content=content)
        return resp


def test_expertA_node_writes_analysis():
    fake_llm = FakeLLM(["This is expert analysis."])

    state: MedState = {
        "messages": [HumanMessage(content="What is aspirin used for?")],
        "retrieved_context": "Aspirin is used for pain and inflammation.",
        "expert1_analysis": "",
    }

    new_state = expertA_node(state, llm=fake_llm)

    assert "expert1_analysis" in new_state
    assert new_state["expert1_analysis"] == "This is expert analysis."



def test_chain_runs_expert1_node():
    # This fake will be used by expertA_node and Judge_node inside the graph
    # Need two replies: one for expert1, one for judge
    fake_llm = FakeLLM(["Fake expert answer.", "Fake judge answer."])

    chain = build_chain(fake_llm)

    initial_state: MedState = {
        "messages": [HumanMessage(content="Tell me about aspirin.")],
        "retrieved_context": "Aspirin info here.",
        "expert1_analysis": "",
    }

    final_state = chain.invoke(initial_state)

    # Make sure the expert1 node ran and wrote to the state
    assert "expert1_analysis" in final_state
    assert final_state["expert1_analysis"] == "Fake expert answer."


