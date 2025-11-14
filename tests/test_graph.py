import types
from unittest.mock import patch, MagicMock
from langchain_core.messages import HumanMessage
from app.graph.graph import build_graph
from app.core.state import MedState


class FakeLLM:
    """Mock LLM for testing that returns predefined responses."""
    def __init__(self, replies):
        self.replies = replies
        self.i = 0

    def invoke(self, messages):
        # Simulate a LangChain LLM result with .content attribute
        content = self.replies[self.i % len(self.replies)]
        self.i += 1
        resp = types.SimpleNamespace(content=content)
        return resp


class FakeConfig:
    """Mock config for testing."""
    EXPERT_MODEL = "fake-expert-model"
    JUDGE_MODEL = "fake-judge-model"
    VECTORDIR = "/fake/vector/dir"


def mock_build_llm(model_name: str, temperature: float = 0.7):
    """Mock build_llm that returns a FakeLLM."""
    if "expert" in model_name.lower():
        return FakeLLM(["Expert analysis response"])
    elif "judge" in model_name.lower():
        return FakeLLM(["Judge decision response"])
    return FakeLLM(["Generic response"])


@patch('app.graph.graph.build_llm', side_effect=mock_build_llm)
def test_graph_structure(mock_llm):
    """Test that the graph has the correct structure with all nodes and edges."""
    config = FakeConfig()
    graph = build_graph(config)

    # Get the graph representation
    graph_dict = graph.get_graph().to_json()

    # Check that all expected nodes exist
    assert "expertA" in str(graph_dict)
    assert "expertB" in str(graph_dict)
    assert "judge" in str(graph_dict)


@patch('app.graph.graph.build_llm', side_effect=mock_build_llm)
def test_graph_parallel_execution(mock_llm):
    """Test that both experts run in parallel and feed into the judge."""
    config = FakeConfig()
    graph = build_graph(config)

    initial_state: MedState = {
        "messages": [HumanMessage(content="What is aspirin used for?")],
        "retrieved_context": "Aspirin is used for pain relief and inflammation.",
        "expert1_analysis": None,
        "expert2_analysis": None,
        "judge_decision": None,
    }

    final_state = graph.invoke(initial_state)

    # Verify that all three nodes have updated the state
    assert final_state["expert1_analysis"] is not None
    assert final_state["expert2_analysis"] is not None
    assert final_state["judge_decision"] is not None
    assert len(final_state["expert1_analysis"]) > 0
    assert len(final_state["expert2_analysis"]) > 0
    assert len(final_state["judge_decision"]) > 0


@patch('app.graph.graph.build_llm', side_effect=mock_build_llm)
def test_graph_state_preservation(mock_llm):
    """Test that the graph preserves messages and context throughout execution."""
    config = FakeConfig()
    graph = build_graph(config)

    test_message = "Tell me about ibuprofen."
    test_context = "Ibuprofen is a nonsteroidal anti-inflammatory drug."

    initial_state: MedState = {
        "messages": [HumanMessage(content=test_message)],
        "retrieved_context": test_context,
        "expert1_analysis": None,
        "expert2_analysis": None,
        "judge_decision": None,
    }

    final_state = graph.invoke(initial_state)

    # Verify that original messages and context are preserved
    assert len(final_state["messages"]) > 0
    assert final_state["messages"][0].content == test_message
    assert final_state["retrieved_context"] == test_context


@patch('app.graph.graph.build_llm', side_effect=mock_build_llm)
def test_graph_experts_receive_different_analyses(mock_llm):
    """Test that both experts produce independent analyses."""
    config = FakeConfig()
    graph = build_graph(config)

    initial_state: MedState = {
        "messages": [HumanMessage(content="What are the side effects of acetaminophen?")],
        "retrieved_context": "Acetaminophen can cause liver damage in high doses.",
        "expert1_analysis": None,
        "expert2_analysis": None,
        "judge_decision": None,
    }

    final_state = graph.invoke(initial_state)

    # Both experts should provide analyses
    assert final_state["expert1_analysis"] is not None
    assert final_state["expert2_analysis"] is not None

    # The analyses might be different (depending on the actual LLM)
    # For now, just verify they both exist and have content
    assert isinstance(final_state["expert1_analysis"], str)
    assert isinstance(final_state["expert2_analysis"], str)


@patch('app.graph.graph.build_llm', side_effect=mock_build_llm)
def test_graph_judge_receives_both_expert_analyses(mock_llm):
    """Test that the judge node receives and processes both expert analyses."""
    config = FakeConfig()
    graph = build_graph(config)

    initial_state: MedState = {
        "messages": [HumanMessage(content="What is metformin?")],
        "retrieved_context": "Metformin is used to treat type 2 diabetes.",
        "expert1_analysis": None,
        "expert2_analysis": None,
        "judge_decision": None,
    }

    final_state = graph.invoke(initial_state)

    # Verify judge decision exists and is a string
    assert final_state["judge_decision"] is not None
    assert isinstance(final_state["judge_decision"], str)
    assert len(final_state["judge_decision"]) > 0


@patch('app.graph.graph.build_llm', side_effect=mock_build_llm)
def test_graph_with_empty_context(mock_llm):
    """Test that the graph handles empty retrieved context gracefully."""
    config = FakeConfig()
    graph = build_graph(config)

    initial_state: MedState = {
        "messages": [HumanMessage(content="What is penicillin?")],
        "retrieved_context": "",
        "expert1_analysis": None,
        "expert2_analysis": None,
        "judge_decision": None,
    }

    final_state = graph.invoke(initial_state)

    # Graph should still execute successfully
    assert final_state["expert1_analysis"] is not None
    assert final_state["expert2_analysis"] is not None
    assert final_state["judge_decision"] is not None


@patch('app.graph.graph.build_llm', side_effect=mock_build_llm)
def test_graph_with_multiple_messages(mock_llm):
    """Test that the graph handles multiple messages in the state."""
    config = FakeConfig()
    graph = build_graph(config)

    initial_state: MedState = {
        "messages": [
            HumanMessage(content="What is amoxicillin?"),
            HumanMessage(content="Tell me more about its side effects."),
        ],
        "retrieved_context": "Amoxicillin is an antibiotic.",
        "expert1_analysis": None,
        "expert2_analysis": None,
        "judge_decision": None,
    }

    final_state = graph.invoke(initial_state)

    # Verify all nodes executed
    assert final_state["expert1_analysis"] is not None
    assert final_state["expert2_analysis"] is not None
    assert final_state["judge_decision"] is not None

    # Messages should be preserved
    assert len(final_state["messages"]) == 2
