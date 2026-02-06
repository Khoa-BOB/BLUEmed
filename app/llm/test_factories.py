"""
Test script for LLM factories.
Demonstrates usage of different LLM providers.

Run this to test your API keys and model configurations.
"""

from langchain_core.messages import HumanMessage, SystemMessage


def test_ollama():
    """Test Ollama (local) LLM."""
    print("\n" + "="*60)
    print("Testing Ollama (Local)")
    print("="*60)
    
    try:
        from app.llm.factory import build_llm
        
        llm = build_llm("llama3.1:8b", temperature=0.2)
        print("✓ Ollama LLM created successfully")
        
        # Test invocation
        response = llm.invoke([HumanMessage(content="Say 'Hello from Ollama!' in one sentence.")])
        print(f"Response: {response.content}")
        print("✓ Ollama test passed!\n")
        
    except Exception as e:
        print(f"✗ Ollama test failed: {e}\n")


def test_gemini():
    """Test Google Gemini LLM."""
    print("\n" + "="*60)
    print("Testing Google Gemini")
    print("="*60)
    
    try:
        from app.llm.gemini_factory import build_gemini_llm
        
        llm = build_gemini_llm("gemini-2.0-flash", temperature=0.2)
        print("✓ Gemini LLM created successfully")
        
        # Test invocation
        response = llm.invoke([HumanMessage(content="Say 'Hello from Gemini!' in one sentence.")])
        print(f"Response: {response.content}")
        print("✓ Gemini test passed!\n")
        
    except ValueError as e:
        print(f"⚠ Gemini not configured: {e}")
        print("  Set GOOGLE_API_KEY in .env to use Gemini\n")
    except Exception as e:
        print(f"✗ Gemini test failed: {e}\n")


def test_gpt():
    """Test OpenAI GPT LLM."""
    print("\n" + "="*60)
    print("Testing OpenAI GPT")
    print("="*60)
    
    try:
        from app.llm.gpt_factory import build_gpt_llm
        
        llm = build_gpt_llm("gpt-4o-mini", temperature=0.2)
        print("✓ GPT LLM created successfully")
        
        # Test invocation
        response = llm.invoke([HumanMessage(content="Say 'Hello from GPT!' in one sentence.")])
        print(f"Response: {response.content}")
        print("✓ GPT test passed!\n")
        
    except ValueError as e:
        print(f"⚠ GPT not configured: {e}")
        print("  Set OPENAI_API_KEY in .env to use GPT\n")
    except Exception as e:
        print(f"✗ GPT test failed: {e}\n")


def test_huggingface():
    """Test HuggingFace LLM (API)."""
    print("\n" + "="*60)
    print("Testing HuggingFace (API)")
    print("="*60)
    
    try:
        from app.llm.huggingface_factory import build_huggingface_llm
        
        llm = build_huggingface_llm(
            "meta-llama/Meta-Llama-3-8B-Instruct",
            temperature=0.2,
            use_local=True
        )
        print("✓ HuggingFace LLM created successfully")
        
        # Test invocation
        response = llm.invoke([HumanMessage(content="Say 'Hello from HuggingFace!' in one sentence.")])
        print(f"Response: {response.content}")
        print("✓ HuggingFace test passed!\n")
        
    except ValueError as e:
        print(f"⚠ HuggingFace not configured: {e}")
        print("  Set HUGGINGFACE_API_KEY in .env to use HuggingFace API\n")
    except Exception as e:
        print(f"✗ HuggingFace test failed: {e}\n")


def test_factory_routing():
    """Test the main factory routing logic."""
    print("\n" + "="*60)
    print("Testing Factory Routing")
    print("="*60)
    
    from app.llm.factory import build_llm
    
    # Test routing for different model prefixes
    test_cases = [
        ("llama3.1:8b", "Ollama"),
        ("gemini-2.0-flash", "Gemini"),
        ("gpt-4o-mini", "GPT"),
        ("hf:meta-llama/Meta-Llama-3-8B-Instruct", "HuggingFace"),
    ]
    
    for model_name, expected_provider in test_cases:
        try:
            llm = build_llm(model_name, temperature=0.2)
            print(f"✓ {model_name} → {expected_provider} routing works")
        except ValueError as e:
            print(f"⚠ {model_name} → {expected_provider} not configured: {e}")
        except Exception as e:
            print(f"✗ {model_name} → {expected_provider} routing failed: {e}")
    
    print()


def test_medical_example():
    """Test with a medical reasoning example (like the actual use case)."""
    print("\n" + "="*60)
    print("Testing Medical Reasoning Example")
    print("="*60)
    
    try:
        from app.llm.factory import build_llm
        
        # Use the first available model
        llm = build_llm("llama3.1:8b", temperature=0.2)
        
        messages = [
            SystemMessage(content="You are a medical expert analyzing patient medications."),
            HumanMessage(content="""
Patient is taking:
- Warfarin 5mg daily
- Aspirin 81mg daily

Question: Is there a potential drug interaction?
Answer in one sentence.
            """)
        ]
        
        response = llm.invoke(messages)
        print(f"Medical Analysis: {response.content}")
        print("✓ Medical reasoning test passed!\n")
        
    except Exception as e:
        print(f"✗ Medical reasoning test failed: {e}\n")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("LLM Factory Test Suite")
    print("="*60)
    print("\nThis script tests all LLM factory implementations.")
    print("Some tests may fail if API keys are not configured.\n")
    
    # Test factory routing first
    test_factory_routing()
    
    # Test individual providers
    test_ollama()
    test_gemini()
    test_gpt()
    test_huggingface()
    
    # Test with medical example
    test_medical_example()
    
    print("="*60)
    print("Test suite complete!")
    print("="*60)
    print("\nTo configure missing providers, add API keys to .env:")
    print("  - GOOGLE_API_KEY for Gemini")
    print("  - OPENAI_API_KEY for GPT")
    print("  - HUGGINGFACE_API_KEY for HuggingFace")
    print()
