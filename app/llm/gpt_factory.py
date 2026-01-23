"""
OpenAI GPT API LLM factory.
"""
from langchain_openai import ChatOpenAI
from config.settings import settings


def build_gpt_llm(model_name: str = "gpt-4o-mini", temperature: float = 0.2):
    """
    Build OpenAI GPT LLM.

    Args:
        model_name: OpenAI model name
            - gpt-4o (best quality, most capable)
            - gpt-4o-mini (fast, cost-effective)
            - gpt-4-turbo (high quality, faster than GPT-4)
            - gpt-3.5-turbo (fastest, most economical)
        temperature: Sampling temperature (0.0-2.0)
            - 0.0: Deterministic, focused responses
            - 0.2-0.7: Balanced creativity and consistency
            - 1.0+: More creative and diverse outputs

    Returns:
        LangChain OpenAI LLM

    Setup:
        1. Get API key from: https://platform.openai.com/api-keys
        2. Set in .env: OPENAI_API_KEY=your_key_here
        
    Usage Example:
        from app.llm.gpt_factory import build_gpt_llm
        llm = build_gpt_llm("gpt-4o-mini", temperature=0.2)
        response = llm.invoke(messages)
    """

    api_key = settings.OPENAI_API_KEY
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY not found in environment. "
            "Get one from https://platform.openai.com/api-keys"
        )

    llm = ChatOpenAI(
        model=model_name,
        openai_api_key=api_key,
        temperature=temperature,
        max_tokens=1024,  # Maximum tokens in response
        timeout=60,  # Request timeout in seconds
        max_retries=2,  # Retry failed requests
    )

    return llm
