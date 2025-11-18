from langchain_ollama import ChatOllama
from config.settings import settings

def build_llm(model_name: str, temperature: float = 0.7):
    """
    Build an LLM from the model name.
    Supports: Ollama, Google Gemini

    Args:
        model_name: The name or path of the model
            - For Ollama: "llama3.1:8b" or full path
            - For Gemini: "gemini" to use Gemini API
        temperature: Sampling temperature for generation

    Returns:
        LLM instance (ChatOllama or ChatGoogleGenerativeAI)
    """
    # Check if using Gemini
    if model_name.lower().startswith("gemini"):
        #Import the factory
        from app.llm.gemini_factory import build_gemini_llm
        
        return build_gemini_llm(settings.GEMINI_MODEL, temperature)

    # Default to Ollama
    return ChatOllama(
        model=model_name,
        temperature=temperature,
        top_p=0.9,
        repeat_penalty=1.05,
    )