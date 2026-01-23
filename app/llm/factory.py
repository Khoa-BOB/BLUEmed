from langchain_ollama import ChatOllama
from config.settings import settings

def build_llm(model_name: str = None, temperature: float = 0.2):
    """
    Build an LLM from the model name.
    Supports: Ollama, Google Gemini, OpenAI GPT, HuggingFace

    Args:
        model_name: The name or path of the model
            - For Ollama: "llama3.1:8b" or full path (default)
            - For Gemini: "gemini-*" (e.g., "gemini-2.0-flash")
            - For GPT: "gpt-*" (e.g., "gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo")
            - For HuggingFace: "hf:model_id" (e.g., "hf:meta-llama/Meta-Llama-3-8B-Instruct")
            - If None, uses EXPERT_MODEL or JUDGE_MODEL from settings
        temperature: Sampling temperature for generation

    Returns:
        LLM instance (ChatOllama, ChatGoogleGenerativeAI, ChatOpenAI, or HuggingFace)
        
    Examples:
        # Use Gemini
        llm = build_llm("gemini-2.0-flash")
        
        # Use GPT-4
        llm = build_llm("gpt-4o-mini")
        
        # Use HuggingFace
        llm = build_llm("hf:meta-llama/Meta-Llama-3-8B-Instruct")
        
        # Use Ollama (default)
        llm = build_llm("llama3.1:8b")
    """
    # Use default from settings if no model specified
    if model_name is None:
        model_name = settings.EXPERT_MODEL
    
    # Check if using Gemini
    if model_name.lower().startswith("gemini"):
        from app.llm.gemini_factory import build_gemini_llm
        return build_gemini_llm(model_name, temperature)
    
    # Check if using OpenAI GPT
    if model_name.lower().startswith("gpt-"):
        from app.llm.gpt_factory import build_gpt_llm
        return build_gpt_llm(model_name, temperature)
    
    # Check if using HuggingFace (prefix with "hf:")
    if model_name.lower().startswith("hf:"):
        from app.llm.huggingface_factory import build_huggingface_llm
        # Remove "hf:" prefix
        hf_model_name = model_name[3:]
        return build_huggingface_llm(hf_model_name, temperature)

    # Default to Ollama (local models)
    return ChatOllama(
        model=model_name,
        temperature=temperature,
        top_p=0.9,
        repeat_penalty=1.05,
    )