"""
HuggingFace LLM factory.
Supports both HuggingFace Hub API and local models.
"""
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from config.settings import settings


def build_huggingface_llm(
    model_name: str = "Qwen/Qwen3-4B-Instruct-2507",
    temperature: float = 0.2,
    use_local: bool = True
):
    """
    Build HuggingFace LLM using either HuggingFace Hub API or local inference.

    Args:
        model_name: HuggingFace model identifier
            Popular models:
            - meta-llama/Meta-Llama-3-8B-Instruct (Llama 3, good balance)
            - meta-llama/Meta-Llama-3-70B-Instruct (Llama 3, most capable)
            - mistralai/Mistral-7B-Instruct-v0.2 (Fast, efficient)
            - mistralai/Mixtral-8x7B-Instruct-v0.1 (Mixture of experts)
            - google/flan-t5-xxl (Instruction-tuned T5)
            - tiiuae/falcon-7b-instruct (Falcon)
        
        temperature: Sampling temperature (0.0-1.0)
            - 0.0: Deterministic outputs
            - 0.2-0.7: Balanced
            - 1.0: Maximum creativity
        
        use_local: If True, use local model (requires model downloaded)
                   If False, use HuggingFace Hub API (default)

    Returns:
        LangChain HuggingFace LLM wrapper

    Setup for HuggingFace Hub API:
        1. Create account at: https://huggingface.co
        2. Get token from: https://huggingface.co/settings/tokens
        3. Set in .env: HUGGINGFACE_API_KEY=your_token_here
        
    Setup for local models:
        1. Install transformers: pip install transformers torch
        2. Download model: huggingface-cli download model_name
        3. Set use_local=True
        
    Usage Examples:
        # Using HuggingFace Hub API (default)
        from app.llm.huggingface_factory import build_huggingface_llm
        llm = build_huggingface_llm("meta-llama/Meta-Llama-3-8B-Instruct")
        
        # Using local model
        llm = build_huggingface_llm("meta-llama/Meta-Llama-3-8B-Instruct", use_local=True)
    """

    if not use_local:
        # Use HuggingFace Hub API (serverless inference)
        api_key = settings.HUGGINGFACE_API_KEY
        if not api_key:
            raise ValueError(
                "HUGGINGFACE_API_KEY not found in environment. "
                "Get one from https://huggingface.co/settings/tokens"
            )

        # Create HuggingFace endpoint
        llm = HuggingFaceEndpoint(
            repo_id=model_name,
            huggingfacehub_api_token=api_key,
            temperature=temperature,
            max_new_tokens=1024,
            top_k=50,
            top_p=0.95,
            typical_p=0.95,
            repetition_penalty=1.03,
        )
        
        # Wrap with ChatHuggingFace for chat interface compatibility
        chat_llm = ChatHuggingFace(llm=llm)
        
        return chat_llm
    
    else:
        # Use local HuggingFace model (requires model downloaded)
        # Note: This requires additional dependencies and GPU for good performance
        try:
            from langchain_huggingface import HuggingFacePipeline
            from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
        except ImportError:
            raise ImportError(
                "Local HuggingFace inference requires additional dependencies. "
                "Install with: pip install transformers torch accelerate"
            )
        
        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",  # Automatically use GPU if available
            torch_dtype="auto",  # Use appropriate dtype
        )
        
        # Create text generation pipeline
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=1024,
            temperature=temperature,
            top_p=0.95,
            repetition_penalty=1.03,
        )
        
        # Wrap pipeline with LangChain
        llm = HuggingFacePipeline(pipeline=pipe)
        
        return llm


def build_huggingface_inference_llm(
    model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct",
    temperature: float = 0.2
):
    """
    Simplified wrapper for HuggingFace Hub API inference (most common use case).
    
    Args:
        model_name: HuggingFace model identifier
        temperature: Sampling temperature
    
    Returns:
        LangChain ChatHuggingFace LLM
    """
    return build_huggingface_llm(
        model_name=model_name,
        temperature=temperature,
        use_local=False
    )
