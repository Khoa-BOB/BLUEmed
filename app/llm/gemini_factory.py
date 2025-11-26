"""
Google Gemini API LLM factory.
"""
from langchain_google_genai import ChatGoogleGenerativeAI
from config.settings import settings
import os


def build_gemini_llm(model_name: str = "gemini-2.0-flash", temperature: float = 0.2):
    """
    Build Google Gemini LLM.

    Args:
        model_name: Gemini model name
            - gemini-1.5-pro (best quality, free tier available)
            - gemini-1.5-flash (fastest, free tier available)
            - gemini-pro (older, stable)
        temperature: Sampling temperature

    Returns:
        LangChain Gemini LLM

    Setup:
        1. Get API key from: https://aistudio.google.com/app/apikey
        2. Set in .env: GOOGLE_API_KEY=your_key_here
    """

    api_key = settings.GOOGLE_API_KEY
    if not api_key:
        raise ValueError(
            "GOOGLE_API_KEY not found in environment. "
            "Get one from https://aistudio.google.com/app/apikey"
        )

    llm = ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=api_key,
        temperature=temperature,
        max_output_tokens=1024,
        convert_system_message_to_human=True
            # Whether to merge any leading SystemMessage into HumanMessage
    )

    return llm
