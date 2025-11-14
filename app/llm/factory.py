from langchain_ollama import ChatOllama

def build_llm(model_name: str, temperature: float = 0.7):
    return ChatOllama(
        model=model_name,
        temperature=temperature,
        top_p=0.9,
        repeat_penalty=1.05,
    )