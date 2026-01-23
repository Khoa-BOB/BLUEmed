# LLM Factory Architecture

## System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                      BlueMed Application                        │
│                                                                 │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────┐ │
│  │   Expert A       │  │   Expert B       │  │    Judge     │ │
│  │   Agent          │  │   Agent          │  │    Agent     │ │
│  └────────┬─────────┘  └────────┬─────────┘  └──────┬───────┘ │
│           │                     │                   │          │
│           └─────────────────────┼───────────────────┘          │
│                                 ▼                               │
└─────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                    LLM Factory Router                           │
│                  (app/llm/factory.py)                           │
│                                                                 │
│  def build_llm(model_name, temperature):                        │
│      if model_name.startswith("gemini"):                        │
│          return build_gemini_llm(...)                           │
│      if model_name.startswith("gpt-"):                          │
│          return build_gpt_llm(...)                              │
│      if model_name.startswith("hf:"):                           │
│          return build_huggingface_llm(...)                      │
│      else:                                                      │
│          return ChatOllama(...)  # Default                      │
└─────────────────────────────────────────────────────────────────┘
                                  │
                ┌─────────────────┼─────────────────┬──────────┐
                ▼                 ▼                 ▼          ▼
    ┌───────────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
    │ Ollama Factory    │ │ Gemini       │ │ GPT          │ │ HuggingFace  │
    │ (Default)         │ │ Factory      │ │ Factory      │ │ Factory      │
    │                   │ │              │ │              │ │              │
    │ ChatOllama        │ │ build_gemini │ │ build_gpt    │ │ build_hf     │
    │                   │ │ _llm()       │ │ _llm()       │ │ _llm()       │
    └─────────┬─────────┘ └──────┬───────┘ └──────┬───────┘ └──────┬───────┘
              │                  │                │                │
              ▼                  ▼                ▼                ▼
    ┌───────────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
    │ Local Models      │ │ Google       │ │ OpenAI       │ │ HuggingFace  │
    │ - llama3.1:8b     │ │ Gemini API   │ │ API          │ │ Hub API      │
    │ - mistral:7b      │ │              │ │              │ │ or Local     │
    │ - mixtral:8x7b    │ │ Models:      │ │ Models:      │ │              │
    │                   │ │ - gemini-2.0 │ │ - gpt-4o     │ │ Models:      │
    │ Free, Offline     │ │ - gemini-1.5 │ │ - gpt-4o-mini│ │ - llama-3    │
    │ No API key        │ │              │ │ - gpt-3.5    │ │ - mistral    │
    │                   │ │ Free tier:   │ │              │ │ - mixtral    │
    │                   │ │ 15 RPM       │ │ Paid only    │ │ - falcon     │
    └───────────────────┘ └──────────────┘ └──────────────┘ └──────────────┘
```

---

## Configuration Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                         .env File                               │
│                                                                 │
│  # Model Selection                                              │
│  EXPERT_MODEL=gpt-4o-mini                                       │
│  JUDGE_MODEL=gemini-1.5-pro                                     │
│                                                                 │
│  # API Keys (optional - only needed for cloud providers)        │
│  OPENAI_API_KEY=sk-...                                          │
│  GOOGLE_API_KEY=AIza...                                         │
│  HUGGINGFACE_API_KEY=hf_...                                     │
└─────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                    config/settings.py                           │
│                                                                 │
│  class Settings(BaseSettings):                                  │
│      EXPERT_MODEL: str                                          │
│      JUDGE_MODEL: str                                           │
│      OPENAI_API_KEY: str = ""                                   │
│      GOOGLE_API_KEY: str = ""                                   │
│      HUGGINGFACE_API_KEY: str = ""                              │
│      ...                                                        │
│                                                                 │
│  settings = Settings()  # Loads from .env                       │
└─────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                  Application Runtime                            │
│                                                                 │
│  from app.llm.factory import build_llm                          │
│  from config.settings import settings                           │
│                                                                 │
│  # Build LLMs from settings                                     │
│  expert_llm = build_llm(settings.EXPERT_MODEL)   # gpt-4o-mini │
│  judge_llm = build_llm(settings.JUDGE_MODEL)     # gemini-1.5  │
│                                                                 │
│  # Inject into agents                                           │
│  graph.add_node("expert", partial(expert_node, llm=expert_llm)) │
│  graph.add_node("judge", partial(judge_node, llm=judge_llm))   │
└─────────────────────────────────────────────────────────────────┘
```

---

## Factory Pattern Implementation

### 1. Main Factory (Router)

**File:** `app/llm/factory.py`

```python
def build_llm(model_name: str = None, temperature: float = 0.2):
    """
    Central factory that routes to specific provider factories.
    """
    # 1. Get model name from settings if not provided
    if model_name is None:
        model_name = settings.EXPERT_MODEL
    
    # 2. Route based on model name prefix
    if model_name.startswith("gemini"):
        return build_gemini_llm(model_name, temperature)
    
    if model_name.startswith("gpt-"):
        return build_gpt_llm(model_name, temperature)
    
    if model_name.startswith("hf:"):
        return build_huggingface_llm(model_name[3:], temperature)
    
    # 3. Default to Ollama (local)
    return ChatOllama(model=model_name, temperature=temperature)
```

**Benefits:**
- ✅ Single entry point
- ✅ Automatic routing
- ✅ Easy to extend
- ✅ Backward compatible

---

### 2. Provider Factories

#### Ollama Factory (Built-in)

```python
# In factory.py - no separate factory file needed
return ChatOllama(
    model=model_name,      # e.g., "llama3.1:8b"
    temperature=temperature,
    top_p=0.9,
    repeat_penalty=1.05,
)
```

**Characteristics:**
- No API key required
- Runs locally
- Free and private
- Requires Ollama installed

---

#### Gemini Factory

**File:** `app/llm/gemini_factory.py`

```python
def build_gemini_llm(model_name: str, temperature: float):
    """
    Build Google Gemini LLM.
    """
    api_key = settings.GOOGLE_API_KEY
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found")
    
    return ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=api_key,
        temperature=temperature,
        max_output_tokens=1024,
        convert_system_message_to_human=True
    )
```

**Characteristics:**
- Requires GOOGLE_API_KEY
- Generous free tier (15 RPM)
- Excellent quality
- Large context (up to 2M tokens)

---

#### GPT Factory ⭐ NEW

**File:** `app/llm/gpt_factory.py`

```python
def build_gpt_llm(model_name: str, temperature: float):
    """
    Build OpenAI GPT LLM.
    """
    api_key = settings.OPENAI_API_KEY
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found")
    
    return ChatOpenAI(
        model=model_name,
        openai_api_key=api_key,
        temperature=temperature,
        max_tokens=1024,
        timeout=60,
        max_retries=2,
    )
```

**Characteristics:**
- Requires OPENAI_API_KEY
- No free tier (paid only)
- Best-in-class quality
- Fast and reliable

---

#### HuggingFace Factory ⭐ NEW

**File:** `app/llm/huggingface_factory.py`

```python
def build_huggingface_llm(model_name: str, temperature: float, use_local: bool):
    """
    Build HuggingFace LLM (API or local).
    """
    if not use_local:
        # API mode (serverless inference)
        api_key = settings.HUGGINGFACE_API_KEY
        if not api_key:
            raise ValueError("HUGGINGFACE_API_KEY not found")
        
        llm = HuggingFaceEndpoint(
            repo_id=model_name,
            huggingfacehub_api_token=api_key,
            temperature=temperature,
            max_new_tokens=1024,
        )
        
        return ChatHuggingFace(llm=llm)
    
    else:
        # Local mode (requires transformers)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
        return HuggingFacePipeline(pipeline=pipe)
```

**Characteristics:**
- Two modes: API (default) or local
- Thousands of models available
- API requires HUGGINGFACE_API_KEY
- Local requires GPU for good performance

---

## Usage Patterns

### Pattern 1: Uniform Provider (Simplest)

```python
# .env
EXPERT_MODEL=gemini-2.0-flash
JUDGE_MODEL=gemini-2.0-flash
GOOGLE_API_KEY=your_key
```

**Use when:**
- Simple setup
- Single provider is sufficient
- Consistent behavior across agents

---

### Pattern 2: Mixed Providers (Optimized)

```python
# .env
EXPERT_MODEL=gpt-4o-mini        # Fast, cost-effective
JUDGE_MODEL=gemini-1.5-pro      # High quality
OPENAI_API_KEY=your_openai_key
GOOGLE_API_KEY=your_gemini_key
```

**Use when:**
- Want to optimize cost/quality tradeoff
- Different agents have different requirements
- Leveraging free tiers

---

### Pattern 3: Development vs Production

```python
# .env.development
EXPERT_MODEL=llama3.1:8b   # Free, local
JUDGE_MODEL=llama3.1:8b

# .env.production
EXPERT_MODEL=gpt-4o-mini   # Fast, paid
JUDGE_MODEL=gpt-4o         # Best quality
OPENAI_API_KEY=your_key
```

**Use when:**
- Testing locally without costs
- Production needs higher quality

---

## Decision Tree: Which Provider to Use?

```
Start: Need an LLM
    │
    ▼
┌─────────────────────┐
│ Do you need         │
│ multimodal (vision)?│
└──────┬──────────────┘
       │
   Yes │    No
       ▼     ▼
    ┌────┐ ┌──────────────────────┐
    │GPT │ │ Budget available?    │
    │4o  │ └──────┬───────────────┘
    └────┘        │
           Paid   │   Free
                  ▼    ▼
              ┌────┐ ┌────────────────────┐
              │GPT │ │ Internet required? │
              │4o  │ └──────┬─────────────┘
              └────┘        │
                     Yes    │   No
                            ▼    ▼
                        ┌─────┐ ┌──────┐
                        │Gemini│ │Ollama│
                        │2.0  │ │      │
                        └─────┘ └──────┘

Quality Priority:     gpt-4o > gemini-1.5-pro > gpt-4o-mini > gemini-2.0-flash
Cost Priority:        ollama > gemini > hf (free tier) > gpt-4o-mini > gpt-4o
Speed Priority:       gpt-4o-mini > gemini-2.0-flash > ollama > gpt-4o
Context Size:         gemini-1.5-pro (2M) > gemini-2.0 (1M) > gpt (128K)
```

---

## Extension Guide

### Adding a New Provider

To add support for a new LLM provider (e.g., Anthropic Claude):

1. **Create factory file:**
   ```python
   # app/llm/claude_factory.py
   from langchain_anthropic import ChatAnthropic
   from config.settings import settings
   
   def build_claude_llm(model_name: str, temperature: float):
       api_key = settings.ANTHROPIC_API_KEY
       if not api_key:
           raise ValueError("ANTHROPIC_API_KEY not found")
       
       return ChatAnthropic(
           model=model_name,
           anthropic_api_key=api_key,
           temperature=temperature,
       )
   ```

2. **Update main factory:**
   ```python
   # In app/llm/factory.py
   def build_llm(model_name: str = None, temperature: float = 0.2):
       # ... existing code ...
       
       # Add Claude routing
       if model_name.startswith("claude-"):
           from app.llm.claude_factory import build_claude_llm
           return build_claude_llm(model_name, temperature)
       
       # ... rest of code ...
   ```

3. **Update settings:**
   ```python
   # In config/settings.py
   class Settings(BaseSettings):
       # ... existing fields ...
       
       # Anthropic Claude
       ANTHROPIC_API_KEY: str = ""
       CLAUDE_MODEL: str = "claude-3-opus"
   ```

4. **Update requirements.txt:**
   ```
   langchain-anthropic==0.1.0
   ```

5. **Document in README:**
   - Add to supported providers list
   - Add setup instructions
   - Add usage examples

---

## Testing

### Unit Tests

```python
# app/llm/test_factories.py

def test_factory_routing():
    """Test that factory routes correctly."""
    from app.llm.factory import build_llm
    
    # Test each provider
    ollama_llm = build_llm("llama3.1:8b")
    assert isinstance(ollama_llm, ChatOllama)
    
    gemini_llm = build_llm("gemini-2.0-flash")
    assert isinstance(gemini_llm, ChatGoogleGenerativeAI)
    
    gpt_llm = build_llm("gpt-4o-mini")
    assert isinstance(gpt_llm, ChatOpenAI)
    
    hf_llm = build_llm("hf:meta-llama/Meta-Llama-3-8B-Instruct")
    assert isinstance(hf_llm, ChatHuggingFace)
```

### Integration Tests

```python
def test_medical_reasoning():
    """Test LLM on medical task."""
    from app.llm.factory import build_llm
    from langchain_core.messages import HumanMessage, SystemMessage
    
    llm = build_llm("gpt-4o-mini")
    
    messages = [
        SystemMessage(content="You are a medical expert."),
        HumanMessage(content="Is aspirin contraindicated with warfarin?")
    ]
    
    response = llm.invoke(messages)
    assert response.content  # Check response exists
    assert len(response.content) > 0  # Check not empty
```

---

## Best Practices

### 1. Environment-Based Configuration

✅ **Good:**
```python
# Use settings
llm = build_llm(settings.EXPERT_MODEL)
```

❌ **Bad:**
```python
# Hardcoded
llm = build_llm("gpt-4o")
```

### 2. Error Handling

✅ **Good:**
```python
try:
    llm = build_llm("gpt-4o-mini")
except ValueError as e:
    print(f"Configuration error: {e}")
    llm = build_llm("llama3.1:8b")  # Fallback
```

### 3. Cost Optimization

✅ **Good:**
```python
# Use cheaper model for initial analysis
expert_llm = build_llm("gpt-4o-mini", temperature=0.2)

# Use premium model for final decision
judge_llm = build_llm("gpt-4o", temperature=0.1)
```

### 4. Temperature Tuning

```python
# Factual tasks (medical analysis)
llm = build_llm("gpt-4o-mini", temperature=0.1)  # Low = deterministic

# Creative tasks (generating alternatives)
llm = build_llm("gpt-4o-mini", temperature=0.7)  # Higher = creative
```

---

## Performance Considerations

### Latency

| Provider | Typical Latency | Notes |
|----------|----------------|-------|
| Ollama | 1-5s | Depends on hardware |
| Gemini | 2-4s | Fast API |
| GPT | 1-3s | Very fast |
| HuggingFace (API) | 3-10s | Variable |
| HuggingFace (local) | 5-20s | Depends on GPU |

### Cost (per 1M tokens)

| Provider | Model | Input | Output |
|----------|-------|-------|--------|
| Ollama | Any | Free | Free |
| Gemini | 2.0-flash | Free* | Free* |
| Gemini | 1.5-pro | $1.25 | $5.00 |
| GPT | 4o-mini | $0.15 | $0.60 |
| GPT | 4o | $5.00 | $15.00 |
| HuggingFace | Various | Free* | Free* |

*With rate limits

---

## Summary

The LLM factory architecture provides:

✅ **Flexibility** - Support for 4 providers  
✅ **Simplicity** - Single interface for all  
✅ **Extensibility** - Easy to add new providers  
✅ **Reliability** - Proper error handling  
✅ **Performance** - Optimized settings  
✅ **Documentation** - Comprehensive guides  

**Start simple, scale as needed!**
