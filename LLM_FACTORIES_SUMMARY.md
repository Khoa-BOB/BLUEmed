# LLM Factories Implementation Summary

## Overview

Successfully implemented factory patterns for **OpenAI GPT** and **HuggingFace** LLMs, expanding the BlueMed system to support 4 different LLM providers:

1. ✅ **Ollama** (Local models) - Existing
2. ✅ **Google Gemini** (Cloud API) - Existing  
3. ✅ **OpenAI GPT** (Cloud API) - **NEW**
4. ✅ **HuggingFace** (Cloud API + Local) - **NEW**

---

## Files Created/Modified

### New Factory Files

1. **`app/llm/gpt_factory.py`** - OpenAI GPT factory implementation
   - Supports GPT-4o, GPT-4o-mini, GPT-4-turbo, GPT-3.5-turbo
   - Configurable temperature, max_tokens, timeout, retries
   - Comprehensive error handling and setup instructions

2. **`app/llm/huggingface_factory.py`** - HuggingFace factory implementation
   - Supports HuggingFace Hub API (serverless inference)
   - Supports local model inference (with transformers)
   - Popular models: Llama 3, Mistral, Mixtral, Falcon, FLAN-T5
   - Two modes: remote API (default) or local inference

### Modified Files

3. **`app/llm/factory.py`** - Main factory router
   - Added routing logic for GPT models (prefix: `gpt-`)
   - Added routing logic for HuggingFace models (prefix: `hf:`)
   - Made model_name parameter optional (defaults to settings)
   - Enhanced documentation with usage examples

4. **`config/settings.py`** - Configuration settings
   - Added `OPENAI_API_KEY` and `GPT_MODEL` settings
   - Added `HUGGINGFACE_API_KEY` and `HUGGINGFACE_MODEL` settings
   - All API keys are optional (only needed if using that provider)

5. **`requirements.txt`** - Dependencies
   - Added `langchain-openai==0.3.14`
   - Added `langchain-huggingface==0.1.2`

6. **`README.md`** - Main documentation
   - Updated Configuration section with new LLM providers
   - Added table showing all supported model formats
   - Added reference to detailed LLM documentation

### Documentation Files

7. **`app/llm/README.md`** - Comprehensive LLM documentation (NEW)
   - Detailed setup instructions for each provider
   - Usage examples and code snippets
   - Model comparison table
   - Troubleshooting guide
   - Cost and performance considerations

8. **`app/llm/test_factories.py`** - Test suite (NEW)
   - Individual tests for each provider
   - Factory routing tests
   - Medical reasoning example test
   - Helpful error messages for missing API keys

9. **`.env.example`** - Environment template (NEW)
   - Complete template with all configuration options
   - Quick setup examples for different scenarios
   - Comments explaining each setting

---

## Implementation Details

### 1. GPT Factory (`gpt_factory.py`)

**Features:**
- Uses `langchain-openai` package
- Supports all OpenAI chat models (GPT-4o, GPT-4o-mini, etc.)
- Configurable parameters: temperature, max_tokens, timeout, retries
- Requires `OPENAI_API_KEY` in environment

**Usage:**
```python
from app.llm.gpt_factory import build_gpt_llm

# Direct usage
llm = build_gpt_llm("gpt-4o-mini", temperature=0.2)

# Via main factory
from app.llm.factory import build_llm
llm = build_llm("gpt-4o-mini")
```

**Supported Models:**
- `gpt-4o` - Most capable, multimodal
- `gpt-4o-mini` - Fast and cost-effective
- `gpt-4-turbo` - High quality, faster than GPT-4
- `gpt-3.5-turbo` - Fastest, most economical

---

### 2. HuggingFace Factory (`huggingface_factory.py`)

**Features:**
- Uses `langchain-huggingface` package
- Two modes: **API** (default) or **Local** inference
- Supports thousands of HuggingFace models
- Wraps with `ChatHuggingFace` for chat interface compatibility

**Usage:**
```python
from app.llm.huggingface_factory import build_huggingface_llm

# API mode (default - serverless)
llm = build_huggingface_llm(
    "meta-llama/Meta-Llama-3-8B-Instruct",
    temperature=0.2,
    use_local=False
)

# Local mode (requires model downloaded)
llm = build_huggingface_llm(
    "meta-llama/Meta-Llama-3-8B-Instruct",
    use_local=True
)

# Via main factory (prefix with "hf:")
from app.llm.factory import build_llm
llm = build_llm("hf:meta-llama/Meta-Llama-3-8B-Instruct")
```

**Popular Models:**
- `meta-llama/Meta-Llama-3-8B-Instruct` - Llama 3 (8B)
- `meta-llama/Meta-Llama-3-70B-Instruct` - Llama 3 (70B)
- `mistralai/Mistral-7B-Instruct-v0.2` - Mistral 7B
- `mistralai/Mixtral-8x7B-Instruct-v0.1` - Mixtral MoE
- `google/flan-t5-xxl` - T5 instruction-tuned

---

### 3. Main Factory Routing (`factory.py`)

**Enhanced Routing Logic:**
```python
def build_llm(model_name: str = None, temperature: float = 0.2):
    # Defaults to settings.EXPERT_MODEL if None
    if model_name is None:
        model_name = settings.EXPERT_MODEL
    
    # Route based on model name prefix
    if model_name.startswith("gemini"):
        return build_gemini_llm(model_name, temperature)
    
    if model_name.startswith("gpt-"):
        return build_gpt_llm(model_name, temperature)
    
    if model_name.startswith("hf:"):
        hf_model_name = model_name[3:]  # Remove "hf:" prefix
        return build_huggingface_llm(hf_model_name, temperature)
    
    # Default: Ollama
    return ChatOllama(model=model_name, temperature=temperature, ...)
```

**Benefits:**
- Single entry point for all LLM providers
- Automatic routing based on model name
- Backward compatible with existing code
- Easy to add new providers

---

## Configuration Guide

### Quick Setup Examples

#### Example 1: Use Ollama (No API Keys Needed)
```bash
# .env
EXPERT_MODEL=llama3.1:8b
JUDGE_MODEL=llama3.1:8b
```

#### Example 2: Use Gemini (Free Tier Available)
```bash
# .env
GOOGLE_API_KEY=your_gemini_key
EXPERT_MODEL=gemini-2.0-flash
JUDGE_MODEL=gemini-2.0-flash
```

#### Example 3: Use GPT-4o-mini
```bash
# .env
OPENAI_API_KEY=your_openai_key
EXPERT_MODEL=gpt-4o-mini
JUDGE_MODEL=gpt-4o-mini
```

#### Example 4: Use HuggingFace
```bash
# .env
HUGGINGFACE_API_KEY=your_hf_token
EXPERT_MODEL=hf:meta-llama/Meta-Llama-3-8B-Instruct
JUDGE_MODEL=hf:mistralai/Mistral-7B-Instruct-v0.2
```

#### Example 5: Mix Providers
```bash
# .env
OPENAI_API_KEY=your_openai_key
GOOGLE_API_KEY=your_gemini_key

# Use GPT for experts (fast, cost-effective)
EXPERT_MODEL=gpt-4o-mini

# Use Gemini for judge (high quality, generous free tier)
JUDGE_MODEL=gemini-1.5-pro
```

---

## Testing

### Run Test Suite

```bash
# Test all factories
python app/llm/test_factories.py
```

The test suite will:
- ✅ Test factory routing logic
- ✅ Test each provider (if configured)
- ⚠️ Skip providers without API keys (with helpful message)
- ✅ Run a medical reasoning example

### Expected Output

```
============================================================
LLM Factory Test Suite
============================================================

Testing Factory Routing
============================================================
✓ llama3.1:8b → Ollama routing works
✓ gemini-2.0-flash → Gemini routing works
⚠ gpt-4o-mini → GPT not configured: OPENAI_API_KEY not found
⚠ hf:meta-llama/Meta-Llama-3-8B-Instruct → HuggingFace not configured

============================================================
Testing Ollama (Local)
============================================================
✓ Ollama LLM created successfully
Response: Hello from Ollama!
✓ Ollama test passed!

... (more tests)
```

---

## Model Comparison

| Provider | Model | Cost | Speed | Quality | Context | Free Tier |
|----------|-------|------|-------|---------|---------|-----------|
| Ollama | llama3.1:8b | Free | Fast | Good | 128K | Unlimited |
| Gemini | gemini-2.0-flash | Low | Fast | Excellent | 1M | 15 RPM |
| Gemini | gemini-1.5-pro | Medium | Medium | Best | 2M | 2 RPM |
| OpenAI | gpt-4o-mini | Low | Fast | Excellent | 128K | No |
| OpenAI | gpt-4o | High | Medium | Best | 128K | No |
| HuggingFace | llama-3-8b | Free* | Fast | Good | 8K | Limited |

*Free for HF Hub API with rate limits; local inference is free but requires GPU.

---

## Key Features

### 1. **Unified Interface**
- All providers use the same LangChain interface
- Code works seamlessly across providers
- Easy to switch providers without changing agent code

### 2. **Flexible Configuration**
- Environment-based configuration
- Can mix different providers for different agents
- Optional API keys (only needed if using that provider)

### 3. **Comprehensive Documentation**
- Detailed setup instructions for each provider
- Code examples and usage patterns
- Troubleshooting guides

### 4. **Production Ready**
- Error handling with helpful messages
- Timeout and retry logic (GPT)
- Performance optimization settings (HuggingFace)

### 5. **Cost Optimization**
- Use free Ollama for development
- Use Gemini's generous free tier
- Use GPT-4o-mini for cost-effective cloud API
- Mix providers based on task importance

---

## Usage in BlueMed System

### Current Usage Pattern

The BlueMed system uses LLMs in multiple places:

1. **Expert Agents** (`app/agents/expertA.py`, `app/agents/expertB.py`)
   - Analyze medical notes
   - Generate debate arguments
   - Use retrieved knowledge from RAG

2. **Judge Agent** (`app/agents/judge.py`)
   - Evaluate expert arguments
   - Make final decision on medical errors

3. **Graph Chain** (`app/graph/chain.py`)
   - Injects LLMs into agent nodes
   - Uses `build_llm()` from factory

### Example Integration

```python
# In app/graph/chain.py
from app.llm.factory import build_llm

# Build LLM from settings
llm = build_llm()  # Uses EXPERT_MODEL from .env

# Or specify model directly
judge_llm = build_llm("gpt-4o")
expert_llm = build_llm("gemini-2.0-flash")

# Inject into agents
graph.add_node("expert1", partial(expertA_node, llm=expert_llm))
graph.add_node("judge", partial(Judge_node, llm=judge_llm))
```

---

## Migration Guide

### From Ollama-Only Setup

**Before:**
```bash
# .env
EXPERT_MODEL=llama3.1:8b
JUDGE_MODEL=llama3.1:8b
```

**After (using GPT):**
```bash
# .env
OPENAI_API_KEY=your_key_here
EXPERT_MODEL=gpt-4o-mini
JUDGE_MODEL=gpt-4o-mini
```

**No code changes needed!** The factory automatically routes to the correct provider.

### From Gemini to Mixed Providers

**Before:**
```bash
EXPERT_MODEL=gemini-2.0-flash
JUDGE_MODEL=gemini-2.0-flash
```

**After:**
```bash
# Use fast GPT for experts
EXPERT_MODEL=gpt-4o-mini

# Use high-quality Gemini for judge
JUDGE_MODEL=gemini-1.5-pro

# Both API keys needed
OPENAI_API_KEY=your_openai_key
GOOGLE_API_KEY=your_gemini_key
```

---

## Next Steps

### Recommended Setup for Development

```bash
# Start with Ollama (free, no API keys)
EXPERT_MODEL=llama3.1:8b
JUDGE_MODEL=llama3.1:8b
```

### Recommended Setup for Production

```bash
# Option 1: All Gemini (generous free tier)
GOOGLE_API_KEY=your_key
EXPERT_MODEL=gemini-2.0-flash
JUDGE_MODEL=gemini-1.5-pro

# Option 2: Mixed (cost-optimized)
OPENAI_API_KEY=your_openai_key
GOOGLE_API_KEY=your_gemini_key
EXPERT_MODEL=gpt-4o-mini      # Fast and cheap
JUDGE_MODEL=gemini-1.5-pro    # High quality

# Option 3: All GPT (best performance)
OPENAI_API_KEY=your_key
EXPERT_MODEL=gpt-4o-mini
JUDGE_MODEL=gpt-4o
```

---

## Troubleshooting

### "API key not found in environment"

**Solution:** Add the required API key to `.env` file:
```bash
echo "OPENAI_API_KEY=your_key" >> .env
echo "GOOGLE_API_KEY=your_key" >> .env
echo "HUGGINGFACE_API_KEY=your_key" >> .env
```

### "Model not found" (Ollama)

**Solution:** Pull the model first:
```bash
ollama pull llama3.1:8b
ollama list  # See all models
```

### Rate Limit Errors

**Gemini:** Free tier is 15 RPM. Add delays or upgrade to paid tier.
**HuggingFace:** Limited free tier. Consider paid plan or local inference.
**OpenAI:** Set up billing and increase rate limits.

### Import Errors

**Solution:** Install missing dependencies:
```bash
pip install -r requirements.txt

# For local HuggingFace (optional)
pip install transformers torch accelerate
```

---

## Resources

### Get API Keys

- **OpenAI GPT:** https://platform.openai.com/api-keys
- **Google Gemini:** https://aistudio.google.com/app/apikey
- **HuggingFace:** https://huggingface.co/settings/tokens

### Documentation

- **LLM Factory Guide:** [app/llm/README.md](app/llm/README.md)
- **Test Suite:** [app/llm/test_factories.py](app/llm/test_factories.py)
- **Environment Template:** [.env.example](.env.example)
- **Main README:** [README.md](README.md)

### Model Information

- **OpenAI Models:** https://platform.openai.com/docs/models
- **Gemini Models:** https://ai.google.dev/models
- **HuggingFace Models:** https://huggingface.co/models

---

## Summary

✅ **Implemented:** OpenAI GPT and HuggingFace factory patterns  
✅ **Enhanced:** Main factory with intelligent routing  
✅ **Updated:** Configuration settings and documentation  
✅ **Created:** Comprehensive test suite  
✅ **Added:** Dependencies to requirements.txt  
✅ **Documented:** Full setup and usage guides  

**BlueMed now supports 4 LLM providers with seamless switching and mixing!**
