# LLM Factory Documentation

This directory contains factory functions for building Language Learning Models (LLMs) from various providers.

## Supported Providers

1. **Ollama** (Local Models) - Default
2. **Google Gemini** (Cloud API)
3. **OpenAI GPT** (Cloud API) 
4. **HuggingFace** (Cloud API or Local)

---

## Quick Start

### Basic Usage

```python
from app.llm.factory import build_llm

# Use Ollama (default - local)
llm = build_llm("llama3.1:8b")

# Use Google Gemini
llm = build_llm("gemini-2.0-flash")

# Use OpenAI GPT
llm = build_llm("gpt-4o-mini")

# Use HuggingFace
llm = build_llm("hf:meta-llama/Meta-Llama-3-8B-Instruct")

# Invoke the LLM
from langchain_core.messages import HumanMessage
response = llm.invoke([HumanMessage(content="Hello!")])
print(response.content)
```

---

## Provider Details

### 1. Ollama (Local Models)

**Use when:** You want to run models locally without API costs or internet dependency.

**Setup:**
```bash
# Install Ollama: https://ollama.ai
curl https://ollama.ai/install.sh | sh

# Pull a model
ollama pull llama3.1:8b
```

**Usage:**
```python
from app.llm.factory import build_llm

# Use any Ollama model
llm = build_llm("llama3.1:8b", temperature=0.2)
llm = build_llm("mistral:7b", temperature=0.5)
```

**Popular Models:**
- `llama3.1:8b` - Meta's Llama 3.1 (8B parameters)
- `llama3.1:70b` - Meta's Llama 3.1 (70B parameters)
- `mistral:7b` - Mistral 7B
- `mixtral:8x7b` - Mixtral Mixture of Experts
- `phi3:mini` - Microsoft Phi-3 Mini

**Configuration:** No API key needed. Set `OLLAMA_URL` in `.env` if using remote Ollama server.

---

### 2. Google Gemini

**Use when:** You want high-quality responses with generous free tier and multimodal capabilities.

**Setup:**
```bash
# 1. Get API key: https://aistudio.google.com/app/apikey
# 2. Add to .env
echo "GOOGLE_API_KEY=your_api_key_here" >> .env
echo "GEMINI_MODEL=gemini-2.0-flash" >> .env
```

**Usage:**
```python
from app.llm.factory import build_llm

# Use via main factory
llm = build_llm("gemini-2.0-flash")

# Or use directly
from app.llm.gemini_factory import build_gemini_llm
llm = build_gemini_llm("gemini-2.0-flash", temperature=0.2)
```

**Available Models:**
- `gemini-2.0-flash` - Latest, fastest, best price/performance
- `gemini-1.5-pro` - Best quality, longer context (128K tokens)
- `gemini-1.5-flash` - Fast, efficient
- `gemini-pro` - Older, stable version

**Free Tier:** 15 RPM (requests per minute), very generous for development.

---

### 3. OpenAI GPT

**Use when:** You need state-of-the-art performance and reliability.

**Setup:**
```bash
# 1. Get API key: https://platform.openai.com/api-keys
# 2. Add to .env
echo "OPENAI_API_KEY=your_api_key_here" >> .env
echo "GPT_MODEL=gpt-4o-mini" >> .env
```

**Usage:**
```python
from app.llm.factory import build_llm

# Use via main factory
llm = build_llm("gpt-4o-mini")

# Or use directly
from app.llm.gpt_factory import build_gpt_llm
llm = build_gpt_llm("gpt-4o", temperature=0.2)
```

**Available Models:**
- `gpt-4o` - Most capable model (multimodal)
- `gpt-4o-mini` - Fast, cost-effective, great for most tasks
- `gpt-4-turbo` - High quality, faster than GPT-4
- `gpt-3.5-turbo` - Fastest, most economical

**Pricing (as of 2026):**
- `gpt-4o-mini`: ~$0.15/1M input tokens, ~$0.60/1M output tokens
- `gpt-4o`: ~$5/1M input tokens, ~$15/1M output tokens

---

### 4. HuggingFace

**Use when:** You want access to thousands of open-source models via API or local inference.

#### Option A: HuggingFace Hub API (Recommended)

**Setup:**
```bash
# 1. Create account: https://huggingface.co
# 2. Get token: https://huggingface.co/settings/tokens
# 3. Add to .env
echo "HUGGINGFACE_API_KEY=your_token_here" >> .env
echo "HUGGINGFACE_MODEL=meta-llama/Meta-Llama-3-8B-Instruct" >> .env
```

**Usage:**
```python
from app.llm.factory import build_llm

# Use via main factory (prefix with "hf:")
llm = build_llm("hf:meta-llama/Meta-Llama-3-8B-Instruct")

# Or use directly
from app.llm.huggingface_factory import build_huggingface_llm
llm = build_huggingface_llm("mistralai/Mistral-7B-Instruct-v0.2")
```

**Popular Models:**
- `meta-llama/Meta-Llama-3-8B-Instruct` - Llama 3 (8B)
- `meta-llama/Meta-Llama-3-70B-Instruct` - Llama 3 (70B)
- `mistralai/Mistral-7B-Instruct-v0.2` - Mistral 7B
- `mistralai/Mixtral-8x7B-Instruct-v0.1` - Mixtral MoE
- `google/flan-t5-xxl` - T5 instruction-tuned
- `tiiuae/falcon-7b-instruct` - Falcon 7B

#### Option B: Local HuggingFace Models

**Setup:**
```bash
# Install dependencies
pip install transformers torch accelerate

# Download model
huggingface-cli download meta-llama/Meta-Llama-3-8B-Instruct
```

**Usage:**
```python
from app.llm.huggingface_factory import build_huggingface_llm

# Use local model (requires GPU for good performance)
llm = build_huggingface_llm(
    "meta-llama/Meta-Llama-3-8B-Instruct",
    use_local=True
)
```

**Note:** Local inference requires significant GPU memory (8B models need ~16GB VRAM).

---

## Environment Configuration

Update your `.env` file with the API keys you plan to use:

```bash
# Ollama (optional - only if using remote server)
OLLAMA_URL=http://localhost:11434

# Google Gemini
GOOGLE_API_KEY=your_gemini_key_here
GEMINI_MODEL=gemini-2.0-flash

# OpenAI GPT
OPENAI_API_KEY=your_openai_key_here
GPT_MODEL=gpt-4o-mini

# HuggingFace
HUGGINGFACE_API_KEY=your_hf_token_here
HUGGINGFACE_MODEL=meta-llama/Meta-Llama-3-8B-Instruct

# Model selection for experts and judge
EXPERT_MODEL=llama3.1:8b  # Can be any supported model
JUDGE_MODEL=llama3.1:8b   # Can be any supported model
```

---

## Advanced Usage

### Temperature Control

Temperature controls randomness in responses:
- `0.0` - Deterministic, focused (best for factual tasks)
- `0.2-0.5` - Balanced (default for medical analysis)
- `0.7-1.0` - Creative (better for brainstorming)
- `1.0+` - Very creative (experimental)

```python
# Low temperature for medical decisions (deterministic)
llm = build_llm("gpt-4o-mini", temperature=0.1)

# Higher temperature for creative tasks
llm = build_llm("gemini-2.0-flash", temperature=0.8)
```

### Using Different Models for Different Agents

```python
from app.llm.factory import build_llm

# Use GPT-4 for judge (highest quality)
judge_llm = build_llm("gpt-4o", temperature=0.2)

# Use GPT-4o-mini for experts (cost-effective)
expert_llm = build_llm("gpt-4o-mini", temperature=0.2)

# Or mix providers
judge_llm = build_llm("gemini-1.5-pro")
expert_llm = build_llm("gpt-4o-mini")
```

### Direct Factory Usage

```python
# Import specific factory
from app.llm.gpt_factory import build_gpt_llm
from app.llm.gemini_factory import build_gemini_llm
from app.llm.huggingface_factory import build_huggingface_llm

# Build with more control
gpt_llm = build_gpt_llm(
    model_name="gpt-4o",
    temperature=0.2
)

gemini_llm = build_gemini_llm(
    model_name="gemini-1.5-pro",
    temperature=0.3
)

hf_llm = build_huggingface_llm(
    model_name="mistralai/Mistral-7B-Instruct-v0.2",
    temperature=0.2,
    use_local=False
)
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

*Free for HF Hub API with rate limits; local inference is free but requires hardware.

---

## Troubleshooting

### "API key not found in environment"

Make sure your `.env` file is in the project root and contains the required keys:
```bash
# Check if .env exists
ls -la .env

# Check if keys are set
cat .env | grep API_KEY
```

### "Model not found" (Ollama)

Pull the model first:
```bash
ollama pull llama3.1:8b
ollama list  # See available models
```

### "Rate limit exceeded"

- **Gemini Free Tier:** 15 requests/minute. Add delays between requests.
- **HuggingFace Free:** Limited rate. Consider paid plan or local inference.
- **OpenAI:** Set up billing and increase rate limits.

### "Out of memory" (Local HuggingFace)

Use smaller models or quantized versions:
```python
# Use smaller model
llm = build_huggingface_llm("google/flan-t5-base", use_local=True)

# Or use CPU (slower but works with less memory)
# Requires setting device_map="cpu" in huggingface_factory.py
```

---

## Installation

Install required dependencies:

```bash
# Core dependencies
pip install -r requirements.txt

# For local HuggingFace inference (optional)
pip install transformers torch accelerate
```

---

## Examples

See `test_init.py` and `main.py` for working examples of the LLM factories in action.

For the complete medical decision system, see the agent implementations in `app/agents/`.

---

## Contributing

When adding support for new LLM providers:

1. Create a new factory file: `app/llm/{provider}_factory.py`
2. Implement `build_{provider}_llm()` function
3. Update `app/llm/factory.py` to route to your factory
4. Add required API keys to `config/settings.py`
5. Update this README with usage instructions
6. Add dependencies to `requirements.txt`

Follow the pattern established in `gemini_factory.py` and `gpt_factory.py`.
