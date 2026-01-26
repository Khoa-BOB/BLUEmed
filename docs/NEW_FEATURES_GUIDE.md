# BLUEmed New Features Guide

## Overview

This document describes the new features added to the BLUEmed medical error detection system.

## Table of Contents

1. [New Features Summary](#new-features-summary)
2. [Feature 1: OpenAI GPT Support](#feature-1-openai-gpt-support)
3. [Feature 2: Judge Knowledge Retrieval](#feature-2-judge-knowledge-retrieval)
4. [Feature 3: Weighted Voting System](#feature-3-weighted-voting-system)
5. [Hardware Requirements](#hardware-requirements)
6. [Configuration](#configuration)
7. [Running the Pipeline](#running-the-pipeline)

---

## New Features Summary

### 1. **OpenAI GPT Support**
- Added support for OpenAI's GPT-4o and GPT-4o-mini models
- Can be used for expert agents and/or judge
- Provides state-of-the-art reasoning capabilities

### 2. **Judge Knowledge Retrieval (Optional)**
- Judge can now access both WebMD and Mayo Clinic knowledge bases
- Allows verification of expert arguments against source knowledge
- Configurable via `USE_JUDGE_RETRIEVER` flag

### 3. **Weighted Voting System (Optional)**
- Dynamic weight allocation across agents:
  - **Judge: 40%** (default)
  - **Expert A (Mayo Clinic): 30%** (default)
  - **Expert B (WebMD): 30%** (default)
- Fully configurable weights
- Provides ensemble decision-making

---

## Feature 1: OpenAI GPT Support

### How to Use

1. **Get OpenAI API Key**
   - Visit: https://platform.openai.com/api-keys

2. **Configure `.env` file**
   ```env
   OPENAI_API_KEY=sk-your-api-key-here

   # Use GPT-4o for experts and judge
   EXPERT_MODEL=gpt-4o
   JUDGE_MODEL=gpt-4o

   # Or use GPT-4o-mini for faster/cheaper inference
   # EXPERT_MODEL=gpt-4o-mini
   # JUDGE_MODEL=gpt-4o-mini
   ```

3. **Available Models**
   - `gpt-4o` - Latest GPT-4 Optimized (recommended)
   - `gpt-4o-mini` - Faster and more cost-efficient
   - `gpt-4-turbo` - GPT-4 Turbo
   - `gpt-3.5-turbo` - Legacy model

### Cost Considerations

| Model | Input | Output |
|-------|--------|---------|
| gpt-4o | $2.50/1M tokens | $10.00/1M tokens |
| gpt-4o-mini | $0.150/1M tokens | $0.600/1M tokens |

---

## Feature 2: Judge Knowledge Retrieval

### How It Works

1. Judge receives arguments from both experts
2. If `USE_JUDGE_RETRIEVER=True`, the judge retrieves relevant medical knowledge from both sources
3. Judge uses retrieved knowledge to verify expert claims
4. Final decision is based on arguments + evidence verification

### How to Enable

```env
USE_JUDGE_RETRIEVER=True
```

### Advantages

- **Improved Accuracy**: Judge can fact-check expert claims
- **Reduced Hallucination**: Grounded in source knowledge
- **Transparency**: Retrieved sources are logged in results

---

## Feature 3: Weighted Voting System

### How It Works

1. Each agent makes an independent classification (CORRECT or INCORRECT)
2. Classifications are converted to scores (0.0 = CORRECT, 1.0 = INCORRECT)
3. Weighted average is calculated:
   ```
   weighted_score = (judge_score * 0.4) + (expert_a_score * 0.3) + (expert_b_score * 0.3)
   ```
4. Final decision: INCORRECT if `weighted_score >= 0.5`, else CORRECT

### Customizing Weights

```env
USE_WEIGHTED_VOTING=True
WEIGHT_JUDGE=0.4        # 40%
WEIGHT_EXPERT_A=0.3     # 30%
WEIGHT_EXPERT_B=0.3     # 30%
```

### Output Example

```
WEIGHTED VOTING RESULTS
============================================================

Individual Classifications:
  Judge:    INCORRECT
  Expert A: INCORRECT (Mayo Clinic)
  Expert B: CORRECT (WebMD)

Weights Applied:
  Judge:    40.0%
  Expert A: 30.0%
  Expert B: 30.0%

Weighted Score: 0.700
  (0.0 = CORRECT, 1.0 = INCORRECT, threshold = 0.5)

Majority decision

WEIGHTED FINAL DECISION: INCORRECT
   Confidence: 40.0%
============================================================
```

---

## Hardware Requirements

### Can this run on 8GB RAM M2 Mac?

**Yes, with cloud APIs:**

- **Recommended**: Use `gemini-2.0-flash` or `gpt-4o-mini`
- **100 test cases**: 2-3 hours
- **Cost**: Free (Gemini) or ~$0.50-1 (GPT-4o-mini)

### Can this run on HPC GPU clusters (e.g., Tiger)?

**Yes, highly recommended:**

- Use local models (Llama 3.1, Mistral) with GPU acceleration
- 100 test cases can complete in 1-2 hours
- No API costs

---

## Configuration

### Configuration Options Summary

| Setting | Default | Description |
|---------|---------|-------------|
| `OPENAI_API_KEY` | "" | OpenAI API key for GPT models |
| `USE_JUDGE_RETRIEVER` | False | Enable judge knowledge retrieval |
| `USE_WEIGHTED_VOTING` | False | Enable weighted voting system |
| `WEIGHT_JUDGE` | 0.4 | Judge weight in voting |
| `WEIGHT_EXPERT_A` | 0.3 | Expert A (Mayo) weight |
| `WEIGHT_EXPERT_B` | 0.3 | Expert B (WebMD) weight |

### Example Configurations

#### Baseline (No new features)
```env
EXPERT_MODEL=gemini-2.0-flash
JUDGE_MODEL=gemini-2.0-flash
USE_RETRIEVER=True
USE_JUDGE_RETRIEVER=False
USE_WEIGHTED_VOTING=False
```

#### Full Features (GPT-4o + Judge Retrieval + Weighted Voting)
```env
EXPERT_MODEL=gpt-4o
JUDGE_MODEL=gpt-4o
USE_RETRIEVER=True
USE_JUDGE_RETRIEVER=True
USE_WEIGHTED_VOTING=True
WEIGHT_JUDGE=0.4
WEIGHT_EXPERT_A=0.3
WEIGHT_EXPERT_B=0.3
```

---

## Running the Pipeline

### Single Case Testing

```bash
python main.py
```

### Batch Prediction (100 Test Cases)

```bash
python evaluation/scripts/batch_predict.py \
  --test-file test_data/test_cases_100.json \
  --output-dir logs/debates \
  --chunk-size 20 \
  --delay 15.0
```

### Evaluation

```bash
python evaluation/evaluate.py
```

---

## Files Changed

### New Files Created
1. `app/rag/judge_retriever.py` - Judge knowledge retrieval module
2. `app/utils/weighted_voting.py` - Weighted voting system
3. `docs/NEW_FEATURES_GUIDE.md` - This documentation

### Files Modified
1. `config/settings.py` - Added new configuration options
2. `app/core/state.py` - Added fields for judge retrieval and weighted voting
3. `app/graph/graph.py` - Added weighted voting node
4. `evaluation/scripts/batch_predict.py` - Added new features to result output

---

## Troubleshooting

### Issue: OpenAI API Error

**Error:** `OPENAI_API_KEY not found in environment`

**Solution:**
```bash
# Add to .env file
OPENAI_API_KEY=sk-your-key-here
```

### Issue: Judge Retrieval Returns Empty Results

**Solution:**
```bash
# Ensure vector databases are built
python preprocessing/build_vectorstore.py
```

### Issue: Weighted Voting Not Applied

**Solution:**
```bash
# Verify in .env
USE_WEIGHTED_VOTING=True
```

---

**Last Updated:** 2026-01-26
**Version:** 2.0 (With Weighted Voting + Judge Retrieval + GPT Support)
