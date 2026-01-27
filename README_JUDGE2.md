# BLUEmed Judge 2.0 Branch

## Overview

The `judge2.0` branch introduces significant improvements to the BLUEmed medical error detection system, including enhanced multi-agent decision making, comprehensive evaluation metrics, and agent performance analysis.

---

## What's Changed from Main

### Summary of Changes

| Category | Files Changed | Description |
|----------|---------------|-------------|
| **New Features** | 4 new modules | Judge retrieval, weighted voting, agent tracking |
| **Evaluation** | 12 new scripts | Comprehensive metrics, fairness analysis, agent performance |
| **Configuration** | settings.py | New toggles for features |
| **Core Pipeline** | 6 modified files | Enhanced expert/judge workflow |

### Commits in judge2.0

1. **`900021d`** - Add judge retrieval and weighted voting features
2. **`9f79ed2`** - Add agent performance tracking and weight optimization to evaluation

---

## New Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Medical Note Input                            │
└─────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     RAG Retrieval (Optional)                         │
│  ┌─────────────────────┐       ┌─────────────────────┐              │
│  │   Mayo Clinic DB    │       │     WebMD DB        │              │
│  │   (Expert A)        │       │   (Expert B)        │              │
│  └─────────────────────┘       └─────────────────────┘              │
└─────────────────────────────────────────────────────────────────────┘
                                   │
                    ┌──────────────┴──────────────┐
                    ▼                              ▼
┌───────────────────────────┐    ┌───────────────────────────┐
│       EXPERT A            │    │       EXPERT B            │
│   (Mayo Clinic Source)    │    │   (WebMD Source)          │
│                           │    │                           │
│  • Analyzes medical note  │    │  • Analyzes medical note  │
│  • Retrieves evidence     │    │  • Retrieves evidence     │
│  • Provides wrong/correct │    │  • Provides wrong/correct │
│    term pairs             │    │    term pairs             │
│  • Classification +       │    │  • Classification +       │
│    confidence             │    │    confidence             │
└───────────────────────────┘    └───────────────────────────┘
                    │                              │
                    └──────────────┬───────────────┘
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    JUDGE (with Optional Retrieval)                   │
│                                                                      │
│  • Reviews both expert arguments                                     │
│  • [NEW] Can retrieve from BOTH sources to verify claims            │
│  • Determines winner (Expert A / Expert B / Neither)                │
│  • Makes final classification (CORRECT / INCORRECT)                 │
└─────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│               [NEW] WEIGHTED VOTING (Optional)                       │
│                                                                      │
│  • Judge:     40% weight (configurable)                             │
│  • Expert A:  30% weight (configurable)                             │
│  • Expert B:  30% weight (configurable)                             │
│                                                                      │
│  weighted_score = (judge * 0.4) + (expert_a * 0.3) + (expert_b * 0.3)│
│  Final = INCORRECT if weighted_score >= 0.5 else CORRECT            │
└─────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         FINAL OUTPUT                                 │
│                                                                      │
│  • predicted_label: 0 (CORRECT) or 1 (INCORRECT)                    │
│  • winner: "Expert A" / "Expert B" / "Neither"                      │
│  • confidence_score: 1-10                                           │
│  • reasoning: Judge's explanation                                   │
│  • expert_a / expert_b: Full argument details                       │
└─────────────────────────────────────────────────────────────────────┘
```

---

## New Features

### 1. Judge Knowledge Retrieval

The judge can now access both Mayo Clinic and WebMD knowledge bases to verify expert claims.

**Enable:**
```env
USE_JUDGE_RETRIEVER=True
```

**How it works:**
- Judge receives arguments from both experts
- Retrieves relevant documents from BOTH sources
- Cross-references expert claims with source knowledge
- Makes more informed decisions

### 2. Weighted Voting System

Ensemble decision-making with configurable weights.

**Enable:**
```env
USE_WEIGHTED_VOTING=True
WEIGHT_JUDGE=0.4        # 40%
WEIGHT_EXPERT_A=0.3     # 30%
WEIGHT_EXPERT_B=0.3     # 30%
```

**Decision Formula:**
```
weighted_score = (judge_score * 0.4) + (expert_a_score * 0.3) + (expert_b_score * 0.3)
final_decision = "INCORRECT" if weighted_score >= 0.5 else "CORRECT"
```

### 3. Agent Performance Tracking

New evaluation system tracks which expert performs better.

**Run evaluation:**
```bash
python evaluation/scripts/evaluate.py --results-dir logs/debates
```

**Output includes:**
- Total cases per expert winner
- Accuracy when each expert wins
- Error type breakdown per expert
- Recommended optimal weights

### 4. Confidence Score Calculation

The judge evaluates experts using a structured scoring system:

#### Decision Process

```
1. If BOTH experts agree → Use their consensus with high confidence

2. If experts disagree, evaluate each expert on:
   A. Did they quote a specific wrong term? (0 or 1)
   B. Did they name a specific correct term? (0 or 1)
   C. Are the terms mutually exclusive (not synonyms)? (0 or 1)
   D. Did they explain clinical impact? (0 or 1)
   E. Were they consistent across rounds? (0 or 1)

3. Winner = expert with higher score (if tied, prefer CORRECT classification)
```

#### Expert Scoring Criteria (0-5 points each)

| Criterion | Question | Points |
|-----------|----------|--------|
| **A** | Did they quote a specific wrong term? | 0 or 1 |
| **B** | Did they name a specific correct term? | 0 or 1 |
| **C** | Are the terms mutually exclusive (not synonyms)? | 0 or 1 |
| **D** | Did they explain clinical impact? | 0 or 1 |
| **E** | Were they consistent across rounds? | 0 or 1 |

**Synonym Consideration:** If an expert claims INCORRECT but the terms could be synonyms or valid alternatives, their score on criterion C = 0.

#### Confidence Score (1-10)

| Scenario | Starting Score | Adjustments |
|----------|----------------|-------------|
| Both experts agree | **7** | +/- for argument strength |
| Experts disagree | **4** | +/- for reasoning quality |

**Adjustments:**
- Strong, well-supported arguments → Add points
- Weak reasoning or missing evidence → Subtract points

#### Judge Output Format

```json
{
  "Final Answer": "CORRECT" or "INCORRECT",
  "Winner": "Expert A" or "Expert B" or "Neither",
  "Confidence Score": 1-10,
  "Expert A Score": 0-5,
  "Expert B Score": 0-5,
  "Score Breakdown": {
    "Expert A": {"A": 0/1, "B": 0/1, "C": 0/1, "D": 0/1, "E": 0/1},
    "Expert B": {"A": 0/1, "B": 0/1, "C": 0/1, "D": 0/1, "E": 0/1}
  },
  "Reasoning": "Brief explanation"
}
```

#### Weighted Voting Confidence (Optional)

When `USE_WEIGHTED_VOTING=True`, an additional confidence layer is calculated:

```python
# weighted_score: 0.0 = all CORRECT, 1.0 = all INCORRECT
weighted_score = (judge * 0.4) + (expert_a * 0.3) + (expert_b * 0.3)

# Confidence = how far from the 0.5 decision threshold
voting_confidence = abs(weighted_score - 0.5) * 2.0  # Scale to 0-100%
```

| Agreement | Weighted Score | Voting Confidence |
|-----------|----------------|-------------------|
| All agree INCORRECT | 1.0 | 100% |
| All agree CORRECT | 0.0 | 100% |
| 2 vs 1 split | ~0.7 | ~40% |
| Perfect split | 0.5 | 0% |

---

## Evaluation Output Example

```
============================================================
EVALUATION RESULTS
============================================================

Total Samples: 597
Correct Predictions: 368
Accuracy: 0.6164

--- Primary Metrics (Positive Class: INCORRECT) ---
Precision: 0.6213
Recall:    0.6752
F1 Score:  0.6471

--- Confusion Matrix ---
                  Predicted
                CORRECT  INCORRECT
Actual CORRECT      158      128
Actual INCORRECT    101      210

============================================================
AGENT/EXPERT PERFORMANCE ANALYSIS
============================================================

Total Cases: 597
Expert A Won: 443
Expert B Won: 125
Neither Won: 29

--- Expert A Performance ---
  Total Wins: 443
  Correct: 269
  Incorrect: 174
  Accuracy When Winning: 0.6072

  Error Types (Correct Cases):
    NA: 108
    diagnosis: 75
    management: 37
    treatment: 27

--- Expert B Performance ---
  Total Wins: 125
  Correct: 84
  Incorrect: 41
  Accuracy When Winning: 0.6720

============================================================
WEIGHT OPTIMIZATION ANALYSIS
============================================================

--- Current Expert Performance ---
  Expert A: 60.7% accuracy (443 wins)
  Expert B: 67.2% accuracy (125 wins)

--- Recommended Weights ---
  Expert A Weight: 47.5%
  Expert B Weight: 52.5%
  Reasoning: Based on relative accuracy: A=60.7% vs B=67.2%

--- Threshold Recommendation ---
  Expert B has higher accuracy (67.2% vs 60.7%).
  Consider weight_a=0.47, weight_b=0.53
  Accuracy difference: 6.5%

--- Expert A Strengths (by error type) ---
  treatment: 71.1% vs 46.2% (+24.9%)

--- Expert B Strengths (by error type) ---
  management: 66.7% vs 54.4% (+12.3%)
  diagnosis: 91.3% vs 85.2% (+6.1%)

============================================================
```

---

## File Changes Detail

### New Files Created

| File | Description |
|------|-------------|
| `app/rag/judge_retriever.py` | Judge knowledge retrieval from both sources |
| `app/utils/weighted_voting.py` | Weighted voting system implementation |
| `evaluation/scripts/evaluate.py` | Comprehensive evaluation with agent tracking |
| `evaluation/scripts/batch_predict.py` | Batch prediction runner |
| `evaluation/scripts/fairness_evaluation.py` | Demographic fairness analysis |
| `evaluation/scripts/demographic_analysis.py` | Demographic breakdown |
| `evaluation/scripts/error_analysis_with_plots.py` | Error analysis visualizations |
| `evaluation/scripts/mcnemar-test.py` | Statistical significance testing |
| `docs/NEW_FEATURES_GUIDE.md` | Feature documentation |

### Modified Files

| File | Changes |
|------|---------|
| `config/settings.py` | Added `USE_JUDGE_RETRIEVER`, `USE_WEIGHTED_VOTING`, weight settings |
| `app/core/state.py` | Added fields for judge retrieval and weighted voting results |
| `app/graph/graph.py` | Added weighted voting node to pipeline |
| `app/agents/expertA.py` | Enhanced argument extraction |
| `app/agents/expertB.py` | Enhanced argument extraction |
| `app/core/prompts.py` | Refined expert and judge prompts |

---

## Configuration Options

### Full .env Configuration

```env
# Model Selection
EXPERT_MODEL=gemini-2.0-flash      # or gpt-4o, gpt-4o-mini
JUDGE_MODEL=gemini-2.0-flash       # or gpt-4o, gpt-4o-mini

# API Keys
GOOGLE_API_KEY=your-gemini-key
OPENAI_API_KEY=your-openai-key     # Optional, for GPT models

# RAG Settings
USE_RETRIEVER=True                  # Enable expert RAG retrieval
USE_JUDGE_RETRIEVER=False           # [NEW] Enable judge retrieval

# Weighted Voting Settings
USE_WEIGHTED_VOTING=False           # [NEW] Enable weighted voting
WEIGHT_JUDGE=0.4                    # Judge weight (40%)
WEIGHT_EXPERT_A=0.3                 # Expert A weight (30%)
WEIGHT_EXPERT_B=0.3                 # Expert B weight (30%)

# Embedding Settings
EMBEDDING_MODEL=gemini-embedding-001
GOOGLE_API_KEY_EMBED=your-embed-key
```

---

## Running the Pipeline

### 1. Single Case Test
```bash
python main.py
```

### 2. Batch Prediction
```bash
python evaluation/scripts/batch_predict.py \
  --test-file test_data/test_cases.json \
  --output-dir logs/debates \
  --chunk-size 20
```

### 3. Run Evaluation
```bash
python evaluation/scripts/evaluate.py \
  --results-dir logs/debates \
  --output-dir evaluation/results
```

### 4. Fairness Analysis
```bash
python evaluation/scripts/fairness_evaluation.py \
  --results-dir logs/debates
```

---

## Key Metrics Tracked

| Metric | Description |
|--------|-------------|
| **Accuracy** | Overall prediction accuracy |
| **Precision** | True positives / (True + False positives) |
| **Recall** | True positives / (True + False negatives) |
| **F1 Score** | Harmonic mean of precision and recall |
| **ROC-AUC** | Area under ROC curve (requires sklearn) |
| **PR-AUC** | Area under precision-recall curve |
| **Expert A Wins** | Cases where Expert A's argument won |
| **Expert B Wins** | Cases where Expert B's argument won |
| **Accuracy per Winner** | How accurate each expert is when they win |
| **Recommended Weights** | Optimal weights based on performance |

---

## Migration from Main

To use judge2.0 features on your existing setup:

1. **Pull the branch:**
   ```bash
   git fetch origin judge2.0
   git checkout judge2.0
   ```

2. **Update .env (optional new features):**
   ```env
   USE_JUDGE_RETRIEVER=True
   USE_WEIGHTED_VOTING=True
   ```

3. **Re-run evaluation:**
   ```bash
   python evaluation/scripts/evaluate.py --results-dir logs/debates
   ```

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 2.0 | 2026-01-27 | Judge retrieval, weighted voting, agent performance tracking |
| 1.0 | 2026-01-24 | Initial hybrid RAG with expert debate |

---

**Branch:** `judge2.0`
**Last Updated:** 2026-01-27
