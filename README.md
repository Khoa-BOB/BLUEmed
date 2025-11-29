# BlueMed: Multi-Agent Debate System for Medical Error Detection

A sophisticated AI system that leverages Large Language Models (LLMs) and Retrieval-Augmented Generation (RAG) to detect terminology substitution errors in clinical notes through expert debate.

## Table of Contents

- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Implementation Details](#implementation-details)
- [Workflow](#workflow)
- [Methodology](#methodology)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Project Structure](#project-structure)
- [Documentation](#documentation)

---

## Overview

BlueMed implements a multi-agent debate system for medical error detection, based on the paper "Error Detection in Medical Note through Multi Agent Debate" (BioNLP 2025). The system uses two AI experts representing different medical knowledge sources (Mayo Clinic and WebMD) who engage in structured debate to identify **substitution errors** - cases where one medical term is incorrectly used in place of another.

### Key Features

- **Multi-Round Expert Debate**: Two AI experts engage in up to 2 rounds of debate
- **Asymmetric Knowledge Sources**: Expert A uses Mayo Clinic knowledge, Expert B uses WebMD
- **Hybrid RAG System**: Combines dense retrieval, sparse retrieval (BM25), and online search
- **Query Decomposition**: Breaks complex medical notes into focused queries for better retrieval
- **Safety Layer**: Rule-based validation to prevent false positives
- **Consensus Detection**: Automatically skips Round 2 if experts agree in Round 1
- **Independent Judge**: Makes final decisions based solely on expert arguments
- **Comprehensive Logging**: Full debate transcripts saved as JSON for auditing

---

## System Architecture

### High-Level Architecture

```
┌─────────────────┐
│  Medical Note   │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│   Smart Hybrid Retriever (Optional)     │
│  ┌────────────────────────────────────┐ │
│  │  Query Decomposition               │ │
│  ├────────────────────────────────────┤ │
│  │  Dense Search (Vector/Chroma)      │ │
│  │  Sparse Search (BM25)              │ │
│  │  Online Search (Direct Crawler)    │ │
│  ├────────────────────────────────────┤ │
│  │  Reciprocal Rank Fusion (RRF)     │ │
│  └────────────────────────────────────┘ │
└─────────┬──────────────────┬────────────┘
          │                  │
          ▼                  ▼
    ┌─────────┐        ┌─────────┐
    │Expert A │        │Expert B │
    │ (Mayo)  │        │ (WebMD) │
    └────┬────┘        └────┬────┘
         │                  │
         └─────────┬────────┘
                   ▼
         ┌──────────────────┐
         │ Consensus Check  │
         └─────────┬────────┘
                   │
         ┌─────────┴─────────┐
         │                   │
    (Agree)              (Disagree)
         │                   │
         ▼                   ▼
    ┌────────┐         ┌──────────┐
    │ Judge  │         │ Round 2  │
    └────────┘         └────┬─────┘
                            │
                            ▼
                       ┌────────┐
                       │ Judge  │
                       └────┬───┘
                            │
                            ▼
                  ┌──────────────────┐
                  │ Hybrid Safety    │
                  │ Layer Validation │
                  └────────┬─────────┘
                           │
                           ▼
                  ┌──────────────────┐
                  │ Final Decision:  │
                  │ CORRECT/INCORRECT│
                  └──────────────────┘
```

### Component Overview

| Component | Purpose | Key Technology |
|-----------|---------|----------------|
| **LangGraph StateGraph** | Orchestrates debate workflow | LangGraph |
| **Expert Agents** | Analyze medical notes and debate | Google Gemini 2.0 Flash |
| **Judge Agent** | Evaluates arguments and decides | Google Gemini 2.0 Flash |
| **Smart Hybrid Retriever** | Retrieves relevant medical knowledge | Chroma + BM25 + Direct Crawler |
| **Safety Layer** | Rule-based validation | Custom Python logic |
| **Vector Database** | Stores embedded medical documents | ChromaDB |

---

## Implementation Details

### 1. LangGraph State Management

The system uses a `MedState` TypedDict that flows through the debate graph:

```python
MedState = {
    messages: List[AnyMessage],           # LangGraph message history
    medical_note: str,                    # Input clinical note
    current_round: int,                   # Current debate round (1-2)
    max_rounds: int,                      # Maximum rounds (default: 2)
    expertA_arguments: List[DebateArgument],
    expertA_retrieved_docs: List,
    expertB_arguments: List[DebateArgument],
    expertB_retrieved_docs: List,
    judge_decision: Optional[str],
    final_answer: Optional[str]           # "CORRECT" or "INCORRECT"
}
```

### 2. Expert Agents (A & B)

**Location**: `app/agents/expertA.py`, `app/agents/expertB.py`

**Key Responsibilities**:
- **Round 1**: Retrieve relevant medical documents and generate initial argument
- **Round 2**: Review opponent's Round 1 argument and generate counter-argument
- **Word Limit**: 300 words per argument
- **Knowledge Sources**: Expert A uses Mayo Clinic, Expert B uses WebMD

**Implementation**:
```python
def expertA_node(state: MedState, llm) -> dict:
    # Round 1: Retrieve documents using smart hybrid retriever
    if current_round == 1 and USE_RETRIEVER:
        retrieved_docs = smart_hybrid_retriever.retrieve_with_decomposition(
            note=medical_note,
            expert="A",
            k_per_query=2,
            max_total=5
        )
    # Round 2: Reuse Round 1 documents, include opponent's argument
    else:
        retrieved_docs = state.get("expertA_retrieved_docs", [])

    # Generate argument using LLM
    response = llm.invoke([system_message, user_prompt])

    return {
        "expertA_arguments": updated_arguments,
        "expertA_retrieved_docs": retrieved_docs
    }
```

### 3. Judge Agent

**Location**: `app/agents/judge.py`

**Key Responsibilities**:
- Evaluates all expert arguments (2-4 depending on consensus)
- **Does NOT see the medical note** - ensures independence
- Verifies correct application of the "Two-Term Rule"
- Applies hybrid safety layer for validation
- Outputs structured JSON decision

**Judge Decision Structure**:
```json
{
  "Final Answer": "CORRECT" or "INCORRECT",
  "Winner": "Expert A" or "Expert B",
  "Confidence Score": 1-10,
  "Reasoning": "Detailed explanation"
}
```

### 4. Hybrid RAG System

**Location**: `app/rag/smart_hybrid_retriever.py`

The RAG system combines three retrieval methods:

1. **Dense Retrieval** (Vector Search)
   - Uses ChromaDB with Google Gemini embeddings
   - Semantic similarity matching
   - Files: `chroma_retriever.py`, `vector_builder.py`

2. **Sparse Retrieval** (BM25)
   - Keyword-based matching
   - Uses rank-bm25 library
   - File: `hybrid_retriever.py`

3. **Online Search** (Direct Web Crawler)
   - Direct crawling of Mayo Clinic and WebMD search pages
   - Real-time web content extraction
   - No third-party search engines (avoids rate limiting)
   - Files: `online_search.py`, `direct_crawler.py`

**Query Decomposition**:
```python
def retrieve_with_decomposition(note, expert, k_per_query, max_total):
    # 1. Decompose medical note into focused queries
    queries = decompose_query(note)

    # 2. For each query, perform hybrid search
    all_docs = []
    for query in queries:
        dense_docs = chroma_retrieve(query)      # Vector search
        sparse_docs = bm25_retrieve(query)       # Keyword search
        online_docs = web_search(query)          # Web search

        # 3. Fuse results using Reciprocal Rank Fusion
        fused_docs = reciprocal_rank_fusion([dense_docs, sparse_docs, online_docs])
        all_docs.extend(fused_docs[:k_per_query])

    # 4. Deduplicate and return top results
    return deduplicate(all_docs)[:max_total]
```

### 5. Hybrid Safety Layer

**Location**: `app/utils/safety_rules.py`

Implements rule-based validation to prevent false positives:

**Core Function**: `hybrid_safety_predict()`

**Safety Checks**:
1. **Two-Term Rule**: Both wrong term and correct term must be present
2. **Culture/Lab Confirmation**: Lab results are definitive, not errors
3. **Process Issues**: Missing tests are process issues, not substitution errors
4. **Side Effects**: Medication reactions are expected, not diagnosis errors
5. **Hierarchical Terms**: Specific vs. general medical terms are valid
6. **Expert Consensus**: Strong agreement with evidence is trusted

**Example**:
```python
# If classification is INCORRECT but no term pairs found → override to CORRECT
if classification == "INCORRECT" and not (wrong_term and correct_term):
    return {
        "final_classification": "CORRECT",
        "rule_applied": "RULE_HARD_TWO_TERM",
        "reason": "No wrong+correct term pair detected."
    }
```

---

## Workflow

### Complete Execution Flow

```
┌────────────────────────────────────────────────┐
│ 1. INITIALIZATION                              │
│    - Build LangGraph                           │
│    - Initialize LLM models                     │
│    - Pre-load retriever (if enabled)           │
└────────────┬───────────────────────────────────┘
             │
             ▼
┌────────────────────────────────────────────────┐
│ 2. INPUT                                       │
│    - Receive medical note from user            │
│    - Create initial state                      │
└────────────┬───────────────────────────────────┘
             │
             ▼
┌────────────────────────────────────────────────┐
│ 3. ROUND 1 (Parallel Execution)                │
│    ┌──────────────────┐  ┌──────────────────┐  │
│    │ Expert A         │  │ Expert B         │  │
│    │ - Retrieve Mayo  │  │ - Retrieve WebMD │  │
│    │ - Generate arg   │  │ - Generate arg   │  │
│    └──────────────────┘  └──────────────────┘  │
└────────────┬───────────────────────────────────┘
             │
             ▼
┌────────────────────────────────────────────────┐
│ 4. CONSENSUS CHECK                             │
│    - Compare Expert A & B classifications      │
│    ┌─────────────┐         ┌────────────────┐  │
│    │ Both Agree  │         │ Disagree       │  │
│    │ → Skip R2   │         │ → Continue R2  │  │
│    └──────┬──────┘         └────────┬───────┘  │
└───────────┼─────────────────────────┼──────────┘
            │                         │
            │                         ▼
            │         ┌────────────────────────────┐
            │         │ 5. ROUND 2 (Parallel)      │
            │         │    ┌──────────────────┐    │
            │         │    │ Expert A         │    │
            │         │    │ - See B's arg    │    │
            │         │    │ - Counter-argue  │    │
            │         │    └──────────────────┘    │
            │         │    ┌──────────────────┐    │
            │         │    │ Expert B         │    │
            │         │    │ - See A's arg    │    │
            │         │    │ - Counter-argue  │    │
            │         │    └──────────────────┘    │
            │         └────────────┬───────────────┘
            │                      │
            └──────────────────────┘
                       │
                       ▼
┌────────────────────────────────────────────────┐
│ 6. JUDGE EVALUATION                            │
│    - Review all 2-4 arguments                  │
│    - Verify two-term rule application          │
│    - Apply hybrid safety layer                 │
│    - Generate JSON decision                    │
└────────────┬───────────────────────────────────┘
             │
             ▼
┌────────────────────────────────────────────────┐
│ 7. OUTPUT & LOGGING                            │
│    - Display final decision                    │
│    - Save debate log to JSON                   │
│    - Return: CORRECT or INCORRECT              │
└────────────────────────────────────────────────┘
```

### Typical Execution Time

| Stage | Duration | Notes |
|-------|----------|-------|
| Initialization | 2-5s | One-time setup |
| RAG Retrieval (Round 1) | 3-8s per expert | Parallel execution |
| Expert Argument (Round 1) | 2-4s per expert | Parallel execution |
| Consensus Check | <1s | Quick comparison |
| Round 2 (if needed) | 2-4s per expert | Parallel execution |
| Judge Evaluation | 2-5s | Sequential |
| Safety Layer | <1s | Rule-based checks |
| **Total** | **10-30s** | Varies by consensus |

---

## Methodology

### The "Two-Term Rule" Framework

The core decision framework requires experts to identify **exactly one type of error**:

**SUBSTITUTION ERROR**: One wrong medical term incorrectly replacing the correct term.

**Critical Constraint**: Both experts must quote:
1. The **wrong term** from the note
2. The **correct term** that should be used

**If either is missing → Classification: CORRECT**

### What is NOT Considered an Error

| Scenario | Why Not an Error | Example |
|----------|------------------|---------|
| **Premature diagnosis** | Unconfirmed ≠ wrong | Diagnosis made before lab results |
| **Missing tests** | Process issue, not substitution | Should have ordered CT scan |
| **Culture results** | Definitive confirmation | Culture confirms MRSA |
| **Medication side effects** | Expected reactions | Nausea from chemotherapy |
| **Wrong reasoning** | Process issue | Incorrect diagnostic logic |
| **Incomplete documentation** | Vague ≠ wrong | Missing clinical details |

### Expert Debate Strategy

**Round 1**: Initial Analysis
- Retrieve 5 relevant documents from respective sources
- Analyze medical note for substitution errors
- Clearly state: wrong term, correct term, classification
- Provide evidence from retrieved knowledge

**Round 2**: Counter-Argumentation (if needed)
- Review opponent's Round 1 argument
- Reuse documents from Round 1 (optimization)
- Address opponent's points directly
- Strengthen or revise own argument

### Judge Evaluation Criteria

The judge evaluates based on:

1. **Two-Term Rule Compliance**: Did the expert provide both wrong and correct terms?
2. **Evidence Quality**: Is the argument supported by medical knowledge?
3. **Logical Consistency**: Does the reasoning make sense?
4. **Specificity**: Are the terms clearly identified in the note?

**Judge does NOT**:
- See the original medical note
- Make independent medical assessments
- Override expert consensus with strong evidence

### Safety Layer Logic

```python
# Consensus Override (Highest Priority)
if both_experts_agree_INCORRECT and both_have_term_pairs:
    return "INCORRECT"  # Trust strong consensus

# Hard Two-Term Rule
if classification == "INCORRECT" and no_term_pairs_found:
    return "CORRECT"  # Override due to missing evidence

# Rule-Based Checks (if no term pairs)
if classification == "INCORRECT" and no_term_pairs:
    if culture_confirmation_in_note:
        return "CORRECT"  # Lab-confirmed, not error
    if process_issue_indicators >= 2:
        return "CORRECT"  # Process issue, not substitution
    if side_effect_discussion:
        return "CORRECT"  # Side effect, not diagnosis error

# Expert Disagreement
if experts_disagree and no_strong_evidence:
    return "CORRECT"  # Conservative approach

# Default: Trust judge's classification
return judge_classification
```

---

## Installation

### Prerequisites

- Python 3.12+
- Google Gemini API key (for LLM and embeddings)
- 2GB+ disk space (for vector databases)

### Setup Steps

1. **Clone the repository**:
```bash
git clone <repository-url>
cd BlueMed
```

2. **Create virtual environment**:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

4. **Configure environment variables**:

Create a `.env` file in the root directory:

```bash
# LLM Configuration
EXPERT_MODEL=gemini-2.0-flash
JUDGE_MODEL=gemini-2.0-flash

# Google API Keys
GOOGLE_API_KEY=your_gemini_api_key_here
GOOGLE_API_KEY_EMBED=your_gemini_api_key_here

# RAG Configuration
USE_RETRIEVER=True
EMBEDDING_MODEL=gemini-embedding-001
PERSIST_DIR=vectordb/chroma_gemini

# Rate Limiting (for embedding generation)
EMBED_BATCH_SIZE=5
EMBED_BATCH_DELAY=1.0
EMBED_RPM=15
EMBED_TPM=1000000

# Optional: Ollama (for local LLMs)
OLLAMA_URL=http://localhost:11434

# Optional: Force rebuild vector databases
FORCE_REBUILD=False
```

5. **Build vector databases** (if using RAG):

The system will automatically build vector databases on first run, or you can manually trigger:

```bash
python -c "from app.rag.vector_builder import build_all_vector_dbs; build_all_vector_dbs()"
```

This will index documents from:
- `BlueMed_data/MayoClinic/` → Mayo Clinic knowledge base
- `BlueMed_data/WebMD/` → WebMD knowledge base

---

## Usage

### CLI Interface

**Basic usage**:
```bash
python main.py
```

The system will prompt you to enter a medical note. Press Enter twice when done.

**Example session**:
```
================================================================================
MULTI-AGENT DEBATE SYSTEM FOR MEDICAL ERROR DETECTION
================================================================================

Building the debate graph...
Graph built successfully!

Enter the medical note to analyze (press Enter twice when done):
54-year-old woman with a painful, rapidly growing leg lesion for 1 month.
History includes Crohn's disease, diabetes, hypertension, and previous anterior uveitis.
Examination revealed a 4-cm tender ulcerative lesion with necrotic base and purplish borders,
along with pitting edema and dilated veins. Diagnosed as a venous ulcer.

================================================================================
STARTING MULTI-AGENT DEBATE
================================================================================

Round 1: Initial arguments from both experts...
[Expert A] Retrieving Mayo Clinic knowledge...
[Expert B] Retrieving WebMD knowledge...

✓ Round 1 Consensus Detected - Skipping Round 2

================================================================================
DEBATE RESULTS
================================================================================

[... Expert arguments and judge decision ...]

FINAL ANSWER: INCORRECT

📝 Debate log saved to: logs/debates/debate_20251129_143022.json
```

### Streamlit Web UI

**Launch the web interface**:
```bash
streamlit run ui/judge_ui/app.py
```

Navigate to `http://localhost:8501` in your browser.

**Features**:
- Interactive medical note input
- Real-time debate visualization
- Downloadable markdown reports
- Session history
- Dummy mode for testing

### Programmatic Usage

```python
from config.settings import settings
from app.graph.graph import build_graph
from langchain_core.messages import HumanMessage

# Build graph
graph = build_graph(settings)

# Create initial state
initial_state = {
    "messages": [HumanMessage(content="Analyze this medical note for errors")],
    "medical_note": "Your medical note here...",
    "current_round": 1,
    "max_rounds": 2,
    "expertA_arguments": [],
    "expertA_retrieved_docs": [],
    "expertB_arguments": [],
    "expertB_retrieved_docs": [],
    "judge_decision": None,
    "final_answer": None
}

# Run debate
result = graph.invoke(initial_state)

# Extract results
final_answer = result.get("final_answer")  # "CORRECT" or "INCORRECT"
judge_decision = result.get("judge_decision")
expert_a_args = result.get("expertA_arguments")
expert_b_args = result.get("expertB_arguments")
```

---

## Configuration

### Environment Variables

| Variable | Description | Default | Options |
|----------|-------------|---------|---------|
| `EXPERT_MODEL` | LLM model for experts | `gemini-2.0-flash` | Any Gemini/Ollama model |
| `JUDGE_MODEL` | LLM model for judge | `gemini-2.0-flash` | Any Gemini/Ollama model |
| `GOOGLE_API_KEY` | Gemini API key | Required | Your API key |
| `GOOGLE_API_KEY_EMBED` | Embedding API key | Required | Your API key |
| `USE_RETRIEVER` | Enable RAG system | `True` | `True` / `False` |
| `EMBEDDING_MODEL` | Embedding model | `gemini-embedding-001` | Gemini embedding models |
| `PERSIST_DIR` | Vector DB directory | `vectordb/chroma_gemini` | Any path |
| `FORCE_REBUILD` | Force rebuild DBs | `False` | `True` / `False` |
| `EMBED_BATCH_SIZE` | Embedding batch size | `5` | Integer |
| `EMBED_BATCH_DELAY` | Delay between batches | `1.0` | Float (seconds) |
| `EMBED_RPM` | Requests per minute | `15` | Integer |
| `EMBED_TPM` | Tokens per minute | `1000000` | Integer |

### Retrieval Configuration

Edit expert nodes to adjust retrieval parameters:

```python
# In app/agents/expertA.py or expertB.py
retrieved_docs = smart_hybrid_retriever.retrieve_with_decomposition(
    note=medical_note,
    expert="A",
    k_per_query=2,        # Documents per query
    max_total=5,          # Total documents
    use_dense=True,       # Vector search
    use_sparse=True,      # BM25 search
    use_online=True       # Web search
)
```

### Debate Configuration

Edit `main.py` to change debate parameters:

```python
initial_state = {
    # ...
    "max_rounds": 2,  # Change to 1 or 3 for different debate lengths
}
```

---

## Project Structure

```
BlueMed/
├── app/
│   ├── agents/
│   │   ├── expertA.py              # Expert A (Mayo Clinic)
│   │   ├── expertB.py              # Expert B (WebMD)
│   │   └── judge.py                # Judge agent
│   ├── core/
│   │   ├── state.py                # MedState TypedDict
│   │   └── prompts.py              # System prompts
│   ├── graph/
│   │   └── graph.py                # LangGraph workflow
│   ├── llm/
│   │   └── factory.py              # LLM initialization
│   ├── rag/
│   │   ├── smart_hybrid_retriever.py   # Main RAG system
│   │   ├── hybrid_retriever.py         # Retrieval fusion
│   │   ├── chroma_retriever.py         # Vector search
│   │   ├── vector_builder.py           # DB builder
│   │   ├── metadata_filter.py          # Strategy selection
│   │   ├── online_search.py            # Web search
│   │   └── direct_crawler.py           # Web scraping
│   └── utils/
│       └── safety_rules.py         # Hybrid safety layer
├── config/
│   └── settings.py                 # Pydantic settings
├── ui/
│   └── judge_ui/
│       └── app.py                  # Streamlit interface
├── BlueMed_data/
│   ├── MayoClinic/                 # Mayo Clinic knowledge
│   │   ├── symptoms/
│   │   ├── diseases_conditions/
│   │   └── drugs_supplements/
│   └── WebMD/                      # WebMD knowledge
├── vectordb/
│   └── chroma_gemini/              # Vector databases
│       ├── mayo_clinic/
│       └── webmd/
├── tests/                          # Test suite
├── logs/
│   └── debates/                    # Debate JSON logs
├── Notes/                          # Documentation
│   ├── QUICKSTART.md
│   ├── RUN_GUIDE.md
│   ├── DEBATE_SYSTEM.md
│   ├── HYBRID_RAG_GUIDE.md
│   └── GEMINI_SETUP.md
├── main.py                         # CLI entry point
├── requirements.txt                # Dependencies
├── .env                            # Environment variables
└── README.md                       # This file
```

---

## Documentation

Comprehensive guides are available in the `Notes/` directory:

| Document | Purpose |
|----------|---------|
| `QUICKSTART.md` | Quick setup and first run |
| `RUN_GUIDE.md` | Detailed execution instructions |
| `DEBATE_SYSTEM.md` | System architecture and design |
| `HYBRID_RAG_GUIDE.md` | RAG system deep dive |
| `GEMINI_SETUP.md` | Google Gemini API setup |
| `LOGGING.md` | Debate logging and analysis |
| `RATE_LIMIT_FIXES.md` | API rate limit handling |
| `DIRECT_CRAWLER_GUIDE.md` | Web crawling documentation |

---

## Example Use Cases

### 1. Detecting Terminology Substitution

**Input**:
```
Patient diagnosed with pseudoseizures based on EEG results.
```

**Expected Output**: `INCORRECT`
- Wrong term: "pseudoseizures"
- Correct term: "psychogenic non-epileptic seizures" or "PNES"

### 2. Process Issues (NOT an Error)

**Input**:
```
Patient diagnosed with bacterial infection without culture confirmation.
```

**Expected Output**: `CORRECT`
- Reason: Missing test is a process issue, not a substitution error
- No wrong term replacing a correct term

### 3. Lab-Confirmed Diagnosis (NOT an Error)

**Input**:
```
Culture results confirm MRSA infection.
```

**Expected Output**: `CORRECT`
- Reason: Lab confirmation is definitive, not a substitution error

---

## Performance Optimization

### Caching and Efficiency

1. **Document Reuse**: Round 1 documents are reused in Round 2
2. **Singleton Retriever**: Hybrid retriever initialized once before parallel execution
3. **Lazy BM25 Indexing**: Sparse retrievers build indices on-demand
4. **Consensus Detection**: Skips Round 2 when experts agree (saves ~40% time)

### Rate Limiting

Embedding generation respects API limits:
- Batches: 5 documents per batch
- Delay: 1 second between batches
- RPM: 15 requests per minute
- TPM: 1M tokens per minute

### Scalability

| Factor | Limit | Notes |
|--------|-------|-------|
| Medical Note Length | ~10K tokens | LLM context window |
| Retrieved Documents | 5-10 per expert | Balance accuracy/cost |
| Debate Rounds | 2 max | Most issues resolved |
| Vector DB Size | ~10K documents | Current knowledge base |

---

## Troubleshooting

### Common Issues

**Issue**: `GOOGLE_API_KEY not found`
- **Solution**: Create `.env` file with your API key

**Issue**: Vector database not found
- **Solution**: Run `python -c "from app.rag.vector_builder import build_all_vector_dbs; build_all_vector_dbs()"`

**Issue**: Rate limit errors
- **Solution**: Adjust `EMBED_BATCH_SIZE` and `EMBED_BATCH_DELAY` in `.env`

**Issue**: Experts always disagree
- **Solution**: Check prompts in `app/core/prompts.py` for clarity

**Issue**: Judge output not parsing
- **Solution**: Check `main.py:extract_judge_decision()` for JSON format handling

---

## Contributing

Contributions are welcome! Please ensure:
1. Code follows existing structure
2. Tests pass: `pytest tests/`
3. Documentation is updated
4. Commit messages are clear

---

## License

[Specify your license here]

---

## Citation

If you use this system in research, please cite:

```
@inproceedings{bluemed2025,
  title={Error Detection in Medical Note through Multi Agent Debate},
  author={[Authors]},
  booktitle={BioNLP 2025},
  year={2025}
}
```

---

## Contact

For questions or issues:
- Open an issue on GitHub
- Email: [your-email]
- Documentation: See `Notes/` directory

---

## Acknowledgments

- Medical knowledge sources: Mayo Clinic, WebMD
- LLM provider: Google Gemini
- Framework: LangChain, LangGraph
- Vector database: ChromaDB
