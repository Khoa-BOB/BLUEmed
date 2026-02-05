#!/bin/bash

#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=bigTiger
#SBATCH --gres=gpu:1
#SBATCH --time=0-10:00:00
#SBATCH --nodelist=itiger03 
#SBATCH --job-name=batch-predict
#SBATCH --output=/project/syou/nlp-final/BLUEmed/evaluation/slurm/logs/output_batch-predict-%j.log
#SBATCH --error=/project/syou/nlp-final/BLUEmed/evaluation/slurm/logs/error_batch-predict-%j.log

# ==============================================================================
# BLUEmed Batch Prediction SLURM Script for iTiger HPC
# ==============================================================================
# Supports Multiple LLM Providers:
#   - Ollama (local): Set EXPERT_MODEL/JUDGE_MODEL to "llama3.1:8b" etc.
#   - Google Gemini: Set to "gemini-2.0-flash" (requires GOOGLE_API_KEY)
#   - OpenAI GPT: Set to "gpt-4o", "gpt-4o-mini" (requires OPENAI_API_KEY)
#   - HuggingFace: Set to "hf:meta-llama/Meta-Llama-3-8B-Instruct" (requires HUGGINGFACE_API_KEY)
#
# HuggingFace Cache:
#   - Cache location set via TRANSFORMERS_CACHE and HF_HOME environment variables
#   - Default: $(pwd)/hf_cache (inside project directory)
#   - Models downloaded once and reused across runs
#
# Usage:
#   sbatch evaluation/slurm/batch_predict.sh
#
# With custom parameters:
#   sbatch evaluation/slurm/batch_predict.sh --test-file test_data/test.json
#
# Monitor job:
#   squeue -u $USER
#   tail -f evaluation/slurm/logs/output_batch-predict-<jobid>.log
# ==============================================================================


# ===== SLURM RESOURCE GUIDE =====
# Available GPUs:
#   itiger01:    gpu:h100_80gb:N  (8 available, 80GB each) - FASTEST
#   itiger02-06: gpu:rtx_6000:N   (8 each, ~48GB each)
#   itiger07-11: gpu:rtx_5000:N   (8 each, ~16GB each)
#
# Examples:
#   --gres=gpu:h100_80gb:2 --nodelist=itiger01      # 2x H100
#   --gres=gpu:rtx_6000:4  --nodelist=itiger03      # 4x RTX 6000
#   --gres=gpu:rtx_5000:2  --nodelist=itiger07      # 2x RTX 5000
#
# Memory: 64G-512G available per node
# CPUs: 64 per node (use ~8 per GPU)

###############################################################################


# Record start time
START_TIME=$(date +%s)
START_DATETIME=$(date)

# Print job info
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $START_DATETIME"
echo "Working Directory: $(pwd)"
echo "=========================================="

# Navigate to project directory (adjust this path for your HPC setup)
PROJECT_DIR="${PROJECT_DIR:-/project/syou/nlp-final/BLUEmed}"
cd "$PROJECT_DIR" || { echo "ERROR: Cannot cd to $PROJECT_DIR"; exit 1; }
echo "Working in: $(pwd)"


# ------------------------------------------------------------------------------
# Virtual Environment Setup
# ------------------------------------------------------------------------------

# Conda setup
ENV_NAME="${ENV_NAME:-bluemed_env}"
# PYTHON_VERSION="${PYTHON_VERSION:-3.11}"


# Initialize conda in the shell session
source ~/.bashrc
source "$(conda info --base)/etc/profile.d/conda.sh"

echo "Activating environment '$ENV_NAME'"
conda activate $ENV_NAME
echo "Using conda env: $CONDA_DEFAULT_ENV"


# Verify Python environment
echo "Python: $(which python)"
echo "Python version: $(python --version)"

#HF cache in specified folder
CACHE_DIR="${CACHE_DIR:-$(pwd)/hf_cache}"
mkdir -p "$CACHE_DIR"
export TRANSFORMERS_CACHE="$CACHE_DIR"
export HF_HOME="$CACHE_DIR"
echo "Using Hugging Face cache at: $CACHE_DIR"

# ------------------------------------------------------------------------------
# Environment Variables
# ------------------------------------------------------------------------------

# Load environment variables from .env file
if [ -f ".env" ]; then
    echo "Loading environment variables from .env..."
    set -a
    source <(grep -v '^#' .env | sed 's/#.*//' | sed 's/[[:space:]]*$//')
    set +a
else
    echo "WARNING: .env file not found!"
    echo "Please create .env with GOOGLE_API_KEY and other required variables."
    exit 1
fi
# Verify API key is set (don't print the actual key)
if [ -z "$GOOGLE_API_KEY" ]; then
    echo "WARNING: GOOGLE_API_KEY not set in .env (required for Gemini)"
fi

if [ -z "$OPENAI_API_KEY" ]; then
    echo "WARNING: OPENAI_API_KEY not set in .env (required for GPT models)"
fi

if [ -z "$HUGGINGFACE_API_KEY" ]; then
    echo "WARNING: HUGGINGFACE_API_KEY not set in .env (required for HF inference API)"
fi

# Check which model is being used
echo ""
echo "Model Configuration:"
echo "  EXPERT_MODEL: $EXPERT_MODEL"
echo "  JUDGE_MODEL: $JUDGE_MODEL"
echo ""

# Determine which API key is needed
if [[ "$EXPERT_MODEL" == gemini* ]] || [[ "$JUDGE_MODEL" == gemini* ]]; then
    if [ -z "$GOOGLE_API_KEY" ]; then
        echo "ERROR: Gemini model selected but GOOGLE_API_KEY not set"
        exit 1
    else
        echo "Google API key configured for Gemini"
    fi
fi

if [[ "$EXPERT_MODEL" == gpt-* ]] || [[ "$JUDGE_MODEL" == gpt-* ]]; then
    if [ -z "$OPENAI_API_KEY" ]; then
        echo "ERROR: GPT model selected but OPENAI_API_KEY not set"
        exit 1
    else
        echo "OpenAI API key configured for GPT"
    fi
fi

if [[ "$EXPERT_MODEL" == hf:* ]] || [[ "$JUDGE_MODEL" == hf:* ]]; then
    if [ -z "$HUGGINGFACE_API_KEY" ]; then
        echo "ERROR: HuggingFace model selected but HUGGINGFACE_API_KEY not set"
        exit 1
    else
        echo "HuggingFace API key configured for HuggingFace"
    fi
fi

# ------------------------------------------------------------------------------
# Run Parameters (can be overridden via command line)
# ------------------------------------------------------------------------------

# Default values
TEST_FILE="${TEST_FILE:-test_data/test.json}"
OUTPUT_DIR="${OUTPUT_DIR:-logs/zeroshot/debates_multiagents_gpt5.2}"
CHUNK_SIZE="${CHUNK_SIZE:-20}"
DELAY="${DELAY:-30.0}"  # API rate limit delay in seconds

# Parse command line arguments passed to sbatch
while [[ $# -gt 0 ]]; do
    case $1 in
        --test-file)
            TEST_FILE="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --chunk-size)
            CHUNK_SIZE="$2"
            shift 2
            ;;
        --delay)
            DELAY="$2"
            shift 2
            ;;
        --reset)
            RESET_FLAG="--reset"
            shift
            ;;
        *)
            shift
            ;;
    esac
done


# Print run configuration
echo ""
echo "=========================================="
echo "Run Configuration:"
echo "  Test File: $TEST_FILE"
echo "  Output Dir: $OUTPUT_DIR"
echo "  Full Output Path: $(pwd)/$OUTPUT_DIR"
echo "  Chunk Size: $CHUNK_SIZE"
echo "  Delay: ${DELAY}s"
echo "  Reset: ${RESET_FLAG:-yes}"
echo "=========================================="
echo ""

# Verify test file exists
if [ ! -f "$TEST_FILE" ]; then
    echo "ERROR: Test file not found: $TEST_FILE"
    echo "Full path: $(pwd)/$TEST_FILE"
    echo "Available test files:"
    ls -la test_data/*.json 2>/dev/null || echo "  No test files found in test_data/"
    exit 1
fi

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"
echo "Output directory verified: $OUTPUT_DIR"
echo ""

# ------------------------------------------------------------------------------
# Run Batch Prediction
# ------------------------------------------------------------------------------

echo "Starting batch prediction..."
echo ""

python3 evaluation/batch_predict.py \
    --test-file "$TEST_FILE" \
    --output-dir "$OUTPUT_DIR" \
    --chunk-size "$CHUNK_SIZE" \
    --delay "$DELAY" \
    $RESET_FLAG

# Capture exit code
EXIT_CODE=$?

# ------------------------------------------------------------------------------
# Post-processing
# ------------------------------------------------------------------------------
# Calculate total running time
END_TIME=$(date +%s)
END_DATETIME=$(date)
TOTAL_SECONDS=$((END_TIME - START_TIME))
HOURS=$((TOTAL_SECONDS / 3600))
MINUTES=$(((TOTAL_SECONDS % 3600) / 60))
SECONDS=$((TOTAL_SECONDS % 60))

echo ""
echo "=========================================="
echo "Batch prediction completed with exit code: $EXIT_CODE"
echo "Start Time:  $START_DATETIME"
echo "End Time:    $END_DATETIME"
echo "Total Time:  ${HOURS}h ${MINUTES}m ${SECONDS}s"

echo "=========================================="

# Run evaluation if prediction was successful
if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "Running evaluation..."
    python3 evaluation/evaluate.py --results-dir "$OUTPUT_DIR"
fi

# Print summary of results
echo ""
echo "Results saved to: $OUTPUT_DIR"
echo "Full path: $(pwd)/$OUTPUT_DIR"
echo "Evaluation results: evaluation/results/"

# List generated files with detailed info
echo ""
echo "Checking for result files..."
RESULT_COUNT=$(ls "$OUTPUT_DIR"/result_*.json 2>/dev/null | wc -l)
echo "Found $RESULT_COUNT result files in $OUTPUT_DIR"

if [ $RESULT_COUNT -gt 0 ]; then
    echo ""
    echo "Generated result files (last 10):"
    ls -lh "$OUTPUT_DIR"/result_*.json 2>/dev/null | tail -10

    echo ""
    echo "First result file:"
    ls "$OUTPUT_DIR"/result_*.json 2>/dev/null | head -1
else
    echo "WARNING: No result_*.json files found!"
    echo "Checking for any .json files in $OUTPUT_DIR:"
    ls -la "$OUTPUT_DIR"/*.json 2>/dev/null | tail -10

    echo ""
    echo "Directory contents:"
    ls -la "$OUTPUT_DIR" 2>/dev/null
fi

# Check checkpoint status
if [ -f "evaluation/checkpoint.json" ]; then
    echo ""
    echo "Checkpoint status:"
    python3 -c "import json; data=json.load(open('evaluation/checkpoint.json')); print(f\"Processed: {data['processed_count']} cases\"); print(f\"Sample IDs: {data['processed_ids'][:5]}...\")"
fi

# CRITICAL DEBUG: Check if files are being written elsewhere
echo ""
echo "Searching for result files in other locations..."
find . -name "result_*.json" -type f 2>/dev/null | head -20

exit $EXIT_CODE
