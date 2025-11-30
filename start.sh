#!/bin/bash

# BLUEmed - Streamlit UI Startup Script
# This script starts the Streamlit UI with direct graph execution (no API server needed)

set -e  # Exit on error

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "======================================================================"
echo "BLUEmed - Medical Note Analysis System"
echo "======================================================================"
echo ""
echo "Working directory: $(pwd)"
echo ""

# Check if .env exists
if [ ! -f .env ]; then
    echo "❌ Error: .env file not found!"
    echo "Please copy .env.example to .env and configure your settings."
    exit 1
fi

# Load environment variables
export $(grep -v '^#' .env | sed 's/#.*//' | grep -v '^[[:space:]]*$' | xargs)

echo "✓ Environment loaded"
echo ""

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"
echo ""

# Install/check dependencies
echo "Checking dependencies..."
if ! pip show streamlit > /dev/null 2>&1; then
    echo "Installing dependencies..."
    pip install -r requirements.txt
fi
echo "✓ Dependencies OK"
echo ""

# Create logs directory
mkdir -p logs/debates
echo "✓ Logs directory ready"
echo ""

# Start Streamlit UI
echo "======================================================================"
echo "Starting Streamlit UI (port 8501)..."
echo "======================================================================"
streamlit run ui/judge_ui/app.py --server.port 8501 --server.address localhost &
UI_PID=$!
echo "✓ Streamlit UI started (PID: $UI_PID)"
echo ""

echo "======================================================================"
echo "✅ BLUEmed is now running!"
echo "======================================================================"
echo ""
echo "🌐 Streamlit UI: http://localhost:8501"
echo "📁 Logs saved to: logs/debates/"
echo ""
echo "Press Ctrl+C to stop the service"
echo "======================================================================"
echo ""

# Create trap to kill process on exit
trap "echo ''; echo 'Shutting down...'; kill $UI_PID 2>/dev/null; exit 0" INT TERM

# Wait for the process
wait
