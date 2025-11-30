#!/bin/bash

# BLUEmed - Complete System Startup Script
# This script starts both the API server and the Streamlit UI

set -e  # Exit on error

echo "======================================================================"
echo "BLUEmed - Medical Note Analysis System"
echo "======================================================================"
echo ""

# Check if .env exists
if [ ! -f .env ]; then
    echo "❌ Error: .env file not found!"
    echo "Please copy .env.example to .env and configure your settings."
    exit 1
fi

# Load environment variables
export $(grep -v '^#' .env | xargs)

echo "✓ Environment loaded"
echo ""

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"
echo ""

# Install/check dependencies
echo "Checking dependencies..."
if ! pip show fastapi > /dev/null 2>&1; then
    echo "Installing dependencies..."
    pip install -r requirements.txt
fi
echo "✓ Dependencies OK"
echo ""

# Start API server in background
echo "======================================================================"
echo "Starting API Server (port 8000)..."
echo "======================================================================"
python3 api_server.py > logs/api_server.log 2>&1 &
API_PID=$!
echo "✓ API server started (PID: $API_PID)"
echo "  Logs: logs/api_server.log"
echo ""

# Wait for API to be ready
echo "Waiting for API server to be ready..."
for i in {1..30}; do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo "✓ API server is ready!"
        break
    fi
    if [ $i -eq 30 ]; then
        echo "❌ API server failed to start within 30 seconds"
        kill $API_PID 2>/dev/null
        exit 1
    fi
    sleep 1
    echo -n "."
done
echo ""
echo ""

# Start Streamlit UI
echo "======================================================================"
echo "Starting Streamlit UI (port 8501)..."
echo "======================================================================"
export API_URL="http://localhost:8000/analyze"
streamlit run ui/judge_ui/app.py --server.port 8501 --server.address localhost &
UI_PID=$!
echo "✓ Streamlit UI started (PID: $UI_PID)"
echo ""

echo "======================================================================"
echo "✅ BLUEmed is now running!"
echo "======================================================================"
echo ""
echo "🌐 Streamlit UI: http://localhost:8501"
echo "🔧 API Server:   http://localhost:8000"
echo "📊 API Docs:     http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop both services"
echo "======================================================================"
echo ""

# Create trap to kill both processes on exit
trap "echo ''; echo 'Shutting down...'; kill $API_PID $UI_PID 2>/dev/null; exit 0" INT TERM

# Wait for both processes
wait
