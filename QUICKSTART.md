# 🚀 Quick Start - BLUEmed UI Integration

## One-Command Startup

```bash
./start.sh
```

Then open: **http://localhost:8501**

## Manual Startup

**Terminal 1** - API Server:
```bash
python3 api_server.py
```

**Terminal 2** - Streamlit UI:
```bash
cd ui/judge_ui && streamlit run app.py
```

## Test Integration

```bash
# Start API server first, then run:
python3 test_integration.py
```

## Configuration

Edit `.env` file:
```bash
# Required
GOOGLE_API_KEY=your_key_here
EXPERT_MODEL=gemini-2.0-flash
JUDGE_MODEL=gemini-2.0-flash

# Optional
API_KEY=secret_key  # Leave empty for no auth
USE_RETRIEVER=true
```

## UI Modes

1. **Dummy Mode**: Works without backend (for testing UI)
2. **HTTP API Mode**: Full system with backend analysis

Switch in sidebar and configure:
- API URL: `http://localhost:8000/analyze`
- API Key: (if you set one in .env)

## Troubleshooting

**Port already in use?**
```bash
lsof -i :8000  # Check API port
lsof -i :8501  # Check UI port
kill <PID>     # Kill process if needed
```

**Dependencies missing?**
```bash
pip install -r requirements.txt
```

**Check API health:**
```bash
curl http://localhost:8000/health
```

## Full Documentation

See **[UI_INTEGRATION.md](./UI_INTEGRATION.md)** for complete guide.

---

**Quick Access:**
- 🌐 UI: http://localhost:8501
- 🔧 API: http://localhost:8000
- 📚 API Docs: http://localhost:8000/docs
