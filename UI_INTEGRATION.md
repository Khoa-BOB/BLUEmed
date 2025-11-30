# BLUEmed - UI Integration Guide

## Overview

The UI integration consists of three main components:

1. **Backend (main.py)**: Multi-agent debate system with LangGraph
2. **API Server (api_server.py)**: FastAPI REST API exposing the backend
3. **UI (ui/judge_ui/app.py)**: Streamlit web interface

## 🏗️ Architecture

```
┌─────────────────┐
│  Streamlit UI   │  (Port 8501)
│   app.py        │
└────────┬────────┘
         │ HTTP Requests
         ▼
┌─────────────────┐
│  FastAPI Server │  (Port 8000)
│  api_server.py  │
└────────┬────────┘
         │ Python Calls
         ▼
┌─────────────────┐
│  LangGraph      │
│  Backend        │
│  main.py        │
└─────────────────┘
```

## Quick Start

### Option 1: Automated Startup (Recommended)

```bash
# Make the startup script executable
chmod +x start.sh

# Run the complete system
./start.sh
```

This will:
- Start API server on port 8000
- Start Streamlit UI on port 8501
- Monitor both services
- Handle graceful shutdown (Ctrl+C)

### Option 2: Manual Startup

**Terminal 1 - API Server:**
```bash
python3 api_server.py
# Or with custom port:
# PORT=8080 python3 api_server.py
```

**Terminal 2 - Streamlit UI:**
```bash
cd ui/judge_ui
streamlit run app.py --server.port 8501
```

## Configuration

### Environment Variables (.env)

```bash
# Required
EXPERT_MODEL=gemini-2.0-flash
JUDGE_MODEL=gemini-2.0-flash
GOOGLE_API_KEY=your_key_here
OLLAMA_URL=http://localhost:11434
PERSIST_DIR=./vectordb

# Optional API Security
API_KEY=your_secret_key_here  # Leave empty for no auth

# RAG Configuration
USE_RETRIEVER=true
GOOGLE_API_KEY_EMBED=your_embed_key_here
EMBEDDING_MODEL=gemini-embedding-001
```

### UI Configuration

The Streamlit UI supports two modes:

1. **Dummy Mode** (no backend): For UI testing and demos
2. **HTTP API Mode**: Connects to real backend via FastAPI

Set the API URL in the UI sidebar or via environment:
```bash
export JUDGE_API_URL=http://localhost:8000/analyze
export JUDGE_API_KEY=your_secret_key_here  # if using API_KEY
```

## API Endpoints

### Health Check
```bash
GET /health
```

Response:
```json
{
  "status": "healthy",
  "models": {
    "expert": "gemini-2.0-flash",
    "judge": "gemini-2.0-flash"
  },
  "retriever_enabled": true
}
```

### Analyze Medical Note
```bash
POST /analyze
Content-Type: application/json
Authorization: Bearer <API_KEY>  # if configured

{
  "medical_note": "54-year-old woman with...",
  "max_rounds": 2
}
```

Response:
```json
{
  "request_id": "uuid-here",
  "medical_note": "...",
  "expertA_arguments": [
    {"round": 1, "content": "..."},
    {"round": 2, "content": "..."}
  ],
  "expertB_arguments": [...],
  "expertA_retrieved_docs": ["..."],
  "expertB_retrieved_docs": ["..."],
  "judge_decision": {
    "final_answer": "CORRECT",
    "confidence_score": 8,
    "winner": "Expert A",
    "reasoning": "..."
  },
  "model_info": {...}
}
```

## Testing

### Test API Server
```bash
# Health check
curl http://localhost:8000/health

# Analyze note (with auth)
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your_api_key" \
  -d '{
    "medical_note": "54-year-old woman with painful leg lesion...",
    "max_rounds": 2
  }'
```

### Test UI
1. Open browser to `http://localhost:8501`
2. Switch to "HTTP API" mode in sidebar
3. Enter API URL: `http://localhost:8000/analyze`
4. Enter API Key (if configured)
5. Submit a medical note

## Project Structure

```
project/repo/
├── api_server.py          # FastAPI backend server ⭐ NEW
├── main.py                # LangGraph debate system
├── start.sh               # Automated startup script ⭐ NEW
├── requirements.txt       # Python dependencies (updated)
├── .env                   # Configuration
├── .env.example           # Configuration template
├── app/
│   ├── agents/            # Expert A, Expert B, Judge
│   ├── graph/             # LangGraph workflow
│   ├── rag/               # Retrieval system
│   ├── core/              # State, prompts
│   └── utils/             # Safety rules
├── ui/
│   └── judge_ui/
│       ├── app.py         # Streamlit UI
│       ├── models.py      # Pydantic models
│       ├── judge_client.py # API client
│       └── utils.py       # Helper functions
├── config/
│   └── settings.py        # Settings management
└── logs/                  # Log files
```

## Security

### API Authentication

If `API_KEY` is set in `.env`, all API requests must include:
```
Authorization: Bearer <API_KEY>
```

### CORS Configuration

The API server allows all origins by default. For production:

```python
# In api_server.py
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://your-frontend.com"],  # Specify allowed origins
    ...
)
```

## Troubleshooting

### API Server won't start
- Check port 8000 is not in use: `lsof -i :8000`
- Check logs: `tail -f logs/api_server.log`
- Verify .env configuration

### UI can't connect to API
- Verify API URL in sidebar: `http://localhost:8000/analyze`
- Check API health: `curl http://localhost:8000/health`
- Verify API key if authentication is enabled

### Graph initialization fails
- Check GOOGLE_API_KEY is valid
- Verify OLLAMA_URL if using Ollama models
- Check embedding model is accessible

### RAG retrieval errors
- Verify vectordb exists: `ls -la vectordb/`
- Check GOOGLE_API_KEY_EMBED is set
- Ensure embedding model matches configuration

## Monitoring

### Check running services
```bash
# API server
curl http://localhost:8000/health

# Streamlit UI
curl http://localhost:8501
```

### View logs
```bash
# API server logs
tail -f logs/api_server.log

# Debate logs (JSON)
ls -la logs/debates/
```

## Deployment

### Local Development
Use `start.sh` for development with auto-reload enabled.

### Production
```bash
# API Server (with Gunicorn)
gunicorn api_server:app -w 4 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:8000

# Streamlit UI
streamlit run ui/judge_ui/app.py --server.port 8501 --server.headless true
```

### Docker (Optional)
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8000 8501
CMD ["./start.sh"]
```

## Contributing

When making changes:
1. Backend changes → Update `api_server.py` models if needed
2. UI changes → Test both Dummy and HTTP API modes
3. Model changes → Update both `ui/judge_ui/models.py` and `api_server.py`

## Support

For issues or questions:
- Check logs in `logs/` directory
- Review configuration in `.env`
- Verify all dependencies are installed
- Test API endpoints with curl

---

**Last Updated**: November 29, 2025
