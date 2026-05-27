# Backend

FastAPI backend for the Stock Analysis Tool.

## Current Status

This backend currently contains:

- FastAPI application setup
- Basic configuration
- Root endpoint
- Health check endpoint

## Run Locally in Codespaces

From the repository root:

```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload