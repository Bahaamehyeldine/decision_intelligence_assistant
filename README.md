# Decision Intelligence Assistant

A full-stack AI application that classifies customer support tickets and generates intelligent responses using RAG, ML, and LLM — with a four-way comparison to help you decide what to deploy in production.

## What It Does

Every support query runs through four systems simultaneously:

| System | What it does | Latency | Cost |
|--------|-------------|---------|------|
| RAG Answer | LLM answer grounded in similar past tickets | ~5000ms | ~$0.000005 |
| Non-RAG Answer | LLM answer from training knowledge only | ~2000ms | ~$0.000002 |
| ML Classifier | Trained Gradient Boosting priority prediction | ~2ms | $0 |
| LLM Zero-shot | LLM directly asked "is this urgent?" | ~1000ms | ~$0.000001 |

## Architecture

User Query → React Frontend (port 3000) → FastAPI Backend (port 8000)
Backend runs: RAG Pipeline, Non-RAG Pipeline, ML Classifier, LLM Zero-shot
RAG uses: sentence-transformers + Qdrant (port 6333) + Ollama llama3.2
ML uses: 13 hand-crafted features + TF-IDF/SVD + Gradient Boosting (F1=0.9675)

## Tech Stack

- Frontend: React + TypeScript + Vite + Tailwind CSS + Nginx
- Backend: FastAPI + Python 3.11
- ML Model: Gradient Boosting (scikit-learn), F1=0.9675
- Embeddings: sentence-transformers all-MiniLM-L6-v2
- Vector DB: Qdrant
- LLM: Ollama llama3.2 (local inference)
- Orchestration: Docker Compose

## Prerequisites

- Docker Desktop with WSL2 integration enabled
- Ollama installed with llama3.2 model
- 8GB+ RAM, NVIDIA GPU recommended

## Quick Start

Step 1 - Clone the repository:
git clone https://github.com/Bahaamehyeldine/decision_intelligence_assistant.git
cd decision_intelligence_assistant

Step 2 - Start Ollama with all interfaces:
Windows PowerShell: $env:OLLAMA_HOST = "0.0.0.0" then ollama serve
Linux/Mac: OLLAMA_HOST=0.0.0.0 ollama serve
Pull model if needed: ollama pull llama3.2

Step 3 - Start all services:
docker compose up --build

Step 4 - Index tweets into Qdrant (first time only):
cd backend && pip install -r requirements.txt
python -c "
import sys; sys.path.insert(0, '.')
import pandas as pd
from rag.vectorstore import create_collection, index_tweets
df = pd.read_csv('../data/processed/labeled_tweets.csv')
create_collection(recreate=True)
index_tweets(df)
print('Done')
"

Step 5 - Open the application:
http://localhost:3000

## Stop

docker compose down

## Run Locally without Docker

Terminal 1: docker run -p 6333:6333 qdrant/qdrant
Terminal 2: cd backend && source .venv/bin/activate && uvicorn main:app --reload --port 8000
Terminal 3: cd frontend && npm install && npm run dev
Visit: http://localhost:5173

## Dataset

Customer Support on Twitter — 2.8M tweets from major brands.
https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter
After filtering: 168,712 English inbound tweets

## Labeling Function (Weak Supervision)

10 signals with confidence scoring — threshold >= 3 = URGENT:
- Problem keywords with negation handling
- Financial keywords (always urgent)
- VADER + RoBERTa sentiment
- Exclamation marks + negative sentiment
- ALL CAPS detection, urgent emojis, word repetition
- Service down keywords, urgent questions with time pressure

Result: Normal 144,057 (85.4%) / Urgent 24,630 (14.6%)

## ML Model Performance

| Model | F1 | Precision | Recall |
|-------|-----|-----------|--------|
| Logistic Regression | 0.8872 | 0.8151 | 0.9733 |
| Random Forest | 0.9498 | 0.9644 | 0.9357 |
| Gradient Boosting | 0.9675 | 0.9912 | 0.9396 |

No overfitting: Train F1=0.9659 vs Test F1=0.9675 (gap=0.0016)
Top features: sentiment (32.9%), problem_keywords_count (30.3%), has_financial_keyword (23.5%)

## Deployment Recommendation

At 10,000 tickets/hour — deploy the ML Classifier.

| Criterion | ML Classifier | LLM Zero-shot |
|-----------|--------------|---------------|
| Latency | ~2ms | ~1000ms |
| Cost per ticket | $0 | ~$0.000001 |
| Accuracy (F1) | 0.9675 | ~0.85 estimated |
| Explainability | Full feature importance | Black box |

ML classifier is 500x faster, costs nothing, higher measured accuracy.
Exception: Use LLM zero-shot for sarcasm/informal language edge cases.

## API Endpoints

POST /query     Body: {"query": "...", "top_k": 5}
GET  /health    Check all service statuses
GET  /logs      Recent query logs
GET  /stats     Aggregate statistics

## Known Limitations

- ML misses urgent tweets using sarcasm or informal language (5.5% miss rate)
- LLM answers require Ollama running locally
- Qdrant must be re-indexed after each fresh Docker deployment
- RoBERTa feature has zero importance (redundant with VADER)

## Author

Bahaamehyeldine — AI Engineering Bootcamp Project 3
