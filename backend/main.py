"""
main.py
-------
FastAPI application entry point for the Decision Intelligence Assistant.
Initializes the app, registers routes, and configures middleware.
"""

from __future__ import annotations

import logging
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import router

# ── Logging Setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# ── App Setup ─────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Decision Intelligence Assistant",
    description="RAG + ML support ticket classification system",
    version="1.0.0"
)

# ── CORS Middleware ───────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routes ────────────────────────────────────────────────────────────────────
app.include_router(router)


# ── Health Check ──────────────────────────────────────────────────────────────
@app.get("/")
async def root():
    return {"status": "ok", "service": "Decision Intelligence Assistant"}


# ── Startup ───────────────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup_event():
    logger.info("Decision Intelligence Assistant starting up...")
    logger.info("Ollama host: %s", os.getenv("OLLAMA_HOST", "http://localhost:11434"))
    logger.info("Qdrant host: %s", os.getenv("QDRANT_HOST", "localhost"))