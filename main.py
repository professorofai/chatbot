"""FastAPI application for Physiotherapy Clinic Chatbot.

Endpoints:
- POST /chat: Process user messages with emergency detection, semantic FAQ matching,
  contextual boosting, confidence thresholding, and database logging.
- GET /logs: Retrieve recent conversation and audit logs.
- GET /health: Health check and model/index diagnostics.
- GET /: Web-based interactive chatbot UI.
"""

from __future__ import annotations

import logging
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import db
import nlp

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("physio_app")

# Global FAQ Matcher
matcher: Optional[nlp.FAQMatcher] = None

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan context for startup and shutdown initialization."""
    global matcher
    logger.info("Initializing database...")
    db.init_db()

    logger.info("Initializing NLP model and FAQ embeddings...")
    matcher = nlp.FAQMatcher()
    logger.info("FastAPI Physiotherapy Chatbot ready.")
    yield


app = FastAPI(
    title="Physiotherapy Clinic Chatbot API",
    description="FastAPI chatbot backend with NLTK preprocessing, Sentence Transformers FAQ matching, and SQLite persistence.",
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request & Response Models
class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, description="User's query or message")
    session_id: Optional[str] = Field(
        default=None, description="Optional conversation session ID"
    )
    variation_index: Optional[int] = Field(
        default=None, description="Optional variation index for diverse answer phrasing"
    )


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    confidence: float
    matched_faq_id: Optional[str] = None
    category: Optional[str] = None
    flag: str
    suggested_questions: Optional[List[str]] = None
    available_answers: Optional[List[str]] = None


class HealthResponse(BaseModel):
    status: str
    model: str
    faqs_count: int
    embeddings_count: int
    database: str
    timestamp: str


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(payload: ChatRequest):
    """Process incoming chat query:

    1. Checks emergency keywords -> returns urgent-care advisory if matched.
    2. Uses last 3 messages and session's last_category to boost candidate FAQs.
    3. Matches against FAQs using all-MiniLM-L6-v2 vector embeddings.
    4. Evaluates confidence against 0.55 threshold; returns fallback with suggestions if under.
    5. Logs every exchange into SQLite messages table.
    """
    if matcher is None:
        raise HTTPException(status_code=503, detail="NLP engine is initializing")

    # Generate session ID if not provided
    sess_id = payload.session_id.strip() if payload.session_id else f"sess_{uuid.uuid4().hex[:10]}"

    try:
        result = nlp.process_chat(
            session_id=sess_id,
            message=payload.message,
            matcher=matcher,
            variation_index=payload.variation_index,
        )
        return ChatResponse(**result)
    except Exception as e:
        logger.exception("Error processing chat message: %s", e)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.get("/logs", response_model=List[Dict[str, Any]])
async def logs_endpoint(
    session_id: Optional[str] = Query(
        default=None, description="Filter logs by session ID"
    ),
    limit: int = Query(default=100, ge=1, le=500, description="Max logs to return"),
):
    """Retrieve logged conversation exchanges for monitoring and auditing."""
    try:
        logs = db.get_logs(session_id=session_id, limit=limit)
        return logs
    except Exception as e:
        logger.exception("Error retrieving logs: %s", e)
        raise HTTPException(status_code=500, detail="Failed to fetch logs")


@app.get("/health", response_model=HealthResponse)
async def health_endpoint():
    """Health check returning system status, model state, and FAQ counts."""
    faqs_count = len(matcher.faqs) if matcher else 0
    emb_count = len(matcher.variant_texts) if matcher else 0

    return HealthResponse(
        status="healthy" if matcher is not None else "initializing",
        model=nlp.DEFAULT_MODEL_NAME,
        faqs_count=faqs_count,
        embeddings_count=emb_count,
        database="connected",
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


# Serve Static Web UI
if not STATIC_DIR.exists():
    STATIC_DIR.mkdir(parents=True, exist_ok=True)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
async def serve_ui():
    """Serve the interactive chatbot web interface."""
    index_file = STATIC_DIR / "index.html"
    if not index_file.exists():
        return {"message": "Physiotherapy Chatbot API is running. UI file missing."}
    return FileResponse(index_file)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)

