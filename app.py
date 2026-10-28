"""FastAPI wrapper around the Mini-RAG service."""
import logging
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, HTTPException, Request
from ollama import ResponseError
from pydantic import BaseModel, Field

from rag import LLM_MODEL, RagService

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("mini_rag.api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.rag = RagService()
    logger.info("RAG service ready (model=%s)", LLM_MODEL)
    yield


app = FastAPI(
    title="Mini-RAG API",
    description="Ask questions about 'Hands-On Machine Learning with Scikit-Learn and TensorFlow'.",
    version="1.0.0",
    lifespan=lifespan,
)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000, examples=["What is overfitting?"])


class Source(BaseModel):
    page: str
    snippet: str


class AskResponse(BaseModel):
    question: str
    answer: str
    sources: list[Source]


@app.get("/health")
def health(request: Request):
    rag: RagService = request.app.state.rag
    try:
        chunks = rag.chunk_count()
    except Exception:
        logger.exception("Vector store check failed")
        chunks = 0
    return {"status": "ok" if chunks else "empty_index", "llm_model": LLM_MODEL, "chunks": chunks}


@app.post("/ask", response_model=AskResponse)
def ask(body: AskRequest, request: Request):
    rag: RagService = request.app.state.rag
    question = body.question.strip()

    if rag.chunk_count() == 0:
        raise HTTPException(503, "Vector store is empty. Run `python ingest.py` first.")

    try:
        result = rag.ask(question)
    except (ConnectionError, httpx.ConnectError):
        logger.exception("Cannot reach Ollama")
        raise HTTPException(503, "Cannot reach Ollama. Is it running and is OLLAMA_BASE_URL correct?")
    except ResponseError as e:
        logger.exception("Ollama returned an error")
        raise HTTPException(502, f"Ollama error: {e}")
    except Exception:
        logger.exception("Unexpected error while answering")
        raise HTTPException(500, "Internal error while generating the answer.")

    logger.info("Answered question (%d chars) using %d sources", len(question), len(result["sources"]))
    return AskResponse(question=question, **result)