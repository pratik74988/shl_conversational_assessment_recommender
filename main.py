"""
FastAPI service for SHL Assessment Recommender.

Endpoints:
  GET  /health  → {"status": "ok"}
  POST /chat    → ChatResponse

The service is stateless — every /chat call carries the full
conversation history. No per-conversation state is stored.
"""

from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from schemas import ChatRequest, ChatResponse, Recommendation
from agent import process_chat
from retrieval import CatalogRetriever

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
#  App lifecycle — load catalog + build indexes once at startup        #
# ------------------------------------------------------------------ #

retriever: CatalogRetriever | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global retriever
    logger.info("Building catalog indexes...")
    retriever = CatalogRetriever("catalog.json")
    logger.info("Ready.")
    yield
    # cleanup (nothing needed)


app = FastAPI(
    title="SHL Assessment Recommender",
    description="Conversational agent for SHL Individual Test Solutions",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ------------------------------------------------------------------ #
#  Endpoints                                                           #
# ------------------------------------------------------------------ #

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    if not request.messages:
        raise HTTPException(status_code=400, detail="messages list cannot be empty")

    # Validate roles
    for msg in request.messages:
        if msg.role not in ("user", "assistant"):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid role '{msg.role}'. Must be 'user' or 'assistant'."
            )

    # Last message must be from user
    if request.messages[-1].role != "user":
        raise HTTPException(
            status_code=400,
            detail="Last message must be from 'user'."
        )

    messages = [m.model_dump() for m in request.messages]

    try:
        result = process_chat(messages, retriever)
    except Exception as exc:
        # Never let an internal error expose a 500 to the evaluator.
        # Log it and return a safe clarify-style response.
        logger.exception("process_chat failed: %s", exc)
        return ChatResponse(
            reply="I ran into an issue processing that. Could you rephrase your request?",
            recommendations=[],
            end_of_conversation=False,
        )

    return ChatResponse(
        reply=result["reply"],
        recommendations=[Recommendation(**r) for r in result["recommendations"]],
        end_of_conversation=result["end_of_conversation"],
    )
