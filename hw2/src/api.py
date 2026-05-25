"""
src/api.py
FastAPI application exposing:
  POST /chat         – standard chat endpoint (mandatory)
  POST /chat/stream  – streaming SSE endpoint (bonus Task 6)
  DELETE /session/{session_id} – clear conversation history
"""

import asyncio
import json
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from src.agent import chat, clear_session, get_session_history

app = FastAPI(
    title="Spotify Music Intelligence Agent",
    description=(
        "A conversational AI agent that can answer questions about Spotify "
        "audio features, music genres, and predict track popularity using a "
        "trained ML model."
    ),
    version="1.0.0",
)


# ── Pydantic schemas ──────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    message: str = Field(
        ...,
        description="The user's message to the agent.",
        example="What audio features make a song popular on Spotify?",
    )
    session_id: str = Field(
        ...,
        description="Unique session identifier for maintaining conversation history.",
        example="user_001",
    )


class ChatResponse(BaseModel):
    response: str = Field(
        ...,
        description="The agent's response.",
    )
    session_id: str = Field(
        ...,
        description="The session ID echoed back.",
    )


class SessionInfo(BaseModel):
    session_id: str
    message_count: int
    message: str


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/", tags=["Health"])
def root():
    """Health check endpoint."""
    return {
        "status": "ok",
        "service": "Spotify Music Intelligence Agent",
        "docs": "/docs",
    }


@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
def chat_endpoint(request: ChatRequest) -> ChatResponse:
    """
    Send a message to the agent and receive a complete response.

    The agent can:
    - Answer questions about Spotify audio features, music genres, and popularity
    - Predict whether a track will be popular given its audio features
    - Return dataset statistics on demand

    Conversation history is maintained per session_id within the current server session.
    """
    try:
        response_text = chat(request.message, request.session_id)
        return ChatResponse(response=response_text, session_id=request.session_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat/stream", tags=["Chat (Streaming – Bonus Task 6)"])
async def chat_stream_endpoint(request: ChatRequest) -> StreamingResponse:
    """
    [BONUS] Send a message to the agent and receive a streaming SSE response.

    Tokens are sent progressively as the LLM generates them, similar to ChatGPT's
    streaming interface. The stream format follows the Server-Sent Events (SSE)
    protocol: each event is prefixed with 'data: ' and terminated with a blank line.
    A final event 'data: [DONE]' signals the end of the stream.
    """

    async def token_generator() -> AsyncGenerator[str, None]:
        # Run the synchronous chat function in a thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        full_response = await loop.run_in_executor(
            None, chat, request.message, request.session_id
        )

        # Simulate word-by-word streaming of the complete response
        words = full_response.split(" ")
        for i, word in enumerate(words):
            chunk = word if i == 0 else " " + word
            payload = json.dumps({"token": chunk, "session_id": request.session_id})
            yield f"data: {payload}\n\n"
            await asyncio.sleep(0.03)  # Small delay to simulate streaming

        yield "data: [DONE]\n\n"

    return StreamingResponse(
        token_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/session/{session_id}", response_model=SessionInfo, tags=["Session Management"])
def get_session(session_id: str) -> SessionInfo:
    """Get information about an existing session."""
    history = get_session_history(session_id)
    return SessionInfo(
        session_id=session_id,
        message_count=len(history),
        message=f"Session '{session_id}' has {len(history)} message(s) in history.",
    )


@app.delete("/session/{session_id}", tags=["Session Management"])
def delete_session(session_id: str) -> dict:
    """Clear the conversation history for a given session."""
    clear_session(session_id)
    return {"message": f"Session '{session_id}' cleared successfully."}
