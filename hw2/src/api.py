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
    try:
        response_text = chat(request.message, request.session_id)
        return ChatResponse(response=response_text, session_id=request.session_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat/stream", tags=["Chat (Streaming – Bonus Task 6)"])
async def chat_stream_endpoint(request: ChatRequest) -> StreamingResponse:
   async def token_generator() -> AsyncGenerator[str, None]:
        loop = asyncio.get_event_loop()
        full_response = await loop.run_in_executor(
            None, chat, request.message, request.session_id
        )

        words = full_response.split(" ")
        for i, word in enumerate(words):
            chunk = word if i == 0 else " " + word
            payload = json.dumps({"token": chunk, "session_id": request.session_id})
            yield f"data: {payload}\n\n"
            await asyncio.sleep(0.03)  

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
    clear_session(session_id)
    return {"message": f"Session '{session_id}' cleared successfully."}
