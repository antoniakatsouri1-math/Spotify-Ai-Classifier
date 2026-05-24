"""
main.py
Entry point for the Spotify Music Intelligence Agent (HW2).

Startup sequence:
  1. Load environment variables from .env
  2. Ensure the vector store is built (or loaded from disk)
  3. Start the FastAPI application with uvicorn

Usage:
    python main.py
    # or
    uvicorn main:app --reload --host 0.0.0.0 --port 8000
"""

import os
from pathlib import Path

import uvicorn
from dotenv import load_dotenv

# Load .env file before any other imports that may need API keys
load_dotenv()

# Verify that at least one LLM API key is present
provider = os.getenv("LLM_PROVIDER", "openai").lower()
if provider == "anthropic":
    if not os.getenv("ANTHROPIC_API_KEY"):
        raise EnvironmentError(
            "LLM_PROVIDER is set to 'anthropic' but ANTHROPIC_API_KEY is not set. "
            "Please add it to your .env file."
        )
elif provider == "openai":
    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError(
            "LLM_PROVIDER is set to 'openai' but OPENAI_API_KEY is not set. "
            "Please add it to your .env file."
        )

# Verify HW1 model artifacts are present
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "best_model.pt"
SCALER_PATH = BASE_DIR / "models" / "scaler.pkl"

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"best_model.pt not found at {MODEL_PATH}. "
        "Please copy your HW1 best_model.pt (PyTorch Neural Network) into the models/ directory."
    )
if not SCALER_PATH.exists():
    raise FileNotFoundError(
        f"scaler.pkl not found at {SCALER_PATH}. "
        "Please copy your HW1 scaler.pkl into the models/ directory."
    )

print("[main] HW1 model artifacts found ✓")

# Pre-build or load the vector store at startup
print("[main] Initializing RAG vector store...")
from src.rag import get_vector_store
get_vector_store()
print("[main] Vector store ready ✓")

# Import the FastAPI app
from src.api import app  # noqa: E402 (import after env validation)

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,  # Set to True during development
        log_level="info",
    )
