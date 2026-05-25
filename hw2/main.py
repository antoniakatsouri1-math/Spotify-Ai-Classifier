import os
from pathlib import Path

import uvicorn
from dotenv import load_dotenv

load_dotenv()

provider = os.getenv("LLM_PROVIDER", "groq").lower()
if provider == "anthropic":
    if not os.getenv("ANTHROPIC_API_KEY"):
        raise EnvironmentError("LLM_PROVIDER is 'anthropic' but ANTHROPIC_API_KEY is not set.")
elif provider == "openai":
    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError("LLM_PROVIDER is 'openai' but OPENAI_API_KEY is not set.")
elif provider == "google":
    if not os.getenv("GOOGLE_API_KEY"):
        raise EnvironmentError("LLM_PROVIDER is 'google' but GOOGLE_API_KEY is not set.")
elif provider == "groq":
    if not os.getenv("GROQ_API_KEY"):
        raise EnvironmentError("LLM_PROVIDER is 'groq' but GROQ_API_KEY is not set.")

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "best_model.pt"
SCALER_PATH = BASE_DIR / "models" / "scaler.pkl"

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"best_model.pt not found at {MODEL_PATH}.")
if not SCALER_PATH.exists():
    raise FileNotFoundError(f"scaler.pkl not found at {SCALER_PATH}.")

print("[main] HW1 model artifacts found ✓")

print("[main] Initializing RAG vector store...")
from src.rag import get_vector_store
get_vector_store()
print("[main] Vector store ready ✓")

from src.api import app

if __name__ == "__main__":
    uvicorn.run(
        "src.api:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
    )
