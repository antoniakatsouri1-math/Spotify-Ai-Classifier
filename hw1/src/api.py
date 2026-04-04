import sys
import os
import joblib
import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, "src"))

app = FastAPI(title="Spotify AI-Track Classifier", version="1.2.0")

try:
    MODELS_PATH = os.path.join(BASE_DIR, "models")
    scaler = joblib.load(os.path.join(MODELS_PATH, "scaler.pkl"))
    best_model = joblib.load(os.path.join(MODELS_PATH, "best_model.pkl"))
    print("✓ Spotify models loaded successfully!")
except Exception as e:
    print(f"✗ Critical Error loading models: {e}")
    scaler = None
    best_model = None


class TrackFeatures(BaseModel):
    acousticness: float = Field(..., ge=0.0, le=1.0)
    danceability: float = Field(..., ge=0.0, le=1.0)
    duration_ms: float = Field(..., gt=0)
    energy: float = Field(..., ge=0.0, le=1.0)
    instrumentalness: float = Field(..., ge=0.0, le=1.0)
    key: int = Field(..., ge=0, le=11)
    liveness: float = Field(..., ge=0.0, le=1.0)
    loudness: float = Field(..., ge=-60.0, le=0.0)
    mode: int = Field(..., ge=0, le=1)
    speechiness: float = Field(..., ge=0.0, le=1.0)
    tempo: float = Field(..., gt=0)
    time_signature: int = Field(..., ge=1, le=7)
    valence: float = Field(..., ge=0.0, le=1.0)
    popularity: int = Field(..., ge=0, le=100)
    short_form: int = Field(..., ge=0, le=1)


def _process_features(t: TrackFeatures):
    base = [t.acousticness, t.danceability, t.duration_ms, t.energy,
            t.instrumentalness, t.key, t.liveness, t.loudness, t.mode,
            t.speechiness, t.tempo, t.time_signature, t.valence,
            t.popularity, t.short_form]

    base.append(t.energy / (t.acousticness + 1e-6))
    base.append(t.danceability * t.valence)
    base.append(t.loudness + 60.0)

    return np.array(base).reshape(1, -1)

@app.get("/")
def home():
    return {"status": "Online", "docs": "/docs"}


@app.post("/predict")
def predict(features: TrackFeatures):
    if best_model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Models not loaded")

    try:
        X = _process_features(features)
        X_scaled = scaler.transform(X)

        if hasattr(best_model, "predict_proba"):
            prob = float(best_model.predict_proba(X_scaled)[0, 1])
        else:
            best_model.eval()
            with torch.no_grad():
                prob = float(best_model(torch.tensor(X_scaled, dtype=torch.float32)).item())

        prediction = int(prob >= 0.5)
        return {
            "prediction": prediction,
            "label": "AI-Generated" if prediction == 1 else "Human",
            "probability": round(prob, 4)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))