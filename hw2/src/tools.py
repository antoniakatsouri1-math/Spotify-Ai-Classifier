import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from src.rag import get_vector_store, retrieve

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "best_model.pt"
SCALER_PATH = BASE_DIR / "models" / "scaler.pkl"
DATASET_PATH = BASE_DIR / "data" / "dataset.csv"

_vectorstore = None
_model = None
_scaler = None
_dataset = None

BASE_FEATURES = [
    "acousticness", "danceability", "duration_ms", "energy",
    "instrumentalness", "key", "liveness", "loudness", "mode",
    "speechiness", "tempo", "time_signature", "valence",
    "popularity", "short_form",
]
ENGINEERED_FEATURES = BASE_FEATURES + [
    "energy_acousticness_ratio",
    "danceability_valence_product",
    "loudness_positive",
]
INPUT_SIZE = len(ENGINEERED_FEATURES)  # 18


class SpotifyClassifier(nn.Module):
    def __init__(self, input_size=INPUT_SIZE):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 32), nn.LeakyReLU(),
            nn.Linear(32, 1), nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


def _get_vectorstore():
    global _vectorstore
    if _vectorstore is None:
        _vectorstore = get_vector_store()
    return _vectorstore


def _get_model_and_scaler():
    global _model, _scaler
    if _scaler is None:
        with open(SCALER_PATH, "rb") as f:
            _scaler = pickle.load(f)
    if _model is None:
        m = SpotifyClassifier(input_size=INPUT_SIZE)
        state = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        m.load_state_dict(state)
        m.eval()
        _model = m
    return _model, _scaler


def _get_dataset():
    global _dataset
    if _dataset is None:
        _dataset = pd.read_csv(DATASET_PATH)
    return _dataset


def _preprocess_input(features):
    _, scaler = _get_model_and_scaler()
    df = pd.DataFrame([features])
    df["energy_acousticness_ratio"] = df["energy"] / (df["acousticness"] + 1e-6)
    df["danceability_valence_product"] = df["danceability"] * df["valence"]
    df["loudness_positive"] = df["loudness"] + 60.0
    for col in ENGINEERED_FEATURES:
        if col not in df.columns:
            df[col] = 0
    X = df[ENGINEERED_FEATURES].values.astype(float)
    return torch.tensor(scaler.transform(X), dtype=torch.float32)


def _retrieval_fn(query: str) -> str:
    return retrieve(query, _get_vectorstore())


def _prediction_fn(input_json: str) -> str:
    print(f"[DEBUG] prediction_fn called with: {input_json}")
    try:
        features = json.loads(input_json)
    except json.JSONDecodeError as e:
        return f"Error parsing JSON: {e}"
    try:
        X = _preprocess_input(features)
        model, _ = _get_model_and_scaler()
        with torch.no_grad():
            prob_ai = float(model(X).squeeze())
        prob_human = 1.0 - prob_ai
        if prob_ai >= 0.5:
            return (
                f"Prediction: AI-GENERATED (probability: {prob_ai * 100:.1f}%)\n"
                f"energy={features.get('energy', 'N/A')}, "
                f"acousticness={features.get('acousticness', 'N/A')}, "
                f"instrumentalness={features.get('instrumentalness', 'N/A')}."
            )
        else:
            return (
                f"Prediction: HUMAN-MADE (probability of being human: {prob_human * 100:.1f}%)\n"
                f"acousticness={features.get('acousticness', 'N/A')}, "
                f"instrumentalness={features.get('instrumentalness', 'N/A')}."
            )
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"Error running prediction: {e}"


def _stats_fn(column_name: str) -> str:
    try:
        df = _get_dataset()
        if column_name not in df.columns:
            return f"Column '{column_name}' not found. Available: {', '.join(df.columns)}"
        col = df[column_name]
        if pd.api.types.is_numeric_dtype(col):
            if column_name == "ai_generated":
                counts = col.value_counts().sort_index()
                return (
                    f"ai_generated distribution:\n"
                    f"  Human (0):        {counts.get(0, 0):,}\n"
                    f"  AI-Generated (1): {counts.get(1, 0):,}\n"
                    f"  AI ratio:         {col.mean() * 100:.1f}%"
                )
            s = col.describe()
            return (
                f"Stats for '{column_name}':\n"
                f"  Mean={s['mean']:.4f}, Std={s['std']:.4f}, "
                f"  Min={s['min']:.4f}, Median={s['50%']:.4f}, Max={s['max']:.4f}"
            )
        top5 = col.value_counts().head(5)
        return f"Top values for '{column_name}':\n" + "\n".join(f"  {v}: {c:,}" for v, c in top5.items())
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"Error: {e}"


class RetrievalInput(BaseModel):
    query: str = Field(description="The user question to search the knowledge base for")


class PredictionInput(BaseModel):
    input_json: str = Field(
        description="JSON string with audio features: acousticness, danceability, duration_ms, energy, instrumentalness, key, liveness, loudness, mode, speechiness, tempo, time_signature, valence, popularity, short_form")


class StatsInput(BaseModel):
    column_name: str = Field(description="Name of the dataset column to get statistics for")


retrieval_tool = StructuredTool.from_function(
    func=_retrieval_fn,
    name="retrieval_tool",
    description="Search the Spotify music knowledge base. Use for questions about audio features, AI-generated music, genres, or music recommendation systems.",
    args_schema=RetrievalInput,
)

prediction_tool = StructuredTool.from_function(
    func=_prediction_fn,
    name="prediction_tool",
    description="Predict if a Spotify track is Human-made or AI-Generated using the trained neural network. Requires audio features as a JSON string.",
    args_schema=PredictionInput,
)

dataset_stats_tool = StructuredTool.from_function(
    func=_stats_fn,
    name="dataset_stats_tool",
    description="Return statistics for any column in the Spotify 2026 dataset. Use when asked about averages, distributions, or counts of features like tempo, energy, or ai_generated.",
    args_schema=StatsInput,
)

ALL_TOOLS = [retrieval_tool, prediction_tool, dataset_stats_tool]