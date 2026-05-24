"""
src/tools.py
LangGraph tool definitions:
  1. retrieval_tool     – answers domain knowledge questions via RAG
  2. prediction_tool    – classifies a track as Human or AI-Generated using the HW1 Neural Network
  3. dataset_stats_tool – returns summary statistics for dataset columns (BONUS)
"""

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from langchain.tools import tool

from src.rag import get_vector_store, retrieve

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "best_model.pt"   # PyTorch Neural Network
SCALER_PATH = BASE_DIR / "models" / "scaler.pkl"
DATASET_PATH = BASE_DIR / "data" / "dataset.csv"

# ── Lazy-loaded globals ───────────────────────────────────────────────────────
_vectorstore = None
_model = None
_scaler = None
_dataset = None

# ── Input features (15) — must match HW1 training column order exactly ────────
# Dropped from raw data: artist_name, track_id, track_name, scenario, ai_generated
BASE_FEATURES = [
    "acousticness",
    "danceability",
    "duration_ms",
    "energy",
    "instrumentalness",
    "key",
    "liveness",
    "loudness",
    "mode",
    "speechiness",
    "tempo",
    "time_signature",
    "valence",
    "popularity",
    "short_form",
]

# After feature engineering the model sees 18 features
ENGINEERED_FEATURES = BASE_FEATURES + [
    "energy_acousticness_ratio",
    "danceability_valence_product",
    "loudness_positive",
]

INPUT_SIZE = len(ENGINEERED_FEATURES)  # 18


# ── Neural Network architecture (must match HW1 exactly) ─────────────────────
class SpotifyClassifier(nn.Module):
    """
    Architecture: Input(18) → Dense(128, ReLU) → Dropout(0.3)
                            → Dense(64,  ReLU) → Dropout(0.2)
                            → Dense(32,  LeakyReLU)
                            → Dense(1,   Sigmoid)
    """
    def __init__(self, input_size: int = INPUT_SIZE):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.network(x)


# ── Loaders ───────────────────────────────────────────────────────────────────
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
        model = SpotifyClassifier(input_size=INPUT_SIZE)
        state = torch.load(MODEL_PATH, map_location="cpu")
        # Handle both raw state_dict and checkpoint dicts
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        model.load_state_dict(state)
        model.eval()
        _model = model

    return _model, _scaler


def _get_dataset() -> pd.DataFrame:
    global _dataset
    if _dataset is None:
        _dataset = pd.read_csv(DATASET_PATH)
    return _dataset


# ── Preprocessing (mirrors HW1 preprocessing.py exactly) ─────────────────────
def _preprocess_input(features: dict) -> torch.Tensor:
    """
    1. Build single-row DataFrame
    2. Apply feature engineering (3 derived features, same as HW1)
    3. Assemble columns in training order
    4. Scale with saved StandardScaler
    5. Return as torch.Tensor
    """
    _, scaler = _get_model_and_scaler()

    df = pd.DataFrame([features])

    # Feature engineering — must match HW1 src/preprocessing.py exactly
    df["energy_acousticness_ratio"] = (
        df["energy"] / (df["acousticness"] + 1e-6)
    )
    df["danceability_valence_product"] = df["danceability"] * df["valence"]
    df["loudness_positive"] = df["loudness"] + 60.0

    # Fill any missing columns with 0
    for col in ENGINEERED_FEATURES:
        if col not in df.columns:
            df[col] = 0

    X = df[ENGINEERED_FEATURES].values.astype(float)
    X_scaled = scaler.transform(X)

    return torch.tensor(X_scaled, dtype=torch.float32)


# ─────────────────────────────────────────────────────────────────────────────
# TOOL 1 – Retrieval (RAG)
# ─────────────────────────────────────────────────────────────────────────────
@tool
def retrieval_tool(query: str) -> str:
    """
    Search the Spotify music knowledge base and return relevant information.
    Use this tool when the user asks a factual or conceptual question about
    music, Spotify audio features, AI-generated music, music recommendation
    systems, or the differences between human-made and AI-generated tracks.

    Args:
        query: The user's question as a natural-language string.

    Returns:
        Relevant passages from the knowledge base as a single string.
    """
    vs = _get_vectorstore()
    return retrieve(query, vs)


# ─────────────────────────────────────────────────────────────────────────────
# TOOL 2 – AI vs Human Prediction (HW1 Neural Network)
# ─────────────────────────────────────────────────────────────────────────────
@tool
def prediction_tool(input_json: str) -> str:
    """
    Predict whether a Spotify track was created by a human artist or generated
    by AI, using the trained Neural Network from HW1.

    Use this tool when the user provides specific audio feature values and wants
    to know if a track is Human-made or AI-Generated.

    Args:
        input_json: A JSON string with the following 15 fields:
            - acousticness    (float, 0.0–1.0): Confidence the track is acoustic
            - danceability    (float, 0.0–1.0): Suitability for dancing
            - duration_ms     (int):            Track duration in milliseconds
            - energy          (float, 0.0–1.0): Intensity and activity level
            - instrumentalness(float, 0.0–1.0): Likelihood of no vocals
            - key             (int, 0–11):      Musical key (0=C, 1=C#, ..., 11=B)
            - liveness        (float, 0.0–1.0): Presence of live audience
            - loudness        (float, dB):      Overall loudness, typically -60 to 0
            - mode            (int, 0 or 1):    Minor (0) or Major (1)
            - speechiness     (float, 0.0–1.0): Presence of spoken words
            - tempo           (float, BPM):     Estimated beats per minute
            - time_signature  (int, 3–7):       Beats per measure
            - valence         (float, 0.0–1.0): Musical positiveness
            - popularity      (int, 0–100):     Spotify popularity score
            - short_form      (int, 0 or 1):    Whether the track is short-form (1) or not (0)

    Returns:
        A human-readable string with the prediction (Human / AI-Generated)
        and the probability.

    Example input:
        '{"acousticness": 0.35, "danceability": 0.68, "duration_ms": 210000,
          "energy": 0.72, "instrumentalness": 0.05, "key": 5,
          "liveness": 0.12, "loudness": -8.5, "mode": 1,
          "speechiness": 0.04, "tempo": 122.0, "time_signature": 4,
          "valence": 0.55, "popularity": 40, "short_form": 0}'
    """
    try:
        features = json.loads(input_json)
    except json.JSONDecodeError as e:
        return f"Error: Could not parse input JSON. Details: {e}"

    try:
        X = _preprocess_input(features)
        model, _ = _get_model_and_scaler()

        with torch.no_grad():
            prob_ai = float(model(X).squeeze())

        prob_human = 1.0 - prob_ai
        prediction = 1 if prob_ai >= 0.5 else 0

        if prediction == 1:
            summary = (
                f"Prediction: AI-GENERATED (probability: {prob_ai*100:.1f}%)\n"
                f"The model classifies this track as likely AI-generated. "
                f"Key signals include a high energy/acousticness ratio "
                f"(energy={features.get('energy','N/A')}, "
                f"acousticness={features.get('acousticness','N/A')}) "
                f"and instrumentalness={features.get('instrumentalness','N/A')}. "
                f"AI music systems tend to optimise simultaneously for high energy "
                f"and low acousticness, which drives the engineered ratio feature."
            )
        else:
            summary = (
                f"Prediction: HUMAN-MADE (probability of being human: {prob_human*100:.1f}%)\n"
                f"The model classifies this track as likely created by a human artist. "
                f"With acousticness={features.get('acousticness','N/A')} and "
                f"instrumentalness={features.get('instrumentalness','N/A')}, "
                f"the audio profile is more consistent with human musical production."
            )

        return summary

    except Exception as e:
        return f"Error running prediction: {e}"


# ─────────────────────────────────────────────────────────────────────────────
# TOOL 3 – Dataset Statistics (BONUS)
# ─────────────────────────────────────────────────────────────────────────────
@tool
def dataset_stats_tool(column_name: str) -> str:
    """
    Return summary statistics for a specific column in the Spotify 2026 dataset.
    Use this tool when the user asks about the distribution, average, range,
    or statistics of any audio feature in the dataset (e.g. 'average tempo',
    'how many AI tracks are there', 'distribution of acousticness').

    Args:
        column_name: Name of the column (e.g. 'acousticness', 'energy',
                     'tempo', 'ai_generated', 'popularity').

    Returns:
        A human-readable summary of the column's statistics.
    """
    try:
        df = _get_dataset()

        if column_name not in df.columns:
            available = ", ".join(df.columns.tolist())
            return (
                f"Column '{column_name}' not found in the dataset. "
                f"Available columns: {available}"
            )

        col = df[column_name]

        if pd.api.types.is_numeric_dtype(col):
            stats = col.describe()
            # Special case: ai_generated is binary, show class counts
            if column_name == "ai_generated":
                counts = col.value_counts().sort_index()
                return (
                    f"Column 'ai_generated' — class distribution:\n"
                    f"  Human (0):       {counts.get(0, 0):,}\n"
                    f"  AI-Generated (1):{counts.get(1, 0):,}\n"
                    f"  Total:           {len(col):,}\n"
                    f"  AI ratio:        {col.mean()*100:.1f}%"
                )
            return (
                f"Statistics for '{column_name}':\n"
                f"  Count:  {int(stats['count']):,}\n"
                f"  Mean:   {stats['mean']:.4f}\n"
                f"  Std:    {stats['std']:.4f}\n"
                f"  Min:    {stats['min']:.4f}\n"
                f"  25th %: {stats['25%']:.4f}\n"
                f"  Median: {stats['50%']:.4f}\n"
                f"  75th %: {stats['75%']:.4f}\n"
                f"  Max:    {stats['max']:.4f}"
            )
        else:
            value_counts = col.value_counts()
            top5 = value_counts.head(5)
            top5_str = "\n".join(
                f"    {val}: {count:,}" for val, count in top5.items()
            )
            return (
                f"Statistics for '{column_name}' (categorical):\n"
                f"  Total rows:     {len(col):,}\n"
                f"  Unique values:  {col.nunique():,}\n"
                f"  Missing values: {col.isna().sum():,}\n"
                f"  Top 5 values:\n{top5_str}"
            )

    except Exception as e:
        return f"Error computing statistics for '{column_name}': {e}"


# ── Export all tools ──────────────────────────────────────────────────────────
ALL_TOOLS = [retrieval_tool, prediction_tool, dataset_stats_tool]
