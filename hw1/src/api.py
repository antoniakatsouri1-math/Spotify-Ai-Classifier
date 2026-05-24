"""
api.py  —  Task 5: FastAPI /predict endpoint
--------------------------------------------
Exposes the best model as a REST API.

Run with:
    uvicorn src.api:app --reload

Then visit:
    http://127.0.0.1:8000/docs   (Swagger UI)
    http://127.0.0.1:8000/redoc
"""

import os
import joblib
import numpy as np
import pandas as pd
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# ── Lazy-loaded globals ───────────────────────────────────────────────────
_model        = None
_scaler       = None
_label_enc    = None
_prep_arts    = None   # fill, bounds, enc, ohe_template

MODELS_DIR = os.environ.get("MODELS_DIR", "models")


def _load_artifacts():
    global _model, _scaler, _label_enc, _prep_arts
    if _model is None:
        _model     = joblib.load(os.path.join(MODELS_DIR, "best_model.pkl"))
        _scaler    = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
        _label_enc = joblib.load(os.path.join(MODELS_DIR, "label_encoder.pkl"))
        _prep_arts = joblib.load(os.path.join(MODELS_DIR, "preprocessing_artifacts.pkl"))


# ══════════════════════════════════════════════════════════════════════════
# Pydantic schema  (covers all numerical + key categorical features)
# Students can omit fields they don't have; None → will be imputed.
# ══════════════════════════════════════════════════════════════════════════

class StudentFeatures(BaseModel):
    # Demographics
    age:                          Optional[float] = Field(None, example=20)
    gender:                       Optional[str]   = Field(None, example="Male")
    country:                      Optional[str]   = Field(None, example="USA")
    development_level:            Optional[str]   = Field(None, example="Developed")

    # Socioeconomic
    family_income_usd:            Optional[float] = Field(None, example=45000)
    urban_rural:                  Optional[str]   = Field(None, example="Urban")
    internet_access:              Optional[str]   = Field(None, example="Yes")
    internet_speed_mbps:          Optional[float] = Field(None, example=50)

    # Education
    education_level:              Optional[str]   = Field(None, example="Undergraduate")
    field_of_study:               Optional[str]   = Field(None, example="Engineering")
    academic_motivation:          Optional[float] = Field(None, example=7)
    online_learning_hours:        Optional[float] = Field(None, example=2)

    # Social Media Behaviour
    daily_social_media_hours:     Optional[float] = Field(None, example=5)
    session_frequency_per_day:    Optional[float] = Field(None, example=8)
    avg_session_length_minutes:   Optional[float] = Field(None, example=25)
    late_night_usage_hours:       Optional[float] = Field(None, example=1.5)
    educational_content_pct:      Optional[float] = Field(None, example=20)
    entertainment_content_pct:    Optional[float] = Field(None, example=50)
    short_video_pct:              Optional[float] = Field(None, example=40)
    news_content_pct:             Optional[float] = Field(None, example=10)

    # Engagement
    daily_likes:                  Optional[float] = Field(None, example=30)
    daily_comments:               Optional[float] = Field(None, example=5)
    content_creation_hours:       Optional[float] = Field(None, example=0.5)

    # Cognitive / Academic
    attention_span_minutes:       Optional[float] = Field(None, example=20)
    study_hours_per_day:          Optional[float] = Field(None, example=3)
    class_attendance_pct:         Optional[float] = Field(None, example=80)
    productivity_score:           Optional[float] = Field(None, example=6)

    # Psychological
    stress_level:                 Optional[float] = Field(None, example=6)
    anxiety_score:                Optional[float] = Field(None, example=5)
    depression_score:             Optional[float] = Field(None, example=4)
    avg_daily_sleep_hours:        Optional[float] = Field(None, example=6.5)

    # Economic
    ad_exposure_per_day:          Optional[float] = Field(None, example=15)
    impulse_purchases_per_month:  Optional[float] = Field(None, example=2)
    monthly_digital_spending_usd: Optional[float] = Field(None, example=30)


class PredictionResponse(BaseModel):
    prediction:   int
    label:        str
    probabilities: dict


# ══════════════════════════════════════════════════════════════════════════
# App
# ══════════════════════════════════════════════════════════════════════════

app = FastAPI(
    title="Student Behavioral Segment Predictor",
    description=(
        "Predicts whether a student falls into **High Risk**, **Balanced**, "
        "or **Focused** behavioral segment based on their social media usage "
        "and academic/psychological profile."
    ),
    version="1.0.0",
)


@app.on_event("startup")
def startup_event():
    _load_artifacts()


@app.get("/")
def root():
    return {"message": "Student Behavioral Segment API — visit /docs for Swagger UI"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(features: StudentFeatures):
    """
    Predict the behavioral segment for a student.
    Missing fields are imputed using training-set statistics.
    """
    _load_artifacts()

    # Convert input to DataFrame
    row = pd.DataFrame([features.dict()])

    # ── Re-apply the same preprocessing steps ─────────────────────────
    from src.preprocessing import (
        add_features, apply_imputer, apply_outlier_bounds,
        apply_encoders, apply_scaler,
    )

    arts = _prep_arts
    row = add_features(row)
    row = apply_imputer(row, arts["fill"])
    row = apply_outlier_bounds(row, arts["bounds"])
    row = apply_encoders(row, arts["enc"], arts["ohe_template"])
    row = apply_scaler(row, _scaler)

    # ── Predict ────────────────────────────────────────────────────────
    try:
        proba = _model.predict_proba(row)[0]
        pred  = int(np.argmax(proba))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")

    label_names = list(_label_enc.classes_)
    return PredictionResponse(
        prediction=pred,
        label=label_names[pred],
        probabilities={name: round(float(p), 4)
                       for name, p in zip(label_names, proba)},
    )
