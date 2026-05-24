"""
main.py
-------
Entry point: runs the complete Homework 1 pipeline end-to-end.

Usage:
    python main.py                      # full run (incl. tuning)
    python main.py --no-tuning          # skip RandomizedSearchCV (faster)

The dataset is expected at:   data/global_student_digital_behavior_dataset.csv
All outputs land in:          models/
"""

import os
import sys
import argparse
import joblib
import numpy as np

# ── Path setup ────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

DATA_PATH   = os.path.join(ROOT, "data", "global_student_digital_behavior_dataset.csv")
MODELS_DIR  = os.path.join(ROOT, "models")


def main(run_tuning: bool = True):
    print("\n" + "="*65)
    print("  HOMEWORK 1 — Student Social Media Behavioral Segment Pipeline")
    print("="*65)

    # ── TASK 2: Preprocessing ─────────────────────────────────────────
    print("\n── TASK 2: Preprocessing ──────────────────────────────────────")
    from src.preprocessing import run_full_pipeline

    (X_train, X_val, X_test,
     y_train, y_val, y_test,
     label_enc, artifacts) = run_full_pipeline(DATA_PATH, MODELS_DIR)

    n_classes     = len(label_enc.classes_)
    feature_names = list(X_train.columns)
    label_names   = list(label_enc.classes_)

    print(f"\n  Classes: {label_names}")
    print(f"  Features after encoding: {len(feature_names)}")

    # ── TASK 3a: Classical ML (XGBoost + optional tuning) ─────────────
    print("\n── TASK 3a: Classical ML Model ────────────────────────────────")
    from src.train_classical import train_and_save

    classical_model, val_auc_cl = train_and_save(
        X_train, y_train, X_val, y_val,
        n_classes   = n_classes,
        models_dir  = MODELS_DIR,
        run_tuning  = run_tuning,
    )

    # ── TASK 3b: Neural Network ────────────────────────────────────────
    print("\n── TASK 3b: Neural Network ────────────────────────────────────")
    from src.train_neural import train_neural

    nn_model, nn_device = train_neural(
        X_train, y_train, X_val, y_val,
        n_classes  = n_classes,
        models_dir = MODELS_DIR,
    )

    # ── TASK 4: Evaluation & Comparison ───────────────────────────────
    print("\n── TASK 4: Evaluation & Model Comparison ──────────────────────")
    from src.evaluate import run_full_evaluation

    metrics_cl, metrics_nn, winner = run_full_evaluation(
        classical_model = classical_model,
        nn_model        = nn_model,
        nn_device       = nn_device,
        X_test          = X_test,
        y_test          = y_test,
        label_names     = label_names,
        feature_names   = feature_names,
        models_dir      = MODELS_DIR,
    )

    # ── Summary ───────────────────────────────────────────────────────
    print("\n" + "="*65)
    print("  PIPELINE COMPLETE")
    print("="*65)
    print(f"  Winner:            {winner}")
    print(f"  XGBoost  ROC-AUC:  {metrics_cl['roc_auc']:.4f}")
    print(f"  NN       ROC-AUC:  {metrics_nn['roc_auc']:.4f}")
    print(f"\n  Saved artefacts in:  {MODELS_DIR}/")
    print("  • classical_model.pkl")
    print("  • neural_network.pt")
    print("  • best_model.pkl")
    print("  • scaler.pkl")
    print("  • label_encoder.pkl")
    print("  • preprocessing_artifacts.pkl")
    print("  • pca_scree.png  |  pca_loadings.png  |  pca_scatter.png")
    print("  • nn_loss_curve.png")
    print("  • cm_xgboost.png  |  cm_neural_network.png")
    print("  • feature_importance.png  |  model_comparison.png")
    print("="*65 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-tuning", action="store_true",
                        help="Skip RandomizedSearchCV (faster run)")
    args = parser.parse_args()
    main(run_tuning=not args.no_tuning)
