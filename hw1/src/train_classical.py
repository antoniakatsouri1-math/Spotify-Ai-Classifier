"""
train_classical.py
------------------
Classical ML training logic: XGBoost multi-class classifier.
Includes Task 6 hyperparameter tuning via RandomizedSearchCV.
"""

import numpy as np
import joblib
import os
import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import roc_auc_score

RANDOM_STATE = 42


def train_xgboost(X_train, y_train, X_val, y_val, n_classes: int):
    """
    Train a baseline XGBoost classifier with sensible defaults.
    Uses eval_set for early stopping on validation data.
    """
    print("\n[classical]  Training baseline XGBoost ...")
    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric="mlogloss",
        early_stopping_rounds=20,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=0,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    val_pred_proba = model.predict_proba(X_val)
    val_auc = roc_auc_score(y_val, val_pred_proba, multi_class="ovr", average="macro")
    print(f"[classical]  Baseline XGBoost — val macro ROC-AUC: {val_auc:.4f}")
    return model, val_auc


def tune_xgboost(X_train, y_train, X_val, y_val, n_classes: int):
    """
    Task 6: RandomizedSearchCV over XGBoost hyperparameters.
    Fitted on X_train with StratifiedKFold; confirmed on X_val before test.
    """
    print("\n[tuning]  Starting RandomizedSearchCV (this may take a few minutes) ...")

    param_dist = {
        "n_estimators":      [100, 200, 300, 400],
        "max_depth":         [4, 5, 6, 8, 10],
        "learning_rate":     [0.01, 0.05, 0.1, 0.2],
        "subsample":         [0.6, 0.7, 0.8, 0.9, 1.0],
        "colsample_bytree":  [0.5, 0.6, 0.7, 0.8, 1.0],
        "min_child_weight":  [1, 3, 5, 7],
        "gamma":             [0, 0.1, 0.2, 0.5],
    }

    base = XGBClassifier(
        use_label_encoder=False,
        eval_metric="mlogloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=0,
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    search = RandomizedSearchCV(
        estimator=base,
        param_distributions=param_dist,
        n_iter=40,
        scoring="roc_auc_ovr_weighted",
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=1,
        refit=True,
    )
    search.fit(X_train, y_train)

    best_model = search.best_estimator_
    print(f"[tuning]  Best params: {search.best_params_}")

    val_pred_proba = best_model.predict_proba(X_val)
    val_auc = roc_auc_score(y_val, val_pred_proba, multi_class="ovr", average="macro")
    print(f"[tuning]  Tuned XGBoost — val macro ROC-AUC: {val_auc:.4f}")
    return best_model, val_auc, search.best_params_


def get_feature_importances(model, feature_names, top_n: int = 15):
    """Return a sorted DataFrame of feature importances."""
    import pandas as pd
    fi = pd.DataFrame({
        "feature": feature_names,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False).head(top_n)
    return fi


def train_and_save(X_train, y_train, X_val, y_val,
                   n_classes: int,
                   models_dir: str = "models",
                   run_tuning: bool = True):
    """
    Full classical training flow:
      1. Train baseline XGBoost
      2. (Optional) Tune with RandomizedSearchCV
      3. Save model as models/classical_model.pkl
    Returns the final model and val AUC.
    """
    os.makedirs(models_dir, exist_ok=True)

    baseline_model, baseline_auc = train_xgboost(
        X_train, y_train, X_val, y_val, n_classes
    )

    if run_tuning:
        tuned_model, tuned_auc, best_params = tune_xgboost(
            X_train, y_train, X_val, y_val, n_classes
        )
        if tuned_auc >= baseline_auc:
            final_model = tuned_model
            final_auc   = tuned_auc
            print(f"[classical]  Using TUNED model  (AUC {tuned_auc:.4f} ≥ baseline {baseline_auc:.4f})")
        else:
            final_model = baseline_model
            final_auc   = baseline_auc
            print(f"[classical]  Using BASELINE model  (AUC {baseline_auc:.4f} > tuned {tuned_auc:.4f})")
    else:
        final_model = baseline_model
        final_auc   = baseline_auc

    path = os.path.join(models_dir, "classical_model.pkl")
    joblib.dump(final_model, path)
    print(f"[classical]  Saved → {path}")
    return final_model, final_auc
