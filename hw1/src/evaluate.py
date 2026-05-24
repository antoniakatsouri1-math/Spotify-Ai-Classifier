"""
evaluate.py
-----------
Evaluation, visualisation, and model comparison for Task 4.
All evaluation is performed on the held-out TEST set only.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report,
)


# ══════════════════════════════════════════════════════════════════════════
# Core metrics
# ══════════════════════════════════════════════════════════════════════════

def compute_metrics(y_true, y_pred, y_proba, label_names):
    """
    Returns a dict of classification metrics for multi-class problems.
    """
    return {
        "accuracy":  accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall":    recall_score(y_true, y_pred,    average="macro", zero_division=0),
        "f1":        f1_score(y_true, y_pred,        average="macro", zero_division=0),
        "roc_auc":   roc_auc_score(y_true, y_proba,  multi_class="ovr", average="macro"),
    }


# ══════════════════════════════════════════════════════════════════════════
# Confusion matrix
# ══════════════════════════════════════════════════════════════════════════

def plot_confusion_matrix(y_true, y_pred, label_names, title: str, out_path: str):
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=label_names, yticklabels=label_names, ax=ax)
    # overlay raw counts
    for i in range(len(label_names)):
        for j in range(len(label_names)):
            ax.text(j + 0.5, i + 0.72, f"n={cm[i,j]}",
                    ha="center", va="center", fontsize=7, color="grey")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[eval]  Confusion matrix saved → {out_path}")


# ══════════════════════════════════════════════════════════════════════════
# Feature importance (classical model)
# ══════════════════════════════════════════════════════════════════════════

def plot_feature_importance(model, feature_names, out_path: str, top_n: int = 20):
    fi = pd.Series(model.feature_importances_, index=feature_names)
    fi = fi.nlargest(top_n).sort_values()

    fig, ax = plt.subplots(figsize=(8, 6))
    fi.plot(kind="barh", ax=ax, color="#4C72B0")
    ax.set_title(f"XGBoost — Top {top_n} Feature Importances")
    ax.set_xlabel("Importance score")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[eval]  Feature importance plot saved → {out_path}")


# ══════════════════════════════════════════════════════════════════════════
# Comparison table
# ══════════════════════════════════════════════════════════════════════════

def print_comparison_table(metrics_classical: dict, metrics_nn: dict):
    df = pd.DataFrame({
        "XGBoost (classical)": metrics_classical,
        "Neural Network":      metrics_nn,
    }).T
    df = df[["accuracy", "precision", "recall", "f1", "roc_auc"]]
    df.columns = ["Accuracy", "Precision (macro)", "Recall (macro)", "F1 (macro)", "ROC-AUC (macro)"]
    print("\n" + "="*70)
    print("MODEL COMPARISON — TEST SET")
    print("="*70)
    print(df.round(4).to_string())
    print("="*70)
    return df


def save_comparison_table(df: pd.DataFrame, out_path: str):
    df.to_csv(out_path)
    print(f"[eval]  Comparison table saved → {out_path}")


# ══════════════════════════════════════════════════════════════════════════
# Full evaluation run
# ══════════════════════════════════════════════════════════════════════════

def evaluate_model(model, X_test, y_test, label_names,
                   model_name: str, models_dir: str,
                   predict_fn=None):
    """
    Evaluate a single model on the test set.
    predict_fn: optional callable(model, X) → (y_pred, y_proba)
                (used for neural network wrapper)
    """
    if predict_fn is not None:
        y_pred, y_proba = predict_fn(model, X_test)
    else:
        y_pred  = model.predict(X_test)
        y_proba = model.predict_proba(X_test)

    metrics = compute_metrics(y_test, y_pred, y_proba, label_names)

    print(f"\n[{model_name}]  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=label_names, digits=4))

    slug = model_name.lower().replace(" ", "_")
    plot_confusion_matrix(
        y_test, y_pred, label_names,
        title=f"{model_name} — Confusion Matrix (Test Set)",
        out_path=os.path.join(models_dir, f"cm_{slug}.png"),
    )
    return metrics, y_pred, y_proba


def run_full_evaluation(classical_model,
                        nn_model, nn_device,
                        X_test, y_test,
                        label_names,
                        feature_names,
                        models_dir: str = "models"):
    """
    Evaluate both models, plot comparison, designate best_model.pkl.
    """
    import joblib
    from src.train_neural import predict_proba_nn

    os.makedirs(models_dir, exist_ok=True)

    # ── Classical ──────────────────────────────────────────────────────
    metrics_cl, _, _ = evaluate_model(
        classical_model, X_test, y_test, label_names,
        model_name="XGBoost", models_dir=models_dir,
    )
    plot_feature_importance(
        classical_model, feature_names,
        out_path=os.path.join(models_dir, "feature_importance.png"),
    )

    # ── Neural Network ─────────────────────────────────────────────────
    def nn_predict(model, X):
        proba = predict_proba_nn(model, X, nn_device)
        pred  = proba.argmax(axis=1)
        return pred, proba

    metrics_nn, _, _ = evaluate_model(
        nn_model, X_test, y_test, label_names,
        model_name="Neural Network", models_dir=models_dir,
        predict_fn=nn_predict,
    )

    # ── Comparison table ───────────────────────────────────────────────
    comparison_df = print_comparison_table(metrics_cl, metrics_nn)
    save_comparison_table(
        comparison_df,
        os.path.join(models_dir, "model_comparison.csv"),
    )

    # ── Designate best model ───────────────────────────────────────────
    if metrics_cl["roc_auc"] >= metrics_nn["roc_auc"]:
        best_model  = classical_model
        winner      = "XGBoost"
        best_metric = metrics_cl["roc_auc"]
        joblib.dump(best_model, os.path.join(models_dir, "best_model.pkl"))
    else:
        import torch
        winner      = "Neural Network"
        best_metric = metrics_nn["roc_auc"]
        # copy neural network weights as best_model
        ckpt = torch.load(os.path.join(models_dir, "neural_network.pt"), map_location="cpu")
        torch.save(ckpt, os.path.join(models_dir, "best_model.pkl"))

    print(f"\n[eval]  🏆  Best model: {winner}  (ROC-AUC={best_metric:.4f})")
    print(f"[eval]  Saved → models/best_model.pkl")

    # ── Bar chart comparison ───────────────────────────────────────────
    _plot_metric_bars(metrics_cl, metrics_nn, models_dir)

    return metrics_cl, metrics_nn, winner


def _plot_metric_bars(m_cl, m_nn, out_dir):
    metrics_keys = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    labels = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
    x = np.arange(len(labels))
    w = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    bars1 = ax.bar(x - w/2, [m_cl[k] for k in metrics_keys], w,
                   label="XGBoost", color="#4C72B0")
    bars2 = ax.bar(x + w/2, [m_nn[k] for k in metrics_keys], w,
                   label="Neural Network", color="#DD8452")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison — Test Set Metrics")
    ax.legend()

    for bar in list(bars1) + list(bars2):
        h = bar.get_height()
        ax.annotate(f"{h:.3f}",
                    xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", fontsize=8)

    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "model_comparison.png"), dpi=150)
    plt.close(fig)
    print("[eval]  Comparison bar chart saved → models/model_comparison.png")
