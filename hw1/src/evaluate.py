import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, roc_curve
)



def compute_metrics(y_true, y_pred, y_prob) -> dict:
    return {
        "Accuracy":  accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall":    recall_score(y_true, y_pred, zero_division=0),
        "F1-Score":  f1_score(y_true, y_pred, zero_division=0),
        "ROC-AUC":   roc_auc_score(y_true, y_prob),
    }


def plot_confusion_matrix(y_true, y_pred, model_name: str,
                          plots_dir: str = "plots") -> None:
    os.makedirs(plots_dir, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    plt.colorbar(im, ax=ax)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Human (0)", "AI-Gen (1)"])
    ax.set_yticklabels(["Human (0)", "AI-Gen (1)"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {model_name}")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black",
                    fontsize=14, fontweight="bold")
    plt.tight_layout()
    fname = model_name.lower().replace(" ", "_") + "_confusion_matrix.png"
    plt.savefig(os.path.join(plots_dir, fname), dpi=150)
    plt.close()


def plot_roc_curves(y_true, rf_probs, nn_probs,
                   rf_auc: float, nn_auc: float,
                   plots_dir: str = "plots") -> None:
    os.makedirs(plots_dir, exist_ok=True)
    fpr_rf, tpr_rf, _ = roc_curve(y_true, rf_probs)
    fpr_nn, tpr_nn, _ = roc_curve(y_true, nn_probs)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fpr_rf, tpr_rf, color="#2196F3", lw=2,
            label=f"Random Forest (AUC = {rf_auc:.4f})")
    ax.plot(fpr_nn, tpr_nn, color="#F44336", lw=2,
            label=f"Neural Network (AUC = {nn_auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — Test Set")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "roc_curves.png"), dpi=150)
    plt.close()


def plot_feature_importances(importances: list, plots_dir: str = "plots") -> None:
    os.makedirs(plots_dir, exist_ok=True)
    names  = [x[0] for x in importances]
    values = [x[1] for x in importances]

    fig, ax = plt.subplots(figsize=(9, max(4, len(names) * 0.4)))
    y_pos = range(len(names))
    ax.barh(list(y_pos), values, color="#2196F3", alpha=0.85)
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(names, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Importance")
    ax.set_title("Random Forest — Feature Importances (Top Features)")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "rf_feature_importances.png"), dpi=150)
    plt.close()

def print_comparison(rf_metrics: dict, nn_metrics: dict) -> pd.DataFrame:
    df = pd.DataFrame({
        "Random Forest": rf_metrics,
        "Neural Network": nn_metrics
    })
    print("\n" + "=" * 52)
    print("           MODEL COMPARISON — TEST SET")
    print("=" * 52)
    print(df.to_string(float_format="{:.4f}".format))
    print("=" * 52 + "\n")
    return df


def save_comparison_plot(rf_metrics: dict, nn_metrics: dict,
                         plots_dir: str = "plots") -> None:
    os.makedirs(plots_dir, exist_ok=True)
    metrics = list(rf_metrics.keys())
    rf_vals = list(rf_metrics.values())
    nn_vals = list(nn_metrics.values())

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width/2, rf_vals, width, label="Random Forest", color="#2196F3", alpha=0.85)
    ax.bar(x + width/2, nn_vals, width, label="Neural Network", color="#F44336", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison — Test Set Metrics")
    ax.legend()
    for i, (rv, nv) in enumerate(zip(rf_vals, nn_vals)):
        ax.text(i - width/2, rv + 0.01, f"{rv:.3f}", ha="center", fontsize=8)
        ax.text(i + width/2, nv + 0.01, f"{nv:.3f}", ha="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "model_comparison.png"), dpi=150)
    plt.close()


def evaluate_all(
    y_test,
    rf_probs: np.ndarray,
    nn_probs: np.ndarray,
    threshold: float = 0.5,
    plots_dir: str = "plots"
) -> tuple:
    rf_preds = (rf_probs >= threshold).astype(int)
    nn_preds = (nn_probs >= threshold).astype(int)

    rf_metrics = compute_metrics(y_test, rf_preds, rf_probs)
    nn_metrics = compute_metrics(y_test, nn_preds, nn_probs)

    df_cmp = print_comparison(rf_metrics, nn_metrics)

    plot_confusion_matrix(y_test, rf_preds, "Random Forest", plots_dir)
    plot_confusion_matrix(y_test, nn_preds, "Neural Network", plots_dir)
    plot_roc_curves(y_test, rf_probs, nn_probs,
                    rf_metrics["ROC-AUC"], nn_metrics["ROC-AUC"], plots_dir)
    save_comparison_plot(rf_metrics, nn_metrics, plots_dir)

    return rf_metrics, nn_metrics, df_cmp