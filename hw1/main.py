import argparse
import os
import pickle
import shutil
import numpy as np

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(BASE_DIR, "data", "dataset.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
PLOTS_DIR  = os.path.join(BASE_DIR, "plots")
SRC_DIR    = os.path.join(BASE_DIR, "src")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR,  exist_ok=True)

import sys
sys.path.insert(0, BASE_DIR)

from src.preprocessing  import load_and_clean, split_data, build_pipeline
from src.train_classical import (train_random_forest, tune_random_forest,
                                  get_feature_importances, save_model)
from src.train_neural    import (train_neural_network, save_neural_network,
                                  predict_proba_nn, experiment_activation_functions)
from src.evaluate        import evaluate_all, plot_feature_importances


def main(data_path: str = DATA_PATH, tune: bool = False):
    print("\n" + "=" * 60)
    print("  Spotify AI-Track Classifier  ·  HW1 Pipeline")
    print("=" * 60 + "\n")

    print(">>> TASK 1: Loading dataset …")
    df = load_and_clean(data_path)
    print(f"    Shape after cleaning: {df.shape}")
    print(f"    Columns: {df.columns.tolist()}\n")

    print(">>> TASK 2: Splitting data (80/10/10, stratified) …")
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)

    print("\n>>> TASK 2: Building preprocessing pipeline …")
    (X_train_s, X_val_s, X_test_s,
     feature_names, scaler, pca_info, extra_params) = build_pipeline(
        X_train, X_val, X_test, y_train, plots_dir=PLOTS_DIR
    )
    print(f"    Feature count after engineering: {len(feature_names)}")
    print(f"    PCA: {pca_info['n_for_90pct']} components explain ≥90% variance")
    print(f"    Top PC1 features: {pca_info['top_pc1_features']}\n")

    scaler_path = os.path.join(MODELS_DIR, "scaler.pkl")
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    print(f"    Scaler saved → {scaler_path}\n")

    print(">>> TASK 3a: Training Random Forest …")
    if tune:
        print("    [Bonus Task 6] Running RandomizedSearchCV …")
        rf_model = tune_random_forest(X_train_s, y_train, X_val_s, y_val)
    else:
        rf_model = train_random_forest(X_train_s, y_train, X_val_s, y_val)

    rf_path = os.path.join(MODELS_DIR, "classical_model.pkl")
    save_model(rf_model, rf_path)

    importances = get_feature_importances(rf_model, feature_names)
    print(f"    Top-5 features: {[n for n, _ in importances[:5]]}")
    plot_feature_importances(importances, plots_dir=PLOTS_DIR)

    print("\n>>> TASK 3b: Activation function experiment …")
    act_results = experiment_activation_functions(
        X_train_s, y_train, X_val_s, y_val, plots_dir=PLOTS_DIR
    )

    print("\n>>> TASK 3b: Training final Neural Network (LeakyReLU) …")
    nn_model = train_neural_network(
        X_train_s, y_train, X_val_s, y_val,
        plots_dir=PLOTS_DIR
    )
    nn_path = os.path.join(MODELS_DIR, "neural_network.pt")
    save_neural_network(nn_model, nn_path)

    print("\n>>> TASK 4: Evaluating both models on TEST set …")
    rf_probs = rf_model.predict_proba(X_test_s)[:, 1]
    nn_probs = predict_proba_nn(nn_model, X_test_s)

    rf_metrics, nn_metrics, df_cmp = evaluate_all(
        y_test, rf_probs, nn_probs, plots_dir=PLOTS_DIR
    )

    best_is_rf = rf_metrics["ROC-AUC"] >= nn_metrics["ROC-AUC"]
    best_label = "Random Forest" if best_is_rf else "Neural Network"
    print(f"\n>>> BEST MODEL: {best_label} "
          f"(ROC-AUC RF={rf_metrics['ROC-AUC']:.4f} "
          f"vs NN={nn_metrics['ROC-AUC']:.4f})")

    best_path = os.path.join(MODELS_DIR, "best_model.pkl")
    if best_is_rf:
        shutil.copy(rf_path, best_path)
    else:

        import torch
        from src.train_neural import SpotifyNet, load_neural_network
        nn_loaded = load_neural_network(nn_path, input_dim=X_train_s.shape[1])
        with open(best_path, "wb") as f:
            pickle.dump(nn_loaded, f)
    print(f"    Best model saved → {best_path}\n")

    print(">>> Pipeline complete. Plots saved to:", PLOTS_DIR)
    print("    Models saved to:",  MODELS_DIR)
    print()
    return rf_metrics, nn_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=DATA_PATH, help="Path to dataset CSV")
    parser.add_argument("--tune", action="store_true",
                        help="Run RandomizedSearchCV hyperparameter tuning (bonus)")
    args = parser.parse_args()
    main(data_path=args.data, tune=args.tune)
