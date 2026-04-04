import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import pickle
import os

def load_and_clean(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    drop_cols = [c for c in ["artist_name", "track_id", "track_name", "scenario"] if c in df.columns]
    df = df.drop(columns=drop_cols)
    df = df.dropna(subset=["ai_generated"])
    df["ai_generated"] = df["ai_generated"].astype(int)
    return df

def split_data(df: pd.DataFrame, target: str = "ai_generated", random_state: int = 42):
    X = df.drop(columns=[target])
    y = df[target]

    X_tv, X_test, y_tv, y_test = train_test_split(
        X, y, test_size=0.10, random_state=random_state, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_tv, y_tv, test_size=0.1111, random_state=random_state, stratify=y_tv
    )
    print(f"Split sizes → Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    print(f"Class ratio (train) → {y_train.value_counts(normalize=True).to_dict()}")
    return X_train, X_val, X_test, y_train, y_val, y_test


def fit_imputer(X_train: pd.DataFrame):
    num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    medians = X_train[num_cols].median()
    return {"medians": medians, "num_cols": num_cols}


def apply_imputer(X: pd.DataFrame, imputer_params: dict) -> pd.DataFrame:
    X = X.copy()
    for col in imputer_params["num_cols"]:
        if col in X.columns:
            X[col] = X[col].fillna(imputer_params["medians"][col])
    return X

def fit_outlier_bounds(X_train: pd.DataFrame, cols: list = None):
    if cols is None:
        cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    bounds = {}
    for col in cols:
        Q1 = X_train[col].quantile(0.25)
        Q3 = X_train[col].quantile(0.75)
        IQR = Q3 - Q1
        bounds[col] = (Q1 - 1.5 * IQR, Q3 + 1.5 * IQR)
    return bounds


def apply_outlier_winsorize(X: pd.DataFrame, bounds: dict) -> pd.DataFrame:
    X = X.copy()
    for col, (lo, hi) in bounds.items():
        if col in X.columns:
            X[col] = X[col].clip(lower=lo, upper=hi)
    return X

def encode(X: pd.DataFrame) -> pd.DataFrame:
     return X.copy()

def add_features(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    X["energy_acousticness_ratio"] = X["energy"] / (X["acousticness"] + 1e-6)
    X["danceability_valence_product"] = X["danceability"] * X["valence"]
    X["loudness_positive"] = X["loudness"] + 60.0
    return X


def fit_scaler(X_train: pd.DataFrame):
    scaler = StandardScaler()
    scaler.fit(X_train)
    return scaler


def apply_scaler(X: pd.DataFrame, scaler: StandardScaler) -> np.ndarray:
    return scaler.transform(X)


def run_pca(X_train_scaled: np.ndarray, feature_names: list,
            y_train: pd.Series, plots_dir: str = "plots") -> dict:
    os.makedirs(plots_dir, exist_ok=True)
    n_components = min(X_train_scaled.shape[1], X_train_scaled.shape[0])
    pca = PCA(n_components=n_components, random_state=42)
    pca.fit(X_train_scaled)

    evr = pca.explained_variance_ratio_
    cumulative = np.cumsum(evr)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(range(1, len(evr) + 1), evr, alpha=0.7, color="#2196F3", label="Individual")
    ax.plot(range(1, len(evr) + 1), cumulative, "o-", color="#F44336", label="Cumulative")
    ax.axhline(0.90, color="gray", linestyle="--", linewidth=0.8, label="90% threshold")
    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Explained Variance Ratio")
    ax.set_title("PCA Scree Plot")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "pca_scree.png"), dpi=150)
    plt.close()

    n_show = min(3, pca.n_components_)
    loadings = pd.DataFrame(
        pca.components_[:n_show].T,
        index=feature_names,
        columns=[f"PC{i+1}" for i in range(n_show)]
    )
    fig, ax = plt.subplots(figsize=(8, max(6, len(feature_names) * 0.35)))
    im = ax.imshow(loadings.values, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(n_show))
    ax.set_xticklabels(loadings.columns)
    ax.set_yticks(range(len(feature_names)))
    ax.set_yticklabels(feature_names, fontsize=8)
    plt.colorbar(im, ax=ax)
    ax.set_title("PCA Loadings (first 3 PCs)")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "pca_loadings.png"), dpi=150)
    plt.close()

    X_2d = pca.transform(X_train_scaled)[:, :2]
    fig, ax = plt.subplots(figsize=(8, 6))
    for label, color, name in [(0, "#2196F3", "Human"), (1, "#F44336", "AI-Generated")]:
        mask = y_train.values == label
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                   alpha=0.3, s=5, c=color, label=name)
    ax.set_xlabel(f"PC1 ({evr[0]:.1%} var)")
    ax.set_ylabel(f"PC2 ({evr[1]:.1%} var)")
    ax.set_title("PCA 2D Projection — Training Set")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "pca_scatter.png"), dpi=150)
    plt.close()

    n_for_90 = int(np.searchsorted(cumulative, 0.90)) + 1
    top3_features = loadings["PC1"].abs().nlargest(3).index.tolist()

    print(f"PCA: {n_for_90} components explain ≥90% of variance")
    print(f"Top features in PC1: {top3_features}")
    return {"pca": pca, "evr": evr, "cumulative": cumulative,
            "n_for_90pct": n_for_90, "top_pc1_features": top3_features}


def build_pipeline(X_train, X_val, X_test, y_train,
                   plots_dir: str = "plots"):
    num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()

    imp = fit_imputer(X_train)
    X_train = apply_imputer(X_train, imp)
    X_val   = apply_imputer(X_val,   imp)
    X_test  = apply_imputer(X_test,  imp)

    bounds = fit_outlier_bounds(X_train, cols=num_cols)
    X_train = apply_outlier_winsorize(X_train, bounds)
    X_val   = apply_outlier_winsorize(X_val,   bounds)
    X_test  = apply_outlier_winsorize(X_test,  bounds)

    X_train = encode(X_train)
    X_val   = encode(X_val)
    X_test  = encode(X_test)

    X_train = add_features(X_train)
    X_val   = add_features(X_val)
    X_test  = add_features(X_test)

    feature_names = X_train.columns.tolist()

    scaler = fit_scaler(X_train)
    X_train_s = apply_scaler(X_train, scaler)
    X_val_s   = apply_scaler(X_val,   scaler)
    X_test_s  = apply_scaler(X_test,  scaler)

    pca_info = run_pca(X_train_s, feature_names, y_train, plots_dir=plots_dir)

    return (X_train_s, X_val_s, X_test_s,
            feature_names, scaler, pca_info,
            {"imputer": imp, "bounds": bounds})