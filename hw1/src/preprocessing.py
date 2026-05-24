"""
preprocessing.py
----------------
All preprocessing logic for the Student Social Media Impact pipeline.
Designed to be fully reusable in Homework 2 and the Final Project.

IMPORTANT: Split-first, preprocess-second discipline is enforced here.
All statistics (medians, IQR bounds, scaler params, encoders) are fitted
on X_train only, then applied identically to X_val and X_test.
"""

import numpy as np
import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import warnings
warnings.filterwarnings("ignore")

# ── Column name for the target ─────────────────────────────────────────────
TARGET_COL = "behavioral_segment"          # High Risk / Balanced / Focused

# ── Columns we drop before modelling (IDs, free-text, leaky) ──────────────
DROP_COLS = []                             # extend if needed after inspection

# ── Numerical columns that need scaling ───────────────────────────────────
# (populated dynamically after encoding)

RANDOM_STATE = 42


# ══════════════════════════════════════════════════════════════════════════
# 1.  LOAD
# ══════════════════════════════════════════════════════════════════════════

def load_data(path: str) -> pd.DataFrame:
    """Load CSV and do minimal column-name normalisation."""
    df = pd.read_csv(path)
    # lower-case, replace spaces/hyphens with underscores
    df.columns = (
        df.columns.str.strip()
                  .str.lower()
                  .str.replace(r"[\s\-]+", "_", regex=True)
    )
    print(f"[load]  shape={df.shape}")
    print(f"[load]  columns: {list(df.columns)}")
    return df


# ══════════════════════════════════════════════════════════════════════════
# 2.  FIND TARGET
# ══════════════════════════════════════════════════════════════════════════

def find_target_column(df: pd.DataFrame) -> str:
    """
    Try to locate the behavioural-segment column by a fuzzy match so the
    pipeline is robust to minor naming differences in the CSV.
    """
    candidates = [c for c in df.columns if "segment" in c or "behavior" in c or "behaviour" in c]
    if candidates:
        col = candidates[0]
        print(f"[target] using '{col}' as target")
        return col
    raise ValueError(
        f"Cannot auto-detect target column. "
        f"Set TARGET_COL manually. Available columns: {list(df.columns)}"
    )


# ══════════════════════════════════════════════════════════════════════════
# 3.  SPLIT  (split FIRST, preprocess SECOND)
# ══════════════════════════════════════════════════════════════════════════

def split_data(df: pd.DataFrame, target_col: str):
    """
    Stratified 80 / 10 / 10 split.
    Returns X_train, X_val, X_test, y_train, y_val, y_test (all DataFrames/Series).
    """
    # drop rows where target is missing
    df = df.dropna(subset=[target_col]).reset_index(drop=True)

    X = df.drop(columns=[target_col] + DROP_COLS)
    y = df[target_col]

    # first split: 90% train+val  |  10% test
    X_tv, X_test, y_tv, y_test = train_test_split(
        X, y, test_size=0.10, random_state=RANDOM_STATE, stratify=y
    )
    # second split: ~88.9% of 90% = 80% train  |  ~11.1% = 10% val
    X_train, X_val, y_train, y_val = train_test_split(
        X_tv, y_tv, test_size=0.1111, random_state=RANDOM_STATE, stratify=y_tv
    )

    print(f"[split]  train={X_train.shape}, val={X_val.shape}, test={X_test.shape}")
    print(f"[split]  class distribution (train):\n{y_train.value_counts(normalize=True).round(3)}")
    return X_train, X_val, X_test, y_train, y_val, y_test


# ══════════════════════════════════════════════════════════════════════════
# 4.  MISSING VALUES
# ══════════════════════════════════════════════════════════════════════════

def fit_imputer(X_train: pd.DataFrame) -> dict:
    """Compute fill values from training set only."""
    fill = {}
    for col in X_train.columns:
        if X_train[col].isnull().sum() == 0:
            continue
        if X_train[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
            fill[col] = X_train[col].median()
        else:
            mode_vals = X_train[col].mode()
            fill[col] = mode_vals[0] if len(mode_vals) else "Unknown"
    print(f"[imputer]  {len(fill)} columns need imputation")
    return fill


def apply_imputer(X: pd.DataFrame, fill: dict) -> pd.DataFrame:
    X = X.copy()
    for col, val in fill.items():
        if col in X.columns:
            X[col] = X[col].fillna(val)
    # any remaining categoricals → "Unknown"
    for col in X.select_dtypes(include="object").columns:
        X[col] = X[col].fillna("Unknown")
    return X


# ══════════════════════════════════════════════════════════════════════════
# 5.  OUTLIER TREATMENT  (IQR Winsorising)
# ══════════════════════════════════════════════════════════════════════════

def fit_outlier_bounds(X_train: pd.DataFrame) -> dict:
    """Compute IQR bounds on training set only."""
    bounds = {}
    num_cols = X_train.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        q1 = X_train[col].quantile(0.25)
        q3 = X_train[col].quantile(0.75)
        iqr = q3 - q1
        bounds[col] = (q1 - 1.5 * iqr, q3 + 1.5 * iqr)
    print(f"[outliers]  IQR bounds computed for {len(bounds)} numerical columns")
    return bounds


def apply_outlier_bounds(X: pd.DataFrame, bounds: dict) -> pd.DataFrame:
    """Winsorise — cap values at training-set IQR bounds."""
    X = X.copy()
    for col, (lo, hi) in bounds.items():
        if col in X.columns:
            X[col] = X[col].clip(lower=lo, upper=hi)
    return X


# ══════════════════════════════════════════════════════════════════════════
# 6.  ENCODING
# ══════════════════════════════════════════════════════════════════════════

def fit_encoders(X_train: pd.DataFrame) -> dict:
    """
    Strategy:
    - Binary columns (2 unique values) → LabelEncoder
    - Nominal categoricals (≤ 15 unique) → One-Hot (drop_first=True)
    - High-cardinality categoricals (> 15 unique) → drop (or target-encode in future)
    Returns a dict with keys 'binary', 'ohe_cols', 'drop_cols'.
    """
    binary_encoders = {}
    ohe_cols = []
    drop_cols = []

    cat_cols = X_train.select_dtypes(include="object").columns
    for col in cat_cols:
        n_unique = X_train[col].nunique()
        if n_unique == 2:
            le = LabelEncoder()
            le.fit(X_train[col].astype(str))
            binary_encoders[col] = le
        elif n_unique <= 15:
            ohe_cols.append(col)
        else:
            drop_cols.append(col)
            print(f"[encoder]  dropping high-cardinality column '{col}' ({n_unique} unique)")

    print(f"[encoder]  binary={list(binary_encoders.keys())}, "
          f"OHE={ohe_cols}, dropped={drop_cols}")
    return {"binary": binary_encoders, "ohe_cols": ohe_cols, "drop_cols": drop_cols}


def apply_encoders(X: pd.DataFrame, enc: dict, ohe_template: list = None) -> pd.DataFrame:
    """
    Apply encoders fitted on training set.
    ohe_template: list of dummy columns from training set (for alignment).
    """
    X = X.copy()

    # drop high-cardinality
    X = X.drop(columns=[c for c in enc["drop_cols"] if c in X.columns], errors="ignore")

    # binary
    for col, le in enc["binary"].items():
        if col in X.columns:
            X[col] = X[col].astype(str).map(
                lambda v, le=le: le.transform([v])[0] if v in le.classes_ else 0
            )

    # one-hot
    if enc["ohe_cols"]:
        X = pd.get_dummies(X, columns=enc["ohe_cols"], drop_first=True)

    # align columns to training template
    if ohe_template is not None:
        for col in ohe_template:
            if col not in X.columns:
                X[col] = 0
        X = X[ohe_template]

    return X


# ══════════════════════════════════════════════════════════════════════════
# 7.  FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════

def add_features(X: pd.DataFrame) -> pd.DataFrame:
    """
    Create at least 2 new domain-informed features.
    Works on both raw and already-imputed data (before encoding).

    Feature 1 – digital_intensity:
        Captures combined screen-time load = daily_social_media_hours * session_frequency_per_day
        Models the multiplicative stress of both duration and frequency.

    Feature 2 – sleep_debt_score:
        1 / (avg_daily_sleep_hours + 1)   (bounded, non-linear)
        Students sleeping very little score near 1; well-rested students near 0.
        Mirrors the engagement_score pattern from the assignment example.

    Feature 3 – productive_ratio:
        online_learning_hours / (daily_social_media_hours + 1)
        Ratio of educational vs. total digital consumption — a proxy for
        whether digital time is being used constructively.
    """
    X = X.copy()

    # ── helper: graceful column lookup ────────────────────────────────────
    def col(candidates):
        for c in candidates:
            if c in X.columns:
                return c
        return None

    sm_col   = col(["daily_social_media_hours", "social_media_hours",
                     "time_spent_on_social_media", "avg_daily_social_media_hours"])
    freq_col = col(["session_frequency_per_day", "sessions_per_day",
                    "social_media_sessions_per_day"])
    sleep_col= col(["avg_daily_sleep_hours", "sleep_hours", "daily_sleep_hours",
                    "average_sleep_hours"])
    learn_col= col(["online_learning_hours", "educational_hours",
                    "hours_on_educational_content"])

    if sm_col and freq_col:
        X["digital_intensity"] = X[sm_col] * X[freq_col]

    if sleep_col:
        X["sleep_debt_score"] = 1.0 / (X[sleep_col].clip(lower=0) + 1.0)

    if learn_col and sm_col:
        X["productive_ratio"] = X[learn_col] / (X[sm_col] + 1.0)

    new_feats = [c for c in ["digital_intensity", "sleep_debt_score", "productive_ratio"]
                 if c in X.columns]
    print(f"[features]  added {new_feats}")
    return X


# ══════════════════════════════════════════════════════════════════════════
# 8.  SCALING
# ══════════════════════════════════════════════════════════════════════════

def fit_scaler(X_train: pd.DataFrame) -> StandardScaler:
    num_cols = X_train.select_dtypes(include=[np.number]).columns
    scaler = StandardScaler()
    scaler.fit(X_train[num_cols])
    print(f"[scaler]  fitted on {len(num_cols)} numerical columns")
    return scaler


def apply_scaler(X: pd.DataFrame, scaler: StandardScaler) -> pd.DataFrame:
    X = X.copy()
    num_cols = [c for c in scaler.feature_names_in_ if c in X.columns]
    X[num_cols] = scaler.transform(X[num_cols])
    return X


# ══════════════════════════════════════════════════════════════════════════
# 9.  LABEL ENCODING FOR TARGET
# ══════════════════════════════════════════════════════════════════════════

def fit_label_encoder(y_train: pd.Series) -> LabelEncoder:
    le = LabelEncoder()
    le.fit(y_train.astype(str))
    print(f"[label_enc]  classes: {list(le.classes_)}")
    return le


def encode_target(y: pd.Series, le: LabelEncoder) -> np.ndarray:
    return le.transform(y.astype(str))


# ══════════════════════════════════════════════════════════════════════════
# 10.  PCA  (exploratory only)
# ══════════════════════════════════════════════════════════════════════════

def run_pca(X_train_scaled: pd.DataFrame,
            y_train_enc: np.ndarray,
            label_enc: LabelEncoder,
            out_dir: str = "models") -> None:
    """
    Run exploratory PCA on scaled training features.
    Saves: scree plot, loading heatmap, 2-D scatter.
    """
    os.makedirs(out_dir, exist_ok=True)
    X_np = X_train_scaled.select_dtypes(include=[np.number]).values
    n_components = min(X_np.shape[1], 20)
    pca_full = PCA(n_components=n_components, random_state=RANDOM_STATE)
    pca_full.fit(X_np)

    # ── Scree plot ─────────────────────────────────────────────────────
    explained = pca_full.explained_variance_ratio_
    cumulative = np.cumsum(explained)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(range(1, n_components + 1), explained * 100, alpha=0.7,
           color="#4C72B0", label="Individual")
    ax.plot(range(1, n_components + 1), cumulative * 100, "o-",
            color="#DD8452", label="Cumulative")
    ax.axhline(90, linestyle="--", color="grey", linewidth=0.8, label="90% threshold")
    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Explained Variance (%)")
    ax.set_title("PCA Scree Plot — Student Social Media Dataset")
    ax.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "pca_scree.png"), dpi=150)
    plt.close(fig)

    # ── Loading heatmap (top 5 PCs vs top 10 features by loading magnitude) ─
    n_show = min(5, n_components)
    loadings = pd.DataFrame(
        pca_full.components_[:n_show].T,
        index=X_train_scaled.select_dtypes(include=[np.number]).columns,
        columns=[f"PC{i+1}" for i in range(n_show)]
    )
    # select top 10 features by max absolute loading across shown PCs
    top_features = loadings.abs().max(axis=1).nlargest(10).index
    loadings_top = loadings.loc[top_features]

    import seaborn as sns
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(loadings_top, annot=True, fmt=".2f", cmap="coolwarm",
                center=0, linewidths=0.4, ax=ax)
    ax.set_title("PCA Loadings — Top 10 Features (first 5 PCs)")
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "pca_loadings.png"), dpi=150)
    plt.close(fig)

    # ── 2-D scatter ────────────────────────────────────────────────────
    pca2 = PCA(n_components=2, random_state=RANDOM_STATE)
    X_2d = pca2.fit_transform(X_np)

    fig, ax = plt.subplots(figsize=(8, 6))
    palette = ["#E24A33", "#348ABD", "#988ED5"]
    for i, cls in enumerate(label_enc.classes_):
        mask = y_train_enc == i
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                   s=2, alpha=0.3, color=palette[i % len(palette)], label=cls)
    ax.set_xlabel(f"PC1 ({pca2.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca2.explained_variance_ratio_[1]*100:.1f}%)")
    ax.set_title("2-D PCA Projection coloured by Behavioral Segment")
    ax.legend(markerscale=5)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "pca_scatter.png"), dpi=150)
    plt.close(fig)

    # ── Summary printout ───────────────────────────────────────────────
    n90 = int(np.argmax(cumulative >= 0.90)) + 1
    top3 = loadings.abs()["PC1"].nlargest(3).index.tolist()
    print(f"[PCA]  {n90} components explain ≥90% variance")
    print(f"[PCA]  top PC1 contributors: {top3}")
    print("[PCA]  plots saved → models/pca_*.png")


# ══════════════════════════════════════════════════════════════════════════
# 11.  FULL PIPELINE  (convenience wrapper)
# ══════════════════════════════════════════════════════════════════════════

def run_full_pipeline(csv_path: str, models_dir: str = "models"):
    """
    End-to-end preprocessing.  Returns:
        X_train, X_val, X_test  (scaled DataFrames)
        y_train, y_val, y_test  (encoded numpy arrays)
        label_enc               (LabelEncoder for target)
        artifacts               (dict of fitted objects for saving)
    """
    os.makedirs(models_dir, exist_ok=True)

    # 1. Load
    df = load_data(csv_path)
    target_col = find_target_column(df)

    # 2. Split FIRST
    X_train, X_val, X_test, y_train_raw, y_val_raw, y_test_raw = split_data(df, target_col)

    # 3. Feature engineering (before imputation so raw columns are intact)
    X_train = add_features(X_train)
    X_val   = add_features(X_val)
    X_test  = add_features(X_test)

    # 4. Missing values — fit on train only
    fill = fit_imputer(X_train)
    X_train = apply_imputer(X_train, fill)
    X_val   = apply_imputer(X_val,   fill)
    X_test  = apply_imputer(X_test,  fill)

    # 5. Outlier treatment — fit on train only
    bounds = fit_outlier_bounds(X_train)
    X_train = apply_outlier_bounds(X_train, bounds)
    X_val   = apply_outlier_bounds(X_val,   bounds)
    X_test  = apply_outlier_bounds(X_test,  bounds)

    # 6. Encoding — fit on train only
    enc = fit_encoders(X_train)
    X_train = apply_encoders(X_train, enc)
    ohe_template = list(X_train.columns)        # column order from training set
    X_val   = apply_encoders(X_val,   enc, ohe_template)
    X_test  = apply_encoders(X_test,  enc, ohe_template)

    # 7. Target encoding
    label_enc = fit_label_encoder(y_train_raw)
    y_train = encode_target(y_train_raw, label_enc)
    y_val   = encode_target(y_val_raw,   label_enc)
    y_test  = encode_target(y_test_raw,  label_enc)

    # 8. Scaling — fit on train only
    scaler = fit_scaler(X_train)
    X_train = apply_scaler(X_train, scaler)
    X_val   = apply_scaler(X_val,   scaler)
    X_test  = apply_scaler(X_test,  scaler)

    # 9. PCA (exploratory)
    run_pca(X_train, y_train, label_enc, out_dir=models_dir)

    # 10. Save artefacts
    joblib.dump(scaler,    os.path.join(models_dir, "scaler.pkl"))
    joblib.dump(label_enc, os.path.join(models_dir, "label_encoder.pkl"))
    joblib.dump({"fill": fill, "bounds": bounds, "enc": enc,
                 "ohe_template": ohe_template},
                os.path.join(models_dir, "preprocessing_artifacts.pkl"))
    print("[pipeline]  artefacts saved to", models_dir)

    artifacts = {
        "scaler": scaler, "label_enc": label_enc,
        "fill": fill, "bounds": bounds,
        "enc": enc, "ohe_template": ohe_template,
    }
    return X_train, X_val, X_test, y_train, y_val, y_test, label_enc, artifacts
