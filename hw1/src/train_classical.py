import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import roc_auc_score

def train_random_forest(X_train: np.ndarray, y_train,
                        X_val: np.ndarray, y_val,
                        random_state: int = 42) -> RandomForestClassifier:
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    val_auc = roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])
    print(f"[Random Forest] Validation ROC-AUC: {val_auc:.4f}")
    return model


def tune_random_forest(X_train: np.ndarray, y_train,
                       X_val: np.ndarray, y_val,
                       random_state: int = 42) -> RandomForestClassifier:

    param_dist = {
        "n_estimators": [100, 200, 300],
        "max_depth": [10, 15, 20, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2", 0.5],
    }
    base = RandomForestClassifier(
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1
    )
    search = RandomizedSearchCV(
        base, param_dist,
        n_iter=30,
        cv=5,
        scoring="roc_auc",
        random_state=random_state,
        n_jobs=-1,
        verbose=1
    )
    search.fit(X_train, y_train)
    best = search.best_estimator_
    print(f"[Tuned RF] Best params: {search.best_params_}")
    val_auc = roc_auc_score(y_val, best.predict_proba(X_val)[:, 1])
    print(f"[Tuned RF] Validation ROC-AUC: {val_auc:.4f}")
    return best


def get_feature_importances(model: RandomForestClassifier,
                            feature_names: list,
                            top_n: int = 15) -> list:
    importances = model.feature_importances_
    pairs = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
    return pairs[:top_n]


def save_model(model, path: str) -> None:
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"Model saved → {path}")


def load_model(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)
