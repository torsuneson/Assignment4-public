# feature_deeplearning.py
# -----------------------------------------------------------------------------
# Deep learning (MLP) helper library.
#
# Design goal (as requested):
# - Keep the CONFIGURATION section from the standalone version.
# - Keep this file focused on the ALGORITHM (training + predictions), not plotting.
# - All plotting and file outputs should be handled by label_and_train.py.
#
# Public API:
#   run_mlp(df: pd.DataFrame | None = None, dataset_csv: Path = DATASET, ...)
#       -> returns dict with: accuracy, macro_f1, cm, report_text, feature_cols, model
# -----------------------------------------------------------------------------

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import learning_curve, GroupKFold

# --------------------
# CONFIG (kept from standalone)
# --------------------
DATASET = Path("segments_labeled.csv")
PLOT_DIR = Path("plots")
DEEP_LEARNING_DIR = Path("mlp")
PLOT_DIR.mkdir(exist_ok=True)
DEEP_LEARNING_DIR.mkdir(exist_ok=True)

SELECTED_FEATURES_JSON = PLOT_DIR / "selected_features.json"

CLASSES = ["Bridge", "RailJoint", "Turnout", "Other"]
N_SPLITS = 5
RANDOM_SEED = 42

# MLP hyperparameters
HIDDEN_LAYER_SIZES = (128, 64)
ALPHA = 1e-4
MAX_ITER = 120 # set to 1000 in exampels but takes forever to run
EARLY_STOPPING = True
N_ITER_NO_CHANGE = 10


def load_selected_features(default_cols: List[str], selected_features_json: Path = SELECTED_FEATURES_JSON) -> List[str]:
    """Load selected feature list if present; otherwise return default_cols."""
    if not selected_features_json.exists():
        return default_cols
    try:
        obj = json.loads(selected_features_json.read_text(encoding="utf-8"))
        feats = obj.get("selected_features", [])
        feats = [f for f in feats if f in default_cols]
        return feats if feats else default_cols
    except Exception:
        return default_cols


def compute_learning_curve_mlp(estimator, X, y, groups, n_splits=5, scoring="f1_macro"):
    """
    Minimal learning curve: train/validation score vs training set size.
    Returns a dict used by label_and_train.py for plotting.
    """
    X = np.asarray(X)
    y = np.asarray(y)
    groups = np.asarray(groups)

    cv = GroupKFold(n_splits=min(int(n_splits), len(np.unique(groups))))

    # Kör inte för små storlekar (multiclass + GroupKFold kan annars sakna klasser i fold)
    train_sizes = np.linspace(0.2, 1.0, 6)

    sizes, train_scores, val_scores = learning_curve(
        estimator=estimator,
        X=X,
        y=y,
        groups=groups,
        cv=cv,
        train_sizes=train_sizes,
        scoring=scoring,
        n_jobs=1,   # Windows-säkert
    )

    return {
        "train_sizes_abs": sizes,
        "train_mean": train_scores.mean(axis=1),
        "train_std": train_scores.std(axis=1),
        "val_mean": val_scores.mean(axis=1),
        "val_std": val_scores.std(axis=1),
        "scoring": scoring,
    }


def run_mlp(
    df: Optional[pd.DataFrame] = None,
    dataset_csv: Path = DATASET,
    selected_features_json: Path = SELECTED_FEATURES_JSON,
    classes: List[str] = CLASSES,
    n_splits: int = N_SPLITS,
    random_seed: int = RANDOM_SEED,
    hidden_layer_sizes: Tuple[int, ...] = HIDDEN_LAYER_SIZES,
    alpha: float = ALPHA,
    max_iter: int = MAX_ITER,
    early_stopping: bool = EARLY_STOPPING,
    n_iter_no_change: int = N_ITER_NO_CHANGE,
) -> Dict[str, Any]:
    """Train/evaluate MLP with GroupKFold-by-run.

    This function does NOT write files and does NOT plot.
    Those responsibilities are handled by label_and_train.py.

    Returns:
      dict with keys: accuracy, macro_f1, cm, report_text, feature_cols, model
    """
    if df is None:
        if not Path(dataset_csv).exists():
            raise FileNotFoundError("segments_labeled.csv not found. Run label_and_train.py first.")
        df = pd.read_csv(dataset_csv)

    drop_cols = {"run", "label", "seg", "Latitude", "Longitude"}
    all_feat_cols = [c for c in df.columns if c not in drop_cols and pd.api.types.is_numeric_dtype(df[c])]
    feat_cols = load_selected_features(all_feat_cols, selected_features_json=Path(selected_features_json))

    y_str = df["label"].astype(str).to_numpy()
    y = pd.Categorical(y_str, categories=classes).codes
    groups = df["run"].astype(str).to_numpy()

    X = df[feat_cols].to_numpy(dtype=float)

    gkf = GroupKFold(n_splits=min(int(n_splits), len(np.unique(groups))))

    mlp = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            activation="relu",
            alpha=float(alpha),
            max_iter=int(max_iter),
            random_state=int(random_seed),
            early_stopping=bool(early_stopping),
            n_iter_no_change=int(n_iter_no_change),
        ))
    ])

    
    lc = compute_learning_curve_mlp(mlp, X, y, groups, n_splits=n_splits, scoring="f1_macro")
    print("[DL DEBUG] lc keys:", None if lc is None else lc.keys())

    y_true_all, y_pred_all = [], []
    for tr_idx, te_idx in gkf.split(X, y, groups=groups):
        mlp.fit(X[tr_idx], y[tr_idx])
        pred = mlp.predict(X[te_idx])
        y_true_all.append(y[te_idx])
        y_pred_all.append(pred)

    y_true_all = np.concatenate(y_true_all)
    y_pred_all = np.concatenate(y_pred_all)

    acc = accuracy_score(y_true_all, y_pred_all)
    f1m = f1_score(y_true_all, y_pred_all, labels=list(range(len(classes))), average="macro", zero_division=0)
    cm = confusion_matrix(y_true_all, y_pred_all, labels=list(range(len(classes))))

    report_text = "\n".join([
        "=== mlp (deep learning) ===",
        f"accuracy: {acc:.4f}",
        f"macro_f1: {f1m:.4f}",
        classification_report(
            y_true_all, y_pred_all,
            labels=list(range(len(classes))),
            target_names=classes,
            zero_division=0,
        )
    ])

    # Fit on full data for saving later (done here as part of algorithm)
    mlp.fit(X, y)

    return {
        "accuracy": float(acc),
        "macro_f1": float(f1m),
        "cm": cm,
        "report_text": report_text,
        "feature_cols": feat_cols,
        "model": mlp,
        "classes": classes,
        "learning_curve": lc,
    }


if __name__ == "__main__":
    # Standalone execution for quick test 
    res = run_mlp()
    print(res["report_text"])
    print("Selected features:", len(res["feature_cols"]))
