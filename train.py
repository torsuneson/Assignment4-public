# train.py
# -----------------------------------------------------------------------------
# Assignment 4 – Training & evaluation utilities
#
# Responsibilities:
# - Feature selection (reusing Assignment 3 filter/embedded/wrapper modules)
# - Train/evaluate classical models with GroupKFold-by-run
# - Save reports, confusion matrices, plots, and best classical model
# - Orchestrate optional deep learning (MLP) via feature_deeplearning.py
#
# This module contains NO weak-label construction.
# -----------------------------------------------------------------------------

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Any, Tuple

import numpy as np
import pandas as pd

from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA

import joblib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def get_models(random_seed: int) -> Dict[str, object]:
    return {
        "logreg": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=800, class_weight="balanced"))
        ]),
        "svm_rbf": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", C=3.0, gamma="scale", class_weight="balanced"))
        ]),
        "knn": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier(n_neighbors=7))
        ]),
        "rf": RandomForestClassifier(
            n_estimators=400,
            random_state=random_seed,
            class_weight="balanced_subsample",
            n_jobs=-1,
        ),
    }


def plot_macro_f1(scores: pd.DataFrame, out_path: Path, title: str) -> None:
    scores = scores.sort_values("macro_f1", ascending=False)
    plt.figure(figsize=(8, 4.2))
    plt.bar(scores["model"], scores["macro_f1"], color="steelblue")
    plt.ylim(0, 1)
    plt.ylabel("Macro F1")
    plt.title(title)
    for i, v in enumerate(scores["macro_f1"].values):
        plt.text(i, min(0.98, v + 0.02), f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_confusion_matrix(cm: np.ndarray, classes: List[str], out_path: Path, title: str) -> None:
    cm = cm.astype(float)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums != 0)

    plt.figure(figsize=(7, 5.5))
    plt.imshow(cm_norm, interpolation="nearest", cmap="Blues")
    plt.title(title)
    plt.colorbar(fraction=0.046, pad=0.04)
    ticks = np.arange(len(classes))
    plt.xticks(ticks, classes, rotation=45, ha="right")
    plt.yticks(ticks, classes)
    plt.ylabel("True label")
    plt.xlabel("Predicted label")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, f"{cm_norm[i, j]:.2f}", ha="center", va="center", fontsize=9,
                     color="white" if cm_norm[i, j] > 0.5 else "black")

    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_pca_scatter(X: np.ndarray, y: np.ndarray, classes: List[str], out_path: Path, random_seed: int) -> None:
    Xs = StandardScaler().fit_transform(X)
    X2 = PCA(n_components=2, random_state=random_seed).fit_transform(Xs)

    plt.figure(figsize=(7.2, 5.4))
    for i, cls in enumerate(classes):
        mask = (y == i)
        if mask.sum() == 0:
            continue
        plt.scatter(X2[mask, 0], X2[mask, 1], s=10, alpha=0.35, label=f"{cls} (n={mask.sum()})")
    plt.title("PCA (2D) of segment features")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend(loc="best", frameon=False)
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def select_features_assignment3(
    Xdf: pd.DataFrame,
    y_codes: np.ndarray,
    topk_filter: int,
    max_features: int,
    wrapper_max_k: int,
    wrapper_scoring: str,
    seed: int,
) -> List[str]:
    feat_names = list(Xdf.columns)
    Xdf2 = Xdf.copy().apply(pd.to_numeric, errors="coerce")
    Xdf2 = Xdf2.replace([np.inf, -np.inf], np.nan)
    Xdf2 = Xdf2.fillna(Xdf2.median(numeric_only=True)).fillna(0)

    selected: List[str] = []

    # Filter: MI
    try:
        import feature_filters as ff
        scores = ff.f_information_gain(Xdf2.values.astype(float), y_codes)
        try:
            scores = ff._normalize_01(scores)
        except Exception:
            pass
        idx = np.argsort(scores)[::-1][:min(topk_filter, len(feat_names))]
        selected += [feat_names[i] for i in idx]
    except Exception as e:
        print("[FS] Filter MI skipped:", e)

    # Embedded: L1
    try:
        import feature_embedded as fe
        selected += fe.select_features_embedded_l1(Xdf2, y_codes, max_features=max_features, seed=seed)
    except Exception as e:
        print("[FS] Embedded L1 skipped:", e)

    # Wrapper: SFS forward
    try:
        import feature_wrappers as fw
        if hasattr(fw, 'SCORING'):
            fw.SCORING = wrapper_scoring
        if hasattr(fw, '_fit_and_get_features_sfs'):
            k = min(wrapper_max_k, Xdf2.shape[1])
            selected += fw._fit_and_get_features_sfs(Xdf2, y_codes, k=k, direction='forward')
    except Exception as e:
        print("[FS] Wrapper SFS skipped:", e)

    # unique preserve order
    seen = set()
    out = []
    for f in selected:
        if f in feat_names and f not in seen:
            out.append(f)
            seen.add(f)

    return out[:min(max_features, len(out))] if out else feat_names


def train_and_evaluate(
    df: pd.DataFrame,
    classes: List[str],
    n_splits: int,
    random_seed: int,
    use_feature_selection: bool,
    fs_topk_filter: int,
    fs_max_features: int,
    fs_wrapper_max_k: int,
    fs_wrapper_scoring: str,
    plot_dir: Path,
    out_selected_features: Path,
) -> Tuple[str, object, str, np.ndarray, pd.DataFrame, List[str]]:
    drop_cols = {"run", "label", "seg", "Latitude", "Longitude"}
    feature_cols_all = [c for c in df.columns if c not in drop_cols and pd.api.types.is_numeric_dtype(df[c])]

    y_str = df["label"].astype(str).to_numpy()
    y = pd.Categorical(y_str, categories=classes).codes
    groups = df["run"].astype(str).to_numpy()

    Xdf = df[feature_cols_all].copy()

    feature_cols = feature_cols_all
    if use_feature_selection:
        feature_cols = select_features_assignment3(
            Xdf, y,
            topk_filter=fs_topk_filter,
            max_features=fs_max_features,
            wrapper_max_k=fs_wrapper_max_k,
            wrapper_scoring=fs_wrapper_scoring,
            seed=random_seed,
        )
        out_selected_features.write_text(json.dumps({"selected_features": feature_cols}, indent=2), encoding="utf-8")

    X = df[feature_cols].to_numpy(dtype=float)

    plot_pca_scatter(X, y, classes, plot_dir / "pca_features_labeled.png", random_seed=random_seed)

    gkf = GroupKFold(n_splits=min(n_splits, len(np.unique(groups))))
    models = get_models(random_seed)

    report_lines: List[str] = []
    scores_rows: List[Dict[str, float]] = []

    best_name = None
    best_score = -1.0
    best_model = None
    best_cm = None

    for name, model in models.items():
        y_true_all, y_pred_all = [], []
        for tr_idx, te_idx in gkf.split(X, y, groups=groups):
            model.fit(X[tr_idx], y[tr_idx])
            pred = model.predict(X[te_idx])
            y_true_all.append(y[te_idx])
            y_pred_all.append(pred)

        y_true_all = np.concatenate(y_true_all)
        y_pred_all = np.concatenate(y_pred_all)

        acc = accuracy_score(y_true_all, y_pred_all)
        f1m = f1_score(y_true_all, y_pred_all, labels=list(range(len(classes))), average="macro", zero_division=0)
        cm = confusion_matrix(y_true_all, y_pred_all, labels=list(range(len(classes))))

        scores_rows.append({"model": name, "accuracy": float(acc), "macro_f1": float(f1m)})

        report_lines.append(f"\n=== {name} ===")
        report_lines.append(f"accuracy: {acc:.4f}")
        report_lines.append(f"macro_f1: {f1m:.4f}")
        report_lines.append(classification_report(
            y_true_all, y_pred_all,
            labels=list(range(len(classes))),
            target_names=classes,
            zero_division=0
        ))

        if f1m > best_score:
            best_score = f1m
            best_name = name
            best_model = model
            best_cm = cm

    scores_df = pd.DataFrame(scores_rows)
    plot_macro_f1(scores_df, plot_dir / "model_macro_f1.png", title="Model comparison (macro-F1) – classical")
    scores_df.sort_values("macro_f1", ascending=False).to_csv(plot_dir / "model_scores.csv", index=False)

    best_model.fit(X, y)
    plot_confusion_matrix(best_cm, classes, plot_dir / "confusion_matrix_best.png", title=f"Confusion matrix (best: {best_name})")

    report = "\n".join(report_lines)
    return best_name, best_model, report, best_cm, scores_df, feature_cols
