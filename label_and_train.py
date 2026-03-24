# label_and_train.py
# -----------------------------------------------------------------------------
# Backwards-compatible entry point.
# Keeps prior behaviour but delegates to:
#   - label.py (weak labeling + derived impulse-friendly features)
#   - train.py (feature selection + classical training + plotting)
#
# Deep learning orchestration remains in this file as before (uses feature_deeplearning.py).
# -----------------------------------------------------------------------------

from __future__ import annotations


import importlib

import json
import os
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

import joblib

from sklearn.metrics import confusion_matrix

# Local modules
import label as lab
importlib.reload(lab)

import train as tr
import keras_mlp_weighted as kmw
from ensemble_combine import combine_predictions
from ensemble_combine import combine_predictions
import inspect
print(">>> COMBINE FROM:", combine_predictions.__code__.co_filename)
print(">>> COMBINE SRC HEAD:\n", "".join(inspect.getsource(combine_predictions).splitlines(True)[:5]))

# --------------------
# Paths / Settings (same defaults as before)
# --------------------
DATA1_DIR = Path("Data 1")
DATA2_DIR = Path("Data 2")
MANIFEST_PATH = Path("selected_runs.json")

OUT_DATASET = Path("segments_labeled.csv")
OUT_REPORT = Path("model_report.txt")
OUT_CONFUSION_CSV = Path("confusion_matrix.csv")
OUT_BEST_MODEL = Path("best_model.joblib")

PLOT_DIR = Path("plots")
PLOT_DIR.mkdir(exist_ok=True)
OUT_SELECTED_FEATURES = PLOT_DIR / "selected_features.json"

# MLP outputs
MLP_DIR = Path("mlp")
MLP_DIR.mkdir(exist_ok=True)
OUT_REPORT_MLP = MLP_DIR / "model_report_deeplearning.txt"
OUT_CONFUSION_CSV_MLP = MLP_DIR / "confusion_matrix_deeplearning.csv"
OUT_MODEL_MLP = MLP_DIR / "best_mlp_model.joblib"
OUT_PLOT_F1_DL = MLP_DIR / "model_macro_f1_deeplearning.png"
OUT_PLOT_CM_DL = MLP_DIR / "confusion_matrix_deeplearning.png"
OUT_LEARN_DL = MLP_DIR / "learning_curve_mlp.png"

INF_CSV   = "segments_inference.csv"
MODEL_OUT = MLP_DIR / "keras_mlp_weighted.keras"
SCALER_NPZ= MLP_DIR / "keras_scaler.npz"
PRED_OUT  = MLP_DIR / "predictions_keras.csv"

# Labeling params
LABEL_RADIUS_M = 500.0
SHIFT_FWD_S = +1.2
SHIFT_REV_S = -1.0
DT_VIB = 0.002

# CV / classes
N_SPLITS = 5
RANDOM_SEED = 42
CLASSES = ["Bridge", "RailJoint", "Turnout", "Other"]

# Feature selection settings
USE_FEATURE_SELECTION = True
FS_TOPK_FILTER = 20
FS_MAX_FEATURES = 20
FS_WRAPPER_MAX_K = 15
FS_WRAPPER_SCORING = "f1_macro"

# Deep learning orchestration
RUN_DEEPLEARNING = False #True
MLP_ONLY = False
RUN_KERAS_WEIGHTED = False   # kör Keras MLP med class weights som komplement
KERAS_PREDICT_ONLY = True   # run prediction only, require that RUN_KERAS_WEIGHTED har been run once

def _plot_single_macro_f1(val: float, out_path: Path, label: str = "mlp") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.figure(figsize=(4.0, 4.2))
    plt.bar([label], [val], color="slateblue")
    plt.ylim(0, 1)
    plt.ylabel("Macro F1")
    plt.title("Deep learning (MLP) macro-F1")
    plt.text(0, min(0.98, val + 0.02), f"{val:.3f}", ha="center", va="bottom", fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def _plot_confusion_matrix(cm: np.ndarray, classes: List[str], out_path: Path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
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


def _plot_learning_curve_dict(lc: dict, out_path: Path, title: str = "MLP learning curve"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sizes = lc["train_sizes_abs"]
    tr_mu, tr_sd = lc["train_mean"], lc["train_std"]
    va_mu, va_sd = lc["val_mean"], lc["val_std"]
    plt.figure(figsize=(7.0, 4.8))
    plt.plot(sizes, tr_mu, marker="o", label="Train")
    plt.fill_between(sizes, tr_mu-tr_sd, tr_mu+tr_sd, alpha=0.15)
    plt.plot(sizes, va_mu, marker="o", label="Validation")
    plt.fill_between(sizes, va_mu-va_sd, va_mu+va_sd, alpha=0.15)
    plt.xlabel("Number of training-examples")
    plt.ylabel(f"Score ({lc.get('scoring','')})")
    plt.title(title)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def main():

    # --- Predict-only med Keras på ny sträcka (ingen weak labeling krävs) ---
    if KERAS_PREDICT_ONLY:

        # Steg 1 — Bygg inference-dataset om det saknas
        if not os.path.exists(INF_CSV):
            print("[KERAS DEBUG] Building Inference dataset...")
            df_inf = lab.build_inference_dataset(
                manifest_path=MANIFEST_PATH,
                data2_dir=DATA2_DIR,
                dt_vib=DT_VIB,
            )
            df_inf.to_csv(INF_CSV, index=False)
            print(f"[KERAS_PREDICT_ONLY] Saved inference dataset: {INF_CSV} rows={len(df_inf)}")

        # Steg 2 — Kör prediktion med Keras
        print("[KERAS_PREDICT_ONLY] Running Keras prediction...")
        kres = kmw.predict_keras_mlp_weighted(
            model_path=MODEL_OUT,
            scaler_npz=SCALER_NPZ,
            dataset_csv=INF_CSV,
            out_path=PRED_OUT,
        )
        print("[KERAS_PREDICT_ONLY] Saved predictions:", PRED_OUT)

        # Steg 3 — Multi-run ensemble-combination

        #man = json.loads(Path("selected_runs.json").read_text(encoding="utf-8"))

        pred_files = []

        # KERAS_PREDICT_ONLY gör endast EN pred-fil:
        if Path(PRED_OUT).exists():
            pred_files.append(PRED_OUT)
        else:
            print(f"[ENSEMBLE] ERROR: Missing {PRED_OUT}")

        print(f"[ENSABMLE DEBUG] pred_files:{pred_files}")
        if pred_files:

            
            import sys, importlib
            print(sys.path)
            import ensemble_combine
            importlib.reload(ensemble_combine)
            from ensemble_combine import combine_predictions
            print(">>> AFTER RELOAD:", combine_predictions.__code__.co_filename)

            combine_predictions(pred_files, out_path="predictions_combined.csv")
            print("[ENSEMBLE] Done → predictions_combined.csv")
        else:
            print("[ENSEMBLE] No files to combine, skipping ensemble.")

        return

    # MLP-only shortcut (uses existing segments_labeled.csv)
    if MLP_ONLY:
        import feature_deeplearning as fd
        df = pd.read_csv(OUT_DATASET)
        res = fd.run_mlp(df=df)
        print("[MLP_ONLY] keys in res:", list(res.keys()))
        if "learning_curve" in res and res["learning_curve"] is not None:
            _plot_learning_curve_dict(res["learning_curve"], OUT_LEARN_DL)
            print("[MLP_ONLY] learning_curve saved:", OUT_LEARN_DL, OUT_LEARN_DL.exists())
        return

    print("Building labeled dataset...")
    df = lab.build_labeled_dataset(
        manifest_path=MANIFEST_PATH,
        data1_dir=DATA1_DIR,
        data2_dir=DATA2_DIR,
        label_radius_m=LABEL_RADIUS_M,
        dt_vib=DT_VIB,
        shift_fwd_s=SHIFT_FWD_S,
        shift_rev_s=SHIFT_REV_S,
    )
    df.to_csv(OUT_DATASET, index=False)
    print(f"Saved labeled dataset: {OUT_DATASET} rows={len(df)}")
    print("Label distribution:")
    print(df["label"].value_counts(dropna=False))

    print("Training and evaluating classical models (GroupKFold by run)...")
    best_name, best_model, report, cm, scores_df, feature_cols = tr.train_and_evaluate(
        df=df,
        classes=CLASSES,
        n_splits=N_SPLITS,
        random_seed=RANDOM_SEED,
        use_feature_selection=USE_FEATURE_SELECTION,
        fs_topk_filter=FS_TOPK_FILTER,
        fs_max_features=FS_MAX_FEATURES,
        fs_wrapper_max_k=FS_WRAPPER_MAX_K,
        fs_wrapper_scoring=FS_WRAPPER_SCORING,
        plot_dir=PLOT_DIR,
        out_selected_features=OUT_SELECTED_FEATURES,
    )

    OUT_REPORT.write_text(report + f"\n\nBEST CLASSICAL MODEL: {best_name}\n", encoding="utf-8")
    print(f"Saved report: {OUT_REPORT}")

    pd.DataFrame(cm, index=CLASSES, columns=CLASSES).to_csv(OUT_CONFUSION_CSV)
    print(f"Saved confusion matrix (CSV): {OUT_CONFUSION_CSV}")

    joblib.dump({"model": best_model, "classes": CLASSES, "feature_cols": feature_cols}, OUT_BEST_MODEL)
    print(f"Saved best model: {OUT_BEST_MODEL} (best={best_name})")

    # Deep learning
    if RUN_DEEPLEARNING:
        try:
            import feature_deeplearning as fd
            res = fd.run_mlp(df=df)
            OUT_REPORT_MLP.write_text(res["report_text"], encoding="utf-8")
            pd.DataFrame(res["cm"], index=CLASSES, columns=CLASSES).to_csv(OUT_CONFUSION_CSV_MLP)
            joblib.dump({"model": res["model"], "classes": CLASSES, "feature_cols": res["feature_cols"]}, OUT_MODEL_MLP)

            _plot_single_macro_f1(res["macro_f1"], OUT_PLOT_F1_DL, label="mlp")
            _plot_confusion_matrix(res["cm"], CLASSES, OUT_PLOT_CM_DL, title="Confusion matrix (MLP)")

            if "learning_curve" in res and res["learning_curve"] is not None:
                _plot_learning_curve_dict(res["learning_curve"], OUT_LEARN_DL)
                print("[MLP] learning_curve saved:", OUT_LEARN_DL, OUT_LEARN_DL.exists())

            print("[MLP] Saved deep learning outputs in ./mlp:")
            for f in [OUT_REPORT_MLP, OUT_CONFUSION_CSV_MLP, OUT_MODEL_MLP, OUT_PLOT_F1_DL, OUT_PLOT_CM_DL, OUT_LEARN_DL]:
                print(" -", f)

        except Exception as e:
            print("[MLP] Deep learning failed:", e)

    # --- Optional Keras weighted MLP (class_weight) ---
    if RUN_KERAS_WEIGHTED:
        try:
            #import keras_mlp_weighted as kmw

            kres = kmw.run_keras_mlp_weighted(
                df=df,
                dataset_csv=OUT_DATASET,
                classes=CLASSES,
                feature_cols=feature_cols,   # samma features som klassiska modeller
                random_seed=RANDOM_SEED,
                epochs=25,
                batch_size=256,
            )

            # Spara rapport + confusion matrix CSV
            (MLP_DIR / "model_report_keras_weighted.txt").write_text(kres["report_text"], encoding="utf-8")
            pd.DataFrame(kres["cm"], index=CLASSES, columns=CLASSES).to_csv(MLP_DIR / "confusion_matrix_keras_weighted.csv")

            # Plotta med dina befintliga plot-hjälpare
            _plot_confusion_matrix(kres["cm"], CLASSES, MLP_DIR / "confusion_matrix_keras_weighted.png",
                                title="Confusion matrix (Keras weighted MLP)")
            _plot_single_macro_f1(kres["macro_f1"], MLP_DIR / "model_macro_f1_keras_weighted.png",
                                label="keras_mlp_weighted")

            print("[KERAS] Saved weighted MLP outputs in ./mlp")
            print(" -", MLP_DIR / "model_report_keras_weighted.txt")
            print(" -", MLP_DIR / "confusion_matrix_keras_weighted.csv")
            print(" -", MLP_DIR / "confusion_matrix_keras_weighted.png")
            print(" -", MLP_DIR / "model_macro_f1_keras_weighted.png")

        except Exception as e:
            print("[KERAS] Weighted MLP failed (TensorFlow/Keras installed?):", e)

if __name__ == "__main__":
    main()
