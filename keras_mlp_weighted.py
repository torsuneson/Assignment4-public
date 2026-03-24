# keras_mlp_weighted.py
# -----------------------------------------------------------------------------
# Minimal Keras MLP with class_weight support.
# Designed as a drop-in complement to sklearn MLPClassifier.
#
# Public API:
#   run_keras_mlp_weighted(df=None, dataset_csv=..., classes=[...], feature_cols=None, ...)
# Returns a dict compatible with label_and_train.py plotting:
#   {"accuracy":..., "macro_f1":..., "cm":..., "report_text":..., "feature_cols":..., "classes":..., "model":...}
#
# Notes:
# - Uses class weights computed by sklearn's compute_class_weight('balanced')
#   i.e. w_c = n_samples / (n_classes * n_c). See sklearn docs. citeturn156search11turn156search10
# - Requires TensorFlow/Keras to be installed in your environment.
# -----------------------------------------------------------------------------

from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict, Any, List

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from tensorflow import keras

CONF_THRESH = 0.80  # justera t.ex. 0.55–0.8


def _build_model(input_dim: int, n_classes: int, seed: int = 42):
    # Import locally so the module can still be imported without TF installed.
    import tensorflow as tf
    from tensorflow import keras

    tf.random.set_seed(int(seed))

    model = keras.Sequential([
        keras.layers.Input(shape=(input_dim,)),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dense(n_classes, activation="softmax"),
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def predict_keras_mlp_weighted(
    model_path: str = "keras_mlp_weighted.keras",
    scaler_npz: str = "keras_scaler.npz",
    dataset_csv: str = "segments_labeled.csv",
    classes: list[str] | None = None,
    feature_cols: list[str] | None = None,
    out_path: str = "predictions_keras.csv",
    also_eval_if_labels_present: bool = True,
):
    """
    Ladda sparad Keras-modell + skaler och gör prediktioner på datasetet.
    Skriver CSV med (run, seg, Latitude, Longitude om de finns), pred,
    confidence samt prob_klass för alla klasser.
    """

    if classes is None:
        classes = ["Bridge", "RailJoint", "Turnout", "Other"]

    # --- Läs dataset ---
    df = pd.read_csv(dataset_csv)

    # --- Välj features ---
    npz = np.load(scaler_npz, allow_pickle=True)
    mean = npz["mean"].astype(np.float32)
    scale = npz["scale"].astype(np.float32)

    if "feat_cols" in npz.files:
        feat_cols_from_npz = list(npz["feat_cols"])
        missing = [c for c in feat_cols_from_npz if c not in df.columns]
        if missing:
            raise ValueError(f"Saknade featurekolumner från skaler-filen: {missing}")
        chosen_features = feat_cols_from_npz
    else:
        drop_cols = {"run", "label", "seg", "Latitude", "Longitude"}
        all_feat_cols = [c for c in df.columns if c not in drop_cols and pd.api.types.is_numeric_dtype(df[c])]
        chosen_features = feature_cols if feature_cols is not None else all_feat_cols

    X = df[chosen_features].to_numpy(dtype=np.float32)
    if mean.shape[0] != X.shape[1] or scale.shape[0] != X.shape[1]:
        raise ValueError(
            f"Skalerns dimensioner matchar inte: scaler has {mean.shape[0]} features, "
            f"men X har {X.shape[1]} features. Kolla vilka features som användes vid träning."
        )
    Xs = (X - mean) / (scale + 1e-12)

    # --- Ladda modell och prediktera ---
    model = keras.models.load_model(model_path)
    y_prob = model.predict(Xs, verbose=0)
    y_pred = np.argmax(y_prob, axis=1)

    # --- Pred + confidence ---
    pred_labels = [classes[i] for i in y_pred]
    confidence = np.max(y_prob, axis=1)


    # --- Tvinga "Neutral" under tröskel ---
    pred_labels = [
        (pl if cf >= CONF_THRESH else "Neutral")
        for pl, cf in zip(pred_labels, confidence)
    ]


    # --- Bygg output-DataFrame (robust kopiering av metadata + GPS om de finns) ---
    meta_cols = []

    meta_cols = [c for c in ("run", "seg", "Latitude", "Longitude") if c in df.columns]
    out_df = df[meta_cols].copy() if meta_cols else pd.DataFrame(index=df.index)

    # Lägg till pred och sannolikheter
    out_df["pred"] = [classes[i] for i in y_pred]
    out_df["confidence"] = np.max(y_prob, axis=1).astype(float)
    for i, cls in enumerate(classes):
        out_df[f"prob_{cls}"] = y_prob[:, i]


    # --- Spara ---
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"[PRED] Saved predictions -> {out_path} rows={len(out_df)}")

    results = {
        "pred_csv": out_path,
        "classes": classes,
        "feature_cols": chosen_features,
        "n_rows": int(len(out_df)),
    }

    # --- Valfri snabb utvärdering om sanna labels finns i datasetet ---
    if also_eval_if_labels_present and "label" in df.columns:
        y_true = pd.Categorical(df["label"].astype(str), categories=classes).codes
        acc = accuracy_score(y_true, y_pred)
        f1m = f1_score(y_true, y_pred, labels=list(range(len(classes))), average="macro", zero_division=0)
        cm = confusion_matrix(y_true, y_pred, labels=list(range(len(classes))))

        report_text = "\n".join([
            "=== keras_mlp_weighted (predict-only eval) ===",
            f"accuracy: {acc:.4f}",
            f"macro_f1: {f1m:.4f}",
            classification_report(
                y_true, y_pred,
                labels=list(range(len(classes))),
                target_names=classes,
                zero_division=0
            )
        ])
        mlp_dir = Path("mlp"); mlp_dir.mkdir(exist_ok=True)
        (mlp_dir / "model_report_keras_predict.txt").write_text(report_text, encoding="utf-8")
        pd.DataFrame(cm, index=classes, columns=classes).to_csv(mlp_dir / "confusion_matrix_keras_predict.csv")
        print("[PRED] Wrote eval to ./mlp (keras_predict)")

        results.update({
            "accuracy": float(acc),
            "macro_f1": float(f1m),
            "cm": cm,
            "report_text": report_text,
        })

    return results


def run_keras_mlp_weighted(
    df: Optional[pd.DataFrame] = None,
    dataset_csv: Path | str = "segments_labeled.csv",
    classes: List[str] = None,
    feature_cols: Optional[List[str]] = None,
    random_seed: int = 42,
    epochs: int = 25,
    batch_size: int = 256,
    validation_split: float = 0.2,
) -> Dict[str, Any]:
    """Train/evaluate a Keras MLP using class weights.

    This function does not plot; it returns artifacts for the caller.
    """

    if classes is None:
        classes = ["Bridge", "RailJoint", "Turnout", "Other"]

    if df is None:
        df = pd.read_csv(Path(dataset_csv))

    drop_cols = {"run", "label", "seg", "Latitude", "Longitude"}
    all_feat_cols = [c for c in df.columns if c not in drop_cols and pd.api.types.is_numeric_dtype(df[c])]
    if feature_cols is None:
        feature_cols = all_feat_cols

    y_str = df["label"].astype(str).to_numpy()
    y = pd.Categorical(y_str, categories=classes).codes.astype(np.int64)

    X = df[feature_cols].to_numpy(dtype=np.float32)

    # Standardize
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X).astype(np.float32)

    # Class weights (balanced) 
    uniq = np.unique(y)
    w = compute_class_weight(class_weight="balanced", classes=uniq, y=y)
    class_weight = {int(c): float(wi) for c, wi in zip(uniq, w)}

    # Build + train
    model = _build_model(input_dim=Xs.shape[1], n_classes=len(classes), seed=random_seed)

    history = model.fit(
        Xs, y,
        epochs=int(epochs),
        batch_size=int(batch_size),
        validation_split=float(validation_split),
        class_weight=class_weight,
        verbose=0,
        shuffle=True,
    )

    # Evaluate on full dataset (consistent with your current sklearn MLP evaluation style)
    y_prob = model.predict(Xs, verbose=0)
    y_pred = np.argmax(y_prob, axis=1).astype(int)

    acc = accuracy_score(y, y_pred)
    f1m = f1_score(y, y_pred, labels=list(range(len(classes))), average="macro", zero_division=0)
    cm = confusion_matrix(y, y_pred, labels=list(range(len(classes))))

    report_text = "\n".join([
        "=== keras_mlp_weighted (deep learning) ===",
        f"accuracy: {acc:.4f}",
        f"macro_f1: {f1m:.4f}",
        classification_report(
            y, y_pred,
            labels=list(range(len(classes))),
            target_names=classes,
            zero_division=0,
        )
    ])

    # Convert history to simple dict for optional plotting if desired
    hist = {k: [float(x) for x in v] for k, v in history.history.items()}

    
    # Save modell + scaler 
    mlp_dir = Path("mlp"); mlp_dir.mkdir(exist_ok=True)
    model.save(mlp_dir / "keras_mlp_weighted.keras")
    print(f"[KERAS] Save {mlp_dir}/keras_mlp_weighted.keras") 
    np.savez(
        mlp_dir / "keras_scaler.npz",
        mean=scaler.mean_,
        scale=scaler.scale_,
        feat_cols=np.array(feature_cols, dtype=object)
    )
    print(f"[KERAS] Save {mlp_dir}/keras_scaler.npz") 


    return {
        "accuracy": float(acc),
        "macro_f1": float(f1m),
        "cm": cm,
        "report_text": report_text,
        "feature_cols": feature_cols,
        "classes": classes,
        "model": model,
        "scaler": scaler,
        "class_weight": class_weight,
        "history": hist,
    }


if __name__ == "__main__":
    # Minimal smoke-test
    res = run_keras_mlp_weighted()
    print(res["report_text"])
    print("class_weight:", res["class_weight"])
