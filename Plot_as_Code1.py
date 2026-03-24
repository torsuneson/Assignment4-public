# Plot_as_Code1.py
# -----------------------------------------------------------------------------
# Single combined map (like Code 1) with:
#   - Data 1 reference points: Bridge / RailJoint / Turnout
#   - Data 2 travelled GPS points (from selected_runs.json if present)
#   - Data 2 segment pseudo-labels from segments_labeled.csv (from label_and_train.py)
#   - Data 2 *predicted* labels from best_model.joblib (trained in label_and_train.py)
#       computed on the SAME segments_labeled rows for easy visual comparison.
#
# Output:
#   plots/map_code1_compare.html
#   plots/map_code1_compare.png (optional; requires kaleido)
#
# Notes:
# - Uses px.scatter_mapbox (older Plotly compatible). Some Plotly versions do NOT
#   support marker.line for scattermapbox; this script avoids marker.line.
# - "Predicted" points reuse the Latitude/Longitude already stored in
#   segments_labeled.csv (so prediction changes class, not coordinate).
# -----------------------------------------------------------------------------

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Dict, Any, List

import numpy as np
import pandas as pd
import plotly.express as px
import copy

# Optional dependency for predicted labels
try:
    import joblib
except Exception:  # pragma: no cover
    joblib = None

# --------------------
# Settings
# --------------------
DATA1_DIR = Path("Data 1")
DATA2_DIR = Path("Data 2")
MANIFEST = Path("selected_runs.json")
SEGMENTS_LABELED = Path("segments_labeled.csv")
BEST_MODEL = Path("best_model.joblib")

OUT_DIR = Path("plots")
OUT_DIR.mkdir(exist_ok=True)
OUT_HTML = OUT_DIR / "map_code1_compare.html"
OUT_PNG = OUT_DIR / "map_code1_compare.png"

GPS_STRIDE = 10
SEG_STRIDE = 50

# Colors (match Code 1-ish; keep deterministic)
COLOR_MAP = {
    # Data 1
    "Data1 | Bridge": "#d62728",      # red
    "Data1 | RailJoint": "#1f77b4",   # blue
    "Data1 | Turnout": "#2ca02c",     # green
    # Data 2
    "Data2_GPS | Travelled": "#111111",  # black/gray

    # Pseudo labels (ground truth approximation)
    "Data2_Segments | Bridge": "#ff9896",
    "Data2_Segments | RailJoint": "#aec7e8",
    "Data2_Segments | Turnout": "#98df8a",
    "Data2_Segments | Other": "#7f7f7f",

    # Predicted labels (model output)
    "Data2_Predicted | Bridge": "#8c564b",
    "Data2_Predicted | RailJoint": "#17becf",
    "Data2_Predicted | Turnout": "#bcbd22",
    "Data2_Predicted | Other": "#9467bd",
}

# Marker symbols per source (many Plotly versions ignore symbol for mapbox,
# but keeping it here makes legend clearer where supported).
SYMBOL_MAP = {
    "Data1": "circle",
    "Data2_GPS": "circle",
    "Data2_Segments": "circle",
    "Data2_Predicted": "circle",
}

# --------------------
# Helpers
# --------------------

def load_points_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    return df[["Latitude", "Longitude"]].dropna().copy()


def read_single_col(path: Path) -> pd.Series:
    try:
        s = pd.read_csv(path, header=None).iloc[:, 0]
    except Exception:
        return pd.Series(dtype=float)
    return pd.to_numeric(s, errors="coerce").dropna().reset_index(drop=True)


def load_run_gps(run_folder: Path) -> pd.DataFrame:
    lat_p = run_folder / "GPS.latitude.refined.csv"
    lon_p = run_folder / "GPS.longitude.refined.csv"
    if not (lat_p.exists() and lon_p.exists()):
        lat_p = run_folder / "GPS.latitude.csv"
        lon_p = run_folder / "GPS.longitude.csv"

    lat = read_single_col(lat_p)
    lon = read_single_col(lon_p)
    n = int(min(len(lat), len(lon)))
    if n < 2:
        return pd.DataFrame(columns=["Latitude", "Longitude", "source", "class", "run", "layer"])

    df = pd.DataFrame({"Latitude": lat.iloc[:n], "Longitude": lon.iloc[:n]})
    df = df[df["Latitude"].between(-90, 90) & df["Longitude"].between(-180, 180)]
    df = df[~((df["Latitude"] == 0) & (df["Longitude"] == 0))]
    df = df.iloc[::GPS_STRIDE].reset_index(drop=True)
    df["source"] = "Data2_GPS"
    df["class"] = "Travelled"
    df["run"] = run_folder.name
    df["layer"] = df["source"] + " | " + df["class"]
    return df


def load_selected_runs() -> List[str]:
    if not MANIFEST.exists():
        return []
    try:
        m = json.loads(MANIFEST.read_text(encoding="utf-8"))
    except Exception:
        return []
    return [x["run"] for x in m.get("selected", []) if "run" in x]


def center_from_df(df: pd.DataFrame) -> Optional[dict]:
    if df.empty:
        return None
    return {"lat": float(df["Latitude"].median()), "lon": float(df["Longitude"].median())}


def build_predicted_segments(seg_labeled: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Create a DataFrame with predicted labels for the same rows as segments_labeled.csv.

    Requires best_model.joblib which stores {'model', 'classes', 'feature_cols'}.
    """
    if joblib is None:
        print("[PRED] joblib not available -> skipping predicted labels")
        return None
    if not BEST_MODEL.exists():
        print("[PRED] best_model.joblib not found -> skipping predicted labels")
        return None

    bundle = joblib.load(BEST_MODEL)
    model = bundle.get("model")
    classes = bundle.get("classes")
    feat_cols = bundle.get("feature_cols")

    if model is None or classes is None or feat_cols is None:
        print("[PRED] best_model.joblib missing required keys (model/classes/feature_cols)")
        return None

    missing = [c for c in feat_cols if c not in seg_labeled.columns]
    if missing:
        print("[PRED] Missing feature columns in segments_labeled.csv:", missing)
        return None

    X = seg_labeled[feat_cols].to_numpy(dtype=float)
    yhat = model.predict(X)
    pred_lbl = [classes[int(i)] for i in yhat]

    out = seg_labeled[["Latitude", "Longitude"]].copy()
    out["run"] = seg_labeled["run"] if "run" in seg_labeled.columns else "-"
    out["source"] = "Data2_Predicted"
    out["class"] = pred_lbl
    out["layer"] = out["source"] + " | " + out["class"]
    return out


def main():
    frames: List[pd.DataFrame] = []

    # --- Data 1 reference points ---
    bridge = load_points_csv(DATA1_DIR / "converted_coordinates_Resultat_Bridge.csv")
    bridge["source"] = "Data1"; bridge["class"] = "Bridge"; bridge["run"] = "-"
    bridge["layer"] = bridge["source"] + " | " + bridge["class"]

    rail = load_points_csv(DATA1_DIR / "converted_coordinates_Resultat_RailJoint.csv")
    rail["source"] = "Data1"; rail["class"] = "RailJoint"; rail["run"] = "-"
    rail["layer"] = rail["source"] + " | " + rail["class"]

    turn = load_points_csv(DATA1_DIR / "converted_coordinates_Turnout.csv")
    turn["source"] = "Data1"; turn["class"] = "Turnout"; turn["run"] = "-"
    turn["layer"] = turn["source"] + " | " + turn["class"]

    frames += [bridge, rail, turn]

    # --- Data 2 GPS travelled ---
    runs = sorted(set(load_selected_runs()))
    if not runs:
        runs = [p.name for p in DATA2_DIR.iterdir() if p.is_dir()]

    for rn in sorted(runs):
        folder = DATA2_DIR / rn
        if folder.is_dir():
            frames.append(load_run_gps(folder))

    # --- Data 2 segments (pseudo labels) + predicted labels ---
    if SEGMENTS_LABELED.exists():
        seg = pd.read_csv(SEGMENTS_LABELED)
        need = {"Latitude", "Longitude", "label"}
        if need.issubset(seg.columns):
            keep = ["Latitude", "Longitude", "label"] + (["run"] if "run" in seg.columns else [])
            seg2 = seg[keep].dropna().copy()
            seg2 = seg2.rename(columns={"label": "class"})
            seg2["source"] = "Data2_Segments"
            if "run" not in seg2.columns:
                seg2["run"] = "-"
            seg2 = seg2.iloc[::SEG_STRIDE].reset_index(drop=True)
            seg2["layer"] = seg2["source"] + " | " + seg2["class"]
            frames.append(seg2)

            # predicted labels computed on FULL seg dataframe (not downsample), then downsample separately
            pred_full = build_predicted_segments(seg)
            if pred_full is not None:
                pred = pred_full.dropna().iloc[::SEG_STRIDE].reset_index(drop=True)
                frames.append(pred)

                # Print quick sanity counts
                print("[PRED] Predicted label counts:", pd.Series(pred_full["class"]).value_counts().to_dict())

            # Print pseudo label counts
            print("[LAB] segments_labeled label counts:", pd.Series(seg["label"]).value_counts().to_dict())
        else:
            print("segments_labeled.csv missing required columns Latitude/Longitude/label -> skipping segment layers")

    df_all = pd.concat(frames, ignore_index=True)

    print("Counts by source:\n", df_all["source"].value_counts(dropna=False))

    center = center_from_df(df_all[df_all["source"] == "Data2_GPS"])
    if center is None:
        center = center_from_df(df_all)

    fig = px.scatter_mapbox(
        df_all,
        lat="Latitude",
        lon="Longitude",
        color="layer",
        color_discrete_map=COLOR_MAP,
        hover_data={"run": True, "source": True, "class": True},
        zoom=7,
        center=center,
        mapbox_style="open-street-map",
        title="Railway Map: Data 1 vs Data 2 GPS + pseudo-labels + predicted labels"
    )

    # styling (avoid marker.line)
    for tr in fig.data:
        name = getattr(tr, "name", "")  # equals 'layer'
        src = name.split("|")[0].strip() if "|" in name else name
        tr.marker.symbol = SYMBOL_MAP.get(src, "circle")
        if src == "Data1":
            tr.marker.size = 8
            tr.marker.opacity = 0.95
        elif src == "Data2_GPS":
            tr.marker.size = 7
            tr.marker.opacity = 0.65
        elif src == "Data2_Segments":
            tr.marker.size = 7
            tr.marker.opacity = 0.80
        else:  # Data2_Predicted
            tr.marker.size = 7
            tr.marker.opacity = 0.55

    fig.update_layout(height=750, legend_title_text="Layer (source | class)")

    fig.write_html(str(OUT_HTML), include_plotlyjs="cdn")
    print(f"Saved: {OUT_HTML}")

    #fig_png = fig.full_copy()
    #fig_png.update_layout(mapbox_style="carto-positron")  # alt: "carto-darkmatter", "white-bg"


    fig_png = copy.deepcopy(fig)
    fig_png.update_layout(mapbox_style="carto-positron")

    try:
        #fig.write_image(str(OUT_PNG), width=1400, height=900, scale=2)
        fig_png.write_image(str(OUT_PNG), width=1400, height=900, scale=2)  
        print(f"Saved: {OUT_PNG}")
    except Exception as e:
        print("PNG export skipped. To enable, install kaleido: pip install -U kaleido")
        print("Reason:", e)


if __name__ == "__main__":
    main()
