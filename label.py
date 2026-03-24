# label.py
# -----------------------------------------------------------------------------
# Assignment 4 – Labelling utilities
#
# Responsibilities:
# - Load Data 1 reference points
# - Map vibration segments -> GPS indices
# - Improve weak labels with GPS quality gating (satellites + speed jump guard)
# - Build segments_labeled.csv-compatible DataFrame
# - Add derived impulse-friendly features for short-duration events (RailJoints)
#
# This module contains NO model training.
# -----------------------------------------------------------------------------

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
import pandas as pd

# --------------------
# Default settings (can be overridden by caller)
# --------------------
DT_VIB = 0.002  # 500 Hz
SHIFT_FWD_S = +1.2
SHIFT_REV_S = -1.0
LABEL_RADIUS_M = 500.0

GPS_SAT_MIN = 4            # below this -> low quality
GPS_JUMP_FACTOR = 3.0      # allowed jump ~= factor * speed * dt
GPS_MIN_ALLOW_M = 5.0
GPS_MAX_ALLOW_M = 200.0
GPS_WINDOW = 5             # search +/- window for valid GPS point around computed index


def haversine_m(lat1, lon1, lat2, lon2) -> np.ndarray:
    R = 6371000.0
    lat1 = np.radians(lat1); lon1 = np.radians(lon1)
    lat2 = np.radians(lat2); lon2 = np.radians(lon2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))


def load_data1_points(data1_dir: Path) -> Dict[str, np.ndarray]:
    files = {
        "Bridge": data1_dir / "converted_coordinates_Resultat_Bridge.csv",
        "RailJoint": data1_dir / "converted_coordinates_Resultat_RailJoint.csv",
        "Turnout": data1_dir / "converted_coordinates_Turnout.csv",
    }
    out: Dict[str, np.ndarray] = {}
    for k, p in files.items():
        df = pd.read_csv(p)
        df.columns = df.columns.str.strip()
        out[k] = df[["Latitude", "Longitude"]].dropna().to_numpy(dtype=float)
    return out


def label_point(lat: float, lon: float, ref: Dict[str, np.ndarray], radius_m: float) -> str:
    """Nearest-neighbour label among Bridge/RailJoint/Turnout within radius, else Other."""
    best_cls = None
    best_d = float("inf")
    for cls, pts in ref.items():
        if pts.size == 0:
            continue
        dmin = float(np.min(haversine_m(lat, lon, pts[:, 0], pts[:, 1])))
        if dmin < best_d:
            best_d = dmin
            best_cls = cls
    return best_cls if (best_cls is not None and best_d <= radius_m) else "Other"


def infer_direction(lat: pd.Series, lon: pd.Series) -> int:
    """+1 forward, -1 reverse (simple heuristic based on net movement)."""
    if len(lat) < 2:
        return +1
    dlat = float(lat.iloc[-1] - lat.iloc[0])
    dlon = float(lon.iloc[-1] - lon.iloc[0])
    if abs(dlon) >= abs(dlat):
        return +1 if dlon >= 0 else -1
    return +1 if dlat >= 0 else -1


def estimate_gps_dt(vib_n_samples: int, dt_vib: float, gps_n: int) -> float:
    if gps_n <= 1:
        return 1.0
    return (vib_n_samples * dt_vib) / (gps_n - 1)


def _read_single_col(path: Path) -> Optional[np.ndarray]:
    if path is None or (not Path(path).exists()):
        return None
    try:
        s = pd.read_csv(path, header=None).iloc[:, 0]
    except Exception:
        return None
    arr = pd.to_numeric(s, errors="coerce").ffill().bfill().to_numpy(float)
    return arr


def _load_clean_gps(lat_path: Path, lon_path: Path) -> Tuple[pd.Series, pd.Series]:
    lat = pd.read_csv(lat_path, header=None).iloc[:, 0]
    lon = pd.read_csv(lon_path, header=None).iloc[:, 0]
    lat = pd.to_numeric(lat, errors="coerce")
    lon = pd.to_numeric(lon, errors="coerce")
    n = int(min(len(lat), len(lon)))
    lat = lat.iloc[:n].reset_index(drop=True)
    lon = lon.iloc[:n].reset_index(drop=True)
    mask = lat.notna() & lon.notna() & lat.between(-90, 90) & lon.between(-180, 180) & ~((lat == 0) & (lon == 0))
    return lat[mask].reset_index(drop=True), lon[mask].reset_index(drop=True)


def mark_gps_valid(lat: np.ndarray, lon: np.ndarray, speed_mps: Optional[np.ndarray], sats: Optional[np.ndarray], dt_gps: float) -> np.ndarray:
    """Return boolean mask of valid GPS indices using satellites threshold + speed-based jump guard."""
    n = min(len(lat), len(lon))
    valid = np.ones(n, dtype=bool)

    # Satellites gating
    if sats is not None and len(sats) >= n:
        valid &= (sats[:n] >= GPS_SAT_MIN)

    # Jump guard using haversine between consecutive points
    if n >= 3:
        R = 6371000.0
        latr = np.radians(lat[:n]); lonr = np.radians(lon[:n])
        dlat = latr[1:] - latr[:-1]
        dlon = lonr[1:] - lonr[:-1]
        a = np.sin(dlat/2)**2 + np.cos(latr[:-1]) * np.cos(latr[1:]) * np.sin(dlon/2)**2
        dist = 2 * R * np.arcsin(np.sqrt(a))

        if speed_mps is None or len(speed_mps) < n:
            allow = np.full(n-1, 60.0)  # generous fallback
        else:
            allow = np.clip(GPS_JUMP_FACTOR * speed_mps[:n-1] * dt_gps, GPS_MIN_ALLOW_M, GPS_MAX_ALLOW_M)

        bad = dist > allow
        valid[1:][bad] = False

    return valid


def choose_nearest_valid_index(gi: int, valid: np.ndarray) -> int:
    """If gi invalid, search small window around gi for nearest valid index."""
    n = len(valid)
    gi = max(0, min(n-1, int(gi)))
    if valid[gi]:
        return gi
    for d in range(1, GPS_WINDOW+1):
        a = gi - d
        b = gi + d
        if a >= 0 and valid[a]:
            return a
        if b < n and valid[b]:
            return b
    return gi


def add_impulse_features(feat: pd.DataFrame, seg_len: int, dt_vib: float) -> pd.DataFrame:
    """Add derived impulse-friendly features without needing raw vibration.

    Uses existing columns if present (spikecount, peak, rms, p2p, kurtosis, bandpower_*).
    """
    out = feat.copy()

    # Basic derived ratios
    if "peak" in out.columns and "rms" in out.columns:
        out["peak_over_rms"] = out["peak"] / (out["rms"] + 1e-12)

    if "p2p" in out.columns and "rms" in out.columns:
        out["p2p_over_rms"] = out["p2p"] / (out["rms"] + 1e-12)

    # Spike density (spikes per second)
    if "spikecount" in out.columns:
        seg_seconds = float(seg_len) * float(dt_vib)
        out["spike_density"] = out["spikecount"] / max(seg_seconds, 1e-9)

    # If multiple bandpower columns exist, add simple ratios (high/low)
    bp_cols = [c for c in out.columns if "bandpower" in c or c.startswith("bp")]
    if len(bp_cols) >= 2:
        # pick first two deterministically
        bp_cols = sorted(bp_cols)
        out["bandpower_ratio_1"] = out[bp_cols[-1]] / (out[bp_cols[0]] + 1e-12)

    return out

def build_inference_dataset(
    manifest_path: Path,
    data2_dir: Path,
    dt_vib: float = DT_VIB,
) -> pd.DataFrame:
    """
    Bygger ett inference-dataset från selected_runs.json (utan etiketter),
    och bäddar in Latitude/Longitude per segment (mittpunktsprov på GPS).

    Retur: DataFrame med kolumner:
      run, seg, Latitude, Longitude + alla numeriska features
      (inkl. våra derivat: peak_over_rms, p2p_over_rms, spike_density, bandpower_ratio_1)
    """
    print(f"[INFER-DEBUG] build_inference_dataset")
    
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    selected = manifest.get("selected", [])
    if not selected:
        raise RuntimeError("No selected runs in manifest. Run select_runs.py first.")

    rows = []
    last_feat_cols = None  # används för att bygga header vid ev. tomt resultat

    for item in selected:
        run = item["run"]
        folder = Path(data2_dir) / run

        feat_path = folder / "vibration_segments.refined.features.csv"
        if not feat_path.exists():
            print(f"[INFER] Missing features for {run}: {feat_path} - skipping")
            continue

        # GPS: använd refined om finns, annars raw som fallback
        lat_path_ref = folder / "GPS.latitude.refined.csv"
        lon_path_ref = folder / "GPS.longitude.refined.csv"
        lat_path_raw = folder / "GPS.latitude.csv"
        lon_path_raw = folder / "GPS.longitude.csv"

        if lat_path_ref.exists() and lon_path_ref.exists():
            lat_path, lon_path = lat_path_ref, lon_path_ref
        elif lat_path_raw.exists() and lon_path_raw.exists():
            lat_path, lon_path = lat_path_raw, lon_path_raw
            print(f"[INFER] Using RAW GPS for {run}")
        else:
            print(f"[INFER] Missing GPS files for {run}: {lat_path_ref} / {lon_path_ref} (or raw) - skipping")
            continue

        print(f"[INFER-DEBUG] build_inference_dataset lat_path:{lat_path}, lon_path:{lon_path} ")

        # 1) Ladda features
        feat = pd.read_csv(feat_path)
        last_feat_cols = [c for c in feat.columns if c != "seg"]  # spara för header

        # 2) seg_len från meta (för spike_density mm.)
        seg_len = 5000
        meta_path = folder / "refine.refined.meta.json"
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                vibm = meta.get("vibration", {})
                if "seg_len" in vibm:
                    seg_len = int(float(vibm["seg_len"]))
            except Exception:
                pass

        # 3) Lägg till våra derivat-features (samma som i build_labeled_dataset)
        feat = add_impulse_features(feat, seg_len=seg_len, dt_vib=dt_vib)

        # 4) Ladda GPS och städa
        lat, lon = _load_clean_gps(lat_path, lon_path)
        print(f"[INFER-DEBUG] Run={run} GPS loaded: lat={len(lat)} lon={len(lon)}")
        gps_n = int(min(len(lat), len(lon)))
        if gps_n < 2:
            print(f"[INFER] Too few GPS points for {run} - skipping")
            continue

        # 5) Satellites/speed (för quality-gating, om tillgängligt)
        def _read_optional_onecol(p: Path):
            try:
                s = pd.read_csv(p, header=None).iloc[:, 0]
                return pd.to_numeric(s, errors="coerce").to_numpy(float)
            except Exception:
                return None

        sats = _read_optional_onecol(folder / "GPS.satellites.refined.csv")
        if sats is None:
            sats = _read_optional_onecol(folder / "GPS.satellites.csv")
        if sats is not None and len(sats) >= gps_n:
            sats = sats[:gps_n]
        else:
            sats = None

        spd = _read_optional_onecol(folder / "GPS.speed.refined.csv")
        if spd is None:
            spd = _read_optional_onecol(folder / "GPS.speed.csv")
        speed_mps = None
        if spd is not None and len(spd) >= gps_n:
            spd = spd[:gps_n]
            speed_mps = spd / 3.6  # km/h -> m/s

        # 6) Uppskatta gps_dt för tids->index-mappning
        vib_n_samples = int(feat.shape[0] * seg_len)
        gps_dt = estimate_gps_dt(vib_n_samples, dt_vib, gps_n)

        # 7) Quality-mask (om vi har signaler), annars allt True
        if sats is not None or speed_mps is not None:
            valid_gps = mark_gps_valid(
                lat.to_numpy(float),
                lon.to_numpy(float),
                speed_mps=speed_mps,
                sats=sats,
                dt_gps=float(gps_dt),
            )
        else:
            valid_gps = np.ones(gps_n, dtype=bool)

        # 8) Riktning + SHIFT som i labeling
        dir_sign = infer_direction(lat, lon)
        shift_s = SHIFT_FWD_S if dir_sign > 0 else SHIFT_REV_S

        # 9) Mittpunktsindex per segment -> lat/lon
        for _, r in feat.iterrows():
            seg = int(r["seg"])
            t_mid = (seg * seg_len + seg_len / 2) * dt_vib + shift_s
            gi = int(round(t_mid / gps_dt))
            gi = max(0, min(gps_n - 1, gi))

            # Välj närmaste giltiga GPS-punkt om denna är ogiltig
            gi = choose_nearest_valid_index(gi, valid_gps)

            row = {
                "run": run,
                "seg": seg,
                "Latitude": float(lat.iloc[gi]),
                "Longitude": float(lon.iloc[gi]),
            }
            for c in feat.columns:
                if c != "seg":
                    row[c] = float(r[c])
            rows.append(row)

    # 10) Bygg DF – om rows är tom, skapa ändå kolumnheader (inkl. Latitude/Longitude)
    if rows:
        df = pd.DataFrame(rows)
    else:
        base_cols = ["run", "seg", "Latitude", "Longitude"]
        extra = last_feat_cols if last_feat_cols else []
        df = pd.DataFrame(columns=base_cols + extra)

    return df

def build_labeled_dataset(
    manifest_path: Path,
    data1_dir: Path,
    data2_dir: Path,
    label_radius_m: float = LABEL_RADIUS_M,
    dt_vib: float = DT_VIB,
    shift_fwd_s: float = SHIFT_FWD_S,
    shift_rev_s: float = SHIFT_REV_S,
    dt_gps_assumed: Optional[float] = None,
) -> pd.DataFrame:
    """Build labeled dataset from selected_runs.json.

    Keeps dt_gps estimate mapping, but improves GPS sampling choice via quality gating.

    dt_gps_assumed: if provided, used only for jump guard; else estimated from durations.
    """
    ref = load_data1_points(data1_dir)
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    selected = manifest.get("selected", [])
    if not selected:
        raise RuntimeError("No selected runs in manifest. Run select_runs.py first or relax thresholds.")

    rows: List[Dict[str, Any]] = []

    for item in selected:
        run = item["run"]
        folder = Path(data2_dir) / run

        feat_path = folder / "vibration_segments.refined.features.csv"
        lat_path = folder / "GPS.latitude.refined.csv"
        lon_path = folder / "GPS.longitude.refined.csv"

        if not feat_path.exists():
            raise FileNotFoundError(f"Missing features file for {run}: {feat_path}")
        if not (lat_path.exists() and lon_path.exists()):
            raise FileNotFoundError(f"Missing refined GPS for {run}: {lat_path} / {lon_path}")

        feat = pd.read_csv(feat_path)

        # seg_len from meta if present
        meta_path = folder / "refine.refined.meta.json"
        seg_len = 5000
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                vibm = meta.get("vibration", {})
                if "seg_len" in vibm:
                    seg_len = int(float(vibm["seg_len"]))
            except Exception:
                pass

        # Add impulse-friendly derived features (Step 2)
        feat = add_impulse_features(feat, seg_len=seg_len, dt_vib=dt_vib)

        lat, lon = _load_clean_gps(lat_path, lon_path)
        gps_n = int(min(len(lat), len(lon)))
        if gps_n < 2:
            continue
        lat = lat.iloc[:gps_n]
        lon = lon.iloc[:gps_n]

        # Optional quality signals
        sats = _read_single_col(folder / "GPS.satellites.refined.csv")
        if sats is None:
            sats = _read_single_col(folder / "GPS.satellites.csv")
        if sats is not None:
            sats = sats[:gps_n]

        spd = _read_single_col(folder / "GPS.speed.refined.csv")
        if spd is None:
            spd = _read_single_col(folder / "GPS.speed.csv")
        speed_mps = None
        if spd is not None:
            spd = spd[:gps_n]
            # Assume km/h in this dataset
            speed_mps = spd / 3.6

        vib_n_samples = int(feat.shape[0] * seg_len)
        gps_dt = estimate_gps_dt(vib_n_samples, dt_vib, gps_n)

        # Use gps_dt for jump guard if no explicit dt provided
        dt_for_jump = float(dt_gps_assumed) if dt_gps_assumed is not None else float(gps_dt)

        valid_gps = mark_gps_valid(lat.to_numpy(float), lon.to_numpy(float), speed_mps, sats, dt_for_jump)

        dir_sign = infer_direction(lat, lon)
        shift_s = shift_fwd_s if dir_sign > 0 else shift_rev_s

        for _, r in feat.iterrows():
            seg = int(r["seg"])
            t_mid = (seg * seg_len + seg_len / 2) * dt_vib + shift_s
            gi = int(round(t_mid / gps_dt))
            gi = max(0, min(gps_n - 1, gi))

            # Step 1 improvement: choose nearest valid GPS sample
            gi = choose_nearest_valid_index(gi, valid_gps)

            la = float(lat.iloc[gi])
            lo = float(lon.iloc[gi])
            lab = label_point(la, lo, ref, radius_m=label_radius_m)

            row = {"run": run, "seg": seg, "Latitude": la, "Longitude": lo, "label": lab}
            for c in feat.columns:
                if c != "seg":
                    row[c] = float(r[c])
            rows.append(row)

    return pd.DataFrame(rows)

if __name__ == "__main__":
    # Optional convenience: build and save segments_labeled.csv when executed directly
    df = build_labeled_dataset(
        manifest_path=Path("selected_runs.json"),
        data1_dir=Path("Data 1"),
        data2_dir=Path("Data 2"),
    )
    out = Path("segments_labeled.csv")
    df.to_csv(out, index=False)
    print(f"Saved: {out} rows={len(df)}")
    if 'label' in df.columns:
        print(df['label'].value_counts(dropna=False))
