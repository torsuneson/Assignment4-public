# select_runs.py
# -----------------------------------------------------------------------------
# Build manifest selected_runs.json by traversing Data 2/<run>/ folders and
# selecting runs whose GPS track matches Data 1 reference points.
#
# Features:
# - Optional refinement via refine_data.py (creates *.refined.csv + vib features)
# - Uses refined GPS files if present
# - Distance guard (MIN_DISTANCE_KM) to reject stationary runs
# - Reference coverage metric (MIN_REF_COVERAGE)
# - Per-class proximity metrics (bridge_cov / railjoint_cov / turnout_cov)
# - Debug CSV (run_scores_debug.csv) with score/coverage/distance and reasons
# - Ensures at least one selected run has RailJoint proximity (if available)
#
# Jupyter-friendly: import and call build_manifest().
# -----------------------------------------------------------------------------

from __future__ import annotations

import os
import json
from pathlib import Path
from typing import Optional, Dict, List, Any

import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError

import refine_data as ref

# ====================
# USER SETTINGS
# ====================
DATA1_DIR = "Data 1"
DATA2_DIR = "Data 2"
OUT_MANIFEST = "selected_runs.json"
QUICK_MANIFEST = "quick_manifest.json"
DEBUG_CSV = "run_scores_debug.csv"

# Selection criteria (overall GPS match vs. all Data 1 points)
THRESHOLD_M = 250
MIN_SHARE = 0.25 #0.90 innan 

USE_DISTANCE_BAND = False #True take out too many runs 
MIN_DISTANCE_KM_BAND = 50.0
MAX_DISTANCE_KM_BAND = 120.0

# Guards (reject stationary / too-local runs)
USE_DISTANCE_GUARD = True
MIN_DISTANCE_KM = 0.40
GPS_STRIDE_DIST = 5

USE_REF_COVERAGE = True
MIN_REF_COVERAGE = 0.01 #0.05
GPS_STRIDE = 10

# Ensure RailJoint is represented (if possible)
ENSURE_RAILJOINT_RUN = True
MIN_RJ_COV = 0.01          # railjoint_cov threshold for "has railjoint" (fraction of GPS samples)
LOOSE_MIN_SCORE = 0.20     # allow a lower total score for the forced run

# Refinement
REFINE_BEFORE_SELECT = True
REFINE_OVERWRITE = False

# Data2 file hints
LAT_HINT = "GPS.latitude"
LON_HINT = "GPS.longitude"
SPD_HINT = "GPS.speed"
SAT_HINT = "GPS.satellites"
V1_HINTS = ["CH1", "ACCEL"]
V2_HINTS = ["CH2", "ACCEL"]

# Data1 reference filenames
DATA1_FILES = [
    "converted_coordinates_Resultat_Bridge.csv",
    "converted_coordinates_Resultat_RailJoint.csv",
    "converted_coordinates_Turnout.csv",
]

# Polyline/corridor selection (Data1 -> polyline -> GPS corridor match)
USE_CORRIDOR = True
CORRIDOR_RADIUS_M = 500.0
MIN_CORRIDOR_SHARE = 0.60

# Quick route precheck (before refinement)
USE_ROUTE_PREFILTER = True
MIN_BBOX_SHARE = 0.60      # 60% av punkterna inom Data1-bbox (med pad)
BBOX_PAD_DEG = 0.10


# ====================
# Helpers
# ====================


def quick_bbox_from_ref(ref_df: pd.DataFrame, pad_deg: float = 0.10):
    lat_min, lat_max = ref_df["Latitude"].min(), ref_df["Latitude"].max()
    lon_min, lon_max = ref_df["Longitude"].min(), ref_df["Longitude"].max()
    return (lat_min - pad_deg, lat_max + pad_deg, lon_min - pad_deg, lon_max + pad_deg)

def quick_bbox_share(lat_path: str, lon_path: str, bbox, stride: int = 25) -> float:
    lat, lon = _load_clean_gps(lat_path, lon_path)
    if len(lat) < 10:
        return 0.0
    lat_s = lat.iloc[::stride].to_numpy(float)
    lon_s = lon.iloc[::stride].to_numpy(float)
    (lat_min, lat_max, lon_min, lon_max) = bbox
    inside = ((lat_s >= lat_min) & (lat_s <= lat_max) & (lon_s >= lon_min) & (lon_s <= lon_max)).sum()
    return float(inside) / float(len(lat_s))


def read_single_col_csv(path: str) -> pd.Series:
    """Robust single-column CSV reader (handles empty files)."""
    try:
        df = pd.read_csv(path, header=None)
    except EmptyDataError:
        return pd.Series(dtype="float64")
    except Exception as e:
        print(f"Read error in {path}: {e}")
        return pd.Series(dtype="float64")
    if df.shape[1] == 0:
        return pd.Series(dtype="float64")
    s = pd.to_numeric(df.iloc[:, 0], errors="coerce")
    return s.reset_index(drop=True)


def haversine_m(lat1, lon1, lat2, lon2) -> np.ndarray:
    R = 6371000.0
    lat1 = np.radians(lat1); lon1 = np.radians(lon1)
    lat2 = np.radians(lat2); lon2 = np.radians(lon2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))



def find_first_file(folder: Path, must_contain: List[str]) -> Optional[str]:
    """Case-insensitive: filename contains ALL substrings. Searches recursively."""
    subs = [s.lower() for s in must_contain]
    matches = []
    for f in folder.rglob("*"):
        if f.is_file():
            name = f.name.lower()
            if all(sub in name for sub in subs):
                matches.append(f)
    if not matches:
        return None
    matches.sort(key=lambda p: (len(p.parts), str(p)))  # prefer shallow
    return str(matches[0])



def load_data1_points_by_class() -> Dict[str, np.ndarray]:
    """Load Data 1 points per class (Bridge/RailJoint/Turnout)."""
    files = {
        "Bridge": Path(DATA1_DIR) / "converted_coordinates_Resultat_Bridge.csv",
        "RailJoint": Path(DATA1_DIR) / "converted_coordinates_Resultat_RailJoint.csv",
        "Turnout": Path(DATA1_DIR) / "converted_coordinates_Turnout.csv",
    }
    out: Dict[str, np.ndarray] = {}
    for k, p in files.items():
        df = pd.read_csv(p)
        df.columns = df.columns.str.strip()
        out[k] = df[["Latitude", "Longitude"]].dropna().to_numpy(float)
    return out


def build_reference_track() -> pd.DataFrame:
    """All Data 1 points merged into one reference point cloud."""
    paths = [Path(DATA1_DIR) / fn for fn in DATA1_FILES]
    ref_df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    ref_df.columns = ref_df.columns.str.strip()
    ref_df = ref_df[["Latitude", "Longitude"]].dropna().drop_duplicates()
    return ref_df


def _load_clean_gps(lat_path: str, lon_path: str) -> tuple[pd.Series, pd.Series]:
    lat = read_single_col_csv(lat_path)
    lon = read_single_col_csv(lon_path)
    n = int(min(len(lat), len(lon)))
    if n < 1:
        return pd.Series(dtype=float), pd.Series(dtype=float)
    lat = pd.to_numeric(lat.iloc[:n], errors="coerce")
    lon = pd.to_numeric(lon.iloc[:n], errors="coerce")
    mask = (
        lat.notna() & lon.notna() &
        lat.between(-90, 90) & lon.between(-180, 180) &
        ~((lat == 0) & (lon == 0))
    )
    return lat[mask].reset_index(drop=True), lon[mask].reset_index(drop=True)


def score_against_reference(lat_path: str, lon_path: str, ref_lat: np.ndarray, ref_lon: np.ndarray) -> float:
    lat, lon = _load_clean_gps(lat_path, lon_path)
    if len(lat) < 10:
        return 0.0
    ok = 0
    for la, lo in zip(lat, lon):
        ok += (haversine_m(la, lo, ref_lat, ref_lon).min() <= THRESHOLD_M)
    return ok / float(len(lat))


def total_distance_km(lat_path: str, lon_path: str, stride: int = GPS_STRIDE_DIST) -> float:
    lat, lon = _load_clean_gps(lat_path, lon_path)
    if len(lat) < 2:
        return 0.0
    lat_s = lat.iloc[::stride].to_numpy(float)
    lon_s = lon.iloc[::stride].to_numpy(float)
    if len(lat_s) < 2:
        return 0.0
    d = haversine_m(lat_s[:-1], lon_s[:-1], lat_s[1:], lon_s[1:])
    return float(np.nansum(d)) / 1000.0


def ref_coverage(lat_path: str, lon_path: str, ref_lat: np.ndarray, ref_lon: np.ndarray, stride: int = GPS_STRIDE) -> float:
    lat, lon = _load_clean_gps(lat_path, lon_path)
    if len(lat) < 10 or len(ref_lat) == 0:
        return 0.0
    lat_s = lat.iloc[::stride].to_numpy(float)
    lon_s = lon.iloc[::stride].to_numpy(float)
    if len(lat_s) < 10:
        return 0.0
    covered = 0
    for rla, rlo in zip(ref_lat, ref_lon):
        covered += (haversine_m(rla, rlo, lat_s, lon_s).min() <= THRESHOLD_M)
    return covered / float(len(ref_lat))


def class_coverage(lat_path: str, lon_path: str, pts: np.ndarray, threshold_m: float = THRESHOLD_M, stride: int = GPS_STRIDE) -> float:
    """Fraction of GPS samples close to a given class point cloud."""
    lat, lon = _load_clean_gps(lat_path, lon_path)
    if len(lat) < 10 or pts.size == 0:
        return 0.0
    lat_s = lat.iloc[::stride].to_numpy(float)
    lon_s = lon.iloc[::stride].to_numpy(float)
    if len(lat_s) < 10:
        return 0.0
    close = 0
    for la, lo in zip(lat_s, lon_s):
        if haversine_m(la, lo, pts[:, 0], pts[:, 1]).min() <= threshold_m:
            close += 1
    return close / float(len(lat_s))


def _make_selected_entry(run_name: str,
                         score: float,
                         corridor_score: float,
                         corridor_dir: str, 
                         cov: float,
                         dist_km: float,
                         railjoint_cov: float,
                         bridge_cov: float,
                         turnout_cov: float,
                         lat_path: str,
                         lon_path: str,
                         spd_raw: Optional[str],
                         v1_raw: Optional[str],
                         v2_raw: Optional[str],
                         sat_raw: Optional[str],
                         refine_meta: Any) -> Dict[str, Any]:
    """Build a selected[] entry (same shape for normal and forced picks)."""
    spd_path = str(ref.prefer_refined(Path(spd_raw))) if spd_raw else None
    v1_path = str(ref.prefer_refined(Path(v1_raw))) if v1_raw else None
    v2_path = str(ref.prefer_refined(Path(v2_raw))) if v2_raw else None

    return {
        "run": run_name,
        "score": float(score),
        "ref_coverage": float(cov),
        "corridor_score": float(corridor_score),
        "corridor_dir": corridor_dir,
        "distance_km": float(dist_km),
        "railjoint_cov": float(railjoint_cov),
        "bridge_cov": float(bridge_cov),
        "turnout_cov": float(turnout_cov),
        "files": {
            "latitude": lat_path,
            "longitude": lon_path,
            "speed": spd_path,
            "vibration1": v1_path,
            "vibration2": v2_path,
            "satellites": sat_raw,
        },
        "refine": refine_meta,
    }


def order_points_along_track(latlon: np.ndarray) -> np.ndarray:
    """
    Sortera Data1-punkter längs huvudriktningen (PCA-axel) så vi får en polyline-ish ordning.
    latlon: Nx2 [lat, lon]
    """
    X = latlon.astype(float)
    mu = X.mean(axis=0)
    Xc = X - mu
    _, _, vt = np.linalg.svd(Xc, full_matrices=False)
    dir1 = vt[0]  # 2D riktning
    s = Xc @ dir1
    return X[np.argsort(s)]

def smooth_polyline(poly: np.ndarray, win: int = 9) -> np.ndarray:
    """Enkel glättning (moving average) längs polylinens index."""
    if len(poly) < win:
        return poly
    k = win // 2
    out = poly.copy()
    for i in range(len(poly)):
        a = max(0, i - k)
        b = min(len(poly), i + k + 1)
        out[i] = poly[a:b].mean(axis=0)
    return out

def corridor_share(lat_path: str, lon_path: str, poly: np.ndarray,
                   radius_m: float = 500.0, stride: int = GPS_STRIDE) -> float:
    """
    Andel GPS-samples som ligger inom radius_m från polylinens vertexar (snabb approx).
    """
    lat, lon = _load_clean_gps(lat_path, lon_path)
    if len(lat) < 10 or poly.size == 0:
        return 0.0
    lat_s = lat.iloc[::stride].to_numpy(float)
    lon_s = lon.iloc[::stride].to_numpy(float)
    if len(lat_s) < 10:
        return 0.0

    inside = 0
    for la, lo in zip(lat_s, lon_s):
        d = haversine_m(la, lo, poly[:, 0], poly[:, 1])
        if float(np.min(d)) <= radius_m:
            inside += 1
    return inside / float(len(lat_s))

def build_selected_runs_from_quick_manifest_no_thresholds(
    quick_manifest_path: str = QUICK_MANIFEST,
    out_manifest_path: str = OUT_MANIFEST,
    overwrite_refine: bool = False,
) -> Dict[str, object]:
    """
    Bygger selected_runs.json baserat på quick_manifest.json:
      - använder ENDAST entries med quality == 'good'
      - kör refine för dessa (om REFINE_BEFORE_SELECT=True)
      - beräknar ALLA scores/coverage-mått
      - INGA trösklar används för att filtrera bort runs (helt okritiskt)
    """

    qm = json.loads(Path(quick_manifest_path).read_text(encoding="utf-8"))
    rows = qm.get("selected", [])
    good = [r for r in rows if str(r.get("quality", "")).lower() == "good"]
    print(f"[QUICK->FULL] total={len(rows)} good={len(good)} from {quick_manifest_path}")

    # Bygg referenser (som vanliga build_manifest)
    ref_df = build_reference_track()
    ref_lat = ref_df["Latitude"].to_numpy(dtype=float)
    ref_lon = ref_df["Longitude"].to_numpy(dtype=float)

    ref_latlon = ref_df[["Latitude", "Longitude"]].to_numpy(float)
    poly = smooth_polyline(order_points_along_track(ref_latlon), win=9)
    poly_rev = poly[::-1].copy()

    pts_by = load_data1_points_by_class()

    selected: List[Dict[str, Any]] = []
    debug_rows: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []

    for r in good:
        run = r.get("run")
        if not run:
            continue

        folder = Path(DATA2_DIR) / run
        if not folder.is_dir():
            errors.append({"run": run, "error": "folder_not_found"})
            continue

        # 1) refine (endast good)
        refine_meta = None
        if REFINE_BEFORE_SELECT:
            try:
                refine_meta = ref.refine_run_folder(folder, overwrite=overwrite_refine)
            except Exception as e:
                refine_meta = {"error": str(e)}
                errors.append({"run": run, "error": str(e)})

        # 2) hitta filer (använd gärna paths från quick_manifest, annars rekursivt)
        lat_raw = r.get("files", {}).get("latitude") or find_first_file(folder, [LAT_HINT])
        lon_raw = r.get("files", {}).get("longitude") or find_first_file(folder, [LON_HINT])
        if not (lat_raw and lon_raw):
            errors.append({"run": run, "error": "missing_lat_lon"})
            continue

        sat_raw = find_first_file(folder, [SAT_HINT])
        spd_raw = find_first_file(folder, [SPD_HINT])
        v1_raw = find_first_file(folder, V1_HINTS)
        v2_raw = find_first_file(folder, V2_HINTS)

        # 3) använd refined om de finns
        lat_path = str(ref.prefer_refined(Path(lat_raw)))
        lon_path = str(ref.prefer_refined(Path(lon_raw)))

        # 4) beräkna alla mått (utan att använda dem som krav)
        score = score_against_reference(lat_path, lon_path, ref_lat, ref_lon)
        cov = ref_coverage(lat_path, lon_path, ref_lat, ref_lon) if USE_REF_COVERAGE else float("nan")
        dist_km = total_distance_km(lat_path, lon_path) if USE_DISTANCE_GUARD else float("nan")

        corr_fwd = corridor_share(lat_path, lon_path, poly, radius_m=CORRIDOR_RADIUS_M, stride=GPS_STRIDE) if USE_CORRIDOR else float("nan")
        corr_rev = corridor_share(lat_path, lon_path, poly_rev, radius_m=CORRIDOR_RADIUS_M, stride=GPS_STRIDE) if USE_CORRIDOR else float("nan")
        corr_score = float(max(corr_fwd, corr_rev))
        corr_dir = "fwd" if corr_fwd >= corr_rev else "rev"

        bridge_cov = class_coverage(lat_path, lon_path, pts_by.get("Bridge", np.empty((0, 2))))
        railjoint_cov = class_coverage(lat_path, lon_path, pts_by.get("RailJoint", np.empty((0, 2))))
        turnout_cov = class_coverage(lat_path, lon_path, pts_by.get("Turnout", np.empty((0, 2))))

        # 5) Bygg entry i exakt “ordinarie” format (men vi tillåter allt)
        entry = _make_selected_entry(
            run,
            score,
            corr_score,
            corr_dir,
            cov,
            dist_km,
            railjoint_cov,
            bridge_cov,
            turnout_cov,
            lat_path,
            lon_path,
            spd_raw,
            v1_raw,
            v2_raw,
            sat_raw,
            refine_meta,
        )
        entry["quality"] = r.get("quality")
        entry["bbox_share"] = float(r.get("bbox_share", float("nan")))
        entry["source_quick_manifest"] = quick_manifest_path
        selected.append(entry)

        debug_rows.append({
            "run": run,
            "quality": r.get("quality"),
            "score": float(score),
            "ref_coverage": float(cov) if cov == cov else float("nan"),
            "distance_km": float(dist_km) if dist_km == dist_km else float("nan"),
            "corridor_score": float(corr_score),
            "corridor_dir": corr_dir,
            "railjoint_cov": float(railjoint_cov),
            "bridge_cov": float(bridge_cov),
            "turnout_cov": float(turnout_cov),
        })

        print(f"[QUICK->FULL] {run}: score={score:.3f} refcov={cov:.3f} dist_km={dist_km:.1f} rj_cov={railjoint_cov:.3f}")

    manifest = {
        "reference": "Data 1 (Bridge/RailJoint/Turnout)",
        "source_quick_manifest": quick_manifest_path,
        "note": "Built from quick_manifest quality==good. Scores computed but NOT used as thresholds.",
        "selected": selected,
        "errors": errors,
    }

    with open(out_manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    # valfritt men bra: debug dump
    try:
        pd.DataFrame(debug_rows).to_csv("run_scores_debug_quick_full.csv", index=False)
    except Exception:
        pass

    print(f"[QUICK->FULL] Saved {out_manifest_path} selected={len(selected)} errors={len(errors)}")
    return manifest

# ====================
# Manifest builder
# ====================
def build_manifest() -> Dict[str, object]:
    ref_df = build_reference_track()
    bbox = quick_bbox_from_ref(ref_df, pad_deg=BBOX_PAD_DEG)

    ref_lat = ref_df["Latitude"].to_numpy(dtype=float)
    ref_lon = ref_df["Longitude"].to_numpy(dtype=float)

    # Build a polyline-like reference from Data1 points (for corridor matching)
    ref_latlon = ref_df[["Latitude", "Longitude"]].to_numpy(float)
    poly = smooth_polyline(order_points_along_track(ref_latlon), win=9)
    poly_rev = poly[::-1].copy()

    pts_by = load_data1_points_by_class()

    selected: List[Dict[str, Any]] = []
    debug_rows: List[Dict[str, Any]] = []

    data2 = Path(DATA2_DIR)

    for run_name in sorted(os.listdir(data2)):
        folder = data2 / run_name
        if not folder.is_dir():
            continue

        lat_raw = find_first_file(folder, [LAT_HINT])
        lon_raw = find_first_file(folder, [LON_HINT])
        if not (lat_raw and lon_raw):
            continue

        sat_raw = find_first_file(folder, [SAT_HINT])
        spd_raw = find_first_file(folder, [SPD_HINT])
        v1_raw = find_first_file(folder, V1_HINTS)
        v2_raw = find_first_file(folder, V2_HINTS)

        # Cheap route prefilter BEFORE refine
        bbox_share = quick_bbox_share(str(lat_raw), str(lon_raw), bbox) if USE_ROUTE_PREFILTER else 1.0
        if USE_ROUTE_PREFILTER and bbox_share < MIN_BBOX_SHARE:
            print(f"[{run_name}] bbox_share={bbox_share:.2f} -> skip refine + skip run")
            continue

        # refine (optional) - only for likely corridor runs
        refine_meta = None
        if REFINE_BEFORE_SELECT:
            try:
                refine_meta = ref.refine_run_folder(folder, overwrite=REFINE_OVERWRITE)
            except Exception as e:
                print(f"[REFINE ERROR] {run_name}: {e}")
                refine_meta = {"error": str(e)}

        # prefer refined for scoring
        lat_path = str(ref.prefer_refined(Path(lat_raw)))
        lon_path = str(ref.prefer_refined(Path(lon_raw)))

        score = score_against_reference(lat_path, lon_path, ref_lat, ref_lon)
        cov = ref_coverage(lat_path, lon_path, ref_lat, ref_lon) if USE_REF_COVERAGE else float("nan")
        dist_km = total_distance_km(lat_path, lon_path) if USE_DISTANCE_GUARD else float("nan")

        # Corridor score (forward/back)
        corr_fwd = corridor_share(lat_path, lon_path, poly, radius_m=CORRIDOR_RADIUS_M, stride=GPS_STRIDE) if USE_CORRIDOR else float("nan")
        corr_rev = corridor_share(lat_path, lon_path, poly_rev, radius_m=CORRIDOR_RADIUS_M, stride=GPS_STRIDE) if USE_CORRIDOR else float("nan")
        corr_score = float(max(corr_fwd, corr_rev))
        corr_dir = "fwd" if corr_fwd >= corr_rev else "rev"

        bridge_cov = class_coverage(lat_path, lon_path, pts_by.get("Bridge", np.empty((0, 2))))
        railjoint_cov = class_coverage(lat_path, lon_path, pts_by.get("RailJoint", np.empty((0, 2))))
        turnout_cov = class_coverage(lat_path, lon_path, pts_by.get("Turnout", np.empty((0, 2))))

        reason: List[str] = []

        if score < MIN_SHARE:
            if (USE_CORRIDOR and corr_score >= MIN_CORRIDOR_SHARE) or (USE_REF_COVERAGE and cov >= MIN_REF_COVERAGE):
                pass
            else:
                reason.append(f"score<{MIN_SHARE}")

        if USE_REF_COVERAGE and cov < MIN_REF_COVERAGE:
            reason.append(f"coverage<{MIN_REF_COVERAGE}")

        if USE_DISTANCE_GUARD and dist_km < MIN_DISTANCE_KM:
            reason.append(f"distance<{MIN_DISTANCE_KM}km")

        if USE_DISTANCE_BAND and not (MIN_DISTANCE_KM_BAND <= dist_km <= MAX_DISTANCE_KM_BAND):
            reason.append(f"distance_not_in_{MIN_DISTANCE_KM_BAND}-{MAX_DISTANCE_KM_BAND}km")

        if not (v1_raw and v2_raw):
            reason.append("missing_vibration_files")

        debug_rows.append({
            "run": run_name,
            "score": float(score),
            "corridor_score": float(corr_score),
            "corridor_dir": corr_dir,
            "ref_coverage": float(cov),
            "distance_km": float(dist_km),
            "railjoint_cov": float(railjoint_cov),
            "bridge_cov": float(bridge_cov),
            "turnout_cov": float(turnout_cov),
            "lat": lat_path,
            "lon": lon_path,
            "speed": spd_raw,
            "v1": v1_raw,
            "v2": v2_raw,
            "sat": sat_raw,
            "reason": ";".join(reason) if reason else "SELECTED_CANDIDATE",
        })

        reason_str = ";".join(reason) if reason else "SELECTED_CANDIDATE"
        print(f"[{run_name}] score={score:.3f} refcov={cov:.3f} dist_km={dist_km:.3f} rj_cov={railjoint_cov:.3f} -> {reason_str}")

        if reason:
            continue

        selected.append(
            _make_selected_entry(
                run_name,
                score,
                corr_score,
                corr_dir,
                cov,
                dist_km,
                railjoint_cov,
                bridge_cov,
                turnout_cov,
                lat_path,
                lon_path,
                spd_raw,
                v1_raw,
                v2_raw,
                sat_raw,
                refine_meta,
            )
        )

    # ---------------- Finalize after scanning all runs ----------------

    # Ensure at least one RailJoint run if possible
    if ENSURE_RAILJOINT_RUN:
        has_rj = any(s.get("railjoint_cov", 0.0) >= MIN_RJ_COV for s in selected)
        if not has_rj:
            candidates = [r for r in debug_rows if r["score"] >= LOOSE_MIN_SCORE and r.get("railjoint_cov", 0.0) >= MIN_RJ_COV]
            if candidates:
                pick = max(candidates, key=lambda r: r["railjoint_cov"])
                pick_run = pick["run"]

                if not any(s["run"] == pick_run for s in selected):
                    print(f"Forcing RailJoint run into selection: {pick_run} railjoint_cov={pick['railjoint_cov']:.3f} score={pick['score']:.3f}")

                    folder = Path(DATA2_DIR) / pick_run
                    lat_raw = find_first_file(folder, [LAT_HINT])
                    lon_raw = find_first_file(folder, [LON_HINT])
                    sat_raw = find_first_file(folder, [SAT_HINT])
                    spd_raw = find_first_file(folder, [SPD_HINT])
                    v1_raw = find_first_file(folder, V1_HINTS)
                    v2_raw = find_first_file(folder, V2_HINTS)

                    refine_meta = None
                    if REFINE_BEFORE_SELECT:
                        try:
                            refine_meta = ref.refine_run_folder(folder, overwrite=REFINE_OVERWRITE)
                        except Exception as e:
                            print(f"[REFINE ERROR forced] {pick_run}: {e}")
                            refine_meta = {"error": str(e)}

                    lat_path = str(ref.prefer_refined(Path(lat_raw))) if lat_raw else None
                    lon_path = str(ref.prefer_refined(Path(lon_raw))) if lon_raw else None

                    if lat_path and lon_path and v1_raw and v2_raw:
                        score = score_against_reference(lat_path, lon_path, ref_lat, ref_lon)
                        cov = ref_coverage(lat_path, lon_path, ref_lat, ref_lon) if USE_REF_COVERAGE else float("nan")
                        dist_km = total_distance_km(lat_path, lon_path) if USE_DISTANCE_GUARD else float("nan")

                        bridge_cov = class_coverage(lat_path, lon_path, pts_by.get("Bridge", np.empty((0, 2))))
                        railjoint_cov = class_coverage(lat_path, lon_path, pts_by.get("RailJoint", np.empty((0, 2))))
                        turnout_cov = class_coverage(lat_path, lon_path, pts_by.get("Turnout", np.empty((0, 2))))

                        corr_fwd = corridor_share(lat_path, lon_path, poly, radius_m=CORRIDOR_RADIUS_M, stride=GPS_STRIDE) if USE_CORRIDOR else float("nan")
                        corr_rev = corridor_share(lat_path, lon_path, poly_rev, radius_m=CORRIDOR_RADIUS_M, stride=GPS_STRIDE) if USE_CORRIDOR else float("nan")
                        corr_score = float(max(corr_fwd, corr_rev))
                        corr_dir = "fwd" if corr_fwd >= corr_rev else "rev"

                        forced_entry = _make_selected_entry(
                            pick_run,
                            score,
                            corr_score,
                            corr_dir,
                            cov,
                            dist_km,
                            railjoint_cov,
                            bridge_cov,
                            turnout_cov,
                            lat_path,
                            lon_path,
                            spd_raw,
                            v1_raw,
                            v2_raw,
                            sat_raw,
                            refine_meta,
                        )
                        forced_entry["forced"] = True
                        selected.append(forced_entry)
                    else:
                        print(f"Could not force {pick_run}: missing required files after discovery")
            else:
                print("No RailJoint candidate found to force (within thresholds).")

    # sort and write outputs
    selected.sort(key=lambda x: x["score"], reverse=True)
    pd.DataFrame(debug_rows).sort_values("score", ascending=False).to_csv(DEBUG_CSV, index=False)
    print(f"Saved debug scores to {DEBUG_CSV}")

    manifest = {
        "reference": "Data 1 (Bridge/RailJoint/Turnout)",
        "threshold_m": THRESHOLD_M,
        "min_share": MIN_SHARE,
        "min_distance_km": MIN_DISTANCE_KM if USE_DISTANCE_GUARD else None,
        "min_ref_coverage": MIN_REF_COVERAGE if USE_REF_COVERAGE else None,
        "ensure_railjoint_run": ENSURE_RAILJOINT_RUN,
        "selected": selected,
    }

    with open(OUT_MANIFEST, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"Saved manifest to {OUT_MANIFEST}")
    print(f"Selected runs: {len(selected)}")
    if selected:
        print("Top 5:", [(x["run"], round(x["score"], 3), round(x.get("railjoint_cov", 0.0), 3)) for x in selected[:5]])

    return manifest



def build_quick_manifest(out_name: str = QUICK_MANIFEST) -> Dict[str, object]:
    ref_df = build_reference_track()
    bbox = quick_bbox_from_ref(ref_df, pad_deg=BBOX_PAD_DEG)

    selected = []
    data2 = Path(DATA2_DIR)

    for run_name in sorted(os.listdir(data2)):
        folder = data2 / run_name
        if not folder.is_dir():
            continue

        lat_raw = find_first_file(folder, [LAT_HINT])
        lon_raw = find_first_file(folder, [LON_HINT])
        if not (lat_raw and lon_raw):
            continue

        # quick bbox check
        bbox_share = quick_bbox_share(lat_raw, lon_raw, bbox)
        if bbox_share < MIN_BBOX_SHARE:
            continue

        # quick distance check
        dist_km = total_distance_km(lat_raw, lon_raw, stride=GPS_STRIDE_DIST)
        if USE_DISTANCE_BAND and not (MIN_DISTANCE_KM_BAND <= dist_km <= MAX_DISTANCE_KM_BAND):
            continue

        selected.append({
            "run": run_name,
            "distance_km": float(dist_km),
            "bbox_share": float(bbox_share),
            "files": {"latitude": lat_raw, "longitude": lon_raw}
        })

        print(f"[QUICK {run_name}] dist_km={dist_km:.1f} bbox_share={bbox_share:.2f}")

    manifest = {
        "type": "quick_manifest",
        "min_bbox_share": MIN_BBOX_SHARE,
        "distance_band_km": [MIN_DISTANCE_KM_BAND, MAX_DISTANCE_KM_BAND] if USE_DISTANCE_BAND else None,
        "selected": selected
    }

    with open(out_name, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"Saved quick manifest: {out_name} selected={len(selected)}")
    return manifest


if __name__ == "__main__":
    #build_manifest()
     build_quick_manifest(QUICK_MANIFEST)