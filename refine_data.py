# refine_data.py
# -----------------------------------------------------------------------------
# Preprocess/refine Data 2 raw files (GPS + speed + vibration) and cache outputs
# next to the original files as *.refined.*
#
# Design goals:
# - Jupyter-friendly (pure functions)
# - Minimal dependencies: numpy, pandas
# - Safe defaults: remove obvious GPS junk ((0,0), out-of-range), filter GPS spikes
# - Produce refined GPS lat/lon and speed for cleaner plotting + selection
# - Produce refined vibration signals + per-segment features + spike/event detection
#
# NOTE: This module does NOT assume absolute timestamps in Data 2; it uses sample
# index as time base (consistent with your current Code 2 approach).
# -----------------------------------------------------------------------------

from __future__ import annotations

import os
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, List

import numpy as np
import pandas as pd

# =====================
# Defaults / settings
# =====================

REFINED_SUFFIX = ".refined"

# GPS
MIN_SATS_DEFAULT = 4
MAX_JUMP_M_DEFAULT = 120.0     # max allowed jump between consecutive valid points
EMA_ALPHA_DEFAULT = 0.20       # smoothing strength for GPS and speed
DO_SMOOTH_GPS_DEFAULT = True

# Speed
SPEED_CLIP_KMH_DEFAULT = 250.0  # hard cap to remove obvious spikes
DO_SMOOTH_SPEED_DEFAULT = True

# Vibration
DT_VIB_DEFAULT = 0.002          # seconds/sample (500 Hz)
SEG_DUR_S_DEFAULT = 10.0        # seconds
HP_WINDOW_DEFAULT = 501         # samples for moving-average high-pass (~1s at 500Hz)
LP_WINDOW_DEFAULT = 11          # samples for moving-average low-pass (mild)
ROBUST_Z_MEDIAN_DEFAULT = True
SPIKE_Z_DEFAULT = 6.0           # z-score threshold for spike detection
SPIKE_MIN_SEP_S_DEFAULT = 1.0   # minimum separation between spikes (seconds)


# =====================
# Utility
# =====================

def read_single_col_csv(path: os.PathLike | str) -> pd.Series:
    """Read a single-column CSV without header into a numeric Series."""
    try:
        df = pd.read_csv(path, header=None)
    except Exception:
        return pd.Series(dtype="float64")
    if df.shape[1] == 0:
        return pd.Series(dtype="float64")
    s = pd.to_numeric(df.iloc[:, 0], errors="coerce")
    return s.reset_index(drop=True)


def write_single_col_csv(path: os.PathLike | str, series: pd.Series | np.ndarray) -> None:
    """Write a single-column CSV without header."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(series, np.ndarray):
        series = pd.Series(series)
    series.to_csv(p, index=False, header=False)


def haversine_m(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Vectorized haversine distance in meters."""
    R = 6371000.0
    lat1 = np.radians(lat1); lon1 = np.radians(lon1)
    lat2 = np.radians(lat2); lon2 = np.radians(lon2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))


def ema(series: pd.Series, alpha: float) -> pd.Series:
    return series.ewm(alpha=alpha, adjust=False).mean()


def moving_average(x: np.ndarray, window: int) -> np.ndarray:
    """Centered moving average, same length output."""
    window = int(window)
    if window <= 1:
        return x
    if window % 2 == 0:
        window += 1
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(x, kernel, mode="same")


def robust_zscore(x: np.ndarray) -> np.ndarray:
    """Robust z-score using median and MAD."""
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale < 1e-12:
        # fallback to std
        m = np.nanmean(x)
        s = np.nanstd(x)
        return (x - m) / (s + 1e-12)
    return (x - med) / (scale + 1e-12)


# =====================
# GPS refinement
# =====================

@dataclass
class GPSRefineConfig:
    min_sats: int = MIN_SATS_DEFAULT
    max_jump_m: float = MAX_JUMP_M_DEFAULT
    ema_alpha: float = EMA_ALPHA_DEFAULT
    do_smooth: bool = DO_SMOOTH_GPS_DEFAULT


def refine_gps_series(
    lat: pd.Series,
    lon: pd.Series,
    sats: Optional[pd.Series] = None,
    cfg: GPSRefineConfig = GPSRefineConfig(),
) -> Tuple[pd.Series, pd.Series, Dict[str, float]]:
    """Clean + spike-filter + interpolate + (optional) smooth GPS."""
    n = int(min(len(lat), len(lon)))
    lat = pd.to_numeric(lat.iloc[:n], errors="coerce").reset_index(drop=True)
    lon = pd.to_numeric(lon.iloc[:n], errors="coerce").reset_index(drop=True)

    # Basic validity
    good = (
        lat.between(-90, 90)
        & lon.between(-180, 180)
        & ~((lat == 0) & (lon == 0))
    )

    # Satellites quality
    if sats is not None and len(sats) >= n:
        sats_n = pd.to_numeric(sats.iloc[:n], errors="coerce")
        sats_n = sats_n.ffill().bfill()
        good &= (sats_n >= cfg.min_sats)

    # Apply mask -> NaN
    lat2 = lat.where(good, np.nan)
    lon2 = lon.where(good, np.nan)

    # Consecutive jump filter (forward pass)
    last_lat = last_lon = None
    removed_jump = 0
    for i in range(n):
        la = lat2.iat[i]
        lo = lon2.iat[i]
        if pd.isna(la) or pd.isna(lo):
            continue
        if last_lat is None:
            last_lat, last_lon = float(la), float(lo)
            continue
        d = float(haversine_m(la, lo, last_lat, last_lon))
        if d > cfg.max_jump_m:
            lat2.iat[i] = np.nan
            lon2.iat[i] = np.nan
            removed_jump += 1
        else:
            last_lat, last_lon = float(la), float(lo)

    # If too little left -> return original cleaned (no jump filter)
    if lat2.notna().sum() < 10:
        lat2 = lat.where(lat.between(-90, 90) & ~((lat == 0) & (lon == 0)), np.nan)
        lon2 = lon.where(lon.between(-180, 180) & ~((lat == 0) & (lon == 0)), np.nan)

    # Interpolate gaps
    lat2 = lat2.interpolate(limit_direction="both")
    lon2 = lon2.interpolate(limit_direction="both")

    # Smooth jitter
    if cfg.do_smooth:
        lat2 = ema(lat2, cfg.ema_alpha)
        lon2 = ema(lon2, cfg.ema_alpha)

    meta = {
        "n": float(n),
        "removed_invalid": float((~good).sum()),
        "removed_jump": float(removed_jump),
        "lat_span": float(lat2.max() - lat2.min()),
        "lon_span": float(lon2.max() - lon2.min()),
    }
    return lat2, lon2, meta


# =====================
# Speed refinement
# =====================

@dataclass
class SpeedRefineConfig:
    ema_alpha: float = EMA_ALPHA_DEFAULT
    do_smooth: bool = DO_SMOOTH_SPEED_DEFAULT
    clip_kmh: float = SPEED_CLIP_KMH_DEFAULT


def refine_speed_series(speed: pd.Series, cfg: SpeedRefineConfig = SpeedRefineConfig()) -> Tuple[pd.Series, Dict[str, float]]:
    """Clean + clip + smooth + provide a normalized copy for plotting."""
    s = pd.to_numeric(speed, errors="coerce").reset_index(drop=True)
    # Remove negatives, clip hard
    s = s.where(s >= 0, np.nan)
    s = s.clip(lower=0, upper=cfg.clip_kmh)

    # Interpolate gaps
    s = s.interpolate(limit_direction="both")

    if cfg.do_smooth:
        s = ema(s, cfg.ema_alpha)

    meta = {
        "n": float(len(s)),
        "min": float(np.nanmin(s)) if len(s) else 0.0,
        "max": float(np.nanmax(s)) if len(s) else 0.0,
        "mean": float(np.nanmean(s)) if len(s) else 0.0,
    }
    return s, meta


# =====================
# Vibration refinement
# =====================

@dataclass
class VibrationRefineConfig:
    dt: float = DT_VIB_DEFAULT
    seg_dur_s: float = SEG_DUR_S_DEFAULT
    hp_window: int = HP_WINDOW_DEFAULT
    lp_window: int = LP_WINDOW_DEFAULT
    robust_z: bool = ROBUST_Z_MEDIAN_DEFAULT
    spike_z: float = SPIKE_Z_DEFAULT
    spike_min_sep_s: float = SPIKE_MIN_SEP_S_DEFAULT


def preprocess_vibration_channel(x: np.ndarray, cfg: VibrationRefineConfig) -> np.ndarray:
    """Remove DC/trend (high-pass via moving average), mild low-pass, then z-score."""
    x = x.astype(float)
    # high-pass: subtract moving average (trend)
    trend = moving_average(x, cfg.hp_window)
    hp = x - trend
    # low-pass: mild MA to reduce high-frequency noise
    lp = moving_average(hp, cfg.lp_window)
    # z-score (robust)
    if cfg.robust_z:
        z = robust_zscore(lp)
    else:
        z = (lp - np.nanmean(lp)) / (np.nanstd(lp) + 1e-12)
    return z


def segment_vibration(z1: np.ndarray, z2: np.ndarray, cfg: VibrationRefineConfig) -> Tuple[np.ndarray, int]:
    seg_len = int(round(cfg.seg_dur_s / cfg.dt))
    n = min(len(z1), len(z2))
    nseg = n // seg_len
    if nseg <= 0:
        return np.empty((0, seg_len, 2)), seg_len
    arr = np.stack([z1[:nseg*seg_len], z2[:nseg*seg_len]], axis=1)  # (N,2)
    segs = arr.reshape(nseg, seg_len, 2)
    return segs, seg_len


def features_per_segment(segs: np.ndarray, cfg: VibrationRefineConfig) -> pd.DataFrame:
    """Compute per-segment features for both channels.

    Adds:
      - Base: RMS, peak, peak-to-peak, kurtosis (per channel)
      - A-features (impulse-friendly): crest_factor, impulse_factor, zcr
      - Simple FFT bandpower (numpy rfft) in a configurable band

    Notes:
      - Uses chunked FFT to keep memory usage reasonable.
    """
    if segs.size == 0:
        return pd.DataFrame()

    x1 = segs[:, :, 0]
    x2 = segs[:, :, 1]

    def _basic_feats(x: np.ndarray):
        # x: (n_segments, seg_len)
        rms = np.sqrt(np.mean(x**2, axis=1))
        peak = np.max(np.abs(x), axis=1)
        p2p = np.max(x, axis=1) - np.min(x, axis=1)
        # kurtosis approx without scipy (per-segment)
        m = np.mean(x, axis=1, keepdims=True)
        xc = x - m
        v = np.mean(xc**2, axis=1) + 1e-12
        k = np.mean(xc**4, axis=1) / (v**2)
        return rms, peak, p2p, k

    def _a_feats(x: np.ndarray, rms: np.ndarray, peak: np.ndarray):
        # Crest factor: peak/rms
        crest = peak / (rms + 1e-12)
        # Impulse factor: peak/mean(|x|)
        mean_abs = np.mean(np.abs(x), axis=1)
        impulse = peak / (mean_abs + 1e-12)
        # Zero-crossing rate (sign changes)
        # Use product of adjacent samples to detect sign changes
        zcr = np.mean((x[:, 1:] * x[:, :-1]) < 0, axis=1)
        return crest, impulse, zcr

    def _bandpower_fft(x: np.ndarray, fs: float, fmin: float, fmax: float, chunk: int = 256):
        """Bandpower via numpy rfft + trapezoidal integration (chunked)."""
        nseg, n = x.shape
        if n < 8:
            return np.zeros(nseg, dtype=float)
        f = np.fft.rfftfreq(n, d=1.0 / fs)
        i0 = int(np.searchsorted(f, fmin, side="left"))
        i1 = int(np.searchsorted(f, fmax, side="right"))
        i0 = max(0, min(len(f) - 1, i0))
        i1 = max(i0 + 1, min(len(f), i1))

        out = np.zeros(nseg, dtype=float)
        # PSD scaling (density-ish): |X|^2/(fs*n)
        for s in range(0, nseg, chunk):
            sl = slice(s, min(s + chunk, nseg))
            X = np.fft.rfft(x[sl, :], axis=1)
            Pxx = (np.abs(X) ** 2) / (fs * n)
            out[sl] = np.trapz(Pxx[:, i0:i1], f[i0:i1], axis=1)
        return out

    # Base features
    rms1, peak1, p2p1, kurt1 = _basic_feats(x1)
    rms2, peak2, p2p2, kurt2 = _basic_feats(x2)

    # A-features
    crest1, imp1, zcr1 = _a_feats(x1, rms1, peak1)
    crest2, imp2, zcr2 = _a_feats(x2, rms2, peak2)

    # Spike counts within segment (using cfg.spike_z threshold on absolute value)
    spike_thr = float(cfg.spike_z)
    spikes1 = np.sum(np.abs(x1) >= spike_thr, axis=1)
    spikes2 = np.sum(np.abs(x2) >= spike_thr, axis=1)

    # FFT bandpower (one simple band by default)
    fs = 1.0 / float(cfg.dt)
    fmin, fmax = 30.0, 120.0
    bp1 = _bandpower_fft(x1, fs, fmin, fmax)
    bp2 = _bandpower_fft(x2, fs, fmin, fmax)

    df = pd.DataFrame({
        "seg": np.arange(segs.shape[0]),
        "rms_ch1": rms1,
        "peak_ch1": peak1,
        "p2p_ch1": p2p1,
        "kurt_ch1": kurt1,
        "crest_ch1": crest1,
        "impulse_ch1": imp1,
        "zcr_ch1": zcr1,
        "spikecount_ch1": spikes1,
        "bp30_120_ch1": bp1,

        "rms_ch2": rms2,
        "peak_ch2": peak2,
        "p2p_ch2": p2p2,
        "kurt_ch2": kurt2,
        "crest_ch2": crest2,
        "impulse_ch2": imp2,
        "zcr_ch2": zcr2,
        "spikecount_ch2": spikes2,
        "bp30_120_ch2": bp2,
    })

    # Combined helper features
    df["energy"] = df["rms_ch1"] + df["rms_ch2"]
    df["peak"] = np.maximum(df["peak_ch1"], df["peak_ch2"])
    df["bp30_120"] = df["bp30_120_ch1"] + df["bp30_120_ch2"]
    df["spikecount"] = df["spikecount_ch1"] + df["spikecount_ch2"]

    return df

def detect_spikes(z: np.ndarray, cfg: VibrationRefineConfig) -> np.ndarray:
    """Return indices of spike peaks in 1D signal using z-score threshold + min separation."""
    zz = robust_zscore(np.abs(z)) if cfg.robust_z else (np.abs(z) - np.mean(np.abs(z))) / (np.std(np.abs(z)) + 1e-12)
    idx = np.where(zz >= cfg.spike_z)[0]
    if len(idx) == 0:
        return idx

    # Non-maximum suppression with min separation
    min_sep = int(round(cfg.spike_min_sep_s / cfg.dt))
    selected = []
    last = -10**12
    for i in idx:
        if i - last >= min_sep:
            selected.append(i)
            last = i
    return np.array(selected, dtype=int)


def refine_vibration(
    v1: pd.Series,
    v2: pd.Series,
    cfg: VibrationRefineConfig = VibrationRefineConfig(),
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, float]]:
    """Return refined sample-level dataframe, per-segment features, spikes, and metadata."""
    n = int(min(len(v1), len(v2)))
    if n < 10:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {"n": float(n)}

    x1 = pd.to_numeric(v1.iloc[:n], errors="coerce").ffill().bfill().to_numpy()
    x2 = pd.to_numeric(v2.iloc[:n], errors="coerce").ffill().bfill().to_numpy()

    z1 = preprocess_vibration_channel(x1, cfg)
    z2 = preprocess_vibration_channel(x2, cfg)

    refined = pd.DataFrame({
        "sample": np.arange(n),
        "ch1": z1,
        "ch2": z2,
    })

    segs, seg_len = segment_vibration(z1, z2, cfg)
    feat = features_per_segment(segs, cfg)

    spikes1 = detect_spikes(z1, cfg)
    spikes2 = detect_spikes(z2, cfg)
    spikes = pd.DataFrame({
        "sample": np.unique(np.concatenate([spikes1, spikes2])) if (len(spikes1) or len(spikes2)) else np.array([], dtype=int),
    })
    if not spikes.empty:
        spikes["time_s"] = spikes["sample"] * cfg.dt

    meta = {
        "n": float(n),
        "seg_len": float(seg_len),
        "n_segments": float(segs.shape[0]),
        "n_spikes": float(len(spikes)),
    }
    return refined, feat, spikes, meta


# =====================
# Folder-level API
# =====================

@dataclass
class RunFiles:
    latitude: Optional[Path] = None
    longitude: Optional[Path] = None
    speed: Optional[Path] = None
    satellites: Optional[Path] = None
    vibration1: Optional[Path] = None
    vibration2: Optional[Path] = None


def find_first_file(folder: os.PathLike | str, must_contain_substrings: List[str]) -> Optional[Path]:
    folder = Path(folder)
    try:
        files = list(folder.iterdir())
    except Exception:
        return None
    for f in files:
        if not f.is_file():
            continue
        name = f.name
        ok = True
        for sub in must_contain_substrings:
            if sub not in name:
                ok = False
                break
        if ok:
            return f
    return None


def prefer_refined(path: Path) -> Path:
    """Return refined sibling if exists, else original."""
    # path like GPS.latitude.csv -> GPS.latitude.refined.csv
    if path is None:
        return path
    p = Path(path)
    if p.suffix.lower() != ".csv":
        return p
    refined = p.with_name(p.stem + REFINED_SUFFIX + p.suffix)
    return refined if refined.exists() else p


def discover_run_files(folder: os.PathLike | str) -> RunFiles:
    folder = Path(folder)
    return RunFiles(
        latitude=find_first_file(folder, ["GPS.latitude"]),
        longitude=find_first_file(folder, ["GPS.longitude"]),
        speed=find_first_file(folder, ["GPS.speed"]),
        satellites=find_first_file(folder, ["GPS.satellites"]),
        vibration1=find_first_file(folder, ["CH1", "ACCEL"]),
        vibration2=find_first_file(folder, ["CH2", "ACCEL"]),
    )


def refine_run_folder(
    folder: os.PathLike | str,
    gps_cfg: GPSRefineConfig = GPSRefineConfig(),
    spd_cfg: SpeedRefineConfig = SpeedRefineConfig(),
    vib_cfg: VibrationRefineConfig = VibrationRefineConfig(),
    overwrite: bool = False,
) -> Dict[str, object]:
    """Refine GPS + speed + vibration for a run folder. Writes refined files and returns metadata."""
    folder = Path(folder)
    print(f"=== Refining folder: {folder} ===")
    files = discover_run_files(folder)

    meta: Dict[str, object] = {"folder": str(folder)}

    # --- GPS ---
    if files.latitude and files.longitude:
        lat_in = files.latitude
        lon_in = files.longitude
        lat_out = lat_in.with_name(lat_in.stem + REFINED_SUFFIX + lat_in.suffix)
        lon_out = lon_in.with_name(lon_in.stem + REFINED_SUFFIX + lon_in.suffix)

        if overwrite or (not lat_out.exists()) or (not lon_out.exists()):
            lat = read_single_col_csv(lat_in)
            lon = read_single_col_csv(lon_in)
            sats = read_single_col_csv(files.satellites) if files.satellites else None
            lat_r, lon_r, gps_meta = refine_gps_series(lat, lon, sats=sats, cfg=gps_cfg)
            write_single_col_csv(lat_out, lat_r)
            write_single_col_csv(lon_out, lon_r)
            meta["gps"] = gps_meta
        else:
            meta["gps"] = {"cached": True}

        meta["gps_lat_refined"] = str(lat_out)
        meta["gps_lon_refined"] = str(lon_out)

    # --- SPEED ---
    if files.speed:
        spd_in = files.speed
        spd_out = spd_in.with_name(spd_in.stem + REFINED_SUFFIX + spd_in.suffix)
        if overwrite or (not spd_out.exists()):
            spd = read_single_col_csv(spd_in)
            spd_r, spd_meta = refine_speed_series(spd, cfg=spd_cfg)
            write_single_col_csv(spd_out, spd_r)

            # Save a normalized variant (0-1) to make plotting comparable
            spd_norm_out = spd_in.with_name(spd_in.stem + REFINED_SUFFIX + ".norm" + spd_in.suffix)
            s = spd_r.to_numpy()
            smin, smax = float(np.nanmin(s)), float(np.nanmax(s))
            sn = (s - smin) / (smax - smin + 1e-12)
            write_single_col_csv(spd_norm_out, sn)

            meta["speed"] = spd_meta
            meta["gps_speed_refined"] = str(spd_out)
            meta["gps_speed_norm"] = str(spd_norm_out)
        else:
            meta["speed"] = {"cached": True}
            meta["gps_speed_refined"] = str(spd_out)

    # --- VIBRATION ---
    if files.vibration1 and files.vibration2:
        v1_in = files.vibration1
        v2_in = files.vibration2

        v1_out = v1_in.with_name(v1_in.stem + REFINED_SUFFIX + v1_in.suffix)
        v2_out = v2_in.with_name(v2_in.stem + REFINED_SUFFIX + v2_in.suffix)
        feat_out = folder / ("vibration_segments" + REFINED_SUFFIX + ".features.csv")
        spikes_out = folder / ("vibration" + REFINED_SUFFIX + ".spikes.csv")
        meta_out = folder / ("refine" + REFINED_SUFFIX + ".meta.json")

        if overwrite or (not v1_out.exists()) or (not v2_out.exists()) or (not feat_out.exists()):
            v1 = read_single_col_csv(v1_in)
            v2 = read_single_col_csv(v2_in)
            refined_df, feat_df, spikes_df, vib_meta = refine_vibration(v1, v2, cfg=vib_cfg)

            # Save refined channels as single-column CSV (keeps your existing loaders compatible)
            write_single_col_csv(v1_out, refined_df["ch1"] if not refined_df.empty else pd.Series(dtype="float64"))
            write_single_col_csv(v2_out, refined_df["ch2"] if not refined_df.empty else pd.Series(dtype="float64"))

            # Save features + spikes
            if not feat_df.empty:
                feat_df.to_csv(feat_out, index=False)
            else:
                pd.DataFrame().to_csv(feat_out, index=False)

            if spikes_df is not None and not spikes_df.empty:
                spikes_df.to_csv(spikes_out, index=False)
            else:
                pd.DataFrame(columns=["sample", "time_s"]).to_csv(spikes_out, index=False)

            meta["vibration"] = vib_meta
            meta["vibration1_refined"] = str(v1_out)
            meta["vibration2_refined"] = str(v2_out)
            meta["vibration_features"] = str(feat_out)
            meta["vibration_spikes"] = str(spikes_out)

            # Write a folder-level meta file for debugging/repro
            try:
                with open(meta_out, "w", encoding="utf-8") as f:
                    json.dump(meta, f, ensure_ascii=False, indent=2)
            except Exception:
                pass
        else:
            meta["vibration"] = {"cached": True}
            meta["vibration1_refined"] = str(v1_out)
            meta["vibration2_refined"] = str(v2_out)
            meta["vibration_features"] = str(feat_out)
            meta["vibration_spikes"] = str(spikes_out)

    return meta
