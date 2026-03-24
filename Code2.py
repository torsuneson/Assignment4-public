import os
import json
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

import dash
from dash import dcc, html, Input, Output

import tkinter as tk
from tkinter import filedialog

# extra 
import overpass


# ============================================================
# MINIMAL EXTENSION (Batch + Manifest) for Code 2
# ------------------------------------------------------------
# Goal: Reuse Code 2 as much as possible.
# - The original pipeline is kept: load CSVs (single col + timestamp=index),
#   merge GPS, merge vibration, segment vibration, build Plotly figures, Dash callback.
# - Added: optional manifest-driven batch mode to avoid manual file picking.
# ============================================================

# --------------------
# User-configurable flags (Jupyter-friendly)
# --------------------
USE_MANIFEST = True
MANIFEST_PATH = "selected_runs.json"   # created by select_runs.py / notebook cell
EXPORT_DIR = "Code2_exports"           # where outputs are saved
BATCH_MODE = True                       # if True and manifest exists -> loop all selected runs
EXPORT_ONLY = True                      # if True in batch mode -> do NOT start Dash server

# What vibration snapshots to export per run
EXPORT_VIB_SELECTION = ("max_rms", "mid")  # options: 'first', 'mid', 'max_rms'

# --------------------
# Original constants from Code 2
# --------------------
dt_vibration = 0.002  # seconds per sample (e.g. 500 Hz sampling rate)
segment_duration_seconds = 10
segment_length = int(segment_duration_seconds / dt_vibration)


# ============================================================
# Original file selection dictionary
# ============================================================
files = {
    "latitude": None,
    "longitude": None,
    "vibration1": None,
    "vibration2": None,
    "speed": None
}


# ============================================================
# Helpers (small) - keep Code2 behavior
# ============================================================

def load_file(key):
    """Original Tkinter-based file picker."""
    file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
    if file_path:
        files[key] = file_path
        print(f"{key.capitalize()} file loaded: {file_path}")


def load_csv_as_code2(file_path, key):
    """Load each CSV into a DataFrame and add a 'timestamp' using the row index (Code2 style)."""
    df = pd.read_csv(file_path, header=None, names=[key])
    df['timestamp'] = df.index
    return df


def segment_vibration_code2(df_vibration_merged):
    """Data Preprocessing and Segmentation for Vibration Data (Code2 style)."""
    if df_vibration_merged.empty:
        return np.array([])

    num_segments = len(df_vibration_merged) // segment_length
    segs = []
    for i in range(num_segments):
        seg = df_vibration_merged.iloc[i * segment_length: (i + 1) * segment_length][["vibration1", "vibration2"]].values
        segs.append(seg)
    segs = np.array(segs)
    return segs


def pick_segment_index(segments, mode):
    if segments.size == 0:
        return 0
    nseg = segments.shape[0]
    if mode == "first":
        return 0
    if mode == "mid":
        return nseg // 2
    if mode == "max_rms":
        rms = np.sqrt(np.mean(segments ** 2, axis=1))  # (nseg, 2)
        score = rms[:, 0] + rms[:, 1]
        return int(np.argmax(score))
    return 0


def safe_write_png(fig, out_png, width=1200, height=800):
    """Try PNG export (requires kaleido). If it fails, keep HTML as fallback."""
    try:
        fig.write_image(out_png, width=width, height=height)
        return True
    except Exception as e:
        print(f"PNG export failed for {out_png}: {e}")
        return False


def auto_center_zoom(df_gps, pad=0.08):

    # Filter out 0.0 points in gps data 
    d = df_gps.dropna(subset=["Latitude", "Longitude"]).copy()
    df_gps = d[
            (d["Latitude"].between(-90, 90)) &
            (d["Longitude"].between(-180, 180)) &
            ~((d["Latitude"] == 0) & (d["Longitude"] == 0))
        ]
    

    lat_min, lat_max = df_gps["Latitude"].min(), df_gps["Latitude"].max()
    lon_min, lon_max = df_gps["Longitude"].min(), df_gps["Longitude"].max()

    # padding så punkter inte hamnar i kanten
    lat_pad = (lat_max - lat_min) * pad if lat_max > lat_min else 0.01
    lon_pad = (lon_max - lon_min) * pad if lon_max > lon_min else 0.01

    lat_min -= lat_pad; lat_max += lat_pad
    lon_min -= lon_pad; lon_max += lon_pad

    center = {"lat": (lat_min + lat_max) / 2, "lon": (lon_min + lon_max) / 2}

    # grov zoom baserad på spann (fungerar bra i praktiken)
    span = max(lat_max - lat_min, lon_max - lon_min)
    if span > 5:   zoom = 5
    elif span > 2: zoom = 6
    elif span > 1: zoom = 7
    elif span > 0.5: zoom = 8
    elif span > 0.25: zoom = 9
    elif span > 0.12: zoom = 10
    elif span > 0.06: zoom = 11
    else: zoom = 12

    return center, zoom


def fetch_railway_geojson_bbox(min_lon, min_lat, max_lon, max_lat, timeout=360):
    """
    Hämtar OSM railway-geometrier (ways + relations) inom bbox och returnerar GeoJSON FeatureCollection.
    bbox-ordning i Overpass QL: (south_lat, west_lon, north_lat, east_lon). [5](https://dev.overpass-api.de/overpass-doc/en/full_data/bbox.html)
    """
    api = overpass.API(timeout=timeout)

    # Filtrera vilka railway-typer du vill ha (justera vid behov)
    # Exkluderar t.ex. abandoned/construction om du vill
    query = f"""
    (
      way["railway"]({min_lat},{min_lon},{max_lat},{max_lon});
      relation["railway"]({min_lat},{min_lon},{max_lat},{max_lon});
    );
    (._;>;);
    """

    # responseformat="geojson" ger FeatureCollection direkt. [1](https://pypi.org/project/overpass/)[2](https://gis.stackexchange.com/questions/314549/saving-overpass-query-results-to-geojson-file-with-python)
    geojson = api.get(query, responseformat="geojson")
    return geojson

# Exempel:
# rail_geojson = fetch_railway_geojson_bbox(14.55, 61.00, 14.58, 61.02)

# ============================================================
# Core pipeline wrapped as a function (same code, minimal move)
# ============================================================

def run_pipeline(current_files, run_name="run"):
    """Run the original Code2 pipeline for a given set of file paths.

    Returns:
        df_gps, segments, map_fig
    """

    # --------------------
    # Load each CSV into a DataFrame and add a 'timestamp' using the row index.
    # --------------------
    dataframes = {}
    for key, file_path in current_files.items():
        if file_path:
            df = load_csv_as_code2(file_path, key)
            dataframes[key] = df
        else:
            print(f"{key.capitalize()} file not selected for {run_name}.")

    # --------------------
    # Create GPS DataFrame by merging latitude and longitude.
    # --------------------
    if "latitude" in dataframes and "longitude" in dataframes:
        df_gps = pd.merge(dataframes["latitude"], dataframes["longitude"], on="timestamp")
        df_gps = df_gps.rename(columns={"latitude": "Latitude", "longitude": "Longitude"})
        df_gps["PointIndex"] = df_gps.index
    else:
        print("Latitude or Longitude data is missing.")
        df_gps = pd.DataFrame(columns=["Latitude", "Longitude", "PointIndex"])

    # --------------------
    # Merge the two vibration signals on 'timestamp'
    # --------------------
    if "vibration1" in dataframes and "vibration2" in dataframes:
        df_vibration_merged = pd.merge(
            dataframes["vibration1"],
            dataframes["vibration2"],
            on="timestamp"
        )
    else:
        print("Vibration data files are missing.")
        df_vibration_merged = pd.DataFrame()

    # --------------------
    # Segmentation (Code2)
    # --------------------
    segments = segment_vibration_code2(df_vibration_merged)
    if segments.size:
        print(f"[{run_name}] Segmented vibration data shape: {segments.shape}")
    else:
        print(f"[{run_name}] No vibration data available for segmentation.")

    # --------------------
    # Build the Interactive GPS map using Plotly Express. (Code2)
    # --------------------
    if not df_gps.empty:
        center, zoom = auto_center_zoom(df_gps)
        #rail_geojson = fetch_railway_geojson_bbox(
        #    df_gps["Longitude"].min(), 
        #    df_gps["Latitude"].min(), 
        #    df_gps["Longitude"].max(), 
        #    df_gps["Latitude"].max()
        #    )

### moved into auto_center_zoom insted       
        ##print("BEFORE:", run_name,
        ##    "LAT:", df_gps["Latitude"].min(), df_gps["Latitude"].median(), df_gps["Latitude"].max(),
        ##    "LON:", df_gps["Longitude"].min(), df_gps["Longitude"].median(), df_gps["Longitude"].max())

        # Remove invalid GPS points (common placeholder values)
        #df_gps = df_gps.dropna(subset=["Latitude", "Longitude"])

        # Filter out zeros and out-of-range values
        #df_gps = df_gps[
        #    (df_gps["Latitude"].between(-90, 90)) &
        #    (df_gps["Longitude"].between(-180, 180)) &
        #    ~((df_gps["Latitude"] == 0) & (df_gps["Longitude"] == 0))
        #].copy()

        #print("AFTER: ", run_name,
        #    "LAT:", df_gps["Latitude"].min(), df_gps["Latitude"].median(), df_gps["Latitude"].max(),
        #    "LON:", df_gps["Longitude"].min(), df_gps["Longitude"].median(), df_gps["Longitude"].max())


        map_fig = px.scatter_map(
            df_gps,
            lat="Latitude", lon="Longitude",
            custom_data=["PointIndex"],
            hover_data=["PointIndex"],
            title=f"GPS Points with Vibration Data ({run_name})",
            center=center,
            zoom=zoom,
            map_style="open-street-map"
        )

        map_fig.update_layout(
            margin=dict(l=0, r=0, t=40, b=0)
        )

        """map_fig.update_layout(
        map=dict(
            style="open-street-map",
            layers=[
                dict(
                    sourcetype="geojson",
                    source=rail_geojson,
                    type="line",
                    color="black",
                    line=dict(width=2),
                    opacity=0.8,
                    )
                ]
            )
        )"""

        #Debug
        #print(run_name,
        #    df_gps["Latitude"].min(), df_gps["Latitude"].max(),
        #    df_gps["Longitude"].min(), df_gps["Longitude"].max())

    else:
        map_fig = go.Figure()
        map_fig.update_layout(title=f"No GPS Data Available ({run_name})", height=600)

    return df_gps, segments, map_fig


# ============================================================
# Export helpers (map + selected vibration snapshots)
# ============================================================

def export_outputs(run_name, df_gps, segments, map_fig):
    os.makedirs(EXPORT_DIR, exist_ok=True)
    out_dir = os.path.join(EXPORT_DIR, run_name)
    os.makedirs(out_dir, exist_ok=True)

    # Save map
    map_html = os.path.join(out_dir, "map.html")
    map_fig.write_html(map_html)
    safe_write_png(map_fig, os.path.join(out_dir, "map.png"))

    # Create and save vibration snapshots using the SAME plotting logic as callback
    time_axis = np.arange(segment_length) * dt_vibration

    def make_vib_fig_for_point(point_index):
        # same logic as original callback: fallback to last segment
        if segments.size == 0:
            fig = go.Figure()
            fig.update_layout(
                title="No Vibration Data Available",
                xaxis_title="Time (s)",
                yaxis_title="Acceleration"
            )
            return fig

        if point_index < segments.shape[0]:
            selected_segment = segments[point_index]
        else:
            selected_segment = segments[-1]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=time_axis, y=selected_segment[:, 0], mode='lines', name='Vibration Channel 1'))
        fig.add_trace(go.Scatter(x=time_axis, y=selected_segment[:, 1], mode='lines', name='Vibration Channel 2'))
        fig.update_layout(
            title=f"Vibration Signal for GPS Point {point_index} ({run_name})",
            xaxis_title="Time (s)",
            yaxis_title="Acceleration"
        )
        return fig

    for mode in EXPORT_VIB_SELECTION:
        idx = pick_segment_index(segments, mode)
        vib_fig = make_vib_fig_for_point(idx)
        vib_fig.write_html(os.path.join(out_dir, f"vibration_{mode}.html"))
        safe_write_png(vib_fig, os.path.join(out_dir, f"vibration_{mode}.png"))

    # Save summary
    summary = {
        "run": run_name,
        "gps_points": int(len(df_gps)),
        "segments": int(segments.shape[0]) if segments.size else 0,
        "segment_length": int(segment_length),
        "dt_vibration": float(dt_vibration),
        "files": current_files
    }
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


# ============================================================
# Mode selection
# ============================================================

# Hide the root Tk window (original Code2 behavior)
root = tk.Tk()
root.withdraw()

manifest = None
if USE_MANIFEST and os.path.exists(MANIFEST_PATH):
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

if manifest and BATCH_MODE:
    # --------------------
    # BATCH: loop over all selected runs in the manifest
    # --------------------
    os.makedirs(EXPORT_DIR, exist_ok=True)
    selected = manifest.get("selected", [])
    print(f"Manifest loaded: {MANIFEST_PATH}. Runs selected: {len(selected)}")

    for entry in selected:
        run_name = entry.get("run", "run")
        current_files = entry.get("files", {})

        # Ensure keys exist as Code2 expects
        current_files = {
            "latitude": current_files.get("latitude"),
            "longitude": current_files.get("longitude"),
            "vibration1": current_files.get("vibration1"),
            "vibration2": current_files.get("vibration2"),
            "speed": current_files.get("speed"),
        }

        print(f"\n=== Processing {run_name} ===")
        df_gps, segments, map_fig = run_pipeline(current_files, run_name=run_name)
        export_outputs(run_name, df_gps, segments, map_fig)

    print(f"\nBatch export finished. Output folder: {EXPORT_DIR}")

else:
    # --------------------
    # INTERACTIVE: original Tkinter selection + Dash app
    # --------------------
    print("Select Latitude File")
    load_file("latitude")
    print("Select Longitude File")
    load_file("longitude")
    print("Select Vibration 1 File")
    load_file("vibration1")
    print("Select Vibration 2 File")
    load_file("vibration2")
    print("Select Speed File")
    load_file("speed")

    current_files = dict(files)

    df_gps, segments, map_fig = run_pipeline(current_files, run_name="interactive")

    vib_empty_fig = go.Figure()
    vib_empty_fig.update_layout(
        title="Vibration Signal",
        xaxis_title="Time (s)",
        yaxis_title="Acceleration"
    )

    app = dash.Dash(__name__)
    app.layout = html.Div([
        html.Div([
            dcc.Graph(id="gps-map", figure=map_fig)
        ], style={'width': '48%', 'display': 'inline-block', 'vertical-align': 'top'}),
        html.Div([
            dcc.Graph(id="vibration-plot", figure=vib_empty_fig)
        ], style={'width': '48%', 'display': 'inline-block', 'vertical-align': 'top'})
    ])

    @app.callback(
        Output('vibration-plot', 'figure'),
        [Input('gps-map', 'clickData')]
    )
    def update_vibration_plot(clickData):
        if clickData is None:
            return vib_empty_fig

        point_index = clickData['points'][0]['customdata'][0]

        if segments.size == 0:
            empty_fig = go.Figure()
            empty_fig.update_layout(
                title="No Vibration Data Available",
                xaxis_title="Time (s)",
                yaxis_title="Acceleration"
            )
            return empty_fig

        if point_index < segments.shape[0]:
            selected_segment = segments[point_index]
        else:
            selected_segment = segments[-1]

        time_axis = np.arange(segment_length) * dt_vibration
        vib_fig = go.Figure()
        vib_fig.add_trace(go.Scatter(
            x=time_axis,
            y=selected_segment[:, 0],
            mode='lines',
            name='Vibration Channel 1'
        ))
        vib_fig.add_trace(go.Scatter(
            x=time_axis,
            y=selected_segment[:, 1],
            mode='lines',
            name='Vibration Channel 2'
        ))
        vib_fig.update_layout(
            title=f"Vibration Signal for GPS Point {point_index}",
            xaxis_title="Time (s)",
            yaxis_title="Acceleration"
        )
        return vib_fig

    if not EXPORT_ONLY:
        app.run_server(debug=True, port=8060)
    else:
        print("EXPORT_ONLY=True -> Dash server not started.")
