# Plot_as_Code1.py
#
# Karta på GPS-koordinater med predikterade klasser (alla körningar ihop).
# 1) Läser predictions_combined.csv (måste ha: run, seg, pred)
# 2) Läser segments_inference.csv (måste ha: run, seg, Latitude, Longitude)
# 3) Inner join på (run, seg) för att få pred + lat/lon
# 4) Plottar EN karta (alla körningar sammanlagt), sparar HTML + PNG
#
# Tips: pip install -U plotly kaleido

from pathlib import Path
import pandas as pd
import plotly.express as px

# ---- In/Ut ----
PRED_CSV = "predictions_combined.csv"       
INFER_CSV  = "segments_inference.csv"        
OUT_DIR    = Path("plots_map")
HTML_OUT   = OUT_DIR / "map_all.html"
PNG_OUT    = OUT_DIR / "map_all.png"

# ---- Plot inställningar ----
MAP_STYLE = "open-street-map"   # kräver ingen token
FIG_W, FIG_H = 1200, 800

# Konsekventa klassfärger
CLASSES = ["Bridge", "RailJoint", "Turnout", "Other", "Neutral"]
CLASS_COLORS = {
    "Bridge":    "#1f77b4",  # blå
    "RailJoint": "#ff7f0e",  # orange
    "Turnout":   "#2ca02c",  # grön
    "Other":     "#d62728",  # röd
    "Neutral":   "#9e9e9e",  # grå
}


def load_predictions(pred_path: str) -> pd.DataFrame:
    dfp = pd.read_csv(pred_path)
    need_p = {"run", "seg", "pred"}
    miss_p = [c for c in need_p if c not in dfp.columns]
    if miss_p:
        raise ValueError(f"{pred_path} saknar kolumner: {miss_p}")
    dfp["run"] = dfp["run"].astype(str)
    dfp["seg"] = pd.to_numeric(dfp["seg"], errors="coerce")
    # säkra kända klassnamn
    dfp["pred"] = dfp["pred"].where(dfp["pred"].isin(CLASSES), "Other")
    return dfp

def load_inference(inf_path: str) -> pd.DataFrame:
    dfi = pd.read_csv(inf_path)
    need_i = {"run", "seg", "Latitude", "Longitude"}
    miss_i = [c for c in need_i if c not in dfi.columns]
    if miss_i:
        raise ValueError(f"{inf_path} saknar kolumner: {miss_i}")
    dfi["run"] = dfi["run"].astype(str)
    dfi["seg"] = pd.to_numeric(dfi["seg"], errors="coerce")
    # rensa orimliga koordinater
    dfi = dfi.dropna(subset=["Latitude", "Longitude"])
    dfi = dfi[dfi["Latitude"].between(-90, 90) & dfi["Longitude"].between(-180, 180)]
    # behåll bara nycklar + lat/lon (undvik onödig bloat)
    return dfi[["run", "seg", "Latitude", "Longitude"]]

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1) Ladda predictions
    df_pred = load_predictions(PRED_CSV)

    # 2) Har predictions redan GPS? Använd direkt, annars joina med inference.
    if {"Latitude", "Longitude"}.issubset(df_pred.columns):
        # Inget behov av join – prediction-filen har GPS
        df = df_pred.copy()
    else:
        # Hämta lat/lon från inference och joina på run/seg
        df_inf = load_inference(INFER_CSV)
        df = df_pred.merge(df_inf, on=["run", "seg"], how="inner")
        if df.empty:
            raise ValueError(
                "Join gav inga rader. Kontrollera att 'run' och 'seg' matchar mellan "
                f"{PRED_CSV} och {INFER_CSV}, samt att inference-filen har lat/lon."
            )

    # 3) Centra karta kring median
    lat_c = float(df["Latitude"].median())
    lon_c = float(df["Longitude"].median())

    # 4) Plot – EN karta för ALLA körningar
    fig = px.scatter_mapbox(
        df,
        lat="Latitude",
        lon="Longitude",
        color="pred",
        category_orders={"pred": CLASSES},
        color_discrete_map=CLASS_COLORS,
        hover_data={
            "run": True, "seg": True, "pred": True,
            "Latitude": ":.6f", "Longitude": ":.6f"
        },
        zoom=9,
        height=FIG_H,
        width=FIG_W,
        title="'Predicted  Classes, Combined runs from selected quick manifest'",
    )


    fig.update_traces(marker=dict(size=8, opacity=0.90))
    # Overwrite Neutral
    for i, tr in enumerate(fig.data):
        if tr.name == "Neutral":
            fig.data[i].marker.update(size=4, opacity=0.25)



    fig.update_layout(
        mapbox_style=MAP_STYLE,
        mapbox_center={"lat": lat_c, "lon": lon_c},
        margin={"l": 10, "r": 10, "t": 50, "b": 10},
        legend_title_text="Predikterad klass",
    )

    # 5) Spara HTML + PNG
    fig.write_html(str(HTML_OUT))  # interaktiv karta med riktiga tiles

    try:
        # Försök först med vald kartstil (t.ex. open-street-map)
        fig.write_image(str(PNG_OUT), width=FIG_W, height=FIG_H)
    except Exception as e:
        # Fallback: PNG utan tilehämtning (white-bg)
        print("[WARN] PNG via Kaleido misslyckades, provar white-bg. Detaljer:", e)
        fig_no_tiles = fig
        fig_no_tiles.update_layout(mapbox_style="white-bg")  # tile-fri bakgrund
        fig_no_tiles.write_image(str(PNG_OUT), width=FIG_W, height=FIG_H)

    print("[SAVED]", HTML_OUT)
    print("[SAVED]", PNG_OUT)

if __name__ == "__main__":
    main()