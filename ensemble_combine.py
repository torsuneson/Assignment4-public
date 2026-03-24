import pandas as pd
import numpy as np
from pathlib import Path

GRID_METERS = 5  # 5 meters grid resolution

def latlon_to_grid(lat, lon, meters=GRID_METERS):
    # Approx conversion meters -> degrees (lat)
    lat_deg = meters / 111111  
    # Lon shrink with latitude
    lon_deg = meters / (111111 * np.cos(np.radians(lat)))
    glat = np.round(lat / lat_deg) * lat_deg
    glon = np.round(lon / lon_deg) * lon_deg
    return glat, glon

def combine_predictions(pred_files, out_path="predictions_combined.csv"):
    import numpy as np
    import pandas as pd

    print(f"[COMBINE DEBUG] Enter")
    dfs = []
    for pf in pred_files:
        df = pd.read_csv(pf)

        # RÄTT GPS-kontroll
        if "Latitude" not in df.columns or "Longitude" not in df.columns:
            print("[COMBINE] Skipping file without GPS:", pf)
            continue
        dfs.append(df)

    if not dfs:
        raise RuntimeError("No valid prediction files with GPS found.")

    df = pd.concat(dfs, ignore_index=True)

    print(f"[COMBINE DEBUG] laton_to_grid")
    # --- gridning (behåll din befintliga latlon_to_grid om du redan har den) ---
    def latlon_to_grid(lat, lon, meters=5):
        lat_deg = meters / 111_111.0
        lon_deg = meters / (111_111.0 * np.cos(np.radians(lat)))
        glat = np.round(lat / lat_deg) * lat_deg
        glon = np.round(lon / lon_deg) * lon_deg
        return glat, glon

    glat, glon = latlon_to_grid(df["Latitude"].to_numpy(), df["Longitude"].to_numpy())
    df["grid_lat"] = glat
    df["grid_lon"] = glon

    prob_cols = [c for c in df.columns if c.startswith("prob_")]
    groups = df.groupby(["grid_lat", "grid_lon"], as_index=False)

    #if prob_cols:
        # Soft-ensemble (medel av sannolikheter → argmax till pred)
        #res = groups[prob_cols].mean()
        #best = res[prob_cols].idxmax(axis=1)          # t.ex. "prob_RailJoint"
        #res["pred"] = best.str.replace("prob_", "", regex=False)
        #res["max_prob"] = res[prob_cols].max(axis=1)
        #res.loc[res["max_prob"] < CONF_THRESH_ENS, "pred"] = "Neutral"
        
        #PER_CLASS_T = {"Bridge": 0.60, "RailJoint": 0.65, "Turnout": 0.60, "Other": 0.55}
        #res["pred"] = [
        #    (cls if row[f"prob_{cls}"] >= PER_CLASS_T[cls] else "Neutral")
        #    for cls, (_, row) in zip(res["pred"], res.iterrows())
        #]

        
    #else:
        # Majority voting
        #res = groups["pred"].agg(lambda s: s.mode().iloc[0]).reset_index()

            # --- SOFT-ENSEMBLE (prob_* finns) ---
    prob_cols = [c for c in df.columns if c.startswith("prob_")]
    groups = df.groupby(["grid_lat", "grid_lon"], as_index=False)

    if prob_cols:
        # 1) Medelvärde per gridcell för alla prob_*
        res = groups[prob_cols].mean()

        # 2) Skapa bas-pred via argmax över prob_*
        res["pred"] = res[prob_cols].idxmax(axis=1).str.replace("prob_", "", regex=False)

        # 3) Tröskling: klass-specifik eller global
        #    a) KLASS-SPECIFIK (ändra siffror vid behov)
        PER_CLASS_T = {"Bridge": 0.80, "RailJoint": 0.90, "Turnout": 0.90, "Other": 0.95}

        #    b) Beräkna sannolikhet för vald klass (det är max-prob i soft-ensemble)
        res["pred_prob"] = res[prob_cols].max(axis=1)
        thr_s = res["pred"].map(PER_CLASS_T)  # tröskel per rad baserat på vald klass

        #    c) Sätt Neutral om under tröskel
        res.loc[res["pred_prob"] < thr_s, "pred"] = "Neutral"

        # (valfritt) rensa hjälpkolumn
        # res.drop(columns=["pred_prob"], inplace=True)

    else:
        # --- MAJORITY-ENSEMBLE (när prob_* saknas) ---
        # Behåll din befintliga majority-branch här (separerad från soft-ensemble)
        res = groups["pred"].agg(lambda s: s.mode().iloc[0]).reset_index()

    print(f"[COMBINE DEBUG] res names")
    
    
    # ---- Lägg till "verkligt" run/seg per gridcell (mode) ----
    # 1) Mode av 'run' per grid
    run_mode = (
        df.groupby(["grid_lat", "grid_lon"])["run"]
        .agg(lambda s: s.mode().iloc[0] if len(s.mode()) else str(s.iloc[0]))
        .reset_index(name="run")
    )

    # 2) Mode av 'seg' per grid ('seg' är int — casta säkert)
    seg_mode = (
        df.groupby(["grid_lat", "grid_lon"])["seg"]
        .agg(lambda s: int(s.mode().iloc[0]) if len(s.mode()) else int(s.iloc[0]))
        .reset_index(name="seg")
    )

    print(f"[COMBINE DEBUG] infört run seg")

    # 3) Slå ihop run/seg in i res (nyckel = grid_lat/grid_lon)
    res = res.merge(run_mode, on=["grid_lat", "grid_lon"], how="left")
    res = res.merge(seg_mode, on=["grid_lat", "grid_lon"], how="left")

    # ---- Döp om grid -> GPS och ordna kolumnordning ----
    res = res.rename(columns={"grid_lat": "Latitude", "grid_lon": "Longitude"})
    front = ["run", "seg", "Latitude", "Longitude", "pred"]
    other = [c for c in res.columns if c not in front]
    res = res[front + other]

    # ---- Skriv fil ----
    res.to_csv(out_path, index=False)
    print("[COMBINE] Saved:", out_path)
    return res
