"""
Phase 1: Extract and process NYC flood risk data per NTA neighborhood.
Outputs: data/nta_risk.geojson
"""

import os
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterstats import zonal_stats


def find_data_dir():
    p = os.path.expanduser("~/Documents")
    for root, _, _ in os.walk(p):
        if "NYCfloodModel" in root:
            return root
    raise FileNotFoundError("NYCfloodModel directory not found")


DATA_DIR = find_data_dir()
APP_DIR  = os.path.dirname(os.path.abspath(__file__))
OUT_PATH = os.path.join(APP_DIR, "data", "nta_risk.geojson")
os.makedirs(os.path.join(APP_DIR, "data"), exist_ok=True)

COMPOSITE_PATH = os.path.join(DATA_DIR, "composite_flood_risk_map.tif")
HAZARD_PATH    = os.path.join(DATA_DIR, "FloodHazardMapLR.tif")
GEO_PATH       = os.path.join(DATA_DIR, "nyc_neighborhoods.geojson")
POP_PATH       = os.path.join(DATA_DIR, "New_York_City_Population_By_Neighborhood_Tabulation_Areas.csv")
TOP_PATH       = os.path.join(DATA_DIR, "topNTAs.csv")


def normalize(series, clip_percentile=99):
    """Normalize to 0-1, preserving NaN (no-coverage) as NaN."""
    s = pd.to_numeric(series, errors="coerce")   # NaN stays NaN
    valid = s.dropna()
    lo = valid.min()
    hi = np.percentile(valid[valid > 0], clip_percentile) if (valid > 0).any() else 1.0
    hi = max(hi, lo + 1e-12)
    return (s - lo).clip(lower=0) / (hi - lo)   # NaN propagates through


def categorize(score):
    if pd.isna(score):
        return "Unknown"
    if score < 0.33:
        return "Low"
    if score <= 0.66:
        return "Medium"
    return "High"


def run_zonal(gdf_proj, raster_path, nodata_val):
    """Run zonal_stats and return mean values as a list."""
    stats = zonal_stats(
        gdf_proj,
        raster_path,
        stats=["mean", "max"],
        nodata=nodata_val,
        all_touched=False,
    )
    # Keep None as NaN — distinguishes "no coverage" from "genuinely zero"
    means = [s["mean"] for s in stats]   # None preserved
    maxes = [s["max"]  for s in stats]
    return means, maxes


# ── Step 1: Load neighborhoods ─────────────────────────────────────────────────
print("Loading neighborhoods…")
gdf = gpd.read_file(GEO_PATH)
print(f"  {len(gdf)} NTAs, CRS: {gdf.crs}")

# ── Step 2: Reproject to EPSG:2263 to match rasters ───────────────────────────
print("Reprojecting to EPSG:2263…")
gdf_proj = gdf.to_crs("EPSG:2263")

# ── Step 3: Hazard layer — FloodHazardMapLR.tif ────────────────────────────────
# Full coverage; nodata=None keeps all pixels (none are truly nodata).
print("Running zonal_stats for hazard layer…")
with rasterio.open(HAZARD_PATH) as src:
    hazard_nodata = src.nodata
hazard_means, hazard_maxes = run_zonal(gdf_proj, HAZARD_PATH, hazard_nodata)

gdf["hazard_mean_raw"] = hazard_means
gdf["hazard_score"]    = normalize(pd.Series(hazard_means))
gdf["hazard_category"] = gdf["hazard_score"].apply(categorize)

# ── Step 4: Composite layer — composite_flood_risk_map.tif ─────────────────────
# Sparse: zeros mean "outside flood zone". Use nodata=0 so mean is over flood pixels only.
print("Running zonal_stats for composite risk layer…")
composite_means, composite_maxes = run_zonal(gdf_proj, COMPOSITE_PATH, nodata_val=0)

gdf["composite_mean_raw"] = composite_means
# NTAs with no flood-zone pixels genuinely have 0 composite risk
gdf["composite_score"]    = normalize(pd.Series(composite_means))
gdf["composite_category"] = gdf["composite_score"].apply(categorize)

# ── Step 5: Population merge ───────────────────────────────────────────────────
print("Merging population…")
pop = pd.read_csv(POP_PATH)
pop_latest = (
    pop.sort_values("Year")
       .groupby("NTA Code", as_index=False)
       .last()[["NTA Code", "NTA Name", "Population", "Borough"]]
)
gdf = gdf.merge(
    pop_latest,
    left_on="ntacode",
    right_on="NTA Code",
    how="left",
)

# ── Step 6: topNTAs merge ─────────────────────────────────────────────────────
top = pd.read_csv(TOP_PATH)[["ntacode", "composite_flood_risk"]].rename(
    columns={"composite_flood_risk": "raw_composite_score"}
)
gdf = gdf.merge(top, on="ntacode", how="left")

# ── Step 7: Borough ────────────────────────────────────────────────────────────
if "Borough" not in gdf.columns or gdf["Borough"].isna().all():
    gdf["Borough"] = gdf.get("boro_name", "Unknown")
else:
    gdf["Borough"] = gdf["Borough"].fillna(gdf.get("boro_name", "Unknown"))

# ── Step 8: Ranks for both layers ─────────────────────────────────────────────
gdf["hazard_rank"]   = gdf["hazard_score"].rank(ascending=False, method="min").astype("Int64")
gdf["composite_rank"]= gdf["composite_score"].rank(ascending=False, method="min").astype("Int64")
gdf["total_ntas"]    = len(gdf)

# ── Step 9: Save ─────────────────────────────────────────────────────────────
keep = [
    "ntacode", "ntaname", "boro_name", "Borough",
    "NTA Name", "Population",
    "hazard_mean_raw", "hazard_score", "hazard_category", "hazard_rank",
    "composite_mean_raw", "composite_score", "composite_category", "composite_rank",
    "raw_composite_score", "total_ntas",
    "geometry",
]
save_cols = [c for c in keep if c in gdf.columns]
gdf[save_cols].to_file(OUT_PATH, driver="GeoJSON")
print(f"\nSaved → {OUT_PATH}")

# ── Report ────────────────────────────────────────────────────────────────────
out = gdf[save_cols]
print(f"Row count: {len(out)}")
print(f"\nHazard score distribution:")
hs = out["hazard_score"]
print(f"  mean={hs.mean():.3f}  median={hs.median():.3f}  max={hs.max():.3f}")
print(f"  Low={( hs<0.33).sum()}  Med={(hs.between(0.33,0.66)).sum()}  High={(hs>0.66).sum()}")

print(f"\nComposite score distribution:")
cs = out["composite_score"]
print(f"  mean={cs.mean():.3f}  median={cs.median():.3f}  max={cs.max():.3f}")
print(f"  Low={( cs<0.33).sum()}  Med={(cs.between(0.33,0.66)).sum()}  High={(cs>0.66).sum()}")

print("\nTop 5 by Hazard score:")
disp = ["ntaname", "boro_name", "hazard_score", "hazard_category", "composite_score", "composite_category"]
print(out.nlargest(5, "hazard_score")[[c for c in disp if c in out.columns]].to_string(index=False))
