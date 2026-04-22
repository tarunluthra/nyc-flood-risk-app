"""
Fetch NYC infrastructure point data from NYC Open Data.
Saves to data/infrastructure.csv  (run once before starting the app).
"""

import os
import json
import urllib.request
import urllib.parse
import pandas as pd

APP_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_PATH = os.path.join(APP_DIR, "data", "infrastructure.csv")
ENDPOINT = "https://data.cityofnewyork.us/resource"
FACILITIES = "ji82-xba5.json"


def fetch_socrata(resource, where=None, select=None, limit=1000, label=""):
    params = {"$limit": limit}
    if where:
        params["$where"] = where
    if select:
        params["$select"] = select
    url = f"{ENDPOINT}/{resource}?{urllib.parse.urlencode(params)}"
    print(f"  {label}…")
    with urllib.request.urlopen(url, timeout=20) as r:
        return json.loads(r.read())


# ── 1. FDNY Firehouses ─────────────────────────────────────────────────────────
fire_rows = fetch_socrata(
    "hc8x-tcnd.json",
    select="facilityname,latitude,longitude,borough",
    limit=500,
    label="FDNY firehouses",
)
fire_df = pd.DataFrame(fire_rows).rename(
    columns={"facilityname": "name", "borough": "boro"}
)
fire_df["category"] = "Fire Services"

# ── 2. Hospitals & Clinics ────────────────────────────────────────────────────
hosp_types = [
    "ACUTE CARE HOSPITAL",
    "HOSPITAL",
    "DIAGNOSTIC AND TREATMENT CENTER",
    "OUTPATIENT CLINIC",
    "CHILD HEALTH CENTER",
]
hosp_in = ", ".join(f"'{t}'" for t in hosp_types)
hosp_rows = fetch_socrata(
    FACILITIES,
    where=f"facgroup='HEALTH CARE' AND factype IN({hosp_in})",
    select="facname,latitude,longitude,boro",
    limit=1000,
    label="Hospitals & clinics",
)
hosp_df = pd.DataFrame(hosp_rows).rename(columns={"facname": "name"})
hosp_df["category"] = "Hospitals & Clinics"

# ── 3. Other Emergency Services (EMS / ambulance) ────────────────────────────
ems_rows = fetch_socrata(
    FACILITIES,
    where="facgroup='EMERGENCY SERVICES'",
    select="facname,factype,latitude,longitude,boro",
    limit=500,
    label="Emergency services",
)
ems_df = pd.DataFrame(ems_rows).rename(columns={"facname": "name"})
ems_df["category"] = "Other Emergency Services"

# ── 4. Bus Depots & Terminals ─────────────────────────────────────────────────
bus_rows = fetch_socrata(
    FACILITIES,
    where=(
        "facgroup='TRANSPORTATION' AND "
        "(factype LIKE '%BUS%' OR factype LIKE '%DEPOT%' OR factype LIKE '%TERMINAL%')"
    ),
    select="facname,factype,latitude,longitude,boro",
    limit=500,
    label="Bus depots & terminals",
)
bus_df = pd.DataFrame(bus_rows).rename(columns={"facname": "name"}) if bus_rows else pd.DataFrame()
if not bus_df.empty:
    bus_df["category"] = "Bus Depots & Terminals"

# ── Combine & clean ───────────────────────────────────────────────────────────
frames = [fire_df, hosp_df, ems_df]
if not bus_df.empty:
    frames.append(bus_df)

infra = pd.concat(frames, ignore_index=True)
infra["latitude"]  = pd.to_numeric(infra["latitude"],  errors="coerce")
infra["longitude"] = pd.to_numeric(infra["longitude"], errors="coerce")
infra = infra.dropna(subset=["latitude", "longitude"])

# Clip to NYC bounding box
infra = infra[
    infra["latitude"].between(40.45, 40.95)
    & infra["longitude"].between(-74.30, -73.65)
]

infra.to_csv(OUT_PATH, index=False)
print(f"\nSaved {len(infra)} facilities → {OUT_PATH}")
print(infra["category"].value_counts().to_string())
