"""NYC Flood Risk Explorer — Streamlit app."""

import os
import json
import numpy as np
import pandas as pd
import geopandas as gpd
import folium
import plotly.graph_objects as go
import streamlit as st
from streamlit_folium import st_folium

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NYC Flood Risk Explorer",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "nta_risk.geojson")

LAYERS = {
    "Composite Risk (Hazard + Vulnerability)": {
        "score_col":    "composite_score",
        "category_col": "composite_category",
        "rank_col":     "composite_rank",
        "label":        "Composite Risk Score",
        "description":  "Fusion of flood hazard extent and social/infrastructure vulnerability.",
    },
    "Flood Hazard Only": {
        "score_col":    "hazard_score",
        "category_col": "hazard_category",
        "rank_col":     "hazard_rank",
        "label":        "Flood Hazard Score",
        "description":  "Raw flood hazard based on FEMA zones, elevation, and inundation depth.",
    },
}

RISK_COLORS = {
    "Low":     "#2ecc71",
    "Medium":  "#f39c12",
    "High":    "#e74c3c",
    "Unknown": "#666666",
}


@st.cache_data
def load_data():
    gdf = gpd.read_file(DATA_PATH)
    if gdf.crs and gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(4326)
    gdf["display_name"] = gdf["ntaname"].fillna(gdf.get("NTA Name", gdf["ntacode"]))
    for col in ("hazard_score", "composite_score"):
        gdf[col] = pd.to_numeric(gdf[col], errors="coerce").clip(0, 1)
    gdf["Population"] = pd.to_numeric(gdf["Population"], errors="coerce")
    gdf["total_ntas"] = int(gdf["total_ntas"].iloc[0]) if "total_ntas" in gdf.columns else 195
    return gdf


def score_to_color(score):
    """Linear interpolation: green (0) → amber (0.5) → red (1)."""
    if pd.isna(score):
        return "#444444"
    s = float(np.clip(score, 0, 1))
    r = int(np.interp(s, [0, 0.5, 1.0], [46,  243, 231]))
    g = int(np.interp(s, [0, 0.5, 1.0], [204, 156,  76]))
    b = int(np.interp(s, [0, 0.5, 1.0], [113,  18,  60]))
    return f"#{r:02x}{g:02x}{b:02x}"


def build_map(gdf, score_col, category_col, selected_ntacode):
    m = folium.Map(location=[40.7128, -74.0060], zoom_start=11, tiles="CartoDB dark_matter")

    # Pre-compute style into GeoJSON properties so no Python closures needed
    gdf_map = gdf.copy()
    gdf_map["_fill"]   = gdf_map[score_col].apply(score_to_color)
    gdf_map["_stroke"] = "#1a1a1a"
    gdf_map["_weight"] = 1.0
    gdf_map["_fop"]    = 0.80

    # Selected NTA overrides
    mask = gdf_map["ntacode"] == selected_ntacode
    gdf_map.loc[mask, "_fill"]   = "#3498db"
    gdf_map.loc[mask, "_stroke"] = "#ffffff"
    gdf_map.loc[mask, "_weight"] = 3.0
    gdf_map.loc[mask, "_fop"]    = 0.92

    geojson_data = json.loads(gdf_map.to_json())

    folium.GeoJson(
        geojson_data,
        style_function=lambda x: {
            "fillColor":   x["properties"]["_fill"],
            "color":       x["properties"]["_stroke"],
            "weight":      x["properties"]["_weight"],
            "fillOpacity": x["properties"]["_fop"],
        },
        highlight_function=lambda x: {
            "weight": 3,
            "color":  "#ffffff",
            "fillOpacity": 0.95,
        },
        tooltip=folium.GeoJsonTooltip(
            fields=["display_name", "boro_name", category_col, score_col],
            aliases=["Neighborhood:", "Borough:", "Risk Level:", "Score (0-1):"],
            localize=True,
            sticky=False,
            style=(
                "background-color:#1e1e2e;color:#e0e0e0;"
                "font-size:13px;border:1px solid #555;border-radius:4px;padding:6px;"
            ),
        ),
    ).add_to(m)

    # Zoom to selected NTA
    row = gdf[gdf["ntacode"] == selected_ntacode]
    if not row.empty:
        b = row.total_bounds
        m.fit_bounds([[b[1], b[0]], [b[3], b[2]]])

    return m


def gauge_chart(score, label):
    val = 0.0 if pd.isna(score) else float(np.clip(score, 0, 1))
    color = score_to_color(val)
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(val * 100, 1),
        number={"suffix": "%", "font": {"size": 30, "color": "#eee"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#666", "tickfont": {"color": "#aaa"}},
            "bar": {"color": color, "thickness": 0.25},
            "bgcolor": "#1e1e2e",
            "bordercolor": "#333",
            "steps": [
                {"range": [0,  33], "color": "#0d2b0d"},
                {"range": [33, 66], "color": "#2b1f0d"},
                {"range": [66, 100], "color": "#2b0d0d"},
            ],
            "threshold": {
                "line": {"color": "#fff", "width": 2},
                "thickness": 0.85,
                "value": round(val * 100, 1),
            },
        },
        title={"text": label, "font": {"color": "#999", "size": 13}},
    ))
    fig.update_layout(
        height=210,
        margin=dict(l=15, r=15, t=40, b=5),
        paper_bgcolor="#0e1117",
        font_color="#eee",
    )
    return fig


def detail_card(row, layer_cfg, alt_layer_cfg, total_ntas):
    name     = row.get("display_name", "—")
    borough  = row.get("boro_name") or row.get("Borough", "—")
    score    = row.get(layer_cfg["score_col"])
    category = row.get(layer_cfg["category_col"], "Unknown")
    alt_score    = row.get(alt_layer_cfg["score_col"])
    alt_category = row.get(alt_layer_cfg["category_col"], "Unknown")
    pop      = row.get("Population")
    rank     = row.get(layer_cfg["rank_col"])
    color    = RISK_COLORS.get(category, "#666")

    st.markdown(f"## {name}")
    st.markdown(f"**Borough:** {borough}")
    st.divider()

    score_missing = pd.isna(score) or score is None

    col_a, col_b = st.columns(2)
    with col_a:
        score_str = f"{float(score):.3f}" if not score_missing else "—"
        st.metric(layer_cfg["label"], score_str)
        if score_missing:
            st.markdown(
                '<span style="display:inline-block;padding:3px 12px;border-radius:10px;'
                'background:#444;color:#aaa;font-size:13px;">No flood zone coverage</span>',
                unsafe_allow_html=True,
            )
        else:
            badge = (
                f'<span style="display:inline-block;padding:3px 12px;border-radius:10px;'
                f'background:{color};color:#fff;font-weight:bold;font-size:14px;">'
                f'{category}</span>'
            )
            st.markdown(badge, unsafe_allow_html=True)
    with col_b:
        alt_missing = pd.isna(alt_score) or alt_score is None
        alt_score_str = f"{float(alt_score):.3f}" if not alt_missing else "—"
        st.metric(alt_layer_cfg["label"], alt_score_str)
        alt_color = RISK_COLORS.get(alt_category, "#666")
        if alt_missing:
            st.markdown(
                '<span style="display:inline-block;padding:3px 12px;border-radius:10px;'
                'background:#444;color:#aaa;font-size:13px;">No flood zone coverage</span>',
                unsafe_allow_html=True,
            )
        else:
            alt_badge = (
                f'<span style="display:inline-block;padding:3px 12px;border-radius:10px;'
                f'background:{alt_color};color:#fff;font-weight:bold;font-size:14px;">'
                f'{alt_category}</span>'
            )
            st.markdown(alt_badge, unsafe_allow_html=True)

    st.markdown("")
    pop_str = f"{int(pop):,}" if not pd.isna(pop) else "N/A"
    st.markdown(f"**Population:** {pop_str}")

    # Gauge: show active layer score; if missing, fall back to the other layer
    gauge_score = score if not score_missing else alt_score
    gauge_label = layer_cfg["label"] if not score_missing else f"{alt_layer_cfg['label']} (fallback)"
    if gauge_score is not None and not pd.isna(gauge_score):
        st.plotly_chart(gauge_chart(gauge_score, gauge_label), width="stretch")
    else:
        st.markdown("*No raster data available for this area.*")

    if not pd.isna(rank):
        st.markdown(f"**Rank:** #{int(rank)} out of {total_ntas} neighborhoods")

    st.divider()
    if score_missing:
        st.info(
            f"This area has **no composite flood zone coverage** in the model. "
            f"Flood Hazard score: **{alt_score_str}** ({alt_category})."
        )
    else:
        level = category if category not in ("Unknown", None) else "UNKNOWN"
        rank_str = f"#{int(rank)}" if not pd.isna(rank) else "unranked"
        st.info(
            f"This neighborhood has **{level}** flood risk. "
            f"It ranks **{rank_str}** in NYC for flood vulnerability "
            f"({layer_cfg['label']})."
        )


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🌊 NYC Flood Risk")
    st.caption("Explore flood vulnerability across NYC neighborhoods")
    st.divider()

    gdf = load_data()

    active_layer_name = st.radio(
        "**Active Map Layer**",
        options=list(LAYERS.keys()),
        index=0,
    )
    layer_cfg = LAYERS[active_layer_name]
    alt_layer_name = [k for k in LAYERS if k != active_layer_name][0]
    alt_layer_cfg  = LAYERS[alt_layer_name]

    st.caption(layer_cfg["description"])
    st.divider()

    all_names = sorted(gdf["display_name"].dropna().unique().tolist())
    selected_name = st.selectbox(
        "Search neighborhood",
        options=all_names,
        index=0,
    )

    risk_filter = st.multiselect(
        "Filter by risk level",
        options=["Low", "Medium", "High", "Unknown"],
        default=["Low", "Medium", "High", "Unknown"],
    )

    st.divider()
    with st.expander("About"):
        st.markdown(
            "- **Flood Hazard** layer uses `FloodHazardMapLR.tif` — raw hazard "
            "score based on FEMA flood zones and inundation modeling.\n"
            "- **Composite Risk** layer fuses hazard with social vulnerability "
            "(`composite_flood_risk_map.tif`). Only areas inside flood zones are scored.\n"
            "- **Zonal statistics** compute the mean pixel value per NTA polygon "
            "via `rasterstats`, then normalize to 0–1 using 99th-percentile clipping."
        )

# ── Apply filter ───────────────────────────────────────────────────────────────
cat_col = layer_cfg["category_col"]
gdf_filtered = gdf[gdf[cat_col].isin(risk_filter)] if risk_filter else gdf.copy()

selected_row  = gdf[gdf["display_name"] == selected_name].iloc[0]
selected_code = selected_row["ntacode"]
total_ntas    = int(gdf["total_ntas"].iloc[0])

# ── Main panel ─────────────────────────────────────────────────────────────────
st.title("NYC Flood Risk Explorer")
st.caption(f"Viewing: **{active_layer_name}**  ·  {len(gdf_filtered)} neighborhoods shown")

map_col, detail_col = st.columns([6, 4])

with map_col:
    fmap = build_map(
        gdf_filtered,
        score_col=layer_cfg["score_col"],
        category_col=layer_cfg["category_col"],
        selected_ntacode=selected_code,
    )
    st_folium(fmap, use_container_width=True, height=560, returned_objects=[])

with detail_col:
    detail_card(selected_row.to_dict(), layer_cfg, alt_layer_cfg, total_ntas)

# ── Bottom: table + download ───────────────────────────────────────────────────
st.divider()
st.subheader("All Neighborhoods")

table_df = gdf[[
    "display_name", "boro_name",
    "hazard_score", "hazard_category",
    "composite_score", "composite_category",
    "Population",
]].rename(columns={
    "display_name":       "Neighborhood",
    "boro_name":          "Borough",
    "hazard_score":       "Hazard Score",
    "hazard_category":    "Hazard Level",
    "composite_score":    "Composite Score",
    "composite_category": "Composite Level",
})
sort_col = "Hazard Score" if layer_cfg["score_col"] == "hazard_score" else "Composite Score"
table_df = table_df.sort_values(sort_col, ascending=False, na_position="last")

for col in ("Hazard Score", "Composite Score"):
    table_df[col] = table_df[col].round(4)

st.dataframe(table_df.reset_index(drop=True), width="stretch", height=380)

csv = table_df.to_csv(index=False).encode("utf-8")
st.download_button("⬇ Download CSV", csv, "nyc_flood_risk_nta.csv", "text/csv")
