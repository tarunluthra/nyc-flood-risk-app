"""NYC Flood Risk Explorer — Streamlit app."""

import os
import json
import numpy as np
import pandas as pd
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

DATA_PATH  = os.path.join(os.path.dirname(__file__), "data", "nta_risk.geojson")
INFRA_PATH = os.path.join(os.path.dirname(__file__), "data", "infrastructure.csv")

INFRA_STYLE = {
    "Fire Services":           {"color": "#e74c3c", "icon": "fire"},
    "Hospitals & Clinics":     {"color": "#f39c12", "icon": "plus-sign"},
    "Other Emergency Services":{"color": "#9b59b6", "icon": "star"},
    "Bus Depots & Terminals":  {"color": "#2ecc71", "icon": "transfer"},
}

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


def _geom_bounds(geom):
    """Return [minx, miny, maxx, maxy] from a GeoJSON geometry dict."""
    if geom is None:
        return None
    gtype = geom.get("type", "")
    if gtype == "Polygon":
        coords = [c for ring in geom["coordinates"] for c in ring]
    elif gtype == "MultiPolygon":
        coords = [c for poly in geom["coordinates"] for ring in poly for c in ring]
    else:
        return None
    if not coords:
        return None
    xs = [c[0] for c in coords]
    ys = [c[1] for c in coords]
    return [min(xs), min(ys), max(xs), max(ys)]


def _safe_val(v):
    """Convert numpy/NaN values to JSON-safe Python types."""
    if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
        return None
    if hasattr(v, "item"):
        return v.item()
    return v


@st.cache_data
def load_data():
    with open(DATA_PATH, encoding="utf-8") as f:
        raw = json.load(f)
    rows = []
    for feat in raw["features"]:
        row = {k: v for k, v in feat["properties"].items()}
        row["_geometry"] = feat["geometry"]
        rows.append(row)
    df = pd.DataFrame(rows)
    df["display_name"] = df["ntaname"].fillna(df.get("NTA Name", df.get("ntacode", "")))
    for col in ("hazard_score", "composite_score"):
        df[col] = pd.to_numeric(df[col], errors="coerce").clip(0, 1)
    df["Population"] = pd.to_numeric(df["Population"], errors="coerce")
    df["total_ntas"] = int(df["total_ntas"].iloc[0]) if "total_ntas" in df.columns else 195
    return df


@st.cache_data
def load_infrastructure():
    if not os.path.exists(INFRA_PATH):
        return pd.DataFrame()
    df = pd.read_csv(INFRA_PATH)
    df["latitude"]  = pd.to_numeric(df["latitude"],  errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    return df.dropna(subset=["latitude", "longitude"])


def score_to_color(score):
    if pd.isna(score):
        return "#444444"
    s = float(np.clip(score, 0, 1))
    r = int(np.interp(s, [0, 0.5, 1.0], [46,  243, 231]))
    g = int(np.interp(s, [0, 0.5, 1.0], [204, 156,  76]))
    b = int(np.interp(s, [0, 0.5, 1.0], [113,  18,  60]))
    return f"#{r:02x}{g:02x}{b:02x}"


def build_map(df, score_col, category_col, selected_ntacode, infra_df=None, infra_cats=None):
    m = folium.Map(location=[40.7128, -74.0060], zoom_start=11, tiles="CartoDB dark_matter")

    prop_cols = [c for c in df.columns if c != "_geometry"]

    features = []
    for _, row in df.iterrows():
        is_selected = row["ntacode"] == selected_ntacode
        score = row[score_col]
        props = {col: _safe_val(row[col]) for col in prop_cols}
        props["_fill"]   = "#3498db" if is_selected else score_to_color(score)
        props["_stroke"] = "#ffffff" if is_selected else "#1a1a1a"
        props["_weight"] = 3.0 if is_selected else 1.0
        props["_fop"]    = 0.92 if is_selected else 0.80
        features.append({"type": "Feature", "geometry": row["_geometry"], "properties": props})

    geojson_data = {"type": "FeatureCollection", "features": features}

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

    # ── Infrastructure overlay ─────────────────────────────────────────────────
    if infra_df is not None and infra_cats:
        for cat in infra_cats:
            style = INFRA_STYLE.get(cat, {"color": "#aaaaaa", "icon": "info-sign"})
            subset = infra_df[infra_df["category"] == cat]
            grp = folium.FeatureGroup(name=cat, show=True)
            for _, pt in subset.iterrows():
                folium.CircleMarker(
                    location=[pt["latitude"], pt["longitude"]],
                    radius=5,
                    color=style["color"],
                    fill=True,
                    fill_color=style["color"],
                    fill_opacity=0.85,
                    weight=1,
                    tooltip=folium.Tooltip(
                        f"<b>{pt.get('name','—')}</b><br>"
                        f"<span style='color:{style['color']}'>{cat}</span>",
                        sticky=False,
                    ),
                ).add_to(grp)
            grp.add_to(m)
        folium.LayerControl(collapsed=False).add_to(m)

    # Zoom to selected NTA
    sel = df[df["ntacode"] == selected_ntacode]
    if not sel.empty:
        b = _geom_bounds(sel.iloc[0]["_geometry"])
        if b:
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

    score_missing = score is None or (isinstance(score, float) and np.isnan(score))

    col_a, col_b = st.columns(2)
    with col_a:
        score_str = f"{float(score):.3f}" if not score_missing else "—"
        st.metric(layer_cfg["label"], score_str)
        if score_missing:
            st.markdown(
                '<span style="display:inline-block;padding:3px 12px;border-radius:10px;' +
                'background:#444;color:#aaa;font-size:13px;">No flood zone coverage</span>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<span style="display:inline-block;padding:3px 12px;border-radius:10px;' +
                f'background:{color};color:#fff;font-weight:bold;font-size:14px;">{category}</span>',
                unsafe_allow_html=True,
            )
    with col_b:
        alt_missing = alt_score is None or (isinstance(alt_score, float) and np.isnan(alt_score))
        alt_score_str = f"{float(alt_score):.3f}" if not alt_missing else "—"
        st.metric(alt_layer_cfg["label"], alt_score_str)
        alt_color = RISK_COLORS.get(alt_category, "#666")
        if alt_missing:
            st.markdown(
                '<span style="display:inline-block;padding:3px 12px;border-radius:10px;' +
                'background:#444;color:#aaa;font-size:13px;">No flood zone coverage</span>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<span style="display:inline-block;padding:3px 12px;border-radius:10px;' +
                f'background:{alt_color};color:#fff;font-weight:bold;font-size:14px;">{alt_category}</span>',
                unsafe_allow_html=True,
            )

    st.markdown("")
    pop_str = f"{int(pop):,}" if pop is not None and not (isinstance(pop, float) and np.isnan(pop)) else "N/A"
    st.markdown(f"**Population:** {pop_str}")

    gauge_score = score if not score_missing else alt_score
    gauge_label = layer_cfg["label"] if not score_missing else f"{alt_layer_cfg['label']} (fallback)"
    if gauge_score is not None and not (isinstance(gauge_score, float) and np.isnan(gauge_score)):
        st.plotly_chart(gauge_chart(gauge_score, gauge_label), use_container_width=True)
    else:
        st.markdown("*No raster data available for this area.*")

    if rank is not None and not (isinstance(rank, float) and np.isnan(rank)):
        st.markdown(f"**Rank:** #{int(rank)} out of {total_ntas} neighborhoods")

    st.divider()
    if score_missing:
        st.info(
            f"This area has **no composite flood zone coverage** in the model. "
            f"Flood Hazard score: **{alt_score_str}** ({alt_category})."
        )
    else:
        level = category if category not in ("Unknown", None) else "UNKNOWN"
        rank_str = f"#{int(rank)}" if rank is not None and not (isinstance(rank, float) and np.isnan(rank)) else "unranked"
        st.info(
            f"This neighborhood has **{level}** flood risk. "
            f"It ranks **{rank_str}** in NYC for flood vulnerability "
            f"({layer_cfg['label']})."
        )


# ── Load data ──────────────────────────────────────────────────────────────────
gdf       = load_data()
infra_df  = load_infrastructure()
all_names = sorted(gdf["display_name"].dropna().unique().tolist())

if "selected_name" not in st.session_state:
    st.session_state["selected_name"] = all_names[0]

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🌊 NYC Flood Risk")
    st.caption("Explore flood vulnerability across NYC neighborhoods")
    st.divider()

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

    current_idx = all_names.index(st.session_state["selected_name"]) \
        if st.session_state["selected_name"] in all_names else 0

    sidebar_pick = st.selectbox(
        "Search neighborhood",
        options=all_names,
        index=current_idx,
    )
    if sidebar_pick != st.session_state["selected_name"]:
        st.session_state["selected_name"] = sidebar_pick

    risk_filter = st.multiselect(
        "Filter by risk level",
        options=["Low", "Medium", "High", "Unknown"],
        default=["Low", "Medium", "High", "Unknown"],
    )

    st.divider()
    st.markdown("**Key Infrastructure Overlay**")
    infra_cats = []
    for cat, style in INFRA_STYLE.items():
        label = f"🔴 {cat}" if style["color"] == "#e74c3c" else \
                f"🟠 {cat}" if style["color"] == "#f39c12" else \
                f"🟣 {cat}" if style["color"] == "#9b59b6" else \
                f"🟢 {cat}"
        if st.checkbox(label, value=False, key=f"infra_{cat}"):
            infra_cats.append(cat)

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
selected_name = st.session_state["selected_name"]
cat_col = layer_cfg["category_col"]
gdf_filtered = gdf[gdf[cat_col].isin(risk_filter)] if risk_filter else gdf.copy()

selected_row  = gdf[gdf["display_name"] == selected_name].iloc[0]
selected_code = selected_row["ntacode"]
total_ntas    = int(gdf["total_ntas"].iloc[0])

# ── Main panel ─────────────────────────────────────────────────────────────────
st.title("NYC Flood Risk Explorer")
st.caption(f"Viewing: **{active_layer_name}**  ·  {len(gdf_filtered)} neighborhoods shown  ·  click a neighborhood on the map to select it")

map_col, detail_col = st.columns([6, 4])

with map_col:
    fmap = build_map(
        gdf_filtered,
        score_col=layer_cfg["score_col"],
        category_col=layer_cfg["category_col"],
        selected_ntacode=selected_code,
        infra_df=infra_df if infra_cats else None,
        infra_cats=infra_cats,
    )
    map_data = st_folium(
        fmap,
        use_container_width=True,
        height=560,
        returned_objects=["last_object_clicked"],
    )

    if map_data and map_data.get("last_object_clicked"):
        props = map_data["last_object_clicked"].get("properties", {})
        clicked_name = props.get("display_name") or props.get("ntaname")
        if clicked_name and clicked_name in all_names \
                and clicked_name != st.session_state["selected_name"]:
            st.session_state["selected_name"] = clicked_name
            st.rerun()

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

st.dataframe(table_df.reset_index(drop=True), use_container_width=True, height=380)

csv = table_df.to_csv(index=False).encode("utf-8")
st.download_button("⬇ Download CSV", csv, "nyc_flood_risk_nta.csv", "text/csv")
