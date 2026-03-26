import io
import tempfile
from typing import Optional

import altair as alt
import numpy as np
import pandas as pd
import geopandas as gpd
import streamlit as st
import pydeck as pdk

try:
    import h3
    H3_AVAILABLE = True
except ImportError:
    H3_AVAILABLE = False


st.set_page_config(page_title="Competencia entre líneas (H3)", layout="wide")

# =========================
# Helpers
# =========================

REQUIRED_COLUMNS = [
    "FECHA",
    "DATE_TIME",
    "NUM_LINEA",
    "SENTIDO",
    "INTERNO",
    "LONGITUDE",
    "LATITUDE",
    "CANT_TRAX",
]

H3_RES_OPTIONS = {
    "Resolución 7 (~5.16 km²)": 7,
    "Resolución 8 (~0.74 km²)": 8,
    "Resolución 9 (~0.11 km²)": 9,
    "Resolución 10 (~0.016 km²)": 10,
    "Resolución 11 (~0.0023 km²)": 11,
}

TIME_BLOCKS = {
    "Todo el día": (0, 24),
    "Madrugada (0-5)": (0, 6),
    "Punta mañana (6-9)": (6, 10),
    "Media mañana (10-12)": (10, 13),
    "Mediodía (13-15)": (13, 16),
    "Punta tarde (16-19)": (16, 20),
    "Noche (20-23)": (20, 24),
}


@st.cache_data(show_spinner=False)
def load_geojson(uploaded_geojson_bytes: bytes) -> gpd.GeoDataFrame:
    with tempfile.NamedTemporaryFile(suffix=".geojson", delete=False) as tmp:
        tmp.write(uploaded_geojson_bytes)
        tmp_path = tmp.name
    gdf = gpd.read_file(tmp_path, engine="pyogrio")
    return gdf


def normalize_line_value(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    digits = "".join(ch for ch in s if ch.isdigit())
    if digits == "":
        return np.nan
    return int(digits)


def prepare_routes_gdf(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    gdf = gdf.copy()
    gdf.columns = [str(c).strip().lower() for c in gdf.columns]

    required = ["linea", "sentido", "geometry"]
    missing = [c for c in required if c not in gdf.columns]
    if missing:
        raise ValueError(f"Faltan campos en el geojson: {missing}")

    gdf["linea_norm"] = gdf["linea"].apply(normalize_line_value)
    gdf["sentido_norm"] = gdf["sentido"].astype(str).str.strip().str.upper()

    if gdf.crs is None:
        gdf = gdf.set_crs(epsg=4326)
    else:
        gdf = gdf.to_crs(epsg=4326)

    return gdf


def geometry_to_paths(geom):
    if geom is None:
        return []

    if geom.geom_type == "LineString":
        return [[list(coord) for coord in geom.coords]]

    if geom.geom_type == "MultiLineString":
        return [[list(coord) for coord in part.coords] for part in geom.geoms]

    return []


def build_routes_layer_df(
    routes_gdf: gpd.GeoDataFrame,
    selected_line: int,
    selected_sentido: str,
) -> pd.DataFrame:
    gdf = routes_gdf[routes_gdf["linea_norm"] == int(selected_line)].copy()

    if selected_sentido != "TODOS":
        gdf = gdf[gdf["sentido_norm"] == selected_sentido].copy()

    if gdf.empty:
        return pd.DataFrame(columns=["linea_norm", "sentido_norm", "path", "color", "tooltip"])

    rows = []
    for _, row in gdf.iterrows():
        paths = geometry_to_paths(row.geometry)

        for path in paths:
            if len(path) >= 2:
                sentido = row["sentido_norm"]
                color = [60, 120, 255] if sentido == "IDA" else [255, 140, 50]
                rows.append(
                    {
                        "linea_norm": row["linea_norm"],
                        "sentido_norm": sentido,
                        "path": path,
                        "color": color,
                        "tooltip": f"Línea {row['linea_norm']} - {sentido}",
                    }
                )

    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def list_excel_sheets(uploaded_file_bytes: bytes) -> list[str]:
    xls = pd.ExcelFile(io.BytesIO(uploaded_file_bytes))
    return xls.sheet_names


@st.cache_data(show_spinner=False)
def load_excel(uploaded_file_bytes: bytes, sheet_name: str) -> pd.DataFrame:
    return pd.read_excel(io.BytesIO(uploaded_file_bytes), sheet_name=sheet_name)


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().upper() for c in df.columns]
    return df


def validate_columns(df: pd.DataFrame) -> tuple[bool, list[str]]:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    return len(missing) == 0, missing


def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["DATE_TIME"] = pd.to_datetime(df["DATE_TIME"], errors="coerce")
    df["FECHA"] = pd.to_datetime(df["FECHA"], errors="coerce")

    df["NUM_LINEA"] = pd.to_numeric(df["NUM_LINEA"], errors="coerce")
    df["INTERNO"] = df["INTERNO"].astype(str).str.strip()
    df["SENTIDO"] = df["SENTIDO"].astype(str).str.strip().str.upper()

    df["LONGITUDE"] = pd.to_numeric(df["LONGITUDE"], errors="coerce")
    df["LATITUDE"] = pd.to_numeric(df["LATITUDE"], errors="coerce")
    df["CANT_TRAX"] = pd.to_numeric(df["CANT_TRAX"], errors="coerce").fillna(0)

    df = df[
        df["LONGITUDE"].between(-59.5, -57.5)
        & df["LATITUDE"].between(-35.5, -33.0)
    ].copy()

    df = df.dropna(subset=["DATE_TIME", "NUM_LINEA", "LONGITUDE", "LATITUDE"])

    df["FECHA_ONLY"] = df["DATE_TIME"].dt.date
    df["HORA"] = df["DATE_TIME"].dt.hour
    df["MINUTO"] = df["DATE_TIME"].dt.minute
    df["BLOQUE_15M"] = (df["MINUTO"] // 15 + 1).astype(int)
    df["FRANJA_15M"] = (
        df["HORA"].astype(str).str.zfill(2)
        + ":"
        + ((df["BLOQUE_15M"] - 1) * 15).astype(str).str.zfill(2)
        + "-"
        + (df["HORA"]).astype(str).str.zfill(2)
        + ":"
        + (df["BLOQUE_15M"] * 15).clip(upper=60).astype(str).str.zfill(2)
    )

    return df


def filter_by_time_block(df: pd.DataFrame, block_name: str) -> pd.DataFrame:
    start_h, end_h = TIME_BLOCKS[block_name]
    return df[(df["HORA"] >= start_h) & (df["HORA"] < end_h)].copy()


def h3_boundary_lonlat(hex_id: str):
    boundary = h3.cell_to_boundary(hex_id)
    return [[lon, lat] for lat, lon in boundary]


def h3_info(hex_id: str) -> dict:
    res = h3.get_resolution(hex_id)
    area_m2 = h3.cell_area(hex_id, unit="m^2")
    edge_m = h3.average_hexagon_edge_length(res, unit="m")
    return {
        "resolucion": res,
        "area_m2": area_m2,
        "area_km2": area_m2 / 1_000_000,
        "lado_m": edge_m,
    }


@st.cache_data(show_spinner=False)
def assign_points_to_hexes_h3(df_prepared: pd.DataFrame, h3_res: int) -> pd.DataFrame:
    df = df_prepared.copy()
    df["hex_id"] = [
        h3.latlng_to_cell(lat, lon, h3_res)
        for lat, lon in zip(df["LATITUDE"], df["LONGITUDE"])
    ]
    return df


@st.cache_data(show_spinner=False)
def build_hex_polygons_h3(df_prepared: pd.DataFrame, h3_res: int) -> pd.DataFrame:
    hex_ids = pd.Series(
        [h3.latlng_to_cell(lat, lon, h3_res) for lat, lon in zip(df_prepared["LATITUDE"], df_prepared["LONGITUDE"])],
        name="hex_id",
    ).dropna().drop_duplicates()

    if hex_ids.empty:
        return pd.DataFrame(columns=["hex_id", "polygon", "lon", "lat"])

    out = pd.DataFrame({"hex_id": hex_ids.tolist()})
    out["polygon"] = out["hex_id"].apply(lambda hid: [h3_boundary_lonlat(hid)])
    out["lat"] = out["hex_id"].apply(lambda hid: h3.cell_to_latlng(hid)[0])
    out["lon"] = out["hex_id"].apply(lambda hid: h3.cell_to_latlng(hid)[1])
    return out


def compute_metrics(
    df_hex: pd.DataFrame,
    my_line: int,
    selected_lines: list[int],
    min_trx_hex: int,
    min_lines_competing: int,
) -> pd.DataFrame:
    work = df_hex[df_hex["NUM_LINEA"].isin(selected_lines)].copy()

    demand = (
        work.groupby(["hex_id", "NUM_LINEA"], as_index=False)
        .agg(
            trx_linea=("CANT_TRAX", "sum"),
            internos_linea=("INTERNO", pd.Series.nunique),
            registros=("INTERNO", "count"),
        )
    )

    totals = (
        work.groupby("hex_id", as_index=False)
        .agg(
            trx_total_hex=("CANT_TRAX", "sum"),
            internos_total_hex=("INTERNO", pd.Series.nunique),
            lineas_presentes=("NUM_LINEA", pd.Series.nunique),
        )
    )

    met = demand.merge(totals, on="hex_id", how="left")

    met["share_demanda"] = np.where(
        met["trx_total_hex"] > 0,
        met["trx_linea"] / met["trx_total_hex"],
        0,
    )

    met["share_oferta"] = np.where(
        met["internos_total_hex"] > 0,
        met["internos_linea"] / met["internos_total_hex"],
        0,
    )

    met["trx_por_interno"] = np.where(
        met["internos_linea"] > 0,
        met["trx_linea"] / met["internos_linea"],
        0,
    )

    met["indice_captacion"] = np.where(
        met["share_oferta"] > 0,
        met["share_demanda"] / met["share_oferta"],
        np.nan,
    )

    my = met[met["NUM_LINEA"] == my_line].copy()

    my = my[
        (my["trx_total_hex"] >= min_trx_hex)
        & (my["lineas_presentes"] >= min_lines_competing)
    ].copy()

    top_share = (
        met.groupby("hex_id", as_index=False)["share_demanda"]
        .max()
        .rename(columns={"share_demanda": "top_share"})
    )
    my = my.merge(top_share, on="hex_id", how="left")
    my["balance_competitivo"] = 1 - my["top_share"]

    if not my.empty:
        my["trx_norm"] = my["trx_total_hex"] / my["trx_total_hex"].max()
        my["comp_norm"] = my["lineas_presentes"] / my["lineas_presentes"].max()
        my["bal_norm"] = my["balance_competitivo"] / max(my["balance_competitivo"].max(), 1e-9)
        my["score_control"] = 0.5 * my["trx_norm"] + 0.3 * my["comp_norm"] + 0.2 * my["bal_norm"]
    else:
        my["score_control"] = pd.Series(dtype=float)

    return my.sort_values(["score_control", "trx_total_hex"], ascending=[False, False])


def build_line_comparison(df_hex: pd.DataFrame, hex_id: str, selected_lines: list[int]) -> pd.DataFrame:
    sub = df_hex[(df_hex["hex_id"] == hex_id) & (df_hex["NUM_LINEA"].isin(selected_lines))].copy()

    out = (
        sub.groupby("NUM_LINEA", as_index=False)
        .agg(
            trx=("CANT_TRAX", "sum"),
            internos=("INTERNO", pd.Series.nunique),
            registros=("INTERNO", "count"),
        )
    )

    trx_total = out["trx"].sum()
    int_total = out["internos"].sum()

    out["share_demanda"] = np.where(trx_total > 0, out["trx"] / trx_total, 0)
    out["share_oferta"] = np.where(int_total > 0, out["internos"] / int_total, 0)
    out["indice_captacion"] = np.where(out["share_oferta"] > 0, out["share_demanda"] / out["share_oferta"], np.nan)
    out["trx_por_interno"] = np.where(out["internos"] > 0, out["trx"] / out["internos"], 0)

    return out.sort_values("trx", ascending=False)


def build_hourly_evolution(df_hex: pd.DataFrame, hex_id: str, selected_lines: list[int]) -> pd.DataFrame:
    sub = df_hex[(df_hex["hex_id"] == hex_id) & (df_hex["NUM_LINEA"].isin(selected_lines))].copy()
    out = (
        sub.groupby(["HORA", "NUM_LINEA"], as_index=False)
        .agg(
            trx=("CANT_TRAX", "sum"),
            internos=("INTERNO", pd.Series.nunique),
        )
    )
    return out.sort_values(["HORA", "NUM_LINEA"])


def build_hourly_evolution_total(df_filtered: pd.DataFrame, selected_lines: list[int]) -> pd.DataFrame:
    sub = df_filtered[df_filtered["NUM_LINEA"].isin(selected_lines)].copy()
    out = (
        sub.groupby(["HORA", "NUM_LINEA"], as_index=False)
        .agg(
            trx=("CANT_TRAX", "sum"),
            internos=("INTERNO", pd.Series.nunique),
        )
    )
    return out.sort_values(["HORA", "NUM_LINEA"])


def to_download_excel(points_df: pd.DataFrame, compare_df: Optional[pd.DataFrame] = None) -> bytes:
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        points_df.to_excel(writer, sheet_name="Puntos_control", index=False)
        if compare_df is not None and not compare_df.empty:
            compare_df.to_excel(writer, sheet_name="Comparacion_hex", index=False)
    buffer.seek(0)
    return buffer.read()


def color_by_index(x):
    if pd.isna(x):
        return [160, 160, 160, 120]
    if x < 0.8:
        return [220, 70, 70, 160]
    if x <= 1.2:
        return [150, 150, 150, 140]
    return [50, 170, 90, 160]


def color_by_heat_pct(p):
    if pd.isna(p):
        return [200, 200, 200, 120]
    if p <= 0:
        return [245, 245, 245, 80]
    elif p < 0.01:
        return [255, 245, 204, 120]
    elif p < 0.02:
        return [255, 230, 153, 140]
    elif p < 0.05:
        return [255, 204, 102, 160]
    elif p < 0.10:
        return [255, 153, 51, 180]
    else:
        return [230, 85, 13, 200]


def color_by_share_demanda(p):
    if pd.isna(p):
        return [220, 220, 220, 120]
    if p < 0.05:
        return [247, 251, 255, 90]
    elif p < 0.10:
        return [222, 235, 247, 110]
    elif p < 0.20:
        return [198, 219, 239, 130]
    elif p < 0.35:
        return [158, 202, 225, 150]
    elif p < 0.50:
        return [107, 174, 214, 170]
    elif p < 0.70:
        return [49, 130, 189, 190]
    else:
        return [8, 81, 156, 210]


# =========================
# UI
# =========================

st.title("Análisis de competencia entre líneas por hexágonos H3")
st.caption("Subí tu Excel y analizá competencia usando resolución H3 real.")

if not H3_AVAILABLE:
    st.error("Instalá h3: pip install h3")
    st.stop()

with st.expander("Carga de archivos y configuración inicial", expanded=True):
    uploaded_file = st.file_uploader(
        "Subir archivo Excel",
        type=["xlsx", "xls"],
        help="Debe contener: FECHA, DATE_TIME, NUM_LINEA, SENTIDO, INTERNO, LONGITUDE, LATITUDE, CANT_TRAX.",
        key="uploader_excel",
    )

    uploaded_geojson = st.file_uploader(
        "Subir GeoJSON de recorridos",
        type=["geojson", "json"],
        help='Debe contener al menos los campos "linea" y "sentido".',
        key="uploader_geojson",
    )

    if uploaded_file is None:
        st.info("Subí un archivo Excel para empezar.")
        st.stop()

    file_bytes = uploaded_file.getvalue()
    sheets = list_excel_sheets(file_bytes)
    sheet = st.selectbox("Hoja del Excel", sheets, key="sheet_selector")

    raw_df = load_excel(file_bytes, sheet)
    raw_df = standardize_columns(raw_df)

    ok, missing_cols = validate_columns(raw_df)
    if not ok:
        st.error(f"Faltan columnas obligatorias: {missing_cols}")
        st.dataframe(raw_df.head(10), use_container_width=True)
        st.stop()

    df = prepare_dataframe(raw_df)

    routes_gdf = None
    if uploaded_geojson is not None:
        try:
            geojson_bytes = uploaded_geojson.getvalue()
            routes_raw = load_geojson(geojson_bytes)
            routes_gdf = prepare_routes_gdf(routes_raw)
            st.success(f"GeoJSON cargado: {len(routes_gdf):,} recorridos".replace(",", "."))
        except Exception as e:
            st.error(f"No se pudo leer/procesar el GeoJSON: {e}")
            routes_gdf = None

    if df.empty:
        st.error("No quedaron registros válidos luego de la limpieza.")
        st.stop()

    c1, c2 = st.columns(2)
    with c1:
        st.metric("Registros válidos", f"{len(df):,}".replace(",", "."))
        st.metric("Hojas detectadas", len(sheets))
    with c2:
        st.metric("Recorridos cargados", f"{len(routes_gdf):,}".replace(",", ".") if routes_gdf is not None else "0")

    with st.expander("Vista previa de la base procesada", expanded=False):
        st.dataframe(df.head(20), use_container_width=True)

tab_competencia = st.tabs(["Competencia por hexágonos H3"])[0]

with tab_competencia:
    st.sidebar.header("Filtros")

    available_dates = sorted(df["FECHA_ONLY"].dropna().unique())
    selected_date = st.sidebar.selectbox("Fecha", available_dates)

    df_f = df[df["FECHA_ONLY"] == selected_date].copy()

    lineas = sorted(df_f["NUM_LINEA"].dropna().astype(int).unique().tolist())
    my_line = st.sidebar.selectbox("Mi línea", lineas)

    selected_lines_key = "selected_lines_competencia_h3"
    default_comp = [x for x in lineas if x != my_line]
    default_selected = [my_line] + default_comp

    if selected_lines_key not in st.session_state:
        st.session_state[selected_lines_key] = default_selected
    else:
        st.session_state[selected_lines_key] = [
            x for x in st.session_state[selected_lines_key] if x in lineas
        ]
        if not st.session_state[selected_lines_key]:
            st.session_state[selected_lines_key] = default_selected
        if my_line not in st.session_state[selected_lines_key]:
            st.session_state[selected_lines_key] = [my_line] + st.session_state[selected_lines_key]

    selected_lines = st.sidebar.multiselect(
        "Líneas a comparar",
        options=lineas,
        key=selected_lines_key,
    )

    if my_line not in selected_lines:
        selected_lines = [my_line] + selected_lines
        st.session_state[selected_lines_key] = selected_lines

    sentidos = sorted(df_f["SENTIDO"].dropna().unique().tolist())
    selected_sentido = st.sidebar.selectbox("Sentido", ["TODOS"] + sentidos)

    if selected_sentido != "TODOS":
        df_f = df_f[df_f["SENTIDO"] == selected_sentido].copy()

    selected_block = st.sidebar.selectbox("Franja horaria", list(TIME_BLOCKS.keys()))
    df_f = filter_by_time_block(df_f, selected_block)

    selected_hours = st.sidebar.slider("Rango de hora fino", 0, 23, (0, 23))
    df_f = df_f[(df_f["HORA"] >= selected_hours[0]) & (df_f["HORA"] <= selected_hours[1])].copy()

    h3_label = st.sidebar.selectbox("Resolución H3", list(H3_RES_OPTIONS.keys()), index=2)
    h3_res = H3_RES_OPTIONS[h3_label]

    min_trx_hex = st.sidebar.number_input("Mínimo de transacciones totales por hexágono", min_value=1, value=50, step=10)
    min_lines_competing = st.sidebar.number_input("Mínimo de líneas presentes", min_value=1, value=2, step=1)

    show_trx_circles = st.sidebar.checkbox("Mostrar círculos por transacciones totales", value=False)

    map_mode = st.sidebar.selectbox(
        "Modo de visualización del mapa",
        options=[
            "Índice de captación",
            "Mapa de calor por pct_trx_hex",
            "Mapa de calor por share_demanda",
        ],
        index=1,
    )

    if df_f.empty:
        st.warning("No hay datos para los filtros elegidos.")
        st.stop()

    with st.spinner("Generando hexágonos H3 y métricas..."):
        df_hex = assign_points_to_hexes_h3(df_f, h3_res)
        hex_polygons = build_hex_polygons_h3(df_f, h3_res)

        points_control = compute_metrics(
            df_hex=df_hex,
            my_line=my_line,
            selected_lines=selected_lines,
            min_trx_hex=int(min_trx_hex),
            min_lines_competing=int(min_lines_competing),
        ).merge(hex_polygons, on="hex_id", how="left")

        points_control_base_total = points_control["trx_total_hex"].sum() if not points_control.empty else 0

        if points_control_base_total > 0:
            points_control["pct_trx_hex"] = points_control["trx_total_hex"] / points_control_base_total
        else:
            points_control["pct_trx_hex"] = 0.0

        points_control["label_pct"] = ((points_control["pct_trx_hex"] * 100).round(1).astype(str) + "%")
        points_control["label_share_demanda"] = ((points_control["share_demanda"] * 100).round(1).astype(str) + "%")

        map_points_control = points_control.copy()

        exclude_hex = "NINGUNO"
        if not map_points_control.empty:
            exclude_hex = st.sidebar.selectbox(
                "Excluir hexágono del mapa de calor",
                options=["NINGUNO"] + sorted(map_points_control["hex_id"].astype(str).unique().tolist()),
                index=0,
                key="exclude_hex_heatmap_h3",
            )

            if exclude_hex != "NINGUNO":
                map_points_control = map_points_control[
                    map_points_control["hex_id"] != exclude_hex
                ].copy()

        trx_total_filtro = map_points_control["trx_total_hex"].sum() if not map_points_control.empty else 0

        if trx_total_filtro > 0:
            map_points_control["pct_trx_hex"] = map_points_control["trx_total_hex"] / trx_total_filtro
        else:
            map_points_control["pct_trx_hex"] = 0.0

        map_points_control["label_pct"] = ((map_points_control["pct_trx_hex"] * 100).round(1).astype(str) + "%")
        map_points_control["label_share_demanda"] = ((map_points_control["share_demanda"] * 100).round(1).astype(str) + "%")

    evol_total = build_hourly_evolution_total(df_f, selected_lines)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Registros filtrados", f"{len(df_f):,}".replace(",", "."))
    c2.metric("Líneas comparadas", f"{len(selected_lines)}")
    c3.metric("Hexágonos H3 con competencia", f"{points_control['hex_id'].nunique():,}".replace(",", ".") if not points_control.empty else "0")
    c4.metric("Mi línea", str(my_line))

    st.subheader("Mapa de puntos de control H3")
    st.caption(f"Resolución seleccionada: H3 res {h3_res}")

    if exclude_hex != "NINGUNO":
        st.info(f"Mapa de calor recalculado excluyendo el hexágono H3: {exclude_hex}")

    if points_control.empty:
        st.warning("No se encontraron hexágonos competitivos con los umbrales elegidos.")
    else:
        map_df = map_points_control.copy()

        map_df["tooltip"] = (
            "Hex H3: " + map_df["hex_id"].astype(str)
            + "\nTrx total: " + map_df["trx_total_hex"].round(0).astype(int).astype(str)
            + "\n% del total: " + (map_df["pct_trx_hex"] * 100).round(2).astype(str) + "%"
            + "\nLíneas presentes: " + map_df["lineas_presentes"].astype(int).astype(str)
            + "\nShare demanda mi línea: " + (map_df["share_demanda"] * 100).round(1).astype(str) + "%"
            + "\nShare oferta mi línea: " + (map_df["share_oferta"] * 100).round(1).astype(str) + "%"
            + "\nÍndice captación: " + map_df["indice_captacion"].round(2).astype(str)
        )

        if map_mode == "Mapa de calor por pct_trx_hex":
            map_df["fill_color"] = map_df["pct_trx_hex"].apply(color_by_heat_pct)
        elif map_mode == "Mapa de calor por share_demanda":
            map_df["fill_color"] = map_df["share_demanda"].apply(color_by_share_demanda)
        else:
            map_df["fill_color"] = map_df["indice_captacion"].apply(color_by_index)

        # radio aprox. basado en el lado promedio H3
        edge_m = h3.average_hexagon_edge_length(h3_res, unit="m")
        map_df["circle_radius"] = 0.0
        if map_df["trx_total_hex"].max() > 0:
            map_df["circle_radius"] = (
                edge_m * 0.20 + (edge_m * 0.75) * (map_df["trx_total_hex"] / map_df["trx_total_hex"].max())
            )

        center_lat = map_df["lat"].mean()
        center_lon = map_df["lon"].mean()

        hex_layer = pdk.Layer(
            "PolygonLayer",
            data=map_df,
            get_polygon="polygon",
            get_fill_color="fill_color",
            get_line_color=[90, 90, 90, 180],
            line_width_min_pixels=1,
            stroked=True,
            filled=True,
            pickable=True,
            auto_highlight=True,
        )

        text_layer = None
        if map_mode == "Mapa de calor por pct_trx_hex":
            text_layer = pdk.Layer(
                "TextLayer",
                data=map_df,
                get_position="[lon, lat]",
                get_text="label_pct",
                get_size=14,
                get_color=[40, 40, 40, 220],
                get_angle=0,
                get_text_anchor="'middle'",
                get_alignment_baseline="'center'",
                pickable=False,
            )
        elif map_mode == "Mapa de calor por share_demanda":
            text_layer = pdk.Layer(
                "TextLayer",
                data=map_df,
                get_position="[lon, lat]",
                get_text="label_share_demanda",
                get_size=14,
                get_color=[40, 40, 40, 220],
                get_angle=0,
                get_text_anchor="'middle'",
                get_alignment_baseline="'center'",
                pickable=False,
            )

        routes_layer = None
        if routes_gdf is not None:
            routes_df = build_routes_layer_df(
                routes_gdf=routes_gdf,
                selected_line=my_line,
                selected_sentido=selected_sentido,
            )
            if not routes_df.empty:
                routes_layer = pdk.Layer(
                    "PathLayer",
                    data=routes_df,
                    get_path="path",
                    get_color="color",
                    width_scale=1,
                    width_min_pixels=4,
                    pickable=True,
                    auto_highlight=True,
                )

        trx_circle_layer = None
        if show_trx_circles:
            trx_circle_layer = pdk.Layer(
                "ScatterplotLayer",
                data=map_df,
                get_position="[lon, lat]",
                get_radius="circle_radius",
                get_fill_color=[30, 144, 255, 90],
                get_line_color=[20, 90, 160, 180],
                line_width_min_pixels=1,
                stroked=True,
                filled=True,
                pickable=True,
            )

        layers = [hex_layer]
        if trx_circle_layer is not None:
            layers.append(trx_circle_layer)
        if routes_layer is not None:
            layers.append(routes_layer)
        if text_layer is not None:
            layers.append(text_layer)

        view_state = pdk.ViewState(
            latitude=float(center_lat),
            longitude=float(center_lon),
            zoom=10.5,
            pitch=0,
        )

        deck = pdk.Deck(
            layers=layers,
            initial_view_state=view_state,
            tooltip={"text": "{tooltip}"},
            map_style="light",
        )

        st.pydeck_chart(deck, use_container_width=True)

    st.subheader("Ranking de puntos de control")

    show_cols = [
        "hex_id",
        "trx_total_hex",
        "pct_trx_hex",
        "lineas_presentes",
        "trx_linea",
        "internos_linea",
        "share_demanda",
        "share_oferta",
        "trx_por_interno",
        "indice_captacion",
        "score_control",
    ]

    display_df = pd.DataFrame(columns=show_cols)
    comp_hex = pd.DataFrame()

    if points_control.empty:
        st.info("No hay puntos de control para mostrar.")
    else:
        display_df = points_control[show_cols].copy()
        display_df["pct_trx_hex"] = (display_df["pct_trx_hex"] * 100).round(2)
        display_df["share_demanda"] = (display_df["share_demanda"] * 100).round(2)
        display_df["share_oferta"] = (display_df["share_oferta"] * 100).round(2)
        display_df["trx_por_interno"] = display_df["trx_por_interno"].round(2)
        display_df["indice_captacion"] = display_df["indice_captacion"].round(2)
        display_df["score_control"] = display_df["score_control"].round(3)
        st.dataframe(display_df, use_container_width=True, height=420)

    st.subheader("Detalle de un punto de control H3")

    if not points_control.empty:
        selected_hex = st.selectbox("Elegir hexágono H3", points_control["hex_id"].tolist())

        info_hex = h3_info(selected_hex)
        st.caption(
            f"Hexágono seleccionado: res {info_hex['resolucion']} | "
            f"área aprox. {info_hex['area_km2']:.3f} km² | "
            f"lado aprox. {info_hex['lado_m']:.0f} m"
        )

        comp_hex = build_line_comparison(df_hex, selected_hex, selected_lines)
        evol_hex = build_hourly_evolution(df_hex, selected_hex, selected_lines)

        col_a, col_b = st.columns([1, 1])

        with col_a:
            st.markdown("**Comparación por línea en el hexágono**")
            comp_show = comp_hex.copy()
            comp_show["share_demanda"] = (comp_show["share_demanda"] * 100).round(2)
            comp_show["share_oferta"] = (comp_show["share_oferta"] * 100).round(2)
            comp_show["indice_captacion"] = comp_show["indice_captacion"].round(2)
            comp_show["trx_por_interno"] = comp_show["trx_por_interno"].round(2)
            st.dataframe(comp_show, use_container_width=True)

        with col_b:
            st.markdown("**Evolución horaria**")
            if evol_hex.empty:
                st.info("No hay datos para este hexágono.")
            else:
                chart_df = evol_hex.pivot(index="HORA", columns="NUM_LINEA", values="trx").fillna(0)
                st.line_chart(chart_df, use_container_width=True)

    download_bytes = to_download_excel(display_df, comp_hex)
    st.download_button(
        "Descargar resultados en Excel",
        data=download_bytes,
        file_name=f"competencia_lineas_h3_{selected_date}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
