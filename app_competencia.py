import math
import io
import tempfile
from typing import Optional

import numpy as np
import pandas as pd
import geopandas as gpd
import streamlit as st
import pydeck as pdk
from shapely.geometry import Polygon

st.set_page_config(page_title="Competencia entre líneas", layout="wide")

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

HEX_SIZE_OPTIONS = {
    "150 m": 150,
    "250 m": 250,
    "400 m": 400,
    "1000 m": 1000,
    "3000 m": 3000,
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
        return [[[list(coord) for coord in part.coords]] for part in geom.geoms]

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

        flat_paths = []
        for p in paths:
            if len(p) == 1 and isinstance(p[0], list) and len(p[0]) > 0 and isinstance(p[0][0], list):
                flat_paths.append(p[0])
            else:
                flat_paths.append(p)

        for path in flat_paths:
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
def list_excel_sheets(uploaded_file_bytes: bytes, filename: str) -> list[str]:
    xls = pd.ExcelFile(io.BytesIO(uploaded_file_bytes))
    return xls.sheet_names


@st.cache_data(show_spinner=False)
def load_excel(uploaded_file_bytes: bytes, filename: str, sheet_name: str) -> pd.DataFrame:
    df = pd.read_excel(io.BytesIO(uploaded_file_bytes), sheet_name=sheet_name)
    return df


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


def df_to_gdf(df: pd.DataFrame) -> gpd.GeoDataFrame:
    gdf = gpd.GeoDataFrame(
        df.copy(),
        geometry=gpd.points_from_xy(df["LONGITUDE"], df["LATITUDE"]),
        crs="EPSG:4326",
    )
    return gdf.to_crs(epsg=3857)


def create_hexagon(center_x: float, center_y: float, size_m: float) -> Polygon:
    angles_deg = [0, 60, 120, 180, 240, 300]
    points = [
        (
            center_x + size_m * math.cos(math.radians(a)),
            center_y + size_m * math.sin(math.radians(a)),
        )
        for a in angles_deg
    ]
    return Polygon(points)


def build_hex_grid(gdf_3857: gpd.GeoDataFrame, size_m: float) -> gpd.GeoDataFrame:
    minx, miny, maxx, maxy = gdf_3857.total_bounds

    hex_width = 2 * size_m
    hex_height = math.sqrt(3) * size_m
    horiz_spacing = 1.5 * size_m
    vert_spacing = hex_height

    hexes = []
    hex_ids = []

    col = 0
    x = minx - hex_width
    while x < maxx + hex_width:
        y_offset = 0 if col % 2 == 0 else hex_height / 2
        y = miny - hex_height
        while y < maxy + hex_height:
            cy = y + y_offset
            hexes.append(create_hexagon(x, cy, size_m))
            hex_ids.append(f"HX_{col}_{int((cy - miny) // max(1, vert_spacing))}")
            y += vert_spacing
        x += horiz_spacing
        col += 1

    grid = gpd.GeoDataFrame({"hex_id": hex_ids, "geometry": hexes}, crs="EPSG:3857")
    grid = gpd.overlay(
        grid,
        gpd.GeoDataFrame(geometry=[gdf_3857.unary_union.convex_hull], crs="EPSG:3857"),
        how="intersection",
    )
    return grid


@st.cache_data(show_spinner=False)
def assign_points_to_hexes(df_prepared: pd.DataFrame, hex_size_m: int) -> pd.DataFrame:
    gdf = df_to_gdf(df_prepared)
    grid = build_hex_grid(gdf, hex_size_m)

    joined = gpd.sjoin(
        gdf,
        grid[["hex_id", "geometry"]],
        how="inner",
        predicate="within",
    ).drop(columns=["index_right"])

    joined = joined.to_crs(epsg=4326)
    return pd.DataFrame(joined.drop(columns="geometry"))


@st.cache_data(show_spinner=False)
def build_hex_polygons(df_prepared: pd.DataFrame, hex_size_m: int) -> pd.DataFrame:
    gdf = df_to_gdf(df_prepared)
    grid = build_hex_grid(gdf, hex_size_m).to_crs(epsg=4326).copy()

    def polygon_to_coords(poly):
        if poly.geom_type == "Polygon":
            return [[list(coord) for coord in poly.exterior.coords]]
        elif poly.geom_type == "MultiPolygon":
            largest = max(poly.geoms, key=lambda g: g.area)
            return [[list(coord) for coord in largest.exterior.coords]]
        return None

    grid["polygon"] = grid["geometry"].apply(polygon_to_coords)

    grid_3857 = grid.to_crs(epsg=3857)
    centroids_3857 = grid_3857.geometry.centroid
    centroids = gpd.GeoSeries(centroids_3857, crs="EPSG:3857").to_crs(epsg=4326)

    grid["lon"] = centroids.x
    grid["lat"] = centroids.y

    return pd.DataFrame(grid[["hex_id", "polygon", "lon", "lat"]])


def filter_by_time_block(df: pd.DataFrame, block_name: str) -> pd.DataFrame:
    start_h, end_h = TIME_BLOCKS[block_name]
    return df[(df["HORA"] >= start_h) & (df["HORA"] < end_h)].copy()


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


def build_vehicle_trace(df_filtered: pd.DataFrame, linea: int, interno: str, sentido: str) -> pd.DataFrame:
    sub = df_filtered.copy()
    sub["INTERNO"] = sub["INTERNO"].astype(str).str.strip()

    sub = sub[
        (sub["NUM_LINEA"] == linea)
        & (sub["INTERNO"] == str(interno).strip())
        & (sub["SENTIDO"] == sentido)
    ].copy()

    sub = sub.sort_values("DATE_TIME").reset_index(drop=True)
    return sub


def build_vehicle_points_animation_df(trace_df: pd.DataFrame, step_idx: int) -> pd.DataFrame:
    if trace_df.empty:
        return pd.DataFrame(columns=["LONGITUDE", "LATITUDE", "tooltip", "orden", "is_current"])

    sub = trace_df.iloc[: step_idx + 1].copy().reset_index(drop=True)
    sub["orden"] = sub.index + 1
    sub["is_current"] = False
    sub.loc[sub.index[-1], "is_current"] = True

    sub["tooltip"] = (
        "Orden: " + sub["orden"].astype(str)
        + "\nDATE_TIME: " + sub["DATE_TIME"].dt.strftime("%d/%m/%Y %H:%M:%S")
        + "\nCANT_TRAX: " + sub["CANT_TRAX"].fillna(0).round(0).astype(int).astype(str)
    )

    return sub[["LONGITUDE", "LATITUDE", "tooltip", "orden", "is_current"]].copy()


def build_current_position_df(trace_df: pd.DataFrame, step_idx: int) -> pd.DataFrame:
    if trace_df.empty:
        return pd.DataFrame(columns=["LONGITUDE", "LATITUDE", "tooltip"])

    row = trace_df.iloc[[step_idx]].copy()

    row["tooltip"] = (
        "Posición actual"
        + "\nDATE_TIME: " + row["DATE_TIME"].dt.strftime("%d/%m/%Y %H:%M:%S")
        + "\nCANT_TRAX: " + row["CANT_TRAX"].fillna(0).round(0).astype(int).astype(str)
    )

    return row[["LONGITUDE", "LATITUDE", "tooltip"]].copy()


def build_vehicle_points_df(trace_df: pd.DataFrame) -> pd.DataFrame:
    if trace_df.empty:
        return pd.DataFrame(columns=["LONGITUDE", "LATITUDE", "tooltip", "orden"])

    out = trace_df.copy().reset_index(drop=True)
    out["orden"] = out.index + 1
    out["tooltip"] = (
        "Orden: " + out["orden"].astype(str)
        + "\nDATE_TIME: " + out["DATE_TIME"].dt.strftime("%d/%m/%Y %H:%M:%S")
        + "\nCANT_TRAX: " + out["CANT_TRAX"].fillna(0).round(0).astype(int).astype(str)
    )

    return out[["LONGITUDE", "LATITUDE", "orden", "tooltip"]].copy()

def build_vehicle_hourly(trace_df: pd.DataFrame) -> pd.DataFrame:
    if trace_df.empty:
        return pd.DataFrame(columns=["HORA", "trx", "registros"])

    out = (
        trace_df.groupby("HORA", as_index=False)
        .agg(
            trx=("CANT_TRAX", "sum"),
            registros=("DATE_TIME", "count"),
        )
        .sort_values("HORA")
    )
    return out


def to_download_excel(points_df: pd.DataFrame, compare_df: Optional[pd.DataFrame] = None) -> bytes:
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        points_df.to_excel(writer, sheet_name="Puntos_control", index=False)
        if compare_df is not None and not compare_df.empty:
            compare_df.to_excel(writer, sheet_name="Comparacion_hex", index=False)
    buffer.seek(0)
    return buffer.read()


# =========================
# UI
# =========================

st.title("Análisis de competencia entre líneas por hexágonos")
st.caption("Subí tu Excel y analizá competencia o reconstruí el recorrido de un colectivo.")

with st.expander("Carga de archivos y configuración inicial", expanded=True):
    uploaded_file = st.file_uploader(
        "Subir archivo Excel",
        type=["xlsx", "xls"],
        help="El archivo debe contener al menos: FECHA, DATE_TIME, NUM_LINEA, SENTIDO, INTERNO, LONGITUDE, LATITUDE, CANT_TRAX.",
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
    sheets = list_excel_sheets(file_bytes, uploaded_file.name)
    sheet = st.selectbox("Hoja del Excel", sheets, key="sheet_selector")

    raw_df = load_excel(file_bytes, uploaded_file.name, sheet)
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

    col_info1, col_info2 = st.columns(2)

    with col_info1:
        st.metric("Registros válidos", f"{len(df):,}".replace(",", "."))
        st.metric("Hojas detectadas", len(sheets))

    with col_info2:
        if routes_gdf is not None:
            st.metric("Recorridos cargados", f"{len(routes_gdf):,}".replace(",", "."))
        else:
            st.metric("Recorridos cargados", "0")

    with st.expander("Vista previa de la base procesada", expanded=False):
        st.dataframe(df.head(20), use_container_width=True)

    if routes_gdf is not None:
        with st.expander("Vista previa del GeoJSON de recorridos", expanded=False):
            st.dataframe(
                routes_gdf[["linea", "linea_norm", "sentido", "sentido_norm"]].head(20),
                use_container_width=True,
            )

tab_competencia, tab_recorrido = st.tabs([
    "Competencia por hexágonos",
    "Recorrido de un colectivo"
])
# =========================
# SOLAPA 1: COMPETENCIA
# =========================

with tab_competencia:
    st.sidebar.header("Filtros")

    available_dates = sorted(df["FECHA_ONLY"].dropna().unique())
    selected_date = st.sidebar.selectbox("Fecha", available_dates)

    df_f = df[df["FECHA_ONLY"] == selected_date].copy()

    lineas = sorted(df_f["NUM_LINEA"].dropna().astype(int).unique().tolist())
    my_line = st.sidebar.selectbox("Mi línea", lineas)

    default_comp = [x for x in lineas if x != my_line]
    selected_lines = st.sidebar.multiselect(
        "Líneas a comparar",
        options=lineas,
        default=[my_line] + default_comp,
    )

    if my_line not in selected_lines:
        selected_lines = [my_line] + selected_lines

    sentidos = sorted(df_f["SENTIDO"].dropna().unique().tolist())
    selected_sentido = st.sidebar.selectbox("Sentido", ["TODOS"] + sentidos)

    if selected_sentido != "TODOS":
        df_f = df_f[df_f["SENTIDO"] == selected_sentido].copy()

    selected_block = st.sidebar.selectbox("Franja horaria", list(TIME_BLOCKS.keys()))
    df_f = filter_by_time_block(df_f, selected_block)

    selected_hours = st.sidebar.slider("Rango de hora fino", 0, 23, (0, 23))
    df_f = df_f[(df_f["HORA"] >= selected_hours[0]) & (df_f["HORA"] <= selected_hours[1])].copy()

    hex_label = st.sidebar.selectbox("Tamaño de hexágono", list(HEX_SIZE_OPTIONS.keys()), index=1)
    hex_size_m = HEX_SIZE_OPTIONS[hex_label]

    min_trx_hex = st.sidebar.number_input("Mínimo de transacciones totales por hexágono", min_value=1, value=50, step=10)
    min_lines_competing = st.sidebar.number_input("Mínimo de líneas presentes", min_value=1, value=2, step=1)

    show_trx_circles = st.sidebar.checkbox(
        "Mostrar círculos por transacciones totales",
        value=False
    )

    if df_f.empty:
        st.warning("No hay datos para los filtros elegidos.")
        st.stop()

    with st.spinner("Generando hexágonos y métricas..."):
        df_hex = assign_points_to_hexes(df_f, hex_size_m)
        hex_polygons = build_hex_polygons(df_f, hex_size_m)
        points_control = compute_metrics(
            df_hex=df_hex,
            my_line=my_line,
            selected_lines=selected_lines,
            min_trx_hex=int(min_trx_hex),
            min_lines_competing=int(min_lines_competing),
        ).merge(hex_polygons, on="hex_id", how="left")

    evol_total = build_hourly_evolution_total(df_f, selected_lines)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Registros filtrados", f"{len(df_f):,}".replace(",", "."))
    c2.metric("Líneas comparadas", f"{len(selected_lines)}")
    c3.metric("Hexágonos con competencia", f"{points_control['hex_id'].nunique():,}".replace(",", ".") if not points_control.empty else "0")
    c4.metric("Mi línea", str(my_line))

    st.subheader("Mapa de puntos de control")

    if points_control.empty:
        st.warning("No se encontraron hexágonos competitivos con los umbrales elegidos.")
    else:
        map_df = points_control.copy()

        map_df["tooltip"] = (
            "Hex: " + map_df["hex_id"].astype(str)
            + "\nTrx total: " + map_df["trx_total_hex"].round(0).astype(int).astype(str)
            + "\nLíneas presentes: " + map_df["lineas_presentes"].astype(int).astype(str)
            + "\nShare demanda mi línea: " + (map_df["share_demanda"] * 100).round(1).astype(str) + "%"
            + "\nShare oferta mi línea: " + (map_df["share_oferta"] * 100).round(1).astype(str) + "%"
            + "\nÍndice captación: " + map_df["indice_captacion"].round(2).astype(str)
        )

        def color_by_index(x):
            if pd.isna(x):
                return [160, 160, 160, 120]
            elif x < 0.8:
                return [220, 70, 70, 160]
            elif x <= 1.2:
                return [150, 150, 150, 140]
            else:
                return [50, 170, 90, 160]

        map_df["fill_color"] = map_df["indice_captacion"].apply(color_by_index)

        if map_df["trx_total_hex"].max() > 0:
            map_df["circle_radius"] = (
                80 + 320 * (map_df["trx_total_hex"] / map_df["trx_total_hex"].max())
            )
        else:
            map_df["circle_radius"] = 80

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

        view_state = pdk.ViewState(
            latitude=float(center_lat),
            longitude=float(center_lon),
            zoom=10.5,
            pitch=0,
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

        deck = pdk.Deck(
            layers=layers,
            initial_view_state=view_state,
            tooltip={"text": "{tooltip}"},
            map_style="light",
        )

        st.pydeck_chart(deck, use_container_width=True)

        st.markdown(
            """
            **Interpretación de colores**
            - 🔴 Rojo: índice de captación < 0.8
            - ⚪ Gris: índice de captación entre 0.8 y 1.2
            - 🟢 Verde: índice de captación > 1.2
            """
        )

        if show_trx_circles:
            st.markdown("🔵 El tamaño del círculo representa las transacciones totales del hexágono.")

    st.subheader("Ranking de puntos de control")

    show_cols = [
        "hex_id",
        "trx_total_hex",
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
        display_df["share_demanda"] = (display_df["share_demanda"] * 100).round(2)
        display_df["share_oferta"] = (display_df["share_oferta"] * 100).round(2)
        display_df["trx_por_interno"] = display_df["trx_por_interno"].round(2)
        display_df["indice_captacion"] = display_df["indice_captacion"].round(2)
        display_df["score_control"] = display_df["score_control"].round(3)

        st.dataframe(display_df, use_container_width=True, height=420)

    st.subheader("Detalle de un punto de control")

    if not points_control.empty:
        selected_hex = st.selectbox("Elegir hexágono", points_control["hex_id"].tolist())

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

        st.subheader("Evolución horaria total")

        if evol_total.empty:
            st.info("No hay datos para la evolución horaria total.")
        else:
            chart_total = evol_total.pivot(index="HORA", columns="NUM_LINEA", values="trx").fillna(0)
            st.line_chart(chart_total, use_container_width=True)

            trx_total_global = int(df_f[df_f["NUM_LINEA"] == my_line]["CANT_TRAX"].sum())
            st.metric(
                "Transacciones totales del filtro actual",
                f"{trx_total_global:,}".replace(",", ".")
            )

    download_bytes = to_download_excel(display_df, comp_hex)
    st.download_button(
        "Descargar resultados en Excel",
        data=download_bytes,
        file_name=f"competencia_lineas_{selected_date}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

# =========================
# SOLAPA 2: RECORRIDO
# =========================

with tab_recorrido:
    st.subheader("Reconstrucción de recorrido por colectivo")

    if df.empty:
        st.info("No hay datos disponibles.")
    else:
        df_rec = df.copy()

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            fechas_rec = sorted(df_rec["FECHA_ONLY"].dropna().unique())
            selected_date_rec = st.selectbox("Fecha", fechas_rec, key="rec_fecha")

        df_rec = df_rec[df_rec["FECHA_ONLY"] == selected_date_rec].copy()

        with col2:
            lineas_rec = sorted(df_rec["NUM_LINEA"].dropna().astype(int).unique().tolist())
            selected_line_rec = st.selectbox("Línea", lineas_rec, key="rec_linea")

        df_rec = df_rec[df_rec["NUM_LINEA"] == selected_line_rec].copy()

        with col3:
            sentidos_rec = sorted(df_rec["SENTIDO"].dropna().unique().tolist())
            selected_sentido_rec = st.selectbox("Sentido", sentidos_rec, key="rec_sentido")

        df_rec = df_rec[df_rec["SENTIDO"] == selected_sentido_rec].copy()

        with col4:
            internos_rec = sorted(df_rec["INTERNO"].astype(str).str.strip().unique().tolist())
            selected_interno_rec = st.selectbox("Interno", internos_rec, key="rec_interno")

        trace_df = build_vehicle_trace(
            df_filtered=df,
            linea=selected_line_rec,
            interno=selected_interno_rec,
            sentido=selected_sentido_rec,
        )

        if trace_df.empty:
            st.warning("No hay registros para la combinación seleccionada.")
        else:
            st.write(f"Registros del recorrido: {len(trace_df):,}".replace(",", "."))
            vehicle_hourly_df = build_vehicle_hourly(trace_df)

            max_step = len(trace_df) - 1

            step_idx = st.slider(
                "Avance temporal del recorrido",
                min_value=0,
                max_value=max_step,
                value=0,
                step=1,
                key="rec_step"
            )

            current_dt = trace_df.iloc[step_idx]["DATE_TIME"]
            st.markdown(f"**DATE_TIME actual:** {current_dt.strftime('%d/%m/%Y %H:%M:%S')}")

            vehicle_points_df = build_vehicle_points_animation_df(trace_df, step_idx)
            current_position_df = build_current_position_df(trace_df, step_idx)

            center_lat = trace_df["LATITUDE"].mean()
            center_lon = trace_df["LONGITUDE"].mean()

            layers_rec = []

            # recorrido de referencia del geojson
            if routes_gdf is not None:
                routes_ref_df = build_routes_layer_df(
                    routes_gdf=routes_gdf,
                    selected_line=selected_line_rec,
                    selected_sentido=selected_sentido_rec,
                )

                if not routes_ref_df.empty:
                    route_ref_layer = pdk.Layer(
                        "PathLayer",
                        data=routes_ref_df,
                        get_path="path",
                        get_color="color",
                        width_scale=1,
                        width_min_pixels=3,
                        pickable=True,
                    )
                    layers_rec.append(route_ref_layer)

            # puntos ya recorridos hasta el instante actual
            history_layer = pdk.Layer(
                "ScatterplotLayer",
                data=vehicle_points_df,
                get_position="[LONGITUDE, LATITUDE]",
                get_radius=20,
                get_fill_color=[220, 30, 30, 90],
                get_line_color=[120, 0, 0, 120],
                line_width_min_pixels=1,
                stroked=True,
                filled=True,
                pickable=True,
            )

            # posición actual destacada
            current_layer = pdk.Layer(
                "ScatterplotLayer",
                data=current_position_df,
                get_position="[LONGITUDE, LATITUDE]",
                get_radius=60,
                get_fill_color=[0, 102, 255, 180],
                get_line_color=[0, 51, 153, 220],
                line_width_min_pixels=2,
                stroked=True,
                filled=True,
                pickable=True,
            )

            layers_rec.append(history_layer)
            layers_rec.append(current_layer)

            view_state_rec = pdk.ViewState(
                latitude=float(current_position_df["LATITUDE"].iloc[0]),
                longitude=float(current_position_df["LONGITUDE"].iloc[0]),
                zoom=12,
                pitch=0,
            )

            deck_rec = pdk.Deck(
                layers=layers_rec,
                initial_view_state=view_state_rec,
                tooltip={"text": "{tooltip}"},
                map_style="light",
            )

            st.pydeck_chart(deck_rec, use_container_width=True)

            st.markdown(
                f"""
                **Línea:** {selected_line_rec}  
                **Interno:** {selected_interno_rec}  
                **Sentido:** {selected_sentido_rec}
                """
            )

            col_a, col_b = st.columns([1, 1])

            with col_a:
                st.markdown("**Tabla de registros ordenados por DATE_TIME**")
                show_trace = trace_df[
                    ["DATE_TIME", "NUM_LINEA", "SENTIDO", "INTERNO", "LONGITUDE", "LATITUDE", "CANT_TRAX"]
                ].copy()
                st.dataframe(show_trace, use_container_width=True, height=420)

            with col_b:
                st.markdown("**Evolución horaria del interno**")
                if vehicle_hourly_df.empty:
                    st.info("No hay datos horarios.")
                else:
                    chart_vehicle = vehicle_hourly_df.set_index("HORA")[["trx"]]
                    st.line_chart(chart_vehicle, use_container_width=True)

            buffer_rec = io.BytesIO()
            with pd.ExcelWriter(buffer_rec, engine="openpyxl") as writer:
                trace_df.to_excel(writer, sheet_name="Recorrido_interno", index=False)
                vehicle_hourly_df.to_excel(writer, sheet_name="Resumen_horario", index=False)
            buffer_rec.seek(0)

            st.download_button(
                "Descargar recorrido del interno en Excel",
                data=buffer_rec.read(),
                file_name=f"recorrido_linea_{selected_line_rec}_interno_{selected_interno_rec}_{selected_date_rec}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="download_recorrido"
            )

# =========================
# Notas metodológicas
# =========================

with st.expander("Criterio metodológico usado", expanded=False):
    st.markdown(
        """
- La **demanda** se mide con `CANT_TRAX`.
- La **oferta observada** se aproxima con `INTERNO` únicos por hexágono.
- Un **punto de control** es un hexágono donde:
  - está tu línea,
  - hay al menos otra línea,
  - y supera un umbral mínimo de transacciones.
- El **índice de captación** se calcula como:

  `share_demanda / share_oferta`

  Interpretación:
  - mayor a 1: la línea capta más demanda que la oferta que pone
  - menor a 1: pone más oferta de la que capta
        """
    )




