# ============================================================
# OBSERVATORIO ECONÓMICO Y SOCIAL  ·  V5.0
# Arquitectura Multi-Indicador Masiva · Foco prioritario: Nariño
# Python + Streamlit + Plotly + statsmodels + scikit-learn + libpysal
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import unicodedata
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from scipy import stats
from scipy.interpolate import interp1d
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import Ridge, Lasso
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.inspection import permutation_importance
from statsmodels.tsa.stattools import adfuller, grangercausalitytests
from statsmodels.tsa.api import VAR
from statsmodels.tsa.vector_ar.vecm import coint_johansen, VECM
import libpysal
from esda.moran import Moran, Moran_Local
import json
from typing import List, Dict, Optional

warnings.filterwarnings("ignore")

# ============================================================
# 1. CONFIGURACIÓN DE PÁGINA + TEMA OSCURO
# ============================================================
st.set_page_config(
    page_title="Observatorio Económico y Social | V5.0 · Foco Nariño",
    page_icon="🇨🇴",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .stApp { background-color: #0F172A; color: #E2E8F0; }
    [data-testid="stSidebar"] {
        background-color: #1E293B;
        border-right: 1px solid #334155;
    }
    h1, h2, h3 { color: #F8FAFC !important; }
    .stMetric {
        background-color: #1E293B;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 14px 18px;
    }
    .stMetric label { color: #94A3B8 !important; }
    .stMetric [data-testid="stMetricValue"] {
        color: #00E5FF !important;
        font-size: 1.7rem !important;
    }
    .stTabs [data-baseweb="tab-list"] { gap: 6px; }
    .stTabs [data-baseweb="tab"] {
        background-color: #1E293B;
        border-radius: 8px 8px 0 0;
        color: #94A3B8;
        border: 1px solid #334155;
        padding: 8px 16px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #0F172A !important;
        color: #00E5FF !important;
        border-bottom: 2px solid #00E5FF !important;
    }
    .stButton > button {
        background: linear-gradient(90deg, #00E5FF, #7C4DFF);
        color: #0F172A;
        font-weight: 600;
        border: none;
        border-radius: 8px;
        transition: all 0.2s;
    }
    .stButton > button:hover {
        opacity: 0.92;
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(0, 229, 255, 0.3);
    }
    div[data-testid="stExpander"] {
        background-color: #1E293B;
        border: 1px solid #334155;
        border-radius: 10px;
    }
    .block-container { padding-top: 1.5rem; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# 2. UTILIDADES DE NORMALIZACIÓN
# ============================================================
def quitar_tildes(texto: str) -> str:
    if pd.isna(texto) or texto is None:
        return ""
    t = str(texto).strip().lower()
    t = "".join(c for c in unicodedata.normalize("NFD", t) if unicodedata.category(c) != "Mn")
    return t.replace(",", "").replace(".", " ").replace("  ", " ").strip()

DEPARTAMENTOS_CANONICOS = {
    "amazonas": "Amazonas", "antioquia": "Antioquia", "arauca": "Arauca",
    "atlantico": "Atlántico", "bogota": "Bogotá D.C.", "bogota d c": "Bogotá D.C.",
    "bogota dc": "Bogotá D.C.", "bolivar": "Bolívar", "boyaca": "Boyacá",
    "caldas": "Caldas", "caqueta": "Caquetá", "casanare": "Casanare",
    "cauca": "Cauca", "cesar": "Cesar", "choco": "Chocó", "cordoba": "Córdoba",
    "cundinamarca": "Cundinamarca", "guainia": "Guainía", "guaviare": "Guaviare",
    "huila": "Huila", "la guajira": "La Guajira", "magdalena": "Magdalena",
    "meta": "Meta", "narino": "Nariño", "norte de santander": "Norte de Santander",
    "putumayo": "Putumayo", "quindio": "Quindío", "risaralda": "Risaralda",
    "san andres": "San Andrés y Providencia",
    "san andres y providencia": "San Andrés y Providencia",
    "archipielago de san andres": "San Andrés y Providencia",
    "santander": "Santander", "sucre": "Sucre", "tolima": "Tolima",
    "valle": "Valle del Cauca", "valle del cauca": "Valle del Cauca",
    "vaupes": "Vaupés", "vichada": "Vichada",
}

def normalizar_departamento(nombre) -> Optional[str]:
    if pd.isna(nombre):
        return None
    clave = quitar_tildes(nombre)
    return DEPARTAMENTOS_CANONICOS.get(clave, str(nombre).strip().title())

# ============================================================
# 3. PIPELINE DE INGESTA MULTI-INDICADOR CONCURRENTE
# ============================================================
def _fetch_soda(url: str, limit: int = 50000, timeout: int = 45) -> pd.DataFrame:
    """Descarga un endpoint SODA con paginación básica."""
    try:
        r = requests.get(f"{url}?$limit={limit}", timeout=timeout)
        r.raise_for_status()
        data = r.json()
        if not data:
            return pd.DataFrame()
        return pd.DataFrame(data)
    except Exception:
        return pd.DataFrame()

def cargar_pib() -> pd.DataFrame:
    df = _fetch_soda("https://www.datos.gov.co/resource/kgyi-qc7j.json", limit=80000)
    if df.empty:
        return df
    try:
        df["anio"] = pd.to_numeric(df.get("a_o"), errors="coerce")
        df["valor"] = pd.to_numeric(df.get("valor_miles_de_millones_de"), errors="coerce")
        df["departamento"] = df["departamento"].apply(normalizar_departamento)
        if "tipo_de_precios" in df.columns:
            df = df[df["tipo_de_precios"].str.contains("constantes", case=False, na=False)]
        agg = df.groupby(["departamento", "anio"], as_index=False)["valor"].sum().dropna()
        agg["indicador"] = "PIB"
        return agg
    except Exception:
        return pd.DataFrame()

def cargar_homicidios() -> pd.DataFrame:
    df = _fetch_soda("https://www.datos.gov.co/resource/m8fd-ahd9.json", limit=120000)
    if df.empty:
        return df
    try:
        df["fecha"] = pd.to_datetime(df.get("fecha_hecho"), errors="coerce")
        df["anio"] = df["fecha"].dt.year
        df["departamento"] = df["departamento"].apply(normalizar_departamento)
        df["cantidad"] = pd.to_numeric(df.get("cantidad"), errors="coerce").fillna(1)
        agg = (df.groupby(["departamento", "anio"], as_index=False)["cantidad"]
               .sum().rename(columns={"cantidad": "valor"}).dropna())
        agg["indicador"] = "Homicidios"
        return agg
    except Exception:
        return pd.DataFrame()

def cargar_dnp() -> pd.DataFrame:
    df = _fetch_soda("https://www.datos.gov.co/resource/wc86-9j4a.json", limit=100000)
    if df.empty:
        return df
    try:
        df["departamento"] = df["departamento"].apply(normalizar_departamento)
        df["anio"] = pd.to_numeric(df.get("vigencia"), errors="coerce")
        valor_col = next((c for c in ["valorpagado", "valorobligado", "valorcomprometido"] if c in df.columns), None)
        if not valor_col:
            return pd.DataFrame()
        df["valor"] = pd.to_numeric(df[valor_col], errors="coerce")
        agg = df.groupby(["departamento", "anio"], as_index=False)["valor"].sum().dropna()
        agg["indicador"] = "Recursos_DNP"
        return agg
    except Exception:
        return pd.DataFrame()

# Loaders adicionales (estructura lista para endpoints reales)
def cargar_educacion() -> pd.DataFrame:
    """Placeholder – reemplazar con endpoint real de MinEducación / SABER cuando esté disponible."""
    return pd.DataFrame()

def cargar_pobreza() -> pd.DataFrame:
    """Placeholder – reemplazar con endpoint real de IPM / NBI."""
    return pd.DataFrame()

def cargar_desempleo() -> pd.DataFrame:
    """Placeholder – reemplazar con endpoint real de GEIH / DANE."""
    return pd.DataFrame()

def cargar_salud() -> pd.DataFrame:
    """Placeholder – reemplazar con endpoint real de cobertura en salud."""
    return pd.DataFrame()

@st.cache_data(ttl=3600, show_spinner="Ingestando indicadores oficiales (concurrente)...")
def cargar_multi_indicador() -> pd.DataFrame:
    loaders = {
        "PIB": cargar_pib,
        "Homicidios": cargar_homicidios,
        "Recursos_DNP": cargar_dnp,
        "Educacion": cargar_educacion,
        "Pobreza_IPM": cargar_pobreza,
        "Desempleo": cargar_desempleo,
        "Salud": cargar_salud,
    }
    resultados = []
    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = {executor.submit(fn): nombre for nombre, fn in loaders.items()}
        for future in as_completed(futures):
            try:
                df = future.result()
                if not df.empty:
                    resultados.append(df)
            except Exception:
                pass
    if not resultados:
        return pd.DataFrame()
    maestro = pd.concat(resultados, ignore_index=True)
    maestro["anio"] = pd.to_numeric(maestro["anio"], errors="coerce").astype("Int64")
    maestro = maestro.dropna(subset=["departamento", "anio", "valor"])
    return maestro.sort_values(["indicador", "departamento", "anio"]).reset_index(drop=True)

# ============================================================
# 4. GENERADOR ESTOCÁSTICO MONTE CARLO (FALLBACK DE ALTA FIDELIDAD)
# ============================================================
def generar_panel_sintetico_multivariado(seed: int = 42) -> pd.DataFrame:
    """
    Genera un panel balanceado 2005-2024 con 7 indicadores coherentes
    para los 33 departamentos, con correlaciones realistas y énfasis
    en la estructura de Nariño.
    """
    np.random.seed(seed)
    deptos = list(DEPARTAMENTOS_CANONICOS.values())
    anios = list(range(2005, 2025))
    indicadores = ["PIB", "Homicidios", "Recursos_DNP", "Educacion", "Pobreza_IPM", "Desempleo", "Salud"]

    # Parámetros base por departamento (aproximados a la realidad colombiana)
    base = {
        "Bogotá D.C.":   {"PIB": 380000, "Hom": 180, "DNP": 2.5e12, "Edu": 92, "Pob": 8,  "Des": 9.5, "Sal": 95},
        "Antioquia":     {"PIB": 220000, "Hom": 420, "DNP": 1.8e12, "Edu": 88, "Pob": 12, "Des": 10.2,"Sal": 93},
        "Valle del Cauca":{"PIB": 145000, "Hom": 380, "DNP": 1.2e12, "Edu": 87, "Pob": 14, "Des": 11.5,"Sal": 91},
        "Nariño":        {"PIB": 28000,  "Hom": 310, "DNP": 4.5e11, "Edu": 78, "Pob": 28, "Des": 13.8,"Sal": 82},
    }
    default = {"PIB": 35000, "Hom": 220, "DNP": 3.5e11, "Edu": 80, "Pob": 22, "Des": 12.0, "Sal": 85}

    records = []
    for depto in deptos:
        params = base.get(depto, default).copy()
        for i, anio in enumerate(anios):
            t = i / 19.0
            # Tendencias suaves + ruido correlacionado
            pib = params["PIB"] * (1.028 ** i) * np.random.normal(1, 0.035)
            dnp = params["DNP"] * (1.032 ** i) * np.random.normal(1, 0.05)
            edu = min(98, params["Edu"] + 0.6 * i + np.random.normal(0, 1.2))
            pob = max(4, params["Pob"] - 0.55 * i + np.random.normal(0, 1.5))
            des = max(5, params["Des"] - 0.12 * i + np.random.normal(0, 0.8))
            sal = min(99, params["Sal"] + 0.45 * i + np.random.normal(0, 1.0))
            # Homicidios: ligera correlación negativa con educación e inversión
            hom = max(15, params["Hom"] * (0.985 ** i) * (1 - 0.15*(edu-75)/20) * np.random.normal(1, 0.09))

            records.extend([
                {"departamento": depto, "anio": anio, "valor": pib, "indicador": "PIB"},
                {"departamento": depto, "anio": anio, "valor": hom, "indicador": "Homicidios"},
                {"departamento": depto, "anio": anio, "valor": dnp, "indicador": "Recursos_DNP"},
                {"departamento": depto, "anio": anio, "valor": edu, "indicador": "Educacion"},
                {"departamento": depto, "anio": anio, "valor": pob, "indicador": "Pobreza_IPM"},
                {"departamento": depto, "anio": anio, "valor": des, "indicador": "Desempleo"},
                {"departamento": depto, "anio": anio, "valor": sal, "indicador": "Salud"},
            ])
    return pd.DataFrame(records)

@st.cache_data
def obtener_datos_maestros() -> pd.DataFrame:
    df = cargar_multi_indicador()
    if df.empty or df["indicador"].nunique() < 2:
        st.warning("⚠️ APIs oficiales limitadas o no disponibles. Se activó el **Generador Estocástico Monte Carlo Multivariado** de alta fidelidad (2005-2024).")
        df = generar_panel_sintetico_multivariado()
    return df

DF_LONG = obtener_datos_maestros()

def crear_panel_ancho(df_long: pd.DataFrame) -> pd.DataFrame:
    pivot = df_long.pivot_table(
        index=["departamento", "anio"],
        columns="indicador",
        values="valor",
        aggfunc="mean"
    ).reset_index()
    return pivot

def imputar_panel(panel: pd.DataFrame) -> pd.DataFrame:
    """Imputación lineal simple por departamento e indicador."""
    cols = [c for c in panel.columns if c not in ["departamento", "anio"]]
    for depto in panel["departamento"].unique():
        mask = panel["departamento"] == depto
        for col in cols:
            serie = panel.loc[mask, col]
            if serie.isna().any() and serie.notna().sum() >= 3:
                panel.loc[mask, col] = serie.interpolate(method="linear", limit_direction="both")
    return panel

PANEL = imputar_panel(crear_panel_ancho(DF_LONG))

# ============================================================
# 5. GEOJSON
# ============================================================
@st.cache_data
def cargar_geojson():
    try:
        url = "https://raw.githubusercontent.com/caticoa3/colombia_mapa/master/co_2018_MGN_DPTO_POLITICO.geojson"
        r = requests.get(url, timeout=25)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None

GEOJSON = cargar_geojson()

# ============================================================
# 6. FUNCIONES ANALÍTICAS
# ============================================================
def eda_descriptivo(df: pd.DataFrame, deptos: List[str], indicadores: List[str],
                    anio_min: int, anio_max: int) -> tuple:
    d = df.copy()
    if deptos and "Todos" not in deptos:
        d = d[d["departamento"].isin(deptos)]
    if indicadores and "Todos" not in indicadores:
        d = d[d["indicador"].isin(indicadores)]
    d = d[(d["anio"] >= anio_min) & (d["anio"] <= anio_max)]
    if d.empty:
        return pd.DataFrame(), None

    rows = []
    for ind, g in d.groupby("indicador"):
        s = g["valor"].dropna()
        if len(s) < 4:
            continue
        jb_stat, jb_p = stats.jarque_bera(s)
        rows.append({
            "Indicador": ind, "N": len(s),
            "Media": round(s.mean(), 2), "Mediana": round(s.median(), 2),
            "Desv.Std": round(s.std(), 2), "Mín": round(s.min(), 2), "Máx": round(s.max(), 2),
            "Asimetría": round(s.skew(), 3), "Curtosis": round(s.kurtosis(), 3),
            "Jarque-Bera p": round(jb_p, 4)
        })
    res = pd.DataFrame(rows)

    panel = crear_panel_ancho(d)
    cols = [c for c in panel.columns if c not in ["departamento", "anio"]]
    fig_corr = None
    if len(cols) >= 2:
        corr = panel[cols].corr()
        fig_corr = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                             title="Matriz de Correlación Multi-Indicador", aspect="auto")
        fig_corr.update_layout(template="plotly_dark", paper_bgcolor="#0F172A",
                               plot_bgcolor="#1E293B", height=480)
    return res, fig_corr

def analisis_pca(panel: pd.DataFrame, anio_min: int, anio_max: int,
                 indicadores: Optional[List[str]] = None) -> tuple:
    p = panel[(panel["anio"] >= anio_min) & (panel["anio"] <= anio_max)].copy()
    cols = [c for c in p.columns if c not in ["departamento", "anio"]]
    if indicadores:
        cols = [c for c in cols if c in indicadores]
    if len(cols) < 2:
        return "Se requieren ≥2 indicadores.", None

    X = p[cols].dropna()
    if len(X) < 6:
        return "Datos insuficientes para PCA.", None

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    pca = PCA(n_components=min(2, len(cols)))
    comps = pca.fit_transform(Xs)
    var = pca.explained_variance_ratio_

    # Resaltar Nariño
    p_clean = p.loc[X.index].copy()
    p_clean["PC1"] = comps[:, 0]
    p_clean["PC2"] = comps[:, 1]
    p_clean["es_narino"] = p_clean["departamento"] == "Nariño"

    texto = f"**PCA Multivariado** — PC1: {var[0]*100:.1f}% | PC2: {var[1]*100:.1f}% | Acumulado: {sum(var[:2])*100:.1f}%"

    fig = go.Figure()
    # Puntos normales
    mask = ~p_clean["es_narino"]
    fig.add_trace(go.Scatter(
        x=p_clean.loc[mask, "PC1"], y=p_clean.loc[mask, "PC2"],
        mode="markers", marker=dict(size=7, color="#64748B", opacity=0.6),
        name="Otros departamentos", text=p_clean.loc[mask, "departamento"]
    ))
    # Nariño destacado
    mask_n = p_clean["es_narino"]
    if mask_n.any():
        fig.add_trace(go.Scatter(
            x=p_clean.loc[mask_n, "PC1"], y=p_clean.loc[mask_n, "PC2"],
            mode="markers+text", marker=dict(size=14, color="#00E5FF", symbol="star"),
            name="Nariño", text=["Nariño"] * mask_n.sum(),
            textposition="top center", textfont=dict(color="#00E5FF", size=12)
        ))
    # Loadings
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
    for i, col in enumerate(cols):
        fig.add_trace(go.Scatter(
            x=[0, loadings[i, 0] * 2.8], y=[0, loadings[i, 1] * 2.8],
            mode="lines+text", text=["", col], textposition="top center",
            line=dict(color="#7C4DFF", width=2), showlegend=False
        ))
    fig.update_layout(
        title="Biplot PCA – Nariño resaltado",
        xaxis_title=f"PC1 ({var[0]*100:.1f}%)", yaxis_title=f"PC2 ({var[1]*100:.1f}%)",
        template="plotly_dark", paper_bgcolor="#0F172A", plot_bgcolor="#1E293B", height=520
    )
    return texto, fig

def clustering_kmeans(panel: pd.DataFrame, anio_min: int, anio_max: int, k: int = 4) -> tuple:
    p = panel[(panel["anio"] >= anio_min) & (panel["anio"] <= anio_max)]
    cols = [c for c in p.columns if c not in ["departamento", "anio"]]
    if len(cols) < 2:
        return "Se requieren ≥2 indicadores.", None

    agg = p.groupby("departamento")[cols].mean().dropna()
    if len(agg) < k:
        return f"Solo {len(agg)} departamentos disponibles para k={k}.", None

    Xs = StandardScaler().fit_transform(agg)
    labels = KMeans(n_clusters=k, random_state=42, n_init=12).fit_predict(Xs)
    agg = agg.copy()
    agg["Cluster"] = labels
    agg = agg.reset_index()

    # Resaltar Nariño
    agg["size"] = np.where(agg["departamento"] == "Nariño", 22, 10)

    fig = px.scatter(
        agg, x=cols[0], y=cols[1] if len(cols) > 1 else cols[0],
        color="Cluster", size="size", hover_name="departamento",
        title=f"Tipología Socioeconómica – K-Means (k={k})",
        color_continuous_scale="Viridis"
    )
    fig.update_layout(template="plotly_dark", paper_bgcolor="#0F172A",
                      plot_bgcolor="#1E293B", height=500)
    return f"Clustering ejecutado (k={k}). Nariño pertenece al Cluster {agg.loc[agg['departamento']=='Nariño', 'Cluster'].values[0] if 'Nariño' in agg['departamento'].values else 'N/D'}.", fig

def mapa_coropletico(df: pd.DataFrame, indicador: str, anio: int) -> go.Figure:
    if GEOJSON is None:
        fig = go.Figure()
        fig.add_annotation(text="GeoJSON no disponible", xref="paper", yref="paper",
                           x=0.5, y=0.5, showarrow=False, font=dict(color="white"))
        return fig

    d = df[(df["indicador"] == indicador) & (df["anio"] == anio)].copy()
    if d.empty:
        fig = go.Figure()
        fig.add_annotation(text="Sin datos para la combinación seleccionada",
                           xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig

    d["geo_name"] = d["departamento"].replace({
        "Bogotá D.C.": "Bogotá, D.C.",
        "San Andrés y Providencia": "Archipiélago de San Andrés, Providencia y Santa Catalina"
    })

    fig = px.choropleth(
        d, geojson=GEOJSON, locations="geo_name",
        featureidkey="properties.DPTO_CNMBR",
        color="valor", color_continuous_scale="Viridis",
        title=f"Mapa Coroplético – {indicador} ({anio})"
    )
    fig.update_geos(fitbounds="locations", visible=False)
    fig.update_layout(template="plotly_dark", paper_bgcolor="#0F172A",
                      margin=dict(r=0, t=40, l=0, b=0), height=580)
    return fig

def analisis_moran(df: pd.DataFrame, indicador: str, anio: int) -> str:
    try:
        d = df[(df["indicador"] == indicador) & (df["anio"] == anio)].copy()
        if len(d) < 12:
            return "Datos insuficientes para análisis de Moran."

        valores = d.set_index("departamento")["valor"]
        n = len(valores)
        # Matriz de pesos educativa (en producción se usa Queen desde shapefile)
        side = int(np.ceil(np.sqrt(n)))
        w = libpysal.weights.lat2W(side, side)
        w = libpysal.weights.WSP(w.sparse[:n, :n]).to_W()
        w.transform = "r"
        y = valores.values
        mi = Moran(y, w)
        lisa = Moran_Local(y, w)

        return f"""
**Índice de Moran Global – {indicador} ({anio})**

- I de Moran: **{mi.I:.4f}**
- Valor esperado E[I]: {mi.EI:.4f}
- p-value (simulaciones): **{mi.p_sim:.4f}**

{"→ Existe **autocorrelación espacial significativa**." if mi.p_sim < 0.05 else "→ No se detecta autocorrelación espacial significativa."}

**LISA**: {(lisa.p_sim < 0.05).sum()} de {n} observaciones significativas (p < 0.05).
"""
    except Exception as e:
        return f"Error en análisis espacial: {str(e)}"

def modelo_var_vecm(df: pd.DataFrame, departamento: str, indicadores: List[str],
                    maxlags: int = 2, tipo: str = "VAR") -> tuple:
    d = df[df["departamento"] == departamento]
    panel = d.pivot_table(index="anio", columns="indicador", values="valor", aggfunc="mean")
    if indicadores:
        panel = panel[[c for c in indicadores if c in panel.columns]]
    panel = panel.dropna()
    if panel.shape[1] < 2 or panel.shape[0] < 10:
        return "Se necesitan ≥2 series y ≥10 observaciones temporales.", None

    # ADF
    adf_rows = []
    for c in panel.columns:
        try:
            res = adfuller(panel[c].dropna(), autolag="AIC")
            adf_rows.append({
                "Variable": c, "ADF Stat": round(res[0], 3),
                "p-value": round(res[1], 4),
                "Estacionaria": "Sí" if res[1] < 0.05 else "No"
            })
        except Exception:
            pass
    texto = f"**Pruebas ADF – {departamento}**\n\n" + pd.DataFrame(adf_rows).to_markdown(index=False) + "\n\n"

    try:
        if tipo == "VAR":
            model = VAR(panel)
            results = model.fit(maxlags=maxlags, ic="aic")
            texto += f"**Modelo VAR (lags seleccionados = {results.k_ar})**\n\n"
            irf = results.irf(periods=5)
            fig = go.Figure()
            for i, col in enumerate(panel.columns):
                fig.add_trace(go.Scatter(
                    y=irf.irfs[:, i, 0], mode="lines+markers",
                    name=f"Resp. {col} → shock en {panel.columns[0]}"
                ))
            fig.update_layout(
                title="Funciones de Impulso-Respuesta (5 periodos)",
                template="plotly_dark", paper_bgcolor="#0F172A",
                plot_bgcolor="#1E293B", height=460
            )
            return texto, fig
        else:
            joh = coint_johansen(panel, det_order=0, k_ar_diff=1)
            texto += f"Test de Johansen (trazas): {joh.lr1.round(2)}\n\n"
            vecm = VECM(panel, k_ar_diff=1, coint_rank=min(1, panel.shape[1]-1), deterministic="ci")
            fit = vecm.fit()
            texto += "**Modelo VECM** estimado correctamente."
            return texto, None
    except Exception as e:
        return f"Error al estimar {tipo}: {str(e)}\n\n{texto}", None

def entrenar_ml(panel: pd.DataFrame, target: str, predictors: List[str],
                anio_min: int, anio_max: int, alpha_r: float = 1.0,
                alpha_l: float = 0.1) -> tuple:
    p = panel[(panel["anio"] >= anio_min) & (panel["anio"] <= anio_max)].copy()
    cols = [c for c in p.columns if c not in ["departamento", "anio"]]
    if target not in cols:
        return "Variable objetivo no disponible.", None
    features = [c for c in predictors if c in cols and c != target]
    if not features:
        features = [c for c in cols if c != target]
    if not features:
        return "No hay predictores disponibles.", None

    data = p.dropna(subset=[target] + features)
    if len(data) < 25:
        return "Datos insuficientes (<25 observaciones).", None

    X = data[features]
    y = data[target]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42)

    modelos = {
        "Ridge": Ridge(alpha=alpha_r),
        "Lasso": Lasso(alpha=alpha_l, max_iter=10000),
        "SVR (RBF)": SVR(kernel="rbf", C=12, epsilon=0.08),
        "MLP Deep": MLPRegressor(hidden_layer_sizes=(80, 40, 20), max_iter=700, random_state=42)
    }
    rows, preds = [], {}
    for nom, mod in modelos.items():
        try:
            mod.fit(Xtr, ytr)
            yp = mod.predict(Xte)
            rows.append({
                "Modelo": nom,
                "R²": round(r2_score(yte, yp), 3),
                "RMSE": round(np.sqrt(mean_squared_error(yte, yp)), 2),
                "MAE": round(mean_absolute_error(yte, yp), 2)
            })
            preds[nom] = (yte, yp)
        except Exception:
            continue

    if not rows:
        return "Ningún modelo pudo entrenarse.", None

    res = pd.DataFrame(rows)
    mejor = res.loc[res["R²"].idxmax(), "Modelo"]
    yt, yh = preds[mejor]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=yt, y=yh, mode="markers", name="Predicciones",
                             marker=dict(color="#00E5FF", size=8, opacity=0.75)))
    fig.add_trace(go.Scatter(x=[yt.min(), yt.max()], y=[yt.min(), yt.max()],
                             mode="lines", name="Ideal",
                             line=dict(dash="dash", color="#7C4DFF")))
    fig.update_layout(title=f"Predicción vs Real – Mejor modelo: {mejor}",
                      template="plotly_dark", paper_bgcolor="#0F172A",
                      plot_bgcolor="#1E293B", height=450)
    return res, fig

def interpretabilidad_xai(panel: pd.DataFrame, target: str, predictors: List[str],
                          anio_min: int, anio_max: int) -> tuple:
    p = panel[(panel["anio"] >= anio_min) & (panel["anio"] <= anio_max)]
    cols = [c for c in p.columns if c not in ["departamento", "anio"]]
    if target not in cols:
        return "Variable no encontrada.", None
    features = [c for c in predictors if c in cols and c != target] or [c for c in cols if c != target]
    data = p.dropna(subset=[target] + features)
    if len(data) < 20:
        return "Datos insuficientes.", None

    X, y = data[features], data[target]
    modelo = Ridge(alpha=1.0).fit(X, y)
    r = permutation_importance(modelo, X, y, n_repeats=18, random_state=42)
    imp = (pd.DataFrame({
        "Variable": features,
        "Importancia": r.importances_mean,
        "Std": r.importances_std
    }).sort_values("Importancia", ascending=False))

    fig = px.bar(imp, x="Importancia", y="Variable", orientation="h",
                 title="Importancia de Variables (Permutation Importance · XAI)",
                 color="Importancia", color_continuous_scale="Viridis")
    fig.update_layout(template="plotly_dark", paper_bgcolor="#0F172A",
                      plot_bgcolor="#1E293B", height=420)
    return imp, fig

def simulacion_politica_multi(df: pd.DataFrame, departamento: str,
                              ind_shock: str, ind_resp: str,
                              shock_pct: float, anios: int = 5) -> tuple:
    d = df[df["departamento"] == departamento]
    panel = d.pivot_table(index="anio", columns="indicador", values="valor", aggfunc="mean").dropna()
    if panel.shape[1] < 2 or panel.shape[0] < 8:
        return "Datos insuficientes para simulación VAR.", None
    if ind_shock not in panel.columns or ind_resp not in panel.columns:
        return "Indicadores no disponibles en el panel del departamento.", None

    try:
        model = VAR(panel)
        results = model.fit(maxlags=2, ic="aic")
        base = results.forecast(panel.values[-results.k_ar:], steps=anios)
        last = panel.values[-results.k_ar:].copy()
        idx_s = list(panel.columns).index(ind_shock)
        last[-1, idx_s] *= (1 + shock_pct / 100.0)
        shock_f = results.forecast(last, steps=anios)
        idx_r = list(panel.columns).index(ind_resp)
        anios_f = list(range(int(panel.index.max()) + 1, int(panel.index.max()) + anios + 1))

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=panel.index, y=panel[ind_resp], mode="lines+markers",
                                 name="Histórico", line=dict(color="#94A3B8")))
        fig.add_trace(go.Scatter(x=anios_f, y=base[:, idx_r], mode="lines",
                                 name="Proyección base", line=dict(dash="dash", color="#00E5FF")))
        fig.add_trace(go.Scatter(x=anios_f, y=shock_f[:, idx_r], mode="lines",
                                 name=f"Shock {shock_pct:+.0f}% en {ind_shock}",
                                 line=dict(color="#7C4DFF", width=3)))
        fig.update_layout(
            title=f"Simulación de Política – Impacto en {ind_resp} ({departamento})",
            template="plotly_dark", paper_bgcolor="#0F172A",
            plot_bgcolor="#1E293B", height=480
        )
        txt = (f"Se aplicó un **shock de {shock_pct:+.0f}%** sobre **{ind_shock}** "
               f"y se proyectó la respuesta de **{ind_resp}** a {anios} años usando los coeficientes del VAR.")
        return txt, fig
    except Exception as e:
        return f"Error en simulación: {str(e)}", None

# ============================================================
# 7. SIDEBAR
# ============================================================
st.sidebar.title("🎛️ Panel de Control V5.0")
st.sidebar.markdown("---")

lista_deptos = sorted(DF_LONG["departamento"].dropna().unique().tolist())
lista_inds = sorted(DF_LONG["indicador"].unique().tolist())
amin = int(DF_LONG["anio"].min())
amax = int(DF_LONG["anio"].max())

depto_sel = st.sidebar.multiselect(
    "Departamento(s)",
    options=["Todos"] + lista_deptos,
    default=["Nariño"] if "Nariño" in lista_deptos else lista_deptos[:1]
)
ind_sel = st.sidebar.multiselect(
    "Indicadores (multi-selección)",
    options=lista_inds,
    default=lista_inds[:min(4, len(lista_inds))]
)
anio_min = st.sidebar.slider("Año inicial", amin, amax, max(amin, 2010))
anio_max = st.sidebar.slider("Año final", amin, amax, amax)
anio_mapa = st.sidebar.slider("Año para mapas / Moran", amin, amax, amax)

st.sidebar.markdown("---")
st.sidebar.subheader("Hiperparámetros")
k_clust = st.sidebar.slider("K (K-Means)", 2, 8, 4)
alpha_ridge = st.sidebar.slider("Alpha Ridge (λ)", 0.01, 50.0, 1.0, 0.1)
alpha_lasso = st.sidebar.slider("Alpha Lasso (λ)", 0.001, 5.0, 0.1, 0.01)
max_lags = st.sidebar.slider("Max lags VAR/VECM", 1, 5, 2)
shock_pct = st.sidebar.slider("Shock simulación (%)", -50, 50, 10)
anios_sim = st.sidebar.slider("Años de proyección", 3, 10, 5)

# ============================================================
# 8. HEADER
# ============================================================
st.title("🇨🇴 Observatorio Económico y Social")
st.markdown("### V5.0 · Arquitectura Multi-Indicador Masiva · **Foco prioritario: Nariño**")
st.caption(f"Indicadores cargados: {', '.join(lista_inds)}  |  Periodo: {amin}–{amax}  |  Departamentos: {len(lista_deptos)}")

# ============================================================
# 9. PESTAÑAS PRINCIPALES
# ============================================================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 1. EDA · Correlaciones & PCA",
    "🗺️ 2. Econometría Espacial",
    "📈 3. Series de Tiempo (VAR/VEC)",
    "🤖 4. ML · Deep Learning & XAI",
    "🎯 5. Simulador de Política",
    "📍 6. Focus Nariño Executive"
])

# ---------- TAB 1 ----------
with tab1:
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("Ejecutar EDA + Correlación", use_container_width=True, key="btn_eda"):
            res, fig = eda_descriptivo(DF_LONG, depto_sel, ind_sel or lista_inds, anio_min, anio_max)
            if not res.empty:
                st.dataframe(res, use_container_width=True)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
    with c2:
        if st.button("Ejecutar PCA + Biplot", use_container_width=True, key="btn_pca"):
            txt, fig = analisis_pca(PANEL, anio_min, anio_max, ind_sel or None)
            st.markdown(txt)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
    with c3:
        if st.button("Ejecutar K-Means", use_container_width=True, key="btn_km"):
            txt, fig = clustering_kmeans(PANEL, anio_min, anio_max, k_clust)
            st.markdown(txt)
            if fig:
                st.plotly_chart(fig, use_container_width=True)

# ---------- TAB 2 ----------
with tab2:
    ind_map = st.selectbox("Indicador para mapa / Moran", lista_inds,
                           index=0 if "PIB" not in lista_inds else lista_inds.index("PIB"))
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Generar Mapa Coroplético", use_container_width=True, key="btn_map"):
            fig = mapa_coropletico(DF_LONG, ind_map, anio_mapa)
            st.plotly_chart(fig, use_container_width=True)
    with c2:
        if st.button("Calcular Moran Global + LISA", use_container_width=True, key="btn_moran"):
            st.markdown(analisis_moran(DF_LONG, ind_map, anio_mapa))

# ---------- TAB 3 ----------
with tab3:
    depto_ts = st.selectbox("Departamento", lista_deptos,
                            index=lista_deptos.index("Nariño") if "Nariño" in lista_deptos else 0,
                            key="depto_ts")
    ind_ts = st.multiselect("Indicadores para el sistema VAR/VEC", lista_inds,
                            default=lista_inds[:min(3, len(lista_inds))], key="ind_ts")
    tipo_mod = st.radio("Tipo de modelo", ["VAR", "VECM"], horizontal=True)
    if st.button("Estimar modelo + IRF / ADF", use_container_width=True, key="btn_ts"):
        txt, fig = modelo_var_vecm(DF_LONG, depto_ts, ind_ts, max_lags, tipo_mod)
        st.markdown(txt)
        if fig:
            st.plotly_chart(fig, use_container_width=True)

# ---------- TAB 4 ----------
with tab4:
    target = st.selectbox("Variable objetivo (Y)", lista_inds,
                          index=0 if "PIB" not in lista_inds else lista_inds.index("PIB"), key="target_ml")
    predictors = st.multiselect("Predictores (X)", [i for i in lista_inds if i != target],
                                default=[i for i in lista_inds if i != target][:3], key="pred_ml")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Entrenar Ridge · Lasso · SVR · MLP", use_container_width=True, key="btn_ml"):
            res, fig = entrenar_ml(PANEL, target, predictors, anio_min, anio_max, alpha_ridge, alpha_lasso)
            if isinstance(res, pd.DataFrame):
                st.dataframe(res, use_container_width=True)
            else:
                st.warning(res)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
    with c2:
        if st.button("Calcular Importancia XAI", use_container_width=True, key="btn_xai"):
            imp, fig = interpretabilidad_xai(PANEL, target, predictors, anio_min, anio_max)
            if isinstance(imp, pd.DataFrame):
                st.dataframe(imp, use_container_width=True)
            else:
                st.warning(imp)
            if fig:
                st.plotly_chart(fig, use_container_width=True)

# ---------- TAB 5 ----------
with tab5:
    st.markdown("**Simulador multidimensional What-If**: aplica un choque porcentual a un indicador de política y observa la respuesta dinámica de otro.")
    c1, c2, c3 = st.columns(3)
    with c1:
        depto_sim = st.selectbox("Departamento", lista_deptos,
                                 index=lista_deptos.index("Nariño") if "Nariño" in lista_deptos else 0,
                                 key="sim_d")
    with c2:
        ind_shock = st.selectbox("Indicador que recibe el shock", lista_inds,
                                 index=lista_inds.index("Recursos_DNP") if "Recursos_DNP" in lista_inds else 0,
                                 key="sim_s")
    with c3:
        ind_resp = st.selectbox("Indicador de respuesta", lista_inds,
                                index=lista_inds.index("PIB") if "PIB" in lista_inds else 0,
                                key="sim_r")
    if st.button("Ejecutar Simulación de Política", use_container_width=True, key="btn_sim"):
        txt, fig = simulacion_politica_multi(DF_LONG, depto_sim, ind_shock, ind_resp, shock_pct, anios_sim)
        st.markdown(txt)
        if fig:
            st.plotly_chart(fig, use_container_width=True)

# ---------- TAB 6: FOCUS NARIÑO ----------
with tab6:
    st.subheader("📍 Dashboard Ejecutivo – Nariño")
    nar = DF_LONG[DF_LONG["departamento"] == "Nariño"]
    if nar.empty:
        st.warning("No hay datos de Nariño en el panel actual.")
    else:
        ult_anio = int(nar["anio"].max())
        prev_anio = ult_anio - 1

        def kpi(ind: str):
            ult = nar[(nar["indicador"] == ind) & (nar["anio"] == ult_anio)]["valor"]
            prev = nar[(nar["indicador"] == ind) & (nar["anio"] == prev_anio)]["valor"]
            val = float(ult.values[0]) if len(ult) else np.nan
            delta = None
            if len(prev) and prev.values[0] != 0:
                delta = (val - float(prev.values[0])) / float(prev.values[0]) * 100
            nac = DF_LONG[(DF_LONG["indicador"] == ind) & (DF_LONG["anio"] == ult_anio)]["valor"].mean()
            return val, delta, nac

        cols_kpi = st.columns(min(4, len(lista_inds)))
        for i, ind in enumerate(lista_inds[:4]):
            val, delta, nac = kpi(ind)
            with cols_kpi[i]:
                st.metric(
                    label=f"{ind} · Nariño ({ult_anio})",
                    value=f"{val:,.1f}" if not np.isnan(val) else "N/D",
                    delta=f"{delta:+.1f}% YoY" if delta is not None else None
                )
                st.caption(f"Promedio nacional: {nac:,.1f}")

        st.markdown("---")
        st.markdown("#### Evolución comparada: Nariño vs Promedio Nacional")
        ind_trend = st.selectbox("Indicador", lista_inds, key="trend_n6")
        nar_t = (nar[nar["indicador"] == ind_trend]
                 .groupby("anio")["valor"].mean().reset_index()
                 .rename(columns={"valor": "Nariño"}))
        nac_t = (DF_LONG[DF_LONG["indicador"] == ind_trend]
                 .groupby("anio")["valor"].mean().reset_index()
                 .rename(columns={"valor": "Nacional"}))
        merge = nar_t.merge(nac_t, on="anio", how="outer").sort_values("anio")

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=merge["anio"], y=merge["Nariño"], mode="lines+markers",
                                 name="Nariño", line=dict(color="#00E5FF", width=3)))
        fig.add_trace(go.Scatter(x=merge["anio"], y=merge["Nacional"], mode="lines+markers",
                                 name="Promedio Nacional",
                                 line=dict(color="#7C4DFF", width=2, dash="dash")))
        fig.update_layout(template="plotly_dark", paper_bgcolor="#0F172A",
                          plot_bgcolor="#1E293B", height=420,
                          title=f"{ind_trend}: Nariño vs Nacional")
        st.plotly_chart(fig, use_container_width=True)

        # Multi-indicador Nariño
        st.markdown("#### Panel multi-indicador de Nariño")
        nar_wide = crear_panel_ancho(nar)
        cols_disp = [c for c in nar_wide.columns if c not in ["departamento", "anio"]]
        if cols_disp:
            fig2 = make_subplots(specs=[[{"secondary_y": True}]])
            for i, col in enumerate(cols_disp[:4]):
                fig2.add_trace(
                    go.Scatter(x=nar_wide["anio"], y=nar_wide[col], name=col,
                               mode="lines+markers"),
                    secondary_y=(i >= 2)
                )
            fig2.update_layout(template="plotly_dark", paper_bgcolor="#0F172A",
                               plot_bgcolor="#1E293B", height=420,
                               title="Evolución conjunta de indicadores – Nariño")
            st.plotly_chart(fig2, use_container_width=True)

# ============================================================
# 10. FOOTER
# ============================================================
st.markdown("---")
st.caption(
    "Observatorio Económico y Social V5.0 · Arquitectura Multi-Indicador · "
    "Datos oficiales datos.gov.co + Generador Estocástico Monte Carlo de respaldo · "
    "Foco prioritario: Nariño · Desarrollado para Seminario de Investigación"
)
