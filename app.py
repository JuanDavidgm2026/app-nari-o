# ============================================================
# OBSERVATORIO ECONÓMICO Y SOCIAL · STREAMLIT
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import requests
import unicodedata
import warnings
from concurrent.futures import ThreadPoolExecutor
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import Ridge, Lasso
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.inspection import permutation_importance
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.api import VAR
from statsmodels.tsa.vector_ar.vecm import coint_johansen
import libpysal
from esda.moran import Moran, Moran_Local

warnings.filterwarnings("ignore")

# ============================================================
# CONFIG + TEMA ALTO CONTRASTE
# ============================================================
st.set_page_config(
    page_title="Observatorio Económico y Social",
    page_icon="🇨🇴",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.stApp { background-color: #0B1220 !important; color: #F1F5F9 !important; }

section[data-testid="stSidebar"] {
    background-color: #111827 !important;
    border-right: 1px solid #1F2937;
}
section[data-testid="stSidebar"] * { color: #E5E7EB !important; }
section[data-testid="stSidebar"] .stMarkdown h1,
section[data-testid="stSidebar"] .stMarkdown h2,
section[data-testid="stSidebar"] .stMarkdown h3 { color: #22D3EE !important; }

h1, h2, h3, h4 { color: #22D3EE !important; font-weight: 700 !important; }
.stCaption, [data-testid="stCaption"] { color: #94A3B8 !important; }
p, span, label, .stMarkdown, .stText { color: #E2E8F0 !important; }

div[data-baseweb="select"] > div {
    background-color: #1F2937 !important;
    border: 1px solid #374151 !important;
    color: #F8FAFC !important;
}
.stSlider [data-baseweb="slider"] div[role="slider"] {
    background-color: #22D3EE !important;
}

.stButton > button {
    background: linear-gradient(90deg, #0891B2 0%, #7C3AED 100%) !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    box-shadow: 0 2px 8px rgba(8, 145, 178, 0.35);
}
.stButton > button:hover {
    filter: brightness(1.12);
    box-shadow: 0 4px 14px rgba(124, 58, 237, 0.45);
}

button[data-baseweb="tab"] {
    color: #94A3B8 !important;
    font-weight: 600 !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: #22D3EE !important;
    border-bottom: 3px solid #22D3EE !important;
}

div[data-testid="stMetric"] {
    background-color: #1E293B;
    border: 1px solid #334155;
    border-radius: 10px;
    padding: 12px 14px;
}
div[data-testid="stMetric"] label { color: #94A3B8 !important; }
div[data-testid="stMetric"] [data-testid="stMetricValue"] { color: #F8FAFC !important; }

.stAlert {
    background-color: #1E293B !important;
    color: #F1F5F9 !important;
    border: 1px solid #334155 !important;
}
[data-testid="stDataFrame"] {
    border: 1px solid #334155;
    border-radius: 8px;
}
hr { border-color: #334155 !important; }
footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

PLOT_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="#0B1220",
    plot_bgcolor="#111827",
    font=dict(color="#E2E8F0", size=13),
    title_font=dict(color="#22D3EE", size=16),
)

# ============================================================
# NORMALIZACIÓN
# ============================================================
def quitar_tildes(texto):
    if pd.isna(texto) or texto is None:
        return ""
    t = str(texto).strip().lower()
    t = "".join(c for c in unicodedata.normalize("NFD", t) if unicodedata.category(c) != "Mn")
    return t.replace(",", "").replace(".", " ").replace("  ", " ").strip()

DEPARTAMENTOS_CANONICOS = {
    "amazonas": "Amazonas", "antioquia": "Antioquia", "arauca": "Arauca",
    "atlantico": "Atlántico", "bogota": "Bogotá D.C.", "bogota d c": "Bogotá D.C.",
    "bogota dc": "Bogotá D.C.", "bogota, d c": "Bogotá D.C.",
    "bolivar": "Bolívar", "boyaca": "Boyacá", "caldas": "Caldas", "caqueta": "Caquetá",
    "casanare": "Casanare", "cauca": "Cauca", "cesar": "Cesar", "choco": "Chocó",
    "cordoba": "Córdoba", "cundinamarca": "Cundinamarca", "guainia": "Guainía",
    "guaviare": "Guaviare", "huila": "Huila", "la guajira": "La Guajira",
    "magdalena": "Magdalena", "meta": "Meta", "narino": "Nariño",
    "norte de santander": "Norte de Santander", "putumayo": "Putumayo",
    "quindio": "Quindío", "risaralda": "Risaralda",
    "san andres": "San Andrés y Providencia",
    "san andres y providencia": "San Andrés y Providencia",
    "archipielago de san andres": "San Andrés y Providencia",
    "santander": "Santander", "sucre": "Sucre", "tolima": "Tolima",
    "valle": "Valle del Cauca", "valle del cauca": "Valle del Cauca",
    "vaupes": "Vaupés", "vichada": "Vichada",
}

def normalizar_departamento(nombre):
    if pd.isna(nombre):
        return None
    return DEPARTAMENTOS_CANONICOS.get(quitar_tildes(nombre), str(nombre).strip().title())

def _match_geo_name(depto):
    mapa = {
        "Bogotá D.C.": "Bogotá, D.C.",
        "Bogotá": "Bogotá, D.C.",
        "San Andrés y Providencia": "Archipiélago de San Andrés, Providencia y Santa Catalina",
    }
    return mapa.get(depto, depto)

# ============================================================
# CARGA DATOS REALES
# ============================================================
def _fetch(url, limit=100000, timeout=70):
    try:
        r = requests.get(f"{url}?$limit={limit}", timeout=timeout)
        r.raise_for_status()
        data = r.json()
        return pd.DataFrame(data) if data else pd.DataFrame()
    except Exception as e:
        print(f"  Error fetch: {e}")
        return pd.DataFrame()

def cargar_pib():
    df = _fetch("https://www.datos.gov.co/resource/kgyi-qc7j.json", 80000)
    if df.empty:
        return pd.DataFrame()
    try:
        df["anio"] = pd.to_numeric(df.get("a_o"), errors="coerce")
        df["valor"] = pd.to_numeric(df.get("valor_miles_de_millones_de"), errors="coerce")
        df["departamento"] = df["departamento"].apply(normalizar_departamento)
        if "tipo_de_precios" in df.columns:
            df = df[df["tipo_de_precios"].astype(str).str.contains("constantes", case=False, na=False)]
        agg = df.groupby(["departamento", "anio"], as_index=False)["valor"].sum().dropna()
        agg["indicador"] = "PIB"
        return agg
    except Exception:
        return pd.DataFrame()

def cargar_homicidios():
    df = _fetch("https://www.datos.gov.co/resource/m8fd-ahd9.json", 150000)
    if df.empty:
        return pd.DataFrame()
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

def cargar_dnp():
    df = _fetch("https://www.datos.gov.co/resource/wc86-9j4a.json", 100000)
    if df.empty:
        return pd.DataFrame()
    try:
        df["departamento"] = df["departamento"].apply(normalizar_departamento)
        df["anio"] = pd.to_numeric(df.get("vigencia"), errors="coerce")
        col = next((c for c in ["valorpagado", "valorobligado", "valorcomprometido"] if c in df.columns), None)
        if not col:
            return pd.DataFrame()
        df["valor"] = pd.to_numeric(df[col], errors="coerce")
        agg = df.groupby(["departamento", "anio"], as_index=False)["valor"].sum().dropna()
        agg["indicador"] = "Recursos_DNP"
        return agg
    except Exception:
        return pd.DataFrame()

def cargar_educacion():
    df = _fetch("https://www.datos.gov.co/resource/ji8i-4anb.json", 5000)
    if df.empty:
        return pd.DataFrame()
    try:
        col_anio = next((c for c in ["ano", "a_o", "anio", "year"] if c in df.columns), None)
        col_dep = next((c for c in ["departamento", "nombre_departamento"] if c in df.columns), None)
        if not col_anio or not col_dep:
            return pd.DataFrame()
        df["anio"] = pd.to_numeric(df[col_anio], errors="coerce")
        df["departamento"] = df[col_dep].apply(normalizar_departamento)
        records = []
        for api_col, nombre in [
            ("cobertura_neta", "Cobertura_Neta_Educacion"),
            ("tasa_matriculacion_5_16", "Tasa_Matriculacion_5_16"),
            ("cobertura_bruta", "Cobertura_Bruta_Educacion"),
        ]:
            if api_col in df.columns:
                t = df[["departamento", "anio", api_col]].copy()
                t["valor"] = pd.to_numeric(t[api_col], errors="coerce")
                t = t.dropna(subset=["departamento", "anio", "valor"])
                if t["valor"].max() <= 1.5:
                    t["valor"] = t["valor"] * 100
                t["indicador"] = nombre
                records.append(t[["departamento", "anio", "valor", "indicador"]])
        if not records:
            return pd.DataFrame()
        return pd.concat(records, ignore_index=True)
    except Exception:
        return pd.DataFrame()

def cargar_matricula_superior():
    df = _fetch("https://www.datos.gov.co/resource/4hrb-y62g.json", 5000)
    if df.empty:
        return pd.DataFrame()
    try:
        col_anio = next((c for c in ["a_o", "ano", "anio"] if c in df.columns), None)
        col_dep = next((c for c in ["nombre_del_departamento", "departamento", "nombre_departamento"] if c in df.columns), None)
        if not col_anio or not col_dep:
            return pd.DataFrame()
        df["anio"] = pd.to_numeric(df[col_anio], errors="coerce")
        df["departamento"] = df[col_dep].apply(normalizar_departamento)
        niveles = [c for c in ["tecnica_profesional", "tecnologica", "universitaria",
                               "especializacion", "maestria", "doctorado"] if c in df.columns]
        if niveles:
            for c in niveles:
                df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
            df["valor"] = df[niveles].sum(axis=1)
        else:
            nums = [c for c in df.select_dtypes(include=[np.number]).columns if c not in ["anio"]]
            if not nums:
                return pd.DataFrame()
            df["valor"] = pd.to_numeric(df[nums[0]], errors="coerce")
        agg = df.groupby(["departamento", "anio"], as_index=False)["valor"].sum().dropna()
        agg["indicador"] = "Matricula_Superior"
        return agg
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=3600, show_spinner="Cargando datos reales de datos.gov.co…")
def cargar_todo():
    with ThreadPoolExecutor(max_workers=5) as ex:
        partes = [f.result() for f in [
            ex.submit(cargar_pib),
            ex.submit(cargar_homicidios),
            ex.submit(cargar_dnp),
            ex.submit(cargar_educacion),
            ex.submit(cargar_matricula_superior),
        ]]
    partes = [p for p in partes if p is not None and not p.empty]
    if not partes:
        raise RuntimeError("No se cargó ningún indicador real.")
    df = pd.concat(partes, ignore_index=True)
    df["anio"] = pd.to_numeric(df["anio"], errors="coerce")
    df = df.dropna(subset=["departamento", "anio", "valor"])
    df["anio"] = df["anio"].astype(int)
    return df.reset_index(drop=True)

@st.cache_data(ttl=3600)
def cargar_geojson():
    try:
        return requests.get(
            "https://raw.githubusercontent.com/caticoa3/colombia_mapa/master/co_2018_MGN_DPTO_POLITICO.geojson",
            timeout=25
        ).json()
    except Exception:
        return None

def crear_panel_ancho(df):
    return df.pivot_table(index=["departamento", "anio"], columns="indicador",
                          values="valor", aggfunc="sum").reset_index()

# ============================================================
# HELPERS
# ============================================================
def filtrar(df, deptos=None, inds=None, a1=None, a2=None):
    d = df.copy()
    if deptos and "Todos" not in deptos:
        d = d[d["departamento"].isin(deptos)]
    if inds:
        d = d[d["indicador"].isin(inds)]
    if a1 is not None:
        d = d[d["anio"] >= a1]
    if a2 is not None:
        d = d[d["anio"] <= a2]
    return d

def mejor_anio(df_long, ind, anio_pref):
    d = df_long[df_long["indicador"] == ind]
    if d.empty:
        return None
    if (d["anio"] == anio_pref).any():
        return anio_pref
    return int(d["anio"].max())

def lags_seguros(nobs, neqs, solicitado=2):
    if nobs <= neqs + 2:
        return 1
    safe = max(1, (nobs - neqs - 1) // max(neqs, 1))
    return max(1, min(int(solicitado), safe, 3))

def style_fig(fig):
    if fig is not None:
        fig.update_layout(**PLOT_LAYOUT)
    return fig

# ============================================================
# MÓDULOS ANALÍTICOS
# ============================================================
def eda(DF_LONG, PANEL, lista_inds, deptos, inds, a1, a2):
    d = filtrar(DF_LONG, deptos, inds or lista_inds, a1, a2)
    info = ""
    if d.empty:
        d = filtrar(DF_LONG, deptos, inds or lista_inds, None, None)
        if d.empty:
            return "⚠️ No hay datos para esa combinación.", None
        info = f"*(Rango ampliado: {int(d.anio.min())}-{int(d.anio.max())})*\n\n"
    rows = []
    for ind, g in d.groupby("indicador"):
        s = g["valor"].dropna()
        if len(s) < 3:
            continue
        jb, p = stats.jarque_bera(s)
        rows.append({
            "Indicador": ind, "N": len(s),
            "Media": round(float(s.mean()), 2), "Mediana": round(float(s.median()), 2),
            "Desv.Std": round(float(s.std()), 2),
            "Asimetría": round(float(s.skew()), 3), "Curtosis": round(float(s.kurtosis()), 3),
            "Jarque-Bera p": round(float(p), 4),
            "Años": f"{int(g.anio.min())}-{int(g.anio.max())}"
        })
    if not rows:
        return "⚠️ Pocos datos para estadísticos.", None
    res = pd.DataFrame(rows)
    panel = crear_panel_ancho(d)
    cols = [c for c in panel.columns if c not in ["departamento", "anio"]]
    fig = None
    if len(cols) >= 2:
        fig = px.imshow(panel[cols].corr(), text_auto=".2f", color_continuous_scale="RdBu_r",
                        title="Correlación (datos reales)", aspect="auto")
        fig = style_fig(fig)
        fig.update_layout(height=420)
    return info + res.to_markdown(index=False), fig

def pca_fn(PANEL, a1, a2):
    p = PANEL[(PANEL["anio"] >= a1) & (PANEL["anio"] <= a2)].copy()
    cols = [c for c in p.columns if c not in ["departamento", "anio"]]
    if len(cols) < 2:
        return "⚠️ Se necesitan ≥2 indicadores.", None
    X = p[cols].dropna(thresh=2).fillna(p[cols].mean())
    if len(X) < 6:
        X = PANEL[cols].dropna(thresh=2).fillna(PANEL[cols].mean())
        if len(X) < 6:
            return "⚠️ Datos insuficientes para PCA.", None
    Xs = StandardScaler().fit_transform(X)
    pca = PCA(n_components=2)
    comps = pca.fit_transform(Xs)
    var = pca.explained_variance_ratio_
    p_clean = PANEL.loc[X.index].copy()
    p_clean["PC1"], p_clean["PC2"] = comps[:, 0], comps[:, 1]
    p_clean["es_narino"] = p_clean["departamento"] == "Nariño"
    texto = f"**PCA** PC1={var[0]*100:.1f}% · PC2={var[1]*100:.1f}% · Acum={sum(var)*100:.1f}%"
    fig = go.Figure()
    mask = ~p_clean["es_narino"]
    fig.add_trace(go.Scatter(
        x=p_clean.loc[mask, "PC1"], y=p_clean.loc[mask, "PC2"], mode="markers",
        marker=dict(size=7, color="#64748B", opacity=0.7), name="Otros",
        text=p_clean.loc[mask, "departamento"]
    ))
    if p_clean["es_narino"].any():
        fig.add_trace(go.Scatter(
            x=p_clean.loc[p_clean["es_narino"], "PC1"],
            y=p_clean.loc[p_clean["es_narino"], "PC2"],
            mode="markers+text", marker=dict(size=14, color="#22D3EE", symbol="star"),
            name="Nariño", text=["Nariño"] * int(p_clean["es_narino"].sum()),
            textposition="top center"
        ))
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
    for i, col in enumerate(cols):
        fig.add_trace(go.Scatter(
            x=[0, loadings[i, 0] * 2.2], y=[0, loadings[i, 1] * 2.2],
            mode="lines+text", text=["", col], textposition="top center",
            line=dict(color="#A78BFA", width=2), showlegend=False
        ))
    fig.update_layout(
        title="Biplot PCA – Nariño",
        xaxis_title=f"PC1 ({var[0]*100:.1f}%)",
        yaxis_title=f"PC2 ({var[1]*100:.1f}%)",
        height=500
    )
    return texto, style_fig(fig)

def kmeans_fn(PANEL, a1, a2, k):
    p = PANEL[(PANEL["anio"] >= a1) & (PANEL["anio"] <= a2)]
    cols = [c for c in p.columns if c not in ["departamento", "anio"]]
    if len(cols) < 2:
        return "⚠️ ≥2 indicadores requeridos.", None
    agg = p.groupby("departamento")[cols].mean().dropna(thresh=2).fillna(p[cols].mean())
    if len(agg) < int(k):
        return f"⚠️ Solo {len(agg)} departamentos. Reduce K.", None
    Xs = StandardScaler().fit_transform(agg)
    labels = KMeans(n_clusters=int(k), random_state=42, n_init=10).fit_predict(Xs)
    agg = agg.copy()
    agg["Cluster"] = labels
    agg = agg.reset_index()
    agg["size"] = np.where(agg["departamento"] == "Nariño", 22, 10)
    xcol, ycol = cols[0], cols[1] if len(cols) > 1 else cols[0]
    fig = px.scatter(
        agg, x=xcol, y=ycol, color="Cluster", size="size",
        hover_name="departamento", title=f"K-Means k={k}", color_continuous_scale="Viridis"
    )
    fig.update_layout(height=470)
    fig = style_fig(fig)
    cl = agg.loc[agg["departamento"] == "Nariño", "Cluster"].values
    return f"K-Means k={k}. Nariño → Cluster {cl[0] if len(cl) else 'N/D'}", fig

def mapa_fn(DF_LONG, GEOJSON, ind, anio):
    anio_use = mejor_anio(DF_LONG, ind, int(anio))
    if anio_use is None:
        fig = go.Figure()
        fig.add_annotation(text=f"Sin datos de {ind}", xref="paper", yref="paper",
                           x=0.5, y=0.5, showarrow=False, font=dict(color="#F1F5F9", size=14))
        return style_fig(fig)
    d = DF_LONG[(DF_LONG["indicador"] == ind) & (DF_LONG["anio"] == anio_use)].copy()
    d = d.dropna(subset=["valor", "departamento"])
    if d.empty:
        fig = go.Figure()
        fig.add_annotation(text="Sin filas", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return style_fig(fig)
    nota = f" (ajustado desde {anio})" if anio_use != int(anio) else ""
    if GEOJSON is not None:
        try:
            d = d.copy()
            d["geo_name"] = d["departamento"].apply(_match_geo_name)
            props = []
            for ft in GEOJSON.get("features", []):
                p = ft.get("properties", {})
                for key in ("DPTO_CNMBR", "NOMBRE_DPT", "nombre", "name"):
                    if key in p and p[key]:
                        props.append(str(p[key]))
                        break
            props_set = set(props)
            matched = d["geo_name"].isin(props_set).sum()
            if matched < max(5, len(d) // 3):
                def norm_geo(s):
                    return quitar_tildes(s).title()
                props_norm = {norm_geo(p): p for p in props}
                d["geo_name"] = d["departamento"].apply(
                    lambda x: props_norm.get(norm_geo(_match_geo_name(x)), _match_geo_name(x))
                )
                matched = d["geo_name"].isin(props_set).sum()
            if matched >= 5 and d["valor"].nunique() > 1:
                fig = px.choropleth(
                    d, geojson=GEOJSON, locations="geo_name",
                    featureidkey="properties.DPTO_CNMBR", color="valor",
                    color_continuous_scale="Viridis",
                    title=f"{ind} · {anio_use}{nota}"
                )
                fig.update_geos(fitbounds="locations", visible=False, bgcolor="#0B1220")
                fig.update_layout(height=580, margin=dict(r=0, t=50, l=0, b=0))
                return style_fig(fig)
        except Exception as e:
            print("Mapa error:", e)
    d2 = d.sort_values("valor", ascending=True)
    fig = px.bar(
        d2, x="valor", y="departamento", orientation="h",
        title=f"{ind} · {anio_use}{nota} [barras]",
        color="valor", color_continuous_scale="Viridis"
    )
    fig.update_layout(height=max(420, 16 * len(d2)))
    return style_fig(fig)

def moran_fn(DF_LONG, ind, anio):
    anio_use = mejor_anio(DF_LONG, ind, int(anio))
    if anio_use is None:
        return "⚠️ Sin datos de ese indicador.", None
    d = DF_LONG[(DF_LONG["indicador"] == ind) & (DF_LONG["anio"] == anio_use)].copy()
    d = d.dropna(subset=["valor", "departamento"])
    if len(d) < 8:
        return f"⚠️ Pocas observaciones ({len(d)}) para Moran.", None
    try:
        d = d.set_index("departamento")
        y = d["valor"].values.astype(float)
        n = len(y)
        side = int(np.ceil(np.sqrt(n)))
        w = libpysal.weights.lat2W(side, side)
        w = libpysal.weights.WSP(w.sparse[:n, :n]).to_W()
        w.transform = "r"
        mi = Moran(y, w)
        lisa = Moran_Local(y, w)
        y_std = (y - y.mean()) / (y.std() if y.std() > 0 else 1)
        lag = np.asarray(w.sparse @ y_std).flatten()
        quad = lisa.q
        labels_q = {1: "HH (hotspot)", 2: "LH", 3: "LL (coldspot)", 4: "HL"}
        colors_q = {1: "#F87171", 2: "#FBBF24", 3: "#60A5FA", 4: "#C084FC"}
        nombres = list(d.index)
        fig = go.Figure()
        for q in [1, 2, 3, 4]:
            mask = (quad == q)
            if not mask.any():
                continue
            fig.add_trace(go.Scatter(
                x=y_std[mask], y=lag[mask], mode="markers+text",
                text=[nombres[i] if nombres[i] == "Nariño" else "" for i in range(n) if mask[i]],
                textposition="top center",
                marker=dict(size=11, color=colors_q[q], opacity=0.9),
                name=labels_q[q]
            ))
        fig.add_hline(y=0, line_dash="dot", line_color="#64748B")
        fig.add_vline(x=0, line_dash="dot", line_color="#64748B")
        fig.update_layout(
            title=f"Diagrama de Moran / LISA – {ind} ({anio_use})",
            xaxis_title="Valor estandarizado", yaxis_title="Rezago espacial",
            height=480, legend=dict(orientation="h", y=1.08)
        )
        nota = f" (año ajustado a {anio_use})" if anio_use != int(anio) else ""
        sig = (lisa.p_sim < 0.05).sum()
        texto = (
            f"**Moran Global – {ind} {anio_use}**{nota}\n\n"
            f"- I = **{mi.I:.4f}** | p-sim = **{mi.p_sim:.4f}** → "
            f"{'**significativa**' if mi.p_sim < 0.05 else 'no significativa'}\n\n"
            f"**LISA**: {sig}/{n} con p<0.05 · "
            f"HH={(quad==1).sum()} LL={(quad==3).sum()} LH={(quad==2).sum()} HL={(quad==4).sum()}"
        )
        return texto, style_fig(fig)
    except Exception as e:
        return f"Error Moran/LISA: {e}", None

def var_fn(DF_LONG, depto, inds, lags, tipo):
    d = DF_LONG[DF_LONG["departamento"] == depto]
    if d.empty:
        return f"⚠️ Sin datos para {depto}.", None
    panel = d.pivot_table(index="anio", columns="indicador", values="valor", aggfunc="sum")
    cols = [c for c in (inds or list(panel.columns)) if c in panel.columns]
    panel = panel[cols]
    good = [c for c in panel.columns if panel[c].notna().sum() >= 6]
    if len(good) < 2:
        disp = d.groupby("indicador")["anio"].agg(["min", "max", "count"])
        return (f"⚠️ {depto}: se necesitan ≥2 indicadores con ≥6 años.\n\n{disp.to_markdown()}"), None
    panel = panel[good].interpolate(limit_direction="both").dropna()
    nobs, neqs = panel.shape
    if nobs < 6:
        return f"⚠️ Solo {nobs} años útiles en {depto}.", None
    adf_rows = []
    for c in panel.columns:
        try:
            r = adfuller(panel[c], autolag="AIC")
            adf_rows.append({
                "Variable": c, "ADF": round(r[0], 3), "p": round(r[1], 4),
                "Estacionaria": "Sí" if r[1] < 0.05 else "No"
            })
        except Exception:
            pass
    texto = f"**ADF – {depto}**\n\n" + pd.DataFrame(adf_rows).to_markdown(index=False) + "\n\n"
    max_safe = lags_seguros(nobs, neqs, lags)
    try:
        if tipo == "VAR":
            model = VAR(panel)
            res = model.fit(maxlags=max_safe, ic="aic")
            texto += f"**VAR** · lags={res.k_ar} (máx seguro={max_safe}) · vars: {list(panel.columns)}\n\n"
            irf = res.irf(periods=5)
            fig = go.Figure()
            for i, col in enumerate(panel.columns):
                fig.add_trace(go.Scatter(y=irf.irfs[:, i, 0], mode="lines+markers", name=f"Resp. {col}"))
            fig.update_layout(title=f"IRF 5 periodos – {depto}", height=440)
            return texto, style_fig(fig)
        else:
            k_ar = max(1, min(max_safe - 1, 2)) if max_safe > 1 else 1
            while k_ar > 1 and nobs <= neqs * (k_ar + 1) + 2:
                k_ar -= 1
            joh = coint_johansen(panel, det_order=0, k_ar_diff=k_ar)
            rank = int((joh.lr1 > joh.cvt[:, 1]).sum())
            texto += (
                f"**VECM / Johansen** · k_ar_diff={k_ar} · nobs={nobs} · eqs={neqs}\n\n"
                f"Trazas: `{np.round(joh.lr1, 2)}`\n"
                f"Críticos 95%: `{np.round(joh.cvt[:, 1], 2)}`\n"
                f"Rango cointegración (95%): **{rank}**\n\n"
            )
            fig = go.Figure()
            for c in panel.columns:
                s = panel[c]
                s_norm = (s - s.mean()) / (s.std() if s.std() > 0 else 1)
                fig.add_trace(go.Scatter(x=panel.index, y=s_norm, mode="lines+markers", name=c))
            fig.update_layout(
                title=f"Series estandarizadas – {depto} (contexto VECM)",
                height=440, yaxis_title="Z-score"
            )
            return texto, style_fig(fig)
    except Exception as e:
        fig = go.Figure()
        for c in panel.columns:
            fig.add_trace(go.Scatter(x=panel.index, y=panel[c], mode="lines+markers", name=c))
        fig.update_layout(title=f"Series – {depto}", height=420)
        return texto + f"\n⚠️ Error: `{e}`", style_fig(fig)

def ml_fn(PANEL, target, preds, a1, a2, ar, al):
    p = PANEL[(PANEL["anio"] >= a1) & (PANEL["anio"] <= a2)].copy()
    cols = [c for c in p.columns if c not in ["departamento", "anio"]]
    if target not in cols:
        return f"⚠️ Target '{target}' no disponible. {cols}", None
    features = [c for c in (preds or cols) if c in cols and c != target] or [c for c in cols if c != target]
    if not features:
        return "⚠️ No hay predictores.", None
    data = p[[target] + features].dropna(subset=[target])
    data[features] = data[features].fillna(data[features].mean())
    data = data.dropna()
    if len(data) < 15:
        data = PANEL[[target] + features].dropna(subset=[target])
        data[features] = data[features].fillna(data[features].mean()).dropna()
        if len(data) < 15:
            return f"⚠️ Solo {len(data)} filas para ML.", None
    X, y = data[features], data[target]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42)
    modelos = {
        "Ridge": Ridge(alpha=float(ar)),
        "Lasso": Lasso(alpha=float(al), max_iter=10000),
        "SVR": SVR(kernel="rbf", C=10),
        "MLP": MLPRegressor(hidden_layer_sizes=(48, 24), max_iter=400, random_state=42),
    }
    rows, store = [], {}
    for nom, mod in modelos.items():
        try:
            mod.fit(Xtr, ytr)
            yp = mod.predict(Xte)
            rows.append({
                "Modelo": nom, "R²": round(r2_score(yte, yp), 3),
                "RMSE": round(float(np.sqrt(mean_squared_error(yte, yp))), 2),
                "MAE": round(float(mean_absolute_error(yte, yp)), 2)
            })
            store[nom] = (yte, yp)
        except Exception:
            continue
    if not rows:
        return "⚠️ Ningún modelo entrenó.", None
    res = pd.DataFrame(rows)
    mejor = res.loc[res["R²"].idxmax(), "Modelo"]
    yt, yh = store[mejor]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=yt, y=yh, mode="markers", name="Pred",
                             marker=dict(color="#22D3EE", size=9, opacity=0.85)))
    fig.add_trace(go.Scatter(
        x=[float(yt.min()), float(yt.max())], y=[float(yt.min()), float(yt.max())],
        mode="lines", name="Ideal", line=dict(dash="dash", color="#A78BFA", width=2)
    ))
    fig.update_layout(title=f"Mejor: {mejor}", height=430)
    return res.to_markdown(index=False), style_fig(fig)

def xai_fn(PANEL, target, preds, a1, a2):
    p = PANEL[(PANEL["anio"] >= a1) & (PANEL["anio"] <= a2)].copy()
    cols = [c for c in p.columns if c not in ["departamento", "anio"]]
    if target not in cols:
        return f"⚠️ Target no disponible. {cols}", None
    features = [c for c in (preds or cols) if c in cols and c != target] or [c for c in cols if c != target]
    data = p[[target] + features].dropna(subset=[target])
    data[features] = data[features].fillna(data[features].mean()).dropna()
    if len(data) < 12:
        data = PANEL[[target] + features].dropna(subset=[target])
        data[features] = data[features].fillna(data[features].mean()).dropna()
    if len(data) < 12:
        return f"⚠️ Solo {len(data)} filas para XAI.", None
    X, y = data[features], data[target]
    modelo = Ridge(alpha=1.0).fit(X, y)
    r = permutation_importance(modelo, X, y, n_repeats=10, random_state=42)
    imp = pd.DataFrame({
        "Variable": features, "Importancia": r.importances_mean, "Std": r.importances_std
    }).sort_values("Importancia", ascending=False)
    fig = px.bar(imp, x="Importancia", y="Variable", orientation="h",
                 title="XAI – Importancia (datos reales)", color="Importancia",
                 color_continuous_scale="Viridis")
    fig.update_layout(height=400)
    return imp.to_markdown(index=False), style_fig(fig)

def sim_fn(DF_LONG, depto, ind_s, ind_r, shock, anios):
    d = DF_LONG[DF_LONG["departamento"] == depto]
    panel = d.pivot_table(index="anio", columns="indicador", values="valor", aggfunc="sum")
    good = [c for c in panel.columns if panel[c].notna().sum() >= 6]
    if ind_s not in good or ind_r not in good:
        disp = d.groupby("indicador")["anio"].agg(["min", "max", "count"])
        return (
            f"⚠️ No se puede simular **{ind_s} → {ind_r}** en **{depto}**.\n\n"
            f"Se necesitan ≥6 años en ambos indicadores.\n\n"
            f"**Cobertura en {depto}:**\n{disp.to_markdown()}"
        ), None
    cols_use = [ind_s, ind_r] + [c for c in good if c not in (ind_s, ind_r)][:1]
    panel = panel[cols_use].interpolate(limit_direction="both").dropna()
    nobs, neqs = panel.shape
    if nobs < 6:
        return f"⚠️ Solo {nobs} años útiles tras alinear series en {depto}.", None
    max_safe = lags_seguros(nobs, neqs, solicitado=2)
    try:
        model = VAR(panel)
        try:
            res = model.fit(maxlags=max_safe, ic="aic")
        except Exception:
            res = model.fit(maxlags=1)
        base = res.forecast(panel.values[-res.k_ar:], steps=int(anios))
        last = panel.values[-res.k_ar:].copy()
        idx_s = list(panel.columns).index(ind_s)
        last[-1, idx_s] *= (1 + float(shock) / 100.0)
        shock_f = res.forecast(last, steps=int(anios))
        idx_r = list(panel.columns).index(ind_r)
        anios_f = list(range(int(panel.index.max()) + 1,
                              int(panel.index.max()) + int(anios) + 1))
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=panel.index, y=panel[ind_r],
            mode="lines+markers", name="Histórico", line=dict(color="#94A3B8", width=2)
        ))
        fig.add_trace(go.Scatter(
            x=anios_f, y=base[:, idx_r],
            mode="lines", name="Proyección base",
            line=dict(dash="dash", color="#22D3EE", width=2)
        ))
        fig.add_trace(go.Scatter(
            x=anios_f, y=shock_f[:, idx_r],
            mode="lines", name=f"Shock {shock:+.0f}%",
            line=dict(color="#A78BFA", width=3)
        ))
        fig.update_layout(
            title=f"Simulación de política – impacto en {ind_r} ({depto})",
            height=460
        )
        return (
            f"Shock **{shock:+.0f}%** en **{ind_s}** → respuesta de **{ind_r}** "
            f"a {anios} años (VAR, lags={res.k_ar}, nobs={nobs}, eqs={neqs})."
        ), style_fig(fig)
    except Exception as e:
        return f"Error en simulación: {e}", None

def focus_narino(DF_LONG, lista_inds):
    nar = DF_LONG[DF_LONG["departamento"] == "Nariño"]
    if nar.empty:
        return "⚠️ No hay filas de Nariño.", None
    rows = []
    for ind in lista_inds:
        g = nar[nar["indicador"] == ind]
        if g.empty:
            rows.append({"Indicador": ind, "Último año": "—", "Valor": "—", "Δ YoY %": "—", "Prom.Nacional": "—"})
            continue
        ult = int(g["anio"].max())
        val = float(g.loc[g["anio"] == ult, "valor"].iloc[0])
        prev = g[g["anio"] == ult - 1]["valor"]
        delta = ((val - float(prev.iloc[0])) / float(prev.iloc[0]) * 100) if len(prev) and float(prev.iloc[0]) != 0 else None
        nac = DF_LONG[(DF_LONG["indicador"] == ind) & (DF_LONG["anio"] == ult)]["valor"].mean()
        rows.append({
            "Indicador": ind, "Último año": ult, "Valor": round(val, 1),
            "Δ YoY %": round(delta, 1) if delta is not None else "N/D",
            "Prom.Nacional": round(float(nac), 1) if pd.notna(nac) else "—"
        })
    res = pd.DataFrame(rows)
    counts = nar.groupby("indicador")["anio"].nunique().sort_values(ascending=False)
    ind0 = counts.index[0]
    nar_t = nar[nar["indicador"] == ind0].groupby("anio")["valor"].mean().reset_index()
    nac_t = DF_LONG[DF_LONG["indicador"] == ind0].groupby("anio")["valor"].mean().reset_index()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=nar_t["anio"], y=nar_t["valor"], mode="lines+markers",
                             name="Nariño", line=dict(color="#22D3EE", width=3)))
    fig.add_trace(go.Scatter(x=nac_t["anio"], y=nac_t["valor"], mode="lines+markers",
                             name="Nacional", line=dict(color="#A78BFA", width=2, dash="dash")))
    fig.update_layout(title=f"{ind0}: Nariño vs Nacional", height=420)
    return res.to_markdown(index=False), style_fig(fig)

# ============================================================
# APP
# ============================================================
DF_LONG = cargar_todo()
GEOJSON = cargar_geojson()
PANEL = crear_panel_ancho(DF_LONG)

lista_inds = sorted(DF_LONG["indicador"].unique().tolist())
lista_deptos = sorted(DF_LONG["departamento"].unique().tolist())
amin, amax = int(DF_LONG["anio"].min()), int(DF_LONG["anio"].max())
anios_count = DF_LONG.groupby("anio")["indicador"].nunique().sort_values(ascending=False)
anio_default = int(anios_count.index[0])
anios_ok = anios_count[anios_count >= 2].index
a1_def = int(min(anios_ok)) if len(anios_ok) else max(amin, 2015)
a2_def = int(max(anios_ok)) if len(anios_ok) else amax

st.markdown(
    """
    <h1 style="color:#22D3EE; margin-bottom:0.2rem;">
        🇨🇴 Observatorio Económico y Social
    </h1>
    <p style="color:#94A3B8; margin-top:0;">
        Datos reales · Enfoque departamental · Prioridad Nariño
    </p>
    """,
    unsafe_allow_html=True,
)
st.caption(
    f"Indicadores: {', '.join(lista_inds)} · Periodo: {amin}–{amax} · "
    f"Rango recomendado: {a1_def}–{a2_def}"
)

with st.sidebar:
    st.header("Controles")
    depto = st.multiselect(
        "Departamento(s)",
        ["Todos"] + lista_deptos,
        default=["Nariño"] if "Nariño" in lista_deptos else ["Todos"]
    )
    inds = st.multiselect("Indicadores", lista_inds, default=lista_inds)
    a1 = st.slider("Año inicial", amin, amax, a1_def, 1)
    a2 = st.slider("Año final", amin, amax, a2_def, 1)
    anio_m = st.slider("Año mapa / Moran", amin, amax, anio_default, 1)
    k = st.slider("K clusters", 2, 6, 3, 1)
    ar = st.slider("Alpha Ridge", 0.01, 30.0, 1.0, 0.1)
    al = st.slider("Alpha Lasso", 0.001, 3.0, 0.1, 0.01)
    lags = st.slider("Lags VAR", 1, 4, 2, 1)
    shock = st.slider("Shock %", -40, 40, 10, 1)
    anios_s = st.slider("Años proyección", 3, 8, 5, 1)

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 EDA · PCA · Clustering",
    "🗺️ Econometría Espacial",
    "📈 Series de Tiempo (VAR/VEC)",
    "🤖 ML · Deep Learning & XAI",
    "🎯 Simulador de Política",
    "📍 Focus Nariño",
])

with tab1:
    c1, c2, c3 = st.columns(3)
    run_eda = c1.button("EDA + Correlación", type="primary", key="b1")
    run_pca = c2.button("PCA + Biplot", type="primary", key="b2")
    run_km = c3.button("K-Means", type="primary", key="b3")
    if run_eda:
        txt, fig = eda(DF_LONG, PANEL, lista_inds, depto, inds, a1, a2)
        st.markdown(txt)
        if fig is not None:
            st.plotly_chart(fig, width="stretch")
    if run_pca:
        txt, fig = pca_fn(PANEL, a1, a2)
        st.markdown(txt)
        if fig is not None:
            st.plotly_chart(fig, width="stretch")
    if run_km:
        txt, fig = kmeans_fn(PANEL, a1, a2, k)
        st.markdown(txt)
        if fig is not None:
            st.plotly_chart(fig, width="stretch")

with tab2:
    ind_m = st.selectbox("Indicador espacial", lista_inds, index=0, key="ind_m")
    c1, c2 = st.columns(2)
    run_mapa = c1.button("Mapa / Barras", type="primary", key="bm")
    run_moran = c2.button("Moran + LISA", type="primary", key="bmor")
    if run_mapa:
        fig = mapa_fn(DF_LONG, GEOJSON, ind_m, anio_m)
        st.plotly_chart(fig, width="stretch")
    if run_moran:
        txt, fig = moran_fn(DF_LONG, ind_m, anio_m)
        st.markdown(txt)
        if fig is not None:
            st.plotly_chart(fig, width="stretch")

with tab3:
    d_ts = st.selectbox(
        "Departamento", lista_deptos,
        index=lista_deptos.index("Nariño") if "Nariño" in lista_deptos else 0,
        key="d_ts"
    )
    i_ts = st.multiselect(
        "Indicadores VAR/VECM", lista_inds,
        default=lista_inds[:min(3, len(lista_inds))], key="i_ts"
    )
    tipo = st.radio("Modelo", ["VAR", "VECM"], horizontal=True, key="tipo")
    if st.button("Estimar modelo", type="primary", key="bts"):
        txt, fig = var_fn(DF_LONG, d_ts, i_ts, lags, tipo)
        st.markdown(txt)
        if fig is not None:
            st.plotly_chart(fig, width="stretch")

with tab4:
    tgt = st.selectbox(
        "Target", lista_inds,
        index=lista_inds.index("PIB") if "PIB" in lista_inds else 0, key="tgt"
    )
    prd = st.multiselect(
        "Predictores", lista_inds,
        default=[i for i in lista_inds if i != tgt][:4], key="prd"
    )
    c1, c2 = st.columns(2)
    run_ml = c1.button("Entrenar ML", type="primary", key="bml")
    run_xai = c2.button("XAI", type="primary", key="bx")
    if run_ml:
        txt, fig = ml_fn(PANEL, tgt, prd, a1, a2, ar, al)
        st.markdown(txt)
        if fig is not None:
            st.plotly_chart(fig, width="stretch")
    if run_xai:
        txt, fig = xai_fn(PANEL, tgt, prd, a1, a2)
        st.markdown(txt)
        if fig is not None:
            st.plotly_chart(fig, width="stretch")

with tab5:
    ds = st.selectbox(
        "Departamento", lista_deptos,
        index=lista_deptos.index("Nariño") if "Nariño" in lista_deptos else 0,
        key="ds"
    )
    is_ = st.selectbox(
        "Indicador shock", lista_inds,
        index=lista_inds.index("Recursos_DNP") if "Recursos_DNP" in lista_inds else 0,
        key="is_"
    )
    ir = st.selectbox(
        "Indicador respuesta", lista_inds,
        index=lista_inds.index("PIB") if "PIB" in lista_inds else 0,
        key="ir"
    )
    if st.button("Simular", type="primary", key="bs"):
        txt, fig = sim_fn(DF_LONG, ds, is_, ir, shock, anios_s)
        st.markdown(txt)
        if fig is not None:
            st.plotly_chart(fig, width="stretch")

with tab6:
    if st.button("Actualizar dashboard Nariño", type="primary", key="bn"):
        txt, fig = focus_narino(DF_LONG, lista_inds)
        st.markdown(txt)
        if fig is not None:
            st.plotly_chart(fig, width="stretch")

st.markdown("---")
st.caption(
    "Streamlit · Solo datos reales · EDA · PCA · Clustering · Mapas · Moran/LISA · "
    "VAR/VECM · Ridge/Lasso/SVR/MLP · XAI · Simulación · Foco Nariño"
)
