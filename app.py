import unicodedata
import warnings
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

warnings.filterwarnings("ignore")

# Configuración de página
st.set_page_config(
    page_title="Observatorio Departamental Colombia",
    page_icon="🇨🇴",
    layout="wide",
)

# ============================================================
# NORMALIZACIÓN ROBUSTA
# ============================================================
def quitar_tildes(texto):
    if pd.isna(texto) or texto is None:
        return ""
    t = str(texto).strip().lower()
    t = "".join(
        c for c in unicodedata.normalize("NFD", t) if unicodedata.category(c) != "Mn"
    )
    t = t.replace(",", "").replace(".", " ").replace("  ", " ").strip()
    return t

DEPARTAMENTOS_CANONICOS = {
    "amazonas": "Amazonas",
    "antioquia": "Antioquia",
    "arauca": "Arauca",
    "atlantico": "Atlántico",
    "bogota": "Bogotá D.C.",
    "bogota d c": "Bogotá D.C.",
    "bogota dc": "Bogotá D.C.",
    "bolivar": "Bolívar",
    "boyaca": "Boyacá",
    "caldas": "Caldas",
    "caqueta": "Caquetá",
    "casanare": "Casanare",
    "cauca": "Cauca",
    "cesar": "Cesar",
    "choco": "Chocó",
    "cordoba": "Córdoba",
    "cundinamarca": "Cundinamarca",
    "guainia": "Guainía",
    "guaviare": "Guaviare",
    "huila": "Huila",
    "la guajira": "La Guajira",
    "magdalena": "Magdalena",
    "meta": "Meta",
    "narino": "Nariño",
    "norte de santander": "Norte de Santander",
    "putumayo": "Putumayo",
    "quindio": "Quindío",
    "risaralda": "Risaralda",
    "san andres": "San Andrés y Providencia",
    "san andres y providencia": "San Andrés y Providencia",
    "santander": "Santander",
    "sucre": "Sucre",
    "tolima": "Tolima",
    "valle": "Valle del Cauca",
    "valle del cauca": "Valle del Cauca",
    "vaupes": "Vaupés",
    "vichada": "Vichada",
}

def normalizar_departamento(nombre):
    clave = quitar_tildes(nombre)
    return DEPARTAMENTOS_CANONICOS.get(
        clave, str(nombre).strip().title() if nombre else None
    )

# ============================================================
# CARGA DE DATOS (Con Cache de Streamlit)
# ============================================================
@st.cache_data(ttl=3600)
def cargar_pib():
    try:
        url = "https://www.datos.gov.co/resource/kgyi-qc7j.json?$limit=50000&$order=a_o DESC"
        r = requests.get(url, timeout=90)
        r.raise_for_status()
        df = pd.DataFrame(r.json())
        df["anio"] = pd.to_numeric(df["a_o"], errors="coerce")
        df["valor"] = pd.to_numeric(
            df["valor_miles_de_millones_de"], errors="coerce"
        )
        df["departamento"] = df["departamento"].apply(normalizar_departamento)
        df = df[
            df["tipo_de_precios"].str.contains(
                "constantes", case=False, na=False
            )
        ]
        agg = (
            df.groupby(["departamento", "anio"], as_index=False)["valor"]
            .sum()
            .dropna()
        )
        agg["indicador"] = "PIB (miles de millones, constantes 2015)"
        agg["fuente"] = "datos.gov.co/kgyi-qc7j"
        return agg
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def cargar_homicidios():
    try:
        url = "https://www.datos.gov.co/resource/m8fd-ahd9.json?$limit=150000"
        r = requests.get(url, timeout=90)
        r.raise_for_status()
        df = pd.DataFrame(r.json())
        df["fecha"] = pd.to_datetime(df["fecha_hecho"], errors="coerce")
        df["anio"] = df["fecha"].dt.year
        df["departamento"] = df["departamento"].apply(normalizar_departamento)
        df["cantidad"] = pd.to_numeric(
            df.get("cantidad"), errors="coerce"
        ).fillna(1)
        agg = (
            df.groupby(["departamento", "anio"], as_index=False)["cantidad"]
            .sum()
            .rename(columns={"cantidad": "valor"})
            .dropna()
        )
        agg["indicador"] = "Homicidios (víctimas)"
        agg["fuente"] = "MinDefensa/m8fd-ahd9"
        return agg
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def cargar_dnp():
    try:
        url = "https://www.datos.gov.co/resource/wc86-9j4a.json?$limit=80000"
        r = requests.get(url, timeout=90)
        r.raise_for_status()
        df = pd.DataFrame(r.json())
        df["departamento"] = df["departamento"].apply(normalizar_departamento)
        df["anio"] = pd.to_numeric(df.get("vigencia"), errors="coerce")
        valor_col = next(
            (
                c
                for c in ["valorpagado", "valorobligado", "valorcomprometido"]
                if c in df.columns
            ),
            None,
        )
        if not valor_col:
            return pd.DataFrame()
        df["valor"] = pd.to_numeric(df[valor_col], errors="coerce")
        agg = (
            df.groupby(["departamento", "anio"], as_index=False)["valor"]
            .sum()
            .dropna()
        )
        agg["indicador"] = "Recursos públicos DNP (COP)"
        agg["fuente"] = "DNP/wc86-9j4a"
        return agg
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def construir_maestro():
    partes = [cargar_pib(), cargar_homicidios(), cargar_dnp()]
    partes = [p for p in partes if not p.empty]
    if not partes:
        return pd.DataFrame(
            columns=["departamento", "anio", "valor", "indicador", "fuente"]
        )
    maestro = pd.concat(partes, ignore_index=True)
    maestro["anio"] = maestro["anio"].astype(int)
    return maestro.sort_values(["indicador", "departamento", "anio"]).reset_index(
        drop=True
    )

DF = construir_maestro()

# ============================================================
# INTERFAZ STREAMLIT
# ============================================================
st.title("🇨🇴 Observatorio de Actividad Económica y Social por Departamentos")
st.caption(
    "Fuentes oficiales: datos.gov.co (SODA API), MinDefensa, DNP. **Datos reales sin generación sintética.**"
)

if DF.empty:
    st.error(
        "⚠️ No se pudieron obtener los datos de las API públicas de SODA. Intenta recargar la página."
    )
else:
    lista_deptos = sorted(DF["departamento"].dropna().unique())
    lista_inds = sorted(DF["indicador"].unique())
    amin, amax = int(DF["anio"].min()), int(DF["anio"].max())

    with st.sidebar:
        st.header("⚙️ Filtros de Control")
        deptos_sel = st.multiselect(
            "Departamento(s)",
            options=["Todos"] + lista_deptos,
            default=["Nariño"] if "Nariño" in lista_deptos else ["Todos"],
        )
        ind_sel = st.selectbox("Indicador", options=lista_inds)
        
        col1, col2 = st.columns(2)
        with col1:
            amin_ui = st.number_input("Año Inicial", min_value=amin, max_value=amax, value=amin)
        with col2:
            amax_ui = st.number_input("Año Final", min_value=amin, max_value=amax, value=amax)
            
        anio_barras = st.slider("Año para Comparativa de Barras", min_value=amin, max_value=amax, value=amax)

    # Filtrado de datos
    df_f = DF.copy()
    if deptos_sel and "Todos" not in deptos_sel:
        df_f = df_f[df_f["departamento"].isin(deptos_sel)]
    df_f = df_f[
        (df_f["indicador"] == ind_sel)
        & (df_f["anio"] >= amin_ui)
        & (df_f["anio"] <= amax_ui)
    ]

    # Métrica / Estadísticas resumidas
    st.subheader(f"📊 Resumen de Indicador: {ind_sel}")
    if not df_f.empty:
        col_a, col_b, col_c, col_d = st.columns(4)
        col_a.metric("Promedio", f"{df_f['valor'].mean():,.2f}")
        col_b.metric("Mediana", f"{df_f['valor'].median():,.2f}")
        col_c.metric("Valor Mínimo", f"{df_f['valor'].min():,.2f}")
        col_d.metric("Valor Máximo", f"{df_f['valor'].max():,.2f}")
    else:
        st.warning("No hay registros para los filtros seleccionados.")

    tab1, tab2, tab3 = st.tabs(["📈 Tendencias", "📊 Barras Comparativas", "📋 Datos Planos"])

    with tab1:
        if not df_f.empty:
            fig_t = px.line(
                df_f,
                x="anio",
                y="valor",
                color="departamento",
                markers=True,
                title=f"Evolución Temporal: {ind_sel} ({amin_ui} - {amax_ui})",
                template="plotly_white",
            )
            fig_t.update_layout(hovermode="x unified")
            st.plotly_chart(fig_t, use_container_width=True)
        else:
            st.info("Seleccione departamentos e indicadores para visualizar la tendencia.")

    with tab2:
        df_b = DF[(DF["indicador"] == ind_sel) & (DF["anio"] == anio_barras)].nlargest(15, "valor")
        if not df_b.empty:
            fig_b = px.bar(
                df_b,
                x="valor",
                y="departamento",
                orientation="h",
                title=f"Top 15 Departamentos - {ind_sel} ({anio_barras})",
                template="plotly_white",
                color="valor",
                color_continuous_scale="Blues",
            )
            fig_b.update_layout(yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig_b, use_container_width=True)
        else:
            st.info(f"No hay datos para generar comparativa en el año {anio_barras}.")

    with tab3:
        st.dataframe(df_f, use_container_width=True)
