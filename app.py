
import os
import io
import json
import base64
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import classification_report
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import cm

# ============================================================
# CONFIGURACIÓN
# ============================================================

st.set_page_config(
    page_title="Sistema Inteligente IRAS",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded"
)

APP_TITLE = "Sistema Inteligente IRAS"
SUBTITLE = "Predicción del Riesgo de Afectación a la Salud por Inseguridad Alimentaria"
DATA_FILE = "02_dataset_analitico_IRAS_Lima_150_V2.xlsx"

IRAS_LABELS = {0: "Bajo", 1: "Moderado", 2: "Alto", 3: "Crítico"}
IRAS_COLORS = {"Bajo": "#8BD342", "Moderado": "#FFC727", "Alto": "#FF8A27", "Crítico": "#FF3B3B"}

FEATURES = [
    "Ingreso_Mensual",
    "Integrantes_Hogar",
    "Personas_Con_Ingreso",
    "Gasto_Alimentos",
    "Porcentaje_Ingreso_Alimentos",
    "Frecuencia_Comidas",
    "Inflacion_Alimentaria",
    "Indice_Dependencia",
    "Ingreso_Per_Capita",
    "Gasto_Alimentos_Per_Capita",
    "Ingreso_Disponible",
    "Indice_Vulnerabilidad_Alimentaria",
    "Codigo_Distrito",
    "Codigo_Diversidad",
    "Codigo_Laboral",
    "Codigo_Educacion"
]

MAP_DIVERSIDAD = {"Alta": 0, "Media": 1, "Baja": 2}
MAP_LABORAL = {"Formal": 0, "Informal": 1, "Desempleado": 2}
MAP_EDUCACION = {"Superior": 0, "Secundaria": 1, "Primaria": 2}

# ============================================================
# CSS PREMIUM
# ============================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at 18% 15%, rgba(35,196,131,0.18), transparent 28%),
        radial-gradient(circle at 78% 2%, rgba(101,87,255,0.18), transparent 24%),
        linear-gradient(135deg, #061A33 0%, #071B2E 45%, #0A1020 100%);
}

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #031124 0%, #061A33 52%, #061222 100%);
    border-right: 1px solid rgba(255,255,255,.08);
}

.block-container {
    padding-top: 1.0rem;
    padding-bottom: 1.5rem;
}

.premium-header {
    border-radius: 24px;
    padding: 1.4rem 1.6rem;
    background: linear-gradient(135deg, rgba(9,31,57,.94), rgba(17,42,75,.84));
    border: 1px solid rgba(255,255,255,.10);
    box-shadow: 0 18px 45px rgba(0,0,0,.28);
    margin-bottom: 1rem;
    position: relative;
    overflow: hidden;
}

.premium-header:before {
    content: "";
    position: absolute;
    top: -120px;
    right: -120px;
    width: 310px;
    height: 310px;
    background: radial-gradient(circle, rgba(35,196,131,.35), transparent 65%);
    animation: floatGlow 5s infinite alternate ease-in-out;
}

@keyframes floatGlow {
    from { transform: translateY(0px) scale(1); opacity: .65; }
    to { transform: translateY(25px) scale(1.08); opacity: .9; }
}

.header-title {
    font-size: 2.1rem;
    font-weight: 800;
    color: #F8FAFC;
    letter-spacing: -.04em;
    margin: 0;
}

.header-subtitle {
    color: #B6C8DD;
    font-size: 1.02rem;
    margin-top: .2rem;
}

.card {
    background: linear-gradient(180deg, rgba(12,34,59,.94), rgba(7,24,43,.96));
    border: 1px solid rgba(255,255,255,.10);
    border-radius: 22px;
    padding: 1.1rem;
    box-shadow: 0 18px 40px rgba(0,0,0,.25);
    transition: transform .18s ease, border .18s ease, box-shadow .18s ease;
}

.card:hover {
    transform: translateY(-3px);
    border-color: rgba(35,196,131,.36);
    box-shadow: 0 22px 55px rgba(0,0,0,.34);
}

.kpi {
    min-height: 118px;
}

.kpi-title {
    color: #CBD5E1;
    font-size: .78rem;
    letter-spacing: .04em;
    text-transform: uppercase;
    font-weight: 700;
}

.kpi-value {
    color: #F8FAFC;
    font-size: 2rem;
    font-weight: 800;
    margin-top: .15rem;
}

.kpi-caption {
    color: #94A3B8;
    font-size: .85rem;
}

.form-card {
    background: linear-gradient(180deg, rgba(12,34,59,.98), rgba(7,24,43,.98));
    border: 1px solid rgba(255,255,255,.12);
    border-radius: 24px;
    padding: 1rem 1.1rem 1.2rem 1.1rem;
    box-shadow: 0 18px 45px rgba(0,0,0,.28);
}

.result-risk {
    text-align: center;
    padding: 1.4rem;
    border-radius: 22px;
    background: radial-gradient(circle at 50% 20%, rgba(255,59,59,.22), transparent 45%), rgba(6,26,51,.82);
    border: 1px solid rgba(255,255,255,.12);
}

.risk-pill {
    display: inline-block;
    padding: .65rem 2rem;
    border-radius: 14px;
    font-size: 2rem;
    font-weight: 900;
    color: white;
    background: linear-gradient(135deg, #E11D48, #FF3B3B);
    box-shadow: 0 12px 35px rgba(255,59,59,.28);
    margin: .5rem 0;
}

.badge {
    display: inline-block;
    padding: .28rem .65rem;
    border-radius: 999px;
    font-size: .78rem;
    font-weight: 800;
    margin: .2rem .2rem .2rem 0;
}

.badge-red { background: rgba(239,68,68,.18); color: #FCA5A5; border: 1px solid rgba(239,68,68,.36); }
.badge-green { background: rgba(34,197,94,.18); color: #86EFAC; border: 1px solid rgba(34,197,94,.36); }
.badge-yellow { background: rgba(245,158,11,.18); color: #FCD34D; border: 1px solid rgba(245,158,11,.36); }
.badge-blue { background: rgba(59,130,246,.18); color: #93C5FD; border: 1px solid rgba(59,130,246,.36); }

.assistant-box {
    background: linear-gradient(180deg, rgba(13,33,63,.95), rgba(21,18,69,.92));
    border: 1px solid rgba(139,92,246,.32);
    border-radius: 24px;
    padding: 1rem;
    box-shadow: 0 18px 45px rgba(0,0,0,.30);
}

.chat-bubble-user {
    background: linear-gradient(135deg, #5B21B6, #7C3AED);
    color: white;
    border-radius: 18px 18px 6px 18px;
    padding: .75rem .9rem;
    margin: .5rem 0 .5rem auto;
    max-width: 90%;
}

.chat-bubble-ai {
    background: rgba(255,255,255,.08);
    color: #E2E8F0;
    border-radius: 18px 18px 18px 6px;
    padding: .75rem .9rem;
    margin: .5rem auto .5rem 0;
    max-width: 96%;
    border: 1px solid rgba(255,255,255,.08);
}

.step-title {
    font-weight: 800;
    color: #F8FAFC;
    margin-bottom: .5rem;
}

.step-dot {
    display: inline-flex;
    justify-content: center;
    align-items: center;
    width: 32px;
    height: 32px;
    margin-right: .45rem;
    border-radius: 999px;
    background: linear-gradient(135deg, #10B981, #3B82F6);
    color: white;
    font-weight: 900;
}

.footer {
    color: #94A3B8;
    font-size: .8rem;
    text-align: center;
    padding: 1rem;
}

div[data-testid="stMetric"] {
    background: rgba(255,255,255,.055);
    border: 1px solid rgba(255,255,255,.08);
    padding: 1rem;
    border-radius: 18px;
}

.stButton > button {
    background: linear-gradient(135deg, #10B981, #2563EB);
    color: white;
    border: 0;
    border-radius: 14px;
    padding: .75rem 1rem;
    font-weight: 800;
    transition: all .2s ease;
}

.stButton > button:hover {
    transform: translateY(-2px);
    filter: brightness(1.05);
    box-shadow: 0 12px 30px rgba(37,99,235,.28);
}

</style>
""", unsafe_allow_html=True)

# ============================================================
# DATA + MODEL
# ============================================================

@st.cache_data
def load_data():
    path = Path(DATA_FILE)
    if not path.exists():
        st.error(f"No se encontró {DATA_FILE}. Asegúrate de subirlo junto con app.py.")
        st.stop()
    return pd.read_excel(path, sheet_name="Dataset_Analitico_V2")

@st.cache_resource
def train_model(df):
    X = df[FEATURES]
    y = df["Codigo_IRAS"]
    model = RandomForestClassifier(
        n_estimators=500,
        random_state=42,
        class_weight="balanced",
        min_samples_leaf=2
    )
    model.fit(X, y)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_validate(
        model, X, y, cv=cv,
        scoring=["accuracy", "precision_macro", "recall_macro", "f1_macro"]
    )

    metrics = pd.DataFrame({
        "Métrica": ["Accuracy", "Precision macro", "Recall macro", "F1 macro"],
        "Valor": [
            scores["test_accuracy"].mean(),
            scores["test_precision_macro"].mean(),
            scores["test_recall_macro"].mean(),
            scores["test_f1_macro"].mean()
        ]
    })

    importance = pd.DataFrame({
        "Variable": FEATURES,
        "Importancia": model.feature_importances_
    }).sort_values("Importancia", ascending=False)

    return model, metrics, importance

df = load_data()
model, metrics, importance = train_model(df)
district_map = {d: int(df[df["Distrito"] == d]["Codigo_Distrito"].iloc[0]) for d in sorted(df["Distrito"].unique())}
districts = sorted(df["Distrito"].unique())

# ============================================================
# FUNCTIONS
# ============================================================

def normalize_value(value, column):
    mn, mx = float(df[column].min()), float(df[column].max())
    if mx == mn:
        return 0
    return max(0, min(1, (value - mn) / (mx - mn)))

def compute_household(
    distrito, ingreso, integrantes, ingresos_personas, gasto,
    comidas, diversidad, laboral, educacion, programa, inflacion
):
    ingreso = max(float(ingreso), 1)
    integrantes = max(int(integrantes), 1)
    ingresos_personas = max(int(ingresos_personas), 1)
    gasto = max(float(gasto), 0)
    comidas = int(comidas)

    pct = round(gasto / ingreso * 100, 2)
    dependencia = round((integrantes - ingresos_personas) / ingresos_personas, 2)
    ingreso_pc = round(ingreso / integrantes, 2)
    gasto_pc = round(gasto / integrantes, 2)
    disponible = round(ingreso - gasto, 2)

    cod_div = {"Alta": 0, "Media": 1, "Baja": 2}[diversidad]
    cod_lab = {"Formal": 0, "Informal": 1, "Desempleado": 2}[laboral]
    cod_edu = {"Superior": 0, "Secundaria": 1, "Primaria": 2}[educacion]
    cod_dist = district_map.get(distrito, 0)

    score_ingreso = 1 - normalize_value(ingreso_pc, "Ingreso_Per_Capita")
    score_presion = normalize_value(pct, "Porcentaje_Ingreso_Alimentos")
    score_dep = normalize_value(dependencia, "Indice_Dependencia")
    score_dieta = (cod_div / 2) * 0.60 + max(0, min(1, (3 - comidas) / 2)) * 0.40
    score_lab = cod_lab / 2
    score_prog = 0 if programa == "Sí" else 1

    icva = round(
        score_ingreso * 0.25 +
        score_presion * 0.25 +
        score_dep * 0.15 +
        score_dieta * 0.20 +
        score_lab * 0.10 +
        score_prog * 0.05,
        4
    )

    X_new = pd.DataFrame([{
        "Ingreso_Mensual": ingreso,
        "Integrantes_Hogar": integrantes,
        "Personas_Con_Ingreso": ingresos_personas,
        "Gasto_Alimentos": gasto,
        "Porcentaje_Ingreso_Alimentos": pct,
        "Frecuencia_Comidas": comidas,
        "Inflacion_Alimentaria": inflacion,
        "Indice_Dependencia": dependencia,
        "Ingreso_Per_Capita": ingreso_pc,
        "Gasto_Alimentos_Per_Capita": gasto_pc,
        "Ingreso_Disponible": disponible,
        "Indice_Vulnerabilidad_Alimentaria": icva,
        "Codigo_Distrito": cod_dist,
        "Codigo_Diversidad": cod_div,
        "Codigo_Laboral": cod_lab,
        "Codigo_Educacion": cod_edu
    }])

    calc = {
        "pct": pct, "dependencia": dependencia, "ingreso_pc": ingreso_pc,
        "gasto_pc": gasto_pc, "disponible": disponible, "icva": icva,
        "cod_div": cod_div, "cod_lab": cod_lab
    }
    return X_new, calc

def predict_risk(X_new):
    pred = int(model.predict(X_new)[0])
    label = IRAS_LABELS[pred]
    proba = model.predict_proba(X_new)[0]
    probas = {IRAS_LABELS[int(c)]: round(float(p) * 100, 2) for c, p in zip(model.classes_, proba)}
    return label, probas

def explain(calc, X_new):
    factors = []
    if calc["ingreso_pc"] < df["Ingreso_Per_Capita"].quantile(.35):
        factors.append(("Ingreso per cápita", 0.31))
    if calc["pct"] > 50:
        factors.append(("% ingreso en alimentos", 0.24))
    if calc["cod_div"] == 2:
        factors.append(("Diversidad alimentaria", 0.17))
    if calc["cod_lab"] >= 1:
        factors.append(("Situación laboral", 0.11))
    if calc["dependencia"] > df["Indice_Dependencia"].median():
        factors.append(("Dependencia económica", 0.09))
    if X_new["Frecuencia_Comidas"].iloc[0] <= 2:
        factors.append(("Frecuencia de comidas", 0.08))
    if not factors:
        factors = [("Condiciones estables", 0.20), ("Ingreso adecuado", 0.18)]
    return factors

def recommendation_text(risk):
    if risk == "Bajo":
        return [
            "Mantener seguimiento general del hogar.",
            "Promover educación alimentaria preventiva.",
            "Monitorear cambios en precios de alimentos."
        ]
    if risk == "Moderado":
        return [
            "Realizar monitoreo preventivo.",
            "Promover planificación del gasto y dieta variada.",
            "Evaluar acceso a apoyo comunitario si aumenta la presión alimentaria."
        ]
    if risk == "Alto":
        return [
            "Priorizar apoyo alimentario y programas sociales.",
            "Realizar evaluación nutricional preventiva.",
            "Promover educación alimentaria y financiera.",
            "Seguimiento periódico por la municipalidad."
        ]
    return [
        "Priorizar apoyo alimentario inmediato.",
        "Derivar a evaluación nutricional en establecimiento de salud.",
        "Realizar seguimiento municipal preventivo.",
        "Evaluar acceso a programas sociales o comunitarios.",
        "Fomentar capacitación laboral y educación alimentaria."
    ]

def health_interpretation(risk):
    if risk == "Bajo":
        return "El hogar presenta menor vulnerabilidad nutricional asociada a inseguridad alimentaria."
    if risk == "Moderado":
        return "Podría existir riesgo de dieta poco variada o deficiencias de micronutrientes si las condiciones empeoran."
    if risk == "Alto":
        return "El perfil incrementa la vulnerabilidad a malnutrición, anemia o deterioro de calidad dietaria."
    return "El perfil incrementa la probabilidad de malnutrición, deficiencias de micronutrientes, anemia y bajo bienestar nutricional."

def web_search_answer(question):
    """
    Optional web search with Tavily.
    To activate:
    - In Streamlit Cloud secrets, add:
      TAVILY_API_KEY = "your_api_key"
    If no key exists, use local scientific knowledge base.
    """
    kb = {
        "programas": "En Perú, los programas o apoyos relacionados con seguridad alimentaria incluyen Qali Warma/Wasi Mikuna para alimentación escolar, Vaso de Leche, comedores populares, ollas comunes, Programa de Complementación Alimentaria y apoyo municipal focalizado.",
        "salud": "La inseguridad alimentaria se asocia con dietas poco diversas, deficiencias de micronutrientes, malnutrición, anemia y deterioro del bienestar nutricional. No debe interpretarse como diagnóstico clínico individual.",
        "municipalidad": "Una municipalidad debería priorizar hogares con alto IRAS, alta presión alimentaria, baja diversidad alimentaria, empleo informal y alta dependencia económica.",
        "recomendacion": "La intervención recomendada combina apoyo alimentario, educación nutricional, seguimiento social, derivación preventiva a centros de salud y articulación con programas sociales."
    }
    q = question.lower()

    try:
        api_key = st.secrets.get("TAVILY_API_KEY", None)
    except Exception:
        api_key = None

    if api_key:
        try:
            response = requests.post(
                "https://api.tavily.com/search",
                json={
                    "api_key": api_key,
                    "query": question,
                    "search_depth": "basic",
                    "include_answer": True,
                    "max_results": 4
                },
                timeout=12
            )
            data = response.json()
            answer = data.get("answer") or ""
            results = data.get("results", [])
            refs = []
            for r in results[:3]:
                refs.append(f"- {r.get('title','Fuente')}: {r.get('url','')}")
            if answer:
                return answer + "\n\nFuentes consultadas:\n" + "\n".join(refs)
        except Exception:
            pass

    if "programa" in q or "social" in q:
        return kb["programas"]
    if "salud" in q or "anemia" in q or "malnutric" in q:
        return kb["salud"]
    if "municipal" in q or "distrito" in q:
        return kb["municipalidad"]
    return kb["recomendacion"]

def create_pdf_report(risk, probas, calc, recs):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    w, h = A4
    c.setFont("Helvetica-Bold", 17)
    c.drawString(2*cm, h-2*cm, "Reporte IRAS - Riesgo Alimentario y Salud")
    c.setFont("Helvetica", 10)
    c.drawString(2*cm, h-2.7*cm, f"Fecha: {datetime.now().strftime('%d/%m/%Y %H:%M')}")
    c.drawString(2*cm, h-3.3*cm, f"Nivel de riesgo estimado: {risk}")
    y = h-4.2*cm
    c.setFont("Helvetica-Bold", 12)
    c.drawString(2*cm, y, "Probabilidades")
    y -= .6*cm
    c.setFont("Helvetica", 10)
    for k,v in probas.items():
        c.drawString(2.2*cm, y, f"{k}: {v}%")
        y -= .45*cm
    y -= .3*cm
    c.setFont("Helvetica-Bold", 12)
    c.drawString(2*cm, y, "Variables calculadas")
    y -= .6*cm
    c.setFont("Helvetica", 10)
    variables = [
        ("Ingreso per cápita", f"S/ {calc['ingreso_pc']}"),
        ("Presión alimentaria", f"{calc['pct']}%"),
        ("Dependencia económica", f"{calc['dependencia']}"),
        ("Ingreso disponible", f"S/ {calc['disponible']}"),
        ("ICVA", f"{calc['icva']}")
    ]
    for k,v in variables:
        c.drawString(2.2*cm, y, f"{k}: {v}")
        y -= .45*cm
    y -= .3*cm
    c.setFont("Helvetica-Bold", 12)
    c.drawString(2*cm, y, "Recomendaciones")
    y -= .6*cm
    c.setFont("Helvetica", 10)
    for i, rec in enumerate(recs, 1):
        c.drawString(2.2*cm, y, f"{i}. {rec[:85]}")
        y -= .45*cm
    c.setFont("Helvetica-Oblique", 8)
    c.drawString(2*cm, 1.5*cm, "Nota: Prototipo académico. IRAS estima riesgo indirecto; no realiza diagnóstico médico.")
    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    logo_path = Path("assets/iras_logo.svg")
    if logo_path.exists():
        st.image(str(logo_path), width=105)
    st.markdown("## IRAS")
    st.caption("Sistema Inteligente · Lima Metropolitana")
    menu = st.radio(
        "Navegación",
        ["Inicio", "Analizar Hogar", "Dashboard", "Mapa de Riesgo", "IA Asistente", "Reportes", "Metodología"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.success("🟢 Sistema en línea")
    st.info("Proyecto de Ciencia de Datos para apoyo a decisiones preventivas en salud pública.")

# ============================================================
# HEADER
# ============================================================

st.markdown(f"""
<div class="premium-header">
  <div class="header-title">{APP_TITLE}</div>
  <div class="header-subtitle">{SUBTITLE}</div>
</div>
""", unsafe_allow_html=True)

# Default values and session state
if "last_result" not in st.session_state:
    st.session_state.last_result = None

# ============================================================
# PAGES
# ============================================================

if menu in ["Inicio", "Analizar Hogar"]:
    k1, k2, k3, k4 = st.columns(4)
    high_crit = int(df["IRAS"].isin(["Alto", "Crítico"]).sum())
    kpis = [
        ("Hogares analizados", len(df), "Total en dataset"),
        ("Riesgo Alto + Crítico", f"{high_crit} ({high_crit/len(df)*100:.1f}%)", "Requieren atención"),
        ("ICVA promedio", f"{df['Indice_Vulnerabilidad_Alimentaria'].mean():.2f}", "Índice de 0 a 1"),
        ("Distritos", df["Distrito"].nunique(), "Lima Metropolitana")
    ]
    for col, (title, val, cap) in zip([k1,k2,k3,k4], kpis):
        with col:
            st.markdown(f"""
            <div class="card kpi">
                <div class="kpi-title">{title}</div>
                <div class="kpi-value">{val}</div>
                <div class="kpi-caption">{cap}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("")

    left, center, right = st.columns([1.15, 1.25, 1.05])

    with left:
        st.markdown('<div class="form-card">', unsafe_allow_html=True)
        st.markdown('<div class="step-title"><span class="step-dot">1</span>Información del hogar</div>', unsafe_allow_html=True)

        distrito = st.selectbox("Distrito", districts, index=districts.index("San Juan de Lurigancho") if "San Juan de Lurigancho" in districts else 0)
        ingreso = st.number_input("Ingreso mensual del hogar (S/.)", value=1800.0, step=50.0, min_value=1.0)
        c1, c2 = st.columns(2)
        integrantes = c1.number_input("Número de integrantes", value=5, min_value=1, max_value=12, step=1)
        ingresos_personas = c2.number_input("Personas con ingreso", value=1, min_value=1, max_value=6, step=1)
        gasto = st.number_input("Gasto en alimentos (S/.)", value=950.0, step=25.0, min_value=0.0)

        c3, c4 = st.columns(2)
        comidas = c3.selectbox("Frecuencia de comidas al día", [1,2,3,4], index=1)
        diversidad = c4.selectbox("Diversidad alimentaria", ["Alta","Media","Baja"], index=2)

        c5, c6 = st.columns(2)
        laboral = c5.selectbox("Situación laboral del jefe/a", ["Formal","Informal","Desempleado"], index=1)
        educacion = c6.selectbox("Nivel educativo", ["Superior","Secundaria","Primaria"], index=1)

        programa = st.radio("Recibe programa social", ["Sí","No"], horizontal=True, index=1)
        inflacion = st.number_input("Inflación alimentaria (%)", value=5.1, step=0.1, min_value=0.0)

        analyze = st.button("🔎 Analizar Hogar", use_container_width=True)
        st.caption("Los campos marcados son obligatorios. La información es usada únicamente con fines académicos.")
        st.markdown('</div>', unsafe_allow_html=True)

    if analyze:
        X_new, calc = compute_household(distrito, ingreso, integrantes, ingresos_personas, gasto, comidas, diversidad, laboral, educacion, programa, inflacion)
        risk, probas = predict_risk(X_new)
        factors = explain(calc, X_new)
        recs = recommendation_text(risk)
        health = health_interpretation(risk)
        st.session_state.last_result = {
            "risk": risk, "probas": probas, "calc": calc, "factors": factors, "recs": recs, "health": health,
            "inputs": {
                "distrito": distrito, "ingreso": ingreso, "integrantes": integrantes, "personas": ingresos_personas,
                "gasto": gasto, "comidas": comidas, "diversidad": diversidad, "laboral": laboral, "educacion": educacion,
                "programa": programa, "inflacion": inflacion
            }
        }

    result = st.session_state.last_result
    if result is None:
        X_new, calc = compute_household("San Juan de Lurigancho" if "San Juan de Lurigancho" in districts else districts[0], 1800, 5, 1, 950, 2, "Baja", "Informal", "Secundaria", "No", 5.1)
        risk, probas = predict_risk(X_new)
        factors = explain(calc, X_new)
        recs = recommendation_text(risk)
        health = health_interpretation(risk)
        result = {"risk": risk, "probas": probas, "calc": calc, "factors": factors, "recs": recs, "health": health}

    with center:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="step-title"><span class="step-dot">2</span>Resultado del análisis</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="result-risk">
            <div style="color:#CBD5E1; font-weight:700;">NIVEL DE RIESGO IRAS</div>
            <div class="risk-pill">{result['risk'].upper()}</div>
            <div style="color:#CBD5E1;">Probabilidad máxima estimada</div>
            <div style="font-size:2.4rem; font-weight:900; color:#FF5A5A;">{max(result['probas'].values()):.0f}%</div>
        </div>
        """, unsafe_allow_html=True)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("ICVA", f"{result['calc']['icva']:.2f}")
        m2.metric("Ingreso per cápita", f"S/ {result['calc']['ingreso_pc']:.0f}")
        m3.metric("% en alimentos", f"{result['calc']['pct']:.1f}%")
        m4.metric("Dependencia", f"{result['calc']['dependencia']:.1f}")

        st.markdown("#### Interpretación")
        st.write(f"El hogar presenta un riesgo **{result['risk']}** de afectación a la salud asociado a inseguridad alimentaria.")
        st.markdown("#### Riesgo asociado a salud")
        st.warning(result["health"])
        st.markdown('<span class="badge badge-red">Malnutrición</span><span class="badge badge-red">Deficiencias de micronutrientes</span><span class="badge badge-red">Anemia</span><span class="badge badge-red">Bajo bienestar nutricional</span>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with right:
        st.markdown('<div class="assistant-box">', unsafe_allow_html=True)
        st.markdown("### 🤖 IA Asistente  · BETA")
        st.markdown('<div class="chat-bubble-ai">¡Hola! Soy tu asistente IRAS. Puedo responder preguntas sobre inseguridad alimentaria, salud, nutrición, programas sociales y acciones municipales.</div>', unsafe_allow_html=True)
        user_q = st.text_area("Pregunta al asistente", "¿Qué programas sociales pueden apoyar a hogares en inseguridad alimentaria?")
        if st.button("Enviar pregunta", use_container_width=True):
            with st.spinner("Buscando información y generando respuesta..."):
                answer = web_search_answer(user_q)
            st.markdown(f'<div class="chat-bubble-user">{user_q}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="chat-bubble-ai">{answer}</div>', unsafe_allow_html=True)
        st.caption("Para búsqueda web real, configura TAVILY_API_KEY en Streamlit Secrets. Sin API usa base de conocimiento interna.")
        st.markdown('</div>', unsafe_allow_html=True)

    bottom1, bottom2, bottom3, bottom4 = st.columns([1.1,1.1,1.25,.9])

    with bottom1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### Probabilidad por nivel")
        p_df = pd.DataFrame({"Nivel": list(result["probas"].keys()), "Probabilidad": list(result["probas"].values())})
        fig = px.bar(p_df, x="Nivel", y="Probabilidad", color="Nivel", color_discrete_map=IRAS_COLORS, text="Probabilidad")
        fig.update_layout(height=280, margin=dict(l=10,r=10,t=30,b=10), showlegend=False, template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with bottom2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### Factores que más influyen")
        f_df = pd.DataFrame(result["factors"], columns=["Factor","Importancia"])
        fig = px.bar(f_df.sort_values("Importancia"), x="Importancia", y="Factor", orientation="h", text="Importancia")
        fig.update_layout(height=280, margin=dict(l=10,r=10,t=30,b=10), template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with bottom3:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### Recomendaciones para el hogar")
        for i, rec in enumerate(result["recs"], 1):
            st.write(f"**{i}.** {rec}")
        st.markdown('</div>', unsafe_allow_html=True)

    with bottom4:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### Acciones")
        pdf = create_pdf_report(result["risk"], result["probas"], result["calc"], result["recs"])
        st.download_button("⬇ Descargar Reporte PDF", data=pdf, file_name="reporte_IRAS_hogar.pdf", mime="application/pdf", use_container_width=True)
        st.download_button("📊 Exportar Dataset", data=df.to_csv(index=False).encode("utf-8-sig"), file_name="dataset_IRAS.csv", mime="text/csv", use_container_width=True)
        if st.button("🔄 Nuevo análisis", use_container_width=True):
            st.session_state.last_result = None
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

elif menu == "Dashboard":
    st.markdown("## 📊 Dashboard ejecutivo")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Hogares", len(df))
    c2.metric("Distritos", df["Distrito"].nunique())
    c3.metric("ICVA promedio", f"{df['Indice_Vulnerabilidad_Alimentaria'].mean():.2f}")
    c4.metric("Alto + Crítico", int(df["IRAS"].isin(["Alto","Crítico"]).sum()))

    col1, col2 = st.columns(2)
    with col1:
        counts = df["IRAS"].value_counts().reset_index()
        counts.columns = ["IRAS","Hogares"]
        fig = px.pie(counts, names="IRAS", values="Hogares", color="IRAS", color_discrete_map=IRAS_COLORS, hole=.45)
        fig.update_layout(template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        resumen = df.groupby("Distrito")["Indice_Vulnerabilidad_Alimentaria"].mean().sort_values(ascending=False).head(12).reset_index()
        fig = px.bar(resumen, x="Indice_Vulnerabilidad_Alimentaria", y="Distrito", orientation="h", template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Dataset analítico")
    st.dataframe(df, use_container_width=True)

elif menu == "Mapa de Riesgo":
    st.markdown("## 🗺️ Mapa de riesgo por distrito")
    st.info("En esta versión académica se muestra un panel geográfico simulado por ranking distrital. Para mapa real se puede integrar GeoJSON de Lima Metropolitana.")
    resumen = df.groupby("Distrito").agg(
        Vulnerabilidad=("Indice_Vulnerabilidad_Alimentaria","mean"),
        Hogares=("ID_Hogar","count"),
        IRAS=("Codigo_IRAS","mean")
    ).sort_values("Vulnerabilidad", ascending=False).reset_index()
    fig = px.bar(resumen, y="Distrito", x="Vulnerabilidad", color="IRAS", orientation="h", template="plotly_dark", color_continuous_scale=["#8BD342","#FFC727","#FF8A27","#FF3B3B"])
    fig.update_layout(height=650)
    st.plotly_chart(fig, use_container_width=True)

elif menu == "IA Asistente":
    st.markdown("## 🤖 IA Asistente con base de conocimiento")
    st.write("Pregunta sobre inseguridad alimentaria, salud, nutrición, programas sociales o decisiones municipales.")
    q = st.text_area("Tu pregunta", "¿Cómo se relaciona la inseguridad alimentaria con anemia y malnutrición?")
    if st.button("Generar respuesta"):
        with st.spinner("Consultando conocimiento disponible..."):
            st.markdown(web_search_answer(q))

elif menu == "Reportes":
    st.markdown("## 📄 Reportes")
    st.write("Exporta el dataset analítico o usa el módulo de análisis de hogar para descargar un reporte individual.")
    st.download_button("Descargar dataset CSV", df.to_csv(index=False).encode("utf-8-sig"), file_name="dataset_IRAS_analitico.csv", mime="text/csv")

elif menu == "Metodología":
    st.markdown("## 📚 Metodología")
    st.markdown("""
    ### Flujo del proyecto
    1. Construcción del dataset original.
    2. Limpieza y transformación.
    3. Dataset analítico V2.
    4. EDA profesional.
    5. Machine Learning.
    6. Asistente inteligente.
    7. Aplicación web demostrativa.

    ### Interpretación
    IRAS es un indicador de riesgo indirecto. No diagnostica enfermedades.

    ### Fuentes sugeridas
    - OMS: Malnutrition fact sheet.
    - FAO, IFAD, UNICEF, WFP y WHO: SOFI.
    - WFP Perú.
    - INEI.
    - MIDIS.
    """)

st.markdown('<div class="footer">© 2024 Sistema Inteligente IRAS · Proyecto de Ciencia de Datos · Desarrollado con Streamlit</div>', unsafe_allow_html=True)
