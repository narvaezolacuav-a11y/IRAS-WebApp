

import io, json
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import cm
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Sistema Inteligente IRAS", page_icon="🍽️", layout="wide", initial_sidebar_state="expanded")

DATA_FILE = "02_dataset_analitico_IRAS_Lima_150_V2.xlsx"
IRAS_LABELS = {0: "Bajo", 1: "Moderado", 2: "Alto", 3: "Crítico"}
IRAS_NUM = {v:k for k,v in IRAS_LABELS.items()}
IRAS_COLORS = {"Bajo":"#6EE7B7","Moderado":"#FACC15","Alto":"#FB923C","Crítico":"#F43F5E"}
FEATURES = ["Ingreso_Mensual","Integrantes_Hogar","Personas_Con_Ingreso","Gasto_Alimentos","Porcentaje_Ingreso_Alimentos","Frecuencia_Comidas","Inflacion_Alimentaria","Indice_Dependencia","Ingreso_Per_Capita","Gasto_Alimentos_Per_Capita","Ingreso_Disponible","Indice_Vulnerabilidad_Alimentaria","Codigo_Distrito","Codigo_Diversidad","Codigo_Laboral","Codigo_Educacion"]

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800;900&display=swap');
html, body, [class*="css"] {font-family: Inter, sans-serif;}
.stApp{
background: radial-gradient(circle at 15% 12%, rgba(39,224,163,.22), transparent 24%),
radial-gradient(circle at 90% 8%, rgba(168,85,247,.20), transparent 28%),
linear-gradient(135deg,#071A2E 0%,#081E34 50%,#0A1224 100%);
}
section[data-testid="stSidebar"]{background:linear-gradient(180deg,#031124,#071A2E,#071320);border-right:1px solid rgba(255,255,255,.10);}
.block-container{padding-top:1rem;}
.hero{border-radius:26px;padding:1.35rem 1.6rem;background:linear-gradient(135deg,rgba(12,44,77,.92),rgba(21,28,81,.84));border:1px solid rgba(255,255,255,.13);box-shadow:0 18px 45px rgba(0,0,0,.28);position:relative;overflow:hidden;margin-bottom:1rem;}
.hero:before{content:"";position:absolute;width:420px;height:420px;top:-210px;right:-170px;background:radial-gradient(circle,rgba(39,224,163,.35),transparent 60%);animation:pulseGlow 4s infinite alternate ease-in-out;}
@keyframes pulseGlow{from{transform:scale(1);opacity:.55;}to{transform:scale(1.18);opacity:.95;}}
.hero h1{color:#F8FAFC;margin:0;font-weight:900;font-size:2.05rem;letter-spacing:-.04em;position:relative;z-index:1;}
.hero p{color:#C8D8EA;margin:.25rem 0 0 0;position:relative;z-index:1;}
.card{background:linear-gradient(180deg,rgba(14,40,68,.96),rgba(7,24,43,.96));border:1px solid rgba(255,255,255,.11);border-radius:24px;padding:1rem;box-shadow:0 18px 42px rgba(0,0,0,.26);transition:all .22s ease;}
.card:hover{transform:translateY(-4px);border-color:rgba(39,224,163,.44);box-shadow:0 24px 60px rgba(0,0,0,.36);}
.kpi-title{color:#C8D8EA;font-size:.78rem;letter-spacing:.04em;text-transform:uppercase;font-weight:800;}
.kpi-value{font-size:2rem;font-weight:900;color:#F8FAFC;}
.kpi-caption{color:#9FB1C9;font-size:.85rem;}
.form-card{background:linear-gradient(180deg,rgba(14,40,68,.98),rgba(7,24,43,.98));border:1px solid rgba(255,255,255,.13);border-radius:24px;padding:1rem;box-shadow:0 18px 45px rgba(0,0,0,.30);}
.step-dot{display:inline-flex;justify-content:center;align-items:center;width:34px;height:34px;margin-right:.5rem;border-radius:999px;background:linear-gradient(135deg,#27E0A3,#38BDF8);color:#031124;font-weight:900;}
.step-title{font-size:1.04rem;font-weight:900;color:#F8FAFC;margin-bottom:.6rem;}
.risk-pill{display:inline-block;padding:.65rem 2rem;border-radius:16px;font-size:2rem;font-weight:900;color:white;background:linear-gradient(135deg,#F43F5E,#FB7185);box-shadow:0 0 38px rgba(244,63,94,.34);animation:softPulse 1.8s infinite alternate ease-in-out;}
@keyframes softPulse{from{transform:scale(1);filter:brightness(1);}to{transform:scale(1.025);filter:brightness(1.12);}}
.chat-panel{background:linear-gradient(180deg,rgba(19,30,83,.96),rgba(12,24,57,.98));border:1px solid rgba(167,139,250,.35);border-radius:24px;padding:1rem;box-shadow:0 18px 45px rgba(0,0,0,.31);}
.ai-bubble{background:rgba(255,255,255,.08);border:1px solid rgba(255,255,255,.10);border-radius:18px 18px 18px 6px;padding:.8rem .9rem;color:#E2E8F0;margin:.55rem 0;}
.user-bubble{background:linear-gradient(135deg,#7C3AED,#2563EB);border-radius:18px 18px 6px 18px;padding:.8rem .9rem;color:white;margin:.55rem 0 .55rem auto;max-width:92%;}
.badge{display:inline-block;padding:.33rem .7rem;border-radius:999px;font-size:.78rem;font-weight:800;margin:.18rem;background:rgba(244,63,94,.20);color:#FDA4AF;border:1px solid rgba(244,63,94,.40);}
.stButton>button{background:linear-gradient(135deg,#27E0A3,#2563EB,#A855F7);color:white;border:none;border-radius:16px;padding:.78rem 1rem;font-weight:900;transition:all .2s ease;}
.stButton>button:hover{transform:translateY(-2px);filter:brightness(1.07);box-shadow:0 14px 36px rgba(37,99,235,.34);}
div[data-testid="stMetric"]{background:rgba(255,255,255,.06);border:1px solid rgba(255,255,255,.10);border-radius:18px;padding:.9rem;}
.footer{color:#9FB1C9;text-align:center;padding:1rem;font-size:.82rem;}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    p = Path(DATA_FILE)
    if not p.exists():
        st.error(f"No se encontró {DATA_FILE}. Sube el Excel al repositorio.")
        st.stop()
    return pd.read_excel(p, sheet_name="Dataset_Analitico_V2")

@st.cache_resource
def train_model(data):
    model = RandomForestClassifier(n_estimators=500, random_state=42, class_weight="balanced", min_samples_leaf=2)
    model.fit(data[FEATURES], data["Codigo_IRAS"])
    importance = pd.DataFrame({"Variable": FEATURES, "Importancia": model.feature_importances_}).sort_values("Importancia", ascending=False)
    return model, importance

df_base = load_data()
model, importance = train_model(df_base)

if "extra_hogares" not in st.session_state: st.session_state.extra_hogares = []
if "auth" not in st.session_state: st.session_state.auth = {"logged": False, "email": "", "name": ""}
if "users" not in st.session_state: st.session_state.users = {"investigador@iras.pe": {"password": "1234", "name": "Investigador"}}
if "chat_history" not in st.session_state: st.session_state.chat_history = []

def data_actual():
    if st.session_state.extra_hogares:
        return pd.concat([df_base, pd.DataFrame(st.session_state.extra_hogares)], ignore_index=True)
    return df_base.copy()

df = data_actual()
districts = sorted(df_base["Distrito"].unique())
district_map = {d:int(df_base[df_base["Distrito"]==d]["Codigo_Distrito"].iloc[0]) for d in districts}

def normalize(v,col):
    mn,mx=float(df_base[col].min()),float(df_base[col].max())
    return 0 if mx==mn else max(0,min(1,(float(v)-mn)/(mx-mn)))

def compute_household(distrito, ingreso, integrantes, personas_ingreso, gasto, comidas, diversidad, laboral, educacion, programa, inflacion):
    ingreso=max(float(ingreso),1); integrantes=max(int(integrantes),1); personas_ingreso=max(int(personas_ingreso),1); gasto=max(float(gasto),0); comidas=int(comidas)
    pct=round(gasto/ingreso*100,2); dep=round((integrantes-personas_ingreso)/personas_ingreso,2); ipc=round(ingreso/integrantes,2); gpc=round(gasto/integrantes,2); disp=round(ingreso-gasto,2)
    cod_div={"Alta":0,"Media":1,"Baja":2}[diversidad]; cod_lab={"Formal":0,"Informal":1,"Desempleado":2}[laboral]; cod_edu={"Superior":0,"Secundaria":1,"Primaria":2}[educacion]
    icva=round((1-normalize(ipc,"Ingreso_Per_Capita"))*.25 + normalize(pct,"Porcentaje_Ingreso_Alimentos")*.25 + normalize(dep,"Indice_Dependencia")*.15 + ((cod_div/2)*.60+max(0,min(1,(3-comidas)/2))*.40)*.20 + (cod_lab/2)*.10 + (0 if programa=="Sí" else 1)*.05,4)
    X=pd.DataFrame([{"Ingreso_Mensual":ingreso,"Integrantes_Hogar":integrantes,"Personas_Con_Ingreso":personas_ingreso,"Gasto_Alimentos":gasto,"Porcentaje_Ingreso_Alimentos":pct,"Frecuencia_Comidas":comidas,"Inflacion_Alimentaria":inflacion,"Indice_Dependencia":dep,"Ingreso_Per_Capita":ipc,"Gasto_Alimentos_Per_Capita":gpc,"Ingreso_Disponible":disp,"Indice_Vulnerabilidad_Alimentaria":icva,"Codigo_Distrito":district_map.get(distrito,0),"Codigo_Diversidad":cod_div,"Codigo_Laboral":cod_lab,"Codigo_Educacion":cod_edu}])
    return X, {"pct":pct,"dependencia":dep,"ingreso_pc":ipc,"gasto_pc":gpc,"disponible":disp,"icva":icva,"cod_div":cod_div,"cod_lab":cod_lab}

def predict(X):
    pred=int(model.predict(X)[0]); risk=IRAS_LABELS[pred]
    probs={IRAS_LABELS[int(c)]:round(float(p)*100,2) for c,p in zip(model.classes_,model.predict_proba(X)[0])}
    return risk, probs

def level_icva(x):
    return "Baja" if x<.30 else "Moderada" if x<.50 else "Alta" if x<.70 else "Crítica"

def priority(r):
    return {"Bajo":"Seguimiento general","Moderado":"Monitoreo preventivo","Alto":"Prevención focalizada","Crítico":"Atención urgente"}[r]

def factors(calc, X):
    fs=[]
    if calc["ingreso_pc"]<df_base["Ingreso_Per_Capita"].quantile(.35): fs.append(("Ingreso per cápita",.31))
    if calc["pct"]>50: fs.append(("% ingreso en alimentos",.24))
    if calc["cod_div"]==2: fs.append(("Diversidad alimentaria",.17))
    if calc["cod_lab"]>=1: fs.append(("Situación laboral",.12))
    if calc["dependencia"]>df_base["Indice_Dependencia"].median(): fs.append(("Dependencia económica",.09))
    if X["Frecuencia_Comidas"].iloc[0]<=2: fs.append(("Frecuencia de comidas",.08))
    return fs or [("Condiciones estables",.20),("Ingreso suficiente",.15)]

def recs(r):
    if r=="Bajo": return ["Mantener seguimiento general.","Educación alimentaria preventiva.","Monitorear precios de alimentos."]
    if r=="Moderado": return ["Monitoreo preventivo.","Planificación del gasto alimentario.","Evaluar apoyo municipal si aumenta la presión."]
    if r=="Alto": return ["Priorizar apoyo alimentario y programas sociales.","Evaluación nutricional preventiva.","Educación alimentaria y financiera.","Seguimiento municipal."]
    return ["Apoyo alimentario inmediato.","Evaluación nutricional en establecimiento de salud.","Seguimiento municipal periódico.","Acceso a programas sociales.","Capacitación laboral y educación alimentaria."]

def health_text(r):
    return {"Bajo":"Riesgo indirecto bajo; menor vulnerabilidad nutricional asociada a inseguridad alimentaria.","Moderado":"Riesgo indirecto moderado; posible exposición a dieta poco variada y deficiencias de micronutrientes.","Alto":"Riesgo indirecto alto; mayor vulnerabilidad a malnutrición, anemia o deterioro de calidad dietaria.","Crítico":"Riesgo indirecto crítico; alta vulnerabilidad a malnutrición, deficiencias de micronutrientes, anemia y bajo bienestar nutricional."}[r]

def local_ai(q, context):
    q=q.lower()
    if "programa" in q or "social" in q: return "Los hogares con IRAS Alto o Crítico deberían priorizarse para programas alimentarios, comedores populares, ollas comunes, apoyo municipal, Vaso de Leche y seguimiento social."
    if "salud" in q or "anemia" in q or "malnutric" in q: return "La inseguridad alimentaria puede asociarse con dietas poco variadas, deficiencias de micronutrientes, anemia y malnutrición. Esta app no diagnostica; estima riesgo indirecto y recomienda seguimiento preventivo."
    if "municipalidad" in q or "distrito" in q: return "La municipalidad debería priorizar distritos y hogares con mayor ICVA, alta presión alimentaria, baja diversidad alimentaria y empleo informal."
    return "Según el análisis IRAS, conviene revisar ingreso per cápita, presión alimentaria, diversidad alimentaria, dependencia económica y acceso a apoyo social."

def gemini_answer(q, context):
    try:
        api_key=st.secrets.get("GEMINI_API_KEY", None)
    except Exception:
        api_key=None
    if api_key:
        try:
            from google import genai
            client=genai.Client(api_key=api_key)
            prompt=f"Eres el Asistente IRAS. Responde en español, profesional y preventivo. No diagnostiques. Contexto: {context}\nPregunta: {q}"
            resp=client.models.generate_content(model="gemini-2.0-flash", contents=prompt)
            return resp.text
        except Exception as e:
            return f"No pude conectar con Gemini. Uso respuesta local.\n\n{local_ai(q, context)}"
    return local_ai(q, context)

def make_pdf(title, risk, probs, calc, recommendations, summary=None):
    buf=io.BytesIO(); c=canvas.Canvas(buf,pagesize=A4); w,h=A4
    c.setFont("Helvetica-Bold",16); c.drawString(1.7*cm,h-1.5*cm,title)
    c.setFont("Helvetica",9); c.drawString(1.7*cm,h-2.1*cm,f"Fecha: {datetime.now().strftime('%d/%m/%Y %H:%M')}")
    y=h-3.2*cm; c.setFont("Helvetica-Bold",12); c.drawString(1.7*cm,y,f"Nivel IRAS: {risk}"); y-=.7*cm
    c.setFont("Helvetica-Bold",11); c.drawString(1.7*cm,y,"Probabilidades"); y-=.45*cm; c.setFont("Helvetica",9)
    for k,v in probs.items(): c.drawString(2*cm,y,f"{k}: {v}%"); y-=.4*cm
    y-=.3*cm; c.setFont("Helvetica-Bold",11); c.drawString(1.7*cm,y,"Cálculos del hogar"); y-=.45*cm; c.setFont("Helvetica",9)
    for k,v in [("Ingreso per cápita",calc["ingreso_pc"]),("Gasto per cápita",calc["gasto_pc"]),("Presión alimentaria",str(calc["pct"])+"%"),("Dependencia",calc["dependencia"]),("Ingreso disponible",calc["disponible"]),("ICVA",calc["icva"])]:
        c.drawString(2*cm,y,f"{k}: {v}"); y-=.4*cm
    y-=.3*cm; c.setFont("Helvetica-Bold",11); c.drawString(1.7*cm,y,"Recomendaciones"); y-=.45*cm; c.setFont("Helvetica",9)
    for i,r in enumerate(recommendations,1): c.drawString(2*cm,y,f"{i}. {r[:85]}"); y-=.4*cm
    if summary:
        y-=.3*cm; c.setFont("Helvetica-Bold",11); c.drawString(1.7*cm,y,"Resumen general"); y-=.45*cm; c.setFont("Helvetica",9)
        for s in summary: c.drawString(2*cm,y,s[:90]); y-=.4*cm
    c.setFont("Helvetica-Oblique",8); c.drawString(1.7*cm,1.4*cm,"IRAS estima riesgo indirecto; no realiza diagnóstico médico.")
    c.showPage(); c.save(); buf.seek(0); return buf

def login_ui():
    st.markdown("### 🔐 Acceso al sistema")
    t1,t2=st.tabs(["Iniciar sesión","Crear cuenta"])
    with t1:
        email=st.text_input("Correo electrónico"); pw=st.text_input("Contraseña",type="password")
        if st.button("Ingresar",use_container_width=True):
            u=st.session_state.users.get(email)
            if u and u["password"]==pw:
                st.session_state.auth={"logged":True,"email":email,"name":u["name"]}; st.rerun()
            else: st.error("Correo o contraseña incorrectos.")
        st.caption("Usuario demo: investigador@iras.pe · Contraseña: 1234")
    with t2:
        name=st.text_input("Nombre"); email2=st.text_input("Correo"); phone=st.text_input("Teléfono opcional"); pw2=st.text_input("Contraseña nueva",type="password")
        if st.button("Crear cuenta",use_container_width=True):
            if name and email2 and pw2:
                st.session_state.users[email2]={"password":pw2,"name":name,"phone":phone}; st.success("Cuenta creada.")
            else: st.warning("Completa nombre, correo y contraseña.")
    st.info("El inicio real con Google o teléfono requiere Firebase/Auth0/Supabase. Esta versión incluye login demo para presentación.")

if not st.session_state.auth["logged"]:
    st.markdown('<div class="hero"><h1>🍽️ Sistema Inteligente IRAS</h1><p>Plataforma premium de análisis de riesgo alimentario y salud.</p></div>',unsafe_allow_html=True)
    login_ui(); st.stop()

with st.sidebar:
    if Path("assets/iras_logo.svg").exists(): st.image("assets/iras_logo.svg", width=95)
    st.markdown("## IRAS"); st.caption("Sistema Inteligente · Lima Metropolitana")
    st.success(f"🟢 {st.session_state.auth['name']}")
    menu=st.radio("Menú",["Inicio","Analizar Hogar","Dashboard","Mapa de Riesgo","IA Asistente","Gestión de Hogares","Reportes"],label_visibility="collapsed")
    if st.button("Cerrar sesión",use_container_width=True):
        st.session_state.auth={"logged":False,"email":"","name":""}; st.rerun()

st.markdown('<div class="hero"><h1>🍽️ Sistema Inteligente IRAS</h1><p>Predicción del Riesgo de Afectación a la Salud por Inseguridad Alimentaria · Lima Metropolitana</p></div>',unsafe_allow_html=True)

def kpis():
    high=int(df["IRAS"].isin(["Alto","Crítico"]).sum())
    cs=st.columns(4)
    vals=[("Hogares analizados",len(df),"Total actualizado"),("Riesgo Alto + Crítico",f"{high} ({high/len(df)*100:.1f}%)","Requieren atención"),("ICVA promedio",f"{df['Indice_Vulnerabilidad_Alimentaria'].mean():.2f}","Índice 0 a 1"),("Distritos",df["Distrito"].nunique(),"Lima Metropolitana")]
    for c,(a,b,d) in zip(cs,vals):
        c.markdown(f'<div class="card"><div class="kpi-title">{a}</div><div class="kpi-value">{b}</div><div class="kpi-caption">{d}</div></div>',unsafe_allow_html=True)

if menu in ["Inicio","Analizar Hogar"]:
    kpis(); st.markdown("")
    left,center,right=st.columns([1.15,1.25,1.0])
    with left:
        st.markdown('<div class="form-card">',unsafe_allow_html=True)
        st.markdown('<div class="step-title"><span class="step-dot">1</span>Información del Hogar</div>',unsafe_allow_html=True)
        distrito=st.selectbox("Distrito",districts,index=districts.index("San Juan de Lurigancho") if "San Juan de Lurigancho" in districts else 0)
        ingreso=st.number_input("Ingreso mensual del hogar (S/.)",value=1800.0,step=50.0,min_value=1.0)
        a,b=st.columns(2); integrantes=a.number_input("Número de integrantes",value=5,min_value=1,max_value=12); personas=b.number_input("Personas con ingreso",value=1,min_value=1,max_value=6)
        gasto=st.number_input("Gasto en alimentos (S/.)",value=950.0,step=25.0)
        c,d=st.columns(2); comidas=c.selectbox("Frecuencia de comidas al día",[1,2,3,4],index=1); diversidad=d.selectbox("Diversidad alimentaria",["Alta","Media","Baja"],index=2)
        e,f=st.columns(2); laboral=e.selectbox("Situación laboral",["Formal","Informal","Desempleado"],index=1); educacion=f.selectbox("Nivel educativo",["Superior","Secundaria","Primaria"],index=1)
        programa=st.radio("Recibe programa social",["Sí","No"],horizontal=True,index=1); inflacion=st.number_input("Inflación alimentaria (%)",value=5.1,step=.1)
        add_house=st.checkbox("Agregar este hogar al dataset temporal"); analyze=st.button("🔎 Analizar Hogar",use_container_width=True)
        st.markdown('</div>',unsafe_allow_html=True)
    if analyze:
        X,calc=compute_household(distrito,ingreso,integrantes,personas,gasto,comidas,diversidad,laboral,educacion,programa,inflacion); risk,probs=predict(X); result={"X":X,"calc":calc,"risk":risk,"probs":probs,"recs":recs(risk),"facs":factors(calc,X),"distrito":distrito}
        st.session_state.last_result=result
        if add_house:
            new_id=f"N{len(st.session_state.extra_hogares)+1:03d}"
            st.session_state.extra_hogares.append({"ID_Hogar":new_id,"Distrito":distrito,"Ingreso_Mensual":ingreso,"Integrantes_Hogar":integrantes,"Personas_Con_Ingreso":personas,"Gasto_Alimentos":gasto,"Porcentaje_Ingreso_Alimentos":calc["pct"],"Frecuencia_Comidas":comidas,"Diversidad_Alimentaria":diversidad,"Reduccion_Consumo":"Sí" if calc["pct"]>55 else "No","Situacion_Laboral":laboral,"Nivel_Educativo":educacion,"Programa_Social":programa,"Inflacion_Alimentaria":inflacion,"Indice_Dependencia":calc["dependencia"],"IRAS":risk,"Ingreso_Per_Capita":calc["ingreso_pc"],"Gasto_Alimentos_Per_Capita":calc["gasto_pc"],"Ingreso_Disponible":calc["disponible"],"Indice_Vulnerabilidad_Alimentaria":calc["icva"],"Nivel_Vulnerabilidad":level_icva(calc["icva"]),"Codigo_Distrito":X["Codigo_Distrito"].iloc[0],"Codigo_Diversidad":X["Codigo_Diversidad"].iloc[0],"Codigo_Laboral":X["Codigo_Laboral"].iloc[0],"Codigo_Educacion":X["Codigo_Educacion"].iloc[0],"Codigo_IRAS":IRAS_NUM[risk],"PC1":np.nan,"PC2":np.nan,"Cluster_Hogar":np.nan,"Perfil_Cluster":"Nuevo análisis","Prioridad_Intervencion":priority(risk)})
            st.success("Hogar agregado al dataset temporal.")
    if "last_result" not in st.session_state:
        X,calc=compute_household(districts[0],1800,5,1,950,2,"Baja","Informal","Secundaria","No",5.1); risk,probs=predict(X); st.session_state.last_result={"X":X,"calc":calc,"risk":risk,"probs":probs,"recs":recs(risk),"facs":factors(calc,X),"distrito":districts[0]}
    r=st.session_state.last_result
    with center:
        st.markdown('<div class="card">',unsafe_allow_html=True); st.markdown('<div class="step-title"><span class="step-dot">2</span>Resultado del Análisis</div>',unsafe_allow_html=True)
        st.markdown(f'<div style="text-align:center;"><div style="font-weight:800;color:#CBD5E1;">NIVEL DE RIESGO IRAS</div><div class="risk-pill">{r["risk"].upper()}</div><div style="font-size:2.3rem;font-weight:900;color:#F43F5E;">{max(r["probs"].values()):.0f}%</div><div>Probabilidad máxima</div></div>',unsafe_allow_html=True)
        m1,m2,m3,m4=st.columns(4); m1.metric("ICVA",f'{r["calc"]["icva"]:.2f}'); m2.metric("Ingreso per cápita",f'S/ {r["calc"]["ingreso_pc"]:.0f}'); m3.metric("% alimentos",f'{r["calc"]["pct"]:.1f}%'); m4.metric("Dependencia",f'{r["calc"]["dependencia"]:.1f}')
        st.write("**Interpretación:**",f"El hogar presenta riesgo **{r['risk']}** asociado a inseguridad alimentaria."); st.warning(health_text(r["risk"])); st.markdown('<span class="badge">Malnutrición</span><span class="badge">Micronutrientes</span><span class="badge">Anemia</span><span class="badge">Bienestar nutricional</span>',unsafe_allow_html=True); st.markdown('</div>',unsafe_allow_html=True)
    with right:
        st.markdown('<div class="chat-panel">',unsafe_allow_html=True); st.markdown("### 🤖 IA Asistente con Gemini"); st.markdown('<div class="ai-bubble">Puedo responder preguntas sobre el hogar, salud pública, programas sociales, nutrición y acciones municipales.</div>',unsafe_allow_html=True)
        q=st.text_area("Pregunta","¿Qué debería priorizar la municipalidad con este hogar?")
        if st.button("Preguntar a la IA",use_container_width=True):
            context=json.dumps({"riesgo":r["risk"],"probabilidades":r["probs"],"calculos":r["calc"],"distrito":r["distrito"]},ensure_ascii=False)
            ans=gemini_answer(q,context); st.session_state.chat_history.append(("user",q)); st.session_state.chat_history.append(("ai",ans))
        for role,msg in st.session_state.chat_history[-4:]:
            st.markdown(f'<div class="{"user-bubble" if role=="user" else "ai-bubble"}">{msg}</div>',unsafe_allow_html=True)
        st.markdown('</div>',unsafe_allow_html=True)
    b1,b2,b3,b4=st.columns([1.1,1.1,1.25,.9])
    with b1:
        st.markdown('<div class="card">',unsafe_allow_html=True); p=pd.DataFrame({"Nivel":list(r["probs"].keys()),"Probabilidad":list(r["probs"].values())}); fig=px.bar(p,x="Nivel",y="Probabilidad",color="Nivel",color_discrete_map=IRAS_COLORS,text="Probabilidad",template="plotly_dark"); fig.update_layout(height=275,showlegend=False,margin=dict(l=10,r=10,t=30,b=10)); st.plotly_chart(fig,use_container_width=True); st.markdown('</div>',unsafe_allow_html=True)
    with b2:
        st.markdown('<div class="card">',unsafe_allow_html=True); ff=pd.DataFrame(r["facs"],columns=["Factor","Importancia"]); fig=px.bar(ff.sort_values("Importancia"),x="Importancia",y="Factor",orientation="h",template="plotly_dark"); fig.update_layout(height=275,margin=dict(l=10,r=10,t=30,b=10)); st.plotly_chart(fig,use_container_width=True); st.markdown('</div>',unsafe_allow_html=True)
    with b3:
        st.markdown('<div class="card">',unsafe_allow_html=True); st.markdown("### Recomendaciones"); [st.write(f"**{i}.** {rr}") for i,rr in enumerate(r["recs"],1)]; st.markdown('</div>',unsafe_allow_html=True)
    with b4:
        st.markdown('<div class="card">',unsafe_allow_html=True); summary=[f"Hogares totales: {len(df)}",f"Alto/Crítico: {int(df['IRAS'].isin(['Alto','Crítico']).sum())}",f"ICVA promedio: {df['Indice_Vulnerabilidad_Alimentaria'].mean():.2f}"]; pdf=make_pdf("Reporte IRAS del hogar",r["risk"],r["probs"],r["calc"],r["recs"],summary); st.download_button("📄 Descargar PDF",pdf,"Reporte_IRAS_Hogar.pdf","application/pdf",use_container_width=True); st.download_button("📊 Exportar Dataset",df.to_csv(index=False).encode("utf-8-sig"),"IRAS_actualizado.csv","text/csv",use_container_width=True); st.markdown('</div>',unsafe_allow_html=True)

elif menu=="Dashboard":
    kpis(); st.markdown("### Dashboard ejecutivo actualizado")
    c1,c2=st.columns(2)
    with c1:
        d=df["IRAS"].value_counts().reset_index(); d.columns=["IRAS","Hogares"]; fig=px.pie(d,names="IRAS",values="Hogares",hole=.45,color="IRAS",color_discrete_map=IRAS_COLORS,template="plotly_dark"); st.plotly_chart(fig,use_container_width=True)
    with c2:
        resumen=df.groupby("Distrito")["Indice_Vulnerabilidad_Alimentaria"].mean().sort_values(ascending=False).head(12).reset_index(); fig=px.bar(resumen,y="Distrito",x="Indice_Vulnerabilidad_Alimentaria",orientation="h",template="plotly_dark",color="Indice_Vulnerabilidad_Alimentaria",color_continuous_scale="Turbo"); st.plotly_chart(fig,use_container_width=True)
    st.dataframe(df,use_container_width=True)

elif menu=="Mapa de Riesgo":
    st.markdown("## 🗺️ Mapa de riesgo por distrito en tiempo real"); st.caption("Se actualiza con hogares agregados durante la sesión. Para mapa geográfico exacto se puede integrar GeoJSON municipal.")
    resumen=df.groupby("Distrito").agg(Vulnerabilidad=("Indice_Vulnerabilidad_Alimentaria","mean"),Hogares=("ID_Hogar","count"),Riesgo=("Codigo_IRAS","mean")).sort_values("Vulnerabilidad").reset_index()
    fig=px.bar(resumen,y="Distrito",x="Vulnerabilidad",color="Riesgo",orientation="h",template="plotly_dark",color_continuous_scale=["#6EE7B7","#FACC15","#FB923C","#F43F5E"]); fig.update_layout(height=680); st.plotly_chart(fig,use_container_width=True)

elif menu=="IA Asistente":
    st.markdown("## 🤖 IA Asistente con Gemini"); st.info("Para activar Gemini real, agrega GEMINI_API_KEY en Secrets de Streamlit Cloud.")
    q=st.text_area("Escribe cualquier pregunta","¿Qué políticas públicas ayudarían a reducir el riesgo alimentario en Lima Metropolitana?")
    if st.button("Responder con IA"):
        context=f"Dataset actual con {len(df)} hogares. ICVA promedio {df['Indice_Vulnerabilidad_Alimentaria'].mean():.2f}. Hogares Alto/Crítico: {int(df['IRAS'].isin(['Alto','Crítico']).sum())}."
        st.markdown(gemini_answer(q,context))

elif menu=="Gestión de Hogares":
    st.markdown("## ➕ Gestión de hogares analizados"); st.write("Los nuevos hogares se conservan durante la sesión y actualizan dashboards, mapa y reportes.")
    st.dataframe(df,use_container_width=True); st.download_button("Descargar dataset actualizado CSV",df.to_csv(index=False).encode("utf-8-sig"),"IRAS_dataset_actualizado.csv","text/csv")

elif menu=="Reportes":
    st.markdown("## 📄 Reportes"); high=int(df["IRAS"].isin(["Alto","Crítico"]).sum()); fake={"ingreso_pc":round(df["Ingreso_Per_Capita"].mean(),2),"gasto_pc":round(df["Gasto_Alimentos_Per_Capita"].mean(),2),"pct":round(df["Porcentaje_Ingreso_Alimentos"].mean(),2),"dependencia":round(df["Indice_Dependencia"].mean(),2),"disponible":round(df["Ingreso_Disponible"].mean(),2),"icva":round(df["Indice_Vulnerabilidad_Alimentaria"].mean(),4)}
    probs=(df["IRAS"].value_counts(normalize=True)*100).round(2).to_dict(); recomendaciones=["Priorizar distritos con mayor ICVA.","Focalizar hogares con IRAS Alto/Crítico.","Monitorear presión alimentaria y diversidad dietaria.","Articular municipios con programas sociales."]; pdf=make_pdf("Reporte general IRAS","Resumen general",probs,fake,recomendaciones,[f"Hogares: {len(df)}",f"Alto/Crítico: {high}",f"Nuevos en sesión: {len(st.session_state.extra_hogares)}"]); st.download_button("Descargar Reporte General PDF",pdf,"Reporte_General_IRAS.pdf","application/pdf"); st.download_button("Descargar Dataset CSV",df.to_csv(index=False).encode("utf-8-sig"),"Dataset_IRAS_actualizado.csv","text/csv")

st.markdown('<div class="footer">© 2024 Sistema Inteligente IRAS · Ciencia de Datos · Streamlit + Machine Learning + Gemini</div>',unsafe_allow_html=True)
