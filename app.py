
from pathlib import Path
from datetime import datetime
import io, json, math
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import cm
from reportlab.lib.colors import HexColor

st.set_page_config(page_title="IRAS Intelligence Platform", page_icon="🍽️", layout="wide", initial_sidebar_state="expanded")

DATA_FILE = "02_dataset_analitico_IRAS_Lima_150_V2.xlsx"
if not Path(DATA_FILE).exists():
    DATA_FILE = "data/02_dataset_analitico_IRAS_Lima_150_V2.xlsx"

IRAS_LABELS = {0:"Bajo", 1:"Moderado", 2:"Alto", 3:"Crítico"}
IRAS_NUM = {v:k for k,v in IRAS_LABELS.items()}
IRAS_COLORS = {"Bajo":"#00E5A8", "Moderado":"#FFB703", "Alto":"#FB8500", "Crítico":"#E63946"}
FEATURES = ["Ingreso_Mensual","Integrantes_Hogar","Personas_Con_Ingreso","Gasto_Alimentos","Porcentaje_Ingreso_Alimentos","Frecuencia_Comidas","Inflacion_Alimentaria","Indice_Dependencia","Ingreso_Per_Capita","Gasto_Alimentos_Per_Capita","Ingreso_Disponible","Indice_Vulnerabilidad_Alimentaria","Codigo_Distrito","Codigo_Diversidad","Codigo_Laboral","Codigo_Educacion"]
SCOPE = ["inseguridad", "alimentaria", "alimentos", "salud", "anemia", "malnutric", "nutric", "lima", "distrito", "municipalidad", "programa", "social", "midis", "inei", "fao", "oms", "wfp", "iras", "icva", "hogar", "riesgo", "pobreza", "ingreso", "gasto", "dependencia", "diversidad", "comidas", "política", "publica", "pública"]

# -------------------- CSS --------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');
html, body, [class*="css"]{font-family:Inter,sans-serif}
.stApp{background:
radial-gradient(circle at 15% 8%, rgba(0,229,168,.23), transparent 24%),
radial-gradient(circle at 88% 8%, rgba(255,183,3,.23), transparent 27%),
radial-gradient(circle at 70% 92%, rgba(56,189,248,.14), transparent 31%),
linear-gradient(135deg,#061A33 0%,#08223E 48%,#071122 100%)}
section[data-testid="stSidebar"]{background:linear-gradient(180deg,#031124,#061A33,#071320);border-right:1px solid rgba(255,255,255,.1)}
.block-container{padding-top:1rem;padding-bottom:1.5rem}.hero{border-radius:30px;padding:1.35rem 1.7rem;background:linear-gradient(135deg,rgba(6,26,51,.94),rgba(14,40,68,.86));border:1px solid rgba(255,255,255,.14);box-shadow:0 24px 60px rgba(0,0,0,.32);position:relative;overflow:hidden;margin-bottom:1rem}
.hero:before{content:"";position:absolute;width:440px;height:440px;top:-240px;right:-160px;background:radial-gradient(circle,rgba(255,183,3,.42),transparent 60%);animation:pulse 4.5s infinite alternate ease-in-out}.hero:after{content:"";position:absolute;width:340px;height:340px;bottom:-200px;left:18%;background:radial-gradient(circle,rgba(0,229,168,.3),transparent 60%);animation:float 6s infinite alternate ease-in-out}
@keyframes pulse{from{transform:scale(1);opacity:.55}to{transform:scale(1.18);opacity:.95}}@keyframes float{from{transform:translateY(0)}to{transform:translateY(-34px)}}@keyframes soft{from{transform:scale(1);filter:brightness(1)}to{transform:scale(1.025);filter:brightness(1.12)}}
.hero h1{color:#F8FAFC;margin:0;font-weight:900;font-size:2.1rem;position:relative;z-index:2}.hero p{color:#C9D9EC;margin:.25rem 0 0 0;position:relative;z-index:2}
.card,.form-card,.chat-panel{background:linear-gradient(180deg,rgba(14,40,68,.96),rgba(7,24,43,.96));border:1px solid rgba(255,255,255,.12);border-radius:26px;padding:1rem;box-shadow:0 18px 42px rgba(0,0,0,.28);transition:all .22s ease}.card:hover,.form-card:hover,.chat-panel:hover{transform:translateY(-3px);border-color:rgba(255,183,3,.42)}
.kpi-title{color:#C9D9EC;font-size:.77rem;text-transform:uppercase;font-weight:900}.kpi-value{color:#F8FAFC;font-size:2rem;font-weight:900}.kpi-caption{color:#9FB2CC;font-size:.85rem}
.step-dot{display:inline-flex;justify-content:center;align-items:center;width:34px;height:34px;margin-right:.48rem;border-radius:999px;background:linear-gradient(135deg,#00E5A8,#FFB703);color:#061A33;font-weight:900}.step-title{font-size:1.05rem;font-weight:900;color:#F8FAFC;margin-bottom:.65rem}
.risk-pill{display:inline-block;padding:.7rem 2.1rem;border-radius:17px;font-size:2rem;font-weight:900;color:white;background:linear-gradient(135deg,#E63946,#FF6B6B);box-shadow:0 0 42px rgba(230,57,70,.36);animation:soft 1.8s infinite alternate ease-in-out}
.ai-bubble{background:rgba(255,255,255,.08);border:1px solid rgba(255,255,255,.1);border-radius:18px 18px 18px 7px;padding:.85rem .95rem;color:#E2E8F0;margin:.55rem 0}.user-bubble{background:linear-gradient(135deg,#2563EB,#A855F7);border-radius:18px 18px 7px 18px;padding:.85rem .95rem;color:white;margin:.55rem 0 .55rem auto;max-width:92%}
.badge,.solution{display:block;border-radius:16px;margin:.45rem 0;padding:.7rem .85rem;background:linear-gradient(135deg,rgba(0,229,168,.12),rgba(255,183,3,.12));border:1px solid rgba(255,255,255,.1)}
.inline-badge{display:inline-block;border-radius:999px;margin:.16rem;padding:.32rem .75rem;background:rgba(255,183,3,.16);border:1px solid rgba(255,183,3,.35);color:#FDE68A;font-weight:800;font-size:.78rem}
.stButton>button{background:linear-gradient(135deg,#00E5A8,#38BDF8,#FFB703);color:#061A33;border:none;border-radius:16px;padding:.78rem 1rem;font-weight:900}.footer{color:#9FB2CC;text-align:center;padding:1rem;font-size:.82rem}
</style>
""", unsafe_allow_html=True)

# -------------------- Data/model --------------------
@st.cache_data
def load_data():
    p=Path(DATA_FILE)
    if not p.exists():
        st.error("No se encontró el dataset Excel. Sube 02_dataset_analitico_IRAS_Lima_150_V2.xlsx al repositorio.")
        st.stop()
    return pd.read_excel(p, sheet_name="Dataset_Analitico_V2")

@st.cache_resource
def train_rf(data):
    m=RandomForestClassifier(n_estimators=550, random_state=42, class_weight="balanced", min_samples_leaf=2)
    m.fit(data[FEATURES], data["Codigo_IRAS"])
    imp=pd.DataFrame({"Variable":FEATURES,"Importancia":m.feature_importances_}).sort_values("Importancia",ascending=False)
    return m, imp

df_base=load_data()
model, importance=train_rf(df_base)

if "extra" not in st.session_state: st.session_state.extra=[]
if "auth" not in st.session_state: st.session_state.auth={"logged":False,"email":"","name":""}
if "users" not in st.session_state: st.session_state.users={"investigador@iras.pe":{"password":"1234","name":"Investigador","role":"Investigador"}}
if "chat" not in st.session_state: st.session_state.chat=[]

def get_df():
    return pd.concat([df_base,pd.DataFrame(st.session_state.extra)], ignore_index=True) if st.session_state.extra else df_base.copy()

df=get_df()
districts=sorted(df_base["Distrito"].unique())
district_map={d:int(df_base[df_base["Distrito"]==d]["Codigo_Distrito"].iloc[0]) for d in districts}

def norm(v,col):
    mn,mx=float(df_base[col].min()),float(df_base[col].max())
    return 0 if mx==mn else max(0,min(1,(float(v)-mn)/(mx-mn)))

def compute(distrito,ingreso,integrantes,personas,gasto,comidas,diversidad,laboral,educacion,programa,inflacion):
    ingreso=max(float(ingreso),1); integrantes=max(int(integrantes),1); personas=max(int(personas),1); gasto=max(float(gasto),0); comidas=int(comidas)
    pct=round(gasto/ingreso*100,2); dep=round((integrantes-personas)/personas,2); ipc=round(ingreso/integrantes,2); gpc=round(gasto/integrantes,2); disp=round(ingreso-gasto,2)
    cd={"Alta":0,"Media":1,"Baja":2}[diversidad]; cl={"Formal":0,"Informal":1,"Desempleado":2}[laboral]; ce={"Superior":0,"Secundaria":1,"Primaria":2}[educacion]
    icva=round((1-norm(ipc,"Ingreso_Per_Capita"))*.25+norm(pct,"Porcentaje_Ingreso_Alimentos")*.25+norm(dep,"Indice_Dependencia")*.15+((cd/2)*.60+max(0,min(1,(3-comidas)/2))*.40)*.20+(cl/2)*.10+(0 if programa=="Sí" else 1)*.05,4)
    X=pd.DataFrame([{"Ingreso_Mensual":ingreso,"Integrantes_Hogar":integrantes,"Personas_Con_Ingreso":personas,"Gasto_Alimentos":gasto,"Porcentaje_Ingreso_Alimentos":pct,"Frecuencia_Comidas":comidas,"Inflacion_Alimentaria":inflacion,"Indice_Dependencia":dep,"Ingreso_Per_Capita":ipc,"Gasto_Alimentos_Per_Capita":gpc,"Ingreso_Disponible":disp,"Indice_Vulnerabilidad_Alimentaria":icva,"Codigo_Distrito":district_map.get(distrito,0),"Codigo_Diversidad":cd,"Codigo_Laboral":cl,"Codigo_Educacion":ce}])
    return X, {"pct":pct,"dependencia":dep,"ingreso_pc":ipc,"gasto_pc":gpc,"disponible":disp,"icva":icva,"cod_div":cd,"cod_lab":cl}

def predict(X):
    pred=int(model.predict(X)[0]); risk=IRAS_LABELS[pred]
    probs={IRAS_LABELS[int(c)]:round(float(p)*100,2) for c,p in zip(model.classes_,model.predict_proba(X)[0])}
    return risk,probs

def level(x): return "Baja" if x<.30 else "Moderada" if x<.50 else "Alta" if x<.70 else "Crítica"
def priority(r): return {"Bajo":"Seguimiento general","Moderado":"Monitoreo preventivo","Alto":"Prevención focalizada","Crítico":"Atención urgente"}[r]

def facs(calc,X):
    fs=[]
    if calc["ingreso_pc"]<df_base["Ingreso_Per_Capita"].quantile(.35): fs.append(("Ingreso per cápita bajo",.31))
    if calc["pct"]>50: fs.append(("Alta presión alimentaria",.24))
    if calc["cod_div"]==2: fs.append(("Baja diversidad alimentaria",.17))
    if calc["cod_lab"]>=1: fs.append(("Inestabilidad laboral",.12))
    if calc["dependencia"]>df_base["Indice_Dependencia"].median(): fs.append(("Dependencia económica elevada",.09))
    if X["Frecuencia_Comidas"].iloc[0]<=2: fs.append(("Frecuencia reducida de comidas",.08))
    return fs or [("Condiciones estables",.20)]

def household_recs(r,calc):
    d={"Bajo":["Seguimiento general del hogar.","Educación alimentaria preventiva.","Monitoreo de precios de alimentos."],
       "Moderado":["Monitoreo preventivo.","Planificación del gasto alimentario.","Evaluar apoyo municipal si aumenta la presión alimentaria."],
       "Alto":["Priorizar apoyo alimentario y programas sociales.","Evaluación nutricional preventiva.","Educación alimentaria y financiera.","Seguimiento municipal mensual."],
       "Crítico":["Activar apoyo alimentario inmediato.","Derivar a evaluación nutricional preventiva.","Seguimiento municipal periódico.","Evaluar acceso a programas sociales o comunitarios.","Capacitación laboral y educación alimentaria."]}[r]
    if calc["pct"]>55: d.append("Reducir presión alimentaria con canastas, comedores u ollas comunes focalizadas.")
    return d

def muni_recs(data):
    res=data.groupby("Distrito").agg(Hogares=("ID_Hogar","count"),ICVA=("Indice_Vulnerabilidad_Alimentaria","mean"),Riesgo=("Codigo_IRAS","mean"),Presion=("Porcentaje_Ingreso_Alimentos","mean"),IngresoPC=("Ingreso_Per_Capita","mean")).sort_values("ICVA", ascending=False)
    top=res.index[0]; val=res.iloc[0]["ICVA"]
    rec=[f"Priorizar el distrito {top}, porque tiene el mayor ICVA promedio ({val:.2f}).",
         "Empadronar hogares con IRAS Alto y Crítico.",
         "Articular apoyo alimentario con comedores, ollas comunes y programas sociales.",
         "Coordinar campañas preventivas de evaluación nutricional.",
         "Ejecutar talleres de educación alimentaria y planificación del gasto.",
         "Monitorear mensualmente presión alimentaria, diversidad de dieta e ingreso per cápita."]
    return top,rec,res

def health(r):
    return {"Bajo":"Riesgo indirecto bajo; menor vulnerabilidad nutricional asociada a inseguridad alimentaria.","Moderado":"Riesgo indirecto moderado; posible exposición a dieta poco variada y deficiencias de micronutrientes.","Alto":"Riesgo indirecto alto; mayor vulnerabilidad a malnutrición, anemia o deterioro de calidad dietaria.","Crítico":"Riesgo indirecto crítico; alta vulnerabilidad a malnutrición, deficiencias de micronutrientes, anemia y bajo bienestar nutricional."}[r]

def in_scope(q): return any(k in q.lower() for k in SCOPE)

def local_ai(q,ctx):
    if not in_scope(q): return "Solo puedo responder preguntas relacionadas con el proyecto IRAS: inseguridad alimentaria, salud pública, hogares, distritos, programas sociales, nutrición, municipalidades, indicadores, modelos o recomendaciones."
    if "municipal" in q.lower() or "distrito" in q.lower():
        top,rec,_=muni_recs(df); return f"El distrito prioritario actual es {top}. Recomendaciones: " + " ".join(rec[:4])
    if "salud" in q.lower() or "anemia" in q.lower() or "malnutric" in q.lower():
        return "La inseguridad alimentaria puede asociarse con dietas poco diversas, deficiencias de micronutrientes, anemia y malnutrición. El sistema estima riesgo indirecto y recomienda seguimiento preventivo."
    return "La decisión debe basarse en IRAS, ICVA, presión alimentaria, ingreso per cápita, diversidad alimentaria, dependencia económica y situación laboral."

def gemini(q,ctx):
    if not in_scope(q): return local_ai(q,ctx)
    try: key=st.secrets.get("GEMINI_API_KEY",None)
    except Exception: key=None
    if key:
        try:
            from google import genai
            client=genai.Client(api_key=key)
            prompt=f"Eres el Asistente IRAS. Responde solo sobre inseguridad alimentaria, salud pública, hogares, Lima Metropolitana, municipalidades, programas sociales, IRAS e ICVA. No diagnostiques enfermedades. Contexto: {ctx}. Pregunta: {q}"
            return client.models.generate_content(model="gemini-2.0-flash", contents=prompt).text
        except Exception:
            return "No pude conectar con Gemini. Respuesta local: " + local_ai(q,ctx)
    return local_ai(q,ctx)

def pdf_report(title,risk,probs,calc,recs,summary=None):
    buf=io.BytesIO(); c=canvas.Canvas(buf,pagesize=A4); w,h=A4
    c.setFillColor(HexColor("#061A33")); c.rect(0,h-3.1*cm,w,3.1*cm,fill=1,stroke=0)
    c.setFillColor(HexColor("#F8FAFC")); c.setFont("Helvetica-Bold",16); c.drawString(1.7*cm,h-1.35*cm,title)
    c.setFont("Helvetica",9); c.drawString(1.7*cm,h-2*cm,f"Fecha: {datetime.now().strftime('%d/%m/%Y %H:%M')}")
    y=h-3.8*cm; c.setFillColor(HexColor("#111827")); c.setFont("Helvetica-Bold",12); c.drawString(1.7*cm,y,f"Nivel IRAS: {risk}"); y-=.6*cm
    c.setFont("Helvetica-Bold",10); c.drawString(1.7*cm,y,"Probabilidades"); y-=.4*cm; c.setFont("Helvetica",9)
    for k,v in probs.items(): c.drawString(2*cm,y,f"{k}: {v}%"); y-=.35*cm
    y-=.25*cm; c.setFont("Helvetica-Bold",10); c.drawString(1.7*cm,y,"Cálculos"); y-=.4*cm; c.setFont("Helvetica",9)
    for k,v in [("Ingreso per cápita",calc["ingreso_pc"]),("Gasto pc",calc["gasto_pc"]),("Presión alimentaria",str(calc["pct"])+"%"),("Dependencia",calc["dependencia"]),("Disponible",calc["disponible"]),("ICVA",calc["icva"])]:
        c.drawString(2*cm,y,f"{k}: {v}"); y-=.35*cm
    y-=.25*cm; c.setFont("Helvetica-Bold",10); c.drawString(1.7*cm,y,"Recomendaciones"); y-=.4*cm; c.setFont("Helvetica",8.5)
    for i,rec in enumerate(recs,1): c.drawString(2*cm,y,f"{i}. {rec[:92]}"); y-=.35*cm
    if summary:
        y-=.25*cm; c.setFont("Helvetica-Bold",10); c.drawString(1.7*cm,y,"Resumen territorial"); y-=.4*cm; c.setFont("Helvetica",8.5)
        for s in summary[:5]: c.drawString(2*cm,y,str(s)[:92]); y-=.35*cm
    c.setFont("Helvetica-Oblique",8); c.drawString(1.7*cm,1.4*cm,"IRAS estima riesgo indirecto; no realiza diagnóstico médico. Uso académico y preventivo.")
    c.showPage(); c.save(); buf.seek(0); return buf

def login():
    st.markdown('<div class="hero"><h1>🍽️ IRAS Intelligence Platform</h1><p>Plataforma para apoyar decisiones frente a inseguridad alimentaria en Lima Metropolitana.</p></div>',unsafe_allow_html=True)
    st.info("Login real con Google o teléfono requiere Firebase/Auth0/Supabase. Esta versión incluye demostración de inicio de sesión para presentación.")
    t1,t2,t3=st.tabs(["Iniciar sesión","Crear cuenta demo","Autenticación profesional"])
    with t1:
        e=st.text_input("Correo electrónico"); p=st.text_input("Contraseña",type="password")
        if st.button("Ingresar",use_container_width=True):
            u=st.session_state.users.get(e)
            if u and u["password"]==p: st.session_state.auth={"logged":True,"email":e,"name":u["name"],"role":u.get("role","Investigador")}; st.rerun()
            else: st.error("Correo o contraseña incorrectos.")
        st.caption("Usuario demo: investigador@iras.pe · Contraseña: 1234")
    with t2:
        n=st.text_input("Nombre completo"); e2=st.text_input("Correo"); rol=st.selectbox("Rol",["Investigador","Municipalidad","Operador territorial"]); tel=st.text_input("Teléfono opcional"); p2=st.text_input("Contraseña nueva",type="password")
        if st.button("Crear cuenta demo",use_container_width=True):
            if n and e2 and p2: st.session_state.users[e2]={"password":p2,"name":n,"phone":tel,"role":rol}; st.success("Cuenta demo creada.")
            else: st.warning("Completa nombre, correo y contraseña.")
    with t3:
        st.markdown("- Google Login: Firebase/Auth0/Supabase.\n- Teléfono/SMS: Firebase Authentication.\n- Base persistente: Supabase/PostgreSQL o Firestore.\n- Roles: administrador, municipalidad, investigador y operador territorial.")

if not st.session_state.auth["logged"]:
    login(); st.stop()

with st.sidebar:
    if Path("assets/iras_logo.svg").exists(): st.image("assets/iras_logo.svg", width=92)
    st.markdown("## IRAS Intelligence"); st.caption("Sistema Inteligente · Lima Metropolitana"); st.success(f"🟢 {st.session_state.auth['name']}")
    menu=st.radio("Menú",["Inicio","Dashboard Ejecutivo","Mapa Inteligente","Analizar Hogar","Gestión de Hogares","Analítica Avanzada","Machine Learning","IA Especialista IRAS","Panel Municipal","Reportes Ejecutivos","Base de Conocimiento","Perfil y Configuración"],label_visibility="collapsed")
    if st.button("Cerrar sesión",use_container_width=True): st.session_state.auth={"logged":False,"email":"","name":""}; st.rerun()

st.markdown('<div class="hero"><h1>🍽️ IRAS Intelligence Platform</h1><p>Predicción, explicación y recomendaciones frente al riesgo alimentario y de salud.</p></div>',unsafe_allow_html=True)

def kpis():
    high=int(df["IRAS"].isin(["Alto","Crítico"]).sum()); top,_,_=muni_recs(df)
    vals=[("Hogares analizados",len(df),"Dataset actualizado"),("Riesgo Alto + Crítico",f"{high} ({high/len(df)*100:.1f}%)","Requieren atención"),("ICVA promedio",f"{df['Indice_Vulnerabilidad_Alimentaria'].mean():.2f}","Índice 0 a 1"),("Distrito prioritario",top,"Mayor vulnerabilidad")]
    cols=st.columns(4)
    for c,(a,b,d) in zip(cols,vals): c.markdown(f'<div class="card"><div class="kpi-title">{a}</div><div class="kpi-value">{b}</div><div class="kpi-caption">{d}</div></div>',unsafe_allow_html=True)

if menu in ["Inicio","Dashboard Ejecutivo"]:
    kpis(); c1,c2=st.columns([1.05,1.15])
    with c1:
        d=df["IRAS"].value_counts().reset_index(); d.columns=["IRAS","Hogares"]; st.plotly_chart(px.pie(d,names="IRAS",values="Hogares",hole=.45,color="IRAS",color_discrete_map=IRAS_COLORS,template="plotly_dark"),use_container_width=True)
    with c2:
        res=df.groupby("Distrito")["Indice_Vulnerabilidad_Alimentaria"].mean().sort_values(ascending=False).head(12).reset_index(); st.plotly_chart(px.bar(res,y="Distrito",x="Indice_Vulnerabilidad_Alimentaria",orientation="h",template="plotly_dark",color="Indice_Vulnerabilidad_Alimentaria",color_continuous_scale="Turbo"),use_container_width=True)
    b1,b2,b3=st.columns(3)
    b1.metric("Ingreso per cápita promedio", f"S/ {df['Ingreso_Per_Capita'].mean():.0f}")
    b2.metric("Presión alimentaria promedio", f"{df['Porcentaje_Ingreso_Alimentos'].mean():.1f}%")
    b3.metric("Dependencia promedio", f"{df['Indice_Dependencia'].mean():.2f}")

elif menu=="Mapa Inteligente":
    st.markdown("## 🗺️ Mapa inteligente de Lima Metropolitana")
    st.info("Para mapa geográfico exacto se puede integrar GeoJSON municipal. Esta versión muestra ranking territorial en tiempo real actualizado con hogares agregados.")
    res=df.groupby("Distrito").agg(Vulnerabilidad=("Indice_Vulnerabilidad_Alimentaria","mean"),Hogares=("ID_Hogar","count"),Riesgo=("Codigo_IRAS","mean")).sort_values("Vulnerabilidad").reset_index()
    fig=px.bar(res,y="Distrito",x="Vulnerabilidad",color="Riesgo",orientation="h",template="plotly_dark",color_continuous_scale=["#00E5A8","#FFB703","#FB8500","#E63946"]); fig.update_layout(height=690); st.plotly_chart(fig,use_container_width=True)
    st.dataframe(res.sort_values("Vulnerabilidad", ascending=False).round(3), use_container_width=True)

elif menu=="Analizar Hogar":
    kpis(); left,center,right=st.columns([1.15,1.25,1.0])
    with left:
        st.markdown('<div class="form-card">',unsafe_allow_html=True); st.markdown('<div class="step-title"><span class="step-dot">1</span>Información del hogar</div>',unsafe_allow_html=True)
        distrito=st.selectbox("Distrito",districts,index=districts.index("San Juan de Lurigancho") if "San Juan de Lurigancho" in districts else 0)
        ingreso=st.number_input("Ingreso mensual (S/.)",value=1800.0,step=50.0,min_value=1.0)
        a,b=st.columns(2); integrantes=a.number_input("Integrantes",value=5,min_value=1,max_value=12); personas=b.number_input("Personas con ingreso",value=1,min_value=1,max_value=6)
        gasto=st.number_input("Gasto alimentos (S/.)",value=950.0,step=25.0,min_value=0.0)
        c,d=st.columns(2); comidas=c.selectbox("Comidas al día",[1,2,3,4],index=1); diversidad=d.selectbox("Diversidad",["Alta","Media","Baja"],index=2)
        e,f=st.columns(2); laboral=e.selectbox("Situación laboral",["Formal","Informal","Desempleado"],index=1); educacion=f.selectbox("Educación",["Superior","Secundaria","Primaria"],index=1)
        programa=st.radio("Programa social",["Sí","No"],horizontal=True,index=1); inflacion=st.number_input("Inflación alimentos (%)",value=5.1,step=.1)
        add=st.checkbox("Agregar hogar al dataset temporal"); analyze=st.button("🔎 Analizar hogar",use_container_width=True)
        st.markdown('</div>',unsafe_allow_html=True)
    if analyze:
        X,calc=compute(distrito,ingreso,integrantes,personas,gasto,comidas,diversidad,laboral,educacion,programa,inflacion); risk,probs=predict(X); rr={"X":X,"calc":calc,"risk":risk,"probs":probs,"recs":household_recs(risk,calc),"facs":facs(calc,X),"distrito":distrito}; st.session_state.last_result=rr
        if add:
            st.session_state.extra.append({"ID_Hogar":f"N{len(st.session_state.extra)+1:03d}","Distrito":distrito,"Ingreso_Mensual":ingreso,"Integrantes_Hogar":integrantes,"Personas_Con_Ingreso":personas,"Gasto_Alimentos":gasto,"Porcentaje_Ingreso_Alimentos":calc["pct"],"Frecuencia_Comidas":comidas,"Diversidad_Alimentaria":diversidad,"Reduccion_Consumo":"Sí" if calc["pct"]>55 else "No","Situacion_Laboral":laboral,"Nivel_Educativo":educacion,"Programa_Social":programa,"Inflacion_Alimentaria":inflacion,"Indice_Dependencia":calc["dependencia"],"IRAS":risk,"Ingreso_Per_Capita":calc["ingreso_pc"],"Gasto_Alimentos_Per_Capita":calc["gasto_pc"],"Ingreso_Disponible":calc["disponible"],"Indice_Vulnerabilidad_Alimentaria":calc["icva"],"Nivel_Vulnerabilidad":level(calc["icva"]),"Codigo_Distrito":X["Codigo_Distrito"].iloc[0],"Codigo_Diversidad":X["Codigo_Diversidad"].iloc[0],"Codigo_Laboral":X["Codigo_Laboral"].iloc[0],"Codigo_Educacion":X["Codigo_Educacion"].iloc[0],"Codigo_IRAS":IRAS_NUM[risk],"PC1":np.nan,"PC2":np.nan,"Cluster_Hogar":np.nan,"Perfil_Cluster":"Nuevo análisis","Prioridad_Intervencion":priority(risk)})
            st.success("Hogar agregado. Dashboard, mapa y reportes se actualizarán.")
    if "last_result" not in st.session_state:
        X,calc=compute(districts[0],1800,5,1,950,2,"Baja","Informal","Secundaria","No",5.1); risk,probs=predict(X); st.session_state.last_result={"X":X,"calc":calc,"risk":risk,"probs":probs,"recs":household_recs(risk,calc),"facs":facs(calc,X),"distrito":districts[0]}
    r=st.session_state.last_result
    with center:
        st.markdown('<div class="card">',unsafe_allow_html=True); st.markdown('<div class="step-title"><span class="step-dot">2</span>Resultado del análisis</div>',unsafe_allow_html=True)
        st.markdown(f'<div style="text-align:center"><div style="font-weight:900;color:#CBD5E1">NIVEL DE RIESGO IRAS</div><div class="risk-pill">{r["risk"].upper()}</div><div style="font-size:2.4rem;font-weight:900;color:#FFB703">{max(r["probs"].values()):.0f}%</div><div>Probabilidad máxima</div></div>',unsafe_allow_html=True)
        m1,m2,m3,m4=st.columns(4); m1.metric("ICVA",f'{r["calc"]["icva"]:.2f}'); m2.metric("Ingreso pc",f'S/ {r["calc"]["ingreso_pc"]:.0f}'); m3.metric("% alimentos",f'{r["calc"]["pct"]:.1f}%'); m4.metric("Dependencia",f'{r["calc"]["dependencia"]:.1f}')
        st.write("**Interpretación:**",f"El hogar presenta riesgo **{r['risk']}** asociado a inseguridad alimentaria."); st.warning(health(r["risk"])); st.markdown('<span class="inline-badge">Malnutrición</span><span class="inline-badge">Micronutrientes</span><span class="inline-badge">Anemia</span><span class="inline-badge">Bienestar nutricional</span>',unsafe_allow_html=True); st.markdown('</div>',unsafe_allow_html=True)
    with right:
        st.markdown('<div class="chat-panel">',unsafe_allow_html=True); st.markdown("### 🤖 IA Especialista IRAS"); st.markdown('<div class="ai-bubble">Respondo solo preguntas vinculadas al proyecto.</div>',unsafe_allow_html=True)
        q=st.text_area("Pregunta","¿Qué debería priorizar la municipalidad con este hogar?")
        if st.button("Preguntar a la IA",use_container_width=True):
            ctx=json.dumps({"riesgo":r["risk"],"probabilidades":r["probs"],"calculos":r["calc"],"distrito":r["distrito"]},ensure_ascii=False); ans=gemini(q,ctx); st.session_state.chat.append(("user",q)); st.session_state.chat.append(("ai",ans))
        for role,msg in st.session_state.chat[-4:]: st.markdown(f'<div class="{"user-bubble" if role=="user" else "ai-bubble"}">{msg}</div>',unsafe_allow_html=True)
        st.markdown('</div>',unsafe_allow_html=True)
    b1,b2,b3,b4=st.columns([1.1,1.1,1.25,.9])
    with b1:
        st.markdown('<div class="card">',unsafe_allow_html=True); p=pd.DataFrame({"Nivel":list(r["probs"].keys()),"Probabilidad":list(r["probs"].values())}); fig=px.bar(p,x="Nivel",y="Probabilidad",color="Nivel",color_discrete_map=IRAS_COLORS,text="Probabilidad",template="plotly_dark"); fig.update_layout(height=275,showlegend=False,margin=dict(l=10,r=10,t=30,b=10)); st.plotly_chart(fig,use_container_width=True); st.markdown('</div>',unsafe_allow_html=True)
    with b2:
        st.markdown('<div class="card">',unsafe_allow_html=True); ff=pd.DataFrame(r["facs"],columns=["Factor","Importancia"]); fig=px.bar(ff.sort_values("Importancia"),x="Importancia",y="Factor",orientation="h",template="plotly_dark",color="Importancia",color_continuous_scale="Turbo"); fig.update_layout(height=275,margin=dict(l=10,r=10,t=30,b=10)); st.plotly_chart(fig,use_container_width=True); st.markdown('</div>',unsafe_allow_html=True)
    with b3:
        st.markdown('<div class="card">',unsafe_allow_html=True); st.markdown("### Recomendaciones"); [st.markdown(f'<div class="solution"><b>{i}.</b> {rec}</div>',unsafe_allow_html=True) for i,rec in enumerate(r["recs"],1)]; st.markdown('</div>',unsafe_allow_html=True)
    with b4:
        st.markdown('<div class="card">',unsafe_allow_html=True); top,muni,_=muni_recs(df); summary=[f"Hogares: {len(df)}",f"Alto/Crítico: {int(df['IRAS'].isin(['Alto','Crítico']).sum())}",f"ICVA promedio: {df['Indice_Vulnerabilidad_Alimentaria'].mean():.2f}",f"Distrito prioritario: {top}"]; pdf=pdf_report("Reporte IRAS del hogar",r["risk"],r["probs"],r["calc"],r["recs"],summary); st.download_button("📄 Descargar PDF",pdf,"Reporte_IRAS_Hogar.pdf","application/pdf",use_container_width=True); st.download_button("📊 Exportar CSV",df.to_csv(index=False).encode("utf-8-sig"),"IRAS_actualizado.csv","text/csv",use_container_width=True); st.markdown('</div>',unsafe_allow_html=True)

elif menu=="Gestión de Hogares":
    st.markdown("## 👨‍👩‍👧 Gestión de Hogares"); st.dataframe(df,use_container_width=True); st.download_button("Descargar dataset actualizado CSV",df.to_csv(index=False).encode("utf-8-sig"),"IRAS_dataset_actualizado.csv","text/csv")

elif menu=="Analítica Avanzada":
    st.markdown("## 📈 Analítica Avanzada")
    cols=["Ingreso_Per_Capita","Porcentaje_Ingreso_Alimentos","Indice_Dependencia","Indice_Vulnerabilidad_Alimentaria","Codigo_IRAS"]
    st.plotly_chart(px.imshow(df[cols].corr(method="spearman"), text_auto=True, template="plotly_dark", color_continuous_scale="RdBu_r"), use_container_width=True)
    fig=px.scatter(df,x="Ingreso_Per_Capita",y="Indice_Vulnerabilidad_Alimentaria",color="IRAS",color_discrete_map=IRAS_COLORS,template="plotly_dark",hover_data=["Distrito","Porcentaje_Ingreso_Alimentos"]); st.plotly_chart(fig,use_container_width=True)

elif menu=="Machine Learning":
    st.markdown("## 🧠 Machine Learning")
    models={"Random Forest":RandomForestClassifier(n_estimators=220,random_state=42,class_weight="balanced"),"Decision Tree":DecisionTreeClassifier(random_state=42,class_weight="balanced"),"Gradient Boosting":GradientBoostingClassifier(random_state=42),"Logistic Regression":Pipeline([("scaler",StandardScaler()),("model",LogisticRegression(max_iter=1500,class_weight="balanced"))])}
    rows=[]; cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
    for name,m in models.items():
        sc=cross_validate(m,df_base[FEATURES],df_base["Codigo_IRAS"],cv=cv,scoring=["accuracy","precision_macro","recall_macro","f1_macro"])
        rows.append([name,sc["test_accuracy"].mean(),sc["test_precision_macro"].mean(),sc["test_recall_macro"].mean(),sc["test_f1_macro"].mean()])
    met=pd.DataFrame(rows,columns=["Modelo","Accuracy","Precision","Recall","F1"])
    st.dataframe(met.round(3),use_container_width=True)
    st.plotly_chart(px.bar(met,x="Modelo",y="F1",template="plotly_dark",color="F1",color_continuous_scale="Turbo"),use_container_width=True)
    st.markdown("### Importancia de variables")
    st.plotly_chart(px.bar(importance.head(12),y="Variable",x="Importancia",orientation="h",template="plotly_dark",color="Importancia",color_continuous_scale="Turbo"),use_container_width=True)

elif menu=="IA Especialista IRAS":
    st.markdown("## 🤖 IA Especialista IRAS"); st.info("Responde únicamente preguntas relacionadas con IRAS, inseguridad alimentaria, salud pública, hogares, distritos, municipalidades, programas sociales y recomendaciones. Para Gemini real configura GEMINI_API_KEY.")
    q=st.text_area("Pregunta","¿Qué políticas públicas ayudarían a reducir el riesgo alimentario en el distrito con mayor ICVA?")
    if st.button("Responder con IA"):
        top,muni,_=muni_recs(df); ctx=f"{len(df)} hogares. Distrito prioritario: {top}. ICVA promedio {df['Indice_Vulnerabilidad_Alimentaria'].mean():.2f}. Recomendaciones: {muni}"; st.markdown(gemini(q,ctx))

elif menu=="Panel Municipal":
    st.markdown("## 🏛️ Panel Municipal"); top,rec,res=muni_recs(df); st.success(f"Distrito prioritario actual: {top}")
    for i,r in enumerate(rec,1): st.markdown(f'<div class="solution"><b>{i}.</b> {r}</div>',unsafe_allow_html=True)
    st.dataframe(res.round(3),use_container_width=True)

elif menu=="Reportes Ejecutivos":
    st.markdown("## 📑 Reportes Ejecutivos"); top,rec,_=muni_recs(df); fake={"ingreso_pc":round(df["Ingreso_Per_Capita"].mean(),2),"gasto_pc":round(df["Gasto_Alimentos_Per_Capita"].mean(),2),"pct":round(df["Porcentaje_Ingreso_Alimentos"].mean(),2),"dependencia":round(df["Indice_Dependencia"].mean(),2),"disponible":round(df["Ingreso_Disponible"].mean(),2),"icva":round(df["Indice_Vulnerabilidad_Alimentaria"].mean(),4)}; probs=(df["IRAS"].value_counts(normalize=True)*100).round(2).to_dict(); summary=[f"Hogares: {len(df)}",f"Alto/Crítico: {int(df['IRAS'].isin(['Alto','Crítico']).sum())}",f"ICVA promedio: {df['Indice_Vulnerabilidad_Alimentaria'].mean():.2f}",f"Distrito prioritario: {top}"]; pdf=pdf_report("Reporte general IRAS","Resumen general",probs,fake,rec,summary); st.download_button("Descargar Reporte General PDF",pdf,"Reporte_General_IRAS.pdf","application/pdf"); st.download_button("Descargar Dataset CSV",df.to_csv(index=False).encode("utf-8-sig"),"Dataset_IRAS_actualizado.csv","text/csv")

elif menu=="Base de Conocimiento":
    st.markdown("## 📚 Base de Conocimiento")
    st.markdown("""
    <div class="card">
    <b>IRAS:</b> Índice de Riesgo de Afectación a la Salud por Inseguridad Alimentaria.<br><br>
    <b>ICVA:</b> Índice Compuesto de Vulnerabilidad Alimentaria.<br><br>
    <b>Fuentes sugeridas:</b> OMS, FAO, WFP, MIDIS e INEI.<br><br>
    <b>Nota:</b> El sistema estima riesgo indirecto, no diagnóstico médico.
    </div>
    """, unsafe_allow_html=True)

elif menu=="Perfil y Configuración":
    st.markdown("## 👤 Perfil y Configuración")
    st.write("Usuario:", st.session_state.auth.get("name"))
    st.write("Correo:", st.session_state.auth.get("email"))
    st.write("Rol:", st.session_state.auth.get("role","Investigador"))
    st.info("Para producción real: Firebase/Auth0/Supabase Auth + Supabase/PostgreSQL para guardar usuarios, hogares, reportes e historial.")

st.markdown('<div class="footer">© 2024 IRAS Intelligence Platform · Ciencia de Datos · Machine Learning · Gemini</div>',unsafe_allow_html=True)
