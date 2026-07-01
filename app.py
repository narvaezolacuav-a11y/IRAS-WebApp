
from pathlib import Path
from datetime import datetime
import io, json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import cm
from reportlab.lib.colors import HexColor

st.set_page_config(page_title="IRAS Lima | Sistema Inteligente", page_icon="🍽️", layout="wide", initial_sidebar_state="expanded")
DATA_FILE="02_dataset_analitico_IRAS_Lima_150_V2.xlsx"
GEOJSON_FILE="assets/lima_distritos.geojson"
IRAS_LABELS={0:"Bajo",1:"Moderado",2:"Alto",3:"Crítico"}
IRAS_NUM={v:k for k,v in IRAS_LABELS.items()}
IRAS_COLORS={"Bajo":"#00E5A8","Moderado":"#FFB703","Alto":"#FB8500","Crítico":"#E63946"}
FEATURES=["Ingreso_Mensual","Integrantes_Hogar","Personas_Con_Ingreso","Gasto_Alimentos","Porcentaje_Ingreso_Alimentos","Frecuencia_Comidas","Inflacion_Alimentaria","Indice_Dependencia","Ingreso_Per_Capita","Gasto_Alimentos_Per_Capita","Ingreso_Disponible","Indice_Vulnerabilidad_Alimentaria","Codigo_Distrito","Codigo_Diversidad","Codigo_Laboral","Codigo_Educacion"]
SCOPE=["iras","icva","inseguridad","alimentaria","alimentos","salud","anemia","malnutric","nutric","lima","distrito","municipalidad","programa","social","midis","inei","fao","oms","wfp","hogar","riesgo","pobreza","ingreso","gasto","dependencia","diversidad","comidas","política","publica","pública","recomendación","recomendaciones","intervención","intervencion"]

st.markdown('''
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800;900&display=swap');
html,body,[class*="css"]{font-family:Inter,sans-serif}.stApp{background:radial-gradient(circle at 15% 8%,rgba(0,229,168,.23),transparent 24%),radial-gradient(circle at 88% 8%,rgba(255,183,3,.23),transparent 27%),radial-gradient(circle at 70% 92%,rgba(56,189,248,.14),transparent 31%),linear-gradient(135deg,#061A33 0%,#08223E 48%,#071122 100%)}section[data-testid="stSidebar"]{background:linear-gradient(180deg,#031124,#061A33,#071320);border-right:1px solid rgba(255,255,255,.1)}.block-container{padding-top:1rem;padding-bottom:1.5rem}.hero{border-radius:30px;padding:1.35rem 1.7rem;background:linear-gradient(135deg,rgba(6,26,51,.94),rgba(14,40,68,.86));border:1px solid rgba(255,255,255,.14);box-shadow:0 24px 60px rgba(0,0,0,.32);position:relative;overflow:hidden;margin-bottom:1rem}.hero:before{content:"";position:absolute;width:440px;height:440px;top:-240px;right:-160px;background:radial-gradient(circle,rgba(255,183,3,.42),transparent 60%);animation:pulse 4.5s infinite alternate ease-in-out}.hero:after{content:"";position:absolute;width:340px;height:340px;bottom:-200px;left:18%;background:radial-gradient(circle,rgba(0,229,168,.30),transparent 60%);animation:float 6s infinite alternate ease-in-out}@keyframes pulse{from{transform:scale(1);opacity:.55}to{transform:scale(1.18);opacity:.95}}@keyframes float{from{transform:translateY(0)}to{transform:translateY(-34px)}}@keyframes soft{from{transform:scale(1);filter:brightness(1)}to{transform:scale(1.025);filter:brightness(1.12)}}.hero h1{color:#F8FAFC;margin:0;font-weight:900;font-size:2.1rem;position:relative;z-index:2}.hero p{color:#C9D9EC;margin:.25rem 0 0 0;position:relative;z-index:2}.card,.form-card,.chat-panel{background:linear-gradient(180deg,rgba(14,40,68,.96),rgba(7,24,43,.96));border:1px solid rgba(255,255,255,.12);border-radius:26px;padding:1rem;box-shadow:0 18px 42px rgba(0,0,0,.28);transition:all .22s ease}.card:hover,.form-card:hover,.chat-panel:hover{transform:translateY(-3px);border-color:rgba(255,183,3,.42)}.kpi-title{color:#C9D9EC;font-size:.77rem;text-transform:uppercase;font-weight:900}.kpi-value{color:#F8FAFC;font-size:2rem;font-weight:900}.kpi-caption{color:#9FB2CC;font-size:.85rem}.step-dot{display:inline-flex;justify-content:center;align-items:center;width:34px;height:34px;margin-right:.48rem;border-radius:999px;background:linear-gradient(135deg,#00E5A8,#FFB703);color:#061A33;font-weight:900}.step-title{font-size:1.05rem;font-weight:900;color:#F8FAFC;margin-bottom:.65rem}.risk-pill{display:inline-block;padding:.70rem 2.1rem;border-radius:17px;font-size:2rem;font-weight:900;color:white;background:linear-gradient(135deg,#E63946,#FF6B6B);box-shadow:0 0 42px rgba(230,57,70,.36);animation:soft 1.8s infinite alternate ease-in-out}.ai-bubble{background:rgba(255,255,255,.08);border:1px solid rgba(255,255,255,.10);border-radius:18px 18px 18px 7px;padding:.85rem .95rem;color:#E2E8F0;margin:.55rem 0}.user-bubble{background:linear-gradient(135deg,#2563EB,#A855F7);border-radius:18px 18px 7px 18px;padding:.85rem .95rem;color:white;margin:.55rem 0 .55rem auto;max-width:92%}.solution{display:block;border-radius:16px;margin:.45rem 0;padding:.70rem .85rem;background:linear-gradient(135deg,rgba(0,229,168,.12),rgba(255,183,3,.12));border:1px solid rgba(255,255,255,.10)}.inline-badge{display:inline-block;border-radius:999px;margin:.16rem;padding:.32rem .75rem;background:rgba(255,183,3,.16);border:1px solid rgba(255,183,3,.35);color:#FDE68A;font-weight:800;font-size:.78rem}.stButton>button{background:linear-gradient(135deg,#00E5A8,#38BDF8,#FFB703);color:#061A33;border:none;border-radius:16px;padding:.78rem 1rem;font-weight:900}.footer{color:#9FB2CC;text-align:center;padding:1rem;font-size:.82rem}
</style>
''', unsafe_allow_html=True)

@st.cache_data
def load_data():
    p=Path(DATA_FILE)
    if not p.exists():
        st.error("No se encontró el dataset Excel. Sube 02_dataset_analitico_IRAS_Lima_150_V2.xlsx al repositorio.")
        st.stop()
    return pd.read_excel(p, sheet_name="Dataset_Analitico_V2")

@st.cache_resource
def train_model(data):
    model=RandomForestClassifier(n_estimators=500, random_state=42, class_weight="balanced", min_samples_leaf=2)
    model.fit(data[FEATURES], data["Codigo_IRAS"])
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores=cross_validate(model, data[FEATURES], data["Codigo_IRAS"], cv=cv, scoring=["accuracy","precision_macro","recall_macro","f1_macro"])
    metrics=pd.DataFrame({"Métrica":["Accuracy","Precision macro","Recall macro","F1 macro"],"Valor":[scores["test_accuracy"].mean(),scores["test_precision_macro"].mean(),scores["test_recall_macro"].mean(),scores["test_f1_macro"].mean()]})
    importance=pd.DataFrame({"Variable":FEATURES,"Importancia":model.feature_importances_}).sort_values("Importancia", ascending=False)
    return model, metrics, importance

df_base=load_data(); model, model_metrics, importance=train_model(df_base)
if "extra_hogares" not in st.session_state: st.session_state.extra_hogares=[]
if "chat_history" not in st.session_state: st.session_state.chat_history=[]

def current_df():
    return pd.concat([df_base,pd.DataFrame(st.session_state.extra_hogares)], ignore_index=True) if st.session_state.extra_hogares else df_base.copy()

df=current_df(); districts=sorted(df_base["Distrito"].unique()); district_map={d:int(df_base[df_base["Distrito"]==d]["Codigo_Distrito"].iloc[0]) for d in districts}

def norm(v,col):
    mn,mx=float(df_base[col].min()),float(df_base[col].max())
    return 0 if mx==mn else max(0,min(1,(float(v)-mn)/(mx-mn)))

def compute_household(distrito, ingreso, integrantes, personas, gasto, comidas, diversidad, laboral, educacion, programa, inflacion):
    ingreso=max(float(ingreso),1); integrantes=max(int(integrantes),1); personas=max(int(personas),1); gasto=max(float(gasto),0); comidas=int(comidas)
    pct=round(gasto/ingreso*100,2); dep=round((integrantes-personas)/personas,2); ipc=round(ingreso/integrantes,2); gpc=round(gasto/integrantes,2); disp=round(ingreso-gasto,2)
    cd={"Alta":0,"Media":1,"Baja":2}[diversidad]; cl={"Formal":0,"Informal":1,"Desempleado":2}[laboral]; ce={"Superior":0,"Secundaria":1,"Primaria":2}[educacion]
    icva=round((1-norm(ipc,"Ingreso_Per_Capita"))*.25+norm(pct,"Porcentaje_Ingreso_Alimentos")*.25+norm(dep,"Indice_Dependencia")*.15+((cd/2)*.60+max(0,min(1,(3-comidas)/2))*.40)*.20+(cl/2)*.10+(0 if programa=="Sí" else 1)*.05,4)
    X=pd.DataFrame([{"Ingreso_Mensual":ingreso,"Integrantes_Hogar":integrantes,"Personas_Con_Ingreso":personas,"Gasto_Alimentos":gasto,"Porcentaje_Ingreso_Alimentos":pct,"Frecuencia_Comidas":comidas,"Inflacion_Alimentaria":inflacion,"Indice_Dependencia":dep,"Ingreso_Per_Capita":ipc,"Gasto_Alimentos_Per_Capita":gpc,"Ingreso_Disponible":disp,"Indice_Vulnerabilidad_Alimentaria":icva,"Codigo_Distrito":district_map.get(distrito,0),"Codigo_Diversidad":cd,"Codigo_Laboral":cl,"Codigo_Educacion":ce}])
    return X,{"pct":pct,"dependencia":dep,"ingreso_pc":ipc,"gasto_pc":gpc,"disponible":disp,"icva":icva,"cod_div":cd,"cod_lab":cl}

def predict_household(X):
    pred=int(model.predict(X)[0]); risk=IRAS_LABELS[pred]
    probs={IRAS_LABELS[int(c)]:round(float(p)*100,2) for c,p in zip(model.classes_,model.predict_proba(X)[0])}
    return risk,probs

def level_icva(x): return "Baja" if x<.30 else "Moderada" if x<.50 else "Alta" if x<.70 else "Crítica"
def priority(risk): return {"Bajo":"Seguimiento general","Moderado":"Monitoreo preventivo","Alto":"Prevención focalizada","Crítico":"Atención urgente"}[risk]

def risk_factors(calc,X):
    factors=[]
    if calc["ingreso_pc"]<df_base["Ingreso_Per_Capita"].quantile(.35): factors.append(("Ingreso per cápita bajo",.31))
    if calc["pct"]>50: factors.append(("Alta presión alimentaria",.24))
    if calc["cod_div"]==2: factors.append(("Baja diversidad alimentaria",.17))
    if calc["cod_lab"]>=1: factors.append(("Inestabilidad laboral",.12))
    if calc["dependencia"]>df_base["Indice_Dependencia"].median(): factors.append(("Dependencia económica elevada",.09))
    if X["Frecuencia_Comidas"].iloc[0]<=2: factors.append(("Frecuencia reducida de comidas",.08))
    return factors or [("Condiciones relativamente estables",.20)]

def household_recommendations(risk,calc):
    recs={"Bajo":["Mantener seguimiento general del hogar.","Promover educación alimentaria preventiva.","Monitorear variaciones en precios de alimentos."],"Moderado":["Realizar monitoreo preventivo del hogar.","Orientar en planificación del gasto alimentario.","Evaluar apoyo municipal si aumenta la presión alimentaria."],"Alto":["Priorizar apoyo alimentario y programas sociales.","Recomendar evaluación nutricional preventiva.","Promover educación alimentaria y financiera.","Realizar seguimiento municipal mensual."],"Crítico":["Activar apoyo alimentario inmediato.","Derivar a evaluación nutricional preventiva.","Realizar seguimiento municipal periódico.","Evaluar acceso a programas sociales o comunitarios.","Promover capacitación laboral y educación alimentaria."]}[risk]
    if calc["pct"]>55: recs.append("Reducir presión alimentaria mediante canastas, comedores u ollas comunes focalizadas.")
    return recs

def health_interpretation(risk):
    return {"Bajo":"Riesgo indirecto bajo: menor vulnerabilidad nutricional asociada a inseguridad alimentaria.","Moderado":"Riesgo indirecto moderado: posible exposición a dieta poco variada y deficiencias de micronutrientes.","Alto":"Riesgo indirecto alto: mayor vulnerabilidad a malnutrición, anemia o deterioro de calidad dietaria.","Crítico":"Riesgo indirecto crítico: alta vulnerabilidad a malnutrición, deficiencias de micronutrientes, anemia y bajo bienestar nutricional."}[risk]

def municipal_recommendations(data):
    resumen=data.groupby("Distrito").agg(Hogares=("ID_Hogar","count"),ICVA=("Indice_Vulnerabilidad_Alimentaria","mean"),Riesgo_Promedio=("Codigo_IRAS","mean"),Presion_Alimentaria=("Porcentaje_Ingreso_Alimentos","mean"),Ingreso_PC=("Ingreso_Per_Capita","mean")).sort_values("ICVA", ascending=False)
    top=resumen.index[0]; top_row=resumen.iloc[0]
    recs=[f"Priorizar el distrito {top}, porque presenta el mayor ICVA promedio ({top_row['ICVA']:.2f}).","Identificar hogares con IRAS Alto y Crítico para intervención focalizada.","Articular apoyo alimentario con comedores populares, ollas comunes y programas sociales.","Coordinar campañas preventivas de evaluación nutricional con centros de salud.","Ejecutar talleres de educación alimentaria y planificación del gasto familiar.","Monitorear mensualmente presión alimentaria, diversidad alimentaria e ingreso per cápita."]
    return top,recs,resumen

def is_project_question(q): return any(k in q.lower() for k in SCOPE)
def local_ai(q,context):
    if not is_project_question(q): return "Solo puedo responder preguntas relacionadas con el proyecto IRAS: inseguridad alimentaria, riesgo en salud, hogares, distritos de Lima Metropolitana, programas sociales, municipalidades, indicadores IRAS/ICVA o recomendaciones."
    if "municipal" in q.lower() or "distrito" in q.lower():
        top,recs,_=municipal_recommendations(df); return f"El distrito prioritario actual es {top}. Se recomienda: "+" ".join(recs[:4])
    if "salud" in q.lower() or "anemia" in q.lower() or "malnutric" in q.lower(): return "La inseguridad alimentaria puede asociarse con dietas poco diversas, deficiencias de micronutrientes, anemia y malnutrición. El sistema no diagnostica; estima riesgo indirecto y sugiere seguimiento preventivo."
    return "La decisión debe basarse en IRAS, ICVA, presión alimentaria, ingreso per cápita, diversidad alimentaria, dependencia económica y situación laboral."

def gemini_answer(q,context):
    if not is_project_question(q): return local_ai(q,context)
    try: api_key=st.secrets.get("GEMINI_API_KEY",None)
    except Exception: api_key=None
    if api_key:
        try:
            from google import genai
            client=genai.Client(api_key=api_key)
            prompt=f"Eres el Asistente IRAS. Responde solo sobre inseguridad alimentaria, salud pública, hogares, Lima Metropolitana, municipalidades, programas sociales, IRAS e ICVA. No diagnostiques enfermedades. Sé claro, profesional y útil para un proyecto universitario de Ciencia de Datos. Contexto: {context}. Pregunta: {q}"
            return client.models.generate_content(model="gemini-2.0-flash", contents=prompt).text
        except Exception: return "No pude conectar con Gemini. Respuesta local: "+local_ai(q,context)
    return local_ai(q,context)

def build_pdf(title,risk,probs,calc,factors,recs,municipal_recs,chart_summary):
    buffer=io.BytesIO(); c=canvas.Canvas(buffer,pagesize=A4); w,h=A4
    c.setFillColor(HexColor("#061A33")); c.rect(0,0,w,h,fill=1,stroke=0); c.setFillColor(HexColor("#FFB703")); c.setFont("Helvetica-Bold",22); c.drawString(2*cm,h-3*cm,title); c.setFillColor(HexColor("#F8FAFC")); c.setFont("Helvetica",11); c.drawString(2*cm,h-3.8*cm,"Sistema Inteligente IRAS · Lima Metropolitana"); c.drawString(2*cm,h-4.4*cm,f"Fecha: {datetime.now().strftime('%d/%m/%Y %H:%M')}"); c.setFont("Helvetica-Oblique",9); c.drawString(2*cm,2*cm,"IRAS estima riesgo indirecto; no realiza diagnóstico médico."); c.showPage()
    y=h-2*cm; c.setFillColor(HexColor("#111827")); c.setFont("Helvetica-Bold",16); c.drawString(2*cm,y,"1. Resultado del análisis"); y-=.8*cm; c.setFont("Helvetica-Bold",12); c.drawString(2*cm,y,f"Nivel IRAS estimado: {risk}"); y-=.55*cm; c.setFont("Helvetica",9)
    for k,v in probs.items(): c.drawString(2.3*cm,y,f"{k}: {v}%"); y-=.38*cm
    y-=.35*cm; c.setFont("Helvetica-Bold",12); c.drawString(2*cm,y,"2. Cálculos principales"); y-=.55*cm; c.setFont("Helvetica",9)
    for k,v in [("Ingreso per cápita",f"S/ {calc['ingreso_pc']}"),("Gasto alimentario per cápita",f"S/ {calc['gasto_pc']}"),("Presión alimentaria",f"{calc['pct']}%"),("Dependencia económica",calc["dependencia"]),("Ingreso disponible",f"S/ {calc['disponible']}"),("ICVA",calc["icva"] )]: c.drawString(2.3*cm,y,f"{k}: {v}"); y-=.38*cm
    y-=.35*cm; c.setFont("Helvetica-Bold",12); c.drawString(2*cm,y,"3. Factores que explican el riesgo"); y-=.55*cm; c.setFont("Helvetica",9)
    for f,imp in factors: c.drawString(2.3*cm,y,f"{f}: importancia aproximada {imp}"); y-=.38*cm
    y-=.35*cm; c.setFont("Helvetica-Bold",12); c.drawString(2*cm,y,"4. Interpretación de gráficos"); y-=.55*cm; c.setFont("Helvetica",9)
    for line in chart_summary: c.drawString(2.3*cm,y,line[:100]); y-=.38*cm
    c.showPage(); y=h-2*cm; c.setFont("Helvetica-Bold",16); c.drawString(2*cm,y,"5. Recomendaciones"); y-=.8*cm; c.setFont("Helvetica-Bold",12); c.drawString(2*cm,y,"Para el hogar"); y-=.55*cm; c.setFont("Helvetica",9)
    for i,rec in enumerate(recs,1): c.drawString(2.3*cm,y,f"{i}. {rec[:95]}"); y-=.38*cm
    y-=.35*cm; c.setFont("Helvetica-Bold",12); c.drawString(2*cm,y,"Para la municipalidad"); y-=.55*cm; c.setFont("Helvetica",9)
    for i,rec in enumerate(municipal_recs,1): c.drawString(2.3*cm,y,f"{i}. {rec[:95]}"); y-=.38*cm
    c.setFont("Helvetica-Oblique",8); c.drawString(2*cm,1.5*cm,"Documento académico. El modelo estima riesgo indirecto y requiere validación con datos reales."); c.showPage(); c.save(); buffer.seek(0); return buffer

def show_kpis():
    high=int(df["IRAS"].isin(["Alto","Crítico"]).sum()); top,_,_=municipal_recommendations(df)
    vals=[("Hogares analizados",len(df),"Dataset actualizado"),("Riesgo Alto + Crítico",f"{high} ({high/len(df)*100:.1f}%)","Requieren intervención"),("ICVA promedio",f"{df['Indice_Vulnerabilidad_Alimentaria'].mean():.2f}","Índice 0 a 1"),("Distrito prioritario",top,"Mayor vulnerabilidad")]
    cols=st.columns(4)
    for c,(title,val,cap) in zip(cols,vals): c.markdown(f'<div class="card"><div class="kpi-title">{title}</div><div class="kpi-value">{val}</div><div class="kpi-caption">{cap}</div></div>', unsafe_allow_html=True)

def probability_chart(probs):
    p=pd.DataFrame({"Nivel":list(probs.keys()),"Probabilidad":list(probs.values())}); fig=px.bar(p,x="Nivel",y="Probabilidad",color="Nivel",color_discrete_map=IRAS_COLORS,text="Probabilidad",template="plotly_dark"); fig.update_layout(height=280,showlegend=False,margin=dict(l=10,r=10,t=30,b=10)); return fig

def factors_chart(factors):
    ff=pd.DataFrame(factors,columns=["Factor","Importancia"]); fig=px.bar(ff.sort_values("Importancia"),x="Importancia",y="Factor",orientation="h",template="plotly_dark",color="Importancia",color_continuous_scale="Turbo"); fig.update_layout(height=280,margin=dict(l=10,r=10,t=30,b=10)); return fig

with st.sidebar:
    logo=Path("assets/iras_logo.svg")
    if logo.exists(): st.image(str(logo),width=92)
    st.markdown("## IRAS Lima"); st.caption("Sistema Inteligente")
    menu=st.radio("Menú",["Inicio","Analizar Hogar","Dashboard Distrital","Mapa Inteligente","IA Especialista IRAS","Soluciones Municipales","Reportes"],label_visibility="collapsed")
    st.markdown("---"); st.success("🟢 Sistema activo"); st.caption("Proyecto de Ciencia de Datos")

st.markdown('<div class="hero"><h1>🍽️ Sistema Inteligente IRAS</h1><p>Predicción del riesgo de afectación a la salud por inseguridad alimentaria en hogares de Lima Metropolitana.</p></div>', unsafe_allow_html=True)

if menu=="Inicio":
    show_kpis(); c1,c2=st.columns([1,1])
    with c1:
        d=df["IRAS"].value_counts().reset_index(); d.columns=["IRAS","Hogares"]; fig=px.pie(d,names="IRAS",values="Hogares",hole=.45,color="IRAS",color_discrete_map=IRAS_COLORS,template="plotly_dark"); st.plotly_chart(fig,use_container_width=True)
    with c2:
        resumen=df.groupby("Distrito")["Indice_Vulnerabilidad_Alimentaria"].mean().sort_values(ascending=False).head(10).reset_index(); fig=px.bar(resumen,y="Distrito",x="Indice_Vulnerabilidad_Alimentaria",orientation="h",template="plotly_dark",color="Indice_Vulnerabilidad_Alimentaria",color_continuous_scale="Turbo"); st.plotly_chart(fig,use_container_width=True)

elif menu=="Analizar Hogar":
    show_kpis(); left,center,right=st.columns([1.1,1.25,1])
    with left:
        st.markdown('<div class="form-card">',unsafe_allow_html=True); st.markdown('<div class="step-title"><span class="step-dot">1</span>Información del hogar</div>',unsafe_allow_html=True)
        distrito=st.selectbox("Distrito",districts,index=districts.index("San Juan de Lurigancho") if "San Juan de Lurigancho" in districts else 0)
        ingreso=st.number_input("Ingreso mensual del hogar (S/.)",value=1800.0,step=50.0,min_value=1.0)
        a,b=st.columns(2); integrantes=a.number_input("Integrantes",value=5,min_value=1,max_value=12); personas=b.number_input("Personas con ingreso",value=1,min_value=1,max_value=6)
        gasto=st.number_input("Gasto mensual en alimentos (S/.)",value=950.0,step=25.0,min_value=0.0)
        c,d=st.columns(2); comidas=c.selectbox("Comidas al día",[1,2,3,4],index=1); diversidad=d.selectbox("Diversidad alimentaria",["Alta","Media","Baja"],index=2)
        e,f=st.columns(2); laboral=e.selectbox("Situación laboral",["Formal","Informal","Desempleado"],index=1); educacion=f.selectbox("Nivel educativo",["Superior","Secundaria","Primaria"],index=1)
        programa=st.radio("Recibe programa social",["Sí","No"],horizontal=True,index=1); inflacion=st.number_input("Inflación alimentaria (%)",value=5.1,step=.1)
        add_house=st.checkbox("Agregar hogar al dataset temporal"); analyze=st.button("🔎 Analizar hogar",use_container_width=True); st.markdown('</div>',unsafe_allow_html=True)
    if analyze:
        X,calc=compute_household(distrito,ingreso,integrantes,personas,gasto,comidas,diversidad,laboral,educacion,programa,inflacion); risk,probs=predict_household(X); factors=risk_factors(calc,X); recs=household_recommendations(risk,calc); st.session_state.last_result={"X":X,"calc":calc,"risk":risk,"probs":probs,"factors":factors,"recs":recs,"distrito":distrito}
        if add_house:
            st.session_state.extra_hogares.append({"ID_Hogar":f"N{len(st.session_state.extra_hogares)+1:03d}","Distrito":distrito,"Ingreso_Mensual":ingreso,"Integrantes_Hogar":integrantes,"Personas_Con_Ingreso":personas,"Gasto_Alimentos":gasto,"Porcentaje_Ingreso_Alimentos":calc["pct"],"Frecuencia_Comidas":comidas,"Diversidad_Alimentaria":diversidad,"Reduccion_Consumo":"Sí" if calc["pct"]>55 else "No","Situacion_Laboral":laboral,"Nivel_Educativo":educacion,"Programa_Social":programa,"Inflacion_Alimentaria":inflacion,"Indice_Dependencia":calc["dependencia"],"IRAS":risk,"Ingreso_Per_Capita":calc["ingreso_pc"],"Gasto_Alimentos_Per_Capita":calc["gasto_pc"],"Ingreso_Disponible":calc["disponible"],"Indice_Vulnerabilidad_Alimentaria":calc["icva"],"Nivel_Vulnerabilidad":level_icva(calc["icva"]),"Codigo_Distrito":X["Codigo_Distrito"].iloc[0],"Codigo_Diversidad":X["Codigo_Diversidad"].iloc[0],"Codigo_Laboral":X["Codigo_Laboral"].iloc[0],"Codigo_Educacion":X["Codigo_Educacion"].iloc[0],"Codigo_IRAS":IRAS_NUM[risk],"PC1":np.nan,"PC2":np.nan,"Cluster_Hogar":np.nan,"Perfil_Cluster":"Nuevo análisis","Prioridad_Intervencion":priority(risk)})
            st.success("Hogar agregado temporalmente. El dashboard, mapa y reportes se actualizarán durante la sesión.")
    if "last_result" not in st.session_state:
        X,calc=compute_household(districts[0],1800,5,1,950,2,"Baja","Informal","Secundaria","No",5.1); risk,probs=predict_household(X); st.session_state.last_result={"X":X,"calc":calc,"risk":risk,"probs":probs,"factors":risk_factors(calc,X),"recs":household_recommendations(risk,calc),"distrito":districts[0]}
    r=st.session_state.last_result; top_dist,muni_recs,_=municipal_recommendations(df)
    with center:
        st.markdown('<div class="card">',unsafe_allow_html=True); st.markdown('<div class="step-title"><span class="step-dot">2</span>Resultado del análisis</div>',unsafe_allow_html=True); st.markdown(f'<div style="text-align:center"><div style="font-weight:900;color:#CBD5E1">NIVEL DE RIESGO IRAS</div><div class="risk-pill">{r["risk"].upper()}</div><div style="font-size:2.4rem;font-weight:900;color:#FFB703">{max(r["probs"].values()):.0f}%</div><div>Probabilidad máxima</div></div>',unsafe_allow_html=True)
        m1,m2,m3,m4=st.columns(4); m1.metric("ICVA",f'{r["calc"]["icva"]:.2f}'); m2.metric("Ingreso pc",f'S/ {r["calc"]["ingreso_pc"]:.0f}'); m3.metric("% alimentos",f'{r["calc"]["pct"]:.1f}%'); m4.metric("Dependencia",f'{r["calc"]["dependencia"]:.1f}')
        st.write("**Interpretación:**",f"El hogar presenta riesgo **{r['risk']}** asociado a inseguridad alimentaria."); st.warning(health_interpretation(r["risk"])); st.markdown('<span class="inline-badge">Malnutrición</span><span class="inline-badge">Micronutrientes</span><span class="inline-badge">Anemia</span><span class="inline-badge">Bienestar nutricional</span>',unsafe_allow_html=True); st.markdown('</div>',unsafe_allow_html=True)
    with right:
        st.markdown('<div class="chat-panel">',unsafe_allow_html=True); st.markdown("### 🤖 IA Especialista IRAS"); st.markdown('<div class="ai-bubble">Respondo solo preguntas del proyecto: inseguridad alimentaria, salud, hogares, distritos, municipalidades e indicadores.</div>',unsafe_allow_html=True)
        q=st.text_area("Pregunta","¿Qué debería priorizar la municipalidad con este hogar?")
        if st.button("Preguntar a la IA",use_container_width=True):
            ctx=json.dumps({"riesgo":r["risk"],"probabilidades":r["probs"],"calculos":r["calc"],"distrito":r["distrito"],"distrito_prioritario":top_dist},ensure_ascii=False); ans=gemini_answer(q,ctx); st.session_state.chat_history.append(("user",q)); st.session_state.chat_history.append(("ai",ans))
        for role,msg in st.session_state.chat_history[-4:]: st.markdown(f'<div class="{"user-bubble" if role=="user" else "ai-bubble"}">{msg}</div>',unsafe_allow_html=True)
        st.markdown('</div>',unsafe_allow_html=True)
    b1,b2,b3,b4=st.columns([1.1,1.1,1.25,.9])
    with b1: st.markdown('<div class="card">',unsafe_allow_html=True); st.plotly_chart(probability_chart(r["probs"]),use_container_width=True); st.caption("Interpretación: la barra más alta representa el nivel de riesgo estimado por el modelo."); st.markdown('</div>',unsafe_allow_html=True)
    with b2: st.markdown('<div class="card">',unsafe_allow_html=True); st.plotly_chart(factors_chart(r["factors"]),use_container_width=True); st.caption("Interpretación: muestra los factores que más influyen en el riesgo del hogar."); st.markdown('</div>',unsafe_allow_html=True)
    with b3:
        st.markdown('<div class="card">',unsafe_allow_html=True); st.markdown("### Recomendaciones para el hogar")
        for i,rec in enumerate(r["recs"],1): st.markdown(f'<div class="solution"><b>{i}.</b> {rec}</div>',unsafe_allow_html=True)
        st.markdown('</div>',unsafe_allow_html=True)
    with b4:
        st.markdown('<div class="card">',unsafe_allow_html=True); chart_summary=[f"El modelo asigna mayor probabilidad al nivel {r['risk']}.",f"La presión alimentaria del hogar es {r['calc']['pct']}%.",f"El ICVA calculado es {r['calc']['icva']}.","El gráfico de factores permite identificar las causas principales del riesgo.",f"El distrito prioritario del dataset actual es {top_dist}."]; pdf=build_pdf("Reporte IRAS del hogar",r["risk"],r["probs"],r["calc"],r["factors"],r["recs"],muni_recs,chart_summary); st.download_button("📄 Descargar PDF",pdf,"Reporte_IRAS_Hogar.pdf","application/pdf",use_container_width=True); st.download_button("📊 Exportar CSV",df.to_csv(index=False).encode("utf-8-sig"),"IRAS_actualizado.csv","text/csv",use_container_width=True); st.markdown('</div>',unsafe_allow_html=True)

elif menu=="Dashboard Distrital":
    show_kpis(); col1,col2=st.columns(2)
    with col1:
        d=df["IRAS"].value_counts().reset_index(); d.columns=["IRAS","Hogares"]; fig=px.pie(d,names="IRAS",values="Hogares",hole=.45,color="IRAS",color_discrete_map=IRAS_COLORS,template="plotly_dark"); st.plotly_chart(fig,use_container_width=True); st.caption("Interpretación: proporción de hogares por nivel IRAS.")
    with col2:
        resumen=df.groupby("Distrito")["Indice_Vulnerabilidad_Alimentaria"].mean().sort_values(ascending=False).head(12).reset_index(); fig=px.bar(resumen,y="Distrito",x="Indice_Vulnerabilidad_Alimentaria",orientation="h",template="plotly_dark",color="Indice_Vulnerabilidad_Alimentaria",color_continuous_scale="Turbo"); st.plotly_chart(fig,use_container_width=True); st.caption("Interpretación: distritos con mayor vulnerabilidad promedio.")
    st.dataframe(df,use_container_width=True)

elif menu=="Mapa Inteligente":
    st.markdown("## 🗺️ Mapa Inteligente de Lima Metropolitana"); st.info("La app incluye integración para GeoJSON municipal. Si subes assets/lima_distritos.geojson, mostrará un mapa geográfico exacto; si no, mostrará ranking territorial dinámico.")
    resumen=df.groupby("Distrito").agg(Vulnerabilidad=("Indice_Vulnerabilidad_Alimentaria","mean"),Hogares=("ID_Hogar","count"),Riesgo=("Codigo_IRAS","mean")).reset_index(); geo_path=Path(GEOJSON_FILE)
    if geo_path.exists():
        try:
            with open(geo_path,"r",encoding="utf-8") as f: geojson=json.load(f)
            props=geojson["features"][0]["properties"]; possible=["Distrito","distrito","DISTRITO","NOMB_DIST","NOMBDIST","name","NAME"]; geo_col=next((c for c in possible if c in props),None)
            if geo_col:
                fig=px.choropleth_mapbox(resumen,geojson=geojson,locations="Distrito",featureidkey=f"properties.{geo_col}",color="Vulnerabilidad",color_continuous_scale=["#00E5A8","#FFB703","#FB8500","#E63946"],mapbox_style="carto-darkmatter",zoom=9,center={"lat":-12.0464,"lon":-77.0428},opacity=.75,hover_data=["Distrito","Hogares","Riesgo"]); fig.update_layout(height=700,margin=dict(l=0,r=0,t=0,b=0)); st.plotly_chart(fig,use_container_width=True)
            else: st.warning("No pude detectar la propiedad del distrito en el GeoJSON. Usa Distrito, distrito, NOMB_DIST o name.")
        except Exception as e: st.error(f"No se pudo cargar el GeoJSON: {e}")
    else:
        resumen2=resumen.sort_values("Vulnerabilidad"); fig=px.bar(resumen2,y="Distrito",x="Vulnerabilidad",color="Riesgo",orientation="h",template="plotly_dark",color_continuous_scale=["#00E5A8","#FFB703","#FB8500","#E63946"]); fig.update_layout(height=700); st.plotly_chart(fig,use_container_width=True); st.caption("Interpretación: ranking territorial actualizado según ICVA promedio.")
    st.dataframe(resumen.sort_values("Vulnerabilidad",ascending=False).round(3),use_container_width=True)

elif menu=="IA Especialista IRAS":
    st.markdown("## 🤖 IA Especialista IRAS"); st.info("El asistente responde únicamente preguntas relacionadas con IRAS, inseguridad alimentaria, salud pública, hogares, distritos, municipalidades, programas sociales e indicadores. Para usar Gemini real, configura GEMINI_API_KEY en Streamlit Secrets.")
    q=st.text_area("Escribe tu pregunta","¿Qué políticas públicas ayudarían a reducir el riesgo alimentario en el distrito con mayor ICVA?")
    if st.button("Responder con IA"):
        top,recs,_=municipal_recommendations(df); ctx=f"Dataset actual: {len(df)} hogares. Distrito prioritario: {top}. ICVA promedio: {df['Indice_Vulnerabilidad_Alimentaria'].mean():.2f}. Recomendaciones municipales: {recs}"; st.markdown(gemini_answer(q,ctx))

elif menu=="Soluciones Municipales":
    st.markdown("## Soluciones Municipales"); top,recs,resumen=municipal_recommendations(df); st.success(f"Distrito prioritario actual: {top}")
    for i,rec in enumerate(recs,1): st.markdown(f'<div class="solution"><b>{i}.</b> {rec}</div>',unsafe_allow_html=True)
    st.markdown("### Plan de acción sugerido"); plan=pd.DataFrame({"Horizonte":["Corto plazo","Corto plazo","Mediano plazo","Mediano plazo","Largo plazo"],"Acción":["Identificar hogares Alto y Crítico","Activar apoyo alimentario focalizado","Monitorear diversidad alimentaria y presión alimentaria","Articular salud, municipalidad y programas sociales","Implementar sistema permanente de vigilancia alimentaria"],"Responsable":["Municipalidad","Municipalidad + comunidad","Área social municipal","Municipalidad + centro de salud","Municipalidad + academia"],"Indicador":["N° hogares priorizados","N° apoyos entregados","ICVA promedio","N° derivaciones preventivas","Dashboard actualizado"]}); st.dataframe(plan,use_container_width=True); st.markdown("### Ranking distrital"); st.dataframe(resumen.round(3),use_container_width=True)

elif menu=="Reportes":
    st.markdown("## Reportes Ejecutivos"); top,recs,resumen=municipal_recommendations(df); fake_calc={"ingreso_pc":round(df["Ingreso_Per_Capita"].mean(),2),"gasto_pc":round(df["Gasto_Alimentos_Per_Capita"].mean(),2),"pct":round(df["Porcentaje_Ingreso_Alimentos"].mean(),2),"dependencia":round(df["Indice_Dependencia"].mean(),2),"disponible":round(df["Ingreso_Disponible"].mean(),2),"icva":round(df["Indice_Vulnerabilidad_Alimentaria"].mean(),4)}; probs=(df["IRAS"].value_counts(normalize=True)*100).round(2).to_dict(); chart_summary=[f"El dataset contiene {len(df)} hogares analizados.",f"El distrito prioritario actual es {top}.",f"El ICVA promedio del dataset es {df['Indice_Vulnerabilidad_Alimentaria'].mean():.2f}.",f"Los hogares Alto y Crítico son {int(df['IRAS'].isin(['Alto','Crítico']).sum())}.","El ranking distrital permite orientar intervenciones municipales focalizadas."]; pdf=build_pdf("Reporte General IRAS","Resumen general",probs,fake_calc,[],recs,recs,chart_summary); st.download_button("Descargar Reporte General PDF",pdf,"Reporte_General_IRAS.pdf","application/pdf"); st.download_button("Descargar Dataset CSV",df.to_csv(index=False).encode("utf-8-sig"),"Dataset_IRAS_actualizado.csv","text/csv")

st.markdown('<div class="footer">© 2024 Sistema Inteligente IRAS · Ciencia de Datos · Machine Learning · Salud Pública</div>', unsafe_allow_html=True)
