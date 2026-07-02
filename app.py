
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score


# =====================================================
# CONFIGURACIÓN
# =====================================================

st.set_page_config(
    page_title="Inseguridad Alimentaria - Lima Metropolitana",
    layout="wide"
)

st.title(" Predicción y Clasificación de Riesgo Alimentario")
st.markdown("Aplicativo de ciencia de datos para Lima Metropolitana.")


# =====================================================
# CARGA DE DATOS
# =====================================================

@st.cache_data
def cargar_datos():
    return pd.read_excel("Entregable_3_Dataset_Transformado_Inseguridad_Alimentaria.xlsx")

df = cargar_datos()


# =====================================================
# ENTRENAMIENTO SIMPLE DEL MODELO
# =====================================================

@st.cache_resource
def entrenar_modelos(df):
    df_model = df.copy()

    encoders = {}
    columnas_cat = [
        "Distrito", "Zona", "Tipo_Empleo", "Nivel_Educativo",
        "Programa_Social", "Acceso_Agua", "Acceso_Desague",
        "Estado_Nutricional", "Nivel_Riesgo"
    ]

    for col in columnas_cat:
        le = LabelEncoder()
        df_model[col] = le.fit_transform(df_model[col].astype(str))
        encoders[col] = le

    variables = [
        "Año",
        "Distrito",
        "Ingreso_Laboral",
        "Gasto_Alimentos",
        "Inflacion_Alimentaria",
        "Integrantes_Hogar",
        "Porcentaje_Gasto_Alimentos",
        "Indice_Vulnerabilidad"
    ]

    X = df_model[variables]
    y_pred = df_model["Probabilidad_Enfermedad_Alimentaria"]
    y_class = df_model["Nivel_Riesgo"]

    X_train, X_test, y_train_pred, y_test_pred = train_test_split(
        X, y_pred, test_size=0.2, random_state=42
    )

    _, _, y_train_class, y_test_class = train_test_split(
        X, y_class, test_size=0.2, random_state=42
    )

    modelo_pred = RandomForestRegressor(n_estimators=200, random_state=42)
    modelo_class = RandomForestClassifier(n_estimators=200, random_state=42)

    modelo_pred.fit(X_train, y_train_pred)
    modelo_class.fit(X_train, y_train_class)

    pred_test = modelo_pred.predict(X_test)
    class_test = modelo_class.predict(X_test)

    mae = mean_absolute_error(y_test_pred, pred_test)
    r2 = r2_score(y_test_pred, pred_test)
    acc = accuracy_score(y_test_class, class_test)

    return modelo_pred, modelo_class, encoders, variables, mae, r2, acc


modelo_pred, modelo_class, encoders, variables, mae, r2, acc = entrenar_modelos(df)


# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.header(" Parámetros")

anio = st.sidebar.selectbox(
    "Selecciona el año a predecir",
    list(range(2026, 2036))
)

st.sidebar.markdown("---")
st.sidebar.write("El modelo estima qué distrito tendría mayor probabilidad de padecer enfermedades alimentarias asociadas a inseguridad alimentaria.")


# =====================================================
# MÉTRICAS DEL ENTRENAMIENTO
# =====================================================

st.subheader(" Estado del modelo entrenado")

c1, c2, c3 = st.columns(3)

c1.metric("Error MAE", f"{mae:.2f}")
c2.metric("R²", f"{r2:.2f}")
c3.metric("Accuracy clasificación", f"{acc*100:.1f}%")


# =====================================================
# PREDICCIÓN FUTURA POR DISTRITO
# =====================================================

st.subheader(" Predicción para año futuro")

base = (
    df.groupby("Distrito")
    .agg({
        "Ingreso_Laboral": "mean",
        "Gasto_Alimentos": "mean",
        "Inflacion_Alimentaria": "mean",
        "Integrantes_Hogar": "mean",
        "Porcentaje_Gasto_Alimentos": "mean",
        "Indice_Vulnerabilidad": "mean"
    })
    .reset_index()
)

base["Año"] = anio

# Codificar distrito
base_model = base.copy()
base_model["Distrito_Texto"] = base_model["Distrito"]
base_model["Distrito"] = encoders["Distrito"].transform(base_model["Distrito"].astype(str))

X_futuro = base_model[variables]

base["Probabilidad_Predicha"] = modelo_pred.predict(X_futuro)
base["Nivel_Riesgo_Codificado"] = modelo_class.predict(X_futuro)
base["Nivel_Riesgo"] = encoders["Nivel_Riesgo"].inverse_transform(base["Nivel_Riesgo_Codificado"].astype(int))

base = base.sort_values("Probabilidad_Predicha", ascending=False)

top = base.iloc[0]


col1, col2, col3 = st.columns(3)

col1.metric("Distrito con mayor probabilidad", top["Distrito"])
col2.metric("Probabilidad estimada", f"{top['Probabilidad_Predicha']:.2f}%")
col3.metric("Nivel de riesgo", top["Nivel_Riesgo"])


st.success(
    f"Para el año {anio}, el distrito con mayor probabilidad estimada es "
    f"{top['Distrito']}, con una probabilidad de {top['Probabilidad_Predicha']:.2f}%."
)


# =====================================================
# TABLA DE CLASIFICACIÓN
# =====================================================

st.subheader(" Clasificación de distritos según riesgo")

tabla = base[[
    "Distrito",
    "Año",
    "Ingreso_Laboral",
    "Gasto_Alimentos",
    "Probabilidad_Predicha",
    "Nivel_Riesgo"
]].copy()

tabla["Ingreso_Laboral"] = tabla["Ingreso_Laboral"].round(2)
tabla["Gasto_Alimentos"] = tabla["Gasto_Alimentos"].round(2)
tabla["Probabilidad_Predicha"] = tabla["Probabilidad_Predicha"].round(2)

st.dataframe(tabla, use_container_width=True)


# =====================================================
# GRÁFICO SIMPLE
# =====================================================

st.subheader(" Ranking de probabilidad por distrito")

top10 = tabla.head(10)

fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(top10["Distrito"], top10["Probabilidad_Predicha"])
ax.set_title(f"Top 10 distritos con mayor probabilidad - {anio}")
ax.set_xlabel("Distrito")
ax.set_ylabel("Probabilidad (%)")
ax.tick_params(axis="x", rotation=75)
ax.grid(axis="y")

st.pyplot(fig)


# =====================================================
# BUSCADOR DE DISTRITO
# =====================================================

st.subheader(" Consulta individual por distrito")

distrito = st.selectbox(
    "Selecciona un distrito",
    sorted(tabla["Distrito"].unique())
)

fila = tabla[tabla["Distrito"] == distrito].iloc[0]

c1, c2, c3 = st.columns(3)

c1.metric("Distrito", distrito)
c2.metric("Probabilidad", f"{fila['Probabilidad_Predicha']:.2f}%")
c3.metric("Clasificación", fila["Nivel_Riesgo"])


# =====================================================
# DESCARGA
# =====================================================

st.subheader(" Descargar resultados")

csv = tabla.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Descargar resultados CSV",
    data=csv,
    file_name=f"resultados_prediccion_{anio}.csv",
    mime="text/csv"
)


# =====================================================
# INTERPRETACIÓN
# =====================================================

st.subheader(" Interpretación")

st.write(
    f"""
    El modelo predictivo estima la probabilidad de enfermedad alimentaria por distrito
    para el año seleccionado. El modelo de clasificación asigna un nivel de riesgo
    según las condiciones económicas y alimentarias.

    En este escenario, **{top['Distrito']}** aparece como el distrito con mayor
    probabilidad, por lo que sería considerado prioritario para intervención.
    """
)

