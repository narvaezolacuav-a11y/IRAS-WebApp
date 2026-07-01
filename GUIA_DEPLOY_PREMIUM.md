# Guía de despliegue — IRAS WebApp Premium

## 1. Crear repositorio
Crea un repositorio público en GitHub llamado:

`iras-webapp-premium`

## 2. Subir archivos
Sube al repositorio:

- `app.py`
- `requirements.txt`
- `README.md`
- `02_dataset_analitico_IRAS_Lima_150_V2.xlsx`
- carpeta `.streamlit`
- carpeta `assets`

## 3. Publicar en Streamlit Cloud
Entra a:

`https://share.streamlit.io`

Configura:

- Repository: `iras-webapp-premium`
- Branch: `main`
- Main file path: `app.py`

Luego presiona **Deploy**.

## 4. Link esperado
Obtendrás un enlace parecido a:

`https://iras-webapp-premium.streamlit.app`

## Búsqueda web real en IA Asistente
La app funciona sin API usando una base de conocimiento interna.  
Para activar búsqueda web real, crea una API key de Tavily y agrega en Streamlit Secrets:

```toml
TAVILY_API_KEY = "tu_api_key"
```

## Nota
El dataset es sintético académico. IRAS estima riesgo indirecto y no diagnostica enfermedades.