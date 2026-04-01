# 🏦 Análisis de Riesgo de Crédito PyME — Cordada

App interactiva construida con Streamlit para analizar el riesgo de default en una cartera de créditos PyME. 4 puntos: exploración de datos, modelo predictivo, selección de variables e impacto de una campaña comercial.

---

## Estructura del proyecto

```
├── app.py                          # App Streamlit
├── analisis_riesgo_pyme_final.ipynb  # Notebook de análisis
├── dataset_riesgo_pymes.csv        # Dataset
├── requirements.txt
└── README.md
```

---

## Instalación

### 1. Clonar el repositorio

```bash
git clone https://github.com/crcordova/cordadav1
cd cordadav1
```

### 2. Crear entorno virtual

```bash
python -m venv venv
```

Activar:

- **Windows:** `venv\Scripts\activate`
- **Mac/Linux:** `source venv/bin/activate`

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

---

## Ejecutar la app

```bash
streamlit run app.py
```

La app abre automáticamente en `http://localhost:8501`

---

## Contenido de la app

| Pestaña | Contenido |
|---------|-----------|
| **P1 — Exploración & Anomalías** | Resumen del dataset, desbalance de clases, detección de data leakage, valores imposibles y distribuciones |
| **P2 & P3 — Modelo & Variables** | Justificación de variables excluidas, modelo LightGBM, curvas ROC y PR, matriz de confusión, análisis de umbral interactivo |
| **P4 — Campaña Comercio** | Simulación del impacto de "1 mes de gracia", concentración sectorial, señal de quiebras por rubro |
| **Resumen Ejecutivo** | Dashboard compacto con métricas clave y recomendaciones para Gerencia |

---

