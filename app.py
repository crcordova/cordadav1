import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score,
    confusion_matrix, classification_report
)
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

# ── Configuración ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Riesgo PyME — Cordada",
    page_icon="🏦",
    layout="wide"
)

TEMPLATE   = "plotly_white"
COLOR_OK   = "#4CAF8A"
COLOR_BAD  = "#E05C5C"
COLOR_BLUE = "#5B8FF9"
SEED       = 42

# ── Carga y pipeline (cacheado) ────────────────────────────────────────────
@st.cache_data
def load_and_prepare():
    df_raw = pd.read_csv("dataset_riesgo_pymes.csv")

    df = df_raw.copy()
    df = df[df["estado_empresa"] == "Activa"].copy()
    col_c = "consultas_equifax_ultimos_6m"
    df[col_c] = df[col_c].fillna(0).clip(lower=0)
    mediana_ant = df["antiguedad_empresa_meses"].median()
    df["antiguedad_empresa_meses"] = df["antiguedad_empresa_meses"].fillna(mediana_ant)

    return df_raw, df

@st.cache_resource
def train_model(df):
    TARGET   = "target_default"
    FEATURES = ["monto_solicitado", "antiguedad_empresa_meses", "score_equifax",
                "tasa_interes_asignada", "rubro", ]

    df_model = df[FEATURES + [TARGET]].copy()
    le = LabelEncoder()
    df_model["rubro_enc"] = le.fit_transform(df_model["rubro"])
    df_model = df_model.drop(columns=["rubro"])

    X = df_model.drop(columns=[TARGET])
    y = df_model[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y)

    model = lgb.LGBMClassifier(
        n_estimators=300, learning_rate=0.05, num_leaves=31,
        class_weight="balanced", random_state=SEED, verbose=-1)
    model.fit(X_train, y_train)

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")

    return model, le, X, y, X_train, X_test, y_train, y_test, y_prob, y_pred, cv_scores

df_raw, df = load_and_prepare()
model, le, X, y, X_train, X_test, y_train, y_test, y_prob, y_pred, cv_scores = train_model(df)

auc = roc_auc_score(y_test, y_prob)
ap  = average_precision_score(y_test, y_prob)

# ── Header ─────────────────────────────────────────────────────────────────
st.title("🏦 Desafío Data Scientist — Cartera de Créditos PyME")
st.caption("Cordada | Análisis de Riesgo de Default")

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 P1 — Exploración & Anomalías",
    "🤖 P2 & P3 — Modelo & Variables",
    "🎯 P4 — Campaña Comercio",
    "📋 Resumen Ejecutivo"
])

# ══════════════════════════════════════════════════════════════════════════
# TAB 1 — EXPLORACIÓN
# ══════════════════════════════════════════════════════════════════════════
with tab1:
    st.header("Pregunta 1 — Exploración del Dataset & Detección de Anomalías")
    st.markdown("> Identificar problemas que harían **peligroso entrenar un modelo a ciegas**.")

    # 1.1 Resumen
    st.subheader("1.1 Resumen de variables")
    resumen = pd.DataFrame({
        "dtype":   df_raw.dtypes,
        "nulos":   df_raw.isnull().sum(),
        "nulos_%": (df_raw.isnull().mean() * 100).round(2),
        "únicos":  df_raw.nunique(),
        "min":     df_raw.min(),
        "max":     df_raw.max(),
    })
    st.dataframe(resumen, use_container_width=True)

    # 1.2 Desbalance
    st.subheader("1.2 Variable objetivo — Desbalance de clases")
    counts = df_raw["target_default"].value_counts()
    pcts   = df_raw["target_default"].value_counts(normalize=True) * 100

    fig = make_subplots(rows=1, cols=2,
        specs=[[{"type":"xy"}, {"type":"domain"}]],
        subplot_titles=["Cantidad por clase", "Proporción"])
    fig.add_trace(go.Bar(
        x=["No mora (0)", "Mora >90d (1)"], y=counts.values,
        marker_color=[COLOR_OK, COLOR_BAD],
        text=[f"{v:,}<br>({p:.1f}%)" for v, p in zip(counts.values, pcts.values)],
        textposition="outside", showlegend=False), row=1, col=1)
    fig.add_trace(go.Pie(
        labels=["No mora (0)", "Mora >90d (1)"], values=counts.values,
        marker_colors=[COLOR_OK, COLOR_BAD], hole=0.35, textinfo="label+percent"),
        row=1, col=2)
    fig.update_layout(height=380, template=TEMPLATE,
        title_text="⚠️ Desbalance: 88.2% vs 11.8%")
    st.plotly_chart(fig, use_container_width=True)
    st.info(f"Ratio desbalance: {counts[0]/counts[1]:.1f}:1 — usar `class_weight='balanced'` o ajustar umbral.")

    # 1.3 Estado empresa
    st.subheader("1.3 Anomalía: `estado_empresa = En_Quiebra`")
    quiebra = df_raw[df_raw["estado_empresa"] == "En_Quiebra"]
    col1, col2, col3 = st.columns(3)
    col1.metric("Registros En_Quiebra", len(quiebra))
    col2.metric("Default rate en quiebra", f"{quiebra['target_default'].mean()*100:.1f}%")
    col3.metric("Default rate general", f"{df_raw['target_default'].mean()*100:.1f}%")
    st.warning("Ninguna institución debería prestar a empresa en quiebra. Estado probablemente actualizado **post-desembolso** (data leakage temporal). Solución: eliminar estos registros.")

    # 1.4 Data leakage unidad_gestion
    st.subheader("1.4 Anomalía: `unidad_gestion_asignada` — Data Leakage severo")
    default_unidad = (df_raw.groupby("unidad_gestion_asignada")["target_default"]
                      .agg(["mean","count"]).rename(columns={"mean":"tasa","count":"n"})
                      .sort_values("tasa", ascending=False).reset_index())
    default_unidad["tasa_pct"] = (default_unidad["tasa"] * 100).round(1)

    fig = px.bar(default_unidad, x="unidad_gestion_asignada", y="tasa_pct",
                 color="tasa_pct", text="tasa_pct",
                 color_continuous_scale=[COLOR_OK, COLOR_BAD],
                 labels={"tasa_pct":"Tasa Default (%)","unidad_gestion_asignada":""},
                 template=TEMPLATE)
    fig.update_traces(texttemplate="%{text}%", textposition="outside")
    fig.update_coloraxes(showscale=False)
    fig.add_hline(y=df_raw["target_default"].mean()*100, line_dash="dash",
                  annotation_text=f"Media: {df_raw['target_default'].mean()*100:.1f}%")
    fig.update_layout(height=380, title="🚨 Unidad_Activos_Especiales = 100% default")
    st.plotly_chart(fig, use_container_width=True)
    st.error("Esta unidad gestiona créditos **ya en mora** — se asigna DESPUÉS del evento. Incluirla es data leakage directo.")

    # 1.5 Consultas negativas
    st.subheader("1.5 Anomalía: `consultas_equifax_ultimos_6m` — Valores negativos y nulos")
    col_c = "consultas_equifax_ultimos_6m"
    c1, c2, c3 = st.columns(3)
    c1.metric("Nulos", f"{df_raw[col_c].isnull().sum()} ({df_raw[col_c].isnull().mean()*100:.1f}%)")
    c2.metric("Valores negativos", int((df_raw[col_c] < 0).sum()))
    c3.metric("Correlación con target", f"{df_raw[col_c].corr(df_raw['target_default']):.4f}")
    st.warning("Una cantidad de consultas no puede ser negativa. Correlación con target ≈ 0.016 → señal muy débil. Solución: clip a 0.")

    # 1.6 Distribuciones
    st.subheader("1.6 Distribuciones de variables numéricas por clase")
    num_cols = ["monto_solicitado","antiguedad_empresa_meses","score_equifax","tasa_interes_asignada"]
    fig = make_subplots(rows=2, cols=2,
        subplot_titles=[c.replace("_"," ").title() for c in num_cols])
    positions = [(1,1),(1,2),(2,1),(2,2)]
    for col, (r, c) in zip(num_cols, positions):
        for val, color, name in [(0, COLOR_OK, "No mora"), (1, COLOR_BAD, "Mora")]:
            fig.add_trace(go.Histogram(
                x=df_raw[df_raw["target_default"]==val][col],
                name=name, marker_color=color, opacity=0.6,
                showlegend=(r==1 and c==1), nbinsx=30), row=r, col=c)
    fig.update_layout(barmode="overlay", height=500, template=TEMPLATE)
    st.plotly_chart(fig, use_container_width=True)

    # 1.7 Tabla resumen anomalías
    st.subheader("1.7 Resumen de anomalías detectadas")
    st.markdown("""
| # | Problema | Variable | Riesgo si se ignora | Solución |
|---|----------|----------|---------------------|----------|
| 1 | **Desbalance de clases** (88/12) | `target_default` | Accuracy engañosa, modelo predice siempre 0 | `class_weight`, umbral ajustado |
| 2 | **Data leakage** post-evento | `unidad_gestion_asignada` | AUC inflado artificialmente | Excluir del modelo |
| 3 | **Estado empresa post-mora** | `estado_empresa = En_Quiebra` | Contaminación del entrenamiento | Eliminar 16 registros |
| 4 | **Valores imposibles** | `consultas_equifax_ultimos_6m` | Ruido en features | Clip a 0 |
| 5 | **Nulos** (9.5%) | `consultas_equifax_ultimos_6m` | Sesgo si MNAR | Imputar con 0 |
| 6 | **Alta cardinalidad** | `id_ejecutivo_venta` (899 IDs) | Overfitting | Excluir |
| 7 | **Identificador** | `id_cliente` | Overfitting | Excluir |
""")

# ══════════════════════════════════════════════════════════════════════════
# TAB 2 — MODELO & VARIABLES
# ══════════════════════════════════════════════════════════════════════════
with tab2:
    st.header("Pregunta 3 — Selección de Variables")
    st.markdown("""
| Variable | Decisión | Categoría | Justificación |
|---|---|---|---|
| `unidad_gestion_asignada` | ❌ Excluir | **Data Leakage** | `Unidad_Activos_Especiales` = 100% default. Se asigna *después* de la mora. |
| `id_ejecutivo_venta` | ❌ Excluir | **Alta cardinalidad** | 899 ejecutivos únicos, no generalizable. |
| `consultas_equifax_ultimos_6m` | ❌ Excluir | **Sin poder discriminante** | Correlación con target = 0.016. |
| `estado_empresa` | ❌ Excluir | **Variable constante** | Tras limpieza queda con un solo valor. |
| `id_cliente` | ❌ Excluir | **Identificador** | Sin poder predictivo. |
""")

    # Score Equifax vs Default
    st.subheader("Poder discriminante: Score Equifax")
    fig = px.box(df_raw, x="target_default", y="score_equifax",
                 color="target_default",
                 color_discrete_map={0: COLOR_OK, 1: COLOR_BAD},
                 labels={"target_default":"Default","score_equifax":"Score Equifax"},
                 template=TEMPLATE)
    fig.update_layout(showlegend=False, height=380,
        title="Score Equifax: clientes en mora vs sin mora")
    st.plotly_chart(fig, use_container_width=True)

    # Default por rubro
    st.subheader("Default rate por rubro")
    default_rubro = (df_raw.groupby("rubro")["target_default"]
                     .agg(["mean","count"]).rename(columns={"mean":"tasa","count":"n"})
                     .sort_values("tasa", ascending=False).reset_index())
    default_rubro["tasa_pct"] = (default_rubro["tasa"]*100).round(1)
    fig = px.bar(default_rubro, x="rubro", y="tasa_pct",
                 color="tasa_pct", text="tasa_pct",
                 color_continuous_scale=[COLOR_OK, COLOR_BAD],
                 template=TEMPLATE)
    fig.update_traces(texttemplate="%{text}%", textposition="outside")
    fig.update_coloraxes(showscale=False)
    fig.add_hline(y=df_raw["target_default"].mean()*100, line_dash="dash",
                  annotation_text=f"Media: {df_raw['target_default'].mean()*100:.1f}%")
    fig.update_layout(height=380)
    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.header("Pregunta 2 — Modelo Base y Métrica para Gerencia de Riesgo")

    # Métricas principales
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("ROC-AUC (test)", f"{auc:.4f}")
    c2.metric("PR-AUC (test)", f"{ap:.4f}")
    c3.metric("ROC-AUC CV media", f"{cv_scores.mean():.4f}")
    c4.metric("CV std", f"±{cv_scores.std():.4f}")

    # Curvas ROC y PR
    st.subheader("Curvas ROC y Precision-Recall")
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    prec, rec, _ = precision_recall_curve(y_test, y_prob)

    fig = make_subplots(rows=1, cols=2,
        subplot_titles=[f"Curva ROC (AUC={auc:.3f})",
                        f"Curva Precision-Recall (AP={ap:.3f})"])
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines",
        line=dict(color=COLOR_BLUE, width=2), name="Modelo"), row=1, col=1)
    fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines",
        line=dict(color="gray", dash="dash"), name="Azar"), row=1, col=1)
    fig.add_trace(go.Scatter(x=rec, y=prec, mode="lines",
        line=dict(color=COLOR_BAD, width=2), name="Modelo PR"), row=1, col=2)
    fig.add_hline(y=y_test.mean(), line_dash="dash", line_color="gray",
                  annotation_text=f"Baseline: {y_test.mean():.2f}", row=1, col=2)
    fig.update_layout(height=420, template=TEMPLATE, showlegend=False)
    fig.update_xaxes(title_text="FPR", row=1, col=1)
    fig.update_yaxes(title_text="TPR", row=1, col=1)
    fig.update_xaxes(title_text="Recall", row=1, col=2)
    fig.update_yaxes(title_text="Precision", row=1, col=2)
    st.plotly_chart(fig, use_container_width=True)

    # Matriz de confusión
    st.subheader("Matriz de confusión (umbral=0.5)")
    cm = confusion_matrix(y_test, y_pred)
    labels = ["No mora (0)", "Mora (1)"]
    fig = px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
                    x=labels, y=labels,
                    labels=dict(x="Predicho", y="Real"), template=TEMPLATE)
    fig.update_coloraxes(showscale=False)
    fig.update_layout(height=350)
    st.plotly_chart(fig, use_container_width=True)

    tn, fp, fn, tp = cm.ravel()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("✅ Verdaderos Negativos", tn, help="Buenos correctamente aprobados")
    c2.metric("⚠️ Falsos Positivos", fp, help="Buenos rechazados — costo: perder negocio")
    c3.metric("🚨 Falsos Negativos", fn, help="Morosos aprobados — costo: pérdida crediticia")
    c4.metric("✅ Verdaderos Positivos", tp, help="Morosos correctamente detectados")

    # Análisis de umbral
    st.subheader("Trade-off de negocio: análisis de umbral")
    thresholds = np.arange(0.1, 0.9, 0.05)
    rows = []
    for t in thresholds:
        y_pred_t = (y_prob >= t).astype(int)
        tn_t, fp_t, fn_t, tp_t = confusion_matrix(y_test, y_pred_t).ravel()
        recall_t    = tp_t / (tp_t + fn_t) if (tp_t + fn_t) > 0 else 0
        precision_t = tp_t / (tp_t + fp_t) if (tp_t + fp_t) > 0 else 0
        rows.append({"umbral": round(t,2), "FP": fp_t, "FN": fn_t,
                     "recall": recall_t, "precision": precision_t})
    df_thresh = pd.DataFrame(rows)

    fig = make_subplots(rows=1, cols=2,
        subplot_titles=["FP vs FN según umbral", "Recall vs Precision según umbral"])
    fig.add_trace(go.Scatter(x=df_thresh["umbral"], y=df_thresh["FP"],
        mode="lines+markers", name="FP (buenos rechazados)",
        line=dict(color=COLOR_BLUE, width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_thresh["umbral"], y=df_thresh["FN"],
        mode="lines+markers", name="FN (morosos aprobados)",
        line=dict(color=COLOR_BAD, width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_thresh["umbral"], y=df_thresh["recall"],
        mode="lines+markers", name="Recall",
        line=dict(color=COLOR_BAD, width=2)), row=1, col=2)
    fig.add_trace(go.Scatter(x=df_thresh["umbral"], y=df_thresh["precision"],
        mode="lines+markers", name="Precision",
        line=dict(color=COLOR_OK, width=2)), row=1, col=2)
    for col_idx in [1, 2]:
        fig.add_vline(x=0.5, line_dash="dash", line_color="gray", row=1, col=col_idx)
    fig.update_layout(height=420, template=TEMPLATE,
        title_text="¿Qué umbral elegir? — Decisión de negocio")
    fig.update_xaxes(title_text="Umbral de decisión")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
| Error | Qué pasa | Costo |
|-------|----------|-------|
| **Falso Positivo (FP)** | Cliente bueno rechazado | Perdemos negocio, el cliente va a la competencia |
| **Falso Negativo (FN)** | Moroso aprobado | Pérdida financiera directa por el crédito impago |

> Bajar el umbral captura más morosos (↑ Recall) pero rechaza más buenos clientes (↑ FP). La decisión del umbral es de **negocio**, no técnica.
""")

    # Importancia de variables
    st.subheader("Importancia de variables")
    importances = pd.DataFrame({
        "feature": X.columns,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=True)
    fig = px.bar(importances, x="importance", y="feature", orientation="h",
                 color="importance", color_continuous_scale=["#C8D4E3", COLOR_BLUE],
                 template=TEMPLATE)
    fig.update_coloraxes(showscale=False)
    fig.update_layout(height=380, title="Importancia de variables (LightGBM)")
    st.plotly_chart(fig, use_container_width=True)

    # CV scores
    st.subheader("Validación cruzada (5 folds)")
    fig = go.Figure(go.Bar(
        x=[f"Fold {i+1}" for i in range(5)], y=cv_scores,
        marker_color=COLOR_BLUE,
        text=[f"{s:.3f}" for s in cv_scores], textposition="outside"))
    fig.add_hline(y=cv_scores.mean(), line_dash="dash",
                  annotation_text=f"Media: {cv_scores.mean():.3f}")
    fig.update_layout(title="ROC-AUC por fold", template=TEMPLATE,
                      yaxis=dict(range=[0.5, 1.0]), height=350)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("¿Qué métrica presentar a Gerencia de Riesgo?")
    st.markdown(f"""
| Métrica | Tu modelo | Azar | Perfecto | Qué significa para el negocio |
|---------|-----------|------|----------|-------------------------------|
| **ROC-AUC** | {cv_scores.mean():.3f} | 0.500 | 1.0 | Ordena bien el riesgo {cv_scores.mean()*100:.0f} de cada 100 veces |
| **PR-AUC** | {ap:.3f} | {y_test.mean():.3f} | 1.0 | Concentra morosos reales {ap/y_test.mean():.1f}x mejor que al azar |

**¿Por qué NO usar Accuracy?** Un modelo que predice siempre "no mora" tiene **88.2% de accuracy** pero detecta **0 morosos**.

**PR-AUC es la métrica honesta** para datasets desbalanceados. El baseline no es 0.5 sino la tasa de default real ({y_test.mean()*100:.1f}%).
""")

# ══════════════════════════════════════════════════════════════════════════
# TAB 3 — CAMPAÑA COMERCIO
# ══════════════════════════════════════════════════════════════════════════
with tab3:
    st.header("Pregunta 4 — Impacto de la Campaña '1 Mes de Gracia' en Rubro Comercio")
    st.markdown("> ¿Aumentará o disminuirá el riesgo de mora en el segmento Comercio?")

    df_comercio = df[df["rubro"] == "Comercio"].copy()
    df_comercio["rubro_enc"] = le.transform(df_comercio["rubro"])

    # 4.1 Perfil actual
    st.subheader("4.1 Perfil de riesgo actual del rubro Comercio")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("N clientes", f"{len(df_comercio):,}")
    c2.metric("Tasa default", f"{df_comercio['target_default'].mean()*100:.2f}%")
    c3.metric("Score Equifax (media)", f"{df_comercio['score_equifax'].mean():.0f}")
    c4.metric("Antigüedad media", f"{df_comercio['antiguedad_empresa_meses'].mean():.0f} meses")

    rubros_stats = (df.groupby("rubro")["target_default"]
                    .agg(["mean","count"]).rename(columns={"mean":"tasa","count":"n"})
                    .reset_index())
    rubros_stats["tasa_pct"] = (rubros_stats["tasa"]*100).round(2)
    fig = px.bar(rubros_stats.sort_values("tasa_pct", ascending=False),
                 x="rubro", y="tasa_pct", text="tasa_pct",
                 color="rubro",
                 color_discrete_map={"Comercio": COLOR_BAD,
                                     "Agricultura": COLOR_BLUE, "Servicios": COLOR_BLUE,
                                     "Manufactura": COLOR_BLUE, "Construcción": COLOR_BLUE},
                 template=TEMPLATE)
    fig.update_traces(texttemplate="%{text}%", textposition="outside")
    fig.add_hline(y=df["target_default"].mean()*100, line_dash="dash",
                  annotation_text=f"Media global: {df['target_default'].mean()*100:.1f}%")
    fig.update_layout(height=380, showlegend=False,
        title="Tasa de Default por Rubro (Comercio destacado)")
    st.plotly_chart(fig, use_container_width=True)

    # 4.2 Perfil beneficiario
    st.subheader("4.2 ¿Qué tipo de cliente atraería la campaña?")
    st.markdown("""
El "1 mes de gracia" es atractivo principalmente para empresas con **flujo de caja ajustado al inicio**:
- Empresas nuevas (poca antigüedad)
- Montos altos relativos a su capacidad
- Score Equifax más bajo (mayor necesidad de financiamiento)
""")
    q75_monto = df_comercio["monto_solicitado"].quantile(0.75)
    mediana_score = df_comercio["score_equifax"].median()
    df_comercio["perfil_campana"] = (
        (df_comercio["antiguedad_empresa_meses"] < 24) |
        (df_comercio["monto_solicitado"] > q75_monto) |
        (df_comercio["score_equifax"] < mediana_score)
    ).astype(int)

    perfil_stats = []
    for p, label in [(1,"Con perfil campaña"), (0,"Sin perfil campaña")]:
        sub = df_comercio[df_comercio["perfil_campana"]==p]
        perfil_stats.append({"Perfil": label, "N": len(sub),
                              "Tasa default": f"{sub['target_default'].mean()*100:.2f}%"})
    st.dataframe(pd.DataFrame(perfil_stats), use_container_width=True, hide_index=True)

    # 4.3 Simulación
    st.subheader("4.3 Simulación de escenarios")
    st.markdown("Simulamos que la campaña atrae clientes con **menor antigüedad, mayor monto y menor score**.")

    col_sim1, col_sim2, col_sim3 = st.columns(3)
    delta_ant   = col_sim1.slider("Reducción antigüedad (%)", 0, 40, 20)
    delta_monto = col_sim2.slider("Aumento monto (%)", 0, 30, 10)
    delta_score = col_sim3.slider("Reducción score Equifax (pts)", 0, 50, 15)

    X_comercio = df_comercio[X.columns].copy()
    prob_base = model.predict_proba(X_comercio)[:, 1]
    default_rate_base = prob_base.mean()

    X_campana = X_comercio.copy()
    X_campana["antiguedad_empresa_meses"] = (
        X_campana["antiguedad_empresa_meses"] * (1 - delta_ant/100)).clip(lower=1)
    X_campana["monto_solicitado"] = X_campana["monto_solicitado"] * (1 + delta_monto/100)
    X_campana["score_equifax"] = (X_campana["score_equifax"] - delta_score).clip(lower=250)

    prob_campana = model.predict_proba(X_campana)[:, 1]
    default_rate_campana = prob_campana.mean()
    delta_abs = default_rate_campana - default_rate_base
    delta_rel = delta_abs / default_rate_base * 100

    c1, c2, c3 = st.columns(3)
    c1.metric("Tasa default BASE", f"{default_rate_base*100:.2f}%")
    c2.metric("Tasa default CAMPAÑA", f"{default_rate_campana*100:.2f}%",
              delta=f"+{delta_abs*100:.2f} pp", delta_color="inverse")
    c3.metric("Incremento relativo", f"+{delta_rel:.1f}%", delta_color="inverse")

    fig = go.Figure(go.Bar(
        x=["Base (actual)", "Campaña (simulado)"],
        y=[default_rate_base*100, default_rate_campana*100],
        marker_color=[COLOR_OK, COLOR_BAD],
        text=[f"{t:.2f}%" for t in [default_rate_base*100, default_rate_campana*100]],
        textposition="outside", width=0.4))
    fig.add_hline(y=df["target_default"].mean()*100, line_dash="dot", line_color="gray",
                  annotation_text=f"Media global: {df['target_default'].mean()*100:.1f}%")
    fig.update_layout(height=380, template=TEMPLATE,
        title="Impacto estimado de la campaña en tasa de default — Rubro Comercio",
        yaxis=dict(range=[0, max(default_rate_campana*100, default_rate_base*100)*1.4]))
    st.plotly_chart(fig, use_container_width=True)

    # Distribución de probabilidades
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=prob_base, name="Base", marker_color=COLOR_OK,
        opacity=0.6, nbinsx=30, histnorm="probability density"))
    fig.add_trace(go.Histogram(x=prob_campana, name="Campaña", marker_color=COLOR_BAD,
        opacity=0.6, nbinsx=30, histnorm="probability density"))
    fig.update_layout(barmode="overlay", height=380, template=TEMPLATE,
        title="Distribución de P(default): Base vs Campaña",
        xaxis_title="P(default)", yaxis_title="Densidad")
    st.plotly_chart(fig, use_container_width=True)

    # 4.5 Concentración sectorial
    st.subheader("4.5 Riesgo de concentración sectorial")
    st.markdown("""
Una campaña dirigida exclusivamente a Comercio no solo aumenta el riesgo individual —
también **concentra la cartera en un solo sector**, introduciendo riesgo sistémico.
Si el sector enfrenta una crisis, todos los créditos se ven afectados simultáneamente.
""")
    cartera_actual = (df.groupby("rubro")
                      .agg(n_clientes=("target_default","count"))
                      .reset_index())
    cartera_actual["pct"] = cartera_actual["n_clientes"] / cartera_actual["n_clientes"].sum() * 100

    cartera_campana = cartera_actual.copy()
    n_nuevos = int(cartera_actual.loc[cartera_actual["rubro"]=="Comercio","n_clientes"].values[0] * 0.30)
    cartera_campana.loc[cartera_campana["rubro"]=="Comercio","n_clientes"] += n_nuevos
    cartera_campana["pct"] = cartera_campana["n_clientes"] / cartera_campana["n_clientes"].sum() * 100

    fig = make_subplots(rows=1, cols=2, specs=[[{"type":"domain"},{"type":"domain"}]],
        subplot_titles=["Cartera actual", "Cartera post-campaña (+30% Comercio)"])
    colors_r = [COLOR_BAD, COLOR_BLUE, COLOR_OK, "#F6C94E", "#A78BFA"]
    fig.add_trace(go.Pie(labels=cartera_actual["rubro"], values=cartera_actual["n_clientes"],
        marker_colors=colors_r, textinfo="label+percent", hole=0.35), row=1, col=1)
    fig.add_trace(go.Pie(labels=cartera_campana["rubro"], values=cartera_campana["n_clientes"],
        marker_colors=colors_r, textinfo="label+percent", hole=0.35), row=1, col=2)
    fig.update_layout(height=400, template=TEMPLATE,
        title="Concentración sectorial: antes y después de la campaña")
    st.plotly_chart(fig, use_container_width=True)

    pct_antes   = cartera_actual.loc[cartera_actual["rubro"]=="Comercio","pct"].values[0]
    pct_despues = cartera_campana.loc[cartera_campana["rubro"]=="Comercio","pct"].values[0]
    st.warning(f"Participación de Comercio: **{pct_antes:.1f}%** → **{pct_despues:.1f}%** (+{pct_despues-pct_antes:.1f} pp). Mayor concentración = mayor riesgo sistémico.")

    # 4.6 Señal de quiebras
    st.subheader("4.6 Señal adicional: empresas En_Quiebra por rubro")
    st.markdown("""
Los 16 registros `En_Quiebra` fueron excluidos del modelo por data leakage,
pero son **evidencia descriptiva válida** sobre la fragilidad estructural de cada sector.
Un índice > 1.0 significa que ese rubro tiene **más quiebras de las esperadas** por su tamaño.
""")
    quiebras = df_raw[df_raw["estado_empresa"] == "En_Quiebra"].copy()
    quiebras_rubro = quiebras["rubro"].value_counts().reset_index()
    quiebras_rubro.columns = ["rubro", "n_quiebras"]
    dist_rubro = df_raw["rubro"].value_counts(normalize=True).reset_index()
    dist_rubro.columns = ["rubro", "pct_cartera"]
    quiebras_rubro = quiebras_rubro.merge(dist_rubro, on="rubro")
    quiebras_rubro["n_esperado"] = (quiebras_rubro["pct_cartera"] * len(quiebras)).round(1)
    quiebras_rubro["indice"] = (quiebras_rubro["n_quiebras"] / quiebras_rubro["n_esperado"]).round(2)
    quiebras_rubro = quiebras_rubro.sort_values("n_quiebras", ascending=False)

    fig = make_subplots(rows=1, cols=2,
        subplot_titles=["Quiebras observadas vs esperadas", "Índice de sobrerrepresentación"])
    fig.add_trace(go.Bar(x=quiebras_rubro["rubro"], y=quiebras_rubro["n_quiebras"],
        name="Observado", marker_color=COLOR_BAD), row=1, col=1)
    fig.add_trace(go.Bar(x=quiebras_rubro["rubro"], y=quiebras_rubro["n_esperado"],
        name="Esperado", marker_color=COLOR_BLUE, opacity=0.6), row=1, col=1)
    colors_idx = [COLOR_BAD if v > 1 else COLOR_OK for v in quiebras_rubro["indice"]]
    fig.add_trace(go.Bar(x=quiebras_rubro["rubro"], y=quiebras_rubro["indice"],
        marker_color=colors_idx,
        text=quiebras_rubro["indice"], texttemplate="%{text:.2f}x",
        textposition="outside", showlegend=False), row=1, col=2)
    fig.add_hline(y=1.0, line_dash="dash", line_color="gray",
                  annotation_text="proporcional", row=1, col=2)
    fig.update_layout(height=400, template=TEMPLATE, barmode="group",
        title="¿Qué rubros concentran más quiebras de lo esperado?")
    st.plotly_chart(fig, use_container_width=True)

    # 4.7 Conclusiones
    st.subheader("4.7 Conclusiones y recomendación para Gerencia")
    st.error("""
**La campaña aumentaría el riesgo de mora por cuatro vías:**

1. **Selección adversa** — el beneficio atrae empresas con mayor necesidad de liquidez inicial.
2. **Empresas más nuevas y montos mayores** — los predictores más importantes del modelo.
3. **Concentración sectorial** — si Comercio crece en la cartera, una crisis del sector afecta toda la cohorte simultáneamente (riesgo sistémico).
4. **Señal estructural de fragilidad** — los datos de quiebras muestran si Comercio está sobrerrepresentado en insolvencias severas.
""")
    st.markdown("""
**Recomendación 1:** Antes de lanzar la campaña masiva, realizar un **piloto controlado** (A/B test) con ~500 clientes de Comercio, asignando aleatoriamente el mes de gracia. Esto permitirá medir el efecto causal real con un modelo de diferencias en diferencias o regresión con variable de tratamiento.  

**Recomendación 2**: se recomendienda una estrategia de pricing inteligente, lograr segmentar a los clientes y en base a cada segmento asignar diferentes tasas y diferentes periodos de gracia.  

**Recomendación 3**: hacer campañar para diversificar cartera no para concentrar  
Si la campaña se lanza de todas formas, mitigar con:
- Score Equifax mínimo más alto para clientes de Comercio
- Límite de monto para empresas con antigüedad < 12 meses
- Cupo máximo de Comercio en la cartera total (ej. no superar 35%)
- Monitoreo mensual de la cohorte de campaña vs histórica

| Modelo para medir efecto causal | Cuándo usarlo |
|---|---|
| **Simulación con modelo predictivo** (este análisis) | Sin datos históricos de la campaña |
| **Regresión logística con interacción** `rubro × mes_gracia` | Con piloto y grupo control |
| **Propensity Score Matching** | Sin grupo control, comparando clientes similares |
""")

# ══════════════════════════════════════════════════════════════════════════
# TAB 4 — RESUMEN EJECUTIVO
# ══════════════════════════════════════════════════════════════════════════
with tab4:
    st.header("📋 Resumen Ejecutivo para Gerencia de Riesgo")

    st.subheader("Dataset")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total registros", f"{len(df_raw):,}")
    c2.metric("Tasa de default", f"{df_raw['target_default'].mean()*100:.1f}%")
    c3.metric("Registros usados (post-limpieza)", f"{len(df):,}")
    c4.metric("Variables en el modelo", len(X.columns))

    st.divider()
    st.subheader("Anomalías detectadas (P1)")
    st.markdown("""
| Problema | Impacto | Solución aplicada |
|----------|---------|-------------------|
| Desbalance 88/12 | Accuracy engañosa | `class_weight='balanced'` |
| Data leakage `unidad_gestion_asignada` | AUC inflado | Excluida del modelo |
| 16 registros `En_Quiebra` | Contaminación | Eliminados |
| Valores negativos en consultas | Ruido | Clip a 0 |
| 899 ejecutivos únicos | Overfitting | Excluido |
""")

    st.divider()
    st.subheader("Rendimiento del modelo base (P2)")
    c1, c2, c3 = st.columns(3)
    c1.metric("ROC-AUC (CV)", f"{cv_scores.mean():.3f} ± {cv_scores.std():.3f}",
              help="0.5=azar, 1.0=perfecto")
    c2.metric("PR-AUC", f"{ap:.3f}",
              help=f"Baseline={y_test.mean():.3f} | {ap/y_test.mean():.1f}x mejor que azar")
    c3.metric("Morosos detectados (recall)", f"{tp/(tp+fn)*100:.1f}%",
              help="Con umbral=0.5")

    st.info(f"""
**Interpretación para Gerencia:**
- El modelo identifica correctamente al cliente más riesgoso el **{cv_scores.mean()*100:.0f}% de las veces** (vs 50% al azar).
- De cada 100 clientes que el modelo marca como riesgosos, **{ap/y_test.mean():.1f}x más** son morosos reales que si se eligieran al azar.
- El modelo es un **punto de partida**. Con más variables (historial de pagos, flujo de caja) el rendimiento mejoraría significativamente.
""")

    st.divider()
    st.subheader("Campaña '1 mes de gracia' — Comercio (P4)")
    st.error("""
**Recomendación: NO lanzar la campaña masiva sin un piloto controlado.**

El análisis muestra que la campaña aumentaría la tasa de default en Comercio por:
selección adversa + concentración sectorial + señal de fragilidad estructural del rubro.

""")

    st.divider()
    st.subheader("Variables del modelo final")
    feat_imp = pd.DataFrame({
        "Variable": X.columns,
        "Importancia": model.feature_importances_
    }).sort_values("Importancia", ascending=False)
    feat_imp["Importancia (%)"] = (feat_imp["Importancia"] / feat_imp["Importancia"].sum() * 100).round(1)
    st.dataframe(feat_imp[["Variable","Importancia (%)"]].reset_index(drop=True),
                 use_container_width=True, hide_index=True)
