import io
import json
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# =========================
# Configuración del proyecto
# =========================
TARGET = "Tension_Liquidez_bin"
EXCLUDED = {
    "ID_Observacion",
    "Fecha",
    "Prob_Tension_Liquidez",
    "Brecha_Caja_90d_MXN",
}
CATEGORICAL_FEATURES = ["Sector"]

EXPECTED_FEATURES = [
    "Sector",
    "Ventas_12M_MXN",
    "Crecimiento_Ventas_pct",
    "Margen_EBITDA_pct",
    "DSO_dias",
    "DIO_dias",
    "DPO_dias",
    "CCC_dias",
    "CxC_MXN",
    "Inventarios_MXN",
    "CxP_MXN",
    "NWC_MXN",
    "Deuda_CP_MXN",
    "Caja_MXN",
    "Linea_Credito_Disponible_MXN",
    "Concentracion_Top5_Clientes_pct",
    "Morosidad_CxC_pct",
    "Inventario_Obsoleto_pct",
    "Volatilidad_Ventas_pct",
    "Estacionalidad_bin",
    "EBITDA_MXN",
]
NUMERIC_FEATURES = [c for c in EXPECTED_FEATURES if c not in CATEGORICAL_FEATURES]

PARAM_GRID = {
    "model__n_estimators": [100, 200, 300],
    "model__max_depth": [None, 5, 10, 15],
    "model__min_samples_split": [2, 5, 10],
    "model__min_samples_leaf": [1, 2, 4],
    "model__max_features": ["sqrt", "log2"],
}

RANDOM_STATE = 42
TEST_SIZE = 0.20

st.set_page_config(
    page_title="Random Forest — Tensión de Liquidez",
    layout="wide",
)


# =========================
# Utilidades
# =========================
def read_uploaded_file(uploaded_file: st.runtime.uploaded_file_manager.UploadedFile):
    """Lee CSV o Excel. En Excel prioriza la hoja Datos_Modelo."""
    raw = uploaded_file.getvalue()
    suffix = Path(uploaded_file.name).suffix.lower()

    if suffix == ".csv":
        # Intento UTF-8; luego latin-1 como respaldo.
        try:
            return pd.read_csv(io.BytesIO(raw)), "CSV"
        except UnicodeDecodeError:
            return pd.read_csv(io.BytesIO(raw), encoding="latin-1"), "CSV"

    if suffix in {".xlsx", ".xls"}:
        xls = pd.ExcelFile(io.BytesIO(raw))
        sheet = "Datos_Modelo" if "Datos_Modelo" in xls.sheet_names else xls.sheet_names[0]
        return pd.read_excel(io.BytesIO(raw), sheet_name=sheet), sheet

    raise ValueError("Formato no soportado. Carga un archivo .csv, .xlsx o .xls")


def validate_and_prepare(df: pd.DataFrame):
    """Valida columnas, objetivo y tipos; devuelve copia preparada + diagnóstico."""
    issues = []
    warnings = []
    info = {}

    missing = [c for c in [TARGET] + EXPECTED_FEATURES if c not in df.columns]
    if missing:
        issues.append("Faltan columnas requeridas: " + ", ".join(missing))
        return None, issues, warnings, info

    work = df.copy()

    # Objetivo: se exige binario 0/1.
    target_numeric = pd.to_numeric(work[TARGET], errors="coerce")
    if target_numeric.isna().any():
        issues.append(f"La variable objetivo '{TARGET}' contiene valores no numéricos o vacíos.")
    else:
        unique_target = sorted(target_numeric.dropna().unique().tolist())
        if not set(unique_target).issubset({0, 1}):
            issues.append(
                f"La variable objetivo '{TARGET}' debe contener únicamente 0 y 1. "
                f"Valores detectados: {unique_target}"
            )
        work[TARGET] = target_numeric.astype(int)

    # Conversión controlada de columnas numéricas.
    coerced_report = {}
    for col in NUMERIC_FEATURES:
        before_non_null = work[col].notna()
        converted = pd.to_numeric(work[col], errors="coerce")
        introduced_na = int((before_non_null & converted.isna()).sum())
        if introduced_na:
            coerced_report[col] = introduced_na
        if converted.notna().sum() == 0:
            issues.append(f"La columna numérica '{col}' no contiene valores numéricos utilizables.")
        work[col] = converted

    if coerced_report:
        warnings.append(
            "Se detectaron valores no numéricos en columnas numéricas; se tratarán como faltantes: "
            + "; ".join(f"{k}={v}" for k, v in coerced_report.items())
        )

    # Sector se mantiene categórico y faltantes se imputan en pipeline.
    work["Sector"] = work["Sector"].astype("string")

    if len(work) < 50:
        warnings.append("La base tiene menos de 50 observaciones; la evaluación puede ser inestable.")

    if work[TARGET].nunique(dropna=True) < 2:
        issues.append("La variable objetivo necesita contener ambas clases (0 y 1).")

    class_counts = work[TARGET].value_counts(dropna=False).to_dict() if TARGET in work else {}
    info["rows"] = len(work)
    info["columns"] = len(work.columns)
    info["class_counts"] = class_counts
    info["missing_by_feature"] = work[EXPECTED_FEATURES].isna().sum().sort_values(ascending=False)
    info["extra_columns"] = [c for c in work.columns if c not in set(EXPECTED_FEATURES + [TARGET])]

    return work, issues, warnings, info


def build_pipeline(n_estimators=200, max_depth=None, min_samples_split=2,
                   min_samples_leaf=1, max_features="sqrt"):
    numeric_pipe = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median"))]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, NUMERIC_FEATURES),
            ("cat", categorical_pipe, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    return Pipeline(steps=[("prep", preprocessor), ("model", model)])


def metric_row(name, y_true, y_pred):
    return {
        "Modelo": name,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
    }


def confusion_df(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return pd.DataFrame(
        cm,
        index=["Real 0", "Real 1"],
        columns=["Predicho 0", "Predicho 1"],
    )


def feature_importance_df(fitted_pipeline):
    feature_names = fitted_pipeline.named_steps["prep"].get_feature_names_out()
    importances = fitted_pipeline.named_steps["model"].feature_importances_
    out = pd.DataFrame({"Variable": feature_names, "Importancia": importances})
    return out.sort_values("Importancia", ascending=False).reset_index(drop=True)


def make_excel_download(metrics_df, best_params, cm_df, importance_df, diagnostics_df):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        metrics_df.to_excel(writer, sheet_name="Metricas", index=False)
        pd.DataFrame(
            [{k.replace("model__", ""): v for k, v in best_params.items()}]
        ).to_excel(writer, sheet_name="Mejores_Parametros", index=False)
        cm_df.to_excel(writer, sheet_name="Matriz_Confusion")
        importance_df.to_excel(writer, sheet_name="Importancia", index=False)
        diagnostics_df.to_excel(writer, sheet_name="Diagnostico", index=False)
    output.seek(0)
    return output.getvalue()


def dataset_key(uploaded_file):
    return hashlib.sha256(uploaded_file.getvalue()).hexdigest()


# =========================
# Interfaz
# =========================
st.title("Random Forest — Tensión de Liquidez")
st.caption(
    "Clasificación binaria de Tension_Liquidez_bin con Random Forest, GridSearchCV y umbral ajustable."
)

with st.expander("Configuración metodológica", expanded=False):
    st.markdown(
        """
- **Objetivo:** `Tension_Liquidez_bin` (1 = tensión, 0 = normal)
- **Train/Test:** 80/20, estratificado, `random_state=42`
- **Imputación:** mediana en numéricas; moda en `Sector`
- **Sector:** One-Hot Encoding
- **Balance:** `class_weight='balanced'`
- **Optimización:** GridSearchCV, 5 folds, `scoring='f1'`
- **Excluidas:** `ID_Observacion`, `Fecha`, `Prob_Tension_Liquidez`, `Brecha_Caja_90d_MXN`
- **Sin escalamiento**
        """
    )

uploaded = st.file_uploader(
    "Carga la base de datos",
    type=["csv", "xlsx", "xls"],
    help="En Excel, si existe una hoja llamada 'Datos_Modelo', se utilizará automáticamente.",
)

if uploaded is None:
    st.info("Carga un archivo CSV o Excel para comenzar.")
    st.stop()

try:
    raw_df, source_sheet = read_uploaded_file(uploaded)
except Exception as exc:
    st.error(f"No fue posible leer el archivo: {exc}")
    st.stop()

prepared_df, issues, warnings, diag = validate_and_prepare(raw_df)

st.subheader("1. Diagnóstico de la base")
c1, c2, c3 = st.columns(3)
c1.metric("Observaciones", diag.get("rows", len(raw_df)))
c2.metric("Columnas", diag.get("columns", len(raw_df.columns)))
c3.metric("Origen", source_sheet)

if issues:
    st.error("La base no puede entrenarse todavía.")
    for msg in issues:
        st.write(f"- {msg}")
    st.stop()

for msg in warnings:
    st.warning(msg)

class_counts = prepared_df[TARGET].value_counts().sort_index()
class_table = pd.DataFrame(
    {
        "Clase": class_counts.index.astype(int),
        "Casos": class_counts.values,
        "Porcentaje": class_counts.values / class_counts.values.sum(),
    }
)
class_table["Significado"] = class_table["Clase"].map({0: "Normal", 1: "Tensión"})
st.dataframe(
    class_table.style.format({"Porcentaje": "{:.1%}"}),
    use_container_width=True,
    hide_index=True,
)

missing = diag["missing_by_feature"]
if int(missing.sum()) > 0:
    st.markdown("**Valores faltantes detectados (serán imputados):**")
    st.dataframe(
        missing[missing > 0].rename("Faltantes").to_frame(),
        use_container_width=True,
    )
else:
    st.success("No se detectaron valores faltantes en los predictores requeridos.")

with st.expander("Variables utilizadas y excluidas"):
    st.write("**Predictores utilizados:**", EXPECTED_FEATURES)
    present_excluded = [c for c in EXCLUDED if c in prepared_df.columns]
    st.write("**Variables excluidas presentes:**", present_excluded)
    if diag.get("extra_columns"):
        st.write("**Columnas adicionales ignoradas:**", diag["extra_columns"])

current_key = dataset_key(uploaded)
if st.session_state.get("dataset_key") != current_key:
    # Invalida entrenamiento previo al cambiar de archivo.
    for key in ["training_results", "dataset_key"]:
        st.session_state.pop(key, None)
    st.session_state["dataset_key"] = current_key

st.subheader("2. Entrenamiento y optimización")
st.write(
    "El modelo base se entrena primero. Después GridSearchCV evalúa 216 combinaciones "
    "con 5 folds (1,080 ajustes) y selecciona la de mayor F1 medio."
)

train_clicked = st.button("Entrenar Random Forest", type="primary")

if train_clicked:
    X = prepared_df[EXPECTED_FEATURES].copy()
    y = prepared_df[TARGET].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    with st.spinner("Entrenando modelo base..."):
        base_pipe = build_pipeline()
        base_pipe.fit(X_train, y_train)
        base_pred = base_pipe.predict(X_test)

    with st.spinner("Ejecutando GridSearchCV..."):
        grid_pipe = build_pipeline()
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        grid = GridSearchCV(
            estimator=grid_pipe,
            param_grid=PARAM_GRID,
            scoring="f1",
            cv=cv,
            n_jobs=-1,
            refit=True,
            return_train_score=False,
        )
        grid.fit(X_train, y_train)

    best_pipe = grid.best_estimator_
    opt_pred_050 = best_pipe.predict(X_test)
    opt_proba = best_pipe.predict_proba(X_test)[:, 1]

    metrics_050 = pd.DataFrame(
        [
            metric_row("Random Forest base", y_test, base_pred),
            metric_row("Random Forest optimizado", y_test, opt_pred_050),
        ]
    )

    importance = feature_importance_df(best_pipe)

    st.session_state["training_results"] = {
        "X_train_n": len(X_train),
        "X_test_n": len(X_test),
        "y_test": y_test.to_numpy(),
        "base_pred": base_pred,
        "opt_proba": opt_proba,
        "metrics_050": metrics_050,
        "best_params": grid.best_params_,
        "best_cv_f1": float(grid.best_score_),
        "importance": importance,
    }

if "training_results" not in st.session_state:
    st.info("Presiona **Entrenar Random Forest** para ejecutar el modelo.")
    st.stop()

res = st.session_state["training_results"]
y_test = np.asarray(res["y_test"])
opt_proba = np.asarray(res["opt_proba"])

st.success("Entrenamiento completado.")
col1, col2, col3 = st.columns(3)
col1.metric("Train", res["X_train_n"])
col2.metric("Test", res["X_test_n"])
col3.metric("Mejor F1 CV", f"{res['best_cv_f1']:.3f}")

st.markdown("**Mejores hiperparámetros encontrados:**")
best_params_clean = {k.replace("model__", ""): v for k, v in res["best_params"].items()}
st.json(best_params_clean)

st.subheader("3. Comparación: modelo base vs optimizado")
metrics_050 = res["metrics_050"].copy()
st.dataframe(
    metrics_050.style.format(
        {"Accuracy": "{:.3f}", "Precision": "{:.3f}", "Recall": "{:.3f}", "F1": "{:.3f}"}
    ),
    use_container_width=True,
    hide_index=True,
)

st.subheader("4. Umbral operativo del modelo optimizado")
threshold = st.slider(
    "Umbral de clasificación",
    min_value=0.05,
    max_value=0.95,
    value=0.50,
    step=0.01,
    help="Si P(tensión) >= umbral, la observación se clasifica como 1.",
)

threshold_pred = (opt_proba >= threshold).astype(int)
threshold_metrics = pd.DataFrame(
    [metric_row(f"Optimizado — umbral {threshold:.2f}", y_test, threshold_pred)]
)

m1, m2, m3, m4 = st.columns(4)
r = threshold_metrics.iloc[0]
m1.metric("Accuracy", f"{r['Accuracy']:.3f}")
m2.metric("Precision", f"{r['Precision']:.3f}")
m3.metric("Recall", f"{r['Recall']:.3f}")
m4.metric("F1", f"{r['F1']:.3f}")

st.subheader("5. Matriz de confusión")
cm = confusion_df(y_test, threshold_pred)

left, right = st.columns([1, 1.3])
with left:
    st.dataframe(cm, use_container_width=True)
    tn, fp, fn, tp = confusion_matrix(y_test, threshold_pred, labels=[0, 1]).ravel()
    st.write(f"**Verdaderos negativos:** {tn}")
    st.write(f"**Falsos positivos:** {fp}")
    st.write(f"**Falsos negativos:** {fn}")
    st.write(f"**Verdaderos positivos:** {tp}")

with right:
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm.values)
    ax.set_xticks([0, 1], ["Predicho 0", "Predicho 1"])
    ax.set_yticks([0, 1], ["Real 0", "Real 1"])
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Clase real")
    ax.set_title(f"Matriz de confusión — umbral {threshold:.2f}")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, int(cm.values[i, j]), ha="center", va="center")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    st.pyplot(fig, clear_figure=True)

st.subheader("6. Importancia de variables — Random Forest optimizado")
importance = res["importance"].copy()

# Para visualización compacta mostramos top 20; la descarga incluye todas.
top_n = min(20, len(importance))
plot_df = importance.head(top_n).sort_values("Importancia", ascending=True)
fig2, ax2 = plt.subplots(figsize=(8, max(5, top_n * 0.28)))
ax2.barh(plot_df["Variable"], plot_df["Importancia"])
ax2.set_xlabel("Importancia")
ax2.set_ylabel("Variable")
ax2.set_title(f"Top {top_n} variables por importancia")
st.pyplot(fig2, clear_figure=True)

st.dataframe(
    importance.style.format({"Importancia": "{:.6f}"}),
    use_container_width=True,
    hide_index=True,
)

st.caption(
    "Nota: las importancias internas del Random Forest pueden repartirse entre variables correlacionadas "
    "o derivadas. Deben interpretarse como relevancia predictiva del modelo, no como causalidad."
)

st.subheader("7. Descargar resultados")

# Diagnóstico resumido para exportación.
diag_rows = [
    {"Elemento": "Observaciones", "Valor": len(prepared_df)},
    {"Elemento": "Predictores", "Valor": len(EXPECTED_FEATURES)},
    {"Elemento": "Train", "Valor": res["X_train_n"]},
    {"Elemento": "Test", "Valor": res["X_test_n"]},
    {"Elemento": "Umbral exportado", "Valor": threshold},
    {"Elemento": "Mejor F1 CV", "Valor": res["best_cv_f1"]},
    {"Elemento": "Clase 0", "Valor": int((prepared_df[TARGET] == 0).sum())},
    {"Elemento": "Clase 1", "Valor": int((prepared_df[TARGET] == 1).sum())},
]
diagnostics_df = pd.DataFrame(diag_rows)

# Exportamos modelo base, optimizado a 0.50 y optimizado al umbral elegido.
export_metrics = pd.concat(
    [
        metrics_050,
        threshold_metrics,
    ],
    ignore_index=True,
)

excel_bytes = make_excel_download(
    metrics_df=export_metrics,
    best_params=res["best_params"],
    cm_df=cm,
    importance_df=importance,
    diagnostics_df=diagnostics_df,
)

st.download_button(
    "Descargar resultados en Excel",
    data=excel_bytes,
    file_name="resultados_random_forest_liquidez.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)
