# Random Forest — Tensión de Liquidez

Aplicación en Streamlit para clasificar `Tension_Liquidez_bin` mediante Random Forest.

## Funciones principales

- Carga de archivos CSV, XLSX y XLS.
- En Excel prioriza automáticamente la hoja `Datos_Modelo`.
- Validación de columnas requeridas y variable objetivo binaria.
- Imputación de faltantes: mediana en numéricas y moda en `Sector`.
- One-Hot Encoding para `Sector`.
- División 80/20 estratificada con `random_state=42`.
- Random Forest base.
- Optimización con GridSearchCV, 5 folds y F1 como métrica.
- `class_weight="balanced"`.
- Comparación Accuracy, Precision, Recall y F1.
- Umbral de clasificación ajustable.
- Matriz de confusión.
- Importancia estándar de variables.
- Descarga de resultados en Excel.

## Variables excluidas

- `ID_Observacion`
- `Fecha`
- `Prob_Tension_Liquidez`
- `Brecha_Caja_90d_MXN`

## Ejecutar

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Estructura esperada

La base debe contener `Tension_Liquidez_bin` y los 21 predictores definidos en `app.py`.

La variable objetivo debe estar codificada como:

- `0`: situación normal
- `1`: tensión de liquidez
