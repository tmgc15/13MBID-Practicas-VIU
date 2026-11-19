
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# 1. Rutas del proyecto
RUTA_DATOS = Path("data/processed/datos_integrados.csv")
RUTA_MODELO = Path("models/modelo_sentimientos.pkl")
RUTA_METRICAS = Path("reports/metricas_entrenamiento.csv")

RUTA_MODELO.parent.mkdir(parents=True, exist_ok=True)
RUTA_METRICAS.parent.mkdir(parents=True, exist_ok=True)


# 2. Configurar MLflow con SQLite
# Compatible con MLflow 3.16.1
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Prediccion_sentimientos_clientes")


# 3. Cargar los datos procesados
df = pd.read_csv(RUTA_DATOS)

X = df.drop(columns=["sentimiento"])
y = df["sentimiento"]


# 4. Separar entrenamiento, validacion y prueba
X_temp, X_test, y_temp, y_test = train_test_split(
    X,
    y,
    test_size=0.10,
    random_state=42,
    stratify=y,
)

X_train, X_val, y_train, y_val = train_test_split(
    X_temp,
    y_temp,
    test_size=0.22,
    random_state=42,
    stratify=y_temp,
)


# 5. Identificar los tipos de variables
text_col = "texto_resena"

num_cols = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

cat_cols = [
    col
    for col in X.select_dtypes(
        include=["object", "string"]
    ).columns
    if col != text_col
]


# 6. Preparar las variables
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ("text", TfidfVectorizer(max_features=300), text_col),
])


# 7. Definir el modelo
modelo = Pipeline([
    ("prep", preprocessor),
    (
        "model",
        LogisticRegression(
            max_iter=2000,
            random_state=42,
        ),
    ),
])


# 8. Iniciar y registrar el experimento en MLflow
with mlflow.start_run(run_name="Regresion_logistica_v1"):

    # Registrar parametros
    mlflow.log_param("algoritmo", "LogisticRegression")
    mlflow.log_param("max_iter", 2000)
    mlflow.log_param("max_features_tfidf", 300)
    mlflow.log_param("random_state", 42)
    mlflow.log_param("test_size", 0.10)
    mlflow.log_param("validation_size_sobre_temporal", 0.22)

    # Entrenar modelo
    modelo.fit(X_train, y_train)

    # Generar predicciones
    pred_val = modelo.predict(X_val)
    pred_test = modelo.predict(X_test)

    # Calcular metricas
    resultados = {
        "accuracy_validacion": accuracy_score(y_val, pred_val),
        "f1_macro_validacion": f1_score(
            y_val,
            pred_val,
            average="macro",
        ),
        "accuracy_test": accuracy_score(y_test, pred_test),
        "f1_macro_test": f1_score(
            y_test,
            pred_test,
            average="macro",
        ),
    }

    # Registrar metricas en MLflow
    for nombre, valor in resultados.items():
        mlflow.log_metric(nombre, valor)

    # Registrar el modelo en MLflow
    mlflow.sklearn.log_model(
        sk_model=modelo,
        name="modelo_sentimientos",
        input_example=X_train.head(3),
    )

    # Guardar el modelo para la API
    joblib.dump(modelo, RUTA_MODELO)

    # Guardar las metricas para DVC
    pd.DataFrame([resultados]).to_csv(
        RUTA_METRICAS,
        index=False,
    )

    # Mostrar resumen
    print("\n=== RESULTADOS DEL ENTRENAMIENTO ===")

    print("Registros de entrenamiento:", len(X_train))
    print("Registros de validacion:", len(X_val))
    print("Registros de prueba:", len(X_test))

    for nombre, valor in resultados.items():
        print(f"{nombre}: {valor:.4f}")

    print("\nModelo guardado en:", RUTA_MODELO)
    print("Metricas guardadas en:", RUTA_METRICAS)
    print("Experimento registrado correctamente en MLflow")
