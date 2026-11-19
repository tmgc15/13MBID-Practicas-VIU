
from pathlib import Path

import pandas as pd


RUTA_MODELO = Path("models/modelo_sentimientos.pkl")
RUTA_METRICAS = Path("reports/metricas_entrenamiento.csv")


def test_archivo_modelo_existe():
    assert RUTA_MODELO.exists(), "No se encontró el modelo entrenado"


def test_archivo_metricas_existe():
    assert RUTA_METRICAS.exists(), "No se encontró el archivo de métricas"


def test_exactitud_minima():
    metricas = pd.read_csv(RUTA_METRICAS)
    accuracy = metricas.loc[0, "accuracy_test"]

    assert accuracy >= 0.80, (
        f"Exactitud insuficiente: {accuracy:.2%}"
    )


def test_f1_macro_minimo():
    metricas = pd.read_csv(RUTA_METRICAS)
    f1_macro = metricas.loc[0, "f1_macro_test"]

    assert f1_macro >= 0.80, (
        f"F1 macro insuficiente: {f1_macro:.2%}"
    )
