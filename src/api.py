
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


# Ubicación del modelo entrenado
RUTA_MODELO = Path("deployment/modelo_sentimientos.pkl")

if not RUTA_MODELO.exists():
    RUTA_MODELO = Path("models/modelo_sentimientos.pkl")

if not RUTA_MODELO.exists():
    raise RuntimeError(
        "No se encontró el modelo entrenado. "
        "Ejecuta primero el pipeline DVC."
    )

modelo = joblib.load(RUTA_MODELO)

app = FastAPI(
    title="API de predicción de sentimientos",
    description=(
        "Servicio de clasificación de reseñas de clientes "
        "mediante un modelo de aprendizaje automático."
    ),
    version="1.0.0",
)


# Datos que debe proporcionar el usuario
class ResenaEntrada(BaseModel):
    texto_resena: str = Field(
        ..., min_length=1, description="Texto de la reseña"
    )
    categoria_producto: str
    canal_compra: str
    plataforma: str
    tiempo_respuesta_horas: float
    problema_resuelto: str
    reclamo_formal: str
    importe_compra: float
    edad: float
    genero: str
    ciudad: str
    antiguedad_cliente_meses: int


@app.get("/")
def inicio():
    return {
        "mensaje": "API de predicción de sentimientos disponible",
        "version": "1.0.0",
    }


@app.get("/salud")
def verificar_salud():
    return {
        "estado": "operativa",
        "modelo_cargado": True,
    }


@app.post("/predecir")
def predecir_sentimiento(resena: ResenaEntrada) -> dict[str, Any]:

    try:
        datos = resena.model_dump()

        # Variables calculadas durante la preparación de datos
        texto = datos["texto_resena"]

        datos["longitud_resena"] = len(texto)

        datos["cantidad_palabras_resena"] = len(texto.split())

        datos["tiempo_respuesta_dias"] = (
            datos["tiempo_respuesta_horas"] / 24
        )

        # Preparar entrada con las mismas variables del entrenamiento
        entrada = pd.DataFrame([datos])

        # Realizar predicción
        prediccion = modelo.predict(entrada)[0]

        return {
            "sentimiento_predicho": str(prediccion),
            "mensaje": "Predicción realizada correctamente",
        }

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Error al realizar la predicción: {error}",
        )
