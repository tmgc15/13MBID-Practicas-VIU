
from pathlib import Path
import pandas as pd

# Rutas de entrada y salida
RUTA_CLIENTES = Path("data/raw/datos_clientes.csv")
RUTA_RESENAS = Path("data/raw/datos_resenas.csv")
RUTA_SALIDA = Path("data/processed/datos_integrados.csv")

# 1. Lectura
clientes = pd.read_csv(RUTA_CLIENTES, sep=";")
resenas = pd.read_csv(RUTA_RESENAS, sep=";")

print("Clientes originales:", len(clientes))
print("Reseñas originales:", len(resenas))

# 2. Eliminación de columnas
clientes = clientes.drop(columns=["canal_registro"])

# 3. Filtrado de edades y eliminación de duplicados
clientes = clientes[
    (clientes["edad"] < 90) | (clientes["edad"].isna())
].copy()

resenas = resenas.drop_duplicates().copy()

# 4. Tratamiento de valores faltantes
clientes["edad"] = clientes.groupby("ciudad")["edad"].transform(
    lambda x: x.fillna(x.median())
)

resenas["tiempo_respuesta_horas"] = (
    resenas.groupby("categoria_producto")["tiempo_respuesta_horas"]
    .transform(lambda x: x.fillna(x.median()))
)

# 5. Integración de los datasets
integrado = pd.merge(
    resenas,
    clientes,
    on="id_cliente",
    how="inner"
)

# 6. Creación de nuevas variables
integrado["longitud_resena"] = integrado["texto_resena"].str.len()

integrado["cantidad_palabras_resena"] = (
    integrado["texto_resena"].str.split().apply(len)
)

integrado["tiempo_respuesta_dias"] = (
    integrado["tiempo_respuesta_horas"] / 24
)

# 7. Eliminación de columnas no utilizadas
# Se elimina calificacion para evitar fuga de información
# hacia la variable objetivo sentimiento.
integrado = integrado.drop(
    columns=[
        "id_cliente",
        "id_resena",
        "calificacion",
        "fecha_resena",
    ]
)

# 8. Exportación del dataset procesado
RUTA_SALIDA.parent.mkdir(parents=True, exist_ok=True)
integrado.to_csv(RUTA_SALIDA, index=False)

# 9. Resumen para verificar el procesamiento
print("\n=== RESULTADOS DEL PROCESAMIENTO ===")
print("Clientes después del filtrado:", len(clientes))
print("Reseñas después de eliminar duplicados:", len(resenas))
print("Dimensiones del dataset integrado:", integrado.shape)
print("Valores faltantes totales:", int(integrado.isna().sum().sum()))
print("Archivo generado:", RUTA_SALIDA)
