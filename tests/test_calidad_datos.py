
import pandas as pd

CLIENTES = pd.read_csv("data/raw/datos_clientes.csv", sep=";")
RESENAS = pd.read_csv("data/raw/datos_resenas.csv", sep=";")


def test_estructura_datasets():
    """Verifica la estructura esperada de los archivos."""
    assert CLIENTES.shape[1] == 6
    assert RESENAS.shape[1] == 13


def test_identificadores_completos():
    """Comprueba que los identificadores no estén vacíos."""
    assert CLIENTES["id_cliente"].notna().all()
    assert RESENAS["id_cliente"].notna().all()
    assert RESENAS["id_resena"].notna().all()


def test_completitud_datos():
    """Verifica umbrales definidos para valores faltantes."""
    assert CLIENTES["edad"].isna().mean() <= 0.03
    assert RESENAS["tiempo_respuesta_horas"].isna().mean() <= 0.04


def test_duplicados():
    """Verifica el porcentaje máximo tolerado de duplicados."""
    assert CLIENTES.duplicated().mean() <= 0.01
    assert RESENAS.duplicated().mean() <= 0.01


def test_validez_edades():
    """Comprueba que las edades inválidas no superen el umbral."""
    edades_invalidas = (
        CLIENTES["edad"].notna()
        & ~CLIENTES["edad"].between(0, 90)
    )
    assert edades_invalidas.mean() <= 0.005


def test_validez_sentimientos():
    """Verifica que las clases correspondan al problema definido."""
    valores_validos = {"Positivo", "Neutral", "Negativo"}
    assert RESENAS["sentimiento"].isin(valores_validos).all()


def test_integridad_referencial():
    """Comprueba la relación de reseñas con clientes."""
    clientes_validos = set(CLIENTES["id_cliente"].dropna())
    assert RESENAS["id_cliente"].isin(clientes_validos).all()
