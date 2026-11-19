
import requests
import streamlit as st

API_URL = "https://one3mbid-api-sentimientos.onrender.com/predecir"

st.set_page_config(
    page_title="Análisis de sentimientos",
    page_icon="📊",
    layout="centered",
)

st.title("Análisis de sentimientos de clientes")
st.write(
    "Esta herramienta utiliza un modelo de aprendizaje automático "
    "para clasificar las reseñas de clientes como positivas, "
    "neutrales o negativas."
)

st.divider()

with st.form("formulario_prediccion"):

    st.subheader("Información de la reseña")

    texto_resena = st.text_area(
        "Escribe la reseña del cliente",
        placeholder="Ejemplo: El producto llegó en excelentes condiciones.",
        height=120,
    )

    categoria_producto = st.text_input(
        "Categoría del producto",
        value="Electrónica",
    )

    canal_compra = st.text_input(
        "Canal de compra",
        value="Online",
    )

    plataforma = st.text_input(
        "Plataforma",
        value="Web",
    )

    st.subheader("Información de atención al cliente")

    tiempo_respuesta_horas = st.number_input(
        "Tiempo de respuesta (horas)",
        min_value=0.0,
        value=2.0,
    )

    problema_resuelto = st.selectbox(
        "¿Se resolvió el problema?",
        ["Sí", "No"],
    )

    reclamo_formal = st.selectbox(
        "¿Se presentó un reclamo formal?",
        ["No", "Sí"],
    )

    importe_compra = st.number_input(
        "Importe de la compra",
        min_value=0.0,
        value=1500.0,
    )

    st.subheader("Información del cliente")

    edad = st.number_input(
        "Edad",
        min_value=0,
        max_value=90,
        value=35,
    )

    genero = st.text_input(
        "Género",
        value="Femenino",
    )

    ciudad = st.text_input(
        "Ciudad",
        value="Santo Domingo",
    )

    antiguedad_cliente_meses = st.number_input(
        "Antigüedad del cliente (meses)",
        min_value=0,
        value=24,
    )

    enviar = st.form_submit_button(
        "Analizar sentimiento",
        use_container_width=True,
    )

if enviar:

    if not texto_resena.strip():
        st.warning("Debes introducir el texto de la reseña.")

    else:

        datos = {
            "texto_resena": texto_resena,
            "categoria_producto": categoria_producto,
            "canal_compra": canal_compra,
            "plataforma": plataforma,
            "tiempo_respuesta_horas": tiempo_respuesta_horas,
            "problema_resuelto": problema_resuelto,
            "reclamo_formal": reclamo_formal,
            "importe_compra": importe_compra,
            "edad": edad,
            "genero": genero,
            "ciudad": ciudad,
            "antiguedad_cliente_meses": antiguedad_cliente_meses,
        }

        try:

            respuesta = requests.post(
                API_URL,
                json=datos,
                timeout=30,
            )

            respuesta.raise_for_status()
            resultado = respuesta.json()

            sentimiento = resultado["sentimiento_predicho"]

            st.subheader("Resultado de la predicción")

            if sentimiento == "Positivo":
                st.success(f"Sentimiento predicho: {sentimiento}")

            elif sentimiento == "Neutral":
                st.info(f"Sentimiento predicho: {sentimiento}")

            elif sentimiento == "Negativo":
                st.error(f"Sentimiento predicho: {sentimiento}")

            else:
                st.write(f"Sentimiento predicho: {sentimiento}")

        except requests.exceptions.RequestException as error:

            st.error(
                "No se pudo conectar con la API. "
                "Comprueba que FastAPI esté en ejecución."
            )

            st.caption(f"Detalle técnico: {error}")

st.divider()

st.caption(
    "Proyecto académico - Máster en Big Data y Data Analysis | VIU"
)
