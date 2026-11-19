
import streamlit as st
import requests
import os

st.set_page_config(page_title="Evaluación de historias", layout="wide")

st.title("Evaluación de historias")
st.divider()

st.title("Predicción de peliculas")

col1, col2 = st.columns(2)

with col1:
    with st.container(border=True):
        st.subheader("Nueva historia")

        titulo = st.text_input("Título de referencia")
        trama = st.text_area("Trama en inglés", height=200)

        predecir = st.button("Ver resultado")

#Consultar la API
resultado = None
error = None

if predecir:
    if trama.strip() == "":
        st.warning("Por favor, ingresa la trama de la película.")
    else:
        try:
            respuesta = requests.post(
                os.getenv("API_URL", "http://127.0.0.1:8000/api/predecir"),
                json={"trama": trama},
                timeout=60
            )
            respuesta.raise_for_status()
            resultado = respuesta.json()

            if "error" in resultado:
                error = resultado["error"]
                resultado = None

        except requests.RequestException:
            error = "No se pudo conectar con la API."

with col2:
    with st.container(border=True):
        st.subheader("Resultado")

        st.write("**Valoración estimada**")

        if resultado:
            st.write(f"{resultado['rating']:.2f} / 10")
        else:
            st.write("Pendiente de predicción")

        st.write("**Géneros sugeridos**")

        if resultado:
            for genero in resultado["generos"]:
                nombre = genero["genero"]
                probabilidad = genero["probabilidad"]

                st.write(f"{nombre}: {probabilidad:.0%}")
                st.progress(float(probabilidad))
        else:
            st.write("Pendiente de predicción")

        st.caption(
            "Las etiquetas pueden coexistir; sus probabilidades "
            "no suman necesariamente 100 %."
        )

        st.divider()
        st.caption("Apoyo para evaluar historias.")

if error:
    st.error(error)
