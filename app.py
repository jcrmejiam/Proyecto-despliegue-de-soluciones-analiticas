import streamlit as st

st.set_page_config(page_title="Evaluación de historias", layout="wide")

st.title("Cine · Evaluación de historias")
st.divider()

st.title("Predicción de peliculas")

col1, col2 = st.columns(2)

with col1:
    with st.container(border=True):
        st.subheader("Nueva historia")

        titulo = st.text_input("Título de referencia")
        trama = st.text_area("Trama en inglés", height=200)

        predecir = st.button("Ver resultado")

with col2:
    with st.container(border=True):
        st.subheader("Resultado")

        st.write("**Valoración estimada**")
        st.write("Pendiente de predicción")

        st.write("**Géneros sugeridos**")
        st.write("Pendiente de predicción")

        st.caption(
            "Las etiquetas pueden coexistir; sus probabilidades "
            "no suman necesariamente 100 %."
        )

        st.divider()
        st.caption("Apoyo para evaluar historias.")

if predecir:
    if trama.strip() == "":
        st.warning("Por favor, ingresa la trama de la película.")
    else:
        st.info("La trama está lista. Falta conectar los modelos de predicción.")