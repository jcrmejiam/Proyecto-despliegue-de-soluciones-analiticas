import streamlit as st
import joblib
import pandas as pd
from sentence_transformers import SentenceTransformer
from tensorflow.keras.models import load_model


@st.cache_resource
def cargar_modelo_generos():
    modelo1 = joblib.load("modelo_generos.pkl")
    vector = joblib.load("vector_generos.pkl")
    mlb = joblib.load("etiquetas_generos.pkl")
    return modelo1, vector, mlb

modelo1, vector, mlb = cargar_modelo_generos()


@st.cache_resource
def cargar_modelo_rating():
    embedding = SentenceTransformer('all-mpnet-base-v2')
    modelo3_red = load_model("modelo_rating_bert.keras", compile=False)
    return embedding, modelo3_red

embedding, modelo3_red = cargar_modelo_rating()



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

with col2:
    with st.container(border=True):
        st.subheader("Resultado")

        st.write("**Valoración estimada**")

        if predecir and trama.strip():
            trama_embedding = embedding.encode([trama])
            rating = modelo3_red.predict(trama_embedding, verbose=0)
            st.write(f"{float(rating[0][0]):.2f} / 10")
        else:
            st.write("Pendiente de predicción")

        st.write("**Géneros sugeridos**")

        if predecir and trama.strip():
            trama_vectorizada = vector.transform([trama])
            probabilidades = modelo1.predict_proba(trama_vectorizada)[0]

            resultados = pd.DataFrame({"Genero": mlb.classes_, "Probabilidad": probabilidades})

            resultados = resultados.sort_values(by="Probabilidad", ascending=False)

            for _, fila in resultados.head(3).iterrows():
                st.write(f"{fila['Genero']}: {fila['Probabilidad']:.0%}")
                st.progress(float(fila["Probabilidad"]))
        else:
            st.write("Pendiente predicción")



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
