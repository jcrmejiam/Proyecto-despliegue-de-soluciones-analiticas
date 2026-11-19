
from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI(title="Prediccion de peliculas")

#Cargar los modelos 
modelo_rating = joblib.load("modelo_rating_ridge.pkl")
vector_rating = joblib.load("vector_rating.pkl")

modelo_generos = joblib.load("modelo_generos.pkl")
vector_generos = joblib.load("vector_generos.pkl")
etiquetas_generos = joblib.load("etiquetas_generos.pkl")


class Pelicula(BaseModel):
    trama: str


@app.get("/api/salud")
def salud():
    return {"estado": "ok"}


@app.post("/api/predecir")
def predecir(pelicula: Pelicula):
    trama = pelicula.trama.strip()

    if not trama:
        return {"error": "Debes ingresar una trama"}

    #Prediccion del rating
    trama_rating = vector_rating.transform([trama])
    rating = float(modelo_rating.predict(trama_rating)[0])

    #Prediccion de los generos
    trama_generos = vector_generos.transform([trama])
    probabilidades = modelo_generos.predict_proba(trama_generos)[0]

    indices = probabilidades.argsort()[::-1][:3]

    generos = [{"genero": str(etiquetas_generos.classes_[i]),
            "probabilidad": round(float(probabilidades[i]), 4)}
        for i in indices]

    return {
        "rating": round(rating, 2),
        "generos": generos
    }
