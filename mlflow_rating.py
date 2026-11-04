
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import load_model
import mlflow

#datos
dataTraining=pd.read_csv("dataTraining.csv")
dataTraining=dataTraining.drop(columns=['Unnamed: 0'])

print("Valores nulos: \n", dataTraining[["plot", "rating", "genres"]].isnull().sum())
print("Registros duplicados: \n", dataTraining.duplicated().sum())

dataTraining=dataTraining.drop_duplicates()

print("Estadisticas del rating:\n", dataTraining["rating"].describe())

x=dataTraining["plot"]
y=dataTraining["rating"]

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=0)

vector=TfidfVectorizer(stop_words='english', max_features=20000)

X_train_vectorizado=vector.fit_transform(X_train)
X_test_vectorizado=vector.transform(X_test)

#MLflow
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Prediccion_rating_peliculas")

#Modelo 1
with mlflow.start_run(run_name="Ridge"):

    modelo1=Ridge(alpha=1.0)

    modelo1.fit(X_train_vectorizado, y_train)
    y_pred1=modelo1.predict(X_test_vectorizado)

    mae=mean_absolute_error(y_test, y_pred1)
    rmse=np.sqrt(mean_squared_error(y_test, y_pred1))
    r2=r2_score(y_test, y_pred1)

    print("mae:", round(mae, 4))
    print("rmse:", round(rmse, 4))
    print("R^2:", round(r2, 4))

    mlflow.log_param("alpha", 1.0)
    mlflow.log_param("vectorizador", "TF-IDF")

    mlflow.log_metric("MAE", mae)
    mlflow.log_metric("RMSE", rmse)
    mlflow.log_metric("R2", r2)

#Modelo 2
with mlflow.start_run(run_name="SVR_lineal"):

    modelo2=SVR(kernel="linear", C=1.0)

    modelo2.fit(X_train_vectorizado, y_train)
    y_pred2=modelo2.predict(X_test_vectorizado)

    mae2=mean_absolute_error(y_test, y_pred2)
    rmse2=np.sqrt(mean_squared_error(y_test, y_pred2))
    r22=r2_score(y_test, y_pred2)

    print("mae:", round(mae2, 4))
    print("rmse:", round(rmse2, 4))
    print("R^2:", round(r22, 4))

    mlflow.log_param("kernel", "linear")
    mlflow.log_param("C", 1.0)
    mlflow.log_param("vectorizador", "TF-IDF")

    mlflow.log_metric("MAE", mae2)
    mlflow.log_metric("RMSE", rmse2)
    mlflow.log_metric("R2", r22)


#Modelo 3, en el notebook ya esta descargado el archivo de embeddings porque es mas facil cargarlos así
X_train_embedding = np.load("X_train_embedding.npy")
X_test_embedding = np.load("X_test_embedding.npy")

modelo3_red = load_model("modelo_rating_bert.keras")