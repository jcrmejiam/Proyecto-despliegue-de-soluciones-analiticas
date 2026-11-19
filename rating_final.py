
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

dataTraining = pd.read_csv("dataTraining.csv")
dataTraining = dataTraining.drop(columns=["Unnamed: 0"])
dataTraining = dataTraining.drop_duplicates()

x = dataTraining["plot"]
y = dataTraining["rating"]

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=0)

vector = TfidfVectorizer(stop_words="english", max_features=20000)
X_train_vectorizado = vector.fit_transform(X_train)
X_test_vectorizado = vector.transform(X_test)

modelo1 = Ridge(alpha=3)
modelo1.fit(X_train_vectorizado, y_train)

y_pred1 = modelo1.predict(X_test_vectorizado)

print("MAE:", round(mean_absolute_error(y_test, y_pred1), 4))
print("RMSE:", round(np.sqrt(mean_squared_error(y_test, y_pred1)), 4))
print("R2:", round(r2_score(y_test, y_pred1), 4))

#Guardar modelo y vectorizador
joblib.dump(modelo1, "modelo_rating_ridge.pkl")
joblib.dump(vector, "vector_rating.pkl")

print("Modelo y vectorizador guardados correctamente")
