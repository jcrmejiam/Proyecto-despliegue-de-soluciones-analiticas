import pandas as pd
import ast
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
import joblib

dataTraining = pd.read_csv("dataTraining.csv")


dataTraining = dataTraining.drop(columns=['Unnamed: 0'])
dataTraining = dataTraining.drop_duplicates()
dataTraining["genres"] = dataTraining["genres"].apply(ast.literal_eval)

######
x = dataTraining["plot"]
y = dataTraining["genres"]


mlb = MultiLabelBinarizer()
y_binario = mlb.fit_transform(y)


X_train, X_test, y_train, y_test = train_test_split(x, y_binario, test_size=0.33, random_state=0)


vector = TfidfVectorizer(stop_words='english', max_features=20000)

X_train_vectorizado = vector.fit_transform(X_train)
X_test_vectorizado = vector.transform(X_test)

#modelo 1 
modelo1 = OneVsRestClassifier(LogisticRegression(max_iter=1000))

modelo1.fit(X_train_vectorizado, y_train)
y_pred1 = modelo1.predict(X_test_vectorizado)


f1 = f1_score(y_test, y_pred1, average='micro', zero_division=0)
precision = precision_score(y_test, y_pred1, average='micro', zero_division=0)
recall = recall_score(y_test, y_pred1, average='micro', zero_division=0)

y_prob1 = modelo1.predict_proba(X_test_vectorizado)
auc = roc_auc_score(y_test, y_prob1, average='micro')
auc_macro = roc_auc_score(y_test, y_prob1, average='macro')

print("Modelo 1 Regresión Logística")
print("F1-score:", round(f1, 4))
print("Precision:", round(precision, 4))
print("Recall:", round(recall, 4))
print("ROC AUC micro:", round(auc, 4))
print("ROC AUC macro:", round(auc_macro, 4))

# el umbral se puede modificar por ahora se hará con 0.3
y_pred1_03 = (y_prob1 >= 0.3).astype(int)


f1_03 = f1_score(y_test, y_pred1_03, average='micro', zero_division=0)
precision_03 = precision_score(y_test, y_pred1_03, average='micro', zero_division=0)
recall_03 = recall_score(y_test, y_pred1_03, average='micro', zero_division=0)

print("\nRegresión Logística con Umbral 0.3")
print("F1-score_umbral_0.3:", round(f1_03, 4))
print("Precision_umbral_0.3:", round(precision_03, 4))
print("Recall_umbral_0.3:", round(recall_03, 4))

# Modelo 2
modelo2 = OneVsRestClassifier(RidgeClassifier(alpha=1.0))
modelo2.fit(X_train_vectorizado, y_train)

y_pred2 = modelo2.predict(X_test_vectorizado)

f1_2 = f1_score(y_test, y_pred2, average='micro', zero_division=0)
precision_2 = precision_score(y_test, y_pred2, average='micro', zero_division=0)
recall_2 = recall_score(y_test, y_pred2, average='micro', zero_division=0)

print("\nModelo 2 Ridge")
print("F1-score:", round(f1_2, 4))
print("Precision:", round(precision_2, 4))
print("Recall:", round(recall_2, 4))

trama = ["A group of astronauts travels to an unknown planet and discovers a dangerous civilization."]

trama_vectorizada = vector.transform(trama)

probabilidades = modelo1.predict_proba(trama_vectorizada)[0]

resultados = pd.DataFrame({
    "Genero": mlb.classes_,
    "Probabilidad": probabilidades})

resultados = resultados.sort_values(by="Probabilidad", ascending=False)

print("\nPredicción de géneros:")
print(resultados.head(5))


#joblib.dump(modelo1, "modelo_generos.pkl")
#joblib.dump(vector, "vector_generos.pkl")
#joblib.dump(mlb, "etiquetas_generos.pkl")
