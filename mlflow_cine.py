"""Ejecuta el entrenamiento del proyecto de cine con registro en MLflow.

Guarde el archivo junto a app.py y requirements.txt. Ejecútelo desde esa
carpeta en la misma instancia EC2 donde MLflow escucha en el puerto 8050:

    python mlflow_cine.py

Puede indicar otro servidor mediante --tracking-uri o MLFLOW_TRACKING_URI.
Para una prueba con la base local, use --tracking-uri sqlite:///mlflow.db.

cine.entrenar prepara los datos, ajusta y evalúa los candidatos y genera
artifacts/mlflow_runs.csv. cine.experimentos abre una corrida por candidato
en cine-trama-rating o cine-trama-generos, registra parámetros y métricas,
y versiona los modelos seleccionados como cine-rating y cine-generos.
"""

import os

from cine.entrenar import main


if __name__ == "__main__":
    os.environ.setdefault("MLFLOW_TRACKING_URI", "http://127.0.0.1:8050")
    main()
