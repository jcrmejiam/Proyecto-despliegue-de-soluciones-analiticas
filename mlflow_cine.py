import os

from cine.entrenar import main


if __name__ == "__main__":
    os.environ.setdefault("MLFLOW_TRACKING_URI", "http://127.0.0.1:8050")
    main()
