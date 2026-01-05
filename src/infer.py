import joblib
import pandas as pd
from pathlib import Path

MODEL_PATH = Path("models/model.joblib")
DATA_PATH = Path("data/processed/water.parquet")


class WaterAnomalyModel:
    def __init__(self):
        self.model = joblib.load(MODEL_PATH)
        self.columns = pd.read_parquet(DATA_PATH).columns.tolist()

    def predict(self, data: dict) -> str:
        if set(data.keys()) != set(map(str, self.columns)):
            raise ValueError("JSON precisa conter todas as colunas")

        df = pd.DataFrame([data])
        df = df[self.columns]

        pred = self.model.predict(df)[0]
        return "ANOMALIA" if pred == -1 else "NORMAL"
