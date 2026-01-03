
from fastapi import FastAPI
import joblib
import pandas as pd
from pathlib import Path

MODEL_PATH = Path("models/model.joblib")
DATA_PATH = Path("data/processed/water.parquet")

app = FastAPI(title="Water Anomaly API")

model = joblib.load(MODEL_PATH)
columns = pd.read_parquet(DATA_PATH).columns.tolist()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(data: dict):
    # valia se tods as colunas existem
    if set(data.keys()) != set(map(str, columns)):
        raise HTTPException(
            status_code=400,
            detail="JSON precisa conter todas as colunas do modelo"
        )

    df = pd.DataFrame([data])
    df = df[columns] #garante ordem correta

    pred = model.predict(df)[0]
    result = "ANOMALIA" if pred == -1 else "NORMAL"
    
    return {"result": result}
