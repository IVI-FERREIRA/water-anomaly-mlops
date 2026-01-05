from fastapi import FastAPI, HTTPException
from src.infer import WaterAnomalyModel

app = FastAPI(title="Water Anomaly API")

model = WaterAnomalyModel()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(data: dict):
    try:
        result = model.predict(data)
        return {"result": result}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
