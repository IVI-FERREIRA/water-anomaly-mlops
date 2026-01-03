import pandas as pd
from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib

DATA_PATH = Path("data/processed/water.parquet")
MODEL_PATH = Path("models/model.joblib")


def load_data():
    return pd.read_parquet(DATA_PATH)


def train_model(df):
    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", IsolationForest(
                n_estimators=100,
                contamination=0.05,
                random_state=42
            ))
        ]
    )
    pipeline.fit(df)
    return pipeline


def main():
    df = load_data()
    model = train_model(df)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    print("Modelo treinado e salvo!")
    print("Total de amostras:", df.shape[0])


if __name__ == "__main__":
    main()

