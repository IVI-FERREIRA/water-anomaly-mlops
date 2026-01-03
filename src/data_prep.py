import pandas as pd
from pathlib import Path

# caminhos
RAW_PATH = Path("data/raw/water.csv")
PROCESSED_PATH = Path("data/processed/water.parquet")

URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "water-treatment/water-treatment.data"
)


def load_data():
    if RAW_PATH.exists():
        df = pd.read_csv(RAW_PATH)
    else:
        df = pd.read_csv(URL, header=None)
        RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(RAW_PATH, index=False)
    return df


def clean_data(df):
    df = df.replace("?", pd.NA)
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.dropna(thresh=int(df.shape[1] * 0.7))
    return df


def main():
    df = load_data()
    df = clean_data(df)

    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(PROCESSED_PATH, index=False)

    print("Dados prontos!")
    print("Formato final:", df.shape)


if __name__ == "__main__":
    main()

