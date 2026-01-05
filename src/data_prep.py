import pandas as pd
from pathlib import Path

# caminhos
RAW_PATH = Path("data/raw/water.csv")
SAMPLE_PATH = Path("data/sample/water_sample.csv")
PROCESSED_PATH = Path("data/processed/water.parquet")


def load_data():
    if RAW_PATH.exists():
        print("Usando dados RAW")
        df = pd.read_csv(RAW_PATH)

    elif SAMPLE_PATH.exists():
        print("Usando dados SAMPLE")
        df = pd.read_csv(SAMPLE_PATH)

    else:
        raise FileNotFoundError(
            "Nenhum dado encontrado. Forneça data/raw ou data/sample."
        )

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
