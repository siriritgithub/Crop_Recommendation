"""
Loads and lightly cleans the crop recommendation dataset.

Note: the raw dataset ships with a 'soil_type' column that is ~99.8% missing
(2195 of 2200 rows are blank/unknown). It is not usable as a model feature,
so it is dropped here rather than imputed or encoded -- imputing a column
that's almost entirely missing would just be injecting noise labeled as
signal. It's kept as raw metadata only for transparency.
"""
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def load_crop_data(filepath: Path = None) -> pd.DataFrame:
    filepath = filepath or (DATA_DIR / "Crop_recommendation.csv")
    return pd.read_csv(filepath)


def clean_crop_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.dropna(subset=["N", "P", "K", "temperature", "humidity", "ph", "rainfall", "label"])
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    if "soil_type" in df.columns:
        df = df.drop(columns=["soil_type"])
    return df


def load_price_data(filepath: Path = None) -> pd.DataFrame:
    filepath = filepath or (DATA_DIR / "crop_price_dataset.csv")
    df = pd.read_csv(filepath)
    df["month"] = pd.to_datetime(df["month"], errors="coerce")
    df = df.dropna(subset=["month", "commodity_name", "avg_modal_price"])
    df["year"] = df["month"].dt.year
    df["month_num"] = df["month"].dt.month
    df["commodity_name"] = df["commodity_name"].astype(str).str.strip()
    return df


if __name__ == "__main__":
    crop_df = clean_crop_data(load_crop_data())
    print("Crop data:", crop_df.shape, "| crops:", crop_df["label"].nunique())
    price_df = load_price_data()
    print("Price data:", price_df.shape, "| commodities:", price_df["commodity_name"].nunique())
