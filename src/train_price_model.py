"""
Trains the crop price prediction model on REAL historical monthly price data
(data/crop_price_dataset.csv) -- commodity, month, and year are used to learn
seasonal + trend patterns in avg_modal_price.

(An earlier version of this project predicted price from user-dragged
"market demand" / "supply" sliders with no real data behind them -- that has
been replaced. This model is fit purely on historical price records.)

Run from the project root: python src/train_price_model.py
"""
import json
from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

from data_preprocessing import load_price_data

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)


def main():
    df = load_price_data()

    le_commodity = LabelEncoder()
    df["commodity_encoded"] = le_commodity.fit_transform(df["commodity_name"])

    features = ["commodity_encoded", "month_num", "year"]
    X = df[features]
    y = df["avg_modal_price"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(
        n_estimators=150, max_depth=14, min_samples_leaf=3, random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Test R^2: {r2:.3f}")
    print(f"Test MAE: Rs {mae:.2f} per quintal")

    joblib.dump(model, MODEL_DIR / "price_prediction_model.pkl")
    joblib.dump(le_commodity, MODEL_DIR / "commodity_encoder.pkl")

    # Save last known price per commodity, used by the app to show a
    # "current vs predicted" comparison and a recent trend chart.
    latest = (
        df.sort_values("month")
        .groupby("commodity_name")
        .tail(12)[["commodity_name", "month", "avg_modal_price"]]
    )
    latest_json = {
        c: g[["month", "avg_modal_price"]].assign(month=lambda x: x["month"].dt.strftime("%Y-%m")).to_dict("records")
        for c, g in latest.groupby("commodity_name")
    }
    with open(MODEL_DIR / "price_history.json", "w") as f:
        json.dump(latest_json, f, indent=2)

    metrics = {"r2": round(r2, 4), "mae": round(mae, 2), "n_samples": int(len(df)),
               "commodities": sorted(df["commodity_name"].unique().tolist())}
    with open(MODEL_DIR / "price_model_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("\nSaved model, encoder, price history, and metrics to", MODEL_DIR)


if __name__ == "__main__":
    main()
