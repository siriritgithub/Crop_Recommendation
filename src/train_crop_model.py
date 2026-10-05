"""
Trains the crop recommendation model.

This is the ONE canonical training script for this model (an earlier version
of this project had two conflicting scripts producing models with different
feature sets -- that's been consolidated into this single source of truth).

Features: N, P, K, temperature, humidity, ph, rainfall  (7 numeric features)
Target:   label (22 crop classes)

Run from the project root: python src/train_crop_model.py
"""
import json
from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report

from data_preprocessing import load_crop_data, clean_crop_data

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]


def main():
    df = clean_crop_data(load_crop_data())
    X = df[FEATURES]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cv_scores = cross_val_score(model, X, y, cv=5)

    print(f"Test accuracy: {acc * 100:.2f}%")
    print(f"5-fold CV accuracy: {cv_scores.mean() * 100:.2f}% (+/- {cv_scores.std() * 100:.2f}%)")
    print("\nClassification report:\n", classification_report(y_test, y_pred))

    joblib.dump(model, MODEL_DIR / "crop_recommendation_model.pkl")

    # Save per-crop feature statistics -- used by the app to generate
    # plain-language "why this crop" explanations without needing SHAP.
    stats = df.groupby("label")[FEATURES].agg(["mean", "std"]).round(2)
    stats_dict = {
        crop: {feat: {"mean": stats.loc[crop, (feat, "mean")], "std": stats.loc[crop, (feat, "std")]}
               for feat in FEATURES}
        for crop in stats.index
    }
    with open(MODEL_DIR / "crop_feature_stats.json", "w") as f:
        json.dump(stats_dict, f, indent=2)

    metrics = {"test_accuracy": round(acc, 4), "cv_accuracy_mean": round(cv_scores.mean(), 4),
               "cv_accuracy_std": round(cv_scores.std(), 4), "n_classes": int(y.nunique()),
               "n_samples": int(len(df))}
    with open(MODEL_DIR / "crop_model_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("\nSaved model, feature stats, and metrics to", MODEL_DIR)


if __name__ == "__main__":
    main()
