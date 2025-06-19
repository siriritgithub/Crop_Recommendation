# train_models.py

import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import os

# Setup model output path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# Load dataset (adjusted path)
data_path = os.path.abspath(os.path.join(BASE_DIR, "..", "data", "Crop_recommendation.csv"))
df = pd.read_csv(data_path)

# Features and target
X = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
y = df['label']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save model
with open(os.path.join(MODEL_DIR, "crop_recommendation_model.pkl"), "wb") as f:
    pickle.dump(model, f)

print("✅ Crop Recommendation Model trained and saved.")
