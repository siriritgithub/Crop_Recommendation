import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib  # for saving the model

def load_data(filepath):
    data = pd.read_csv(filepath)
    return data

def preprocess_data(data):
    data = data.dropna()
    return data

def train_model(data):
    X = data.drop('label', axis=1)
    y = data['label']

    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy*100:.2f}%")

    return model, le

def save_model(model, label_encoder, model_path='crop_model.pkl', le_path='label_encoder.pkl'):
    joblib.dump(model, model_path)
    joblib.dump(label_encoder, le_path)
    print(f"Model saved to {model_path} and label encoder saved to {le_path}")

if __name__ == "__main__":
    filepath = r'C:\crop-recommendation-price-prediction\data\Crop_recommendation.csv'
    data = load_data(filepath)
    data = preprocess_data(data)
    model, le = train_model(data)
    save_model(model, le)
