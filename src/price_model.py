import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pickle

# Load dataset
file_path = r"C:\crop-recommendation-price-prediction\data\crop_price_dataset.csv"
df = pd.read_csv(file_path)

# Convert date
df['month'] = pd.to_datetime(df['month'], errors='coerce')
df['year'] = df['month'].dt.year
df['month_num'] = df['month'].dt.month

# Encode categorical features
le_crop = LabelEncoder()
df['commodity_encoded'] = le_crop.fit_transform(df['commodity_name'])

# Drop rows with missing values
df.dropna(subset=['avg_modal_price', 'commodity_name', 'month_num'], inplace=True)

# Features and Target
X = df[['commodity_encoded', 'month_num']]
y = df['avg_modal_price']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Accuracy
print(f"\n✅ Model Score: {model.score(X_test, y_test) * 100:.2f}%")

# Save model and label encoder
with open('price_model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('crop_encoder.pkl', 'wb') as f:
    pickle.dump(le_crop, f)

print("✅ Model saved as price_model.pkl and crop_encoder.pkl")
