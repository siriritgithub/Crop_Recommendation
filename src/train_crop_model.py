import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import pickle

# Dataset path
dataset_path = r"C:\crop-recommendation-price-prediction\data\Crop_recommendation_clean.csv"

# Load dataset
df = pd.read_csv(dataset_path)

# Clean soil_type column (capitalize, strip)
df['soil_type'] = df['soil_type'].astype(str).str.strip().str.capitalize()

# Define soil types and add 'Unknown' if needed
soil_types = ['Clay', 'Sandy', 'Loamy']
df['soil_type'] = df['soil_type'].apply(lambda x: x if x in soil_types else 'Unknown')
if 'Unknown' in df['soil_type'].unique():
    soil_types.append('Unknown')

# Drop rows with missing essential values
df = df.dropna(subset=['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall', 'soil_type', 'label'])

# Encode soil_type feature
df['soil_encoded'] = df['soil_type'].apply(lambda x: soil_types.index(x))

# Prepare features and target
X = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall', 'soil_encoded']]
y = df['label']  # assuming 'label' column contains crop names

# Encode crop labels
crop_encoder = LabelEncoder()
y_encoded = crop_encoder.fit_transform(y)

# Train RandomForest model
crop_model = RandomForestClassifier(random_state=42)
crop_model.fit(X, y_encoded)

# Save model and encoder
pickle.dump(crop_model, open(r"C:\crop-recommendation-price-prediction\src\crop_model.pkl", 'wb'))
pickle.dump(crop_encoder, open(r"C:\crop-recommendation-price-prediction\src\label_encoder.pkl", 'wb'))

print("✅ Crop model and label encoder saved successfully!")
