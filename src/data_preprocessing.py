import pandas as pd

def load_data(filepath):
    """
    Load dataset from CSV file.
    """
    data = pd.read_csv(filepath)
    return data

def check_missing(data):
    """
    Check for missing values in the dataset.
    """
    missing = data.isnull().sum()
    print("Missing values per column:\n", missing)

def preprocess_data(data):
    """
    Basic preprocessing steps like:
    - Handle missing values (if any)
    - Clean soil_type strings and fix typos
    - Data type conversion
    - Encode categorical columns
    """
    # Fill missing soil_type values
    data['soil_type'] = data['soil_type'].fillna('unknown')

    # Clean soil_type strings
    data['soil_type'] = data['soil_type'].str.strip().str.lower()
    data['soil_type'] = data['soil_type'].replace({'blacl': 'black'})

    # Encode soil_type
    soil_types = sorted(data['soil_type'].unique())
    soil_map = {soil: idx for idx, soil in enumerate(soil_types)}
    data['soil_encoded'] = data['soil_type'].map(soil_map)

    # Convert 'label' column to categorical type safely
    data['label'] = data['label'].astype('category')

    return data, soil_map

def save_clean_data(data, filepath):
    data.to_csv(filepath, index=False)
    print(f"✅ Cleaned data saved to {filepath}")

if __name__ == "__main__":
    filepath = r'C:\crop-recommendation-price-prediction\data\Crop_recommendation.csv'
    df = load_data(filepath)
    print("Data shape:", df.shape)
    check_missing(df)
    df_clean, soil_map = preprocess_data(df)   # Unpack both returned values
    print("Sample data after preprocessing:")
    print(df_clean.head())
    print("\nUnique soil types:", list(soil_map.keys()))
    
    cleaned_filepath = r'C:\crop-recommendation-price-prediction\data\Crop_recommendation_clean.csv'
    save_clean_data(df_clean, cleaned_filepath)
