import pandas as pd

# Load dataset
file_path = r"C:\crop-recommendation-price-prediction\data\crop_price_dataset.csv"
df = pd.read_csv(file_path)

# Display basic info
print("\n🔹 Columns:", df.columns.tolist())
print("🔹 Shape:", df.shape)
print("\n🔹 Sample rows:\n", df.head())

# Check for missing values
print("\n🔹 Missing values:\n", df.isnull().sum())

# Convert 'date' column if available
if 'date' in df.columns:
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    print("\n🔹 Date format conversion done.")

# Extract date parts (for time series)
if 'date' in df.columns:
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month

# Display processed sample
print("\n🔹 Processed data sample:\n", df.head())
