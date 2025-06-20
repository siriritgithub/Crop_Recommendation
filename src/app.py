import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os
import requests

API_KEY = "3a79b73910e3f8025ddd608973a82973"

def get_weather(city):
    try:
        url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"
        response = requests.get(url)
        data = response.json()

        if data["cod"] != 200:
            return None, None

        temperature = data["main"]["temp"]
        rainfall = data.get("rain", {}).get("1h") or data.get("rain", {}).get("3h", 0)
        return temperature, rainfall
    except Exception:
        return None, None

# Load models
@st.cache_resource
def load_models():
    crop_model = joblib.load("models/crop_recommendation_model.pkl")
    price_model = joblib.load("models/price_prediction_model.pkl")
    fert_model = joblib.load("models/fertilizer_model.pkl")
    return crop_model, price_model, fert_model

crop_model, price_model, fert_model = load_models()

# Streamlit UI
st.set_page_config(page_title="Smart Agri Assistant", layout="wide")
st.title("🌾 Smart Agri Assistant")

tab1, tab2, tab3, tab4 = st.tabs([
    "🌱 Crop Recommendation",
    "📈 Price Prediction",
    "🧪 Fertilizer Suggestion",
    "💧 Irrigation Suggestion"
])

# 🌱 Crop Recommendation
with tab1:
    st.header("Crop Recommendation Based on Soil and Weather")
    city = st.text_input("Enter City Name for Weather Data", key="weather_city")
    soil_type = st.selectbox("Select Soil Type", ["Clay", "Sandy", "Loamy"], key="crop_soil")
    N = st.slider("Nitrogen (N)", 0, 140, 70, key="crop_N")
    P = st.slider("Phosphorus (P)", 0, 140, 40, key="crop_P")
    K = st.slider("Potassium (K)", 0, 140, 40, key="crop_K")
    humid_crop = st.slider("Humidity (%)", 0, 100, 50, key="crop_humid")
    ph_crop = st.slider("pH", 0.0, 14.0, 6.5, key="crop_ph")

    if st.button("🌱 Recommend Crop"):
        if city:
            with st.spinner("Fetching weather data..."):
                temp_crop, rainfall_crop = get_weather(city)
            if temp_crop is None:
                st.error("❌ Could not fetch weather data. Please check the city name.")
            else:
                crop_input = np.array([[N, P, K, temp_crop, humid_crop, ph_crop, rainfall_crop]])
                result = crop_model.predict(crop_input)
                st.success(f"🌾 Recommended Crop: **{result[0]}**")
                st.info(f"🌡 Temp: {temp_crop}°C | 🌧 Rainfall: {rainfall_crop} mm | 🧱 Soil: {soil_type}")
        else:
            st.warning("⚠️ Please enter a city name.")

# 📈 Price Prediction
with tab2:
    st.header("Crop Price Prediction")
    crop_name = st.selectbox("Select Crop", ["rice", "wheat", "maize", "cotton", "sugarcane"])
    market_demand = st.slider("Market Demand (0-100)", 0, 100, 50)
    supply = st.slider("Supply Level (0-100)", 0, 100, 50)

    if st.button("📈 Predict Price"):
        crop_idx = ["rice", "wheat", "maize", "cotton", "sugarcane"].index(crop_name)
        input_data = np.array([[crop_idx, market_demand, supply]])
        price = price_model.predict(input_data)
        st.success(f"📊 Predicted Price: ₹{price[0]:.2f} per quintal")

# 🧪 Fertilizer Suggestion
with tab3:
    st.header("Fertilizer Recommendation Based on Soil Nutrients")
    N_fert = st.slider("Nitrogen (N)", 0, 140, 70)
    P_fert = st.slider("Phosphorus (P)", 0, 140, 40)
    K_fert = st.slider("Potassium (K)", 0, 140, 40)

    if st.button("🧪 Suggest Fertilizer"):
        input_data = np.array([[N_fert, P_fert, K_fert]])
        fert_result = fert_model.predict(input_data)
        st.success(f"🌿 Recommended Fertilizer: **{fert_result[0]}**")

        fert_data = {
            "Nutrient": ["Nitrogen", "Phosphorus", "Potassium"],
            "Amount (kg/ha)": [N_fert, P_fert, K_fert]
        }
        df_fert = pd.DataFrame(fert_data)
        st.bar_chart(df_fert.set_index("Nutrient"))

# 💧 Irrigation Suggestion
with tab4:
    st.header("Irrigation Suggestion Based on Weather and Soil")
    city_irri = st.text_input("Enter City for Weather Data", key="irri_city")
    soil_type_irri = st.selectbox("Soil Type", ["Clay", "Sandy", "Loamy"], key="irri_soil")

    if st.button("💧 Get Irrigation Advice"):
        if city_irri:
            with st.spinner("Fetching weather data..."):
                temp_irri, rainfall_irri = get_weather(city_irri)
            if temp_irri is None:
                st.error("Could not fetch weather data.")
            else:
                if rainfall_irri > 200:
                    irrigation = "Low – Natural rainfall is sufficient."
                elif rainfall_irri < 50 and temp_irri > 35 and soil_type_irri == "Sandy":
                    irrigation = "High – Frequent watering needed due to heat and poor water retention."
                elif soil_type_irri == "Clay":
                    irrigation = "Moderate – Clay retains water longer. Avoid overwatering."
                elif soil_type_irri == "Loamy":
                    irrigation = "Moderate – Balanced watering recommended."
                else:
                    irrigation = "Moderate to High – Check moisture levels frequently."

                st.success(f"💧 Irrigation Advice: **{irrigation}**")
                st.info(f"🌡 Temp: {temp_irri}°C | 🌧 Rainfall: {rainfall_irri} mm")
        else:
            st.warning("Please enter a city name.")
