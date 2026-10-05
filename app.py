"""
Smart Agri Assistant -- main Streamlit app.

Single entry point for the whole project (an earlier version had four
different, mostly-empty app files across app/ and src/; this is the one
canonical app now). Run with:

    streamlit run app.py
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import requests
import streamlit as st

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from translations import TRANSLATIONS, t
from govt_schemes import GOVT_SCHEMES, PRIVATE_SCHEMES, INDIAN_STATES_AND_UTS, CROP_SEASONS, get_schemes, get_crop_info
from fertilizer_recommender import get_fertilizer_recommendation, fertilizer_dict, to_per_acre
from chatbot import ask_chatbot

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"
FEEDBACK_PATH = DATA_DIR / "feedback_log.csv"

CROP_FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]


def get_secret(key: str) -> str:
    """st.secrets raises if secrets.toml doesn't exist at all (as opposed to
    just missing a key), which is exactly the state of a fresh checkout of
    this project -- only secrets.toml.example ships in the repo."""
    try:
        return st.secrets.get(key, "")
    except Exception:
        return ""

# ----------------------------------------------------------------------
# Page config + theme
# ----------------------------------------------------------------------
st.set_page_config(page_title="Smart Agri Assistant", page_icon="🌾", layout="wide")

st.markdown("""
<style>
    .stApp { background-color: #FAF7F0; }
    section[data-testid="stSidebar"] { background-color: #1B4332; }
    section[data-testid="stSidebar"] * { color: #F1F8E9 !important; }
    h1, h2, h3 { color: #1B4332; }
    div[data-testid="stMetric"] {
        background-color: #FFFFFF;
        border: 1px solid #D8E2DC;
        border-radius: 12px;
        padding: 14px 16px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    }
    .agri-card {
        background-color: #FFFFFF;
        border: 1px solid #D8E2DC;
        border-left: 5px solid #52796F;
        border-radius: 10px;
        padding: 16px 20px;
        margin-bottom: 12px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .agri-card h4 { margin: 0 0 6px 0; color: #1B4332; }
    .stButton>button {
        background-color: #2D6A4F;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 0.5em 1.4em;
        font-weight: 600;
    }
    .stButton>button:hover { background-color: #1B4332; color: white; }
    div[data-baseweb="tab-list"] { gap: 4px; }
</style>
""", unsafe_allow_html=True)


# ----------------------------------------------------------------------
# Cached loaders
# ----------------------------------------------------------------------
@st.cache_resource
def load_models():
    crop_model = joblib.load(MODEL_DIR / "crop_recommendation_model.pkl")
    price_model = joblib.load(MODEL_DIR / "price_prediction_model.pkl")
    commodity_encoder = joblib.load(MODEL_DIR / "commodity_encoder.pkl")
    return crop_model, price_model, commodity_encoder


@st.cache_data
def load_json(name):
    path = MODEL_DIR / name
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


crop_model, price_model, commodity_encoder = load_models()
crop_stats = load_json("crop_feature_stats.json")
crop_metrics = load_json("crop_model_metrics.json")
price_history = load_json("price_history.json")
price_metrics = load_json("price_model_metrics.json")

CROP_LIST = sorted(crop_model.classes_.tolist())
COMMODITY_LIST = sorted(commodity_encoder.classes_.tolist())


@st.cache_data(ttl=600, show_spinner=False)
def _fetch_weather_cached(city: str, api_key: str):
    """Cached per (city, api_key) for 10 minutes.

    This fixes two real issues from testing: (1) without caching, every
    slider tweak on the page reruns the whole script and re-hits the API
    on every rerun -- slow, and burns API quota; (2) it also caused a
    confusing bug where changing the city but leaving other inputs the
    same appeared to reuse an old result, because of ad-hoc reruns
    happening faster than the network call could return. Caching keyed on
    the exact city string makes results deterministic per city.
    """
    try:
        # Appending ",IN" measurably improves match rate for small Indian
        # towns/villages that share a name with places elsewhere, or that
        # OpenWeatherMap only indexes under a nearby administrative name.
        query = city if "," in city else f"{city},IN"
        url = f"http://api.openweathermap.org/data/2.5/weather?q={query}&appid={api_key}&units=metric"
        resp = requests.get(url, timeout=8)
        data = resp.json()
        if str(data.get("cod")) != "200":
            return None, None, "not_found"
        return data["main"]["temp"], data["main"]["humidity"], "ok"
    except requests.exceptions.RequestException:
        return None, None, "error"


def get_weather(city: str):
    """Live current temperature (C) and humidity (%) only.

    Deliberately NOT used for the model's 'rainfall' feature: the crop
    dataset's rainfall values are cumulative seasonal totals (20-300mm),
    while a live weather API only reports rain over the last hour (almost
    always ~0mm). Feeding that in would silently push every prediction to
    the edge of the training distribution, so rainfall stays a manual
    input everywhere in this app instead.
    """
    api_key = get_secret("OPENWEATHER_API_KEY")
    if not api_key:
        return None, None, "no_key"
    return _fetch_weather_cached(city.strip(), api_key)


def crop_reason(crop: str, inputs: dict) -> str:
    """Plain-language explanation comparing the user's inputs to the
    training-data range typical of this crop (no SHAP needed at this scale)."""
    stats = crop_stats.get(crop)
    if not stats:
        return ""
    notes = []
    for feat, val in inputs.items():
        s = stats.get(feat)
        if not s:
            continue
        mean, std = s["mean"], s["std"] or 1
        z = (val - mean) / std
        if abs(z) < 0.75:
            notes.append(f"**{feat}** ({val}) is close to typical for {crop} (~{mean})")
    if not notes:
        return f"{crop} is still the model's best statistical match for these inputs overall."
    return "; ".join(notes[:2]) + "."


def selectbox_with_other(label: str, options: list, key: str, other_placeholder: str = "Please specify"):
    """A selectbox where the last option is always 'Other (not listed)'.
    Picking it reveals a free-text field so the user is never forced into
    a wrong choice just because their option isn't in the list."""
    full_options = list(options) + ["Other (not listed)"]
    choice = st.selectbox(label, full_options, key=key)
    if choice == "Other (not listed)":
        custom = st.text_input(other_placeholder, key=f"{key}_other")
        return custom if custom else choice
    return choice


# ----------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🌾 Smart Agri Assistant")
    lang = st.selectbox("🌐 Language / भाषा", list(TRANSLATIONS.keys()), key="lang")
    st.markdown("---")
    st.caption(t("app_subtitle", lang))
    st.markdown("---")
    with st.expander("ℹ️ About this project"):
        st.write(
            "A data-science portfolio project combining a crop recommendation model, "
            "a real-data price predictor, rule-based agronomy guidance, and a small "
            "retrieval-grounded farming chatbot."
        )
    with st.expander("📊 Model performance"):
        if crop_metrics:
            st.write(f"**Crop model:** {crop_metrics['test_accuracy']*100:.1f}% test accuracy "
                      f"({crop_metrics['n_classes']} crops, {crop_metrics['n_samples']} samples)")
        if price_metrics:
            st.write(f"**Price model:** R² = {price_metrics['r2']:.2f}, "
                      f"MAE ≈ ₹{price_metrics['mae']:.0f}/quintal "
                      f"({price_metrics['n_samples']} historical records)")

st.title(t("app_title", lang))
st.caption(t("app_subtitle", lang))

tabs = st.tabs([
    t("tab_crop", lang), t("tab_price", lang), t("tab_profit", lang),
    t("tab_fertilizer", lang), t("tab_irrigation", lang),
    t("tab_schemes", lang), t("tab_chatbot", lang), t("tab_feedback", lang),
])
tab_crop, tab_price, tab_profit, tab_fert, tab_irri, tab_schemes, tab_chat, tab_feedback = tabs

# ----------------------------------------------------------------------
# TAB: Crop Recommendation
# ----------------------------------------------------------------------
with tab_crop:
    st.subheader(t("tab_crop", lang))
    col1, col2 = st.columns([1, 1])

    with col1:
        city = st.text_input(t("city_label", lang), key="crop_city",
                              placeholder="e.g. Bhubaneswar (optional)")
        use_live = False
        if city:
            temp_live, hum_live, status = get_weather(city)
            if status == "ok":
                st.success(f"Live weather for {city}: {temp_live}°C, {hum_live}% humidity")
                use_live = True
            elif status == "no_key":
                st.info("Add OPENWEATHER_API_KEY in .streamlit/secrets.toml to fetch live weather.")
            else:
                st.warning("Couldn't fetch weather for that city — using manual values below.")

        soil_type = selectbox_with_other(
            t("soil_label", lang), ["Clay", "Sandy", "Loamy", "Black (Regur)", "Red", "Alluvial"],
            key="crop_soil"
        )
        st.caption("Shown for your reference alongside the result — not fed into the model "
                   "(the source dataset's soil-type column was ~99.8% missing).")

        N = st.slider(t("nitrogen", lang), 0, 140, 70)
        P = st.slider(t("phosphorus", lang), 0, 140, 40)
        K = st.slider(t("potassium", lang), 0, 140, 40)
        temperature = st.slider(t("temperature", lang) + " (°C)", 0.0, 45.0,
                                 float(temp_live) if use_live else 25.0)
        humidity = st.slider(t("humidity", lang), 0, 100,
                              int(hum_live) if use_live else 60)
        ph = st.slider(t("ph", lang), 0.0, 14.0, 6.5)
        rainfall = st.slider(t("rainfall", lang) + " (mm, seasonal total)", 0, 300, 100)

    with col2:
        st.markdown("#### How this works")
        st.write(
            "A Random Forest model trained on 2,200 real soil/climate records "
            "(22 crops) predicts the top 3 best-matching crops for your inputs, "
            "with a plain-language reason for each."
        )
        st.caption("Note: the source dataset's 'soil type' column was ~99.8% missing, "
                   "so soil type isn't used as a prediction input in this app.")

        if st.button(t("recommend_btn", lang), use_container_width=True):
            inputs = {"N": N, "P": P, "K": K, "temperature": temperature,
                      "humidity": humidity, "ph": ph, "rainfall": rainfall}
            X = pd.DataFrame([[inputs[f] for f in CROP_FEATURES]], columns=CROP_FEATURES)
            proba = crop_model.predict_proba(X)[0]
            top_idx = np.argsort(proba)[::-1][:3]

            st.markdown(f"##### {t('top3_crops', lang)}")
            for rank, idx in enumerate(top_idx, start=1):
                crop = crop_model.classes_[idx]
                conf = proba[idx] * 100
                reason = crop_reason(crop, inputs)
                season_info = get_crop_info(crop)
                st.markdown(f"""
                <div class="agri-card">
                    <h4>#{rank} {crop.title()} — {conf:.1f}% match</h4>
                    <p style="margin:0;color:#444;">{reason}</p>
                    <p style="margin:6px 0 0 0;color:#52796F;">
                        📅 <b>{t('season', lang)}:</b> {season_info['season']} — {season_info['reason']}
                    </p>
                </div>
                """, unsafe_allow_html=True)

# ----------------------------------------------------------------------
# TAB: Price Prediction
# ----------------------------------------------------------------------
with tab_price:
    st.subheader(t("tab_price", lang))
    st.caption("Trained on real historical monthly mandi prices "
               f"({price_metrics.get('n_samples', '?')} records, "
               f"{len(COMMODITY_LIST)} commodities).")
    st.info("📍 **This is a national average, not state/district-specific.** "
            "The source data (Agmarknet-style national series) has no real state-level "
            "breakdown, so rather than fake per-state numbers, this predicts the all-India "
            "average modal price only.")

    col1, col2 = st.columns([1, 1])
    with col1:
        commodity = st.selectbox(t("crop_select", lang), COMMODITY_LIST, key="price_commodity")
        month_num = st.selectbox(
            t("month_select", lang), list(range(1, 13)),
            format_func=lambda m: pd.Timestamp(2024, m, 1).strftime("%B"),
            help="The month you want the price estimate FOR — e.g. pick March 2027 to see "
                 "the model's predicted average price for that specific month."
        )
        year = st.number_input(
            "Year", min_value=2024, max_value=2027, value=2026,
            help="The year you want the price estimate for, together with the month above."
        )
        unit = st.radio("Show price per", ["Quintal", "Kg", "Ton"], horizontal=True, key="price_unit")

        if st.button(t("predict_price_btn", lang), use_container_width=True):
            enc = commodity_encoder.transform([commodity])[0]
            price_X = pd.DataFrame([[enc, month_num, year]],
                                    columns=["commodity_encoded", "month_num", "year"])
            pred_per_quintal = price_model.predict(price_X)[0]
            st.session_state["price_pred"] = pred_per_quintal
            st.session_state["price_commodity_used"] = commodity

            unit_factor = {"Quintal": 1, "Kg": 1 / 100, "Ton": 10}
            unit_label = {"Quintal": t("unit_quintal", lang), "Kg": t("unit_kg", lang), "Ton": t("unit_ton", lang)}
            display_price = pred_per_quintal * unit_factor[unit]

            hist = price_history.get(commodity, [])
            trend_note = ""
            if hist:
                last_known = hist[-1]["avg_modal_price"]
                pct_change = (pred_per_quintal - last_known) / last_known * 100
                direction = "📈 increase" if pct_change > 1 else ("📉 decrease" if pct_change < -1 else "➡️ stay roughly flat")
                trend_note = (f"That's a {direction} of {abs(pct_change):.1f}% versus the most "
                               f"recently recorded price (₹{last_known:,.0f}/quintal, {hist[-1]['month']}).")

            st.metric(t("price_result", lang), f"₹{display_price:,.0f} / {unit_label[unit]}")
            if trend_note:
                st.caption(trend_note)

    with col2:
        hist = price_history.get(commodity, [])
        if hist:
            hist_df = pd.DataFrame(hist)
            st.markdown(f"##### Recent price trend — {commodity}")
            st.line_chart(hist_df.set_index("month")["avg_modal_price"])
        else:
            st.info("No recent history available for this commodity.")

# ----------------------------------------------------------------------
# TAB: Profit Calculator
# ----------------------------------------------------------------------
with tab_profit:
    st.subheader(t("tab_profit", lang))
    col1, col2 = st.columns([1, 1])
    with col1:
        area = st.number_input(t("area", lang), min_value=0.1, value=1.0, step=0.5)
        seed_cost = st.number_input(t("seed_cost", lang), min_value=0, value=3000, step=100)
        fert_cost = st.number_input(t("fertilizer_cost", lang), min_value=0, value=4000, step=100)
        labor_cost = st.number_input(t("labor_cost", lang), min_value=0, value=6000, step=100)
        other_cost = st.number_input(t("other_cost", lang), min_value=0, value=2000, step=100)
        yield_qtl = st.number_input(t("expected_yield", lang), min_value=0.0, value=15.0, step=0.5)

        default_price = st.session_state.get("price_pred", 2000.0)
        selling_price = st.number_input(
            t("selling_price", lang), min_value=0.0, value=float(round(default_price, 0)), step=50.0
        )
        if "price_pred" in st.session_state:
            st.caption(f"Pre-filled from your Price Prediction tab result "
                       f"({st.session_state.get('price_commodity_used','')}).")

    with col2:
        if st.button(t("profit_btn", lang), use_container_width=True):
            total_cost = (seed_cost + fert_cost + labor_cost + other_cost) * area
            total_revenue = yield_qtl * area * selling_price
            profit = total_revenue - total_cost

            c1, c2, c3 = st.columns(3)
            c1.metric(t("total_cost", lang), f"₹{total_cost:,.0f}")
            c2.metric(t("total_revenue", lang), f"₹{total_revenue:,.0f}")
            c3.metric(t("profit", lang) if profit >= 0 else t("loss", lang),
                      f"₹{abs(profit):,.0f}", delta=f"{profit:,.0f}")

            chart_df = pd.DataFrame({"Amount (₹)": [total_cost, total_revenue]},
                                     index=["Total Cost", "Total Revenue"])
            st.bar_chart(chart_df)

# ----------------------------------------------------------------------
# TAB: Fertilizer & Pest Guidance
# ----------------------------------------------------------------------
with tab_fert:
    st.subheader(t("tab_fertilizer", lang))
    st.caption("Rule-based agronomy guidance (curated for 5 major crops, with a "
               "data-derived estimate for the rest) rather than a black-box classifier "
               "with no verifiable training data.")

    fert_crop = st.selectbox(t("crop_select", lang), CROP_LIST, key="fert_crop")
    unit_choice = st.radio(
        "Land unit", ["Acre (small farmers)", "Hectare"], horizontal=True, key="fert_unit",
        help="'kg/ha' below means kilograms per hectare (1 hectare ≈ 2.47 acres). "
             "Pick Acre to see the amount for a typical smallholding instead."
    )
    N_have = st.slider(t("nitrogen", lang) + " you currently have", 0, 200, 50, key="fert_N")
    P_have = st.slider(t("phosphorus", lang) + " you currently have", 0, 200, 30, key="fert_P")
    K_have = st.slider(t("potassium", lang) + " you currently have", 0, 200, 30, key="fert_K")

    if st.button(t("fertilizer_btn", lang), use_container_width=True):
        rec = get_fertilizer_recommendation(fert_crop)
        st.markdown(f"""
        <div class="agri-card">
            <h4>{fert_crop.title()}</h4>
            <p>{rec['recommendation']}</p>
            <p style="color:#666;"><em>{rec['note']}</em></p>
            <p><b>🗓 When to apply:</b> {rec.get('when_to_apply', 'Not available for this crop yet.')}</p>
            <p><b>🐛 Pest control:</b> {rec['pesticide']}</p>
            <p><b>⚠️ Precautions (for crop and farmer safety):</b> {rec.get('precautions', '')}</p>
        </div>
        """, unsafe_allow_html=True)

        if rec["N"] != "Unknown":
            use_acre = unit_choice.startswith("Acre")
            unit_label = "kg/acre" if use_acre else "kg/ha"
            rec_N = to_per_acre(rec["N"]) if use_acre else rec["N"]
            rec_P = to_per_acre(rec["P"]) if use_acre else rec["P"]
            rec_K = to_per_acre(rec["K"]) if use_acre else rec["K"]

            st.caption(f"Values below shown in **{unit_label}**. "
                       "\"You have\" values assume the same unit as what you entered above.")
            compare_df = pd.DataFrame({
                "You have": [N_have, P_have, K_have],
                f"Recommended ({unit_label})": [rec_N, rec_P, rec_K],
            }, index=["N", "P", "K"])
            st.bar_chart(compare_df)
            if rec.get("source") == "data_estimate":
                st.caption("⚠️ Recommended values here are dataset averages, not a curated "
                           "agronomy source — treat as a rough reference only.")

# ----------------------------------------------------------------------
# TAB: Irrigation
# ----------------------------------------------------------------------
with tab_irri:
    st.subheader(t("tab_irrigation", lang))
    city_irri = st.text_input(t("city_label", lang), key="irri_city")
    soil_type_irri = selectbox_with_other(
        t("soil_label", lang), ["Clay", "Sandy", "Loamy", "Black (Regur)", "Red", "Alluvial"],
        key="irri_soil"
    )
    rainfall_irri = st.slider(t("rainfall", lang) + " (mm, recent season)", 0, 300, 100, key="irri_rain")

    temp_irri = None
    if city_irri:
        temp_irri, hum_irri, status = get_weather(city_irri)
        if status == "ok":
            st.success(f"Live temperature for {city_irri}: {temp_irri}°C")
        elif status == "no_key":
            st.info("Add OPENWEATHER_API_KEY in secrets.toml for live temperature; using default below.")
        else:
            st.warning("Couldn't fetch weather — using default temperature.")
    if temp_irri is None:
        temp_irri = st.slider(t("temperature", lang) + " (°C)", 0.0, 45.0, 28.0, key="irri_temp_manual")

    if st.button(t("irrigation_btn", lang), use_container_width=True):
        if rainfall_irri > 200:
            advice = "Low — recent rainfall looks sufficient on its own."
        elif rainfall_irri < 50 and temp_irri > 35 and soil_type_irri == "Sandy":
            advice = "High — frequent watering needed (heat + low water retention in sandy soil)."
        elif soil_type_irri == "Clay":
            advice = "Moderate — clay retains water longer; avoid overwatering."
        elif soil_type_irri == "Loamy":
            advice = "Moderate — balanced watering recommended."
        else:
            advice = "Moderate to High — check soil moisture frequently."

        st.markdown(f"""
        <div class="agri-card">
            <h4>💧 {advice.split(' — ')[0]}</h4>
            <p>{advice}</p>
            <p style="color:#666;">🌡 {temp_irri}°C · 🌧 {rainfall_irri}mm recent rainfall · 🧱 {soil_type_irri} soil</p>
        </div>
        """, unsafe_allow_html=True)

# ----------------------------------------------------------------------
# TAB: Government Schemes
# ----------------------------------------------------------------------
with tab_schemes:
    st.subheader(t("tab_schemes", lang))
    st.caption("Curated reference data (not a live government feed) — covers national "
               "schemes plus state-specific ones for the states listed below, and a few "
               "real private-sector farmer support programs.")

    # Full state/UT list, independent of which ones have curated entries --
    # a state with no specific entries still falls back to national schemes.
    all_locations = ["All States"] + INDIAN_STATES_AND_UTS
    state = st.selectbox(t("schemes_location", lang), all_locations, key="schemes_state")

    crop_options = sorted({c for crops in GOVT_SCHEMES.values() for c in crops if c != "All Crops"})
    scheme_crop = selectbox_with_other(
        t("schemes_crop", lang), ["All Crops"] + crop_options, key="schemes_crop",
        other_placeholder="Type your crop (national schemes will still be shown)"
    )

    include_private = st.checkbox("Also show private-sector / cooperative programs", value=True,
                                   key="schemes_private")

    if st.button(t("find_schemes_btn", lang), use_container_width=True):
        # A crop typed into "Other" won't match any curated crop-specific
        # entry -- that's fine, it still gets the general/national schemes.
        results = get_schemes(state, scheme_crop, include_private=include_private)
        has_state_specific = bool(GOVT_SCHEMES.get(state, {}))
        if not has_state_specific and state != "All States":
            st.info(f"No state-specific entries curated for {state} yet — showing national "
                    "schemes that apply everywhere, including here.")

        for s in results:
            badge_color = "#8D5A97" if s.get("category") == "Private" else "#52796F"
            st.markdown(f"""
            <div class="agri-card">
                <h4>{s['name']} <span style="font-size:0.75em;color:{badge_color};">({s['type']})</span></h4>
                <p><b>Benefit:</b> {s['benefit']}</p>
                <p><b>Eligibility:</b> {s['eligibility']}</p>
                <p><b>Apply:</b> {s['apply']}</p>
            </div>
            """, unsafe_allow_html=True)
        if not results:
            st.warning("No schemes found — try 'All Crops' or check the private-sector box above.")

# ----------------------------------------------------------------------
# TAB: AI Chatbot (retrieval-grounded)
# ----------------------------------------------------------------------
with tab_chat:
    st.subheader(t("tab_chatbot", lang))
    st.caption("Answers general farming questions are answered from the model's own "
               "knowledge. Questions about schemes or fertilizer dosage are grounded "
               "in this app's own data via a lightweight retrieval step first.")

    api_key = get_secret("ANTHROPIC_API_KEY")
    if not api_key:
        st.info("No ANTHROPIC_API_KEY set in .streamlit/secrets.toml — running on a small "
                "built-in keyword responder instead (works, but far less capable than the "
                "full AI). Add a key to unlock real conversational answers grounded in "
                "this app's scheme/fertilizer data.")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for role, msg in st.session_state.chat_history:
        with st.chat_message(role):
            st.write(msg)

    question = st.chat_input(t("chatbot_input", lang))
    if question:
        st.session_state.chat_history.append(("user", question))
        with st.chat_message("user"):
            st.write(question)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer = ask_chatbot(question, api_key)
            st.write(answer)
        st.session_state.chat_history.append(("assistant", answer))

# ----------------------------------------------------------------------
# TAB: Feedback
# ----------------------------------------------------------------------
with tab_feedback:
    st.subheader(t("tab_feedback", lang))
    rating = st.slider(t("feedback_rating", lang), 1, 5, 4)
    comment = st.text_area(t("feedback_comment", lang))

    if st.button(t("submit_feedback", lang), use_container_width=True):
        row = pd.DataFrame([{"rating": rating, "comment": comment,
                              "timestamp": pd.Timestamp.now().isoformat()}])
        if FEEDBACK_PATH.exists():
            row.to_csv(FEEDBACK_PATH, mode="a", header=False, index=False)
        else:
            row.to_csv(FEEDBACK_PATH, index=False)
        st.success("Thanks for the feedback!")

    if FEEDBACK_PATH.exists():
        fb_df = pd.read_csv(FEEDBACK_PATH)
        st.markdown("##### Feedback so far")
        st.bar_chart(fb_df["rating"].value_counts().sort_index())
