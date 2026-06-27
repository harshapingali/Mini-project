"""
AI Crop Recommendation System
------------------------------
Deployment notes:
  Place these files in the SAME folder as this script before deploying:
    - crop_recommendation_model.pkl
    - scaler.pkl
    - label_encoder.pkl
    - Crop_recommendation.csv   (OPTIONAL - used to compute accurate ideal
                                  ranges per crop for the "why" explanations.
                                  If absent, built-in guideline ranges are
                                  used instead, and a sidebar note says so.)

  requirements.txt is provided alongside this file.
"""

import os
import numpy as np
import pandas as pd
import joblib
import streamlit as st

# ---------------------------------------------------------------------------
# Page configuration (must be the first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Crop Recommendation",
    page_icon="🌾",
    layout="wide",
)

MODEL_PATH = "crop_recommendation_model.pkl"
SCALER_PATH = "scaler.pkl"
ENCODER_PATH = "label_encoder.pkl"
DATA_PATH = "Crop_recommendation.csv"

FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]

PARAM_INFO = {
    "N": ("Nitrogen", "supports leafy growth and chlorophyll production"),
    "P": ("Phosphorus", "supports root development and flowering"),
    "K": ("Potassium", "improves disease resistance and fruit/grain quality"),
    "temperature": ("Temperature", "affects germination and metabolic rate"),
    "humidity": ("Humidity", "influences transpiration and disease pressure"),
    "ph": ("Soil pH", "affects how easily roots can absorb nutrients"),
    "rainfall": ("Rainfall", "determines how much water is available for growth"),
}

# Built-in fallback guideline ranges (approximate agronomic guidelines, used
# only when Crop_recommendation.csv is not bundled with the app). For best
# accuracy, ship the training CSV so ranges are computed from real data.
FALLBACK_RANGES = {
    "rice":        {"N": (60, 100), "P": (35, 60),  "K": (35, 45),  "temperature": (20, 27), "humidity": (75, 95), "ph": (5.0, 7.5), "rainfall": (180, 300)},
    "maize":       {"N": (60, 100), "P": (35, 60),  "K": (15, 25),  "temperature": (18, 27), "humidity": (55, 75), "ph": (5.5, 7.5), "rainfall": (60, 120)},
    "chickpea":    {"N": (20, 60),  "P": (55, 80),  "K": (75, 90),  "temperature": (17, 22), "humidity": (14, 20), "ph": (6.0, 8.0), "rainfall": (65, 90)},
    "kidneybeans": {"N": (15, 40),  "P": (55, 80),  "K": (15, 25),  "temperature": (15, 22), "humidity": (18, 22), "ph": (5.5, 6.0), "rainfall": (60, 110)},
    "pigeonpeas":  {"N": (15, 40),  "P": (55, 80),  "K": (15, 25),  "temperature": (18, 37), "humidity": (30, 70), "ph": (4.5, 7.0), "rainfall": (90, 200)},
    "mothbeans":   {"N": (15, 40),  "P": (35, 60),  "K": (15, 25),  "temperature": (24, 32), "humidity": (30, 60), "ph": (3.5, 10.0), "rainfall": (25, 65)},
    "mungbean":    {"N": (15, 40),  "P": (35, 60),  "K": (15, 25),  "temperature": (27, 32), "humidity": (75, 90), "ph": (6.0, 7.5), "rainfall": (40, 70)},
    "blackgram":   {"N": (15, 40),  "P": (55, 80),  "K": (15, 25),  "temperature": (25, 35), "humidity": (60, 70), "ph": (6.0, 7.5), "rainfall": (60, 75)},
    "lentil":      {"N": (15, 40),  "P": (55, 80),  "K": (15, 25),  "temperature": (18, 24), "humidity": (60, 70), "ph": (6.0, 7.0), "rainfall": (40, 55)},
    "pomegranate": {"N": (15, 40),  "P": (15, 40),  "K": (35, 45),  "temperature": (18, 25), "humidity": (85, 95), "ph": (6.0, 7.0), "rainfall": (105, 125)},
    "banana":      {"N": (90, 120), "P": (70, 95),  "K": (45, 55),  "temperature": (25, 30), "humidity": (75, 85), "ph": (5.5, 6.5), "rainfall": (95, 115)},
    "mango":       {"N": (15, 40),  "P": (15, 40),  "K": (25, 35),  "temperature": (27, 36), "humidity": (45, 55), "ph": (5.5, 7.0), "rainfall": (85, 105)},
    "grapes":      {"N": (15, 40),  "P": (115, 145),"K": (195, 205),"temperature": (15, 25), "humidity": (80, 85), "ph": (5.5, 6.5), "rainfall": (65, 75)},
    "watermelon":  {"N": (90, 110), "P": (15, 40),  "K": (45, 55),  "temperature": (24, 27), "humidity": (80, 90), "ph": (6.0, 6.8), "rainfall": (40, 55)},
    "muskmelon":   {"N": (90, 110), "P": (15, 40),  "K": (45, 55),  "temperature": (27, 30), "humidity": (90, 95), "ph": (6.0, 6.8), "rainfall": (20, 30)},
    "apple":       {"N": (15, 40),  "P": (115, 145),"K": (195, 205),"temperature": (20, 24), "humidity": (90, 95), "ph": (5.5, 6.5), "rainfall": (100, 120)},
    "orange":      {"N": (15, 40),  "P": (5, 30),   "K": (5, 15),   "temperature": (12, 35), "humidity": (90, 95), "ph": (6.0, 7.5), "rainfall": (100, 120)},
    "papaya":      {"N": (40, 70),  "P": (50, 75),  "K": (45, 55),  "temperature": (23, 44), "humidity": (90, 95), "ph": (6.5, 7.0), "rainfall": (40, 250)},
    "coconut":     {"N": (15, 40),  "P": (5, 30),   "K": (25, 35),  "temperature": (25, 30), "humidity": (90, 100),"ph": (5.5, 6.5), "rainfall": (140, 220)},
    "cotton":      {"N": (100, 140),"P": (35, 60),  "K": (15, 25),  "temperature": (22, 27), "humidity": (75, 85), "ph": (5.5, 7.0), "rainfall": (60, 90)},
    "jute":        {"N": (60, 100), "P": (35, 60),  "K": (35, 45),  "temperature": (23, 27), "humidity": (70, 90), "ph": (6.0, 7.5), "rainfall": (160, 200)},
    "coffee":      {"N": (80, 120), "P": (15, 40),  "K": (25, 35),  "temperature": (23, 28), "humidity": (50, 70), "ph": (6.0, 7.0), "rainfall": (150, 200)},
}


# ---------------------------------------------------------------------------
# Cached loaders - run once per server process, not on every click
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading model artifacts...")
def load_artifacts():
    missing = [p for p in (MODEL_PATH, SCALER_PATH, ENCODER_PATH) if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            "Missing required file(s): " + ", ".join(missing) +
            ". Make sure they are committed in the same folder as app.py before deploying."
        )
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    encoder = joblib.load(ENCODER_PATH)
    return model, scaler, encoder


@st.cache_data(show_spinner=False)
def load_ideal_ranges():
    """
    Build ideal (low, high) ranges per crop & feature.
    Prefers the 25th-75th percentile computed from a bundled training CSV
    for accuracy; falls back to built-in agronomic guideline ranges if the
    CSV isn't shipped with the app.
    """
    if os.path.exists(DATA_PATH):
        try:
            df = pd.read_csv(DATA_PATH)
            label_col = "label" if "label" in df.columns else df.columns[-1]
            ranges = {}
            for crop, group in df.groupby(label_col):
                ranges[str(crop).strip().lower()] = {
                    feat: (group[feat].quantile(0.25), group[feat].quantile(0.75))
                    for feat in FEATURES if feat in group.columns
                }
            return ranges, "dataset"
        except Exception:
            pass  # fall through to built-in ranges
    return FALLBACK_RANGES, "fallback"


def explain_recommendation(crop, values, ranges_table):
    """Return a list of {'ok': bool, 'text': str} explaining the fit for one crop."""
    crop_title = str(crop).title()
    crop_key = str(crop).strip().lower()
    crop_ranges = ranges_table.get(crop_key)

    if not crop_ranges:
        return [{
            "ok": True,
            "text": f"{crop_title} matches the overall pattern the model learned "
                    f"from similar input combinations in training.",
        }]

    reasons = []
    for feat in FEATURES:
        if feat not in crop_ranges:
            continue
        low, high = float(crop_ranges[feat][0]), float(crop_ranges[feat][1])
        value = values[feat]
        name, desc = PARAM_INFO[feat]

        if low <= value <= high:
            text = (f"**{name}** ({value:g}) is within {crop_title}'s ideal range "
                    f"({low:.1f}–{high:.1f}) — {desc}.")
            reasons.append({"ok": True, "text": text})
        elif value < low:
            text = (f"**{name}** ({value:g}) is lower than {crop_title}'s ideal range "
                    f"({low:.1f}–{high:.1f}). Since {name} {desc}, increasing it may help.")
            reasons.append({"ok": False, "text": text})
        else:
            text = (f"**{name}** ({value:g}) is higher than {crop_title}'s ideal range "
                    f"({low:.1f}–{high:.1f}).")
            reasons.append({"ok": False, "text": text})
    return reasons


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.title("🌾 AI Crop Recommendation System")
st.caption("Enter soil nutrients and climate conditions to get the best-suited crop, with the reasoning behind it.")

try:
    model, scaler, encoder = load_artifacts()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()
except Exception as e:
    st.error(f"Could not load model artifacts: {e}")
    st.stop()

ranges_table, ranges_source = load_ideal_ranges()

with st.sidebar:
    st.header("About")
    st.write(
        "Predicts the most suitable crop for given soil nutrients and climate "
        "conditions using a trained ML model, then explains *why* that crop "
        "fits by comparing your inputs against ideal growing ranges."
    )
    if ranges_source == "fallback":
        st.info(
            f"ℹ️ Explanations use built-in approximate guideline ranges. "
            f"Add `{DATA_PATH}` next to app.py to base them on your actual "
            f"training data instead."
        )
    else:
        st.success("✅ Explanations are based on your training dataset's per-crop statistics.")
    st.caption("This tool is informational and does not replace local agronomic advice.")

col1, col2 = st.columns(2)
with col1:
    N = st.number_input("Nitrogen - N (kg/ha)", min_value=0.0, max_value=200.0, value=50.0, step=1.0)
    P = st.number_input("Phosphorus - P (kg/ha)", min_value=0.0, max_value=200.0, value=50.0, step=1.0)
    K = st.number_input("Potassium - K (kg/ha)", min_value=0.0, max_value=250.0, value=50.0, step=1.0)
    temperature = st.number_input("Temperature (°C)", min_value=-10.0, max_value=55.0, value=25.0, step=0.5)
with col2:
    humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=60.0, step=1.0)
    ph = st.number_input("Soil pH", min_value=0.0, max_value=14.0, value=6.5, step=0.1)
    rainfall = st.number_input("Rainfall (mm)", min_value=0.0, max_value=500.0, value=100.0, step=1.0)

if st.button("🔍 Predict Crop", type="primary"):
    values = {"N": N, "P": P, "K": K, "temperature": temperature,
              "humidity": humidity, "ph": ph, "rainfall": rainfall}

    try:
        with st.spinner("Analyzing soil & climate conditions..."):
            data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
            scaled = scaler.transform(data)
            probs = model.predict_proba(scaled)[0]
            top_idx = np.argsort(probs)[::-1][:3]
            crops = encoder.inverse_transform(top_idx)
            scores = probs[top_idx] * 100
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.stop()

    # Soil health score (kept from the original logic)
    soil_health = (
        (N / 140) * 25 +
        (P / 145) * 20 +
        (K / 205) * 20 +
        (1 - abs(ph - 7) / 7) * 15 +
        (humidity / 100) * 10 +
        (rainfall / 300) * 10
    )
    soil_health = int(min(max(round(soil_health), 0), 100))

    m1, m2 = st.columns([1, 2])
    with m1:
        st.metric("Soil Health Score", f"{soil_health}/100")
    with m2:
        st.progress(soil_health / 100)

    st.subheader("🏆 Top Suitable Crops")
    medals = ["🥇", "🥈", "🥉"]
    cols = st.columns(3)
    for i, c in enumerate(cols):
        with c:
            st.metric(f"{medals[i]} {str(crops[i]).title()}", f"{scores[i]:.2f}%")

    top_crop = str(crops[0]).title()
    st.subheader(f"✅ Recommended Crop: {top_crop}")
    st.success(f"**{top_crop}** is the best match for the conditions you entered.")

    st.subheader(f"🌱 Why {top_crop}?")
    reasons = explain_recommendation(crops[0], values, ranges_table)
    matched = [r for r in reasons if r["ok"]]
    mismatched = [r for r in reasons if not r["ok"]]

    if matched:
        st.markdown("**Conditions that support this recommendation:**")
        for r in matched:
            st.write(f"✅ {r['text']}")

    if mismatched:
        st.markdown("**Worth keeping an eye on:**")
        for r in mismatched:
            st.write(f"⚠️ {r['text']}")

    with st.expander(f"See reasoning for runner-up crops ({str(crops[1]).title()}, {str(crops[2]).title()})"):
        for alt_crop in crops[1:]:
            st.markdown(f"**{str(alt_crop).title()}**")
            for r in explain_recommendation(alt_crop, values, ranges_table):
                icon = "✅" if r["ok"] else "⚠️"
                st.write(f"{icon} {r['text']}")