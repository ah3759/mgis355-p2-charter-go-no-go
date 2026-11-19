import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Charter trip go/no-go", page_icon="??")

NOTICE = "Synthetic data: this shows the pipeline works, not that real fish behave this way."
BASE = Path(__file__).parent


def safe_float(value, default=0.0):
    try:
        value = float(value)
        return default if pd.isna(value) else value
    except (TypeError, ValueError):
        return default


@st.cache_resource
def load_artifacts():
    return joblib.load(BASE / "model.joblib"), json.loads((BASE / "model_meta.json").read_text(encoding="utf-8"))


@st.cache_data
def load_test_rows():
    data = pd.read_csv(BASE / "data" / "fishing_trips.csv")
    return data[data["season"] == 3].copy()


def cost_counts(y, probabilities, threshold, run_cost, reschedule_cost):
    predicted = probabilities >= threshold
    actual = y.astype(int)
    fp = int(((predicted == 1) & (actual == 0)).sum())
    fn = int(((predicted == 0) & (actual == 1)).sum())
    total = fp * run_cost + fn * reschedule_cost
    return fp, fn, total


model, meta = load_artifacts()
test = load_test_rows()
features = meta["features"]
threshold = safe_float(meta.get("threshold"), 0.5)
run_cost = safe_float(meta["costs"].get("run_unproductive"))
reschedule_cost = safe_float(meta["costs"].get("reschedule_productive"))

st.warning(NOTICE)
st.title("Charter trip go/no-go")
st.write("For a Carolinas charter captain deciding the evening before whether to run a booked trip; it supports the choice between running as planned and offering a reschedule.")

predict_tab, quality_tab, limits_tab = st.tabs(["Predict", "How good is this", "Limits"])

with predict_tab:
    st.header("Prediction for one booked trip")
    trip_type = st.selectbox("Trip type", meta["allowed_categories"]["trip_type"])
    allowed_months = meta["allowed_months_by_trip_type"][trip_type]
    month = st.selectbox("Month", allowed_months, format_func=lambda m: pd.Timestamp(2024, int(m), 1).strftime("%B"))
    species = st.selectbox("Species targeted", meta["allowed_species_by_trip_type"][trip_type])
    tide = st.selectbox("Tide stage at start", meta["allowed_categories"]["tide_stage_start"])
    ranges = meta["training_numeric_ranges"]
    party = st.slider("Party size", int(ranges["party_size"]["min"]), int(ranges["party_size"]["max"]), 4)
    moon = st.slider("Moon phase", safe_float(ranges["moon_phase"]["min"]), safe_float(ranges["moon_phase"]["max"]), 0.5, 0.001)
    days_front = st.slider("Days since front", int(ranges["days_since_front"]["min"]), int(ranges["days_since_front"]["max"]), 3)
    wind = st.slider("Forecast wind (mph)", safe_float(ranges["fcst_wind_mph"]["min"]), safe_float(ranges["fcst_wind_mph"]["max"]), 10.0, 0.1)
    wave = st.slider("Forecast wave height (ft)", safe_float(ranges["fcst_wave_height_ft"]["min"]), safe_float(ranges["fcst_wave_height_ft"]["max"]), 2.0, 0.1)
    air = st.slider("Forecast air temperature (F)", safe_float(ranges["fcst_air_temp_f"]["min"]), safe_float(ranges["fcst_air_temp_f"]["max"]), 75.0, 0.1)
    water = st.slider("Water temperature (F)", safe_float(ranges["water_temp_f"]["min"]), safe_float(ranges["water_temp_f"]["max"]), 70.0, 0.1)
    trailing = st.slider("Trailing 3-day catch rate", safe_float(ranges["trailing_3day_catch_rate"]["min"]), safe_float(ranges["trailing_3day_catch_rate"]["max"]), 0.55, 0.001)
    row = pd.DataFrame([{
        "trip_type": trip_type, "trip_hours": 4 if trip_type == "inshore" else 8, "month": month, "species_targeted": species, "party_size": party, "tide_stage_start": tide, "moon_phase": moon, "days_since_front": days_front, "fcst_wind_mph": wind, "fcst_wave_height_ft": wave, "fcst_air_temp_f": air, "water_temp_f": water, "trailing_3day_catch_rate": trailing
    }], columns=features)
    probability = safe_float(model.predict_proba(row)[:, 1][0])
    st.metric("Probability of a productive trip", f"{probability:.1%}")
    st.success("Run as planned" if probability >= threshold else "Offer a reschedule")
    st.caption(f"The decision threshold is {threshold:.1%}, computed from the two stated mistake costs.")

with quality_tab:
    st.header("Season 3 test performance")
    selected_threshold = st.slider("Decision threshold", 0.05, 0.95, float(threshold), 0.01)
    probabilities = model.predict_proba(test[features])[:, 1]
    fp, fn, total = cost_counts(test["productive"].to_numpy(), probabilities, selected_threshold, run_cost, reschedule_cost)
    tn = int(((probabilities < selected_threshold) & (test["productive"] == 0)).sum())
    tp = int(((probabilities >= selected_threshold) & (test["productive"] == 1)).sum())
    st.write(f"Confusion matrix: true negatives {tn}, false positives {fp}, false negatives {fn}, true positives {tp}.")
    st.write(f"Model: ${total:,.0f} total, ${total / len(test):,.2f} per trip.")
    st.write(f"Mistake costs: ${run_cost:,.0f} for running an unproductive trip; ${reschedule_cost:,.0f} for rescheduling a productive trip.")
    for name, result in meta["baselines"].items():
        st.write(f"{name.replace('_', ' ').title()}: ${safe_float(result.get('total_cost')):,.0f} total, ${safe_float(result.get('cost_per_trip')):,.2f} per trip.")
    always_running = safe_float(meta["baselines"]["majority"].get("total_cost"))
    avoided = always_running - total
    direction = "avoided" if avoided >= 0 else "added"
    st.info(f"At a threshold of {selected_threshold:.0%}, the model would have {direction} ${abs(avoided):,.0f} compared with always running the trip.")

with limits_tab:
    st.header("Limits and safe use")
    weak = meta["weakest_segment"]
    st.write(f"Weakest measured segment: {weak['dimension']} = {weak['value']}, AUC {safe_float(weak.get('auc')):.3f}.")
    st.write("Do not use this model to promise a catch, guarantee revenue, replace a captain's judgment, or generalize from synthetic data to real fish behavior.")
    st.write("Offshore trips are limited to May through October; the allowed input ranges come from seasons 1 and 2.")
    st.subheader("Input ranges")
    st.json(meta["training_numeric_ranges"])
    st.info(NOTICE)
