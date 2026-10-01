import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("../xgb_model.pkl")

# Page title
st.title("Predictive Maintenance System")
st.write("Predict machine failure risk using operating conditions.")

# Machine inputs
machine_type = st.selectbox(
    "Machine Type",
    ["H", "L", "M"]
)

air_temp = st.number_input(
    "Air Temperature [K]",
    min_value=290.0,
    max_value=310.0,
    value=298.0
)

process_temp = st.number_input(
    "Process Temperature [K]",
    min_value=300.0,
    max_value=320.0,
    value=308.0
)

rotational_speed = st.number_input(
    "Rotational Speed [rpm]",
    min_value=1000,
    max_value=3000,
    value=1500
)

torque = st.number_input(
    "Torque [Nm]",
    min_value=0.0,
    max_value=80.0,
    value=40.0
)

tool_wear = st.number_input(
    "Tool Wear [min]",
    min_value=0,
    max_value=300,
    value=100
)

# Predict button
if st.button("Predict Failure Risk"):

    # Encode machine type exactly as during training
    type_L = 1 if machine_type == "L" else 0
    type_M = 1 if machine_type == "M" else 0

    # Create input in the same order used by XGBoost
    input_data = pd.DataFrame([{
        "air_temp": air_temp,
        "process_temp": process_temp,
        "rotational_speed": rotational_speed,
        "torque": torque,
        "tool_wear": tool_wear,
        "type_L": type_L,
        "type_M": type_M
    }])

    # Get failure probability
    probability = model.predict_proba(input_data)[0][1]

    # Determine risk level
    if probability < 0.10:
        risk = "Low"
    elif probability < 0.30:
        risk = "Medium"
    else:
        risk = "High"

    # Display results
    st.subheader("Prediction Result")

    st.metric(
        "Failure Probability",
        f"{probability * 100:.2f}%"
    )

    st.write(f"### Risk Level: {risk}")

    if risk == "Low":
        st.success("Machine is currently at low predicted failure risk.")

    elif risk == "Medium":
        st.warning("Machine shows moderate predicted failure risk. Consider inspection.")

    else:
        st.error("Machine shows high predicted failure risk. Inspection should be considered.")