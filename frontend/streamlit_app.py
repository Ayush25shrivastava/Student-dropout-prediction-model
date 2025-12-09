import streamlit as st
import requests
import pickle

API_URL = "http://localhost:9000/predict"

st.title("🎓 Student Dropout Prediction System")

st.info("Fill the student details below and click Predict")

feature_columns = pickle.load(open("../backend/feature_columns.pkl", "rb"))

invalid_cols = ["Dropout_Probability", "Predicted_Dropout", "Actual_Dropout", "special_needs_flag"]
feature_columns = [col for col in feature_columns if col not in invalid_cols]

inputs = {}

for col in feature_columns:
    value = st.number_input(
        label=col, 
        min_value=-999999.0, 
        max_value=999999.0, 
        value=0.0, 
        key=col
    )
    inputs[col] = value

if st.button("Predict Dropout Risk"):
    with st.spinner("Predicting... please wait"):
        response = requests.post(API_URL, json={"features": inputs})
        result = response.json()

        if "error" in result:
            st.error(result["error"])
        else:
            st.success(f"Prediction: {result['prediction']}")
            st.metric("Dropout Probability", f"{result['dropout_probability']:.2%}")
