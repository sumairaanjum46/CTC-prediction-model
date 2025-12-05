import streamlit as st
import numpy as np
import joblib

# Load the model
model = joblib.load("model.pkl")

st.title("Simple Linear Regression - Real Time Predictor")

# Input
x = st.number_input("Enter Years of Experience :", value=0.0)

# Predict button
if st.button("Predict"):
    prediction = model.predict([[x]])[0]
    st.success(f"Predicted CTC = {prediction:.2f}")
