import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

# Load trained model (make sure you saved it as 'model.pkl')
model = pickle.load(open('model.pkl', 'rb'))

st.set_page_config(page_title="Diabetes Prediction", layout="centered")
st.title("🩺 Diabetes Prediction App")

st.write("Enter your health details below:")

# Input form
Pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, step=1)
Glucose = st.number_input("Glucose Level", min_value=0, max_value=300)
BloodPressure = st.number_input("Blood Pressure", min_value=0, max_value=200)
SkinThickness = st.number_input("Skin Thickness ", min_value=0, max_value=100)
Insulin = st.number_input("Insulin Level", min_value=0, max_value=900)
BMI = st.number_input("BMI", min_value=0.0, max_value=70.0)
DiabetesPedigreeFunction = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.5)
Age = st.number_input("Age", min_value=1, max_value=120)

# Predict button
if st.button("Predict"):
    # Input to model
    input_data = np.array([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, 
                            BMI, DiabetesPedigreeFunction, Age]])
    
    # Make prediction
    prediction = model.predict(input_data)

    # Show result
    if prediction[0] == 1:
        st.error("⚠️ The model predicts you might have Diabetes.")
    else:
        st.success("✅ The model predicts you are not diabetic.")

