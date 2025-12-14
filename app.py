import streamlit as st
import numpy as np
import pickle

# -------------------------------
# Load saved objects
# -------------------------------

with open("poly.pkl", "rb") as f:
    poly = pickle.load(f)


with open("polynomial_regression_model.pkl", "rb") as f:
    model = pickle.load(f)

try:
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
except:
    scaler = None


# -------------------------------
# Streamlit UI
# -------------------------------

st.title("Edunet Foundation – Health Insurance Cost Prediction")
st.write("Enter customer details to estimate the insurance cost")

age = st.number_input("Age", min_value=0, max_value=100, value=30)
bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0)
#bloodpressure = st.number_input("Blood Pressure", min_value=50, max_value=200, value=120)
children = st.number_input("Number of Children", min_value=0, max_value=5, value=0)

gender = st.selectbox("Gender", ["Female", "Male"])
#diabetic = st.selectbox("Diabetic", ["No", "Yes"])
smoker = st.selectbox("Smoker", ["No", "Yes"])
region = st.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"])


# -------------------------------
# Manual Encoding
# -------------------------------

gender_male = 1 if gender == "Male" else 0
#diabetic_yes = 1 if diabetic == "Yes" else 0
smoker_yes = 1 if smoker == "Yes" else 0

region_northwest = 1 if region == "northwest" else 0
region_southeast = 1 if region == "southeast" else 0
region_southwest = 1 if region == "southwest" else 0
# northeast is the reference category


# -------------------------------
# Prepare input data
# -------------------------------

input_data = np.array([[
    age,
    bmi,
    #bloodpressure,
    children,
    gender_male,
    #diabetic_yes,
    smoker_yes,
    region_northwest,
    region_southeast,
    region_southwest
]])

# Apply polynomial features
input_data = poly.transform(input_data)

# Apply scaling if scaler exists
if scaler is not None:
    input_data = scaler.transform(input_data)


# -------------------------------
# Prediction
# -------------------------------

if st.button("Predict Insurance Cost"):
    prediction = model.predict(input_data)
    st.success(f"Estimated Insurance Cost: ₹ {prediction[0]:,.2f}")