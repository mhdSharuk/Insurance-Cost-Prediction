import pickle
import numpy as np
import pandas as pd
import streamlit as st

import streamlit as st

model_file_name = 'insurance_cost_prediction_best_model_rf.pkl'

@st.cache_resource
def load_model():
  with open(model_file_name, 'rb') as model_file:
    model = pickle.load(model_file)

  return model

model = load_model()

st.set_page_config(
    page_title = 'Insurance Cost Prediction',
    layout = 'wide',
    initial_sidebar_state = 'expanded'
)

st.title('Insurace Cost Prediction')
st.markdown('---')

st.markdown("""
### Welcome to the Insurance Premium Calculator!
This application uses machine learning to estimate your insurance premium based on your personal health information.
Please fill in the form below to get your estimated premium.
""")

col1, col2 = st.columns(2)

with col1:
    st.header("Personal Information")
    
    # Age input
    age = st.slider("Age", min_value=18, max_value=100, value=30, help="Your current age")
    
    # Height input (in cm)
    height = st.number_input("Height (cm)", min_value=100, max_value=250, value=170, help="Your height in centimeters")
    
    # Weight input (in kg)
    weight = st.number_input("Weight (kg)", min_value=30, max_value=200, value=70, help="Your weight in kilograms")
    
    # Number of major surgeries
    surgeries = st.selectbox("Number of Major Surgeries", options=[0, 1, 2, 3, 4, 5], help="Number of major surgeries you've had")

with col2:
    st.header("Health Conditions")
    
    # Diabetes
    diabetes = st.selectbox("Do you have Diabetes?", options=["No", "Yes"])
    diabetes_val = 1 if diabetes == "Yes" else 0
    
    # Blood pressure problems
    bp_problems = st.selectbox("Do you have Blood Pressure Problems?", options=["No", "Yes"])
    bp_val = 1 if bp_problems == "Yes" else 0
    
    # Any transplants
    transplants = st.selectbox("Have you had Any Transplants?", options=["No", "Yes"])
    transplants_val = 1 if transplants == "Yes" else 0
    
    # Chronic diseases
    chronic_diseases = st.selectbox("Do you have Any Chronic Diseases?", options=["No", "Yes"])
    chronic_val = 1 if chronic_diseases == "Yes" else 0
    
    # Known allergies
    allergies = st.selectbox("Do you have Known Allergies?", options=["No", "Yes"])
    allergies_val = 1 if allergies == "Yes" else 0
    
    # History of cancer in family
    cancer_history = st.selectbox("History of Cancer in Family?", options=["No", "Yes"])
    cancer_val = 1 if cancer_history == "Yes" else 0

bmi = weight / ((height/100) ** 2)

st.markdown("---")
st.subheader("Calculated BMI")
st.metric("Body Mass Index (BMI)", f"{bmi:.2f}")

if bmi < 18.5:
    bmi_category = "Underweight"
    bmi_color = "blue"
elif 18.5 <= bmi < 25:
    bmi_category = "Normal weight"
    bmi_color = "green"
elif 25 <= bmi < 30:
    bmi_category = "Overweight"
    bmi_color = "orange"
else:
    bmi_category = "Obese"
    bmi_color = "red"

st.markdown(f"**Category:** :{bmi_color}[{bmi_category}]")

st.markdown("---")
if st.button("Calculate Premium", type="primary", use_container_width=True):
    input_data = np.array([[
        age,
        diabetes_val,
        bp_val,
        transplants_val,
        chronic_val,
        height,
        weight,
        allergies_val,
        cancer_val,
        surgeries
    ]])
    
    prediction = model.predict(input_data)[0]
    
    st.success("Premium Calculated Successfully!")
    
    result_col1, result_col2, result_col3 = st.columns(3)
    
    with result_col1:
        st.metric("Estimated Annual Premium", f"${prediction:,.2f}")
    
    with result_col2:
        monthly_premium = prediction / 12
        st.metric("Monthly Premium", f"${monthly_premium:,.2f}")
    
    with result_col3:
        daily_premium = prediction / 365
        st.metric("Daily Premium", f"${daily_premium:.2f}")
    
    st.markdown("---")
    st.subheader("Risk Assessment")
    
    risk_factors = []
    if diabetes_val:
        risk_factors.append("Diabetes")
    if bp_val:
        risk_factors.append("Blood Pressure Problems")
    if transplants_val:
        risk_factors.append("Previous Transplants")
    if chronic_val:
        risk_factors.append("Chronic Diseases")
    if allergies_val:
        risk_factors.append("Known Allergies")
    if cancer_val:
        risk_factors.append("Family History of Cancer")
    if surgeries > 0:
        risk_factors.append(f"{surgeries} Major Surgeries")
    if bmi >= 30:
        risk_factors.append("Obesity (BMI ≥ 30)")
    elif bmi >= 25:
        risk_factors.append("Overweight (BMI ≥ 25)")
    
    if risk_factors:
        st.warning("**Risk Factors Identified:**")
        for factor in risk_factors:
            st.write(f"• {factor}")
    else:
        st.success("**No major risk factors identified!**")
    
    st.markdown("---")
    st.subheader("Recommendations")
    
    recommendations = []
    
    if bmi >= 30:
        recommendations.append("Consider weight management programs to reduce BMI")
    elif bmi >= 25:
        recommendations.append("Maintain a healthy diet and regular exercise")
    
    if diabetes_val:
        recommendations.append("Regular monitoring of blood sugar levels")
    
    if bp_val:
        recommendations.append("Regular blood pressure monitoring and medication compliance")
    
    if not recommendations:
        recommendations.append("Maintain your current healthy lifestyle")
        recommendations.append("Regular health check-ups are recommended")
    
    for rec in recommendations:
        st.info(f"💡 {rec}")

st.sidebar.header("About This Calculator")
st.sidebar.info("""
This insurance premium calculator uses machine learning to estimate your insurance costs based on various health and personal factors.

**Factors Considered:**
- Age
- Height & Weight (BMI)
- Diabetes
- Blood Pressure Problems
- Transplant History
- Chronic Diseases
- Known Allergies
- Family Cancer History
- Number of Major Surgeries

**Note:** This is an estimate only. Actual premiums may vary based on additional factors and insurance company policies.
""")

st.sidebar.markdown("---")
st.sidebar.markdown("**Developed using Streamlit**")
