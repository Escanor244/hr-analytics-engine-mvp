# app.py

import streamlit as st
import pandas as pd
import joblib

# Load model and encoders
model = joblib.load('models/attrition_model.pkl')
encoders = joblib.load('models/label_encoders.pkl')

# Define burnout estimation function
def estimate_burnout(row):
    score = 0
    if row['OverTime'] == 'Yes': score += 1
    if row['JobSatisfaction'] <= 2: score += 1
    if row['WorkLifeBalance'] <= 2: score += 1
    if row['TotalWorkingYears'] >= 10 and row['YearsSinceLastPromotion'] > 3: score += 1
    return 'High' if score >= 3 else 'Medium' if score == 2 else 'Low'

st.title("HR Analytics Engine 🧠")
st.markdown("Predict Employee Attrition & Estimate Burnout Risk")

# Sidebar Inputs
st.sidebar.header("Enter Employee Info")
input_data = {}

# Collect inputs
input_data['Age'] = st.sidebar.slider("Age", 18, 60, 30)
input_data['BusinessTravel'] = st.sidebar.selectbox("Business Travel", ['Non-Travel', 'Travel_Rarely', 'Travel_Frequently'])
input_data['DailyRate'] = st.sidebar.slider("Daily Rate", 100, 1500, 800)
input_data['Department'] = st.sidebar.selectbox("Department", ['Sales', 'Research & Development', 'Human Resources'])
input_data['DistanceFromHome'] = st.sidebar.slider("Distance From Home", 1, 30, 10)
input_data['Education'] = st.sidebar.selectbox("Education Level", [1, 2, 3, 4, 5])
input_data['EducationField'] = st.sidebar.selectbox("Education Field", ['Life Sciences', 'Medical', 'Marketing', 'Technical Degree', 'Human Resources', 'Other'])
input_data['EnvironmentSatisfaction'] = st.sidebar.selectbox("Environment Satisfaction", [1, 2, 3, 4])
input_data['Gender'] = st.sidebar.selectbox("Gender", ['Male', 'Female'])
input_data['HourlyRate'] = st.sidebar.slider("Hourly Rate", 30, 150, 60)
input_data['JobInvolvement'] = st.sidebar.selectbox("Job Involvement", [1, 2, 3, 4])
input_data['JobLevel'] = st.sidebar.selectbox("Job Level", [1, 2, 3, 4, 5])
input_data['JobRole'] = st.sidebar.selectbox("Job Role", [
    'Sales Executive', 'Research Scientist', 'Laboratory Technician',
    'Manufacturing Director', 'Healthcare Representative', 'Manager',
    'Sales Representative', 'Research Director', 'Human Resources'
])
input_data['JobSatisfaction'] = st.sidebar.selectbox("Job Satisfaction", [1, 2, 3, 4])
input_data['MaritalStatus'] = st.sidebar.selectbox("Marital Status", ['Single', 'Married', 'Divorced'])
input_data['MonthlyIncome'] = st.sidebar.slider("Monthly Income", 1000, 20000, 5000)
input_data['MonthlyRate'] = st.sidebar.slider("Monthly Rate", 1000, 25000, 12000)
input_data['NumCompaniesWorked'] = st.sidebar.slider("Num Companies Worked", 0, 10, 2)
input_data['OverTime'] = st.sidebar.selectbox("OverTime", ['Yes', 'No'])
input_data['PercentSalaryHike'] = st.sidebar.slider("Percent Salary Hike", 10, 25, 15)
input_data['PerformanceRating'] = st.sidebar.selectbox("Performance Rating", [1, 2, 3, 4])
input_data['RelationshipSatisfaction'] = st.sidebar.selectbox("Relationship Satisfaction", [1, 2, 3, 4])
input_data['StockOptionLevel'] = st.sidebar.selectbox("Stock Option Level", [0, 1, 2, 3])
input_data['TotalWorkingYears'] = st.sidebar.slider("Total Working Years", 0, 40, 10)
input_data['TrainingTimesLastYear'] = st.sidebar.slider("Training Times Last Year", 0, 6, 2)
input_data['WorkLifeBalance'] = st.sidebar.selectbox("Work Life Balance", [1, 2, 3, 4])
input_data['YearsAtCompany'] = st.sidebar.slider("Years at Company", 0, 40, 5)
input_data['YearsInCurrentRole'] = st.sidebar.slider("Years in Current Role", 0, 18, 4)
input_data['YearsSinceLastPromotion'] = st.sidebar.slider("Years Since Last Promotion", 0, 15, 3)
input_data['YearsWithCurrManager'] = st.sidebar.slider("Years with Current Manager", 0, 17, 5)

# Convert to DataFrame
user_input = pd.DataFrame([input_data])

# Encode categorical columns
for col, encoder in encoders.items():
    if col in user_input.columns:
        user_input[col] = encoder.transform(user_input[col])
# Align with training columns
input_columns = joblib.load('models/input_columns.pkl')
user_input = user_input.reindex(columns=input_columns, fill_value=0)


# 🔍 Optional: show what the model receives
st.write("🔍 Encoded Input for Model", user_input)

# Predict attrition
try:
    prediction = model.predict(user_input)[0]
    pred_result = 'Yes 🔴' if prediction == 1 else 'No 🟢'
except Exception as e:
    st.error(f"Prediction failed: {e}")
    pred_result = 'Error'

# Estimate burnout
burnout_label = estimate_burnout(input_data)

# Show results
st.subheader("Prediction")
st.write(f"**Will the employee leave?**: {pred_result}")

st.subheader("Burnout Risk")
st.write(f"**Burnout Level**: {burnout_label}")
