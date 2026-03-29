import streamlit as st
import requests
import pandas as pd

API_URL = "https://attrition-backend.onrender.com"

st.set_page_config(page_title="Attrition System", layout="wide")

# -------------------------
# SESSION STATE
# -------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "employee_data" not in st.session_state:
    st.session_state.employee_data = None

if "probability" not in st.session_state:
    st.session_state.probability = None


# -------------------------
# TITLE
# -------------------------
st.title("💼 Employee Attrition Management System")

# -------------------------
# FORM
# -------------------------
with st.form("form"):

    col1, col2, col3 = st.columns(3)

    with col1:
        Age = st.number_input("Age", 18, 60, 30)
        BusinessTravel = st.selectbox("Business Travel", ["Travel_Rarely", "Travel_Frequently", "Non-Travel"])
        DailyRate = st.number_input("Daily Rate", 100, 1500, 800)
        Department = st.selectbox("Department", ["Sales", "Research & Development", "Human Resources"])
        DistanceFromHome = st.number_input("Distance From Home", 1, 50, 10)
        Education = st.selectbox("Education", [1,2,3,4,5])
        EducationField = st.selectbox("Education Field", ["Life Sciences","Medical","Marketing","Technical Degree","Human Resources","Other"])

    with col2:
        EnvironmentSatisfaction = st.selectbox("Environment Satisfaction", [1,2,3,4])
        Gender = st.selectbox("Gender", ["Male", "Female"])
        HourlyRate = st.number_input("Hourly Rate", 30, 100, 65)
        JobInvolvement = st.selectbox("Job Involvement", [1,2,3,4])
        JobLevel = st.selectbox("Job Level", [1,2,3,4,5])
        JobRole = st.selectbox("Job Role", [
            "Sales Executive","Research Scientist","Laboratory Technician",
            "Manufacturing Director","Healthcare Representative",
            "Manager","Sales Representative","Research Director","Human Resources"
        ])
        JobSatisfaction = st.selectbox("Job Satisfaction", [1,2,3,4])

    with col3:
        MaritalStatus = st.selectbox("Marital Status", ["Single","Married","Divorced"])
        MonthlyIncome = st.number_input("Monthly Income", 1000, 20000, 4000)
        MonthlyRate = st.number_input("Monthly Rate", 2000, 30000, 12000)
        NumCompaniesWorked = st.number_input("Companies Worked", 0, 10, 3)
        OverTime = st.selectbox("OverTime", ["Yes","No"])
        PercentSalaryHike = st.number_input("Salary Hike %", 10, 30, 15)
        PerformanceRating = st.selectbox("Performance Rating", [1,2,3,4])

    col4, col5 = st.columns(2)

    with col4:
        RelationshipSatisfaction = st.selectbox("Relationship Satisfaction", [1,2,3,4])
        StockOptionLevel = st.selectbox("Stock Option Level", [0,1,2,3])
        TotalWorkingYears = st.number_input("Total Working Years", 0, 40, 10)
        TrainingTimesLastYear = st.number_input("Training Times Last Year", 0, 10, 3)

    with col5:
        WorkLifeBalance = st.selectbox("Work Life Balance", [1,2,3,4])
        YearsAtCompany = st.number_input("Years At Company", 0, 40, 5)
        YearsInCurrentRole = st.number_input("Years In Role", 0, 20, 3)
        YearsSinceLastPromotion = st.number_input("Years Since Promotion", 0, 15, 1)
        YearsWithCurrManager = st.number_input("Years With Manager", 0, 20, 3)

    submit = st.form_submit_button("🔍 Predict")


# -------------------------
# INPUT DATA
# -------------------------
input_data = {
    "Age": Age,
    "BusinessTravel": BusinessTravel,
    "DailyRate": DailyRate,
    "Department": Department,
    "DistanceFromHome": DistanceFromHome,
    "Education": Education,
    "EducationField": EducationField,
    "EnvironmentSatisfaction": EnvironmentSatisfaction,
    "Gender": Gender,
    "HourlyRate": HourlyRate,
    "JobInvolvement": JobInvolvement,
    "JobLevel": JobLevel,
    "JobRole": JobRole,
    "JobSatisfaction": JobSatisfaction,
    "MaritalStatus": MaritalStatus,
    "MonthlyIncome": MonthlyIncome,
    "MonthlyRate": MonthlyRate,
    "NumCompaniesWorked": NumCompaniesWorked,
    "OverTime": OverTime,
    "PercentSalaryHike": PercentSalaryHike,
    "PerformanceRating": PerformanceRating,
    "RelationshipSatisfaction": RelationshipSatisfaction,
    "StockOptionLevel": StockOptionLevel,
    "TotalWorkingYears": TotalWorkingYears,
    "TrainingTimesLastYear": TrainingTimesLastYear,
    "WorkLifeBalance": WorkLifeBalance,
    "YearsAtCompany": YearsAtCompany,
    "YearsInCurrentRole": YearsInCurrentRole,
    "YearsSinceLastPromotion": YearsSinceLastPromotion,
    "YearsWithCurrManager": YearsWithCurrManager
}


# -------------------------
# PREDICT
# -------------------------
if submit:

    res = requests.post(f"{API_URL}/predict", json=input_data).json()

    if "attrition_probability" in res:

        prob = res["attrition_probability"]

        st.session_state.employee_data = input_data
        st.session_state.probability = prob

        st.metric("Attrition Risk", f"{prob:.2%}")

        chart = pd.DataFrame({
            "Category": ["Stay", "Leave"],
            "Value": [1-prob, prob]
        })

        st.bar_chart(chart.set_index("Category"))

    else:
        st.error(res)


# -------------------------
# CHAT
# -------------------------
st.divider()
st.subheader("🤖 Attrition CoPilot Chat")

user_input = st.chat_input("Ask anything...")

if user_input:

    st.session_state.chat_history.append(
        {"role": "user", "content": user_input}
    )

    payload = {
        "messages": st.session_state.chat_history,
        "employee_data": st.session_state.employee_data,
        "probability": st.session_state.probability
    }

    res = requests.post(f"{API_URL}/chat", json=payload).json()

    if "reply" in res:

        st.session_state.chat_history.append(
            {"role": "assistant", "content": res["reply"]}
        )

    else:
        st.error(res)


# -------------------------
# DISPLAY CHAT
# -------------------------
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])