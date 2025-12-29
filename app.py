import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeRegressor

# ---------- PAGE CONFIG ----------
st.set_page_config(page_title="Salary Predictor", layout="centered")

# ---------- SIMPLE CSS ----------
st.markdown("""
<style>
body {background-color:#f4f6fb;}
h1 {text-align:center; color:#2E7D32;}
</style>
""", unsafe_allow_html=True)

# ---------- TITLE ----------
st.markdown("<h1>🌳 Employee Salary Predictor (Decision Tree)</h1>", unsafe_allow_html=True)

# ---------- LOAD DATA ----------
data = pd.read_csv('emp_sal.csv')   # keep CSV in project folder
X = data.iloc[:, 1:2].values
y = data.iloc[:, 2].values

# ---------- DECISION TREE MODEL ----------
model = DecisionTreeRegressor(
    criterion="absolute_error",
    splitter="random",
    max_depth=3,
    min_samples_split=4,
    random_state=0,
    ccp_alpha=2
)
model.fit(X, y)

# ---------- USER INPUT ----------
level = st.slider("Select Experience Level", 1.0, 10.0, 6.5, 0.1)

# ---------- PREDICTION ----------
if st.button("Predict Salary"):
    salary = model.predict([[level]])[0]
    st.success(f"💰 Predicted Salary: ₹ {salary:,.2f}")

