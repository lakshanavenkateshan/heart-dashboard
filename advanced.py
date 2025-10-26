# --- Clean & colorful dashboard with personal feedback ---
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
import plotly.express as px

# ---------- Load main dataset ----------
df = pd.read_csv("processed_cleveland.csv")
df = df.replace("?", np.nan).apply(pd.to_numeric, errors="coerce")
df = df.fillna(df.mean())

X = df.drop("num", axis=1)
y = (df["num"] > 0).astype(int)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

log_reg = LogisticRegression(max_iter=1000)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
nb = GaussianNB()

ensemble = VotingClassifier(estimators=[
    ('lr', log_reg),
    ('rf', rf),
    ('nb', nb)
], voting='soft')
ensemble.fit(X_scaled, y)

# ---------- Streamlit Dashboard ----------
st.title("Heart Disease Risk Prediction Dashboard")
st.markdown("A smart system that predicts heart risk, visualizes trends, and gives **personalized suggestions.**")

# --- Input Section ---
st.sidebar.header("Enter Patient Data")
user_data = {}
for col in X.columns:
    user_data[col] = st.sidebar.number_input(f"{col}", float(df[col].mean()))
input_df = pd.DataFrame([user_data])
input_scaled = scaler.transform(input_df)

# --- Predict risk ---
prob = ensemble.predict_proba(input_scaled)[0][1]
if prob < 0.3:
    risk_label, color = "Low Risk", "green"
elif prob < 0.7:
    risk_label, color = "Medium Risk", "orange"
else:
    risk_label, color = "High Risk", "red"

st.subheader("Prediction Result")
st.markdown(f"**Heart Disease Risk:** <span style='color:{color}; font-size:20px'>{risk_label}</span>", unsafe_allow_html=True)
st.write(f"🩺 **Probability Score:** {prob*100:.2f}%")

# --- Personalized Suggestion ---
def suggestion(risk):
    if risk == "High Risk":
        return "You are at high risk. Consult a cardiologist immediately and maintain a low-fat, high-fiber diet."
    elif risk == "Medium Risk":
        return "You are at moderate risk. Maintain healthy weight, exercise 30 mins daily, and monitor BP regularly."
    else:
        return "You are at low risk. Keep up your healthy habits and go for annual heart checkups."

st.success(suggestion(risk_label))

# --- What-if Analysis ---
st.subheader("What-if Analysis (Change parameters & simulate)")
sim_inputs = {}
for col in X.columns:
    sim_inputs[col] = st.number_input(f"{col}", float(input_df[col][0]), key=f"sim_{col}")
sim_df = pd.DataFrame([sim_inputs])
sim_scaled = scaler.transform(sim_df)
sim_prob = ensemble.predict_proba(sim_scaled)[0][1]
if sim_prob < 0.3:
    sim_label, sim_color = "Low Risk", "green"
elif sim_prob < 0.7:
    sim_label, sim_color = "Medium Risk", "orange"
else:
    sim_label, sim_color = "High Risk", "red"

st.markdown(f"**Updated Risk:** <span style='color:{sim_color}; font-size:18px'>{sim_label}</span>", unsafe_allow_html=True)
st.write(f"Updated Probability Score: {sim_prob*100:.2f}%")
st.info(suggestion(sim_label))

# --- Feature Importance ---
importances = rf.feature_importances_
feat_imp = pd.Series(importances, index=X.columns).sort_values(ascending=False)
st.subheader("Feature Importance")
st.bar_chart(feat_imp)

# --- Optional Time-series Upload ---
st.subheader("Time-series Trend (Optional)")
st.markdown("Upload a CSV with columns: `patient_id`, `date`, and features.")
time_file = st.file_uploader("Upload optional time-series CSV", type=["csv"])
if time_file is not None:
    ts_df = pd.read_csv(time_file)
    ts_df["date"] = pd.to_datetime(ts_df["date"])
    ts_df = ts_df.sort_values("date")

    # Predict over time
    ts_features = ts_df.drop(["patient_id", "date"], axis=1)
    ts_scaled = scaler.transform(ts_features)
    ts_df["risk_prob"] = ensemble.predict_proba(ts_scaled)[:, 1]
    ts_df["risk_label"] = pd.cut(ts_df["risk_prob"], bins=[0,0.3,0.7,1], labels=["Low","Medium","High"])

    st.write("🩺 Sample of time-series predictions:")
    st.dataframe(ts_df.head())

    fig = px.line(ts_df, x="date", y="risk_prob", color="patient_id",
                  title="Patient Heart Risk Trend Over Time", markers=True)
    st.plotly_chart(fig, use_container_width=True)
