# -------------------------
# ❤️ One-Page Heart Disease Dashboard (Final)
# -------------------------
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Heart Risk Dashboard", layout="wide")

# 🎨 Page Styling
st.markdown("""
    <style>
        [data-testid="stAppViewContainer"] {background: linear-gradient(to right, #f9f9ff, #fff3f3);}
        [data-testid="stHeader"] {background: rgba(255, 0, 0, 0.1);}
        .big-font {font-size:28px !important; font-weight:bold; color:#d6336c;}
        .section {background-color:white; padding:20px; border-radius:15px; box-shadow:0 0 8px rgba(0,0,0,0.1);}
    </style>
""", unsafe_allow_html=True)

# -------------------------
# Upload + Load CSV
# -------------------------
st.markdown("<h2 class='big-font'>❤️ Heart Disease Risk Prediction Dashboard</h2>", unsafe_allow_html=True)
uploaded = st.file_uploader("Upload your dataset (processed_cleveland.csv or similar)", type=["csv"])

if uploaded:
    df = pd.read_csv(uploaded)
else:
    st.warning("Please upload a dataset to begin.")
    st.stop()

# Clean data
df = df.replace("?", np.nan).apply(pd.to_numeric, errors="coerce").fillna(df.mean())

# Model setup
X = df.drop("num", axis=1)
y = (df["num"] > 0).astype(int)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

rf = RandomForestClassifier(n_estimators=200, random_state=42)
lr = LogisticRegression(max_iter=1000)
nb = GaussianNB()
ensemble = VotingClassifier(estimators=[('rf', rf), ('lr', lr), ('nb', nb)], voting='soft')
ensemble.fit(X_scaled, y)
rf.fit(X_scaled, y)  # ✅ Fit the RF model before using feature_importances_

# -------------------------
# Layout
# -------------------------
col1, col2, col3 = st.columns([1,1.5,1])

# Personalized suggestion generator
def suggestion(risk):
    if risk == "High Risk":
        return "⚠️ High risk detected — consult a cardiologist, maintain a low-fat diet, and monitor regularly."
    elif risk == "Medium Risk":
        return "⚠️ Moderate risk — maintain healthy weight, regular exercise, and control cholesterol."
    else:
        return "✅ Low risk — great job! Continue your healthy habits and get yearly checkups."

# --- Left: Single patient prediction ---
with col1:
    st.markdown("<div class='section'>", unsafe_allow_html=True)
    st.subheader("🩺 Single Patient Input")
    patient = {}
    for c in X.columns:
        patient[c] = st.number_input(f"{c}", float(df[c].mean()))
    input_df = pd.DataFrame([patient])
    input_scaled = scaler.transform(input_df)
    prob = ensemble.predict_proba(input_scaled)[0][1]

    if prob < 0.3:
        color, risk = "#28a745", "Low Risk"
    elif prob < 0.7:
        color, risk = "#ff9900", "Medium Risk"
    else:
        color, risk = "#dc3545", "High Risk"

    st.markdown(f"<h4 style='color:{color}'>Predicted Risk: {risk}</h4>", unsafe_allow_html=True)
    st.metric("Probability (%)", f"{prob*100:.2f}")
    st.info(suggestion(risk))
    st.markdown("</div>", unsafe_allow_html=True)

# --- Middle: Feature importance chart ---
with col2:
    st.markdown("<div class='section'>", unsafe_allow_html=True)
    st.subheader("🌿 Feature Importance")
    feat_imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
    fig = px.bar(feat_imp.head(10), x=feat_imp.head(10).values, y=feat_imp.head(10).index,
                 orientation='h', color=feat_imp.head(10).values,
                 color_continuous_scale="Reds", title="Top 10 Important Features")
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# --- Right: Correlation heatmap ---
with col3:
    st.markdown("<div class='section'>", unsafe_allow_html=True)
    st.subheader("📊 Correlation Heatmap")
    fig2, ax = plt.subplots(figsize=(5,4))
    sns.heatmap(df.corr(), cmap="coolwarm", cbar=False, ax=ax)
    st.pyplot(fig2)
    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------
# What-if Analysis (Personalized feedback)
# -------------------------
st.markdown("<br><div class='section'>", unsafe_allow_html=True)
st.subheader("🔍 What-if Analysis (Change parameters & simulate)")
sim_inputs = {}
for col in X.columns:
    sim_inputs[col] = st.number_input(f"{col}", float(df[col].mean()), key=f"sim_{col}")
sim_df = pd.DataFrame([sim_inputs])
sim_scaled = scaler.transform(sim_df)
sim_prob = ensemble.predict_proba(sim_scaled)[0][1]

if sim_prob < 0.3:
    sim_color, sim_label = "#28a745", "Low Risk"
elif sim_prob < 0.7:
    sim_color, sim_label = "#ff9900", "Medium Risk"
else:
    sim_color, sim_label = "#dc3545", "High Risk"

st.markdown(f"<h4 style='color:{sim_color}'>Updated Risk: {sim_label}</h4>", unsafe_allow_html=True)
st.metric("Updated Probability (%)", f"{sim_prob*100:.2f}")
st.success(suggestion(sim_label))
st.markdown("</div>", unsafe_allow_html=True)

# -------------------------
# Optional: Time-series trend (synthetic or uploaded)
# -------------------------
st.markdown("<br><div class='section'>", unsafe_allow_html=True)
st.subheader("📈 Time-series Trend (Optional)")
st.markdown("Upload a CSV with columns: `patient_id`, `date`, and features.")

time_file = st.file_uploader("Upload optional time-series CSV", type=["csv"], key="ts")
if time_file:
    ts_df = pd.read_csv(time_file)
    ts_df["date"] = pd.to_datetime(ts_df["date"])
    ts_df = ts_df.sort_values("date")
    ts_features = ts_df.drop(["patient_id", "date"], axis=1)
    ts_scaled = scaler.transform(ts_features)
    ts_df["risk_prob"] = ensemble.predict_proba(ts_scaled)[:,1]
    ts_df["risk_label"] = pd.cut(ts_df["risk_prob"], bins=[0,0.3,0.7,1], labels=["Low","Medium","High"])

    fig = px.line(ts_df, x="date", y="risk_prob", color="patient_id", title="Heart Risk Trend Over Time", markers=True)
    st.plotly_chart(fig, use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)

st.caption("⚠️ Educational use only — not for clinical diagnosis.")
