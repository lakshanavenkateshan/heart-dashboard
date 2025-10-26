# -------------------------
# ❤️ One-Page Heart Disease Dashboard
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
st.markdown("<h2 class='big-font'>Heart Disease Risk Prediction Dashboard</h2>", unsafe_allow_html=True)
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

# -------------------------
# Layout
# -------------------------
col1, col2, col3 = st.columns([1,1.5,1])

# --- Left: Single patient prediction ---
with col1:
    st.markdown("<div class='section'>", unsafe_allow_html=True)
    st.subheader("🩺 Single Patient Input")
    patient = {}
    for c in X.columns[:6]:
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
    st.markdown("</div>", unsafe_allow_html=True)

# --- Middle: Feature importance chart ---
with col2:
    st.markdown("<div class='section'>", unsafe_allow_html=True)
    st.subheader("Feature Importance")
    rf.fit(X_scaled, y)
    feat_imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
    fig = px.bar(feat_imp.head(10), x=feat_imp.head(10).values, y=feat_imp.head(10).index,
                 orientation='h', color=feat_imp.head(10).values,
                 color_continuous_scale="Reds", title="Top 10 Important Features")
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# --- Right: Correlation heatmap ---
with col3:
    st.markdown("<div class='section'>", unsafe_allow_html=True)
    st.subheader("Correlation Heatmap")
    fig2, ax = plt.subplots(figsize=(5,4))
    sns.heatmap(df.corr(), cmap="coolwarm", cbar=False, ax=ax)
    st.pyplot(fig2)
    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------
# Bottom Row: Batch prediction summary
# -------------------------
st.markdown("<br><div class='section'>", unsafe_allow_html=True)
st.subheader("Batch Prediction Results")

batch_scaled = scaler.transform(X)
probs = ensemble.predict_proba(batch_scaled)[:,1]
df["Risk_Prob"] = probs
df["Risk_Level"] = pd.cut(probs, bins=[0,0.3,0.7,1], labels=["Low","Medium","High"])

colA, colB = st.columns(2)
with colA:
    st.write("Summary:")
    st.dataframe(df[["Risk_Level","Risk_Prob"]].head(10), use_container_width=True)
with colB:
    st.write("Risk Distribution:")
    fig3 = px.histogram(df, x="Risk_Prob", nbins=15, color="Risk_Level",
                        color_discrete_map={"Low":"green","Medium":"orange","High":"red"})
    st.plotly_chart(fig3, use_container_width=True)

st.markdown("</div>", unsafe_allow_html=True)


