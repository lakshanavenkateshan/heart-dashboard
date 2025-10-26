import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import io

st.set_page_config(layout="wide", page_title="Advanced Heart Disease Dashboard")

# -------------------------
# Helper functions
# -------------------------
@st.cache_data
def load_and_prep(path):
    df = pd.read_csv(path)
    df = df.replace("?", np.nan).apply(pd.to_numeric, errors="coerce")
    df = df.fillna(df.mean())
    return df

def risk_category(prob):
    if prob < 0.3:
        return "Low", "green"
    elif prob < 0.7:
        return "Medium", "orange"
    else:
        return "High", "red"

def personalized_recs(top_features):
    recs = []
    for f in top_features:
        if "chol" in f.lower() or "cholesterol" in f.lower():
            recs.append("Reduce dietary saturated fat, increase fiber; consider lipid profile follow-up.")
        elif "thalach" in f.lower() or "max" in f.lower() or "heart" in f.lower():
            recs.append("Increase aerobic exercise gradually and consult cardiologist for stress test.")
        elif "trestbps" in f.lower() or "bp" in f.lower():
            recs.append("Monitor blood pressure, reduce salt intake, regular exercise.")
        elif "age" in f.lower():
            recs.append("Age is non-modifiable; focus on modifiable factors (exercise, diet, BP, lipids).")
        elif "cp" in f.lower() or "chest" in f.lower():
            recs.append("Investigate chest pain with physician; avoid strenuous activity until cleared.")
        else:
            recs.append(f"Address {f}: lifestyle modification and clinical follow-up as appropriate.")
    return list(dict.fromkeys(recs))

# -------------------------
# Load dataset & train
# -------------------------
st.sidebar.header("Data / Model Setup")
csv_path = st.sidebar.text_input("Local CSV path (or leave blank to use default processed_cleveland.csv)", value="processed_cleveland.csv")

try:
    df = load_and_prep(csv_path)
except Exception as e:
    st.sidebar.error(f"Couldn't load {csv_path}: {e}")
    st.stop()

if "num" not in df.columns:
    st.sidebar.error("Dataset must contain 'num' target column (0=no disease, 1-4 = disease).")
    st.stop()

X = df.drop("num", axis=1)
y = (df["num"] > 0).astype(int)

# scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# models
log_reg = LogisticRegression(max_iter=2000)
rf = RandomForestClassifier(n_estimators=200, random_state=42)
nb = GaussianNB()
ensemble = VotingClassifier(estimators=[('lr', log_reg), ('rf', rf), ('nb', nb)], voting='soft')
ensemble.fit(X_scaled, y)
rf.fit(X_scaled, y)

# SHAP explainer
explainer = shap.Explainer(rf, X, feature_perturbation="interventional")

# -------------------------
# Layout - top row
# -------------------------
st.title("Advanced Heart Disease Risk Prediction — Explainable & Interactive")
col1, col2 = st.columns([1,2])

with col1:
    st.subheader("Single Patient (Manual Input)")
    user_inputs = {}
    for c in X.columns:
        default = float(df[c].mean())
        minv = float(df[c].min() if np.isfinite(df[c].min()) else default - 50)
        maxv = float(df[c].max() if np.isfinite(df[c].max()) else default + 50)
        step = (maxv - minv) / 100 if (maxv - minv) != 0 else 1.0
        user_inputs[c] = st.number_input(c, value=default, min_value=minv, max_value=maxv, step=step, format="%.3f")
    input_df = pd.DataFrame([user_inputs])
    input_scaled = scaler.transform(input_df)
    prob = ensemble.predict_proba(input_scaled)[0][1]
    label, col = risk_category(prob)
    st.markdown(f"**Predicted Risk:** <span style='color:{col};font-weight:600'>{label}</span>", unsafe_allow_html=True)
    st.write("Probability:", f"{prob*100:.2f}%")
    # 💬 Personalized suggestion based on risk level
if label == "High":
    st.error("You are at high risk. Schedule a full cardiac check-up soon, reduce stress, avoid smoking, and follow a heart-healthy diet.")
elif label == "Medium":
    st.warning("You are at moderate risk. Maintain healthy weight, exercise 30 mins daily, and monitor BP and cholesterol regularly.")
else:
    st.success("You are at low risk. Keep up your healthy habits, balanced diet, and regular physical activity.")


    # -----------------------------
    # SHAP explanation (Fixed version)
    # -----------------------------
    st.markdown("**Why this prediction? (SHAP)**")
    try:
        shap_values = explainer(input_df)
        if hasattr(shap_values, "values") and shap_values.values.ndim == 3:
            vals = shap_values.values[0][:, 0]
        elif hasattr(shap_values, "values"):
            vals = shap_values.values[0]
        else:
            vals = shap_values[0]
        contribs = pd.Series(vals, index=X.columns).sort_values(ascending=False)
        st.write("Top contributing features for this prediction:")
        st.bar_chart(contribs.head(5))
    except Exception as e:
        st.warning(f"SHAP visualization failed — fallback used. Reason: {e}")
        feat_imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
        st.write("Top risk-contributing features (by model importance):")
        st.bar_chart(feat_imp.head(5))

    # Radar Chart
    st.markdown("**Patient vs Dataset Mean (Radar)**")
    mean_vals = df.mean()[X.columns]
    categories = list(X.columns)
    patient_vals = input_df.iloc[0].values
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=patient_vals, theta=categories, fill='toself', name='Patient'))
    fig.add_trace(go.Scatterpolar(r=mean_vals.values, theta=categories, fill='toself', name='Dataset Mean'))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True)), showlegend=True, height=500)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("What-if / Sensitivity Controls")
    st.markdown("Use sliders to adjust critical features and watch probability update.")
    imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
    top5 = list(imp.index[:6])
    scenario = input_df.copy()
    for f in top5:
        default = float(input_df[f][0])
        minv = float(df[f].min())
        maxv = float(df[f].max())
        s = st.slider(f"Adjust {f}", min_value=minv, max_value=maxv, value=default)
        scenario.at[0, f] = s
    scenario_scaled = scaler.transform(scenario)
    scenario_prob = ensemble.predict_proba(scenario_scaled)[0][1]
    srisk, scol = risk_category(scenario_prob)
    st.markdown(f"**Scenario Risk:** <span style='color:{scol};font-weight:600'>{srisk}</span>", unsafe_allow_html=True)
    st.write("Updated Probability:", f"{scenario_prob*100:.2f}%")

    st.markdown("**Global Feature Importance (Random Forest)**")
    st.bar_chart(imp)

    st.markdown("**Personalized Suggestions**")
    contribs = pd.Series(vals if 'vals' in locals() else rf.feature_importances_, index=X.columns)
    top_pos = contribs.sort_values(ascending=False).head(4).index.tolist()
    recs = personalized_recs(top_pos)
    for r in recs:
        st.write("- ", r)

# -------------------------
# Batch Prediction
# -------------------------
st.markdown("---")
st.header("Batch Prediction / Multi-patient Analysis")

colA, colB = st.columns([1,2])
with colA:
    uploaded = st.file_uploader("Upload CSV for batch prediction", type=["csv"])
    if uploaded:
        try:
            batch_df = pd.read_csv(uploaded)
            st.success("Batch CSV loaded.")
        except Exception as e:
            st.error(f"Failed to read uploaded CSV: {e}")
            batch_df = None
    else:
        st.info("No CSV uploaded.")
        batch_df = None

    if st.button("Download sample template CSV"):
        sample = pd.DataFrame([df.mean()[:].to_dict()])
        csv = sample.to_csv(index=False).encode()
        st.download_button("Download sample CSV", data=csv, file_name="sample_patient_template.csv", mime="text/csv")

with colB:
    if batch_df is not None:
        missing = set(X.columns) - set(batch_df.columns)
        if missing:
            st.error(f"Uploaded CSV missing columns: {missing}")
        else:
            # Clean and preprocess uploaded batch data
            batch_X = batch_df[X.columns].replace("?", np.nan)
            batch_X = batch_X.apply(pd.to_numeric, errors='coerce')
            batch_X = batch_X.fillna(df.mean())
            batch_scaled = scaler.transform(batch_X)

            batch_probs = ensemble.predict_proba(batch_scaled)[:,1]
            batch_df["risk_prob"] = batch_probs
            batch_df["risk_cat"], _ = zip(*[risk_category(p) for p in batch_probs])
            st.write("Batch prediction preview:")
            st.dataframe(batch_df.head(10))

            import plotly.express as px
            fig = px.histogram(batch_df, x="risk_prob", nbins=20, title="Risk probability distribution")
            fig.add_vline(x=0.3, line_dash="dash", annotation_text="0.3 low/med")
            fig.add_vline(x=0.7, line_dash="dash", annotation_text="0.7 med/high")
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("**Top 10 highest-risk entries**")
            st.dataframe(batch_df.sort_values("risk_prob", ascending=False).head(10))
            csv = batch_df.to_csv(index=False).encode()
            st.download_button("Download results CSV", data=csv, file_name="batch_predictions.csv", mime="text/csv")

# -------------------------
# Exploratory Analysis
# -------------------------
st.markdown("---")
st.header("Exploratory Analysis & Trend ")

colX, colY = st.columns(2)
with colX:
    st.subheader("Correlation Heatmap")
    fig_c, ax = plt.subplots(figsize=(10,8))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig_c)

with colY:
    st.subheader("Time-series Trend (if dataset has 'date')")
    st.markdown("Upload a CSV with columns: patient_id, date, and features.")
    ts_uploaded = st.file_uploader("Optional time-series CSV", type=["csv"], key="ts")
    if ts_uploaded:
        try:
            ts = pd.read_csv(ts_uploaded, parse_dates=["date"])
            if not {"patient_id", "date"}.issubset(ts.columns):
                st.error("Time-series CSV must contain 'patient_id' and 'date' columns.")
            else:
                ts_X = ts[X.columns]
                ts_scaled = scaler.transform(ts_X)
                ts_probs = ensemble.predict_proba(ts_scaled)[:,1]
                ts["risk_prob"] = ts_probs
                pid = st.selectbox("Select patient_id to view trend", options=ts["patient_id"].unique())
                p_df = ts[ts["patient_id"] == pid].sort_values("date")
                st.line_chart(data=p_df.set_index("date")["risk_prob"])
        except Exception as e:
            st.error(f"Failed processing time-series CSV: {e}")

st.markdown("---")
st.caption("Model trained on provided dataset; validate before clinical use. SHAP aids interpretability but does not replace medical judgement.")
