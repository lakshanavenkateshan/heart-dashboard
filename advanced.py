# advanced_layout_fixed.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.naive_bayes import GaussianNB
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px

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

def suggestion_text(label):
    if label == "High":
        return "⚠️ High risk — consult a cardiologist soon, avoid smoking, follow heart-healthy diet, and get tests."
    elif label == "Medium":
        return "🩺 Moderate risk — monitor BP/cholesterol, exercise daily, maintain healthy weight."
    else:
        return "💚 Low risk — keep healthy habits; annual check-ups recommended."

# -------------------------
# Sidebar: Data / Model Setup + Manual Input + Suggestion
# -------------------------
st.sidebar.title("Data / Model Setup")

csv_path = st.sidebar.text_input("Local CSV path (or leave blank to use default processed_cleveland.csv)",
                                value="processed_cleveland.csv")

# Optional: allow upload for processed CSV
uploaded_main = st.sidebar.file_uploader("Or upload processed CSV", type=["csv"], key="main_csv")

try:
    if uploaded_main is not None:
        df = pd.read_csv(uploaded_main)
    else:
        df = load_and_prep(csv_path)
except Exception as e:
    st.sidebar.error(f"Couldn't load dataset: {e}")
    st.stop()

if "num" not in df.columns:
    st.sidebar.error("Dataset must contain 'num' target column (0=no disease, 1-4 = disease).")
    st.stop()

# keep original df for means
df = df.replace("?", np.nan).apply(pd.to_numeric, errors="coerce").fillna(df.mean())

X = df.drop("num", axis=1)
y = (df["num"] > 0).astype(int)

# scale and models (train once)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

log_reg = LogisticRegression(max_iter=2000)
rf = RandomForestClassifier(n_estimators=200, random_state=42)
nb = GaussianNB()
ensemble = VotingClassifier(estimators=[('lr', log_reg), ('rf', rf), ('nb', nb)], voting='soft')

# fit models
ensemble.fit(X_scaled, y)
rf.fit(X_scaled, y)

# SHAP explainer (robust)
try:
    explainer = shap.Explainer(rf, X, feature_perturbation="interventional")
except Exception:
    explainer = None

st.sidebar.markdown("### Manual Patient Input (use these to predict)")

# Manual inputs in sidebar (exact same fields)
manual_inputs = {}
for c in X.columns:
    manual_inputs[c] = st.sidebar.number_input(c, float(df[c].mean()), format="%.3f", key=f"manual_{c}")

manual_df = pd.DataFrame([manual_inputs])
manual_scaled = scaler.transform(manual_df)
manual_prob = ensemble.predict_proba(manual_scaled)[0][1]
manual_label, manual_color = risk_category(manual_prob)

st.sidebar.markdown(f"**Prediction:** <span style='color:{manual_color};font-weight:600'>{manual_label}</span>",
                    unsafe_allow_html=True)
st.sidebar.write(f"Probability: {manual_prob*100:.2f}%")

# Personalized suggestion under manual setup
st.sidebar.markdown("**Personalized suggestion:**")
if manual_label == "High":
    st.sidebar.error(suggestion_text("High"))
elif manual_label == "Medium":
    st.sidebar.warning(suggestion_text("Medium"))
else:
    st.sidebar.success(suggestion_text("Low"))

# -------------------------
# Main area: Title + What-if only (right side)
# -------------------------
st.markdown("<h1 style='text-align:center'>Advanced Heart Disease Risk Prediction — Explainable & Interactive</h1>",
            unsafe_allow_html=True)
st.markdown("---")

col_main_left, col_main_right = st.columns([2, 1])

with col_main_left:
    st.header("Single Patient Overview")
    st.write("Manual input prediction (shown on left).")
    # Show a compact SHAP & RF importance area (collapsible to keep main area clean)
    with st.expander("Explainability & Feature Importance (SHAP / Random Forest)"):
        # SHAP per-patient explanation (try robustly)
        st.subheader("SHAP explanation (local)")
        if explainer is not None:
            try:
                shap_values = explainer(manual_df)
                # normalize shap_values shape
                if hasattr(shap_values, "values") and shap_values.values.ndim == 3:
                    vals = shap_values.values[0][:, 0]
                elif hasattr(shap_values, "values"):
                    vals = shap_values.values[0]
                else:
                    vals = shap_values[0]
                contribs = pd.Series(vals, index=X.columns).sort_values(ascending=False)
                st.write("Top contributing features (local):")
                st.bar_chart(contribs.head(10))
            except Exception as e:
                st.warning(f"SHAP failed: {e}")
                st.write("Using RF feature importances instead:")
                st.bar_chart(pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False).head(10))
        else:
            st.write("SHAP unavailable — showing RF importances:")
            st.bar_chart(pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False).head(10))

    # Radar chart (patient vs mean)
    st.subheader("Patient vs Dataset Mean (Radar)")
    mean_vals = df.mean()[X.columns]
    categories = list(X.columns)
    patient_vals = manual_df.iloc[0].values
    radar_fig = go.Figure()
    radar_fig.add_trace(go.Scatterpolar(r=patient_vals, theta=categories, fill='toself', name='Patient'))
    radar_fig.add_trace(go.Scatterpolar(r=mean_vals.values, theta=categories, fill='toself', name='Dataset Mean'))
    radar_fig.update_layout(polar=dict(radialaxis=dict(visible=True)), showlegend=True, height=420)
    st.plotly_chart(radar_fig, use_container_width=True)

with col_main_right:
    st.header("What-if / Sensitivity Controls")
    st.markdown("Use sliders to adjust the most important features and watch probability update.")
    # use top 6 important features
    imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
    top_features = list(imp.index[:6])
    scenario = manual_df.copy()
    for f in top_features:
        default = float(manual_df[f][0])
        minv = float(df[f].min())
        maxv = float(df[f].max())
        scenario.at[0, f] = st.slider(f"Adjust {f}", min_value=minv, max_value=maxv, value=default, step=(maxv-minv)/100)
    scenario_scaled = scaler.transform(scenario)
    scenario_prob = ensemble.predict_proba(scenario_scaled)[0][1]
    scenario_label, scenario_color = risk_category(scenario_prob)
    st.markdown(f"**Scenario Risk:** <span style='color:{scenario_color};font-weight:600'>{scenario_label}</span>",
                unsafe_allow_html=True)
    st.write(f"Updated Probability: {scenario_prob*100:.2f}%")

    # suggestion after what-if
    if scenario_label == "High":
        st.error(suggestion_text("High"))
    elif scenario_label == "Medium":
        st.warning(suggestion_text("Medium"))
    else:
        st.success(suggestion_text("Low"))

    # show global feature importance compact
    st.markdown("**Global Feature Importance (RF)**")
    st.bar_chart(imp.head(10))

# -------------------------
# Batch Prediction / Multi-patient Analysis (separate section)
# -------------------------
st.markdown("---")
st.header("Batch Prediction / Multi-patient Analysis")
colA, colB = st.columns([1,2])
with colA:
    batch_uploaded = st.file_uploader("Upload CSV for batch prediction (must have same feature columns)", type=["csv"], key="batch")
    if st.button("Download sample template CSV"):
        sample = pd.DataFrame([df.mean().to_dict()])
        st.download_button("Download sample CSV", data=sample.to_csv(index=False), file_name="sample_patient_template.csv")
with colB:
    if batch_uploaded is not None:
        try:
            batch_df = pd.read_csv(batch_uploaded)
            # cleaning like main df
            batch_X = batch_df[X.columns].replace("?", np.nan)
            batch_X = batch_X.apply(pd.to_numeric, errors='coerce').fillna(df.mean())
            batch_scaled = scaler.transform(batch_X)
            batch_probs = ensemble.predict_proba(batch_scaled)[:,1]
            batch_df["risk_prob"] = batch_probs
            batch_df["risk_cat"] = [risk_category(p)[0] for p in batch_probs]
            st.write("Batch prediction preview (scrollable):")
            st.dataframe(batch_df, use_container_width=True, height=300)
            # distribution
            fig_batch = px.histogram(batch_df, x="risk_prob", nbins=20, title="Risk probability distribution")
            fig_batch.add_vline(x=0.3, line_dash="dash")
            fig_batch.add_vline(x=0.7, line_dash="dash")
            st.plotly_chart(fig_batch, use_container_width=True)
            st.download_button("Download annotated CSV", data=batch_df.to_csv(index=False), file_name="batch_predictions.csv")
        except Exception as e:
            st.error(f"Failed to process batch CSV: {e}")

# -------------------------
# Time-series Trend + Correlation Heatmap (separate area)
# -------------------------
st.markdown("---")
st.header("Time-series Trend (optional) & Correlation Heatmap")
st.markdown("Upload a CSV with columns: `patient_id`, `date` (YYYY-MM-DD), and all feature columns.")

ts_uploaded = st.file_uploader("Optional time-series CSV", type=["csv"], key="ts")
if ts_uploaded is not None:
    try:
        ts = pd.read_csv(ts_uploaded, parse_dates=["date"])
        if not {"patient_id", "date"}.issubset(ts.columns):
            st.error("Time-series CSV must contain 'patient_id' and 'date' columns.")
        else:
            ts = ts.sort_values(by=["patient_id","date"])
            ts_features = ts[X.columns]
            ts_scaled = scaler.transform(ts_features)
            ts["risk_prob"] = ensemble.predict_proba(ts_scaled)[:,1]
            ts["risk_cat"] = [risk_category(p)[0] for p in ts["risk_prob"]]
            pid = st.selectbox("Select patient_id for trend", options=ts["patient_id"].unique())
            p_df = ts[ts["patient_id"]==pid].set_index("date").sort_index()
            st.line_chart(p_df["risk_prob"])
            st.dataframe(p_df.head(), use_container_width=True)
    except Exception as e:
        st.error(f"Failed processing time-series CSV: {e}")

# Correlation heatmap below
st.subheader("Feature Correlation Heatmap")
fig_c, ax = plt.subplots(figsize=(10,6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax)
st.pyplot(fig_c)

st.caption("Model trained on provided dataset; validate before clinical use. SHAP helps interpretability but does not replace clinical judgement.")
