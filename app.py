import streamlit as st, pandas as pd, joblib, shap
import matplotlib.pyplot as plt

st.set_page_config(page_title="Churn Predictor", layout="wide")
model = joblib.load("models/xgb.pkl")
columns = joblib.load("models/columns.pkl")
defaults = joblib.load("models/defaults.pkl")
options = joblib.load("models/options.pkl")

st.title("Customer Churn Prediction")
st.caption("Predict churn risk and see *why* the model thinks so.")

# Sidebar inputs (key drivers; everything else uses dataset defaults)
st.sidebar.header("Customer details")
inp = dict(defaults)
inp["tenure"] = st.sidebar.slider("Tenure (months)", 0, 72, 12)
inp["MonthlyCharges"] = st.sidebar.slider("Monthly charges", 18.0, 120.0, 70.0)
inp["TotalCharges"] = inp["tenure"] * inp["MonthlyCharges"]
for col in ["Contract", "InternetService", "PaymentMethod", "TechSupport"]:
    inp[col] = st.sidebar.selectbox(col, options[col])

# Encode exactly like training
row = pd.get_dummies(pd.DataFrame([inp])).reindex(columns=columns, fill_value=0).astype(float)

prob = model.predict_proba(row)[0, 1]
c1, c2 = st.columns(2)
c1.metric("Churn probability", f"{prob:.1%}")
c2.metric("Risk level", "High" if prob > 0.6 else "Medium" if prob > 0.3 else "Low")

# SHAP explanation
st.subheader("What is driving this prediction?")
explainer = shap.TreeExplainer(model)
sv = explainer(row)
import io

shap.plots.waterfall(sv[0], max_display=10, show=False)
fig = plt.gcf()
buf = io.BytesIO()
fig.savefig(buf, format="png", bbox_inches="tight", dpi=200)
plt.close(fig)
st.image(buf)
st.info("Red bars push toward churn; blue bars push toward staying.")