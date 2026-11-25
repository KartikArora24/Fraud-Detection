import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Credit Card Fraud Detection", page_icon="💳", layout="wide")

ARTIFACT_PATH = "artifacts/fraud_artifacts.joblib"

@st.cache_resource
def load_artifacts():
    if not os.path.exists(ARTIFACT_PATH):
        return None
    return joblib.load(ARTIFACT_PATH)

artifacts = load_artifacts()

if artifacts is None:
    st.error(
        "Model artifacts (`artifacts/fraud_artifacts.joblib`) not found! "
        "Please place `creditcard.csv` in the project root, run `python train_model.py`, "
        "and commit `artifacts/fraud_artifacts.joblib` to your repository."
    )
    st.stop()

scaler = artifacts["scaler"]
all_columns = artifacts["all_columns"]
best_indices = artifacts["best_indices"]
selected_features = artifacts["selected_features"]
forest = artifacts["forest"]
lof = artifacts["lof"]
if_cutoff = artifacts["if_cutoff"]
lof_cutoff = artifacts["lof_cutoff"]

def predict_transactions(df_input: pd.DataFrame):
    X = pd.DataFrame(0.0, index=df_input.index, columns=all_columns)
    for col in all_columns:
        if col in df_input.columns:
            X[col] = df_input[col].values

    X["Amount"] = np.log1p(np.maximum(X["Amount"], 0))
    scaled = scaler.transform(X)
    features = scaled[:, best_indices]

    if_scores = -forest.score_samples(features)
    lof_scores = -lof.score_samples(features)

    if_preds = (if_scores >= if_cutoff).astype(int)
    lof_preds = (lof_scores >= lof_cutoff).astype(int)

    return if_scores, if_preds, lof_scores, lof_preds

# --- UI ---
st.title("💳 Credit Card Fraud Detection")
st.markdown(
    "Anomaly detection using **Isolation Forest** and **Local Outlier Factor (LOF)** "
    "calibrated at a **70% validation precision** target."
)

tab1, tab2 = st.tabs(["🔍 Single Transaction Simulator", "📂 Batch CSV Prediction"])

with tab1:
    st.subheader("Test a Transaction")
    col_btn1, col_btn2, _ = st.columns([1, 1, 4])
    
    if "preset" not in st.session_state:
        st.session_state.preset = artifacts["sample_normal"]

    if col_btn1.button("Load Sample Genuine"):
        st.session_state.preset = artifacts["sample_normal"]
    if col_btn2.button("Load Sample Fraud"):
        st.session_state.preset = artifacts["sample_fraud"]

    preset = st.session_state.preset

    with st.form("single_tx_form"):
        cols = st.columns(4)
        input_vals = {}
        for idx, feat in enumerate(selected_features):
            with cols[idx % 4]:
                input_vals[feat] = st.number_input(
                    f"{feat}",
                    value=float(round(preset.get(feat, 0.0), 4)),
                    step=0.1,
                    format="%.4f"
                )
        submitted = st.form_submit_button("Analyze Transaction", type="primary")

    if submitted:
        row_df = pd.DataFrame([input_vals])
        if_score, if_pred, lof_score, lof_pred = predict_transactions(row_df)

        st.divider()
        res_col1, res_col2 = st.columns(2)

        with res_col1:
            st.markdown("### 🌲 Isolation Forest")
            if if_pred[0] == 1:
                st.error(f"🚨 **FRAUD ALERT** (Score: `{if_score[0]:.4f}` ≥ Cutoff: `{if_cutoff:.4f}`)")
            else:
                st.success(f"✅ **GENUINE** (Score: `{if_score[0]:.4f}` < Cutoff: `{if_cutoff:.4f}`)")

        with res_col2:
            st.markdown("### 📍 Local Outlier Factor (LOF)")
            if lof_pred[0] == 1:
                st.error(f"🚨 **FRAUD ALERT** (Score: `{lof_score[0]:.4f}` ≥ Cutoff: `{lof_cutoff:.4f}`)")
            else:
                st.success(f"✅ **GENUINE** (Score: `{lof_score[0]:.4f}` < Cutoff: `{lof_cutoff:.4f}`)")

with tab2:
    st.subheader("Upload Transaction CSV")
    st.caption(
        f"Your CSV should contain at least the 8 selected features: "
        f"`{', '.join(selected_features)}` (or the full `creditcard.csv` schema)."
    )
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)
        if_scores, if_preds, lof_scores, lof_preds = predict_transactions(batch_df)

        results_df = batch_df.copy()
        results_df["IF_Anomaly_Score"] = np.round(if_scores, 4)
        results_df["IF_Prediction"] = np.where(if_preds == 1, "Fraud", "Genuine")
        results_df["LOF_Anomaly_Score"] = np.round(lof_scores, 4)
        results_df["LOF_Prediction"] = np.where(lof_preds == 1, "Fraud", "Genuine")

        c1, c2, c3 = st.columns(3)
        c1.metric("Total Transactions", f"{len(results_df):,}")
        c2.metric("Isolation Forest Frauds", f"{int(if_preds.sum()):,}")
        c3.metric("LOF Frauds", f"{int(lof_preds.sum()):,}")

        st.dataframe(results_df.head(50), use_container_width=True)

        csv_out = results_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download Predictions CSV",
            data=csv_out,
            file_name="fraud_predictions.csv",
            mime="text/csv"
        )
