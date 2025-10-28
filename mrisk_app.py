import streamlit as st
import pandas as pd
import pickle
import os
import re

# ------------------------------------------------------------------
# 1. CONFIG & MODEL LOADING
# ------------------------------------------------------------------
MODEL_PATH = "best_tuned_model.pkl"

st.set_page_config(page_title="Maternal High-Risk Pregnancy Predictor", layout="wide")
st.title("Maternal Health – High-Risk Pregnancy Predictor")

if not os.path.exists(MODEL_PATH):
    st.error(f"Model file not found: `{MODEL_PATH}`")
    st.stop()

try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

#st.markdown("Enter **exact values** as used during training (raw strings).")

# ------------------------------------------------------------------
# 2. USER INPUT
# ------------------------------------------------------------------
st.subheader("Enter Patient Details")

# c1, c2, c3 = st.columns(3)

# with c1:
#     name = st.text_input("Patient Name (optional)", "")
#     age = st.number_input("Age", min_value=10, max_value=70, value=25, step=1)
#     gravida = st.selectbox("Gravida", options=["1st", "2nd", "3rd"])
#     tt_injection = st.selectbox("TranslationTT Injection", options=["1st", "2nd", "3rd"])
#     gest_age = st.text_input("Gestational Age (weeks)", value=" ")
#     weight = st.text_input("Weight (kg)", value=" ")

# with c2:
#     height = st.text_input("Height (any unit, e.g. 5.3 ft)", value=" ")
#     bp = st.text_input("Blood Pressure (e.g. 120/80)", value=" ")
#     anemia = st.selectbox("Anemia", options=["Normal", "Mild", "Moderate", "Severe"])
#     jaundice = st.selectbox("Jaundice", options=["Normal", "Yes"])

# with c3:
#     fetal_pos = st.selectbox("Fetal Position", options=["Normal", "Abnormal"])
#     fhr = st.text_input("Fetal Heart Rate (bpm)", value="140")
#     urine_alb = st.selectbox("Urine Test – Albumin", options=["Normal", "Higher"])
#     urine_sug = st.selectbox("Urine Test – Sugar", options=["No", "Yes"])
#     vdrl = st.selectbox("VDRL", options=["Negative", "Positive"])
#     hrsag = st.selectbox("HRsAG", options=["Negative", "Positive"])
c1, c2, c3 = st.columns(3)

with c1:
    name = st.text_input("Patient Name (optional)", "")
    age = st.number_input("Age", min_value=10, max_value=70, value=25, step=1)

    # Dropdowns
    gravida = st.selectbox("Gravida", options=["", "1st", "2nd", "3rd"], index=0)
    tt_injection = st.selectbox("TranslationTT Injection", options=["", "1st", "2nd", "3rd"], index=0)

    # Text inputs – EMPTY by default
    gest_age = st.text_input("Gestational Age (weeks)", value="")
    weight = st.text_input("Weight (kg)", value="")

with c2:
    height = st.text_input("Height (any unit, e.g. 5.3 ft)", value="")
    bp = st.text_input("Blood Pressure (e.g. 120/80)", value="")

    anemia = st.selectbox("Anemia", options=["", "Normal", "Mild", "Moderate", "Severe"], index=0)
    jaundice = st.selectbox("Jaundice", options=["", "Normal", "Yes"], index=0)

with c3:
    fetal_pos = st.selectbox("Fetal Position", options=["", "Normal", "Abnormal"], index=0)
    fhr = st.text_input("Fetal Heart Rate (bpm)", value="")

    urine_alb = st.selectbox("Urine Test – Albumin", options=["", "Normal", "Higher"], index=0)
    urine_sug = st.selectbox("Urine Test – Sugar", options=["", "No", "Yes"], index=0)
    vdrl = st.selectbox("VDRL", options=["", "Negative", "Positive"], index=0)
    hrsag = st.selectbox("HRsAG", options=["", "Negative", "Positive"], index=0)
# ------------------------------------------------------------------
# 3. PREDICTION
# ------------------------------------------------------------------
if st.button("Predict Risk", type="primary"):
    # --- Input validation ---
    errors = []

    try:
        int(gest_age)
    except ValueError:
        errors.append("Gestational Age must be a number")

    try:
        float(weight)
    except ValueError:
        errors.append("Weight must be a number")

    try:
        float(height)
    except ValueError:
        errors.append("Height must be a number")

    try:
        int(fhr)
    except ValueError:
        errors.append("Fetal Heart Rate must be a number")

    bp_match = re.match(r"(\d{2,3})/(\d{2,3})", bp.strip())
    if not bp_match:
        errors.append("Blood Pressure must be in format `120/80`")

    if errors:
        for err in errors:
            st.error(err)
        st.stop()

    bp_str = f"{bp_match.group(1)}/{bp_match.group(2)}"

    # --- Build input DataFrame ---
    input_data = {
        "Age": age,
        "Gravida": gravida,
        "TranslationTT Injection": tt_injection,
        "Gestational Age": gest_age,
        "Weight": weight,
        "Height": height,
        "Blood Pressure": bp_str,
        "Anemia": anemia,
        "Jaundice": jaundice,
        "Fetal Position": fetal_pos,
        "Fetal Heart Rate": fhr,
        "Urine Test – Albumin": urine_alb,
        "Urine Test – Sugar": urine_sug,
        "VDRL": vdrl,
        "HRsAG": hrsag
    }

    input_df = pd.DataFrame([input_data])
    expected_cols = [
        "Age", "Gravida", "TranslationTT Injection", "Gestational Age", "Weight",
        "Height", "Blood Pressure", "Anemia", "Jaundice", "Fetal Position",
        "Fetal Heart Rate", "Urine Test – Albumin", "Urine Test – Sugar",
        "VDRL", "HRsAG"
    ]
    input_df = input_df[expected_cols]

    # --- Predict ---
    try:
        pred = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0]

        # --- CRITICAL FIX: Normalize probability to 0–1 ---
        # Some models return 0–100; we force 0–1
        high_risk_prob = float(proba[1]) if len(proba) > 1 else float(proba[0])
        high_risk_prob = high_risk_prob / 100 if high_risk_prob > 1.0 else high_risk_prob
        high_risk_prob = max(0.0, min(1.0, high_risk_prob))  # Clamp

    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.stop()

    # --- Display Result ---
    st.subheader("Prediction Result")
    patient = name.strip() or "Patient"

    if pred == 1:
        st.error(f"**{patient}** → **HIGH-RISK PREGNANCY**")
    else:
        st.success(f"**{patient}** → **NORMAL PREGNANCY**")

    # FIXED: st.progress expects 0.0 to 1.0
    st.progress(high_risk_prob)
    st.write(f"**High-Risk Probability:** {high_risk_prob:.1%}")

    # Optional: Show input
    with st.expander("View Input Data"):
        st.json(input_data)

# ------------------------------------------------------------------
# 4. FOOTER
# ------------------------------------------------------------------
st.markdown("---")
#st.caption("Model uses 15 raw string features. No preprocessing needed. Fixed progress bar issue.")