import streamlit as st
import pandas as pd
import pickle
import numpy as np

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CreditWise – Loan Approval Predictor",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

/* ── Reset & base ── */
*, *::before, *::after { box-sizing: border-box; }

html, body, [data-testid="stAppViewContainer"] {
    background: #0a0a0f;
    color: #e8e4dc;
    font-family: 'DM Sans', sans-serif;
}

[data-testid="stHeader"] { background: transparent; }
[data-testid="stMainBlockContainer"] { padding: 2rem 3rem; max-width: 1100px; margin: 0 auto; }

/* ── Hero ── */
.hero {
    text-align: center;
    padding: 3.5rem 0 2.5rem;
    position: relative;
}
.hero::before {
    content: '';
    position: absolute;
    top: 0; left: 50%;
    transform: translateX(-50%);
    width: 600px; height: 300px;
    background: radial-gradient(ellipse at center, rgba(99,102,241,0.18) 0%, transparent 70%);
    pointer-events: none;
}
.hero-badge {
    display: inline-block;
    border: 1px solid rgba(99,102,241,0.4);
    color: #a5b4fc;
    font-family: 'DM Sans', sans-serif;
    font-size: 0.72rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    padding: 0.35rem 1rem;
    border-radius: 100px;
    margin-bottom: 1.2rem;
    background: rgba(99,102,241,0.08);
}
.hero h1 {
    font-family: 'Syne', sans-serif;
    font-size: clamp(2.4rem, 5vw, 3.8rem);
    font-weight: 800;
    line-height: 1.08;
    letter-spacing: -0.03em;
    margin: 0 0 0.8rem;
    background: linear-gradient(135deg, #e8e4dc 30%, #a5b4fc 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.hero p {
    color: #8b8680;
    font-size: 1.05rem;
    font-weight: 300;
    max-width: 520px;
    margin: 0 auto;
    line-height: 1.7;
}

/* ── Divider ── */
.divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(99,102,241,0.3), transparent);
    margin: 2rem 0;
}

/* ── Section label ── */
.section-label {
    font-family: 'Syne', sans-serif;
    font-size: 0.68rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #6366f1;
    margin-bottom: 1.2rem;
    display: flex;
    align-items: center;
    gap: 0.6rem;
}
.section-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: rgba(99,102,241,0.2);
}

/* ── Cards ── */
.card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 16px;
    padding: 1.6rem 1.8rem;
    margin-bottom: 1rem;
}

/* ── Streamlit widget overrides ── */
[data-testid="stSelectbox"] > div > div,
[data-testid="stNumberInput"] input {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 10px !important;
    color: #e8e4dc !important;
    font-family: 'DM Sans', sans-serif !important;
}
[data-testid="stSelectbox"] > div > div:hover,
[data-testid="stNumberInput"] input:focus {
    border-color: rgba(99,102,241,0.5) !important;
    box-shadow: 0 0 0 3px rgba(99,102,241,0.08) !important;
}
label[data-testid="stWidgetLabel"] p {
    color: #a09c96 !important;
    font-size: 0.82rem !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 400 !important;
    letter-spacing: 0.01em;
}
[data-testid="stSlider"] > div { accent-color: #6366f1; }

/* ── Submit button ── */
[data-testid="stButton"] > button {
    width: 100%;
    background: linear-gradient(135deg, #6366f1 0%, #818cf8 100%) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 12px !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 0.95rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.04em;
    padding: 0.8rem 2rem !important;
    transition: all 0.2s ease;
    text-transform: uppercase;
}
[data-testid="stButton"] > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 30px rgba(99,102,241,0.35) !important;
}
[data-testid="stButton"] > button:active { transform: translateY(0); }

/* ── Result box ── */
.result-approved {
    background: linear-gradient(135deg, rgba(16,185,129,0.12), rgba(16,185,129,0.04));
    border: 1px solid rgba(16,185,129,0.35);
    border-radius: 20px;
    padding: 2.5rem;
    text-align: center;
    margin-top: 1.5rem;
}
.result-rejected {
    background: linear-gradient(135deg, rgba(239,68,68,0.12), rgba(239,68,68,0.04));
    border: 1px solid rgba(239,68,68,0.35);
    border-radius: 20px;
    padding: 2.5rem;
    text-align: center;
    margin-top: 1.5rem;
}
.result-icon { font-size: 3rem; margin-bottom: 0.8rem; display: block; }
.result-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.9rem;
    font-weight: 800;
    margin: 0 0 0.5rem;
    letter-spacing: -0.02em;
}
.result-approved .result-title { color: #34d399; }
.result-rejected .result-title { color: #f87171; }
.result-subtitle { color: #8b8680; font-size: 0.92rem; }

.prob-bar-wrap {
    margin-top: 1.5rem;
    background: rgba(255,255,255,0.05);
    border-radius: 100px;
    height: 10px;
    overflow: hidden;
}
.prob-bar-fill-green {
    height: 100%;
    border-radius: 100px;
    background: linear-gradient(90deg, #10b981, #34d399);
    transition: width 0.8s ease;
}
.prob-bar-fill-red {
    height: 100%;
    border-radius: 100px;
    background: linear-gradient(90deg, #ef4444, #f87171);
    transition: width 0.8s ease;
}
.prob-label {
    display: flex;
    justify-content: space-between;
    font-size: 0.78rem;
    color: #6b6761;
    margin-top: 0.5rem;
}

/* ── Footer ── */
.footer {
    text-align: center;
    color: #3d3c3a;
    font-size: 0.75rem;
    padding: 3rem 0 1.5rem;
    letter-spacing: 0.04em;
}
</style>
""", unsafe_allow_html=True)

# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open("model_1.pkl", "rb") as f:
        return pickle.load(f)

try:
    model = load_model()
    model_loaded = True
except FileNotFoundError:
    model_loaded = False

# ── Hero section ──────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-badge">AI-Powered Decision Engine</div>
    <h1>CreditWise</h1>
    <p>Enter applicant details below to get an instant loan approval prediction powered by Gaussian Naive Bayes.</p>
</div>
<div class="divider"></div>
""", unsafe_allow_html=True)

if not model_loaded:
    st.error("⚠️  `model_1.pkl` not found. Place it in the same directory as this app and restart.")
    st.stop()

# ── Form ──────────────────────────────────────────────────────────────────────

# Section 1 – Personal Information
st.markdown('<div class="section-label">01 — Personal Information</div>', unsafe_allow_html=True)
with st.container():
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        age = st.number_input("Age", min_value=18, max_value=90, value=35)
    with c2:
        gender = st.selectbox("Gender", ["Male", "Female"])
    with c3:
        marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Widowed"])
    with c4:
        dependents = st.number_input("Dependents", min_value=0, max_value=10, value=1)

    c5, c6 = st.columns(2)
    with c5:
        education = st.selectbox("Education Level", ["High School", "Bachelor's", "Master's", "PhD", "Other"])
    with c6:
        employment_status = st.selectbox("Employment Status", ["Employed", "Self-Employed", "Unemployed", "Part-Time"])

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# Section 2 – Financial Profile
st.markdown('<div class="section-label">02 — Financial Profile</div>', unsafe_allow_html=True)
with st.container():
    c1, c2, c3 = st.columns(3)
    with c1:
        applicant_income = st.number_input("Applicant Income (₹)", min_value=0, value=50000, step=1000)
    with c2:
        coapplicant_income = st.number_input("Co-applicant Income (₹)", min_value=0, value=0, step=1000)
    with c3:
        savings = st.number_input("Savings (₹)", min_value=0, value=20000, step=1000)

    c4, c5, c6 = st.columns(3)
    with c4:
        credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=700)
    with c5:
        existing_loans = st.number_input("Existing Loans", min_value=0, max_value=20, value=1)
    with c6:
        dti_ratio = st.number_input("DTI Ratio", min_value=0.0, max_value=1.0, value=0.3, step=0.01, format="%.2f")

    c7, c8 = st.columns(2)
    with c7:
        employer_category = st.selectbox("Employer Category", ["Government", "Private", "NGO", "Self", "Other"])
    with c8:
        collateral_value = st.number_input("Collateral Value (₹)", min_value=0, value=100000, step=5000)

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# Section 3 – Loan Details
st.markdown('<div class="section-label">03 — Loan Details</div>', unsafe_allow_html=True)
with st.container():
    c1, c2, c3 = st.columns(3)
    with c1:
        loan_amount = st.number_input("Loan Amount (₹)", min_value=1000, value=200000, step=5000)
    with c2:
        loan_term = st.selectbox("Loan Term (months)", [12, 24, 36, 48, 60, 84, 120, 180, 240, 360], index=3)
    with c3:
        loan_purpose = st.selectbox("Loan Purpose", ["Home", "Education", "Business", "Personal", "Medical", "Auto", "Other"])

    property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

st.markdown("<br>", unsafe_allow_html=True)

# ── Predict button ─────────────────────────────────────────────────────────────
predict_clicked = st.button("🔍  Predict Loan Approval")

if predict_clicked:
    input_data = pd.DataFrame([{
        "Applicant_Income": applicant_income,
        "Coapplicant_Income": coapplicant_income,
        "Employment_Status": employment_status,
        "Age": age,
        "Marital_Status": marital_status,
        "Dependents": dependents,
        "Credit_Score": credit_score,
        "Existing_Loans": existing_loans,
        "DTI_Ratio": dti_ratio,
        "Savings": savings,
        "Collateral_Value": collateral_value,
        "Loan_Amount": loan_amount,
        "Loan_Term": loan_term,
        "Loan_Purpose": loan_purpose,
        "Property_Area": property_area,
        "Education_Level": education,
        "Gender": gender,
        "Employer_Category": employer_category,
    }])

    prediction = model.predict(input_data)[0]
    proba = model.predict_proba(input_data)[0]

    # Map prediction to readable string
    approved_label = "Yes"
    approved = str(prediction).strip() in [approved_label, "1", "1.0", "yes", "Yes", "YES", "Approved", "approved"]

    # Try to get the correct class probability
    classes = list(model.classes_)
    try:
        approval_idx = [str(c) for c in classes].index(str(prediction))
        approval_prob = proba[approval_idx]
    except Exception:
        approval_prob = max(proba)

    pct = int(approval_prob * 100)

    if approved:
        bar_class = "prob-bar-fill-green"
        st.markdown(f"""
        <div class="result-approved">
            <span class="result-icon">✅</span>
            <div class="result-title">Loan Approved</div>
            <div class="result-subtitle">Model confidence: <strong style="color:#34d399">{pct}%</strong></div>
            <div class="prob-bar-wrap" style="margin-top:1.2rem">
                <div class="{bar_class}" style="width:{pct}%"></div>
            </div>
            <div class="prob-label"><span>0%</span><span>Approval Probability</span><span>100%</span></div>
        </div>
        """, unsafe_allow_html=True)
    else:
        reject_pct = 100 - pct
        st.markdown(f"""
        <div class="result-rejected">
            <span class="result-icon">❌</span>
            <div class="result-title">Loan Rejected</div>
            <div class="result-subtitle">Model confidence: <strong style="color:#f87171">{reject_pct}%</strong></div>
            <div class="prob-bar-wrap" style="margin-top:1.2rem">
                <div class="prob-bar-fill-red" style="width:{reject_pct}%"></div>
            </div>
            <div class="prob-label"><span>0%</span><span>Rejection Probability</span><span>100%</span></div>
        </div>
        """, unsafe_allow_html=True)

    # ── Key factors summary ────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">Summary — Input Snapshot</div>', unsafe_allow_html=True)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Credit Score", credit_score)
    m2.metric("DTI Ratio", f"{dti_ratio:.2f}")
    m3.metric("Loan Amount", f"₹{loan_amount:,}")
    m4.metric("Monthly Income", f"₹{applicant_income + coapplicant_income:,}")

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown('<div class="footer">CreditWise · Gaussian Naïve Bayes · For demonstration purposes only</div>', unsafe_allow_html=True)