import streamlit as st
import pandas as pd
import numpy as np
import joblib, json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import shap
from xgboost import XGBClassifier, XGBRegressor

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Credit Risk Lending System",
    page_icon="🏦",
    layout="wide"
)

# ── Load models ────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    xgb_calib  = joblib.load("models/xgb_base_for_calibration.pkl")
    platt      = joblib.load("models/platt_scaler.pkl")
    sizer      = joblib.load("models/xgb_loan_sizer.pkl")
    with open("models/features.json") as f:
        features = json.load(f)
    explainer = shap.TreeExplainer(xgb_calib)
    return xgb_calib, platt, sizer, features, explainer

xgb_calib, platt, sizer, FEATURES, explainer = load_models()

SIZER_FEATURES = [
    "log_income", "person_age", "person_emp_length",
    "income_stability", "credit_seniority", "grade_num",
    "ownership_num", "hist_default", "loan_int_rate",
    "cb_person_cred_hist_length", "intent_DEBTCONSOLIDATION",
    "intent_EDUCATION", "intent_HOMEIMPROVEMENT",
    "intent_MEDICAL", "intent_PERSONAL", "intent_VENTURE"
]

GRADE_MAP  = {"A": 6, "B": 5, "C": 4, "D": 3, "E": 2, "F": 1, "G": 0}
GRADE_CAPS = {"A": 0.50, "B": 0.45, "C": 0.35, "D": 0.25, "E": 0.22, "F": 0.20, "G": 0.15}
OWN_MAP    = {"OWN": 3, "MORTGAGE": 2, "RENT": 1, "OTHER": 0}
THRESHOLD  = 0.16
MAX_LOAN   = 35000

# ── Helper functions ───────────────────────────────────────────────────────
def build_feature_row(age, income, emp_length, grade, ownership,
                      hist_def, int_rate, cred_hist, intent, loan_amnt):
    log_income       = np.log1p(income)
    income_stability = emp_length / (age - 18 + 1)
    credit_seniority = cred_hist / (age - 18 + 1)
    loan_to_income   = loan_amnt / (income + 1)
    grade_num        = GRADE_MAP[grade]
    ownership_num    = OWN_MAP[ownership]
    hist_default     = 1 if hist_def == "Yes" else 0
    loan_pct_income  = loan_amnt / (income + 1)

    intents = ["DEBTCONSOLIDATION", "EDUCATION", "HOMEIMPROVEMENT",
               "MEDICAL", "PERSONAL", "VENTURE"]
    intent_cols = {f"intent_{i}": 1 if i == intent else 0 for i in intents}

    row = {
        "person_age":                  age,
        "person_emp_length":           emp_length,
        "loan_amnt":                   loan_amnt,
        "loan_int_rate":               int_rate,
        "loan_percent_income":         loan_pct_income,
        "cb_person_cred_hist_length":  cred_hist,
        "log_income":                  log_income,
        "income_stability":            income_stability,
        "credit_seniority":            credit_seniority,
        "loan_to_income":              loan_to_income,
        "grade_num":                   grade_num,
        "ownership_num":               ownership_num,
        "hist_default":                hist_default,
        **intent_cols
    }
    return pd.DataFrame([row])[FEATURES]


def get_decision(X_row, grade, income):
    raw   = xgb_calib.predict_proba(X_row)[:, 1].reshape(-1, 1)
    pd_   = platt.predict_proba(raw)[:, 1][0]
    decision = "APPROVED" if pd_ < THRESHOLD else "DECLINED"
    credit_score = int(np.clip(850 - (pd_ * 550), 300, 850))

    if pd_ < 0.05:   band = "Very Low Risk"
    elif pd_ < 0.10: band = "Low Risk"
    elif pd_ < 0.16: band = "Moderate Risk"
    elif pd_ < 0.30: band = "High Risk"
    elif pd_ < 0.60: band = "Very High Risk"
    else:            band = "Extreme Risk"

    # Loan limit
    X_sizer = X_row[SIZER_FEATURES]
    af = float(np.clip(sizer.predict(X_sizer)[0], 0.01, 0.83))
    rs = max(0.0, 1 - pd_ * 2)
    gc = GRADE_CAPS.get(grade, 0.20)
    ff = min(af, gc) * rs
    limit = round(min(ff * income, MAX_LOAN), 0)

    # SHAP
    sv = explainer.shap_values(X_row)[0]
    shap_series = pd.Series(sv, index=FEATURES)
    if decision == "DECLINED":
        top = shap_series.nlargest(3)
        direction = "increasing risk"
    else:
        top = shap_series.nsmallest(3).abs()
        direction = "reducing risk"

    reasons = [
        (feat.replace("_", " ").title(), float(shap_series[feat]), direction)
        for feat in top.index
    ]

    return pd_, decision, credit_score, band, limit, reasons, sv


def draw_gauge(score):
    fig, ax = plt.subplots(figsize=(4, 2.2), subplot_kw={"projection": "polar"})
    fig.patch.set_facecolor("none")
    ax.set_facecolor("none")

    score_pct = (score - 300) / 550
    angle     = np.pi * (1 - score_pct)

    zones = [(0.0, 0.33, "#D85A30"), (0.33, 0.66, "#EF9F27"), (0.66, 1.0, "#1D9E75")]
    for start, end, color in zones:
        theta = np.linspace(np.pi * (1 - end), np.pi * (1 - start), 100)
        ax.fill_between(theta, 0.6, 1.0, color=color, alpha=0.85)

    ax.annotate("", xy=(angle, 0.55), xytext=(angle, 0.0),
                arrowprops=dict(arrowstyle="-|>", color="black", lw=2))
    ax.text(np.pi / 2, 0.2, str(score), ha="center", va="center",
            fontsize=20, fontweight="bold", color="black")

    ax.set_xlim(0, np.pi)
    ax.set_ylim(0, 1.1)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.spines["polar"].set_visible(False)
    ax.set_thetamin(0)
    ax.set_thetamax(180)
    plt.tight_layout(pad=0)
    return fig


def draw_shap_bar(reasons, decision):
    fig, ax = plt.subplots(figsize=(5, 2.5))
    fig.patch.set_facecolor("none")
    ax.set_facecolor("none")
    color = "#D85A30" if decision == "DECLINED" else "#1D9E75"
    labels = [r[0] for r in reasons]
    values = [abs(r[1]) for r in reasons]
    ax.barh(labels, values, color=color, alpha=0.85, edgecolor="white")
    ax.set_xlabel("SHAP contribution", fontsize=9)
    ax.set_title("Top reason codes", fontsize=10, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    return fig


# ── UI ─────────────────────────────────────────────────────────────────────
st.title("🏦 Credit Risk Lending System")
st.caption("Enter applicant details in the sidebar to generate a real-time lending decision.")

st.sidebar.header("Applicant Details")

age        = st.sidebar.slider("Age", 18, 80, 30)
income     = st.sidebar.number_input("Annual Income (£)", 5000, 500000, 55000, step=1000)
emp_length = st.sidebar.slider("Employment Length (years)", 0, 40, 5)
grade      = st.sidebar.selectbox("Loan Grade", ["A","B","C","D","E","F","G"])
ownership  = st.sidebar.selectbox("Home Ownership", ["OWN","MORTGAGE","RENT","OTHER"])
hist_def   = st.sidebar.selectbox("Prior Default on Record", ["No","Yes"])
int_rate   = st.sidebar.slider("Interest Rate (%)", 5.0, 24.0, 11.0, step=0.1)
cred_hist  = st.sidebar.slider("Credit History Length (years)", 2, 30, 5)
intent     = st.sidebar.selectbox("Loan Purpose", [
    "PERSONAL","EDUCATION","MEDICAL","DEBTCONSOLIDATION",
    "HOMEIMPROVEMENT","VENTURE"
])
loan_amnt  = st.sidebar.slider("Loan Amount Requested (£)", 500, 35000, 8000, step=500)

X_row = build_feature_row(age, income, emp_length, grade, ownership,
                           hist_def, int_rate, cred_hist, intent, loan_amnt)

pd_, decision, credit_score, band, limit, reasons, sv = get_decision(X_row, grade, income)

# ── Decision banner ────────────────────────────────────────────────────────
if decision == "APPROVED":
    st.success(f"## APPROVED")
else:
    st.error(f"## DECLINED")

# ── Main metrics ───────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
col1.metric("Calibrated PD", f"{pd_:.1%}")
col2.metric("Credit Score",  f"{credit_score}")
col3.metric("Risk Band",     band)
col4.metric("Loan Limit",    f"£{limit:,.0f}" if decision == "APPROVED" else "N/A")

st.divider()

# ── Gauge and SHAP ─────────────────────────────────────────────────────────
col_a, col_b = st.columns([1, 1])

with col_a:
    st.subheader("Credit Score")
    st.pyplot(draw_gauge(credit_score), use_container_width=True)
    st.caption("300 = highest risk   850 = lowest risk")

with col_b:
    st.subheader("Decision Drivers")
    st.pyplot(draw_shap_bar(reasons, decision), use_container_width=True)
    direction = "pushing toward default" if decision == "DECLINED" else "reducing default risk"
    st.caption(f"Top 3 features {direction} for this applicant")

st.divider()

# ── Loan details ───────────────────────────────────────────────────────────
if decision == "APPROVED":
    st.subheader("Loan Details")
    c1, c2, c3 = st.columns(3)
    c1.metric("Requested Amount", f"£{loan_amnt:,}")
    c2.metric("Approved Limit",   f"£{limit:,.0f}")
    c3.metric("Approval Rate",    f"{min(limit/loan_amnt, 1):.0%}" if loan_amnt > 0 else "N/A")

# ── Reason codes ───────────────────────────────────────────────────────────
st.subheader("Reason Codes")
for feat, val, direction in reasons:
    arrow = "↑" if val > 0 else "↓"
    color = "red" if val > 0 else "green"
    st.markdown(f"**{feat}** {arrow} {direction} &nbsp; `SHAP: {abs(val):.3f}`")

st.divider()
st.caption("Model: XGBoost with Platt calibration | Threshold: 0.16 | AUC: 0.954 | Built with CRISP-DM")
