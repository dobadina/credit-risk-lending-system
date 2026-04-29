# Credit Risk Lending System

A data-driven lending decision system that predicts the probability of borrower default, determines how much to lend, and produces auditable approve/decline decisions with plain-English explanations.

Built as a portfolio project following the CRISP-DM methodology using a real-world credit dataset of 32,000 loan records.

---

## The Problem

Most lending decisions are driven by rigid scoring rules that treat all borrowers the same. A 28-year-old renter taking a medical loan carries very different risk from a 45-year-old homeowner consolidating debt, but a simple cutoff score might treat them identically.

This project builds a system that answers three questions for any new applicant:

1. Should we lend to this person at all?
2. If yes, how much should we lend them?
3. Why did the model make that decision?

---

## What the System Does

**Default prediction model**
An XGBoost classifier trained on 19 engineered features achieves AUC-ROC of 0.954, Gini of 0.908, and KS of 0.775 on a held-out test set. Probability scores are calibrated using Platt scaling so that a score of 0.15 genuinely represents a 15% chance of default.

**Threshold optimisation**
The approval threshold is set at 0.16 using a profit maximisation framework rather than the naive 0.50 default. With assumed unit economics of 800 pounds revenue per good loan and 4,000 pounds loss per default, the optimal threshold approves 73.8% of applicants while maximising expected portfolio profit.

**Loan sizing model**
An XGBoost regressor trained only on successfully repaid loans predicts the maximum sustainable loan as a fraction of income. A three layer structure applies the result: affordability model output, compressed by a risk scalar based on default probability, capped by grade-based policy limits.

**Explainability**
Every decision includes SHAP-derived reason codes identifying the top three features driving the outcome. A declined applicant can be told specifically which factors raised their risk score. This satisfies GDPR Article 22 requirements for meaningful explanations of automated decisions.

**Monitoring**
A Population Stability Index implementation detects when the applicant population has shifted enough to warrant model review. PSI on a simulated time split returns 0.028, well within the stable threshold of 0.10.

---

## Key Findings

- Loan to income ratio is the strongest predictor of default, with a mean SHAP value of 1.276
- Loan grade is the second strongest predictor. Grade A borrowers default at 10%. Grade G at 98%.
- Historical default on file, despite appearing significant in exploratory analysis, contributes almost nothing once other financial features are known (SHAP: 0.008)
- The optimal approval threshold of 0.16 is derived from unit economics, not from model outputs. Using 0.50 would leave significant profit on the table.
- The dataset has no origination timestamps. Macro-economic signals are therefore applied as a policy overlay rather than model features, which is the architecturally correct response to this constraint.

---

## Project Structure

credit-risk-lending-system/
├── notebooks/
│   ├── 01_data_understanding_and_preparation.ipynb
│   ├── 02_modelling.ipynb
│   ├── 03_evaluation.ipynb
│   └── 04_deployment.ipynb
├── models/
│   ├── xgb_champion.pkl
│   ├── xgb_base_for_calibration.pkl
│   ├── platt_scaler.pkl
│   ├── xgb_loan_sizer.pkl
│   └── features.json
├── outputs/
│   └── figures/
├── data/
│   ├── raw/
│   └── processed/
└── README.md

---

## Methodology

This project follows the CRISP-DM framework:

| Phase | Notebook | Key outputs |
|---|---|---|
| Business Understanding | 01 | Problem framing, success criteria |
| Data Understanding | 01 | EDA, distributions, bivariate analysis, correlation |
| Data Preparation | 01 | Cleaning, imputation, feature engineering |
| Modelling | 02 | Four model comparison, hyperparameter tuning |
| Evaluation | 03 | Calibration, SHAP, threshold optimisation, score bands |
| Deployment | 04 | Loan sizer, PSI monitoring, model card |

---

## Model Performance

| Model | AUC-ROC | Gini | KS | Train Time |
|---|---|---|---|---|
| Logistic Regression | 0.876 | 0.752 | 0.592 | 0.3s |
| Random Forest | 0.935 | 0.870 | 0.738 | 13.4s |
| XGBoost | 0.951 | 0.902 | 0.764 | 1.2s |
| LightGBM | 0.948 | 0.897 | 0.763 | 2.0s |
| XGBoost tuned | 0.954 | 0.908 | 0.775 | 3.6s |

---

## Limitations

- No origination dates in the dataset, so macro-economic conditions cannot be incorporated into the model directly
- Loan sizer R2 of 0.19 reflects low variance in repayment fractions among good borrowers
- Fairness audit across protected demographic groups has not been conducted
- Model has not been validated across different economic cycles

---

## Tech Stack

Python, XGBoost, scikit-learn, SHAP, pandas, NumPy, matplotlib, seaborn, Google Colab, Git
