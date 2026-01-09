# Healthcare Provider Fraud Detection

This project predicts potentially fraudulent healthcare providers using Medicare‑style claims data and machine learning. By analysing inpatient, outpatient, and beneficiary information, it flags suspicious providers for further investigation and helps reduce financial losses in the healthcare system.[web:152][web:205]

## Project overview

- **Goal:** Predict whether a medical provider is potentially fraudulent based on the claims they file.[web:150][web:167]  
- **Problem type:** Binary classification at provider level (`fraud` vs `non‑fraud`).  
- **Tech stack:** Python, pandas, NumPy, scikit‑learn, Jupyter Notebook.  
- **Core tasks:** Data understanding, provider‑level feature engineering, model training, evaluation, and model interpretation using feature importance.[web:163][web:181]  

This project is inspired by the Kaggle “Healthcare Provider Fraud Detection Analysis” dataset and follows a practical workflow used in fraud analytics.[web:152][web:167]

## Data description

The dataset (Kaggle: Healthcare Provider Fraud Detection Analysis) contains 8 CSV files split into train and test sets.[web:152][web:155]

### Training files

- **Train labels** (`Train-...csv`):  
  - One row per provider.  
  - Columns: `Provider`, `PotentialFraud` (Yes = potentially fraudulent, No = not fraudulent).[web:156][web:150]  

- **Train_Inpatientdata** (`Train_Inpatientdata-...csv`):  
  - Inpatient claim records for admitted patients.  
  - Key fields: `Provider`, `BeneID`, `ClaimID`, `ClaimStartDt`, `ClaimEndDt`, `InscClaimAmtReimbursed`, `AdmissionDt`, `DischargeDt`, diagnosis codes, procedure codes, physician identifiers.[web:160][web:174]  

- **Train_Outpatientdata** (`Train_Outpatientdata-...csv`):  
  - Outpatient claim records for non‑admitted visits.  
  - Fields: `Provider`, `BeneID`, `ClaimID`, `InscClaimAmtReimbursed`, diagnosis and procedure codes, deductibles, physician information.[web:160][web:174]  

- **Train_Beneficiarydata** (`Train_Beneficiarydata-...csv`):  
  - Beneficiary demographics and health information.  
  - Fields: `BeneID`, `Gender`, `Race`, coverage months, chronic condition indicators, and annual inpatient/outpatient reimbursement and deductible amounts (for example, `IPAnnualReimbursementAmt`, `OPAnnualReimbursementAmt`).[web:160][web:178]  

The test files have the same structure but omit the fraud label; they are used for scoring and simulated deployment.[web:152][web:160]

Each provider appears in many claim records and is associated with multiple beneficiaries, so the modeling target is the provider, not individual claims.[web:150][web:162]

## Approach

### 1. Feature engineering (provider level)

The raw claim and beneficiary tables are aggregated to one row per provider.

**Inpatient features**

From `Train_Inpatientdata`:

- Number of inpatient claims per provider.  
- Total and average inpatient reimbursed amount.  
- Number of unique admission and discharge dates.  
- Number of unique attending, operating, and other physicians.  

These features capture how intensively each provider uses inpatient services and how much money is reimbursed through those claims.[web:163][web:181]

**Outpatient features**

From `Train_Outpatientdata`:

- Number of outpatient claims per provider.  
- Total and average outpatient reimbursed amount.  
- Number of unique attending, operating, and other physicians.  

These features reflect outpatient claim volume, cost, and provider‑physician diversity.[web:163][web:181]

**Beneficiary features**

From `Train_Beneficiarydata`, linked via `BeneID`:

- Number of unique beneficiaries per provider.  
- Average Part A and Part B coverage months.  
- Mean values of chronic condition flags:
  - Alzheimer, heart failure, kidney disease, cancer, obstructive pulmonary disease, depression, diabetes, ischemic heart disease, osteoporosis, rheumatoid arthritis, stroke.  
- Average annual inpatient and outpatient reimbursement and deductible amounts per beneficiary (`IPAnnualReimbursementAmt`, `OPAnnualReimbursementAmt`, `IPAnnualDeductibleAmt`, `OPAnnualDeductibleAmt`).[web:160][web:178]  

All aggregates are merged into a single training table:

```text
Provider | inpatient features | outpatient features | beneficiary features | PotentialFraud | Target_bin
