# 🏥 Healthcare Fraud Detection System - Complete Logic Documentation

## Overview

This document explains the complete logic of how the Healthcare Fraud Detection System works, from patient input to fraud detection output.

---

## 📊 System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         FRONTEND (React)                            │
│                    http://localhost:5173                            │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐     │
│  │    Dashboard    │  │  Analyze Claim  │  │    Analytics    │     │
│  │   (Statistics)  │  │   (Input Form)  │  │    (Charts)     │     │
│  └─────────────────┘  └────────┬────────┘  └─────────────────┘     │
└────────────────────────────────┼────────────────────────────────────┘
                                 │ POST /predict
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         BACKEND (FastAPI)                           │
│                    http://localhost:8000                            │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                    /predict Endpoint                          │  │
│  │  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐  │  │
│  │  │ Disease Price  │  │  Rule-Based    │  │   ML Model     │  │  │
│  │  │    Lookup      │  │  Detection     │  │   Prediction   │  │  │
│  │  └───────┬────────┘  └───────┬────────┘  └───────┬────────┘  │  │
│  │          └──────────────────┼──────────────────┘             │  │
│  └──────────────────────────────┼────────────────────────────────┘  │
└─────────────────────────────────┼────────────────────────────────────┘
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                           DATA LAYER                                │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐     │
│  │  claims.db      │  │ disease_prices  │  │   model.pkl     │     │
│  │  (558K claims)  │  │  (6,016 codes)  │  │  (ML Model)     │     │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 🔄 Step-by-Step Logic Flow

### STEP 1: User Input (Frontend)

When a user fills the "Analyze Claim" form, they provide:

| Field | Example | Purpose |
|-------|---------|---------|
| Provider Type | "Private" | Which type of hospital |
| Provider ID | "PRV55001" | Unique hospital identifier |
| Diagnosis Code | "51881" | ICD-9 code for the disease |
| Amount (₹) | 17000 | Claim amount in rupees |
| Stay (days) | 1 | Days in hospital (0 = outpatient) |
| Diagnoses | 6 | Number of diagnosis codes |
| Patient Age | 65 | Patient's age |
| Chronic Conditions | 2 | Number of chronic conditions |

**Frontend sends this data to:**
```
POST http://localhost:8000/predict
```

---

### STEP 2: Disease Price Lookup (Backend)

The first thing the backend does is look up the **expected price** for this specific disease.

#### 2.1 Load Disease Prices
```python
# At startup, backend loads:
disease_prices.csv → Contains 6,016 diagnosis codes with base prices

Example:
┌──────────────┬─────────────────────────────┬────────────┐
│ Diagnosis    │ Disease Name                │ Base Price │
├──────────────┼─────────────────────────────┼────────────┤
│ 51881        │ Acute Respiratory Failure   │ ₹17,635    │
│ 4019         │ Hypertension               │ ₹3,500     │
│ 486          │ Pneumonia                  │ ₹8,200     │
└──────────────┴─────────────────────────────┴────────────┘
```

#### 2.2 Apply Provider Multiplier
Different hospitals charge different rates:

```python
PROVIDER_MULTIPLIERS = {
    'Government': 0.7,   # 30% cheaper (subsidized)
    'Clinic': 1.0,       # Standard pricing
    'Private': 1.8       # 80% premium
}

# Calculation:
Expected Price = Base Price × Provider Multiplier

Example for Acute Respiratory Failure (51881):
- Base Price: ₹17,635
- Provider: Private (1.8x)
- Expected Price: ₹17,635 × 1.8 = ₹31,743
```

#### 2.3 Calculate Fraud Thresholds

```python
FRAUD_THRESHOLDS = {
    'normal_max': 1.5,     # Up to 1.5x expected = Normal
    'elevated_max': 2.5,   # 1.5x to 2.5x = Elevated
    # Above 2.5x = Suspicious
}

For Private hospital with Expected Price ₹31,743:
- Normal Max: ₹31,743 × 1.5 = ₹47,615
- Elevated Max: ₹31,743 × 2.5 = ₹79,359
```

#### 2.4 Classify Into Zone

```python
def classify_price_zone(claim_amount, expected_price):
    if claim_amount <= expected_price * 1.5:
        return "Normal"      # ✅ Fair profit
    elif claim_amount <= expected_price * 2.5:
        return "Elevated"    # ⚠️ Premium pricing
    else:
        return "Suspicious"  # 🚨 Possible fraud
```

**Example:**
```
Claim Amount: ₹17,000
Expected: ₹31,743
Ratio: 17000 / 31743 = 0.54x
Zone: ✅ NORMAL (below 1.5x)
```

---

### STEP 3: Rule-Based Detection (Layer 1)

Before using ML, the system checks for **obvious fraud patterns**:

#### Rule 1: Disease-Specific Pricing
```python
if price_zone == "Suspicious":  # Above 2.5x expected
    flag_as_fraud("Overpriced claim")
```

#### Rule 2: Excessive Diagnoses (Upcoding)
```python
if num_diagnoses > 15:
    flag_as_fraud("Too many diagnoses - possible upcoding")
```

#### Rule 3: Invalid Age
```python
if patient_age > 120 or patient_age < 0:
    flag_as_fraud("Invalid patient age")
```

#### Rule 4: Age + Chronic Conditions Mismatch
```python
if patient_age < 30 and chronic_conditions > 5:
    flag_as_fraud("Young patient with too many chronic conditions")
```

#### Rule 5: Inpatient Claim with 0 Stay
```python
if claim_type == "Inpatient" and length_of_stay == 0:
    flag_as_fraud("Inpatient claim but no hospital stay")
```

**If any rule triggers with high severity (≥40% probability), return immediately as fraud.**

---

### STEP 4: ML Model Prediction (Layer 2)

If rule-based detection doesn't find obvious fraud, the ML model analyzes patterns.

#### 4.1 Model Details
```
Model Type: Gradient Boosting Classifier
Training Data: 5,410 providers
Accuracy: 94.82%
ROC-AUC: 0.9683
```

#### 4.2 Features Used (28 total)
The model uses provider-level aggregated features:

| Feature | Description |
|---------|-------------|
| total_claims | Number of claims by this provider |
| amount_mean | Average claim amount |
| amount_std | Variation in claim amounts |
| num_diagnoses_mean | Average diagnoses per claim |
| chronic_conditions_sum | Total chronic conditions |
| inpatient_ratio | % of inpatient vs outpatient |
| claims_per_patient | How many claims per patient |
| revenue_per_patient | Average revenue per patient |

#### 4.3 ML Prediction
```python
# Scale features
X_scaled = scaler.transform(features)

# Predict probability
probability = model.predict_proba(X_scaled)[:, 1]

# Example: probability = 0.35 (35% fraud likelihood)
```

---

### STEP 5: Combine Results

The final result combines rule-based and ML predictions:

```python
if rule_triggered:
    combined_prob = (ml_probability + rule_probability) / 2
else:
    combined_prob = ml_probability

# Determine risk level
if combined_prob >= 0.7:
    risk_level = "Critical"
    is_fraud = True
elif combined_prob >= 0.5:
    risk_level = "High"
    is_fraud = True
elif combined_prob >= 0.3:
    risk_level = "Medium"
    is_fraud = False
else:
    risk_level = "Low"
    is_fraud = False
```

---

### STEP 6: Calculate GST

```python
GST_RATE = 0.18  # 18% GST on healthcare services

base_amount = 17000
gst_amount = 17000 × 0.18 = 3060
total_with_gst = 17000 + 3060 = 20060
```

---

### STEP 7: Return Response

The API returns all information to the frontend:

```json
{
  "is_fraud": false,
  "probability": 0.25,
  "risk_level": "Low",
  "short_desc": "Acute respiratry failure",
  "long_desc": "Acute respiratry failure",
  "detection_method": "Provider Comparison (Private)",
  "provider_type": "Private",
  
  "price_zone_info": {
    "zone": "Normal",
    "emoji": "✅",
    "ratio": 0.54,
    "explanation": "₹17,000 is within fair profit range"
  },
  
  "expected_price_info": {
    "base_price": 17635.41,
    "expected_without_gst": 31743.74,
    "max_normal": 47615.61,
    "max_elevated": 79359.35
  },
  
  "gst_info": {
    "base_amount": 17000,
    "gst_rate": "18%",
    "gst_amount": 3060,
    "total_with_gst": 20060
  },
  
  "benchmark_info": {
    "provider_type": "Private",
    "expected_average": 10000,
    "p95_threshold": 40000
  }
}
```

---

## 📈 Visual Summary

```
USER INPUT                    PROCESSING                        OUTPUT
───────────                   ──────────                        ──────

┌─────────────┐
│ Diagnosis:  │
│ 51881       │──┐
├─────────────┤  │     ┌──────────────────────┐
│ Amount:     │  │     │   STEP 1: Lookup     │
│ ₹17,000     │──┼────►│   Base Price:        │
├─────────────┤  │     │   ₹17,635            │
│ Provider:   │  │     └──────────┬───────────┘
│ Private     │──┘                │
└─────────────┘                   ▼
                        ┌──────────────────────┐
                        │   STEP 2: Apply      │
                        │   Multiplier (1.8x)  │
                        │   Expected: ₹31,743  │
                        └──────────┬───────────┘
                                   │
                                   ▼
                        ┌──────────────────────┐     ┌─────────────────┐
                        │   STEP 3: Compare    │     │   RESULT:       │
                        │   ₹17,000 vs ₹31,743 │────►│   ✅ NORMAL     │
                        │   Ratio: 0.54x       │     │   Zone          │
                        └──────────┬───────────┘     └─────────────────┘
                                   │
                                   ▼
                        ┌──────────────────────┐     ┌─────────────────┐
                        │   STEP 4: ML Model   │     │   Risk: LOW     │
                        │   Additional check   │────►│   Prob: 25%     │
                        └──────────┬───────────┘     └─────────────────┘
                                   │
                                   ▼
                        ┌──────────────────────┐     ┌─────────────────┐
                        │   STEP 5: Add GST    │     │   Total:        │
                        │   18% = ₹3,060       │────►│   ₹20,060       │
                        └──────────────────────┘     └─────────────────┘
```

---

## 📁 Key Files

| File | Purpose |
|------|---------|
| `backend/main.py` | API endpoints, fraud detection logic |
| `backend/database.py` | SQLAlchemy models, DB connection |
| `backend/icd_lookup.py` | Disease name lookup from ICD codes |
| `data/disease_prices.csv` | 6,016 diagnosis codes with prices |
| `data/claims.csv` | 558,211 claims dataset |
| `data/claims.db` | SQLite database |
| `ml/model.pkl` | Trained ML model |
| `frontend/src/App.jsx` | React UI components |

---

## 🎯 Summary

1. **User enters claim details** → Frontend sends to API
2. **Lookup disease price** → From 6,016 pre-calculated prices
3. **Apply provider multiplier** → Govt 0.7x, Clinic 1.0x, Private 1.8x
4. **Classify into zone** → Normal (≤1.5x), Elevated (≤2.5x), Suspicious (>2.5x)
5. **Run rule-based checks** → Catch obvious fraud patterns
6. **Run ML model** → Detect subtle fraud patterns
7. **Combine results** → Final fraud probability and risk level
8. **Calculate GST** → 18% on base amount
9. **Return response** → Complete breakdown to frontend
