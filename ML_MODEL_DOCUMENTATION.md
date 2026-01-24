# 🤖 ML Model - Complete Technical Documentation

## Overview

This document explains how the Machine Learning model in the Healthcare Fraud Detection System works, from data preparation to final prediction.

---

## 📊 Model Summary

| Property | Value |
|----------|-------|
| **Model Type** | Gradient Boosting Classifier |
| **Training Data** | 5,410 providers (aggregated from 558,211 claims) |
| **Test Accuracy** | 94.82% |
| **ROC-AUC Score** | 0.9683 |
| **Features Used** | 28 |
| **Model File** | `ml/model.pkl` |

---

## 🔄 Complete ML Pipeline

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        ML TRAINING PIPELINE                             │
└─────────────────────────────────────────────────────────────────────────┘

STEP 1: Raw Data                 STEP 2: Aggregation              STEP 3: Features
┌──────────────────┐            ┌──────────────────┐            ┌──────────────────┐
│ claims.csv       │            │ Provider-Level   │            │ 28 Features      │
│ 558,211 claims   │───────────►│ Statistics       │───────────►│ Calculated       │
│                  │            │ 5,410 providers  │            │                  │
└──────────────────┘            └──────────────────┘            └──────────────────┘
                                                                         │
                                                                         ▼
STEP 6: Save                     STEP 5: Evaluate              STEP 4: Train/Test
┌──────────────────┐            ┌──────────────────┐            ┌──────────────────┐
│ model.pkl        │◄───────────│ Accuracy: 94.82% │◄───────────│ 80% Train        │
│ (Saved Model)    │            │ ROC-AUC: 0.9683  │            │ 20% Test         │
└──────────────────┘            └──────────────────┘            └──────────────────┘
```

---

## STEP 1: Raw Data Loading

### 1.1 Source Data
```python
# Load the claims dataset
df = pd.read_csv('data/claims.csv')

# Dataset Shape: 558,211 rows × 15 columns
```

### 1.2 Key Columns
| Column | Type | Example | Description |
|--------|------|---------|-------------|
| claim_id | int | 12345 | Unique claim ID |
| provider_id | str | "PRV55001" | Hospital identifier |
| patient_id | str | "PAT10001" | Patient identifier |
| diagnosis_code | str | "51881" | ICD-9 disease code |
| claim_type | str | "Inpatient" | Inpatient or Outpatient |
| amount | float | 17000.0 | Claim amount (₹) |
| num_diagnoses | int | 3 | Number of diagnoses |
| num_procedures | int | 2 | Number of procedures |
| length_of_stay | int | 5 | Days in hospital |
| patient_age | int | 65 | Patient's age |
| chronic_conditions | int | 2 | Number of chronic conditions |
| **is_fraud** | bool | True/False | **TARGET VARIABLE** |

---

## STEP 2: Provider-Level Aggregation

### 2.1 Why Aggregate?

**Problem:** Individual claims don't have enough info to detect fraud
**Solution:** Look at provider patterns across ALL their claims

```python
# A fraudulent provider might:
# - Submit many high-amount claims
# - Have unusually high number of diagnoses
# - Have consistent overbilling patterns

# By aggregating, we can detect these patterns
```

### 2.2 Aggregation Code
```python
provider_stats = df.groupby('provider_id').agg({
    'claim_id': 'count',                    # Total claims
    'patient_id': 'nunique',                # Unique patients
    'amount': ['mean', 'sum', 'std', 'max', 'min'],
    'deductible': ['mean', 'sum'],
    'amount_per_diagnosis': ['mean', 'max'],
    'num_diagnoses': ['mean', 'sum', 'max'],
    'num_procedures': ['mean', 'sum', 'max'],
    'length_of_stay': ['mean', 'sum', 'max'],
    'patient_age': ['mean', 'std'],
    'chronic_conditions': ['mean', 'sum'],
    'is_fraud': 'first'                     # Provider's fraud label
}).reset_index()
```

### 2.3 Result
```
Individual Claims:        558,211 rows
After Aggregation:        5,410 rows (one per provider)
```

---

## STEP 3: Feature Engineering

### 3.1 All 28 Features

| # | Feature Name | Description | Why Important |
|---|--------------|-------------|---------------|
| 1 | total_claims | Count of claims | Fraudulent providers submit many claims |
| 2 | unique_patients | Unique patient count | Low diversity = suspicious |
| 3 | amount_mean | Average claim amount | High = potential overbilling |
| 4 | amount_sum | Total revenue | Unusually high = red flag |
| 5 | amount_std | Variance in amounts | Low variance = suspicious patterns |
| 6 | amount_max | Maximum single claim | Extreme outliers |
| 7 | amount_min | Minimum single claim | Unusual floor values |
| 8 | deductible_mean | Avg deductible | Manipulation indicator |
| 9 | deductible_sum | Total deductibles | Pattern detection |
| 10 | amount_per_diagnosis_mean | ₹ per diagnosis avg | Overbilling per diagnosis |
| 11 | amount_per_diagnosis_max | Max ₹ per diagnosis | Extreme cases |
| 12 | num_diagnoses_mean | Avg diagnoses/claim | Upcoding detection |
| 13 | num_diagnoses_sum | Total diagnoses | Volume indicator |
| 14 | num_diagnoses_max | Max diagnoses | Extreme upcoding |
| 15 | num_procedures_mean | Avg procedures | Unnecessary procedures |
| 16 | num_procedures_sum | Total procedures | Volume indicator |
| 17 | num_procedures_max | Max procedures | Extreme cases |
| 18 | length_of_stay_mean | Avg hospital days | Extended stays = more billing |
| 19 | length_of_stay_sum | Total hospital days | Pattern detection |
| 20 | length_of_stay_max | Max stay | Extreme cases |
| 21 | patient_age_mean | Avg patient age | Target demographic patterns |
| 22 | patient_age_std | Age variance | Diversity indicator |
| 23 | chronic_conditions_mean | Avg chronic conditions | Complexity indicator |
| 24 | chronic_conditions_sum | Total chronic conditions | Pattern detection |
| 25 | claims_per_patient | Claims ÷ Patients | Repeat claim patterns |
| 26 | avg_diagnoses_per_claim | Diagnoses ÷ Claims | Upcoding ratio |
| 27 | revenue_per_patient | Amount ÷ Patients | Revenue efficiency |
| 28 | inpatient_ratio | % Inpatient claims | Service mix indicator |

### 3.2 Derived Features
```python
# Claims per patient (repeat visits)
provider_stats['claims_per_patient'] = (
    provider_stats['total_claims'] / 
    provider_stats['unique_patients']
)

# Average diagnoses per claim (upcoding indicator)
provider_stats['avg_diagnoses_per_claim'] = (
    provider_stats['num_diagnoses_sum'] / 
    provider_stats['total_claims']
)

# Revenue per patient (value extraction)
provider_stats['revenue_per_patient'] = (
    provider_stats['amount_sum'] / 
    provider_stats['unique_patients']
)

# Inpatient ratio (service mix)
provider_stats['inpatient_ratio'] = (
    inpatient_count / total_claims
)
```

---

## STEP 4: Train/Test Split

### 4.1 Data Preparation
```python
# Separate features (X) and target (y)
X = provider_stats[feature_cols]  # 28 features
y = provider_stats['is_fraud']    # 0 or 1

# Split: 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2,      # 20% for testing
    random_state=42,    # Reproducibility
    stratify=y          # Maintain fraud ratio
)
```

### 4.2 Split Statistics
```
Total Providers:    5,410
Training Set:       4,328 (80%)
Testing Set:        1,082 (20%)
Fraud Ratio:        ~38% (maintained in both sets)
```

### 4.3 Feature Scaling
```python
# Standardize features (mean=0, std=1)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Why scale?
# - Gradient Boosting works better with normalized features
# - Features have different scales (amount_sum in millions, ratios 0-1)
# - Prevents features with large values from dominating
```

---

## STEP 5: Model Training

### 5.1 Model Selection: Gradient Boosting

**Why Gradient Boosting?**
| Advantage | Explanation |
|-----------|-------------|
| Handles mixed feature types | Works with both numeric and categorical |
| Resistant to overfitting | Built-in regularization |
| Captures complex patterns | Non-linear decision boundaries |
| Feature importance | Tells us which features matter most |
| High accuracy | State-of-the-art for tabular data |

### 5.2 Model Configuration
```python
model = GradientBoostingClassifier(
    n_estimators=150,      # Number of trees
    learning_rate=0.1,     # Step size shrinkage
    max_depth=5,           # Maximum tree depth
    min_samples_split=10,  # Minimum samples to split
    min_samples_leaf=5,    # Minimum samples in leaf
    random_state=42        # Reproducibility
)
```

### 5.3 Training Process
```python
# Fit model to training data
model.fit(X_train_scaled, y_train)

# What happens during training:
# 1. First tree learns main patterns
# 2. Calculate errors (residuals)
# 3. Next tree focuses on errors
# 4. Repeat 150 times
# 5. Combine all trees for final prediction
```

### 5.4 How Gradient Boosting Works (Visual)
```
Data ──► Tree 1 ──► Errors ──► Tree 2 ──► Errors ──► ... ──► Tree 150
         │                      │                              │
         ▼                      ▼                              ▼
      Prediction 1         Prediction 2               Prediction 150
         │                      │                              │
         └──────────────────────┴─────────── + ────────────────┘
                                    │
                                    ▼
                           Final Prediction
                        (Weighted Sum of All Trees)
```

---

## STEP 6: Model Evaluation

### 6.1 Accuracy Metrics
```
Training Accuracy:  100.00%
Testing Accuracy:   94.82%
Difference:         5.18% (slight overfitting, acceptable)

Training ROC-AUC:   1.0000
Testing ROC-AUC:    0.9683
```

### 6.2 Cross-Validation (5-Fold)
```
Fold 1: 95.10%
Fold 2: 94.82%
Fold 3: 93.53%
Fold 4: 93.25%
Fold 5: 93.99%
────────────────
Mean:   94.14% ± 1.43%
```

### 6.3 Confusion Matrix
```
                    Predicted
                  Non-Fraud  Fraud
Actual Non-Fraud    620       38     ← False Positive (38)
       Fraud         18      406     ← True Positive (406)
                     │
                     └─ False Negative (18)

Precision: 91.4% (of predicted fraud, how many are actually fraud)
Recall:    95.8% (of actual fraud, how many did I catch)
F1-Score:  93.5% (harmonic mean of precision and recall)
```

### 6.4 Feature Importance (Top 10)
```
┌────────────────────────────┬────────────┐
│ Feature                    │ Importance │
├────────────────────────────┼────────────┤
│ amount_sum                 │   0.142    │
│ revenue_per_patient        │   0.118    │
│ amount_mean                │   0.095    │
│ claims_per_patient         │   0.087    │
│ total_claims               │   0.076    │
│ chronic_conditions_sum     │   0.068    │
│ num_diagnoses_sum          │   0.062    │
│ amount_per_diagnosis_mean  │   0.058    │
│ inpatient_ratio            │   0.054    │
│ length_of_stay_mean        │   0.048    │
└────────────────────────────┴────────────┘
```

---

## STEP 7: Saving the Model

### 7.1 What Gets Saved
```python
model_artifacts = {
    'model': model,           # Trained classifier
    'scaler': scaler,         # Feature scaler
    'feature_cols': feature_cols,  # Feature names
    'training_date': datetime.now(),
    'accuracy': test_accuracy,
    'roc_auc': test_roc_auc
}

joblib.dump(model_artifacts, 'ml/model.pkl')
```

### 7.2 Model File Contents
```
ml/model.pkl (1.0 MB)
├── model          → GradientBoostingClassifier (150 trees)
├── scaler         → StandardScaler (28 features)
├── feature_cols   → List of 28 feature names
├── accuracy       → 0.9482
└── roc_auc        → 0.9683
```

---

## 🔮 Prediction Flow (At Runtime)

### When a New Claim Arrives:

```python
# 1. Load saved model
model_artifacts = joblib.load('ml/model.pkl')
model = model_artifacts['model']
scaler = model_artifacts['scaler']

# 2. Build features from claim
# (Since we trained on provider aggregates,
#  we simulate single-claim as a provider)
features = {
    'total_claims': 1,
    'amount_mean': claim.amount,
    'amount_sum': claim.amount,
    # ... all 28 features
}

# 3. Scale features
X = scaler.transform([features])

# 4. Predict
probability = model.predict_proba(X)[0][1]  # Fraud probability

# 5. Classify
if probability >= 0.7:
    result = "Critical Risk"
elif probability >= 0.5:
    result = "High Risk"
elif probability >= 0.3:
    result = "Medium Risk"
else:
    result = "Low Risk"
```

---

## 📈 Model Performance Visualization

```
                ROC Curve
        ┌────────────────────────┐
    1.0 │            .---------│
        │          .'          │
    0.8 │        .'            │
        │      .'              │
TPR 0.6 │    .'                │
        │  .'  Our Model       │
    0.4 │.'    AUC = 0.9683    │
        │                      │
    0.2 │                      │
        │   Random (AUC=0.5)   │
    0.0 │──────────────────────│
        0.0  0.2  0.4  0.6  0.8  1.0
                  FPR


        Accuracy Timeline (Cross-Validation)
        ┌────────────────────────────────────┐
   96%  │     │                              │
        │  █  │     █                        │
   95%  │  █  │  █  █     █                 │
        │  █  │  █  █     █     █           │
   94%  │  █  │  █  █     █     █     █     │
        │  █  │  █  █  █  █  █  █  █  █     │
   93%  │  █  │  █  █  █  █  █  █  █  █     │
        └──────────────────────────────────┘
          F1   F2   F3   F4   F5   Mean
```

---

## 🎯 Summary

| Step | What Happens | Output |
|------|--------------|--------|
| 1 | Load 558,211 claims | Raw data |
| 2 | Aggregate by provider | 5,410 providers |
| 3 | Engineer 28 features | Feature matrix |
| 4 | Split 80/20 | Train & Test sets |
| 5 | Train Gradient Boosting | 150 trees |
| 6 | Evaluate model | 94.82% accuracy |
| 7 | Save model | model.pkl (1 MB) |
| 8 | Runtime prediction | Fraud probability 0-100% |

---

## 📁 Files

| File | Size | Purpose |
|------|------|---------|
| `ml/train_model.py` | 15 KB | Training script |
| `ml/model.pkl` | 1.0 MB | Saved model + scaler |
| `data/claims.csv` | 40 MB | Training data |

---

## 📚 COMPLETE FUNCTION DEFINITIONS

This section provides detailed definitions of every function used in the ML model.

---

### 🔵 Function 1: `create_provider_features(df)`

**Location:** `ml/train_model.py` (Lines 57-238)

**Purpose:** Aggregates individual claims into provider-level statistics. This is the KEY function that improves accuracy from 62% to 94%.

#### Signature
```python
def create_provider_features(df: pd.DataFrame) -> pd.DataFrame:
```

#### Parameters
| Parameter | Type | Description |
|-----------|------|-------------|
| `df` | `pandas.DataFrame` | Claims data with columns: `provider_id`, `patient_id`, `amount`, `deductible`, `num_diagnoses`, `num_procedures`, `length_of_stay`, `patient_age`, `chronic_conditions`, `is_fraud`, `claim_type` |

#### Returns
| Type | Description |
|------|-------------|
| `pandas.DataFrame` | One row per provider with 28 aggregate features |

#### Internal Steps

**Step 1: Aggregate Claims by Provider**
```python
provider_stats = df.groupby('provider_id').agg({
    'claim_id': 'count',           # Total number of claims
    'patient_id': 'nunique',       # Unique patients treated
    'amount': ['mean', 'sum', 'std', 'max', 'min'],
    'deductible': ['mean', 'sum'],
    'amount_per_diagnosis': ['mean', 'max'],
    'num_diagnoses': ['mean', 'sum', 'max'],
    'num_procedures': ['mean', 'sum', 'max'],
    'length_of_stay': ['mean', 'sum', 'max'],
    'patient_age': ['mean', 'std'],
    'chronic_conditions': ['mean', 'sum'],
    'is_fraud': 'first'
}).reset_index()
```

**Step 2: Flatten Column Names**
```python
# ('amount', 'mean') becomes 'amount_mean'
provider_stats.columns = [
    '_'.join(col).strip('_') if isinstance(col, tuple) else col 
    for col in provider_stats.columns
]
```

**Step 3: Create Derived Features**
```python
# Claims per patient (fraud indicator - high = suspicious)
provider_stats['claims_per_patient'] = (
    provider_stats['total_claims'] / 
    provider_stats['unique_patients'].replace(0, 1)
)

# Revenue per patient (fraud indicator - high = suspicious)
provider_stats['revenue_per_patient'] = (
    provider_stats['amount_sum'] / 
    provider_stats['unique_patients'].replace(0, 1)
)

# Average diagnoses per claim (upcoding indicator)
provider_stats['avg_diagnoses_per_claim'] = (
    provider_stats['num_diagnoses_sum'] / 
    provider_stats['total_claims']
)
```

**Step 4: Calculate Inpatient Ratio**
```python
claim_type_dist = df.groupby(['provider_id', 'claim_type']).size().unstack(fill_value=0)
claim_type_dist['inpatient_ratio'] = (
    claim_type_dist['Inpatient'] / 
    (claim_type_dist['Inpatient'] + claim_type_dist['Outpatient'])
)
```

#### Example Usage
```python
>>> df = pd.read_csv('data/claims.csv')
>>> provider_df = create_provider_features(df)
>>> print(provider_df.shape)
(5410, 29)  # 5,410 providers with 29 columns (28 features + 1 target)
```

---

### 🔵 Function 2: `train_enhanced_model()`

**Location:** `ml/train_model.py` (Lines 245-543)

**Purpose:** Main entry point for training the fraud detection model.

#### Signature
```python
def train_enhanced_model() -> None:
```

#### Parameters
None (uses hardcoded file paths)

#### Returns
None (saves model to `ml/model.pkl`)

#### Internal Steps

**Step 1: Load Data**
```python
df = pd.read_csv("data/claims.csv")
# Loads 558,211 claims
```

**Step 2: Create Provider Features**
```python
provider_df = create_provider_features(df)
# Aggregates to 5,410 providers
```

**Step 3: Prepare X and y**
```python
feature_cols = [c for c in provider_df.columns if c not in ['provider_id', 'is_fraud']]
X = provider_df[feature_cols].fillna(0).replace([np.inf, -np.inf], 0)
y = provider_df['is_fraud']
```

**Step 4: Train/Test Split**
```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

**Step 5: Feature Scaling**
```python
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

**Step 6: Train Gradient Boosting**
```python
clf = GradientBoostingClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    min_samples_split=10,
    min_samples_leaf=5,
    subsample=0.8,
    random_state=42
)
clf.fit(X_train_scaled, y_train)
```

**Step 7: Train Random Forest**
```python
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=5,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train_scaled, y_train)
```

**Step 8: Evaluate & Select Best**
```python
clf_auc = roc_auc_score(y_test, clf.predict_proba(X_test_scaled)[:, 1])
rf_auc = roc_auc_score(y_test, rf.predict_proba(X_test_scaled)[:, 1])
best_model = clf if clf_auc > rf_auc else rf
```

**Step 9: Save Model**
```python
model_artifacts = {
    'model': best_model,
    'scaler': scaler,
    'feature_cols': feature_cols,
    'le_diagnosis': le_diagnosis,
    'le_claim_type': le_claim_type,
    'model_type': 'provider_level'
}
joblib.dump(model_artifacts, 'ml/model.pkl')
```

---

## 📚 SKLEARN LIBRARY FUNCTIONS USED

### `train_test_split()`
**Module:** `sklearn.model_selection`

**Purpose:** Splits data into training and testing sets.

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,                    # Features (DataFrame or array)
    y,                    # Target (Series or array)
    test_size=0.2,        # 20% for testing
    random_state=42,      # Seed for reproducibility
    stratify=y            # Maintain class distribution
)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `X` | array-like | Feature matrix |
| `y` | array-like | Target vector |
| `test_size` | float | Fraction for test set (0.0 to 1.0) |
| `random_state` | int | Random seed for reproducibility |
| `stratify` | array-like | If set, maintains class proportions |

---

### `StandardScaler()`
**Module:** `sklearn.preprocessing`

**Purpose:** Normalizes features to have mean=0 and std=1.

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # Fit + transform
X_test_scaled = scaler.transform(X_test)         # Only transform
```

| Method | Description |
|--------|-------------|
| `fit(X)` | Computes mean and std from X |
| `transform(X)` | Applies normalization: (X - mean) / std |
| `fit_transform(X)` | Does both fit and transform |

**Formula:**
```
z = (x - μ) / σ

Where:
  x = original value
  μ = mean of feature
  σ = standard deviation
  z = normalized value (mean=0, std=1)
```

---

### `GradientBoostingClassifier()`
**Module:** `sklearn.ensemble`

**Purpose:** Builds an ensemble of decision trees sequentially, where each tree corrects errors of the previous ones.

```python
from sklearn.ensemble import GradientBoostingClassifier

model = GradientBoostingClassifier(
    n_estimators=200,      # Number of trees
    learning_rate=0.1,     # Shrinkage (smaller = more robust)
    max_depth=6,           # Max depth per tree
    min_samples_split=10,  # Min samples to split node
    min_samples_leaf=5,    # Min samples in leaf
    subsample=0.8,         # Fraction of samples per tree
    random_state=42        # Reproducibility
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_estimators` | int | 100 | Number of boosting stages (trees) |
| `learning_rate` | float | 0.1 | Step size shrinkage |
| `max_depth` | int | 3 | Max depth of individual trees |
| `min_samples_split` | int | 2 | Min samples required to split |
| `min_samples_leaf` | int | 1 | Min samples in a leaf node |
| `subsample` | float | 1.0 | Fraction of samples for each tree |
| `random_state` | int | None | Seed for reproducibility |

| Method | Description |
|--------|-------------|
| `fit(X, y)` | Train the model |
| `predict(X)` | Predict class labels |
| `predict_proba(X)` | Predict class probabilities |
| `feature_importances_` | Get feature importance scores |

---

### `RandomForestClassifier()`
**Module:** `sklearn.ensemble`

**Purpose:** Builds an ensemble of decision trees in parallel, each trained on random subsets of data.

```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(
    n_estimators=200,          # Number of trees
    max_depth=10,              # Max depth per tree
    min_samples_split=5,       # Min samples to split
    class_weight='balanced',   # Handle imbalanced classes
    random_state=42,           # Reproducibility
    n_jobs=-1                  # Use all CPU cores
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_estimators` | int | 100 | Number of trees in forest |
| `max_depth` | int | None | Max depth of trees |
| `min_samples_split` | int | 2 | Min samples to split |
| `class_weight` | str/dict | None | Weights for classes |
| `random_state` | int | None | Seed for reproducibility |
| `n_jobs` | int | 1 | CPU cores to use (-1 = all) |

---

### `accuracy_score()`
**Module:** `sklearn.metrics`

**Purpose:** Calculates the fraction of correct predictions.

```python
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_true, y_pred)
# Returns: 0.9482 (94.82%)
```

**Formula:**
```
Accuracy = (TP + TN) / (TP + TN + FP + FN)

Where:
  TP = True Positives (correctly predicted fraud)
  TN = True Negatives (correctly predicted legitimate)
  FP = False Positives (legitimate flagged as fraud)
  FN = False Negatives (fraud missed)
```

---

### `roc_auc_score()`
**Module:** `sklearn.metrics`

**Purpose:** Calculates Area Under the ROC Curve - measures ranking ability.

```python
from sklearn.metrics import roc_auc_score

auc = roc_auc_score(y_true, y_prob)
# Returns: 0.9683
```

| Score | Interpretation |
|-------|----------------|
| 1.0 | Perfect classifier |
| 0.9+ | Excellent |
| 0.8-0.9 | Good |
| 0.7-0.8 | Fair |
| 0.5 | Random guess |
| <0.5 | Worse than random |

---

### `classification_report()`
**Module:** `sklearn.metrics`

**Purpose:** Generates precision, recall, F1-score report.

```python
from sklearn.metrics import classification_report

print(classification_report(y_true, y_pred, target_names=['Legitimate', 'Fraudulent']))
```

**Output:**
```
              precision    recall  f1-score   support

  Legitimate       0.97      0.94      0.96       658
  Fraudulent       0.91      0.96      0.93       424

    accuracy                           0.95      1082
   macro avg       0.94      0.95      0.94      1082
weighted avg       0.95      0.95      0.95      1082
```

| Metric | Formula | Description |
|--------|---------|-------------|
| Precision | TP / (TP + FP) | Of predicted positives, how many are correct |
| Recall | TP / (TP + FN) | Of actual positives, how many were caught |
| F1-Score | 2 × (P × R) / (P + R) | Harmonic mean of precision & recall |

---

### `confusion_matrix()`
**Module:** `sklearn.metrics`

**Purpose:** Creates a matrix showing prediction vs actual counts.

```python
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_true, y_pred)
# Returns:
# [[TN, FP],
#  [FN, TP]]
```

**Visual:**
```
                    Predicted
                  Legit    Fraud
Actual Legit    [  TN  ,   FP  ]    ← Non-fraud cases
       Fraud    [  FN  ,   TP  ]    ← Fraud cases
```

---

### `cross_val_score()`
**Module:** `sklearn.model_selection`

**Purpose:** Performs k-fold cross-validation.

```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
# Returns: [0.951, 0.948, 0.935, 0.932, 0.939]

print(f"Mean: {scores.mean():.2%} ± {scores.std()*2:.2%}")
# Output: Mean: 94.14% ± 1.43%
```

| Parameter | Description |
|-----------|-------------|
| `model` | The classifier to evaluate |
| `X` | Feature matrix |
| `y` | Target vector |
| `cv` | Number of folds (default 5) |
| `scoring` | Metric to use ('accuracy', 'roc_auc', etc.) |

---

### `LabelEncoder()`
**Module:** `sklearn.preprocessing`

**Purpose:** Converts categorical labels to integers.

```python
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
le.fit(['Inpatient', 'Outpatient'])  # Learn categories

encoded = le.transform(['Inpatient', 'Outpatient', 'Inpatient'])
# Returns: [0, 1, 0]

decoded = le.inverse_transform([0, 1, 0])
# Returns: ['Inpatient', 'Outpatient', 'Inpatient']
```

---

### `joblib.dump()` / `joblib.load()`
**Module:** `joblib`

**Purpose:** Save and load Python objects efficiently.

```python
import joblib

# Save model
joblib.dump(model_artifacts, 'ml/model.pkl')

# Load model
model_artifacts = joblib.load('ml/model.pkl')
model = model_artifacts['model']
```

| Function | Description |
|----------|-------------|
| `dump(obj, filename)` | Save object to file |
| `load(filename)` | Load object from file |

---

## 📊 PANDAS FUNCTIONS USED

### `groupby().agg()`
**Purpose:** Group data and apply aggregation functions.

```python
provider_stats = df.groupby('provider_id').agg({
    'amount': ['mean', 'sum', 'std', 'max', 'min'],
    'patient_id': 'nunique',  # Count unique
    'claim_id': 'count'       # Count total
})
```

| Aggregation | Description |
|-------------|-------------|
| `'mean'` | Average value |
| `'sum'` | Total sum |
| `'std'` | Standard deviation |
| `'max'` | Maximum value |
| `'min'` | Minimum value |
| `'count'` | Number of rows |
| `'nunique'` | Number of unique values |
| `'first'` | First value in group |

---

### `fillna()` / `replace()`
**Purpose:** Handle missing and infinite values.

```python
# Fill missing values with 0
X = X.fillna(0)

# Replace infinity with 0
X = X.replace([np.inf, -np.inf], 0)
```

---

## 🎯 Summary of All Functions

| Function | Location | Purpose |
|----------|----------|---------|
| `create_provider_features(df)` | train_model.py | Aggregate claims to provider level |
| `train_enhanced_model()` | train_model.py | Main training entry point |
| `train_test_split()` | sklearn | Split data into train/test |
| `StandardScaler()` | sklearn | Normalize features |
| `GradientBoostingClassifier()` | sklearn | Ensemble classifier |
| `RandomForestClassifier()` | sklearn | Ensemble classifier |
| `accuracy_score()` | sklearn | Calculate accuracy |
| `roc_auc_score()` | sklearn | Calculate ROC-AUC |
| `classification_report()` | sklearn | Precision/Recall/F1 |
| `confusion_matrix()` | sklearn | TP/TN/FP/FN matrix |
| `cross_val_score()` | sklearn | K-fold validation |
| `LabelEncoder()` | sklearn | Encode categories |
| `joblib.dump/load()` | joblib | Save/load models |
| `groupby().agg()` | pandas | Aggregate data |
