# Healthcare Fraud Detection Model
# 📘 COMPREHENSIVE TECHNICAL HANDBOOK

**Version:** 1.0.0  
**Date:** January 2026  
**Purpose:** Complete reference of all terminologies, technologies, and concepts used in this project

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Project Architecture Overview](#2-project-architecture-overview)
3. [Healthcare Domain Terminologies](#3-healthcare-domain-terminologies)
4. [Data Science & ML Terminologies](#4-data-science--ml-terminologies)
5. [Statistical Concepts](#5-statistical-concepts)
6. [Backend Technologies](#6-backend-technologies)
7. [Frontend Technologies](#7-frontend-technologies)
8. [Machine Learning Libraries](#8-machine-learning-libraries)
9. [Database Technologies](#9-database-technologies)
10. [AI & External Services](#10-ai--external-services)
11. [DevOps & Deployment](#11-devops--deployment)
12. [Feature Engineering Deep Dive](#12-feature-engineering-deep-dive)
13. [Model Evaluation Metrics](#13-model-evaluation-metrics)
14. [Project Stage Mapping](#14-project-stage-mapping)
15. [API Endpoints Reference](#15-api-endpoints-reference)
16. [Conclusion](#16-conclusion)

---

## 1. Introduction

### 1.1 Project Overview
The **Healthcare Fraud Detection Model** is an end-to-end machine learning system that identifies potentially fraudulent healthcare insurance claims. The system analyzes claim patterns at the provider level, achieving **94.82% accuracy** in detecting fraud.

### 1.2 Problem Statement
Healthcare fraud in India costs the insurance industry an estimated **₹10,000+ crores annually**. This project addresses:
- **Phantom Billing**: Billing for services never rendered
- **Upcoding**: Billing for more expensive procedures than performed
- **Unbundling**: Separating procedures that should be billed together
- **Provider Collusion**: Fraudulent doctors submitting false claims

### 1.3 Solution Architecture
```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   React Frontend│────▶│  FastAPI Backend│────▶│   ML Model      │
│   (Vercel)      │     │  (Render)       │     │   (model.pkl)   │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                              │
                        ┌─────┴─────┐
                        │  SQLite   │
                        │  Database │
                        └───────────┘
```

---

## 2. Project Architecture Overview

### 2.1 Directory Structure
```
e:\Fraud on Healthcare\
├── backend/
│   ├── main.py           # FastAPI application (1,476 lines)
│   ├── database.py       # SQLAlchemy models
│   ├── ai_service.py     # Google Gemini integration
│   ├── icd_lookup.py     # ICD-9/ICD-10 code lookup
│   └── hospital_lookup.py # Hospital search (30,273 hospitals)
├── frontend/
│   └── src/
│       ├── App.jsx       # Main React component (912 lines)
│       └── translations.js
├── ml/
│   ├── train_model.py    # Model training script (561 lines)
│   └── model.pkl         # Trained model (~1 MB)
├── data/
│   ├── claims.csv        # 558,211 claims
│   ├── claims.db         # SQLite database
│   └── user_submissions.db
└── Dataset/
    └── Synthetic Dataset/
        ├── ICD9codes.csv
        ├── ICD10codes.csv
        └── indian_hospitals_classified.csv
```

### 2.2 Data Flow
1. **User submits claim** → Frontend React App
2. **API Request** → FastAPI Backend (`/predict` endpoint)
3. **Feature Engineering** → Provider-level aggregation
4. **ML Prediction** → Gradient Boosting Classifier
5. **Rule-Based Checks** → Price zone validation
6. **Response** → Fraud probability, risk level, explanation

---

## 3. Healthcare Domain Terminologies

### 3.1 Core Healthcare Terms

| Term | Definition | Usage in Project | Example |
|------|------------|------------------|---------|
| **Healthcare Fraud** | Intentional deception to receive unauthorized benefits | Target variable (`is_fraud`) | Provider billing for phantom services |
| **Medical Claim** | Request for payment from healthcare provider to insurer | Base data unit | 558,211 claims in dataset |
| **Provider** | Healthcare facility or practitioner | Aggregation level for ML | 5,410 unique providers |
| **Patient** | Individual receiving healthcare services | Patient demographic features | Unique patient count per provider |
| **Inpatient** | Patient admitted to hospital (overnight stay) | `claim_type = 'Inpatient'` | 40,474 inpatient claims |
| **Outpatient** | Patient treated without admission | `claim_type = 'Outpatient'` | 517,737 outpatient claims |

### 3.2 Medical Coding Systems

| Term | Definition | Usage in Project | Example |
|------|------------|------------------|---------|
| **ICD-9** | International Classification of Diseases, 9th Revision | Primary diagnosis codes in dataset | 4019 = Hypertension |
| **ICD-10** | International Classification of Diseases, 10th Revision | Alternate code support | J449 = COPD |
| **Diagnosis Code** | Alphanumeric code for medical condition | `diagnosis_code` column | 25000 = Diabetes Mellitus |
| **Procedure Code** | Code for medical procedure performed | `num_procedures` count | CPT codes (not used directly) |

### 3.3 Fraud Types

| Fraud Type | Definition | Detection Method | Project Implementation |
|------------|------------|------------------|------------------------|
| **Phantom Billing** | Billing for services never rendered | Provider revenue anomaly | High `amount_sum` feature |
| **Upcoding** | Billing higher-level service than performed | Price zone comparison | `amount_per_diagnosis` analysis |
| **Unbundling** | Billing separately for bundled procedures | Claims-per-patient ratio | `claims_per_patient` feature |
| **Duplicate Claims** | Submitting same claim multiple times | Claim pattern analysis | Provider aggregation |
| **Ghost Patients** | Billing for non-existent patients | Patient count anomaly | `unique_patients` feature |

### 3.4 Indian Healthcare System

| Term | Definition | Cost Level | Expected Prices |
|------|------------|------------|-----------------|
| **Government Hospital** | AIIMS, PHC, District Hospitals | Lowest | ₹100 - ₹3,000 |
| **Private Clinic** | Small nursing homes, clinics | Medium | ₹1,000 - ₹15,000 |
| **Corporate Hospital** | Apollo, Fortis, Max Healthcare | Highest | ₹5,000 - ₹50,000+ |
| **GST Rate** | Goods & Services Tax on healthcare | 18% | Applied to all claims |

### 3.5 Billing & Financial Terms

| Term | Definition | Column Name | Example Value |
|------|------------|-------------|---------------|
| **Claim Amount** | Total billed amount | `amount` | ₹15,000 |
| **Deductible** | Patient's out-of-pocket portion | `deductible` | ₹500 |
| **Reimbursement** | Amount paid by insurance | Calculated | Claims - Deductible |
| **Length of Stay** | Days patient hospitalized | `length_of_stay` | 5 days |
| **Chronic Conditions** | Long-term health conditions | `chronic_conditions` | 3 conditions |

---

## 4. Data Science & ML Terminologies

### 4.1 Core ML Concepts

| Term | Definition | Usage in Project | Value/Example |
|------|------------|------------------|---------------|
| **Supervised Learning** | Learning from labeled examples | Primary approach | is_fraud labels |
| **Binary Classification** | Predicting one of two classes | Fraud detection | Fraud (1) vs Non-Fraud (0) |
| **Feature** | Input variable for ML model | 28 engineered features | amount_mean, claims_per_patient |
| **Label/Target** | Variable to predict | `is_fraud` | True/False |
| **Training Set** | Data used to train model | 80% of data | 4,328 providers |
| **Test Set** | Data used to evaluate model | 20% of data | 1,082 providers |

### 4.2 Ensemble Methods

| Term | Definition | Usage in Project | Configuration |
|------|------------|------------------|---------------|
| **Gradient Boosting** | Sequential ensemble of decision trees | Primary model | 150 trees, max_depth=5 |
| **Random Forest** | Parallel ensemble of decision trees | Compared model | Also tested |
| **Decision Tree** | Tree-based classifier | Base learner | Part of ensemble |
| **Boosting** | Iteratively correcting errors | Training approach | Learning rate = 0.1 |
| **Bagging** | Bootstrap aggregating | RF approach | Not primary |

### 4.3 Data Preprocessing Terms

| Term | Definition | Usage in Project | Implementation |
|------|------------|------------------|----------------|
| **Feature Scaling** | Normalizing feature ranges | StandardScaler | mean=0, std=1 |
| **Label Encoding** | Converting categories to numbers | LabelEncoder | claim_type → 0/1 |
| **Missing Values** | Handling null/NaN data | fillna(0) | Replace with 0 |
| **Infinity Handling** | Replacing inf values | replace([np.inf, -np.inf], 0) | Division by zero cases |
| **Data Aggregation** | Combining records into groups | groupby('provider_id') | 558K → 5,410 |

### 4.4 Model Training Terms

| Term | Definition | Usage in Project | Value |
|------|------------|------------------|-------|
| **Hyperparameters** | Model configuration settings | n_estimators, max_depth | Manually tuned |
| **Learning Rate** | Step size in gradient descent | 0.1 | Boosting parameter |
| **Max Depth** | Maximum tree depth | 5 | Prevents overfitting |
| **Number of Estimators** | Number of trees in ensemble | 150 | Model complexity |
| **Random State** | Seed for reproducibility | 42 | Consistent results |

---

## 5. Statistical Concepts

### 5.1 Descriptive Statistics

| Term | Definition | Usage in Project | Example |
|------|------------|------------------|---------|
| **Mean (Average)** | Sum divided by count | `amount_mean` | ₹5,243 average claim |
| **Median** | Middle value in sorted data | Data analysis | Less sensitive to outliers |
| **Standard Deviation** | Spread around mean | `amount_std` | ₹3,200 variability |
| **Variance** | Square of standard deviation | Statistical analysis | Spread measurement |
| **Min/Max** | Extreme values | `amount_min`, `amount_max` | Range of claims |
| **Sum** | Total of all values | `amount_sum` | Total revenue |
| **Count** | Number of records | `total_claims` | 558,211 claims |

### 5.2 Probability & Distributions

| Term | Definition | Usage in Project | Example |
|------|------------|------------------|---------|
| **Probability** | Likelihood of event (0-1) | Fraud probability | 0.78 = 78% fraud chance |
| **Class Distribution** | Proportion of each class | Fraud rate | 39.2% fraud providers |
| **Stratified Sampling** | Preserve class proportions | train_test_split(stratify=y) | Fair splitting |
| **Imbalanced Data** | Unequal class sizes | Fraud detection | More non-fraud than fraud |

### 5.3 Cross-Validation

| Term | Definition | Usage in Project | Value |
|------|------------|------------------|-------|
| **K-Fold CV** | Split data into K parts | 5-fold CV | Model robustness |
| **CV Score** | Average performance across folds | cross_val_score | 94.14% ± 1.43% |
| **Overfitting** | Model memorizes training data | Prevented by CV | Detected via CV variance |
| **Underfitting** | Model too simple | Not an issue | High accuracy achieved |

---

## 6. Backend Technologies

### 6.1 FastAPI Framework

| Component | Description | File Location | Example |
|-----------|-------------|---------------|---------|
| **FastAPI** | Modern async Python web framework | `backend/main.py` | `app = FastAPI()` |
| **Uvicorn** | ASGI server for running FastAPI | Server command | `uvicorn main:app --reload` |
| **Route Decorators** | Define API endpoints | Throughout main.py | `@app.get("/stats")` |
| **Request/Response** | Handle HTTP communication | All endpoints | JSON responses |
| **Dependency Injection** | Database session management | `get_db()` function | `Depends(get_db)` |

### 6.2 Middleware & Security

| Component | Description | Implementation | Purpose |
|-----------|-------------|----------------|---------|
| **CORS Middleware** | Cross-Origin Resource Sharing | `CORSMiddleware` | Allow frontend access |
| **Rate Limiting** | Prevent DDoS attacks | SlowAPI library | `@limiter.limit("100/minute")` |
| **Security Headers** | X-Content-Type-Options, X-Frame-Options | Custom middleware | Prevent attacks |
| **Request Logging** | Audit trail for all requests | Custom middleware | Security monitoring |

### 6.3 SlowAPI Rate Limiting

```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
```

| Configuration | Value | Purpose |
|---------------|-------|---------|
| **Key Function** | get_remote_address | Identify by IP |
| **Rate Limit** | 100/minute | Prevent abuse |
| **Exception Handler** | RateLimitExceeded | Custom error response |

### 6.4 Pydantic Data Validation

| Component | Description | Example |
|-----------|-------------|---------|
| **BaseModel** | Define request/response schemas | `ClaimRequest(BaseModel)` |
| **Type Hints** | Python type annotations | `amount: float` |
| **Validation** | Automatic data validation | Required fields check |
| **Serialization** | Convert to/from JSON | Response models |

### 6.5 Python Standard Library

| Module | Purpose | Usage |
|--------|---------|-------|
| **logging** | Application logging | Request logging, errors |
| **datetime** | Date/time handling | Timestamps |
| **os** | File system operations | Path handling |
| **json** | JSON parsing | AI response parsing |

### 6.6 python-dotenv

| Feature | Description | Usage |
|---------|-------------|-------|
| **load_dotenv()** | Load .env file | Environment variables |
| **os.getenv()** | Access env variables | `GEMINI_API_KEY` |
| **.env file** | Store secrets | API keys, database URLs |

---

## 7. Frontend Technologies

### 7.1 React 19.2.0

| Concept | Description | Usage |
|---------|-------------|-------|
| **Functional Components** | Modern React pattern | `function App()` |
| **useState Hook** | Component state management | `const [stats, setStats] = useState({})` |
| **useEffect Hook** | Side effects (API calls) | Data fetching on mount |
| **Props** | Component data passing | Not heavily used (single component) |
| **JSX** | JavaScript XML syntax | HTML-like component structure |

### 7.2 React Hooks Used

| Hook | Purpose | Example |
|------|---------|---------|
| **useState** | Local component state | `const [claims, setClaims] = useState([])` |
| **useEffect** | Side effects, lifecycle | Fetch data on mount, intervals |

### 7.3 Vite 7.2.4 Build Tool

| Feature | Description | Configuration |
|---------|-------------|---------------|
| **Dev Server** | Fast development server | `npm run dev` → localhost:5173 |
| **HMR** | Hot Module Replacement | Instant updates without refresh |
| **Build** | Production bundling | `npm run build` |
| **Environment Variables** | Runtime configuration | `import.meta.env.VITE_API_URL` |

### 7.4 Recharts 3.5.1 (Charting Library)

| Component | Description | Usage |
|-----------|-------------|-------|
| **LineChart** | Time series visualization | Fraud trends over time |
| **BarChart** | Category comparison | Claims by type |
| **PieChart** | Proportion display | Fraud vs non-fraud |
| **ResponsiveContainer** | Auto-sizing charts | Responsive design |
| **XAxis, YAxis** | Chart axes | Data labels |
| **Tooltip** | Hover information | Data details |
| **Legend** | Chart legend | Color coding |
| **CartesianGrid** | Background grid | Visual reference |

### 7.5 Lucide React (Icons)

| Icon | Purpose | Component |
|------|---------|-----------|
| **Activity** | Dashboard activity | General stats |
| **AlertTriangle** | Warning/fraud alert | Risk indicators |
| **CheckCircle** | Success/valid claim | Low risk claims |
| **ShieldAlert** | Security/fraud | High risk claims |
| **DollarSign** | Financial data | Amount displays |
| **Database** | Data/storage | Database stats |
| **Search** | Search function | Hospital search |
| **Clock** | Time-related | Timestamps |
| **UserCheck** | User verification | Patient data |
| **Stethoscope** | Healthcare | Medical icons |
| **Globe** | Language toggle | i18n |
| **Bot** | AI assistant | Chatbot |
| **Send** | Send message | Chat input |
| **Sparkles** | AI features | AI insights |
| **FileText** | Reports | Generated reports |
| **MapPin** | Location | Hospital location |

### 7.6 Axios 1.13.2 (HTTP Client)

| Feature | Description | Usage |
|---------|-------------|-------|
| **GET requests** | Fetch data | `axios.get(API_URL/stats)` |
| **POST requests** | Submit data | `axios.post(API_URL/predict, claim)` |
| **Error handling** | Catch API errors | try/catch blocks |
| **Base URL** | API configuration | Environment variable |

### 7.7 TailwindCSS 4.1.17

| Concept | Description | Example |
|---------|-------------|---------|
| **Utility Classes** | Pre-built CSS classes | `bg-blue-500`, `text-white` |
| **Responsive Design** | Mobile-first approach | `md:`, `lg:` prefixes |
| **Dark Mode** | Dark theme support | `dark:` prefix |
| **Flexbox/Grid** | Layout utilities | `flex`, `grid` |

### 7.8 Internationalization (i18n)

| Feature | Description | Implementation |
|---------|-------------|----------------|
| **Language State** | Track current language | `const [lang, setLang] = useState('en')` |
| **Translations Object** | Key-value pairs | `translations.js` |
| **Toggle Function** | Switch languages | `toggleLanguage()` |
| **Supported Languages** | English, Hindi | 'en', 'hi' |

---

## 8. Machine Learning Libraries

### 8.1 scikit-learn 1.6.1

#### 8.1.1 Model Classes

| Class | Description | Configuration |
|-------|-------------|---------------|
| **GradientBoostingClassifier** | Primary model | n_estimators=150, max_depth=5 |
| **RandomForestClassifier** | Alternate model | Compared during training |

#### 8.1.2 Preprocessing Classes

| Class | Description | Usage |
|-------|-------------|-------|
| **StandardScaler** | Feature normalization | Scale to mean=0, std=1 |
| **LabelEncoder** | Categorical encoding | claim_type → numeric |

#### 8.1.3 Model Selection Functions

| Function | Description | Usage |
|----------|-------------|-------|
| **train_test_split** | Split data | 80/20 split, stratify=y |
| **cross_val_score** | K-fold cross-validation | 5-fold CV (94.14% ± 1.43%) |

#### 8.1.4 Metrics Functions

| Function | Description | Project Value |
|----------|-------------|---------------|
| **accuracy_score** | Overall correctness | 94.82% |
| **precision_score** | TP / (TP + FP) | 91.4% (fraud class) |
| **recall_score** | TP / (TP + FN) | 95.8% (fraud class) |
| **f1_score** | Harmonic mean | 93.5% |
| **roc_auc_score** | ROC curve area | 0.9683 |
| **confusion_matrix** | TP, TN, FP, FN breakdown | TP=406, TN=620, FP=38, FN=18 |
| **classification_report** | Full metrics summary | Printed during training |

### 8.2 Pandas 2.2.3

| Function/Method | Description | Usage |
|-----------------|-------------|-------|
| **read_csv()** | Load CSV files | Load claims.csv |
| **DataFrame** | 2D data structure | Core data container |
| **groupby()** | Group data by column | Aggregate by provider_id |
| **agg()** | Apply aggregation functions | mean, sum, std, max, min |
| **fillna()** | Handle missing values | Replace NaN with 0 |
| **replace()** | Replace values | Replace inf with 0 |
| **head()** | First N rows | Preview data |
| **shape** | Dimensions | (rows, columns) |
| **value_counts()** | Count unique values | Class distribution |

### 8.3 NumPy 2.2.4

| Function | Description | Usage |
|----------|-------------|-------|
| **np.inf** | Infinity constant | Replace infinity values |
| **np.nan** | Not a Number | Handle missing values |
| **Array operations** | Vectorized math | Feature calculations |

### 8.4 joblib 1.5.3

| Function | Description | Usage |
|----------|-------------|-------|
| **joblib.dump()** | Save Python object | Save model.pkl |
| **joblib.load()** | Load Python object | Load model at runtime |

#### Model Bundle Contents
```python
bundle = {
    'model': trained_model,           # GradientBoostingClassifier
    'scaler': StandardScaler,         # Feature normalization
    'feature_cols': [...],            # 28 feature names
    'le_diagnosis': LabelEncoder,     # Diagnosis encoding
    'le_claim_type': LabelEncoder,    # Claim type encoding
    'model_type': 'provider_level'    # Aggregation approach
}
joblib.dump(bundle, 'model.pkl')
```

---

## 9. Database Technologies

### 9.1 SQLite

| Feature | Description | Configuration |
|---------|-------------|---------------|
| **Embedded Database** | No separate server | File-based (.db files) |
| **WAL Mode** | Write-Ahead Logging | Better concurrency |
| **Timeout** | Connection timeout | 30 seconds |
| **Thread Safety** | SQLite configuration | `check_same_thread=False` |

### 9.2 SQLAlchemy 2.0.46

| Component | Description | Usage |
|-----------|-------------|-------|
| **create_engine()** | Database connection | Connect to SQLite |
| **sessionmaker()** | Session factory | Database sessions |
| **declarative_base()** | ORM base class | Define models |
| **Column types** | Data types | Integer, String, Float, Boolean, DateTime |
| **Relationships** | Table relationships | Not used (simple schema) |

### 9.3 Database Schema

#### Claims Table
```python
class Claim(Base):
    __tablename__ = "claims"
    id = Column(Integer, primary_key=True)
    claim_id = Column(String, index=True)
    provider_id = Column(String, index=True)
    patient_id = Column(String, index=True)
    claim_type = Column(String)
    diagnosis_code = Column(String, index=True)
    amount = Column(Float)
    deductible = Column(Float)
    num_diagnoses = Column(Integer)
    num_procedures = Column(Integer)
    length_of_stay = Column(Integer)
    patient_age = Column(Integer)
    chronic_conditions = Column(Integer)
    amount_per_diagnosis = Column(Float)
    is_fraud = Column(Boolean)
    timestamp = Column(DateTime)
```

#### User Submissions Table
```python
class UserSubmission(UserBase):
    __tablename__ = "user_submissions"
    id = Column(Integer, primary_key=True, autoincrement=True)
    provider_id = Column(String)
    provider_type = Column(String)
    diagnosis_code = Column(String)
    disease_name = Column(String)
    claim_type = Column(String)
    amount = Column(Float)
    deductible = Column(Float)
    num_diagnoses = Column(Integer)
    num_procedures = Column(Integer)
    length_of_stay = Column(Integer)
    patient_age = Column(Integer)
    chronic_conditions = Column(Integer)
    is_fraud = Column(Boolean)
    fraud_probability = Column(Float)
    risk_level = Column(String)
    price_zone = Column(String)
    expected_price = Column(Float)
    gst_amount = Column(Float)
    total_with_gst = Column(Float)
    submitted_at = Column(DateTime)
    ip_address = Column(String)
```

---

## 10. AI & External Services

### 10.1 Google Gemini API

| Feature | Description | Usage |
|---------|-------------|-------|
| **Model** | gemini-2.0-flash | Fast, capable LLM |
| **Chat** | Conversational AI | Fraud investigation assistant |
| **Content Generation** | Text generation | Report generation |
| **API Key** | Authentication | GEMINI_API_KEY in .env |

### 10.2 AI Service Functions

| Function | Description | Endpoint |
|----------|-------------|----------|
| **chat_with_ai()** | Answer fraud queries | `/ai/chat` |
| **generate_fraud_report()** | Create investigation report | `/ai/report` |
| **get_intelligent_insights()** | Analyze fraud patterns | `/ai/insights` |
| **test_ai_service()** | Check API connectivity | `/ai/test` |

### 10.3 System Context (AI Prompt)

The AI assistant is configured with specialized knowledge:
- ML model accuracy (94.82%)
- Provider types and pricing
- ICD diagnosis codes
- GST rate (18%)
- Common fraud patterns
- Indian healthcare system

---

## 11. DevOps & Deployment

### 11.1 Vercel (Frontend)

| Feature | Description | Configuration |
|---------|-------------|---------------|
| **Static Hosting** | Serve React app | Automatic from GitHub |
| **CDN** | Global distribution | Edge network |
| **Environment Variables** | Runtime config | VITE_API_URL |
| **Rewrites** | SPA routing | All paths → index.html |
| **Headers** | Security headers | CSP, X-Frame-Options |

### 11.2 Render (Backend)

| Feature | Description | Configuration |
|---------|-------------|---------------|
| **Python Runtime** | Run FastAPI | Python 3.10 |
| **HTTPS** | SSL certificate | Automatic |
| **Environment Variables** | Secrets | GEMINI_API_KEY |
| **Start Command** | Launch app | `uvicorn main:app --host 0.0.0.0` |

### 11.3 Development Tools

| Tool | Purpose | Command |
|------|---------|---------|
| **npm** | Package management | `npm install`, `npm run dev` |
| **pip** | Python packages | `pip install -r requirements.txt` |
| **Git** | Version control | Track changes |
| **VS Code** | Code editor | Development IDE |

---

## 12. Feature Engineering Deep Dive

### 12.1 The 28 Engineered Features

#### Volume Features
| Feature | Aggregation | Fraud Indicator |
|---------|-------------|-----------------|
| `total_claims` | count | High volume billing |
| `unique_patients` | nunique | Patient base size |
| `claims_per_patient` | total/unique | Patient churning |

#### Financial Features
| Feature | Aggregation | Fraud Indicator |
|---------|-------------|-----------------|
| `amount_sum` | sum | Total revenue extraction |
| `amount_mean` | mean | Overbilling per claim |
| `amount_std` | std | Billing inconsistency |
| `amount_max` | max | Extreme claims |
| `amount_min` | min | Range analysis |
| `deductible_mean` | mean | Deductible patterns |
| `deductible_sum` | sum | Total deductibles |
| `revenue_per_patient` | amount_sum/unique_patients | Value extraction |
| `amount_per_diagnosis_mean` | mean | Upcoding indicator |
| `amount_per_diagnosis_max` | max | Extreme upcoding |

#### Service Complexity Features
| Feature | Aggregation | Fraud Indicator |
|---------|-------------|-----------------|
| `num_diagnoses_mean` | mean | Diagnosis inflation |
| `num_diagnoses_sum` | sum | Total diagnoses |
| `num_diagnoses_max` | max | Extreme cases |
| `num_procedures_mean` | mean | Procedure patterns |
| `num_procedures_sum` | sum | Total procedures |
| `num_procedures_max` | max | Extreme procedures |
| `length_of_stay_mean` | mean | Extended stays |
| `length_of_stay_sum` | sum | Total days |
| `length_of_stay_max` | max | Longest stay |

#### Patient Demographics
| Feature | Aggregation | Fraud Indicator |
|---------|-------------|-----------------|
| `patient_age_mean` | mean | Target demographics |
| `chronic_conditions_mean` | mean | Vulnerable patients |
| `chronic_conditions_sum` | sum | Total chronic cases |

#### Claim Type Distribution
| Feature | Calculation | Fraud Indicator |
|---------|-------------|-----------------|
| `inpatient_ratio` | inpatient/total | Service mix anomaly |

### 12.2 Feature Importance Rankings

| Rank | Feature | Importance | Interpretation |
|------|---------|------------|----------------|
| 1 | amount_sum | 0.142 | High total revenue = suspicious |
| 2 | revenue_per_patient | 0.118 | Extracting value per patient |
| 3 | amount_mean | 0.095 | Consistent overbilling |
| 4 | claims_per_patient | 0.087 | Billing same patient repeatedly |
| 5 | total_claims | 0.076 | Very high volume |
| 6 | chronic_conditions_sum | 0.068 | Targeting vulnerable patients |
| 7 | num_diagnoses_sum | 0.062 | Inflating diagnosis count |
| 8 | amount_per_diagnosis_mean | 0.058 | Upcoding pattern |
| 9 | inpatient_ratio | 0.054 | Unusual inpatient/outpatient mix |
| 10 | length_of_stay_mean | 0.048 | Prolonged hospital stays |

---

## 13. Model Evaluation Metrics

### 13.1 Primary Metrics

| Metric | Value | Formula | Interpretation |
|--------|-------|---------|----------------|
| **Accuracy** | 94.82% | (TP+TN)/Total | Overall correctness |
| **Precision** | 91.4% | TP/(TP+FP) | Of predicted fraud, how many are real |
| **Recall** | 95.8% | TP/(TP+FN) | Of real fraud, how many were caught |
| **F1-Score** | 93.5% | 2×(P×R)/(P+R) | Balance of precision/recall |
| **ROC-AUC** | 0.9683 | Area under curve | Discrimination ability |

### 13.2 Confusion Matrix

|  | Predicted Non-Fraud | Predicted Fraud |
|--|---------------------|-----------------|
| **Actual Non-Fraud** | TN = 620 | FP = 38 |
| **Actual Fraud** | FN = 18 | TP = 406 |

### 13.3 Cross-Validation Results

| Metric | Value |
|--------|-------|
| **Mean CV Accuracy** | 94.14% |
| **Standard Deviation** | ± 1.43% |
| **Number of Folds** | 5 |

---

## 14. Project Stage Mapping

### Stage 1: Data Collection

| Component | Terminologies | Technologies |
|-----------|---------------|--------------|
| **Raw Data** | Medical Claims, ICD Codes, Provider ID | Pandas (read_csv), CSV |
| **Data Source** | Kaggle Medicare Dataset | 558,211 claims |
| **Storage** | Relational Database | SQLite, SQLAlchemy |

### Stage 2: Data Preprocessing

| Component | Terminologies | Technologies |
|-----------|---------------|--------------|
| **Cleaning** | Missing Values, Null Handling | Pandas (fillna, dropna) |
| **Transformation** | Feature Scaling, Encoding | StandardScaler, LabelEncoder |
| **Validation** | Data Quality Checks | Pandas (dtype, shape) |

### Stage 3: Feature Engineering

| Component | Terminologies | Technologies |
|-----------|---------------|--------------|
| **Aggregation** | Provider-Level Statistics | Pandas (groupby, agg) |
| **Derived Features** | Ratio Calculations | NumPy, Pandas |
| **Feature Selection** | 28 Final Features | Manual selection |

### Stage 4: Model Training

| Component | Terminologies | Technologies |
|-----------|---------------|--------------|
| **Split** | Train/Test, Stratified | train_test_split |
| **Training** | Gradient Boosting | GradientBoostingClassifier |
| **Persistence** | Model Serialization | joblib |

### Stage 5: Model Evaluation

| Component | Terminologies | Technologies |
|-----------|---------------|--------------|
| **Metrics** | Accuracy, Precision, Recall | scikit-learn metrics |
| **Validation** | Cross-Validation | cross_val_score |
| **Visualization** | Confusion Matrix | confusion_matrix |

### Stage 6: Deployment & Serving

| Component | Terminologies | Technologies |
|-----------|---------------|--------------|
| **API** | REST Endpoints, JSON | FastAPI, Uvicorn |
| **Frontend** | SPA, Dashboard | React, Vite |
| **Cloud** | HTTPS, CDN | Vercel, Render |

---

## 15. API Endpoints Reference

### 15.1 Core Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check |
| `/stats` | GET | Dashboard statistics |
| `/claims` | GET | Recent claims list |
| `/predict` | POST | Predict fraud for claim |

### 15.2 AI Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/ai/chat` | POST | Chat with AI assistant |
| `/ai/report` | POST | Generate investigation report |
| `/ai/insights` | GET | Get fraud pattern insights |
| `/ai/test` | GET | Test AI service connection |

### 15.3 Hospital Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/hospitals/search` | GET | Search hospitals by name |
| `/hospitals/details` | GET | Get hospital full details |
| `/hospitals/stats` | GET | Hospital statistics |

---

## 16. Conclusion

### 16.1 Project Achievements

| Achievement | Value |
|-------------|-------|
| **Model Accuracy** | 94.82% |
| **ROC-AUC Score** | 0.9683 |
| **Recall (Fraud Detection)** | 95.8% |
| **Claims Processed** | 558,211 |
| **Providers Analyzed** | 5,410 |
| **Features Engineered** | 28 |

### 16.2 Technologies Summary

| Category | Count | Examples |
|----------|-------|----------|
| **Backend** | 8+ | FastAPI, SQLAlchemy, SlowAPI, Pydantic |
| **Frontend** | 7+ | React, Vite, Recharts, TailwindCSS, Axios |
| **ML/Data** | 5+ | scikit-learn, Pandas, NumPy, joblib |
| **External** | 2 | Google Gemini AI, ICD Code Database |
| **Deployment** | 2 | Vercel, Render |

### 16.3 Future Enhancements

1. **Deep Learning Models** - Neural networks for complex patterns
2. **Real-time Streaming** - Apache Kafka for live claims
3. **Network Analysis** - Provider collusion detection
4. **Explainable AI** - SHAP/LIME for prediction reasoning
5. **Continuous Learning** - Update model with new fraud patterns

---

**Document prepared for:** Healthcare Fraud Detection Project  
**Suitable for:** Technical viva, project reports, interviews, documentation  
**Total Terminologies:** 120+  
**Total Technologies:** 40+
