# Healthcare Fraud Detection System

A Machine Learning-based system to detect fraudulent healthcare claims in real-time.

## 🚀 Quick Start

### Prerequisites
- Python 3.10 or higher
- Node.js 18 or higher
- npm or yarn

### Installation

1. **Clone/Download the Project**
   ```bash
   cd "e:/Fraud on Healthcare"
   ```

2. **Set Up Python Virtual Environment**
   ```bash
   python -m venv venv
   venv\Scripts\activate  # Windows
   # source venv/bin/activate  # Linux/Mac
   ```

3. **Install Python Dependencies**
   ```bash
   pip install fastapi uvicorn sqlalchemy pandas scikit-learn joblib fpdf2
   ```

4. **Install Frontend Dependencies**
   ```bash
   cd frontend
   npm install
   cd ..
   ```

---

## ▶️ Running the Project

### Step 1: Start the Backend API
```bash
venv\Scripts\uvicorn backend.main:app --reload --port 8000
```

You should see:
```
✅ User submissions database ready: user_submissions.db
✅ Model loaded successfully!
✅ Loaded prices for 6016 diagnoses
INFO: Application startup complete.
```

### Step 2: Start the Frontend
Open a **new terminal** and run:
```bash
cd frontend
npm run dev
```

You should see:
```
VITE ready in 800ms
➜ Local: http://localhost:5173/
```

### Step 3: Open the Dashboard
Open your browser and go to:
- **Dashboard**: http://localhost:5173
- **API Docs**: http://localhost:8000/docs

---

## 📁 Project Structure

```
e:/Fraud on Healthcare/
├── backend/               # FastAPI Backend
│   ├── main.py           # API endpoints & fraud detection
│   ├── database.py       # Database models
│   └── icd_lookup.py     # ICD code lookup
├── frontend/             # React Frontend
│   └── src/
│       └── App.jsx       # Main dashboard
├── ml/                   # Machine Learning
│   ├── train_model.py    # Training script
│   └── model.pkl         # Trained model
├── data/                 # Data Files
│   ├── claims.csv        # Main dataset (40MB)
│   ├── claims.db         # SQLite database
│   ├── disease_prices.csv # Pricing data
│   └── user_submissions.db # User entries
├── Dataset/              # Original Kaggle data
├── PROJECT_REPORT.pdf    # Complete documentation
└── README.md             # This file
```

---

## 🔌 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check |
| `/stats` | GET | Dataset statistics |
| `/claims` | GET | Recent claims list |
| `/predict` | POST | Analyze a claim for fraud |
| `/user-submissions` | GET | View saved user entries |
| `/benchmarks` | GET | Provider pricing benchmarks |

### Example: Analyze a Claim
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "provider_id": "PRV001",
    "provider_type": "Clinic",
    "diagnosis_code": "4019",
    "claim_type": "Outpatient",
    "amount": 500,
    "patient_age": 65,
    "num_diagnoses": 2,
    "chronic_conditions": 2
  }'
```

---

## 🧪 Running Tests

```bash
venv\Scripts\python.exe test_api.py
```

Expected output:
```
Total Tests: 10
✓ Passed: 10
❌ Failed: 0

🎉 ALL TESTS PASSED!
```

---

## 🔧 Troubleshooting

### Backend won't start
```bash
# Make sure you're in the project directory
cd "e:/Fraud on Healthcare"

# Activate virtual environment
venv\Scripts\activate

# Check if port 8000 is free
netstat -ano | findstr :8000
```

### Frontend won't start
```bash
cd frontend
npm install  # Reinstall dependencies
npm run dev
```

### Database errors
```bash
# The databases are auto-created on first run
# If issues persist, delete and restart:
del data\user_submissions.db
venv\Scripts\uvicorn backend.main:app --reload --port 8000
```

---

## 📊 Key Features

- **Two-Layer Fraud Detection**: Rule-based + ML model (94.82% accuracy)
- **Disease-Specific Pricing**: Compares against expected costs
- **Real-time Analysis**: Instant fraud risk assessment
- **ICD Code Support**: Both ICD-9 and ICD-10 codes
- **User Submissions**: Auto-saves all analyzed claims
- **GST Calculation**: 18% GST for Indian healthcare

---

## 📚 Documentation

- `PROJECT_REPORT.pdf` - Complete project report
- `PROJECT_LOGIC.md` - System logic explanation
- `ML_MODEL_DOCUMENTATION.md` - ML model details

---

## 👨‍💻 Tech Stack

| Layer | Technology |
|-------|------------|
| Backend | FastAPI, Python 3.10+, SQLAlchemy |
| Frontend | React 18, Vite, Tailwind CSS, Recharts |
| ML | Scikit-learn, Gradient Boosting |
| Database | SQLite |

---

## 📝 License

This project is for educational purposes.

---

**Happy Fraud Detection! 🔍**
