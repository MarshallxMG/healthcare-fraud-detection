# 🏗️ Healthcare Fraud Detection - System Architecture

## 🖼️ Full System Diagram

![System Architecture]

## 🧩 Component Breakdown & Connections

This system is built using a **Microservices-style Architecture** where the Frontend, Backend, and ML components are decoupled but communicate seamlessly.

---

### 1. 🎨 The Frontend (User Interface)
**Tech Stack:** React 18, Vite, Tailwind CSS, Recharts
**Role:** The "Face" of the application.

*   **User Flow:**
    *   User opens `http://localhost:5173`.
    *   React components (`ClaimForm.jsx`, `Dashboard.jsx`) render the UI.
    *   User inputs claim details (Amount, Provider ID, Diagnosis Code).
*   **Connection:**
    *   Connects to Backend via **HTTP Requests (REST API)**.
    *   Uses `fetch` or `axios` to send JSON data to `http://localhost:8000`.

---

### 2. ⚙️ The Backend (API Layer)
**Tech Stack:** Python 3.10, FastAPI, Uvicorn, SQLAlchemy
**Role:** The "Brain" and "Traffic Controller".

*   **How it Connects:**
    *   **Receives:** JSON data from Frontend.
    *   **Validates:** Uses `Pydantic` models (`ClaimInput` schema) to ensure data types are correct (e.g., age is a number).
    *   **Orchestrates:** It calls the helper modules (`icd_lookup.py`, `hospital_lookup.py`) and the ML model.
*   **Key Components:**
    *   **`main.py`**: The entry point. Handles routing (e.g., `/predict`, `/stats`).
    *   **`icd_lookup.py`**: A dictionary lookup for disease names.
    *   **`hospital_lookup.py`**: Searches hospital names.

---

### 3. 🧠 The ML Pipeline (Intelligence Layer)
**Tech Stack:** Scikit-learn, Pandas, Joblib, Gradient Boosting
**Role:** The "Detective" making decisions.

*   **The Connection (Model Loading):**
    *   The model isn't a separate running server; it's a **serialized file** (`model.pkl`).
    *   On Backend startup, `joblib.load('ml/model.pkl')` loads the pre-trained Python objects into memory.
    *   This is efficient—no need to retrain for every request!
*   **The Prediction Flow:**
    *   Backend passes raw input features to the loaded model.
    *   Model runs `predict_proba()` to estimate fraud probability.
    *   Model returns a score (e.g., `0.85`), which the Backend interprets as "Critical Risk".

---

### 4. 💾 The Data Layer (Storage)
**Tech Stack:** SQLite, CSV
**Role:** The "Memory" of the system.

*   **Static Data (Read-Only):**
    *   `claims.csv`: The huge training dataset (History).
    *   `disease_prices.csv`: Benchmark prices for diseases (Reference).
        *   *Connection:* Loaded into Pandas DataFrames on startup for fast lookups.
*   **Dynamic Data (Read/Write):**
    *   `user_submissions.db`: Stores every claim the user tests.
        *   *Connection:* `SQLAlchemy` ORM manages the connection. Logic is in `backend/database.py`.

---

## 🔄 End-to-End Data Flow Example

Let's trace a single request: **"Analyze a ₹50,000 claim for a checkup"**

1.  **Frontend**: User clicks "Analyze". React creates a JSON payload:
    ```json
    { "provider_type": "Government", "amount": 50000, "diagnosis_code": "V700" }
    ```
2.  **API (Layer 1)**: FastAPI receives it.
    *   Checks **Rules**: "Is ₹50k > Standard ₹500 for checkup?" -> **YES**.
    *   *Result:* Flags as "Suspicious" immediately.
3.  **API (Layer 2)**: (If rules didn't catch it)
    *   API converts JSON -> features vector.
    *   Passes vector to `model.predict()`.
    *   Model says: "Fraud Probability: 92%".
4.  **Database**: API saves this attempt to `user_submissions.db` for audit.
5.  **Response**: API sends JSON back to Frontend:
    ```json
    { "is_fraud": true, "risk_level": "Critical", "message": "Price 100x above norm" }
    ```
6.  **Frontend**: Displays a Red Alert 🚨 card to the user.

---

## 🔗 How it all fits together (The "Glue")

| Connection Point | Mechanism | Protocol |
| :--- | :--- | :--- |
| **Frontend ↔ Backend** | REST API Calls | HTTP/JSON |
| **Backend ↔ Model** | In-Memory Function Call | Python Object |
| **Backend ↔ Database** | ORM Query | SQL (SQLite driver) |
| **User ↔ Frontend** | Browser Interaction | HTML/JS Events |
