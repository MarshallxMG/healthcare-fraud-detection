"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    HEALTHCARE FRAUD DETECTION API                            ║
║                    ---------------------------------                          ║
║  Author: Healthcare Analytics Team                                           ║
║  Purpose: Detect fraudulent healthcare claims using ML + Rules               ║
║  Country: India (adapted for Indian healthcare system)                       ║
╚══════════════════════════════════════════════════════════════════════════════╝

This is the main backend API for the Healthcare Fraud Detection System.
It provides endpoints for:
1. Analyzing individual claims for potential fraud
2. Retrieving statistics about claims in the database
3. Getting recent claims for display on the dashboard

The system uses a TWO-LAYER approach:
    Layer 1: Rule-Based Detection (instant flags for obvious anomalies)
    Layer 2: ML Model (pattern recognition for subtle fraud)

INDIAN HEALTHCARE PRICING:
    - Government hospitals (AIIMS, PHC, District Hospitals) - Cheapest
    - Private Clinics / Nursing Homes - Medium pricing
    - Corporate Hospitals (Apollo, Fortis, Max) - Most expensive
"""

# =============================================================================
# IMPORTS
# =============================================================================

from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
from .database import SessionLocal, Claim, UserSubmission, UserSessionLocal
from .icd_lookup import get_disease_info
from .ai_service import chat_with_ai, generate_fraud_report, get_intelligent_insights, test_ai_service
import pandas as pd
from pydantic import BaseModel
from typing import List, Optional
import joblib
import os
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
from . import hospital_lookup

# =============================================================================
# CONFIGURATION
# =============================================================================

# GST Rate for India (Goods & Services Tax on healthcare services)
# As per Indian tax laws, healthcare services attract 18% GST
GST_RATE = 0.18  # 18% GST

# Initialize FastAPI application
app = FastAPI(
    title="Healthcare Fraud Detection API",
    description="AI-powered fraud detection for Indian healthcare claims",
    version="1.0.0"
)

# =============================================================================
# RATE LIMITING (DDoS Protection)
# =============================================================================
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# Initialize rate limiter using client IP address
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# =============================================================================
# SECURITY MIDDLEWARE
# =============================================================================
import logging
from fastapi import Request

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    """Add security headers to all responses."""
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    return response

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all incoming requests for audit trail."""
    logger.info(f"Request: {request.method} {request.url.path} from {request.client.host if request.client else 'unknown'}")
    response = await call_next(request)
    return response

# =============================================================================
# CORS MIDDLEWARE
# =============================================================================
# CORS (Cross-Origin Resource Sharing) allows our frontend to communicate
# with this backend API even though they run on different ports

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get allowed origins from environment (default to localhost for development)
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,  # Restricted to specific origins
    allow_credentials=True,
    allow_methods=["GET", "POST"],  # Only needed methods
    allow_headers=["Content-Type", "Authorization"],  # Only needed headers
)

# =============================================================================
# DATABASE CONNECTION
# =============================================================================

def get_db():
    """
    Database Dependency Injection
    
    Creates a new database session for each API request.
    The session is automatically closed when the request completes.
    This is called a "dependency" in FastAPI.
    
    Usage in endpoints:
        @app.get("/something")
        def my_endpoint(db: Session = Depends(get_db)):
            # db is now available for database operations
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# =============================================================================
# DATABASE SEEDING (For Render deployment)
# =============================================================================
# =============================================================================
# DATABASE SEEDING (For Render deployment)
# =============================================================================
# =============================================================================
# DATABASE SEEDING (For Render deployment)
# =============================================================================
import threading

def run_seeding_logic_sync(force: bool = False):
    """Shared seeding logic (Synchronous for Threading)."""
    db = SessionLocal()
    try:
        count = db.query(Claim).count()
        if count > 0 and not force:
            print(f"✅ Database contains {count} claims. Skipping seed.")
            return

        print("🌱 Seeding database from CSV (Background)...")
        csv_path = os.path.join(BASE_DIR, "data", "claims.csv")
        
        if not os.path.exists(csv_path):
            print(f"⚠️ CSV file not found at {csv_path}")
            return

        # Load CSV in chunks
        chunk_size = 5000 
        total_loaded = 0
        
        # Read CSV with pandas
        for chunk in pd.read_csv(csv_path, chunksize=chunk_size):
            claims_to_add = []
            for _, row in chunk.iterrows():
                claim = Claim(
                    claim_id=str(row.get('claim_id', '')),
                    provider_id=str(row.get('provider_id', '')),
                    patient_id=str(row.get('patient_id', '')),
                    claim_type=str(row.get('claim_type', '')),
                    diagnosis_code=str(row.get('diagnosis_code', '')),
                    amount=float(row.get('amount', 0)),
                    deductible=float(row.get('deductible', 0)),
                    num_diagnoses=int(row.get('num_diagnoses', 1)),
                    num_procedures=int(row.get('num_procedures', 0)),
                    length_of_stay=int(row.get('length_of_stay', 0)),
                    patient_age=int(row.get('patient_age', 0)),
                    chronic_conditions=int(row.get('chronic_conditions', 0)),
                    amount_per_diagnosis=float(row.get('amount_per_diagnosis', 0)),
                    is_fraud=bool(row.get('is_fraud', False)),
                    timestamp=datetime.utcnow()
                )
                claims_to_add.append(claim)
            
            db.add_all(claims_to_add)
            db.commit()
            total_loaded += len(claims_to_add)
            print(f"   Loaded {total_loaded} claims...")
        
        print(f"✅ Successfully seeded {total_loaded} claims!")
            
    except Exception as e:
        print(f"❌ Error seeding database: {e}")
    finally:
        db.close()

def run_background_initialization(force_seed: bool = False):
    """Run all heavy initialization in background."""
    print("⏳ Starting background initialization...")
    
    # 1. Load ML Model
    load_ml_model()
    
    # 2. Load Disease Prices
    load_disease_prices()
    
    # 3. Seed Database
    run_seeding_logic_sync(force_seed)
    
    print("🚀 Background initialization complete!")

@app.on_event("startup")
async def startup_event():
    """Start initialization in background on startup."""
    # Run everything in a separate thread to not block startup
    thread = threading.Thread(target=run_background_initialization, args=(False,))
    thread.daemon = True
    thread.start()
    print("🚀 Server started. Initialization running in background.")

@app.get("/seed")
def manual_seed(force: bool = False):
    """Manually trigger database seeding (GET for browser access)."""
    # Run in background thread to avoid timeout
    thread = threading.Thread(target=run_seeding_logic_sync, args=(force,))
    thread.daemon = True
    thread.start()
    return {"status": "started", "message": "Seeding started in background. Check logs for progress."}

def save_user_submission(claim_input, result):
    """
    Save user submission to database.
    
    Called automatically when a user submits a claim for analysis.
    Stores both the input data and the prediction results.
    """
    try:
        db = UserSessionLocal()
        
        submission = UserSubmission(
            # User Input Fields
            provider_id=claim_input.provider_id,
            provider_type=claim_input.provider_type,
            diagnosis_code=claim_input.diagnosis_code,
            disease_name=result.get('short_desc', 'Unknown'),
            claim_type=claim_input.claim_type,
            amount=claim_input.amount,
            deductible=claim_input.deductible,
            num_diagnoses=claim_input.num_diagnoses,
            num_procedures=claim_input.num_procedures,
            length_of_stay=claim_input.length_of_stay,
            patient_age=claim_input.patient_age,
            chronic_conditions=claim_input.chronic_conditions,
            
            # Prediction Results
            is_fraud=result.get('is_fraud', False),
            fraud_probability=result.get('probability', 0.0),
            risk_level=result.get('risk_level', 'Unknown'),
            price_zone=result.get('price_zone_info', {}).get('zone', 'Unknown') if result.get('price_zone_info') else 'Unknown',
            expected_price=result.get('expected_price_info', {}).get('expected_without_gst', 0) if result.get('expected_price_info') else 0,
            
            # GST Information
            gst_amount=result.get('gst_info', {}).get('gst_amount', 0) if result.get('gst_info') else 0,
            total_with_gst=result.get('gst_info', {}).get('total_with_gst', 0) if result.get('gst_info') else 0,
            
            # Metadata
            submitted_at=datetime.utcnow()
        )
        
        db.add(submission)
        db.commit()
        db.refresh(submission)
        print(f"✅ Saved submission #{submission.id} to database")
        db.close()
        return submission.id
        
    except Exception as e:
        print(f"❌ Error saving submission: {e}")
        return None



# =============================================================================
# HOSPITAL LOOKUP ENDPOINTS
# =============================================================================

@app.get("/hospitals/types")
@limiter.limit("60/minute")
def get_hospital_types_endpoint(request: Request):
    """Get available hospital types for filtering."""
    return {"types": hospital_lookup.get_hospital_types()}

@app.get("/hospitals/search")
@limiter.limit("100/minute")
def search_hospitals_endpoint(request: Request, query: str, type: Optional[str] = None):
    """
    Search hospitals by name with optional type filter.
    
    Parameters:
    - query: Search text (partial name)
    - type: Filter by 'Government', 'Clinic', 'Private'
    """
    results = hospital_lookup.search_hospitals(query, type)
    return {"count": len(results), "hospitals": results}

@app.get("/hospitals/stats")
@limiter.limit("30/minute")
def get_hospital_stats_endpoint(request: Request):
    """Get statistics about the hospital database."""
    return hospital_lookup.get_hospital_stats()


# =============================================================================
# ML MODEL LOADING
# =============================================================================

# Get base directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Path to the trained machine learning model
MODEL_PATH = os.path.join(BASE_DIR, "ml", "model.pkl")
model_artifacts = None

def load_ml_model():
    """Load the ML model artifacts into memory."""
    global model_artifacts
    if os.path.exists(MODEL_PATH):
        try:
            model_artifacts = joblib.load(MODEL_PATH)
            print(f"✅ Model loaded successfully!")
            print(f"   Features used: {model_artifacts.get('feature_cols', [])}")
        except Exception as e:
            print(f"⚠️ Error loading ML model: {e}")
            model_artifacts = None
    else:
        print("⚠️ WARNING: ML Model not found. Using rule-based detection only.")

# =============================================================================
# INDIAN HEALTHCARE PRICING BENCHMARKS
# =============================================================================
"""
Why Provider Type Matters:
--------------------------
In India, healthcare pricing varies SIGNIFICANTLY based on provider type:

1. GOVERNMENT HOSPITALS (AIIMS, District Hospitals, PHCs)
   - Heavily subsidized by government
   - Very low costs (₹100-500 for most treatments)
   - If a government hospital claims ₹25,000 for a checkup → FRAUD!

2. PRIVATE CLINICS / NURSING HOMES
   - Moderate pricing
   - Typical costs: ₹500-3000 for treatments
   - If a clinic claims ₹50,000 for minor treatment → SUSPICIOUS!

3. CORPORATE HOSPITALS (Apollo, Fortis, Max, Medanta)
   - Premium pricing
   - Can legitimately charge ₹5000-50,000 for treatments
   - Higher amounts may still be legitimate here

The fraud detection compares claim amounts against these benchmarks
to identify claims that are unusually high for the provider type.
"""

PROVIDER_BENCHMARKS = {
    'Government': {
        'mean': 1000.0,            # Average claim is ₹1,000 (subsidized)
        'median': 500.0,           # Half the claims are below ₹500
        'p75': 1500.0,             # 75% of claims are below ₹1,500
        'p95_threshold': 3000.0,   # 95% of claims are below ₹3,000
        'description': 'Government Hospital (AIIMS, District Hospital, PHC)'
    },
    'Clinic': {
        'mean': 5000.0,            # Average claim is ₹5,000
        'median': 3000.0,
        'p75': 8000.0,
        'p95_threshold': 15000.0,  # 95% of claims are below ₹15,000
        'description': 'Private Clinic / Nursing Home'
    },
    'Private': {
        'mean': 10000.0,           # Average claim is ₹10,000
        'median': 6000.0,
        'p75': 15000.0,
        'p95_threshold': 40000.0,  # 95% of claims are below ₹40,000
        'description': 'Corporate Hospital (Apollo, Fortis, Max)'
    }
}

# =============================================================================
# DISEASE-SPECIFIC PRICING (Data-Driven from Dataset)
# =============================================================================
"""
WHY DISEASE-SPECIFIC PRICING?
-----------------------------
Different diseases have VERY different costs:
- Simple Checkup: ₹200-500
- Acute Respiratory Failure: ₹15,000-25,000
- Heart Surgery: ₹100,000-500,000

We can't use one threshold for all diseases!
This registry contains average prices for 6,000+ diagnoses from real data.
"""

import pandas as pd

# Load disease prices from CSV (generated from actual claims data)
DISEASE_PRICES = {}

def load_disease_prices():
    """Load disease prices from CSV."""
    global DISEASE_PRICES
    disease_prices_path = os.path.join(BASE_DIR, "data", "disease_prices.csv")
    
    try:
        if os.path.exists(disease_prices_path):
            df_prices = pd.read_csv(disease_prices_path)
            for _, row in df_prices.iterrows():
                DISEASE_PRICES[str(row['diagnosis_code'])] = {
                    'base_price': float(row['avg_price']),
                    'median_price': float(row['median_price']),
                    'min_price': float(row['min_price']),
                    'max_price': float(row['max_price']),
                    'claim_count': int(row['claim_count']),
                    'fraud_rate': float(row['fraud_rate'])
                }
            print(f"✅ Loaded prices for {len(DISEASE_PRICES)} diagnoses")
        else:
            print(f"⚠️ Disease prices file not found: {disease_prices_path}")
            DISEASE_PRICES = {}
    except Exception as e:
        print(f"⚠️ Could not load disease prices: {e}")
        DISEASE_PRICES = {}

# Provider type multipliers (hospitals can charge premium)
PROVIDER_MULTIPLIERS = {
    'Government': 0.7,   # 30% cheaper (subsidized)
    'Clinic': 1.0,       # Standard pricing
    'Private': 1.8       # 80% premium (corporate hospitals)
}

# Fraud detection thresholds (how much above expected is allowed)
FRAUD_THRESHOLDS = {
    'normal_max': 1.5,      # Up to 1.5x expected = Normal (fair profit)
    'elevated_max': 2.5,    # 1.5x to 2.5x expected = Elevated (premium care)
    # Above 2.5x = Suspicious (possible fraud)
}

def get_expected_price(diagnosis_code: str, provider_type: str) -> dict:
    """
    Calculate expected price for a disease at a specific provider type.
    
    Returns:
        dict with base_price, expected_price, min_allowed, max_normal, max_elevated
    """
    code = str(diagnosis_code).strip()
    
    # Get base price from disease registry
    if code in DISEASE_PRICES:
        base_price = DISEASE_PRICES[code]['base_price']
    elif code.lstrip('0') in DISEASE_PRICES:
        base_price = DISEASE_PRICES[code.lstrip('0')]['base_price']
    elif code.zfill(5) in DISEASE_PRICES:
        base_price = DISEASE_PRICES[code.zfill(5)]['base_price']
    else:
        # Unknown diagnosis - use provider benchmark
        base_price = PROVIDER_BENCHMARKS.get(provider_type, PROVIDER_BENCHMARKS['Clinic'])['mean']
    
    # Apply provider multiplier
    multiplier = PROVIDER_MULTIPLIERS.get(provider_type, 1.0)
    expected_price = base_price * multiplier
    
    # Calculate thresholds
    max_normal = expected_price * FRAUD_THRESHOLDS['normal_max']
    max_elevated = expected_price * FRAUD_THRESHOLDS['elevated_max']
    
    # Add GST to all values
    gst_multiplier = 1 + GST_RATE
    
    return {
        'base_price': round(base_price, 2),
        'provider_multiplier': multiplier,
        'expected_without_gst': round(expected_price, 2),
        'expected_with_gst': round(expected_price * gst_multiplier, 2),
        'max_normal': round(max_normal, 2),
        'max_normal_with_gst': round(max_normal * gst_multiplier, 2),
        'max_elevated': round(max_elevated, 2),
        'max_elevated_with_gst': round(max_elevated * gst_multiplier, 2),
        'gst_rate': f"{int(GST_RATE * 100)}%"
    }

def classify_price_zone(claim_amount: float, expected_info: dict) -> dict:
    """
    Classify claim amount into Normal/Elevated/Suspicious zone.
    
    Returns:
        dict with zone, color, and explanation
    """
    max_normal = expected_info['max_normal']
    max_elevated = expected_info['max_elevated']
    expected = expected_info['expected_without_gst']
    
    if claim_amount <= max_normal:
        ratio = claim_amount / expected if expected > 0 else 0
        return {
            'zone': 'Normal',
            'color': 'green',
            'emoji': '✅',
            'explanation': f'₹{claim_amount:,.0f} is within fair profit range (up to {FRAUD_THRESHOLDS["normal_max"]}x expected)',
            'ratio': round(ratio, 2)
        }
    elif claim_amount <= max_elevated:
        ratio = claim_amount / expected if expected > 0 else 0
        return {
            'zone': 'Elevated',
            'color': 'yellow',
            'emoji': '⚠️',
            'explanation': f'₹{claim_amount:,.0f} is {ratio:.1f}x expected - Premium pricing (may be justified)',
            'ratio': round(ratio, 2)
        }
    else:
        ratio = claim_amount / expected if expected > 0 else float('inf')
        return {
            'zone': 'Suspicious',
            'color': 'red',
            'emoji': '🚨',
            'explanation': f'₹{claim_amount:,.0f} is {ratio:.1f}x expected - Possible overcharging!',
            'ratio': round(ratio, 2)
        }

# =============================================================================
# RULE-BASED FRAUD DETECTION (Layer 1)
# =============================================================================
"""
WHY RULE-BASED DETECTION?
-------------------------
Some fraud is SO OBVIOUS that we don't need ML to detect it:

1. ₹50,000 claim for a checkup at a government hospital → FRAUD
2. 20 diagnoses for a single outpatient visit → UPCODING (padding diagnoses)
3. Age 150 years → DATA ERROR/FRAUD
4. Patient with 11 chronic conditions at age 25 → IMPOSSIBLE

These rules catch INSTANT RED FLAGS before we even run the ML model.
This makes the system faster and more reliable.
"""

def apply_fraud_rules(claim, provider_type: str = None) -> dict:
    """
    Applies rule-based fraud detection to a claim.
    
    This is LAYER 1 of our fraud detection system.
    It catches obvious anomalies that don't need machine learning.
    
    Parameters:
    -----------
    claim : ClaimInput
        The claim object containing amount, diagnoses, age, etc.
    provider_type : str
        Type of healthcare provider: "Government", "Clinic", or "Private"
    
    Returns:
    --------
    dict with keys:
        - is_fraud: bool (True if fraud detected)
        - risk_level: str ("Low", "Medium", "High", "Critical")
        - probability: float (0.0 to 1.0)
        - violations: list (human-readable explanations)
        - rule_triggered: bool (True if any rule was triggered)
        - benchmark_used: dict (the pricing benchmark used for comparison)
    
    Example:
    --------
    >>> claim = ClaimInput(amount=50000, provider_type="Government", ...)
    >>> result = apply_fraud_rules(claim, "Government")
    >>> print(result['violations'])
    ["🚨 PEER COMPARISON: ₹50,000 exceeds 95th percentile (₹2,000) for Government by 2400%"]
    """
    violations = []        # List of rule violations (human-readable)
    severity_scores = []   # Severity of each violation (0-100)
    
    # Get the pricing benchmark for this provider type
    # If provider_type is invalid, default to "Clinic" benchmarks
    benchmark = PROVIDER_BENCHMARKS.get(provider_type, PROVIDER_BENCHMARKS['Clinic'])
    
    # =========================================================================
    # RULE 1: DISEASE-SPECIFIC PRICING COMPARISON (Most Important Rule)
    # =========================================================================
    # Compare the claim amount against the EXPECTED PRICE for this specific disease
    # This is much more accurate than generic provider benchmarks!
    
    if provider_type and hasattr(claim, 'diagnosis_code'):
        # Get expected price for this disease at this provider type
        expected_info = get_expected_price(claim.diagnosis_code, provider_type)
        zone_info = classify_price_zone(claim.amount, expected_info)
        
        expected_price = expected_info['expected_without_gst']
        max_normal = expected_info['max_normal']
        max_elevated = expected_info['max_elevated']
        
        # Only flag as fraud if in SUSPICIOUS zone (above 2.5x expected)
        if zone_info['zone'] == 'Suspicious':
            violations.append(
                f"🚨 OVERPRICED: ₹{claim.amount:,.2f} is {zone_info['ratio']}x the expected price "
                f"(₹{expected_price:,.2f}) for this disease at {provider_type} facilities"
            )
            # Severity based on how much above elevated max
            severity_scores.append(min(95, 70 + (zone_info['ratio'] - 2.5) * 10))
        
        # Elevated zone is a warning, not fraud
        elif zone_info['zone'] == 'Elevated':
            violations.append(
                f"⚠️ PREMIUM PRICING: ₹{claim.amount:,.2f} is {zone_info['ratio']}x expected "
                f"(₹{expected_price:,.2f}) - may be justified for premium care"
            )
            severity_scores.append(35)  # Low severity for elevated (not fraud)
    
    # =========================================================================
    # RULE 2: EXCESSIVE DIAGNOSES (Upcoding Detection)
    # =========================================================================
    # Upcoding = Adding fake diagnoses to inflate the bill
    # A single visit shouldn't have more than 10-15 diagnoses
    
    if claim.num_diagnoses > 15:
        violations.append(
            f"📋 UPCODING SUSPECTED: {claim.num_diagnoses} diagnoses is excessive "
            f"(normal is 1-10)"
        )
        severity_scores.append(50)
    
    # =========================================================================
    # RULE 3: EXTREME AMOUNTS (Absolute Thresholds)
    # =========================================================================
    # Some amounts are suspicious regardless of provider type
    
    if claim.amount > 100000:  # ₹1 lakh+
        violations.append(
            f"💰 VERY HIGH AMOUNT: ₹{claim.amount:,.2f} requires manual review"
        )
        severity_scores.append(65)
    
    # =========================================================================
    # RULE 4: SUSPICIOUS PATIENT PROFILE
    # =========================================================================
    # Young patients with many chronic conditions is medically unlikely
    
    if claim.patient_age < 40 and claim.chronic_conditions >= 5:
        violations.append(
            f"👤 UNUSUAL PROFILE: Age {claim.patient_age} with "
            f"{claim.chronic_conditions} chronic conditions"
        )
        severity_scores.append(45)
    
    # =========================================================================
    # CALCULATE FINAL RISK LEVEL
    # =========================================================================
    
    if violations:
        max_severity = max(severity_scores)
        
        # Determine risk level based on highest severity score
        if max_severity >= 70:
            risk_level = "Critical"    # Definitely fraudulent
            is_fraud = True
        elif max_severity >= 50:
            risk_level = "High"        # Very likely fraudulent
            is_fraud = True
        elif max_severity >= 30:
            risk_level = "Medium"      # Suspicious, needs review
            is_fraud = False
        else:
            risk_level = "Low"         # Probably legitimate
            is_fraud = False
        
        # Convert severity to probability (0.0 to 0.99)
        probability = min(max_severity / 100, 0.99)
    else:
        # No rules triggered → Low risk
        risk_level = "Low"
        is_fraud = False
        probability = 0.05  # 5% baseline probability
    
    return {
        "is_fraud": is_fraud,
        "risk_level": risk_level,
        "probability": probability,
        "violations": violations,
        "rule_triggered": len(violations) > 0,
        "benchmark_used": benchmark
    }

# =============================================================================
# DATA MODELS (Pydantic Schemas)
# =============================================================================
"""
These models define the structure of data coming in (ClaimInput)
and going out (PredictionOutput) of the API.

Pydantic automatically validates the data and provides helpful error messages.
"""

class ClaimInput(BaseModel):
    """
    Input schema for claim analysis.
    
    This is what the frontend sends when a user wants to analyze a claim.
    All fields have default values so partial data can be submitted.
    
    Attributes:
    -----------
    provider_id : str
        Unique identifier for the healthcare provider
    provider_type : str
        Type of provider: "Government", "Clinic", or "Private"
    diagnosis_code : str
        ICD-9 or ICD-10 code for the diagnosis (e.g., "4019" for hypertension)
    claim_type : str
        "Inpatient" (admitted to hospital) or "Outpatient" (not admitted)
    amount : float
        Claim amount in Indian Rupees (₹)
    deductible : float
        Amount the patient pays out of pocket
    num_diagnoses : int
        Number of diagnosis codes on the claim
    num_procedures : int
        Number of procedures performed
    length_of_stay : int
        Days spent in hospital (0 for outpatient)
    patient_age : int
        Patient's age in years
    chronic_conditions : int
        Number of chronic conditions (0-11 tracked in Medicare data)
    """
    provider_id: str
    provider_type: str = "Clinic"           # Default to Clinic
    diagnosis_code: str
    claim_type: str = "Outpatient"          # Default to Outpatient
    amount: float
    deductible: float = 0
    num_diagnoses: int = 1
    num_procedures: int = 1
    length_of_stay: int = 0                 # 0 = Outpatient
    patient_age: int = 65                   # Default Medicare age
    chronic_conditions: int = 0


class PredictionOutput(BaseModel):
    """
    Output schema for fraud prediction results.
    
    This is what the API returns after analyzing a claim.
    
    Attributes:
    -----------
    is_fraud : bool
        True if the claim is predicted to be fraudulent
    probability : float
        Fraud probability (0.0 to 1.0)
    risk_level : str
        Human-readable risk: "Low", "Medium", "High", or "Critical"
    short_desc : str
        Short description of the diagnosis
    long_desc : str
        Full description of the diagnosis
    category_desc : str
        Category/chapter of the diagnosis code
    risk_factors : List[str]
        List of reasons why the claim is flagged
    detection_method : str
        How fraud was detected: "Rule-Based" or "ML Model"
    provider_type : str
        The provider type used for comparison
    benchmark_info : dict
        Pricing benchmark details for this provider type
    gst_info : dict
        GST calculation breakdown (18%)
    reason : Optional[str]
        Primary reason for the fraud flag
    """
    is_fraud: bool
    probability: float
    risk_level: str
    short_desc: str
    long_desc: str
    category_desc: str
    risk_factors: List[str]
    detection_method: str
    provider_type: str
    benchmark_info: dict
    gst_info: dict
    price_zone_info: dict = None  # NEW: Disease-specific pricing zone
    expected_price_info: dict = None  # NEW: Expected price breakdown
    reason: Optional[str] = None


# =============================================================================
# API ENDPOINTS
# =============================================================================

@app.get("/")
def read_root():
    """
    Root endpoint - Health check and welcome message.
    
    Use this to verify the API is running.
    
    Returns:
    --------
    dict with welcome message
    """
    return {
        "message": "🏥 Healthcare Fraud Detection API - India",
        "version": "1.0.0",
        "status": "running",
        "features": [
            "Provider Type Comparison (Govt/Clinic/Private)",
            "GST Calculation (18%)",
            "Rule-Based Detection",
            "ML Model Prediction"
        ]
    }


@app.get("/benchmarks")
def get_benchmarks():
    """
    Get pricing benchmarks for all provider types.
    
    Useful for frontend to display expected ranges.
    
    Returns:
    --------
    dict with benchmarks for Government, Clinic, and Private providers
    """
    return PROVIDER_BENCHMARKS


@app.get("/stats")
@limiter.limit("60/minute")
def get_stats(request: Request, db: Session = Depends(get_db)):
    """
    Get aggregate statistics about claims in the database.
    
    Parameters:
    -----------
    db : Session
        Database session (injected by FastAPI)
    
    Returns:
    --------
    dict with:
        - total_claims: Total number of claims
        - fraud_claims: Number of fraudulent claims
        - fraud_percentage: Percentage of fraud
        - inpatient_claims: Number of inpatient claims
        - outpatient_claims: Number of outpatient claims
    """
    # Count total claims
    total_claims = db.query(Claim).count()
    
    # Count fraudulent claims
    fraud_claims = db.query(Claim).filter(Claim.is_fraud == True).count()
    
    # Calculate fraud percentage
    fraud_percentage = (fraud_claims / total_claims * 100) if total_claims > 0 else 0
    
    # Count by claim type
    inpatient_claims = db.query(Claim).filter(Claim.claim_type == "Inpatient").count()
    outpatient_claims = db.query(Claim).filter(Claim.claim_type == "Outpatient").count()
    
    return {
        "total_claims": total_claims,
        "fraud_claims": fraud_claims,
        "fraud_percentage": round(fraud_percentage, 2),
        "inpatient_claims": inpatient_claims,
        "outpatient_claims": outpatient_claims
    }


@app.post("/predict", response_model=PredictionOutput)
@limiter.limit("30/minute")
def predict_fraud(request: Request, claim: ClaimInput):
    """
    🔍 MAIN ENDPOINT: Analyze a claim for potential fraud.
    
    This is the core functionality of the system. It uses a two-layer approach:
    
    LAYER 1: Rule-Based Detection
        - Compares amount against provider type benchmarks
        - Checks for excessive diagnoses (upcoding)
        - Validates patient profile
        - FAST: Returns immediately if obvious fraud found
    
    LAYER 2: ML Model
        - Uses provider-level aggregated features
        - Trained on 5,410 providers with 94.82% accuracy
        - Catches subtle patterns that rules miss
    
    Parameters:
    -----------
    claim : ClaimInput
        The claim details to analyze
    
    Returns:
    --------
    PredictionOutput with fraud prediction and explanations
    
    Example:
    --------
    POST /predict
    {
        "provider_id": "PRV001",
        "provider_type": "Government",
        "diagnosis_code": "4019",
        "amount": 25000,
        "patient_age": 65
    }
    
    Response:
    {
        "is_fraud": true,
        "probability": 0.85,
        "risk_level": "Critical",
        "risk_factors": ["₹25,000 exceeds 95th percentile (₹2,000) for Government..."],
        ...
    }
    """
    
    # Get pricing benchmark for this provider type
    benchmark = PROVIDER_BENCHMARKS.get(claim.provider_type, PROVIDER_BENCHMARKS['Clinic'])
    
    # =========================================================================
    # LAYER 1: Rule-Based Detection
    # =========================================================================
    rule_result = apply_fraud_rules(claim, claim.provider_type)
    
    # Get diagnosis information
    disease_info = get_disease_info(claim.diagnosis_code)
    
    # Build benchmark info for the response
    benchmark_info = {
        "provider_type": claim.provider_type,
        "expected_average": benchmark['mean'],
        "p95_threshold": benchmark['p95_threshold'],
        "your_amount": claim.amount,
        "comparison": "Above 95th percentile" if claim.amount > benchmark['p95_threshold'] 
                      else "Above average" if claim.amount > benchmark['mean']
                      else "Within normal range"
    }
    
    # Calculate GST (18%)
    base_amount = claim.amount
    gst_amount = round(base_amount * GST_RATE, 2)
    total_with_gst = round(base_amount + gst_amount, 2)
    
    gst_info = {
        "base_amount": base_amount,
        "gst_rate": f"{int(GST_RATE * 100)}%",
        "gst_amount": gst_amount,
        "total_with_gst": total_with_gst
    }
    
    # =========================================================================
    # NEW: Disease-Specific Pricing
    # =========================================================================
    # Get expected price based on diagnosis code and provider type
    expected_price_info = get_expected_price(claim.diagnosis_code, claim.provider_type)
    
    # Classify claim into Normal/Elevated/Suspicious zone
    price_zone_info = classify_price_zone(claim.amount, expected_price_info)
    
    # If rule-based detection found HIGH severity fraud, return immediately
    # No need to run ML model for obvious fraud
    if rule_result["rule_triggered"] and rule_result["probability"] >= 0.4:
        result = {
            "is_fraud": rule_result["is_fraud"],
            "probability": rule_result["probability"],
            "risk_level": rule_result["risk_level"],
            "short_desc": disease_info['short_desc'],
            "long_desc": disease_info['long_desc'],
            "category_desc": disease_info['category_desc'],
            "risk_factors": rule_result["violations"],
            "detection_method": f"Rule-Based Detection (vs {claim.provider_type} benchmark)",
            "provider_type": claim.provider_type,
            "benchmark_info": benchmark_info,
            "gst_info": gst_info,
            "price_zone_info": price_zone_info,
            "expected_price_info": expected_price_info,
            "reason": rule_result["violations"][0] if rule_result["violations"] else None
        }
        # Save user submission to database
        save_user_submission(claim, result)
        return result
    
    # =========================================================================
    # LAYER 2: ML Model Prediction
    # =========================================================================
    
    # If model not loaded, return rule-based result
    if not model_artifacts:
        result = {
            "is_fraud": rule_result["is_fraud"],
            "probability": rule_result["probability"],
            "risk_level": rule_result["risk_level"],
            "short_desc": disease_info['short_desc'],
            "long_desc": disease_info['long_desc'],
            "category_desc": disease_info['category_desc'],
            "risk_factors": rule_result["violations"] if rule_result["violations"] else [
                f"✅ Amount ₹{claim.amount:,.2f} is within normal range for {claim.provider_type}"
            ],
            "detection_method": "Rule-Based Only",
            "provider_type": claim.provider_type,
            "benchmark_info": benchmark_info,
            "gst_info": gst_info,
            "price_zone_info": price_zone_info,
            "expected_price_info": expected_price_info,
            "reason": None
        }
        # Save user submission to database
        save_user_submission(claim, result)
        return result
    
    try:
        # Load model components
        model = model_artifacts['model']
        scaler = model_artifacts.get('scaler')
        feature_cols = model_artifacts['feature_cols']
        
        # Build features for ML model
        # The model expects provider-level aggregated features
        # For a single claim, we simulate this with the claim values
        feature_dict = {
            'amount_sum': claim.amount,
            'amount_mean': claim.amount,
            'amount_std': 0,
            'amount_max': claim.amount,
            'amount_min': claim.amount,
            'total_claims': 1,
            'unique_patients': 1,
            'claims_per_patient': 1,
            'length_of_stay_max': claim.length_of_stay,
            'length_of_stay_sum': claim.length_of_stay,
            'length_of_stay_mean': claim.length_of_stay,
            'num_diagnoses_mean': claim.num_diagnoses,
            'num_diagnoses_sum': claim.num_diagnoses,
            'num_diagnoses_max': claim.num_diagnoses,
            'num_procedures_mean': claim.num_procedures,
            'num_procedures_sum': claim.num_procedures,
            'num_procedures_max': claim.num_procedures,
            'patient_age_mean': claim.patient_age,
            'patient_age_std': 0,
            'patient_age_min': claim.patient_age,
            'patient_age_max': claim.patient_age,
            'chronic_conditions_mean': claim.chronic_conditions,
            'chronic_conditions_max': claim.chronic_conditions,
            'inpatient_ratio': 1.0 if claim.claim_type == "Inpatient" else 0.0,
            'claim_type_enc': 1 if claim.claim_type == "Inpatient" else 0,
            'deductible_sum': claim.deductible,
            'deductible_mean': claim.deductible,
            'deductible_max': claim.deductible,
            'high_amount_claims': 1 if claim.amount > 5000 else 0,
            'avg_amount_per_patient': claim.amount,
            'amount_per_diagnosis': claim.amount / max(1, claim.num_diagnoses)
        }
        
        # Create feature vector in the correct order
        features = [feature_dict.get(col, 0) for col in feature_cols]
        feature_df = pd.DataFrame([features], columns=feature_cols)
        
        # Scale features if scaler exists
        if scaler:
            features_scaled = scaler.transform(feature_df)
        else:
            features_scaled = feature_df.values
        
        # Get ML prediction
        ml_prediction = model.predict(features_scaled)[0]
        ml_probability = model.predict_proba(features_scaled)[0][1]
        
        # Combine ML and rule-based results
        if rule_result["rule_triggered"]:
            # Average the probabilities if rules were triggered
            combined_prob = (ml_probability + rule_result["probability"]) / 2
        else:
            combined_prob = ml_probability
        
        # Determine final risk level
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
        
        # Build risk factors list
        risk_factors = rule_result["violations"].copy() if rule_result["violations"] else []
        if not risk_factors:
            risk_factors = [
                f"✅ Amount ₹{claim.amount:,.2f} is within normal range for "
                f"{claim.provider_type} (avg: ₹{benchmark['mean']:,.2f})"
            ]
        
        result = {
            "is_fraud": is_fraud,
            "probability": combined_prob,
            "risk_level": risk_level,
            "short_desc": disease_info['short_desc'],
            "long_desc": disease_info['long_desc'],
            "category_desc": disease_info['category_desc'],
            "risk_factors": risk_factors,
            "detection_method": f"Provider Comparison ({claim.provider_type})",
            "provider_type": claim.provider_type,
            "benchmark_info": benchmark_info,
            "gst_info": gst_info,
            "price_zone_info": price_zone_info,
            "expected_price_info": expected_price_info,
            "reason": risk_factors[0] if is_fraud else None
        }
        
        # Save user submission to database
        save_user_submission(claim, result)
        
        return result
        
    except Exception as e:
        # If ML fails, fall back to rule-based result
        print(f"❌ ML Error: {e}")
        return {
            "is_fraud": rule_result["is_fraud"],
            "probability": rule_result["probability"],
            "risk_level": rule_result["risk_level"],
            "short_desc": disease_info['short_desc'],
            "long_desc": disease_info['long_desc'],
            "category_desc": disease_info['category_desc'],
            "risk_factors": rule_result["violations"] if rule_result["violations"] else ["Analysis complete"],
            "detection_method": "Rule-Based",
            "provider_type": claim.provider_type,
            "benchmark_info": benchmark_info,
            "gst_info": gst_info,
            "price_zone_info": price_zone_info,
            "expected_price_info": expected_price_info,
            "reason": str(e)
        }


@app.get("/claims", response_model=List[dict])
def get_recent_claims(limit: int = 50, db: Session = Depends(get_db)):
    """
    Get the most recent claims from the database.
    
    Used to populate the "Live Transactions" table on the dashboard.
    
    Parameters:
    -----------
    limit : int
        Maximum number of claims to return (default: 50)
    db : Session
        Database session
    
    Returns:
    --------
    List of claim objects with diagnosis information
    """
    # Get recent claims ordered by ID (most recent first)
    claims = db.query(Claim).order_by(Claim.id.desc()).limit(limit).all()
    
    results = []
    for c in claims:
        # Look up diagnosis information
        info = get_disease_info(c.diagnosis_code)
        
        results.append({
            "id": c.id,
            "provider_id": c.provider_id,
            "patient_id": c.patient_id,
            "diagnosis_code": c.diagnosis_code,
            "disease_name": info['short_desc'],      # Human-readable name
            "claim_type": c.claim_type,
            "amount": c.amount,
            "is_fraud": c.is_fraud,
            "timestamp": str(c.timestamp) if c.timestamp else None
        })
    
    return results


@app.get("/user-submissions")
def get_user_submissions(limit: int = 100):
    """
    Get all user-submitted claims with their prediction results.
    
    This endpoint returns claims that were submitted through the Analyze Claim form.
    
    Parameters:
    -----------
    limit : int
        Maximum number of submissions to return (default: 100)
    
    Returns:
    --------
    List of user submissions with input data and prediction results
    """
    try:
        db = UserSessionLocal()
        submissions = db.query(UserSubmission).order_by(
            UserSubmission.submitted_at.desc()
        ).limit(limit).all()
        
        results = []
        for s in submissions:
            results.append({
                "id": s.id,
                "provider_id": s.provider_id,
                "provider_type": s.provider_type,
                "diagnosis_code": s.diagnosis_code,
                "disease_name": s.disease_name,
                "claim_type": s.claim_type,
                "amount": s.amount,
                "patient_age": s.patient_age,
                "is_fraud": s.is_fraud,
                "fraud_probability": s.fraud_probability,
                "risk_level": s.risk_level,
                "price_zone": s.price_zone,
                "expected_price": s.expected_price,
                "gst_amount": s.gst_amount,
                "total_with_gst": s.total_with_gst,
                "submitted_at": str(s.submitted_at) if s.submitted_at else None
            })
        
        db.close()
        return {
            "total_submissions": len(results),
            "submissions": results
        }
        
    except Exception as e:
        print(f"❌ Error fetching submissions: {e}")
        return {"total_submissions": 0, "submissions": [], "error": str(e)}


@app.get("/user-submissions/stats")
def get_submission_stats():
    """
    Get statistics about user submissions.
    """
    try:
        db = UserSessionLocal()
        
        total = db.query(UserSubmission).count()
        fraud_count = db.query(UserSubmission).filter(UserSubmission.is_fraud == True).count()
        
        db.close()
        return {
            "total_submissions": total,
            "fraud_flagged": fraud_count,
            "legitimate": total - fraud_count,
            "fraud_rate": round((fraud_count / total * 100) if total > 0 else 0, 2)
        }
        
    except Exception as e:
        return {"total_submissions": 0, "fraud_flagged": 0, "legitimate": 0, "fraud_rate": 0}


# =============================================================================
# AI ASSISTANT ENDPOINTS
# =============================================================================

class ChatRequest(BaseModel):
    """Request schema for AI chat"""
    message: str
    claim_context: Optional[dict] = None

class ReportRequest(BaseModel):
    """Request schema for report generation"""
    claim_data: dict
    prediction_result: dict


@app.get("/ai/test")
@limiter.limit("20/minute")
def test_ai(request: Request):
    """
    Test AI service connection.
    
    Returns:
    --------
    dict with status and test message from Gemini
    """
    result = test_ai_service()
    return result


@app.post("/ai/chat")
@limiter.limit("10/minute")
async def ai_chat(request: Request, chat_req: ChatRequest):
    """
    AI Chatbot for fraud-related queries.
    
    Send a message and optionally include claim context for more relevant responses.
    
    Parameters:
    -----------
    message : str
        User's question
    claim_context : dict, optional
        Current claim data for context
    
    Returns:
    --------
    dict with AI response
    """
    try:
        response = await chat_with_ai(chat_req.message, chat_req.claim_context)
        return {
            "success": True,
            "response": response
        }
    except Exception as e:
        return {
            "success": False,
            "response": f"Error: {str(e)}"
        }


@app.post("/ai/report")
@limiter.limit("10/minute")
async def generate_report(request: Request, report_req: ReportRequest):
    """
    Generate fraud investigation report using AI.
    
    Parameters:
    -----------
    claim_data : dict
        Original claim input
    prediction_result : dict
        Fraud prediction results
    
    Returns:
    --------
    dict with markdown-formatted report
    """
    try:
        report = await generate_fraud_report(report_req.claim_data, report_req.prediction_result)
        return {
            "success": True,
            "report": report
        }
    except Exception as e:
        return {
            "success": False,
            "report": f"Error generating report: {str(e)}"
        }


@app.get("/ai/insights")
async def get_insights(db: Session = Depends(get_db)):
    """
    Get AI-powered intelligent insights about fraud patterns.
    
    Analyzes overall statistics and recent claims to identify:
    - Key fraud patterns
    - Anomalies
    - Recommendations
    - Priority actions
    
    Returns:
    --------
    dict with insights data
    """
    try:
        # Get stats
        total_claims = db.query(Claim).count()
        fraud_claims = db.query(Claim).filter(Claim.is_fraud == True).count()
        inpatient = db.query(Claim).filter(Claim.claim_type == "Inpatient").count()
        outpatient = db.query(Claim).filter(Claim.claim_type == "Outpatient").count()
        
        stats = {
            "total_claims": total_claims,
            "fraud_claims": fraud_claims,
            "fraud_percentage": round((fraud_claims / total_claims * 100) if total_claims > 0 else 0, 2),
            "inpatient_claims": inpatient,
            "outpatient_claims": outpatient
        }
        
        # Get recent claims
        recent = db.query(Claim).order_by(Claim.id.desc()).limit(50).all()
        recent_claims = []
        for c in recent:
            recent_claims.append({
                "claim_id": c.claim_id,
                "provider_id": c.provider_id,
                "diagnosis_code": c.diagnosis_code,
                "amount": c.amount,
                "is_fraud": c.is_fraud,
                "claim_type": c.claim_type
            })
        
        # Get AI insights
        insights = await get_intelligent_insights(stats, recent_claims)
        
        return {
            "success": True,
            "stats": stats,
            "insights": insights
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "insights": {
                "key_patterns": [],
                "anomalies": [],
                "recommendations": [],
                "risk_summary": "Unable to generate insights",
                "priority_actions": []
            }
        }


# =============================================================================
# END OF FILE
# =============================================================================
"""
To run this API:
    uvicorn backend.main:app --reload --port 8000

API will be available at:
    http://localhost:8000

Interactive docs at:
    http://localhost:8000/docs (Swagger UI)
    http://localhost:8000/redoc (ReDoc)
"""
