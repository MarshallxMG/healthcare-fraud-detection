"""
Database models and connection for Healthcare Fraud Detection
"""
from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import os

# Get the project base directory (parent of backend/)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

# Ensure data directory exists
os.makedirs(DATA_DIR, exist_ok=True)

# Database configuration - use environment variables or default to local paths
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{os.path.join(DATA_DIR, 'claims.db')}")
USER_SUBMISSIONS_DB = os.getenv("USER_SUBMISSIONS_DB", f"sqlite:///{os.path.join(DATA_DIR, 'user_submissions.db')}")

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
user_engine = create_engine(USER_SUBMISSIONS_DB, connect_args={"check_same_thread": False})

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
UserSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=user_engine)

Base = declarative_base()
UserBase = declarative_base()

class Claim(Base):
    """Claims table model - Original dataset"""
    __tablename__ = "claims"
    
    id = Column(Integer, primary_key=True, index=True)
    claim_id = Column(String, index=True)
    provider_id = Column(String, index=True)
    patient_id = Column(String, index=True)
    claim_type = Column(String)  # Inpatient or Outpatient
    diagnosis_code = Column(String, index=True)
    amount = Column(Float)
    deductible = Column(Float)
    num_diagnoses = Column(Integer)
    num_procedures = Column(Integer)
    length_of_stay = Column(Integer)
    patient_age = Column(Integer)
    chronic_conditions = Column(Integer)
    amount_per_diagnosis = Column(Float)
    is_fraud = Column(Boolean, default=False)
    timestamp = Column(DateTime)


class UserSubmission(UserBase):
    """User-submitted claims for analysis - Stores all user entries"""
    __tablename__ = "user_submissions"
    
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    
    # User Input Fields
    provider_id = Column(String, index=True)
    provider_type = Column(String)  # Government, Clinic, Private
    diagnosis_code = Column(String, index=True)
    disease_name = Column(String)  # Looked up from ICD codes
    claim_type = Column(String)  # Inpatient or Outpatient
    amount = Column(Float)
    deductible = Column(Float)
    num_diagnoses = Column(Integer)
    num_procedures = Column(Integer)
    length_of_stay = Column(Integer)
    patient_age = Column(Integer)
    chronic_conditions = Column(Integer)
    
    # Prediction Results
    is_fraud = Column(Boolean, default=False)
    fraud_probability = Column(Float)  # 0.0 to 1.0
    risk_level = Column(String)  # Low, Medium, High, Critical
    price_zone = Column(String)  # Normal, Elevated, Suspicious
    expected_price = Column(Float)
    
    # GST Information
    gst_amount = Column(Float)
    total_with_gst = Column(Float)
    
    # Metadata
    submitted_at = Column(DateTime, default=datetime.utcnow)
    ip_address = Column(String, nullable=True)


# Create tables
Base.metadata.create_all(bind=engine)
UserBase.metadata.create_all(bind=user_engine)

print("✅ User submissions database ready: user_submissions.db")
