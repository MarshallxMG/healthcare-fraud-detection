"""
Healthcare Fraud Detection - Streamlit App
Deployed on Streamlit Cloud
"""
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime

# =============================================================================
# PAGE CONFIG
# =============================================================================
st.set_page_config(
    page_title="Healthcare Fraud Detection",
    page_icon="🏥",
    layout="wide"
)

# =============================================================================
# CONSTANTS
# =============================================================================
GST_RATE = 0.18  # 18% GST

PROVIDER_BENCHMARKS = {
    'Government': {
        'mean': 1000.0,
        'median': 500.0,
        'p75': 1500.0,
        'p95_threshold': 3000.0,
        'description': 'Government Hospital (AIIMS, District Hospital, PHC)'
    },
    'Clinic': {
        'mean': 5000.0,
        'median': 3000.0,
        'p75': 8000.0,
        'p95_threshold': 15000.0,
        'description': 'Private Clinic / Nursing Home'
    },
    'Private': {
        'mean': 10000.0,
        'median': 6000.0,
        'p75': 15000.0,
        'p95_threshold': 40000.0,
        'description': 'Corporate Hospital (Apollo, Fortis, Max)'
    }
}

PROVIDER_MULTIPLIERS = {
    'Government': 0.7,
    'Clinic': 1.0,
    'Private': 1.8
}

FRAUD_THRESHOLDS = {
    'normal_max': 1.5,
    'elevated_max': 2.5,
}

# =============================================================================
# LOAD RESOURCES
# =============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource
def load_ml_model():
    """Load the ML model artifacts."""
    model_path = os.path.join(BASE_DIR, "ml", "model.pkl")
    if os.path.exists(model_path):
        try:
            return joblib.load(model_path)
        except Exception as e:
            st.warning(f"Could not load ML model: {e}")
            return None
    return None

@st.cache_data
def load_disease_prices():
    """Load disease prices from CSV."""
    disease_prices_path = os.path.join(BASE_DIR, "data", "disease_prices.csv")
    if os.path.exists(disease_prices_path):
        try:
            df = pd.read_csv(disease_prices_path)
            prices = {}
            for _, row in df.iterrows():
                prices[str(row['diagnosis_code'])] = {
                    'base_price': float(row['avg_price']),
                    'median_price': float(row['median_price']),
                    'min_price': float(row['min_price']),
                    'max_price': float(row['max_price']),
                    'claim_count': int(row['claim_count']),
                    'fraud_rate': float(row['fraud_rate'])
                }
            return prices
        except Exception as e:
            st.warning(f"Could not load disease prices: {e}")
    return {}

@st.cache_data
def load_icd_codes():
    """Load ICD-9 and ICD-10 codes into lookup dictionary."""
    icd_lookup = {}
    
    # ICD-9 file
    icd9_path = os.path.join(BASE_DIR, "Dataset", "Synthetic Dataset", "ICD9codes.csv")
    if os.path.exists(icd9_path):
        try:
            df9 = pd.read_csv(icd9_path, header=None, dtype=str)
            for _, row in df9.iterrows():
                code = str(row[2]).strip() if pd.notna(row[2]) else str(row[0]).strip()
                short_desc = str(row[3]).strip() if pd.notna(row[3]) else 'Unknown'
                long_desc = str(row[4]).strip() if pd.notna(row[4]) else short_desc
                
                if code and code != 'nan':
                    entry = {
                        'short_desc': short_desc,
                        'long_desc': long_desc,
                        'category_desc': short_desc
                    }
                    icd_lookup[code] = entry
                    icd_lookup[code.lstrip('0')] = entry
                    icd_lookup[code.zfill(4)] = entry
                    icd_lookup[code.zfill(5)] = entry
        except Exception as e:
            st.warning(f"Could not load ICD-9 codes: {e}")
    
    # ICD-10 file
    icd10_path = os.path.join(BASE_DIR, "Dataset", "Synthetic Dataset", "ICD10codes.csv")
    if os.path.exists(icd10_path):
        try:
            df10 = pd.read_csv(icd10_path, header=None, dtype=str)
            for _, row in df10.iterrows():
                full_code = str(row[2]).strip() if len(row) > 2 and pd.notna(row[2]) else str(row[0]).strip()
                long_desc = str(row[3]).strip() if len(row) > 3 and pd.notna(row[3]) else 'Unknown'
                short_desc = str(row[4]).strip() if len(row) > 4 and pd.notna(row[4]) else long_desc
                
                if full_code and full_code != 'nan' and len(full_code) >= 3:
                    entry = {
                        'short_desc': short_desc,
                        'long_desc': long_desc,
                        'category_desc': short_desc
                    }
                    if full_code not in icd_lookup:  # Don't override ICD-9
                        icd_lookup[full_code] = entry
        except Exception as e:
            st.warning(f"Could not load ICD-10 codes: {e}")
    
    return icd_lookup

@st.cache_data
def load_top_expensive_diseases():
    """Load top 10 most expensive diseases from CSV."""
    diseases_path = os.path.join(BASE_DIR, "data", "all_diseases_list.csv")
    if os.path.exists(diseases_path):
        try:
            df = pd.read_csv(diseases_path)
            # Already sorted by Avg Price descending, take top 10
            return df.head(10)
        except Exception as e:
            st.warning(f"Could not load diseases list: {e}")
    return None

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_disease_info(code: str, icd_lookup: dict) -> dict:
    """Get disease information for a diagnosis code."""
    if not code:
        return {
            'short_desc': 'Unknown Diagnosis',
            'long_desc': 'No diagnosis code provided',
            'category_desc': 'Unknown'
        }
    
    code = str(code).strip().upper()
    
    # Try exact match
    if code in icd_lookup:
        return icd_lookup[code]
    
    # Try without dots
    code_no_dot = code.replace('.', '')
    if code_no_dot in icd_lookup:
        return icd_lookup[code_no_dot]
    
    # Try with padding
    for padded in [code.zfill(4), code.zfill(5), code.lstrip('0')]:
        if padded in icd_lookup:
            return icd_lookup[padded]
    
    return {
        'short_desc': f'Diagnosis {code}',
        'long_desc': f'Diagnosis code {code}',
        'category_desc': 'Unknown category'
    }

def get_expected_price(diagnosis_code: str, provider_type: str, disease_prices: dict) -> dict:
    """Calculate expected price for a disease at a specific provider type."""
    code = str(diagnosis_code).strip()
    
    # Get base price from disease registry
    if code in disease_prices:
        base_price = disease_prices[code]['base_price']
    elif code.lstrip('0') in disease_prices:
        base_price = disease_prices[code.lstrip('0')]['base_price']
    elif code.zfill(5) in disease_prices:
        base_price = disease_prices[code.zfill(5)]['base_price']
    else:
        base_price = PROVIDER_BENCHMARKS.get(provider_type, PROVIDER_BENCHMARKS['Clinic'])['mean']
    
    multiplier = PROVIDER_MULTIPLIERS.get(provider_type, 1.0)
    expected_price = base_price * multiplier
    
    max_normal = expected_price * FRAUD_THRESHOLDS['normal_max']
    max_elevated = expected_price * FRAUD_THRESHOLDS['elevated_max']
    
    gst_multiplier = 1 + GST_RATE
    
    return {
        'base_price': round(base_price, 2),
        'provider_multiplier': multiplier,
        'expected_without_gst': round(expected_price, 2),
        'expected_with_gst': round(expected_price * gst_multiplier, 2),
        'max_normal': round(max_normal, 2),
        'max_elevated': round(max_elevated, 2),
        'gst_rate': f"{int(GST_RATE * 100)}%"
    }

def classify_price_zone(claim_amount: float, expected_info: dict) -> dict:
    """Classify claim amount into Normal/Elevated/Suspicious zone."""
    max_normal = expected_info['max_normal']
    max_elevated = expected_info['max_elevated']
    expected = expected_info['expected_without_gst']
    
    if claim_amount <= max_normal:
        ratio = claim_amount / expected if expected > 0 else 0
        return {
            'zone': 'Normal',
            'color': 'green',
            'emoji': '✅',
            'explanation': f'₹{claim_amount:,.0f} is within fair range',
            'ratio': round(ratio, 2)
        }
    elif claim_amount <= max_elevated:
        ratio = claim_amount / expected if expected > 0 else 0
        return {
            'zone': 'Elevated',
            'color': 'orange',
            'emoji': '⚠️',
            'explanation': f'₹{claim_amount:,.0f} is {ratio:.1f}x expected - Premium pricing',
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

def apply_fraud_rules(claim_data: dict, provider_type: str, disease_prices: dict) -> dict:
    """Apply rule-based fraud detection."""
    violations = []
    severity_scores = []
    
    benchmark = PROVIDER_BENCHMARKS.get(provider_type, PROVIDER_BENCHMARKS['Clinic'])
    
    # Rule 1: Disease-specific pricing
    expected_info = get_expected_price(claim_data['diagnosis_code'], provider_type, disease_prices)
    zone_info = classify_price_zone(claim_data['amount'], expected_info)
    
    if zone_info['zone'] == 'Suspicious':
        violations.append(
            f"🚨 OVERPRICED: ₹{claim_data['amount']:,.2f} is {zone_info['ratio']}x the expected price "
            f"(₹{expected_info['expected_without_gst']:,.2f}) for this disease at {provider_type} facilities"
        )
        severity_scores.append(min(95, 70 + (zone_info['ratio'] - 2.5) * 10))
    elif zone_info['zone'] == 'Elevated':
        violations.append(
            f"⚠️ PREMIUM PRICING: ₹{claim_data['amount']:,.2f} is {zone_info['ratio']}x expected"
        )
        severity_scores.append(35)
    
    # Rule 2: Excessive diagnoses
    if claim_data['num_diagnoses'] > 15:
        violations.append(
            f"📋 UPCODING SUSPECTED: {claim_data['num_diagnoses']} diagnoses is excessive"
        )
        severity_scores.append(50)
    
    # Rule 3: Extreme amounts
    if claim_data['amount'] > 100000:
        violations.append(
            f"💰 VERY HIGH AMOUNT: ₹{claim_data['amount']:,.2f} requires manual review"
        )
        severity_scores.append(65)
    
    # Rule 4: Suspicious patient profile
    if claim_data['patient_age'] < 40 and claim_data['chronic_conditions'] >= 5:
        violations.append(
            f"👤 UNUSUAL PROFILE: Age {claim_data['patient_age']} with "
            f"{claim_data['chronic_conditions']} chronic conditions"
        )
        severity_scores.append(45)
    
    # Calculate final risk
    if violations:
        max_severity = max(severity_scores)
        
        if max_severity >= 70:
            risk_level = "Critical"
            is_fraud = True
        elif max_severity >= 50:
            risk_level = "High"
            is_fraud = True
        elif max_severity >= 30:
            risk_level = "Medium"
            is_fraud = False
        else:
            risk_level = "Low"
            is_fraud = False
        
        probability = min(max_severity / 100, 0.99)
    else:
        risk_level = "Low"
        is_fraud = False
        probability = 0.05
    
    return {
        "is_fraud": is_fraud,
        "risk_level": risk_level,
        "probability": probability,
        "violations": violations,
        "rule_triggered": len(violations) > 0,
        "benchmark_used": benchmark
    }

def predict_fraud(claim_data: dict, model_artifacts, disease_prices: dict, icd_lookup: dict) -> dict:
    """Main fraud prediction function."""
    provider_type = claim_data['provider_type']
    benchmark = PROVIDER_BENCHMARKS.get(provider_type, PROVIDER_BENCHMARKS['Clinic'])
    
    # Rule-based detection
    rule_result = apply_fraud_rules(claim_data, provider_type, disease_prices)
    
    # Get disease info
    disease_info = get_disease_info(claim_data['diagnosis_code'], icd_lookup)
    
    # Expected price info
    expected_price_info = get_expected_price(claim_data['diagnosis_code'], provider_type, disease_prices)
    price_zone_info = classify_price_zone(claim_data['amount'], expected_price_info)
    
    # GST calculation
    base_amount = claim_data['amount']
    gst_amount = round(base_amount * GST_RATE, 2)
    total_with_gst = round(base_amount + gst_amount, 2)
    
    gst_info = {
        "base_amount": base_amount,
        "gst_rate": f"{int(GST_RATE * 100)}%",
        "gst_amount": gst_amount,
        "total_with_gst": total_with_gst
    }
    
    benchmark_info = {
        "provider_type": provider_type,
        "expected_average": benchmark['mean'],
        "p95_threshold": benchmark['p95_threshold'],
        "your_amount": claim_data['amount']
    }
    
    # If rule-based found high severity fraud, return immediately
    if rule_result["rule_triggered"] and rule_result["probability"] >= 0.4:
        return {
            "is_fraud": rule_result["is_fraud"],
            "probability": rule_result["probability"],
            "risk_level": rule_result["risk_level"],
            "short_desc": disease_info['short_desc'],
            "long_desc": disease_info['long_desc'],
            "risk_factors": rule_result["violations"],
            "detection_method": f"Rule-Based Detection (vs {provider_type} benchmark)",
            "provider_type": provider_type,
            "benchmark_info": benchmark_info,
            "gst_info": gst_info,
            "price_zone_info": price_zone_info,
            "expected_price_info": expected_price_info
        }
    
    # ML Model prediction (if available)
    if model_artifacts:
        try:
            model = model_artifacts['model']
            scaler = model_artifacts.get('scaler')
            feature_cols = model_artifacts['feature_cols']
            
            # Build features
            feature_dict = {
                'amount_sum': claim_data['amount'],
                'amount_mean': claim_data['amount'],
                'amount_std': 0,
                'amount_max': claim_data['amount'],
                'amount_min': claim_data['amount'],
                'total_claims': 1,
                'unique_patients': 1,
                'claims_per_patient': 1,
                'length_of_stay_max': claim_data['length_of_stay'],
                'length_of_stay_sum': claim_data['length_of_stay'],
                'length_of_stay_mean': claim_data['length_of_stay'],
                'num_diagnoses_mean': claim_data['num_diagnoses'],
                'num_diagnoses_sum': claim_data['num_diagnoses'],
                'num_diagnoses_max': claim_data['num_diagnoses'],
                'num_procedures_mean': claim_data['num_procedures'],
                'num_procedures_sum': claim_data['num_procedures'],
                'num_procedures_max': claim_data['num_procedures'],
                'patient_age_mean': claim_data['patient_age'],
                'patient_age_std': 0,
                'patient_age_min': claim_data['patient_age'],
                'patient_age_max': claim_data['patient_age'],
                'chronic_conditions_mean': claim_data['chronic_conditions'],
                'chronic_conditions_max': claim_data['chronic_conditions'],
                'inpatient_ratio': 1.0 if claim_data['claim_type'] == "Inpatient" else 0.0,
                'claim_type_enc': 1 if claim_data['claim_type'] == "Inpatient" else 0,
                'deductible_sum': claim_data['deductible'],
                'deductible_mean': claim_data['deductible'],
                'deductible_max': claim_data['deductible'],
                'high_amount_claims': 1 if claim_data['amount'] > 5000 else 0,
                'avg_amount_per_patient': claim_data['amount'],
                'amount_per_diagnosis': claim_data['amount'] / max(1, claim_data['num_diagnoses'])
            }
            
            # Create feature array
            features = np.array([[feature_dict.get(col, 0) for col in feature_cols]])
            
            # Scale if scaler available
            if scaler:
                features = scaler.transform(features)
            
            # Predict
            prediction = model.predict(features)[0]
            probability = model.predict_proba(features)[0][1] if hasattr(model, 'predict_proba') else float(prediction)
            
            # Determine risk level
            if probability >= 0.7:
                risk_level = "Critical"
                is_fraud = True
            elif probability >= 0.5:
                risk_level = "High"
                is_fraud = True
            elif probability >= 0.3:
                risk_level = "Medium"
                is_fraud = False
            else:
                risk_level = "Low"
                is_fraud = False
            
            risk_factors = rule_result["violations"] if rule_result["violations"] else [
                f"✅ Amount ₹{claim_data['amount']:,.2f} is within normal range for {provider_type}"
            ]
            
            return {
                "is_fraud": is_fraud,
                "probability": probability,
                "risk_level": risk_level,
                "short_desc": disease_info['short_desc'],
                "long_desc": disease_info['long_desc'],
                "risk_factors": risk_factors,
                "detection_method": "ML Model + Rule-Based",
                "provider_type": provider_type,
                "benchmark_info": benchmark_info,
                "gst_info": gst_info,
                "price_zone_info": price_zone_info,
                "expected_price_info": expected_price_info
            }
            
        except Exception as e:
            st.warning(f"ML Model error: {e}")
    
    # Fallback to rule-based only
    return {
        "is_fraud": rule_result["is_fraud"],
        "probability": rule_result["probability"],
        "risk_level": rule_result["risk_level"],
        "short_desc": disease_info['short_desc'],
        "long_desc": disease_info['long_desc'],
        "risk_factors": rule_result["violations"] if rule_result["violations"] else [
            f"✅ Amount ₹{claim_data['amount']:,.2f} appears legitimate"
        ],
        "detection_method": "Rule-Based Only",
        "provider_type": provider_type,
        "benchmark_info": benchmark_info,
        "gst_info": gst_info,
        "price_zone_info": price_zone_info,
        "expected_price_info": expected_price_info
    }

# =============================================================================
# MAIN APP
# =============================================================================

def main():
    # Load resources
    model_artifacts = load_ml_model()
    disease_prices = load_disease_prices()
    icd_lookup = load_icd_codes()
    
    # Title
    st.title("🏥 Healthcare Fraud Detection")
    st.markdown("*AI-powered fraud detection for Indian healthcare claims*")
    
    st.divider()
    
    # Create tabs for different sections
    tab1, tab2 = st.tabs(["📝 Analyze Claim", "💰 Top 10 Most Expensive Diseases"])
    
    # TAB 1: Analyze Claim
    with tab1:
        st.header("📝 Enter Claim Details")
        
        col1, col2 = st.columns(2)
        
        with col1:
            provider_id = st.text_input("Provider ID", value="PRV001", help="Unique identifier for the healthcare provider")
            
            provider_type = st.selectbox(
                "Provider Type",
                options=["Government", "Clinic", "Private"],
                index=1,
                help="Type of healthcare facility"
            )
            
            diagnosis_code = st.text_input(
                "Diagnosis Code (ICD-9/ICD-10)",
                value="4019",
                help="e.g., 4019 for hypertension"
            )
            
            claim_type = st.selectbox(
                "Claim Type",
                options=["Outpatient", "Inpatient"],
                index=0
            )
            
            amount = st.number_input(
                "Claim Amount (₹)",
                min_value=0.0,
                value=5000.0,
                step=100.0,
                help="Total claim amount in Indian Rupees"
            )
        
        with col2:
            deductible = st.number_input(
                "Deductible (₹)",
                min_value=0.0,
                value=0.0,
                step=50.0
            )
            
            num_diagnoses = st.number_input(
                "Number of Diagnoses",
                min_value=1,
                max_value=30,
                value=1
            )
            
            num_procedures = st.number_input(
                "Number of Procedures",
                min_value=0,
                max_value=20,
                value=1
            )
            
            length_of_stay = st.number_input(
                "Length of Stay (days)",
                min_value=0,
                max_value=365,
                value=0,
                help="0 for outpatient visits"
            )
            
            patient_age = st.number_input(
                "Patient Age",
                min_value=0,
                max_value=120,
                value=45
            )
            
            chronic_conditions = st.number_input(
                "Chronic Conditions",
                min_value=0,
                max_value=11,
                value=0,
                help="Number of chronic conditions (0-11)"
            )
        
        st.divider()
        
        # Analyze Button
        if st.button("🔍 Analyze Claim for Fraud", type="primary", use_container_width=True):
            claim_data = {
                'provider_id': provider_id,
                'provider_type': provider_type,
                'diagnosis_code': diagnosis_code,
                'claim_type': claim_type,
                'amount': amount,
                'deductible': deductible,
                'num_diagnoses': num_diagnoses,
                'num_procedures': num_procedures,
                'length_of_stay': length_of_stay,
                'patient_age': patient_age,
                'chronic_conditions': chronic_conditions
            }
            
            with st.spinner("Analyzing claim..."):
                result = predict_fraud(claim_data, model_artifacts, disease_prices, icd_lookup)
            
            st.divider()
            
            # Results Section
            st.header("📊 Analysis Results")
            
            # Main verdict
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if result['is_fraud']:
                    st.error(f"⚠️ **FRAUD DETECTED**")
                else:
                    st.success(f"✅ **LEGITIMATE CLAIM**")
            
            with col2:
                risk_colors = {
                    "Low": "🟢",
                    "Medium": "🟡",
                    "High": "🟠",
                    "Critical": "🔴"
                }
                st.metric(
                    "Risk Level",
                    f"{risk_colors.get(result['risk_level'], '⚪')} {result['risk_level']}"
                )
            
            with col3:
                st.metric(
                    "Fraud Probability",
                    f"{result['probability']*100:.1f}%"
                )
            
            # Disease Info
            st.subheader("🩺 Disease Information")
            st.info(f"**{result['short_desc']}**\n\n{result['long_desc']}")
            
            # Price Zone
            zone_info = result.get('price_zone_info', {})
            if zone_info:
                zone = zone_info.get('zone', 'Unknown')
                zone_colors = {'Normal': 'success', 'Elevated': 'warning', 'Suspicious': 'error'}
                zone_method = getattr(st, zone_colors.get(zone, 'info'))
                zone_method(f"{zone_info.get('emoji', '')} **Price Zone: {zone}** - {zone_info.get('explanation', '')}")
            
            # Expected Price Info
            expected_info = result.get('expected_price_info', {})
            if expected_info:
                st.subheader("💰 Pricing Comparison")
                cols = st.columns(4)
                cols[0].metric("Expected Price", f"₹{expected_info.get('expected_without_gst', 0):,.2f}")
                cols[1].metric("Your Amount", f"₹{amount:,.2f}")
                cols[2].metric("Max Normal", f"₹{expected_info.get('max_normal', 0):,.2f}")
                cols[3].metric("Max Elevated", f"₹{expected_info.get('max_elevated', 0):,.2f}")
            
            # GST Info
            gst_info = result.get('gst_info', {})
            if gst_info:
                st.subheader("🧾 GST Calculation")
                cols = st.columns(3)
                cols[0].metric("Base Amount", f"₹{gst_info.get('base_amount', 0):,.2f}")
                cols[1].metric(f"GST ({gst_info.get('gst_rate', '18%')})", f"₹{gst_info.get('gst_amount', 0):,.2f}")
                cols[2].metric("Total with GST", f"₹{gst_info.get('total_with_gst', 0):,.2f}")
            
            # Risk Factors
            st.subheader("🔍 Risk Factors")
            for factor in result.get('risk_factors', []):
                st.write(f"• {factor}")
            
            # Detection Method
            st.caption(f"Detection Method: {result.get('detection_method', 'Unknown')}")
    
    # TAB 2: Top 10 Most Expensive Diseases
    with tab2:
        st.header("💰 Top 10 Most Expensive Diseases")
        st.markdown("*Based on historical claim data - sorted by average treatment cost*")
        
        top_diseases = load_top_expensive_diseases()
        if top_diseases is not None and not top_diseases.empty:
            # Create display dataframe
            display_df = top_diseases.copy()
            display_df['Rank'] = range(1, len(display_df) + 1)
            display_df['Avg Price'] = display_df['Avg Price (INR)'].apply(lambda x: f"₹{x:,.0f}")
            display_df['Fraud Risk'] = display_df['Fraud Rate'].apply(
                lambda x: f"🔴 {x:.1f}%" if x >= 60 else (f"🟠 {x:.1f}%" if x >= 45 else f"🟢 {x:.1f}%")
            )
            
            # Display as cards
            for idx, row in display_df.iterrows():
                with st.container():
                    col1, col2, col3, col4 = st.columns([1, 4, 2, 2])
                    with col1:
                        st.markdown(f"### {row['Rank']}")
                    with col2:
                        st.markdown(f"**{row['Disease Name']}**")
                        st.caption(f"ICD Code: `{row['Code']}`")
                    with col3:
                        st.metric("Avg Cost", row['Avg Price'])
                    with col4:
                        st.markdown(f"**Fraud Risk**")
                        st.markdown(row['Fraud Risk'])
                    st.divider()
        else:
            st.info("Disease data not available")
    
    # Sidebar
    with st.sidebar:
        st.header("ℹ️ About")
        st.markdown("""
        This application uses a **two-layer approach** to detect healthcare fraud:
        
        1. **Rule-Based Detection**: Instant flags for obvious anomalies
        2. **ML Model**: Pattern recognition for subtle fraud
        
        ---
        
        **Provider Type Benchmarks:**
        - 🏥 **Government**: Subsidized (avg ₹1,000)
        - 🏨 **Clinic**: Standard (avg ₹5,000)
        - 🏢 **Private**: Premium (avg ₹10,000)
        """)
        
        if model_artifacts:
            st.success("✅ ML Model Loaded")
        else:
            st.warning("⚠️ ML Model not available - using rules only")
        
        st.caption(f"Disease prices loaded: {len(disease_prices)}")
        st.caption(f"ICD codes loaded: {len(icd_lookup)}")

if __name__ == "__main__":
    main()

