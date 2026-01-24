"""
Comprehensive Test Script for Healthcare Fraud Detection API
=============================================================
Tests all aspects of the /predict endpoint with various inputs.
"""

import requests
import random
import json

API_URL = "http://localhost:8000"

# Test cases to run
test_cases = [
    # Test 1: Basic ICD-9 code (common)
    {
        "name": "ICD-9 Hypertension (4019)",
        "data": {
            "provider_id": "PRV55001",
            "provider_type": "Clinic",
            "diagnosis_code": "4019",
            "claim_type": "Outpatient",
            "amount": 500.0,
            "deductible": 50.0,
            "num_diagnoses": 2,
            "num_procedures": 1,
            "length_of_stay": 0,
            "patient_age": 65,
            "chronic_conditions": 2
        },
        "expected": {"disease_name_contains": "Hypertension"}
    },
    
    # Test 2: High-cost ICD-9 code
    {
        "name": "ICD-9 Acute Respiratory Failure (51881)",
        "data": {
            "provider_id": "PRV55002",
            "provider_type": "Private",
            "diagnosis_code": "51881",
            "claim_type": "Inpatient",
            "amount": 17000.0,
            "deductible": 100.0,
            "num_diagnoses": 5,
            "num_procedures": 3,
            "length_of_stay": 5,
            "patient_age": 70,
            "chronic_conditions": 4
        },
        "expected": {"disease_name_contains": "Respiratory"}
    },
    
    # Test 3: V-code (should work now)
    {
        "name": "ICD-9 V-Code Heart Assist (V4321)",
        "data": {
            "provider_id": "PRV55003",
            "provider_type": "Private",
            "diagnosis_code": "V4321",
            "claim_type": "Inpatient", 
            "amount": 25000.0,
            "deductible": 200.0,
            "num_diagnoses": 3,
            "num_procedures": 2,
            "length_of_stay": 7,
            "patient_age": 55,
            "chronic_conditions": 3
        },
        "expected": {"disease_name_contains": "Heart"}
    },
    
    # Test 4: ICD-10 code with dot
    {
        "name": "ICD-10 COPD (J44.1)",
        "data": {
            "provider_id": "PRV55004",
            "provider_type": "Government",
            "diagnosis_code": "J44.1",
            "claim_type": "Outpatient",
            "amount": 800.0,
            "deductible": 50.0,
            "num_diagnoses": 2,
            "num_procedures": 1,
            "length_of_stay": 0,
            "patient_age": 68,
            "chronic_conditions": 2
        },
        "expected": {"disease_name_contains": "pulmonary"}
    },
    
    # Test 5: ICD-10 code without dot
    {
        "name": "ICD-10 Fracture (S72001A)",
        "data": {
            "provider_id": "PRV55005",
            "provider_type": "Private",
            "diagnosis_code": "S72001A",
            "claim_type": "Inpatient",
            "amount": 15000.0,
            "deductible": 100.0,
            "num_diagnoses": 2,
            "num_procedures": 2,
            "length_of_stay": 4,
            "patient_age": 75,
            "chronic_conditions": 1
        },
        "expected": {"disease_name_contains": "femur"}
    },
    
    # Test 6: Government hospital (low amount)
    {
        "name": "Government Hospital Low Amount",
        "data": {
            "provider_id": "PRV55006",
            "provider_type": "Government",
            "diagnosis_code": "486",
            "claim_type": "Inpatient",
            "amount": 500.0,
            "deductible": 25.0,
            "num_diagnoses": 1,
            "num_procedures": 1,
            "length_of_stay": 2,
            "patient_age": 45,
            "chronic_conditions": 0
        },
        "expected": {"risk_level": "Low"}
    },
    
    # Test 7: Suspicious high amount
    {
        "name": "Suspicious High Amount Claim",
        "data": {
            "provider_id": "PRV55007",
            "provider_type": "Clinic",
            "diagnosis_code": "4019",
            "claim_type": "Outpatient",
            "amount": 50000.0,
            "deductible": 100.0,
            "num_diagnoses": 10,
            "num_procedures": 5,
            "length_of_stay": 0,
            "patient_age": 65,
            "chronic_conditions": 3
        },
        "expected": {"should_flag": True}
    },
    
    # Test 8: Edge case - very young patient
    {
        "name": "Young Patient (Age 25)",
        "data": {
            "provider_id": "PRV55008",
            "provider_type": "Clinic",
            "diagnosis_code": "7840",
            "claim_type": "Outpatient",
            "amount": 200.0,
            "deductible": 20.0,
            "num_diagnoses": 1,
            "num_procedures": 1,
            "length_of_stay": 0,
            "patient_age": 25,
            "chronic_conditions": 0
        },
        "expected": {}
    },
    
    # Test 9: Edge case - many chronic conditions
    {
        "name": "Many Chronic Conditions (11)",
        "data": {
            "provider_id": "PRV55009",
            "provider_type": "Private",
            "diagnosis_code": "25000",
            "claim_type": "Outpatient",
            "amount": 3000.0,
            "deductible": 100.0,
            "num_diagnoses": 5,
            "num_procedures": 2,
            "length_of_stay": 0,
            "patient_age": 80,
            "chronic_conditions": 11
        },
        "expected": {}
    },
    
    # Test 10: Unknown diagnosis code
    {
        "name": "Unknown Diagnosis Code (XXXXX)",
        "data": {
            "provider_id": "PRV55010",
            "provider_type": "Clinic",
            "diagnosis_code": "XXXXX",
            "claim_type": "Outpatient",
            "amount": 500.0,
            "deductible": 50.0,
            "num_diagnoses": 1,
            "num_procedures": 1,
            "length_of_stay": 0,
            "patient_age": 50,
            "chronic_conditions": 1
        },
        "expected": {"should_handle_gracefully": True}
    },
]

def run_tests():
    print("=" * 70)
    print("HEALTHCARE FRAUD DETECTION - COMPREHENSIVE API TEST")
    print("=" * 70)
    print()
    
    results = {
        "passed": 0,
        "failed": 0,
        "errors": []
    }
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n--- Test {i}: {test['name']} ---")
        
        try:
            response = requests.post(f"{API_URL}/predict", json=test["data"], timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                # Extract key fields
                disease_short = data.get("short_desc", "N/A")
                disease_long = data.get("long_desc", "N/A")
                risk_level = data.get("risk_level", "N/A")
                probability = data.get("probability", 0) * 100
                is_fraud = data.get("is_fraud", False)
                price_zone = data.get("price_zone_info", {})
                expected_price = data.get("expected_price_info", {})
                gst_info = data.get("gst_info", {})
                
                # Print results
                print(f"  ✓ Status: 200 OK")
                print(f"  Disease: {disease_short}")
                print(f"  Long: {disease_long[:50]}..." if len(str(disease_long)) > 50 else f"  Long: {disease_long}")
                print(f"  Risk Level: {risk_level} ({probability:.1f}%)")
                print(f"  Is Fraud: {is_fraud}")
                
                if price_zone:
                    print(f"  Price Zone: {price_zone.get('zone', 'N/A')} ({price_zone.get('ratio', 0):.2f}x expected)")
                
                if expected_price:
                    print(f"  Expected Price: ₹{expected_price.get('expected_without_gst', 0):,.2f}")
                
                if gst_info:
                    print(f"  GST: ₹{gst_info.get('gst_amount', 0):,.2f} (Total: ₹{gst_info.get('total_with_gst', 0):,.2f})")
                
                # Validate expected conditions
                errors_in_test = []
                
                # Check disease name
                if "disease_name_contains" in test["expected"]:
                    expected_text = test["expected"]["disease_name_contains"].lower()
                    if expected_text not in disease_short.lower() and expected_text not in disease_long.lower():
                        errors_in_test.append(f"Disease name should contain '{expected_text}' but got '{disease_short}'")
                
                # Check if disease name is just a number or very short
                if disease_short in ["9", "1", "0"] or len(disease_short) < 3:
                    errors_in_test.append(f"Disease name looks invalid: '{disease_short}'")
                
                # Check for missing fields
                if not disease_short or disease_short == "N/A":
                    errors_in_test.append("Missing disease short description")
                
                if errors_in_test:
                    results["failed"] += 1
                    for err in errors_in_test:
                        print(f"  ❌ ERROR: {err}")
                        results["errors"].append(f"Test {i} ({test['name']}): {err}")
                else:
                    results["passed"] += 1
                    print(f"  ✓ PASSED")
                    
            else:
                results["failed"] += 1
                error_msg = f"HTTP {response.status_code}: {response.text[:100]}"
                print(f"  ❌ FAILED: {error_msg}")
                results["errors"].append(f"Test {i} ({test['name']}): {error_msg}")
                
        except requests.exceptions.ConnectionError:
            results["failed"] += 1
            print(f"  ❌ ERROR: Cannot connect to API at {API_URL}")
            results["errors"].append(f"Test {i}: Connection error")
            
        except Exception as e:
            results["failed"] += 1
            print(f"  ❌ ERROR: {str(e)}")
            results["errors"].append(f"Test {i} ({test['name']}): {str(e)}")
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"\n  Total Tests: {len(test_cases)}")
    print(f"  ✓ Passed: {results['passed']}")
    print(f"  ❌ Failed: {results['failed']}")
    
    if results["errors"]:
        print(f"\n  ERRORS FOUND:")
        for err in results["errors"]:
            print(f"    - {err}")
    else:
        print(f"\n  🎉 ALL TESTS PASSED!")
    
    print("=" * 70)
    
    return results

if __name__ == "__main__":
    run_tests()
