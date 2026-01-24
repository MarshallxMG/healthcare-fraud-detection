"""
AI Service Module for Healthcare Fraud Detection System
Uses Google Gemini API for intelligent features:
1. Chatbot for Fraud Queries
2. Report Generator
3. Intelligent Insights
"""

import google.generativeai as genai
from typing import Optional, Dict, List
import json
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure Gemini API securely from environment
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    print("⚠️ WARNING: GEMINI_API_KEY not set. AI features will be disabled.")
    model = None
else:
    genai.configure(api_key=GEMINI_API_KEY)
    # Initialize Gemini model
    model = genai.GenerativeModel('gemini-2.0-flash')

# System context for healthcare fraud detection
SYSTEM_CONTEXT = """
You are an AI assistant specialized in Healthcare Fraud Detection for the Indian healthcare system.

Key Knowledge:
- The system uses a Machine Learning model with 94.82% accuracy to detect fraud
- Provider types: Government (cheapest), Clinic (medium), Private/Corporate (expensive)
- ICD-9 and ICD-10 diagnosis codes are used
- GST rate is 18% on healthcare services in India
- Common fraud patterns: upcoding, phantom billing, unbundling, duplicate claims

Fraud Detection Rules:
1. Amount > 3x expected price = Suspicious
2. Age > 120 years = Invalid/Fraud
3. Chronic conditions > 8 at young age = Suspicious
4. Very high claims from Government hospitals = Red flag

Always respond in a helpful, professional manner. Use Indian Rupee (₹) for currency.
"""


async def chat_with_ai(user_message: str, claim_context: Optional[Dict] = None) -> str:
    """
    AI Chatbot for fraud-related queries.
    """
    # Check if model is available
    if model is None:
        return "AI Assistant is not available. Please check the GEMINI_API_KEY in your .env file."
    
    try:
        context = SYSTEM_CONTEXT
        if claim_context:
            context += f"""
            
Current Claim Context:
- Provider: {claim_context.get('provider_id', 'N/A')} ({claim_context.get('provider_type', 'N/A')})
- Diagnosis: {claim_context.get('diagnosis_code', 'N/A')} - {claim_context.get('disease_name', 'N/A')}
- Amount: ₹{claim_context.get('amount', 0):,.2f}
- Risk Level: {claim_context.get('risk_level', 'N/A')}
- Is Fraud: {claim_context.get('is_fraud', False)}
"""
        
        chat = model.start_chat(history=[])
        prompt = f"{context}\n\nUser Question: {user_message}"
        response = chat.send_message(prompt)
        
        return response.text
        
    except Exception as e:
        return f"Sorry, I encountered an error: {str(e)}. Please try again."


async def generate_fraud_report(claim_data: Dict, prediction_result: Dict) -> str:
    """
    Generate a detailed fraud investigation report.
    """
    if model is None:
        return "AI Report generation is not available. Please check the GEMINI_API_KEY."
    
    try:
        prompt = f"""
{SYSTEM_CONTEXT}

Generate a professional FRAUD INVESTIGATION REPORT in markdown format for the following claim:

CLAIM DETAILS:
- Provider ID: {claim_data.get('provider_id', 'N/A')}
- Provider Type: {claim_data.get('provider_type', 'N/A')}
- Diagnosis Code: {claim_data.get('diagnosis_code', 'N/A')}
- Disease: {prediction_result.get('short_desc', 'N/A')}
- Claim Amount: ₹{claim_data.get('amount', 0):,.2f}
- Patient Age: {claim_data.get('patient_age', 'N/A')}
- Claim Type: {claim_data.get('claim_type', 'N/A')}
- Length of Stay: {claim_data.get('length_of_stay', 0)} days
- Number of Diagnoses: {claim_data.get('num_diagnoses', 0)}
- Chronic Conditions: {claim_data.get('chronic_conditions', 0)}

PREDICTION RESULTS:
- Is Fraud: {prediction_result.get('is_fraud', False)}
- Fraud Probability: {prediction_result.get('probability', 0) * 100:.1f}%
- Risk Level: {prediction_result.get('risk_level', 'N/A')}
- Price Zone: {prediction_result.get('price_zone_info', {}).get('zone', 'N/A')}
- Expected Price: ₹{prediction_result.get('expected_price_info', {}).get('expected_without_gst', 0):,.2f}
- Detection Method: {prediction_result.get('detection_method', 'N/A')}
- Risk Factors: {prediction_result.get('risk_factors', [])}

Generate a detailed report with:
1. Executive Summary (2-3 sentences)
2. Claim Analysis (key observations)
3. Risk Assessment (why this was flagged or cleared)
4. Price Comparison (expected vs actual)
5. Recommendation (investigate further, approve, or deny)
6. Next Steps (specific actions to take)

Use proper markdown formatting with headers, bullet points, and emphasis.
"""
        
        response = model.generate_content(prompt)
        return response.text
        
    except Exception as e:
        return f"Error generating report: {str(e)}"


async def get_intelligent_insights(stats: Dict, recent_claims: List[Dict]) -> Dict:
    """
    Generate intelligent insights from fraud data.
    """
    if model is None:
        return {
            "key_patterns": ["AI not available"],
            "anomalies": [],
            "recommendations": ["Check GEMINI_API_KEY in .env file"],
            "risk_summary": "AI insights unavailable. Please configure the Gemini API key.",
            "priority_actions": []
        }
    
    try:
        fraud_claims = [c for c in recent_claims if c.get('is_fraud', False)]
        fraud_count = len(fraud_claims)
        total_count = len(recent_claims)
        
        prompt = f"""
{SYSTEM_CONTEXT}

Analyze the following healthcare fraud data and provide intelligent insights:

OVERALL STATISTICS:
- Total Claims: {stats.get('total_claims', 0):,}
- Fraud Claims: {stats.get('fraud_claims', 0):,}
- Fraud Rate: {stats.get('fraud_percentage', 0):.1f}%
- Inpatient Claims: {stats.get('inpatient_claims', 0):,}
- Outpatient Claims: {stats.get('outpatient_claims', 0):,}

RECENT CLAIMS SAMPLE ({total_count} claims, {fraud_count} flagged):
{json.dumps(recent_claims[:10], indent=2) if recent_claims else 'No data'}

Provide a JSON response with exactly this structure:
{{
    "key_patterns": [
        "Pattern 1 description",
        "Pattern 2 description",
        "Pattern 3 description"
    ],
    "anomalies": [
        "Anomaly 1 description",
        "Anomaly 2 description"
    ],
    "recommendations": [
        "Recommendation 1",
        "Recommendation 2",
        "Recommendation 3"
    ],
    "risk_summary": "One paragraph summary of overall fraud risk",
    "priority_actions": [
        "Action 1 to take immediately",
        "Action 2 to take this week"
    ]
}}

Return ONLY valid JSON, no markdown formatting.
"""
        
        response = model.generate_content(prompt)
        
        try:
            text = response.text.strip()
            if text.startswith("```json"):
                text = text[7:]
            if text.startswith("```"):
                text = text[3:]
            if text.endswith("```"):
                text = text[:-3]
            
            insights = json.loads(text.strip())
            return insights
        except json.JSONDecodeError:
            return {
                "key_patterns": ["Unable to parse AI response"],
                "anomalies": ["Please try again"],
                "recommendations": ["Refresh the insights"],
                "risk_summary": response.text[:500] if response.text else "No insights available",
                "priority_actions": ["Review system configuration"]
            }
        
    except Exception as e:
        return {
            "key_patterns": [],
            "anomalies": [],
            "recommendations": [],
            "risk_summary": f"Error generating insights: {str(e)}",
            "priority_actions": []
        }


def test_ai_service():
    """Test the AI service connection"""
    if model is None:
        return {"status": "error", "message": "GEMINI_API_KEY not configured in .env file"}
    
    try:
        response = model.generate_content("Say 'Healthcare AI Ready!' in one line")
        return {"status": "ok", "message": response.text.strip()}
    except Exception as e:
        return {"status": "error", "message": str(e)}
