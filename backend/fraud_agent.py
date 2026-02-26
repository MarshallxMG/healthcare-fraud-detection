"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    AGENTIC AI - FRAUD INVESTIGATION AGENT                   ║
║                                                                              ║
║  An autonomous AI agent that can investigate healthcare fraud using tools.    ║
║  Built with Google Gemini's native function calling.                         ║
║                                                                              ║
║  The agent has 7 tools:                                                      ║
║    1. query_claims_database  - Search claims with filters                    ║
║    2. run_fraud_prediction   - Analyze a claim through ML + rules            ║
║    3. lookup_disease_price   - Get expected pricing for a disease             ║
║    4. get_provider_history   - Get all claims for a provider                 ║
║    5. get_fraud_statistics   - Get overall dataset statistics                ║
║    6. search_hospital_info   - Look up hospital details                      ║
║    7. generate_investigation_report - Produce a detailed fraud report        ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import google.generativeai as genai
from google.generativeai.types import content_types
from typing import Optional, Dict, List, Any
import json
import os
import uuid
from collections import OrderedDict
from dotenv import load_dotenv

load_dotenv()

# ============================================================================
# GEMINI CONFIGURATION
# ============================================================================

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
agent_model = None

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# ============================================================================
# AGENT SYSTEM PROMPT
# ============================================================================

AGENT_SYSTEM_PROMPT = """You are an expert Healthcare Fraud Investigation Agent for the Indian healthcare system.

You have access to powerful tools that let you autonomously investigate fraud. You should USE THESE TOOLS proactively to answer user questions with REAL DATA rather than giving generic responses.

YOUR CAPABILITIES:
- Query the claims database to find specific claims, providers, or patterns
- Run the ML fraud prediction model on any claim
- Look up expected disease pricing across provider types
- Investigate specific providers by pulling their full claim history
- Get overall fraud statistics from the database
- Search for hospital information
- Generate detailed investigation reports

GUIDELINES:
1. ALWAYS use tools when the user asks about data, claims, providers, statistics, or prices.
2. When investigating a provider, start by getting their full history, then analyze patterns.
3. For suspicious claims, run the fraud prediction model and explain the results.
4. Use Indian Rupee (₹) for all currency amounts.
5. Be thorough - if a question requires multiple tools, use them all.
6. Present findings clearly with bullet points and data.
7. If you find suspicious patterns, explain WHY they are suspicious.
8. Provider types are: Government (cheapest), Clinic (medium), Private (most expensive).
9. The ML model has 94.82% accuracy using Gradient Boosting.
10. GST rate is 18% on healthcare services in India.

FRAUD PATTERNS TO WATCH FOR:
- Amounts significantly above expected price (>2.5x = suspicious)
- Young patients with many chronic conditions
- Excessive diagnoses on a single claim (upcoding)
- Government hospitals with very high claims
- Providers with unusually high fraud rates

When you are done investigating, provide a clear summary of your findings."""

# ============================================================================
# TOOL DECLARATIONS (Gemini Function Calling format)
# ============================================================================

TOOL_DECLARATIONS = [
    {
        "name": "query_claims_database",
        "description": "Search the healthcare claims database with filters. Returns matching claims with details. Use this to find specific claims, look for patterns, or get examples of fraud/legitimate claims.",
        "parameters": {
            "type": "object",
            "properties": {
                "provider_id": {
                    "type": "string",
                    "description": "Filter by provider ID (e.g., 'PRV51234')"
                },
                "is_fraud": {
                    "type": "boolean",
                    "description": "Filter by fraud status. True for fraud only, False for legitimate only."
                },
                "min_amount": {
                    "type": "number",
                    "description": "Minimum claim amount in ₹"
                },
                "max_amount": {
                    "type": "number",
                    "description": "Maximum claim amount in ₹"
                },
                "claim_type": {
                    "type": "string",
                    "description": "Filter by claim type: 'Inpatient' or 'Outpatient'"
                },
                "diagnosis_code": {
                    "type": "string",
                    "description": "Filter by diagnosis code (ICD-9 or ICD-10)"
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of results to return (default: 10, max: 50)"
                },
                "sort_by": {
                    "type": "string",
                    "description": "Sort results by: 'amount_desc', 'amount_asc', or 'recent'. Default: 'recent'"
                }
            },
            "required": []
        }
    },
    {
        "name": "run_fraud_prediction",
        "description": "Run the 2-layer fraud detection system (Rule-Based + ML Model) on a claim. Returns fraud probability, risk level, risk factors, price zone analysis, and GST info.",
        "parameters": {
            "type": "object",
            "properties": {
                "provider_id": {
                    "type": "string",
                    "description": "Provider ID (e.g., 'PRV001')"
                },
                "provider_type": {
                    "type": "string",
                    "description": "Provider type: 'Government', 'Clinic', or 'Private'"
                },
                "diagnosis_code": {
                    "type": "string",
                    "description": "ICD-9 or ICD-10 diagnosis code"
                },
                "claim_type": {
                    "type": "string",
                    "description": "'Inpatient' or 'Outpatient'"
                },
                "amount": {
                    "type": "number",
                    "description": "Claim amount in Indian Rupees (₹)"
                },
                "patient_age": {
                    "type": "integer",
                    "description": "Patient age in years"
                },
                "num_diagnoses": {
                    "type": "integer",
                    "description": "Number of diagnoses on the claim"
                },
                "chronic_conditions": {
                    "type": "integer",
                    "description": "Number of chronic conditions"
                },
                "length_of_stay": {
                    "type": "integer",
                    "description": "Days spent in hospital (0 for outpatient)"
                }
            },
            "required": ["provider_id", "provider_type", "diagnosis_code", "amount"]
        }
    },
    {
        "name": "lookup_disease_price",
        "description": "Get the expected price for a disease/diagnosis at a given provider type. Shows base price, expected price with provider multiplier, and price zones (Normal/Elevated/Suspicious thresholds).",
        "parameters": {
            "type": "object",
            "properties": {
                "diagnosis_code": {
                    "type": "string",
                    "description": "ICD-9 or ICD-10 diagnosis code (e.g., '4019' for hypertension)"
                },
                "provider_type": {
                    "type": "string",
                    "description": "Provider type: 'Government', 'Clinic', or 'Private'. Default: 'Clinic'"
                }
            },
            "required": ["diagnosis_code"]
        }
    },
    {
        "name": "get_provider_history",
        "description": "Get the complete claim history and statistics for a specific provider. Returns all their claims, fraud rate, total amount, average claim, and pattern analysis.",
        "parameters": {
            "type": "object",
            "properties": {
                "provider_id": {
                    "type": "string",
                    "description": "The provider ID to investigate (e.g., 'PRV51234')"
                },
                "limit": {
                    "type": "integer",
                    "description": "Max number of claims to return (default: 20)"
                }
            },
            "required": ["provider_id"]
        }
    },
    {
        "name": "get_fraud_statistics",
        "description": "Get overall fraud statistics from the entire claims database. Returns total claims, fraud count, fraud percentage, inpatient/outpatient breakdown.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    {
        "name": "search_hospital_info",
        "description": "Search for hospital information by name. Returns hospital details, type, and location.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Hospital name to search for (partial name works)"
                },
                "hospital_type": {
                    "type": "string",
                    "description": "Filter by type: 'Government', 'Clinic', or 'Private'"
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "generate_investigation_report",
        "description": "Generate a detailed markdown fraud investigation report for a specific claim. Includes executive summary, risk assessment, price comparison, and recommendations.",
        "parameters": {
            "type": "object",
            "properties": {
                "provider_id": {
                    "type": "string",
                    "description": "Provider ID"
                },
                "provider_type": {
                    "type": "string",
                    "description": "Provider type: 'Government', 'Clinic', or 'Private'"
                },
                "diagnosis_code": {
                    "type": "string",
                    "description": "Diagnosis code"
                },
                "amount": {
                    "type": "number",
                    "description": "Claim amount in ₹"
                },
                "patient_age": {
                    "type": "integer",
                    "description": "Patient age"
                },
                "claim_type": {
                    "type": "string",
                    "description": "'Inpatient' or 'Outpatient'"
                }
            },
            "required": ["provider_id", "provider_type", "diagnosis_code", "amount"]
        }
    }
]


# ============================================================================
# TOOL IMPLEMENTATIONS
# ============================================================================
# These functions execute the actual logic. They are called by the agent loop
# when Gemini decides to use a tool.

def _execute_query_claims(params: dict, db_session_factory, Claim_model, get_disease_info_fn) -> dict:
    """Query the claims database with filters."""
    try:
        db = db_session_factory()
        query = db.query(Claim_model)

        # Apply filters
        if params.get("provider_id"):
            query = query.filter(Claim_model.provider_id == params["provider_id"])
        if params.get("is_fraud") is not None:
            query = query.filter(Claim_model.is_fraud == params["is_fraud"])
        if params.get("min_amount"):
            query = query.filter(Claim_model.amount >= params["min_amount"])
        if params.get("max_amount"):
            query = query.filter(Claim_model.amount <= params["max_amount"])
        if params.get("claim_type"):
            query = query.filter(Claim_model.claim_type == params["claim_type"])
        if params.get("diagnosis_code"):
            query = query.filter(Claim_model.diagnosis_code == params["diagnosis_code"])

        # Sort
        sort_by = params.get("sort_by", "recent")
        if sort_by == "amount_desc":
            query = query.order_by(Claim_model.amount.desc())
        elif sort_by == "amount_asc":
            query = query.order_by(Claim_model.amount.asc())
        else:
            query = query.order_by(Claim_model.id.desc())

        # Limit
        limit = min(params.get("limit", 10), 50)
        claims = query.limit(limit).all()

        results = []
        for c in claims:
            info = get_disease_info_fn(c.diagnosis_code)
            results.append({
                "id": c.id,
                "provider_id": c.provider_id,
                "patient_id": c.patient_id,
                "diagnosis_code": c.diagnosis_code,
                "disease_name": info["short_desc"],
                "claim_type": c.claim_type,
                "amount": c.amount,
                "num_diagnoses": c.num_diagnoses,
                "patient_age": c.patient_age,
                "length_of_stay": c.length_of_stay,
                "is_fraud": c.is_fraud,
            })

        db.close()

        total_query = db_session_factory()
        total_matching = total_query.query(Claim_model)
        if params.get("provider_id"):
            total_matching = total_matching.filter(Claim_model.provider_id == params["provider_id"])
        if params.get("is_fraud") is not None:
            total_matching = total_matching.filter(Claim_model.is_fraud == params["is_fraud"])
        count = total_matching.count()
        total_query.close()

        return {
            "total_matching": count,
            "returned": len(results),
            "claims": results
        }
    except Exception as e:
        return {"error": str(e)}


def _execute_fraud_prediction(params: dict, predict_fn) -> dict:
    """Run the 2-layer fraud detection on a claim."""
    try:
        claim_data = {
            "provider_id": params.get("provider_id", "AGENT_TEST"),
            "provider_type": params.get("provider_type", "Clinic"),
            "diagnosis_code": params.get("diagnosis_code", "4019"),
            "claim_type": params.get("claim_type", "Outpatient"),
            "amount": params.get("amount", 0),
            "deductible": params.get("deductible", 0),
            "num_diagnoses": params.get("num_diagnoses", 1),
            "num_procedures": params.get("num_procedures", 1),
            "length_of_stay": params.get("length_of_stay", 0),
            "patient_age": params.get("patient_age", 65),
            "chronic_conditions": params.get("chronic_conditions", 0),
        }
        result = predict_fn(claim_data)
        return result
    except Exception as e:
        return {"error": str(e)}


def _execute_lookup_price(params: dict, get_expected_price_fn, classify_price_zone_fn) -> dict:
    """Look up expected price for a disease."""
    try:
        code = params.get("diagnosis_code", "4019")
        provider_type = params.get("provider_type", "Clinic")

        expected_info = get_expected_price_fn(code, provider_type)

        # Also show zones for some example amounts
        example_amounts = [
            expected_info.get("expected_without_gst", 0) * 0.5,
            expected_info.get("expected_without_gst", 0),
            expected_info.get("expected_without_gst", 0) * 2,
            expected_info.get("expected_without_gst", 0) * 3,
        ]

        zones = []
        for amt in example_amounts:
            if amt > 0:
                zone = classify_price_zone_fn(amt, expected_info)
                zones.append({"amount": round(amt, 2), "zone": zone["zone"], "explanation": zone["explanation"]})

        return {
            "diagnosis_code": code,
            "provider_type": provider_type,
            "pricing": expected_info,
            "example_zones": zones
        }
    except Exception as e:
        return {"error": str(e)}


def _execute_provider_history(params: dict, db_session_factory, Claim_model, get_disease_info_fn) -> dict:
    """Get provider claim history and stats."""
    try:
        provider_id = params.get("provider_id", "")
        limit = min(params.get("limit", 20), 50)

        db = db_session_factory()
        claims = db.query(Claim_model).filter(
            Claim_model.provider_id == provider_id
        ).order_by(Claim_model.amount.desc()).limit(limit).all()

        total_claims = db.query(Claim_model).filter(
            Claim_model.provider_id == provider_id
        ).count()

        fraud_claims = db.query(Claim_model).filter(
            Claim_model.provider_id == provider_id,
            Claim_model.is_fraud == True
        ).count()

        from sqlalchemy import func
        amount_stats = db.query(
            func.sum(Claim_model.amount),
            func.avg(Claim_model.amount),
            func.min(Claim_model.amount),
            func.max(Claim_model.amount),
        ).filter(Claim_model.provider_id == provider_id).first()

        claim_list = []
        for c in claims:
            info = get_disease_info_fn(c.diagnosis_code)
            claim_list.append({
                "id": c.id,
                "patient_id": c.patient_id,
                "diagnosis_code": c.diagnosis_code,
                "disease_name": info["short_desc"],
                "claim_type": c.claim_type,
                "amount": c.amount,
                "patient_age": c.patient_age,
                "num_diagnoses": c.num_diagnoses,
                "is_fraud": c.is_fraud,
            })

        db.close()

        return {
            "provider_id": provider_id,
            "total_claims": total_claims,
            "fraud_claims": fraud_claims,
            "fraud_rate": round((fraud_claims / total_claims * 100) if total_claims > 0 else 0, 2),
            "total_amount": round(amount_stats[0] or 0, 2),
            "average_claim": round(amount_stats[1] or 0, 2),
            "min_claim": round(amount_stats[2] or 0, 2),
            "max_claim": round(amount_stats[3] or 0, 2),
            "claims_shown": len(claim_list),
            "claims": claim_list
        }
    except Exception as e:
        return {"error": str(e)}


def _execute_fraud_statistics(db_session_factory, Claim_model) -> dict:
    """Get overall fraud statistics."""
    try:
        db = db_session_factory()

        total = db.query(Claim_model).count()
        fraud = db.query(Claim_model).filter(Claim_model.is_fraud == True).count()
        inpatient = db.query(Claim_model).filter(Claim_model.claim_type == "Inpatient").count()
        outpatient = db.query(Claim_model).filter(Claim_model.claim_type == "Outpatient").count()

        from sqlalchemy import func
        amount_stats = db.query(
            func.sum(Claim_model.amount),
            func.avg(Claim_model.amount),
        ).first()

        fraud_amount = db.query(
            func.sum(Claim_model.amount),
        ).filter(Claim_model.is_fraud == True).first()

        db.close()

        return {
            "total_claims": total,
            "fraud_claims": fraud,
            "legitimate_claims": total - fraud,
            "fraud_percentage": round((fraud / total * 100) if total > 0 else 0, 2),
            "inpatient_claims": inpatient,
            "outpatient_claims": outpatient,
            "total_amount": round(amount_stats[0] or 0, 2),
            "average_claim_amount": round(amount_stats[1] or 0, 2),
            "total_fraud_amount": round(fraud_amount[0] or 0, 2),
        }
    except Exception as e:
        return {"error": str(e)}


def _execute_hospital_search(params: dict, search_fn) -> dict:
    """Search for hospital info."""
    try:
        query = params.get("query", "")
        hospital_type = params.get("hospital_type", None)
        results = search_fn(query, hospital_type)
        return results
    except Exception as e:
        return {"error": str(e)}


def _execute_investigation_report(params: dict, predict_fn, report_fn) -> dict:
    """Generate a full investigation report."""
    try:
        # First run prediction
        claim_data = {
            "provider_id": params.get("provider_id", "UNKNOWN"),
            "provider_type": params.get("provider_type", "Clinic"),
            "diagnosis_code": params.get("diagnosis_code", "4019"),
            "claim_type": params.get("claim_type", "Outpatient"),
            "amount": params.get("amount", 0),
            "deductible": 0,
            "num_diagnoses": params.get("num_diagnoses", 1),
            "num_procedures": 1,
            "length_of_stay": params.get("length_of_stay", 0),
            "patient_age": params.get("patient_age", 65),
            "chronic_conditions": params.get("chronic_conditions", 0),
        }

        prediction = predict_fn(claim_data)
        report = report_fn(claim_data, prediction)

        return {
            "prediction_summary": {
                "is_fraud": prediction.get("is_fraud"),
                "probability": prediction.get("probability"),
                "risk_level": prediction.get("risk_level"),
                "risk_factors": prediction.get("risk_factors", []),
            },
            "report": report
        }
    except Exception as e:
        return {"error": str(e)}


# ============================================================================
# SESSION MEMORY (stores conversation history per session)
# ============================================================================

class SessionStore:
    """Simple in-memory session store with LRU eviction."""

    def __init__(self, max_sessions: int = 100):
        self._sessions: OrderedDict[str, List[dict]] = OrderedDict()
        self._max = max_sessions

    def get(self, session_id: str) -> List[dict]:
        if session_id in self._sessions:
            self._sessions.move_to_end(session_id)
            return self._sessions[session_id]
        return []

    def save(self, session_id: str, history: List[dict]):
        self._sessions[session_id] = history
        self._sessions.move_to_end(session_id)
        while len(self._sessions) > self._max:
            self._sessions.popitem(last=False)

    def create_session(self) -> str:
        return str(uuid.uuid4())


session_store = SessionStore()


# ============================================================================
# AGENT LOOP
# ============================================================================

async def run_agent(
    user_message: str,
    session_id: Optional[str],
    # Dependency injection — these are passed from main.py
    db_session_factory,
    Claim_model,
    get_disease_info_fn,
    get_expected_price_fn,
    classify_price_zone_fn,
    predict_fn,
    search_hospital_fn,
    report_fn,
) -> dict:
    """
    Main agent loop using Gemini function calling.

    1. Sends user message to Gemini with tool declarations
    2. If Gemini requests a tool call, executes it and sends results back
    3. Loops until Gemini produces a text response (max 10 iterations)
    4. Returns the response and list of tools used
    """

    if not GEMINI_API_KEY:
        return {
            "response": "⚠️ AI Agent is not available. Please set `GEMINI_API_KEY` in your `.env` file.",
            "tools_used": [],
            "session_id": session_id or "none"
        }

    # Create or restore session
    if not session_id:
        session_id = session_store.create_session()

    history = session_store.get(session_id)

    # Build Gemini tools from declarations
    gemini_tools = genai.protos.Tool(
        function_declarations=[
            genai.protos.FunctionDeclaration(
                name=t["name"],
                description=t["description"],
                parameters=genai.protos.Schema(
                    type=genai.protos.Type.OBJECT,
                    properties={
                        k: genai.protos.Schema(
                            type=_map_type(v.get("type", "string")),
                            description=v.get("description", ""),
                        )
                        for k, v in t["parameters"].get("properties", {}).items()
                    },
                    required=t["parameters"].get("required", []),
                )
            )
            for t in TOOL_DECLARATIONS
        ]
    )

    # Initialize the model with tools
    model = genai.GenerativeModel(
        "gemini-2.5-flash",
        tools=[gemini_tools],
        system_instruction=AGENT_SYSTEM_PROMPT,
    )

    # Start or continue chat
    chat = model.start_chat(history=history)

    tools_used = []
    max_iterations = 10

    # Send user message
    response = chat.send_message(user_message)

    for _ in range(max_iterations):
        # Check if Gemini wants to call a function
        candidate = response.candidates[0]
        part = candidate.content.parts[0]

        if hasattr(part, "function_call") and part.function_call.name:
            fn_call = part.function_call
            fn_name = fn_call.name
            fn_args = dict(fn_call.args) if fn_call.args else {}

            tools_used.append({"tool": fn_name, "args": _sanitize_args(fn_args)})

            # Execute the tool
            tool_result = _dispatch_tool(
                fn_name, fn_args,
                db_session_factory=db_session_factory,
                Claim_model=Claim_model,
                get_disease_info_fn=get_disease_info_fn,
                get_expected_price_fn=get_expected_price_fn,
                classify_price_zone_fn=classify_price_zone_fn,
                predict_fn=predict_fn,
                search_hospital_fn=search_hospital_fn,
                report_fn=report_fn,
            )

            # Send tool result back to Gemini
            response = chat.send_message(
                genai.protos.Content(
                    parts=[
                        genai.protos.Part(
                            function_response=genai.protos.FunctionResponse(
                                name=fn_name,
                                response={"result": tool_result},
                            )
                        )
                    ]
                )
            )
        else:
            # Gemini produced a text response — we're done
            break

    # Extract final text response
    final_text = ""
    for part in response.candidates[0].content.parts:
        if hasattr(part, "text") and part.text:
            final_text += part.text

    if not final_text:
        final_text = "I completed the investigation but couldn't generate a summary. Please try rephrasing your question."

    # Save conversation history (keep last 20 turns to avoid context overflow)
    session_store.save(session_id, list(chat.history[-20:]))

    return {
        "response": final_text,
        "tools_used": tools_used,
        "session_id": session_id,
    }


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _map_type(type_str: str) -> int:
    """Map JSON schema type to Gemini proto type."""
    mapping = {
        "string": genai.protos.Type.STRING,
        "number": genai.protos.Type.NUMBER,
        "integer": genai.protos.Type.NUMBER,
        "boolean": genai.protos.Type.BOOLEAN,
        "array": genai.protos.Type.ARRAY,
        "object": genai.protos.Type.OBJECT,
    }
    return mapping.get(type_str, genai.protos.Type.STRING)


def _sanitize_args(args: dict) -> dict:
    """Make args JSON-serializable for the response."""
    sanitized = {}
    for k, v in args.items():
        try:
            json.dumps(v)
            sanitized[k] = v
        except (TypeError, ValueError):
            sanitized[k] = str(v)
    return sanitized


def _dispatch_tool(
    fn_name: str,
    fn_args: dict,
    db_session_factory,
    Claim_model,
    get_disease_info_fn,
    get_expected_price_fn,
    classify_price_zone_fn,
    predict_fn,
    search_hospital_fn,
    report_fn,
) -> dict:
    """Route a tool call to the correct implementation."""

    if fn_name == "query_claims_database":
        return _execute_query_claims(fn_args, db_session_factory, Claim_model, get_disease_info_fn)

    elif fn_name == "run_fraud_prediction":
        return _execute_fraud_prediction(fn_args, predict_fn)

    elif fn_name == "lookup_disease_price":
        return _execute_lookup_price(fn_args, get_expected_price_fn, classify_price_zone_fn)

    elif fn_name == "get_provider_history":
        return _execute_provider_history(fn_args, db_session_factory, Claim_model, get_disease_info_fn)

    elif fn_name == "get_fraud_statistics":
        return _execute_fraud_statistics(db_session_factory, Claim_model)

    elif fn_name == "search_hospital_info":
        return _execute_hospital_search(fn_args, search_hospital_fn)

    elif fn_name == "generate_investigation_report":
        return _execute_investigation_report(fn_args, predict_fn, report_fn)

    else:
        return {"error": f"Unknown tool: {fn_name}"}


# ============================================================================
# HELPER: Wrap predict_fraud for agent use (without FastAPI Request object)
# ============================================================================

def create_predict_fn(ClaimInput_class, predict_fraud_fn):
    """
    Creates a wrapper that calls predict_fraud without needing a Request object.
    The agent passes a dict, we convert it to ClaimInput and call predict.
    """
    def _predict(claim_data: dict) -> dict:
        claim = ClaimInput_class(**claim_data)
        # Create a mock request with a mock client
        class MockClient:
            host = "127.0.0.1"
        class MockRequest:
            client = MockClient()
            # Attributes needed by rate limiter
            scope = {"type": "http", "path": "/agent/predict"}
            url = type("URL", (), {"path": "/agent/predict"})()
            headers = {}
            query_params = {}
            state = type("State", (), {"_state": {}})()
            method = "POST"
            def __init__(self):
                pass
        try:
            # Call predict_fraud — it may be rate-limited, so we handle that
            result = predict_fraud_fn.__wrapped__(MockRequest(), claim)
        except AttributeError:
            # If no __wrapped__ (no rate limiter), call directly
            result = predict_fraud_fn(MockRequest(), claim)
        except Exception:
            # If rate limiter fails, try calling the function directly
            # by skipping the decorator
            try:
                result = predict_fraud_fn(MockRequest(), claim)
            except Exception as e:
                return {"error": str(e), "is_fraud": False, "probability": 0}

        # Convert Pydantic model to dict if needed
        if hasattr(result, "dict"):
            return result.dict()
        elif hasattr(result, "model_dump"):
            return result.model_dump()
        elif isinstance(result, dict):
            return result
        else:
            return {"result": str(result)}

    return _predict


def create_report_fn(generate_fraud_report_fn):
    """Creates a sync wrapper around the async report generation."""
    import asyncio

    def _report(claim_data: dict, prediction: dict) -> str:
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # We're already in an async context — use a new thread
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(
                        asyncio.run,
                        generate_fraud_report_fn(claim_data, prediction)
                    )
                    return future.result(timeout=30)
            else:
                return loop.run_until_complete(
                    generate_fraud_report_fn(claim_data, prediction)
                )
        except Exception as e:
            return f"Error generating report: {str(e)}"

    return _report
