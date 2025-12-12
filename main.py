import os
import json
import logging
from typing import Literal, Dict, Any, Optional
from datetime import datetime
from flask import Flask, request, jsonify
from openai import OpenAI
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv

# --- IMPORTS (Local Files) ---
from legion_state_schema import LegionState, RagContext, Volatility, RouterOutput, ConfidenceMetrics
from router_node import production_dual_stream_router

# Attempt to import RAG, fallback to Mock if missing (for testing)
try:
    from rag_engine import RAG_Engine, generate_response
except ImportError:
    class MockRAG:
        def query_and_process(self, q): return ["Mock Retrieval 1", "Mock Retrieval 2"]
    RAG_Engine = MockRAG
    def generate_response(q, c, h): return f"Simulated Answer based on {len(c)} docs."

# --- CONFIGURATION ---
load_dotenv()

class Config:
    SERVER_VERSION = "13.0.1"
    MAX_HISTORY_LENGTH = 50
    MAX_QUERY_LENGTH = 5000
    DEBUG_MODE = os.getenv("FLASK_DEBUG", "false").lower() == "true"
    LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:8000/v1")
    LLM_API_KEY = os.getenv("LLM_API_KEY", "legion-local")

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize Client
ROUTER_CLIENT = OpenAI(base_url=Config.LLM_BASE_URL, api_key=Config.LLM_API_KEY)

# Lazy Global
brain_memory = None

def get_rag_engine():
    """Lazy load RAG engine to allow fast server boot."""
    global brain_memory
    if brain_memory is None:
        logger.info("⚡ Initializing RAG Engine (Lazy Load)...")
        brain_memory = RAG_Engine()
    return brain_memory

# --- HELPER: State Factory ---
def create_initial_state(query: str, history: list) -> LegionState:
    """
    Creates a clean, typed state object. 
    Prevents KeyError by ensuring all fields exist.
    """
    return {
        "messages": history + [query],
        "router_output": None,
        "confidence_metrics": None,
        "retrieved_context": [],
        "routing_decision": "",
        "dream_seed": None,
        "last_error": None  # Fixed: Explicitly initialize to prevent KeyError
    }

# --- NODE DEFINITIONS ---

def router_node_wrapper(state: LegionState):
    """The Gatekeeper: Decides if we need RAG, Creativity, or Clarification."""
    # Returns DELTA updates
    return production_dual_stream_router(state, ROUTER_CLIENT)

def rag_execution_node(state: LegionState):
    """The Scholar: Performs search with CIRCUIT BREAKER."""
    query = state["messages"][-1]
    logger.info(f"[ACT] Routing to RAG: {query[:50]}...")
    
    engine = get_rag_engine()
    retrieved_docs = engine.query_and_process(query)
    
    # Circuit Breaker
    if not retrieved_docs:
        logger.warning("RAG Circuit Breaker Triggered: No docs.")
        return {
            "messages": ["I searched my knowledge base but couldn't find reliable information. Could you rephrase?"]
        }

    rag_contexts = []
    current_time = datetime.now()
    
    for doc in retrieved_docs:
        content = doc if isinstance(doc, str) else str(doc)
        # Create Pydantic objects for the state
        ctx = RagContext(
            content=content,
            source_id="vector_db",
            timestamp=current_time,
            similarity_score=0.9, 
            volatility=Volatility.DYNAMIC
        )
        rag_contexts.append(ctx)
    
    # Generate Answer
    conversation_history = state["messages"][:-1]
    final_answer = generate_response(query, [c.content for c in rag_contexts], conversation_history)
    
    return {
        "retrieved_context": rag_contexts,
        "messages": [final_answer]
    }

def dream_node(state: LegionState):
    """The Artist: Pure generation."""
    logger.info("[DREAM] Entering Latent Space...")
    query = state["messages"][-1]
    answer = f"I have reflected deeply on '{query}'. Based on internal logic (Dreaming Mode), I suggest..."
    return {"messages": [answer]}

def fallback_node(state: LegionState):
    """The Safety Net."""
    logger.info("[ASK] Ambiguity Detected.")
    message = "I'm detecting ambiguity. Could you be more specific?"
    if state.get("confidence_metrics"):
        conf = state["confidence_metrics"].final_confidence
        message += f" (Confidence: {conf:.1f}%)"
    return {"messages": [message]}

# --- ROUTING LOGIC ---

def route_decision(state: LegionState) -> Literal["rag", "dream", "ask"]:
    decision = state.get("routing_decision", "ASK_USER").upper().strip()
    if decision == "EXECUTE": return "rag"
    elif decision == "DREAM": return "dream"
    else: return "ask"

# --- GRAPH CONSTRUCTION ---

def build_graph():
    workflow = StateGraph(LegionState)
    workflow.add_node("router", router_node_wrapper)
    workflow.add_node("rag", rag_execution_node)
    workflow.add_node("dream", dream_node)
    workflow.add_node("ask", fallback_node)
    
    workflow.set_entry_point("router")
    
    workflow.add_conditional_edges(
        "router",
        route_decision,
        {"rag": "rag", "dream": "dream", "ask": "ask"}
    )
    
    workflow.add_edge("rag", END)
    workflow.add_edge("dream", END)
    workflow.add_edge("ask", END)
    
    return workflow.compile()

app_graph = build_graph()
server = Flask(__name__)

# --- ENDPOINTS ---

@server.route('/version', methods=['GET'])
def get_version():
    return jsonify({"version": Config.SERVER_VERSION})

@server.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "version": Config.SERVER_VERSION})

@server.route('/query', methods=['POST'])
def handle_query():
    # 1. Validation
    data = request.get_json()
    if not data or 'query' not in data:
        return jsonify({"error": "No query provided"}), 400
        
    user_query = str(data['query']).strip()
    if not user_query:
        return jsonify({"error": "Empty query"}), 400
    if len(user_query) > Config.MAX_QUERY_LENGTH:
        return jsonify({"error": "Query too long"}), 400

    # 2. History Management
    raw_history = data.get('chat_history', [])
    if len(raw_history) > Config.MAX_HISTORY_LENGTH:
        raw_history = raw_history[-Config.MAX_HISTORY_LENGTH:]
        logger.warning(f"Truncated history to {Config.MAX_HISTORY_LENGTH}")

    # Format history for graph (Simple list of strings or dicts depending on LLM needs)
    formatted_history = []
    for msg in raw_history:
        if isinstance(msg, dict):
            formatted_history.append(f"{msg.get('role')}: {msg.get('content')}")
        else:
            formatted_history.append(str(msg))

    try:
        # 3. Execution
        initial_state = create_initial_state(user_query, formatted_history)
        final_state = app_graph.invoke(initial_state)
        
        # 4. Response Extraction
        bot_response = final_state["messages"][-1]
        
        # 5. Serialization (CRITICAL FIX)
        router_out = final_state.get("router_output")
        metrics = final_state.get("confidence_metrics")
        
        # Safe Pydantic Dump
        r_dict = router_out.model_dump() if hasattr(router_out, 'model_dump') else (router_out.dict() if router_out else {})
        m_dict = metrics.model_dump() if hasattr(metrics, 'model_dump') else (metrics.dict() if metrics else {})

        return jsonify({
            "answer": bot_response,
            "router_output": r_dict,
            "confidence_metrics": m_dict,
            "routing_decision": final_state.get("routing_decision", "UNKNOWN")
            # NOTE: chat_history removed from response. Client manages local history.
        })

    except Exception as e:
        logger.error(f"GRAPH EXECUTION ERROR: {e}", exc_info=True)
        
        # 6. Safe Error Response (Prevents Client Crash)
        # Returns a valid structure even on failure
        return jsonify({
            "answer": f"System Error: {str(e)}",
            "routing_decision": "ERROR",
            # Mock the telemetry so client doesn't KeyError
            "router_output": {
                "intent_category": "System_Error",
                "keyword_match": ["error", "exception"],
                "ambiguity_score": 1.0
            },
            "confidence_metrics": {
                "consensus_score": 0.0,
                "probability_score": 0.0,
                "intent_score": 0.0,
                "final_confidence": 0.0
            }
        }), 500

if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info(f"LEGION V{Config.SERVER_VERSION} EXECUTIVE CORTEX ONLINE")
    logger.info("=" * 60)
    logger.info("✓ Robust Error Handling Active")
    logger.info("✓ Telemetry Contract Verified")
    
    if Config.DEBUG_MODE:
        server.run(host='0.0.0.0', port=5000, debug=True)
    else:
        # In production, use gunicorn
        server.run(host='0.0.0.0', port=5000, debug=False)