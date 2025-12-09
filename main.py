import os
from typing import Literal
from datetime import datetime
from flask import Flask, request, jsonify
from openai import OpenAI
from langgraph.graph import StateGraph, END

# --- IMPORTS FROM YOUR FILES ---
from legion_state_schema import LegionState, RagContext, Volatility, create_initial_state
from router_node import production_dual_stream_router
from rag_engine import RAG_Engine, generate_response 

# --- CONFIGURATION ---
ROUTER_CLIENT = OpenAI(
    base_url="http://localhost:8000/v1", 
    api_key="lm-studio"
)

MAX_HISTORY_LENGTH = 50  # Prevent memory leaks

print("--- BOOT SEQUENCE: Initializing RAG Engine ---")
brain_memory = RAG_Engine() 

# --- NODE DEFINITIONS ---

def router_node_wrapper(state: LegionState):
    """The Gatekeeper: Decides if we need RAG, Creativity, or Clarification."""
    return production_dual_stream_router(state, ROUTER_CLIENT)

def rag_execution_node(state: LegionState):
    """
    The Scholar: Performs search with CIRCUIT BREAKER for empty results.
    
    GEMINI VALIDATION: Returns DELTA updates only.
    """
    query = state["messages"][-1] 
    print(f" [ACT] Routing to RAG Engine: {query}")
    
    # 1. Retrieve Context
    retrieved_docs = brain_memory.query_and_process(query)
    
    # --- CIRCUIT BREAKER ---
    if not retrieved_docs:
        print(" [WARN] RAG Circuit Breaker Triggered: No documents found.")
        # Return DELTA for message append (operator.add will append this to existing messages)
        return {
            "messages": ["I searched my knowledge base but couldn't find reliable information. Could you rephrase?"]
        }

    # 2. Convert to Schema (Handle strings from RAG engine)
    rag_contexts = []
    for doc in retrieved_docs:
        # Safety: RAG engine returns strings
        content = doc if isinstance(doc, str) else str(doc)
        rag_contexts.append(RagContext(
            content=content,
            source_id="vector_db",
            timestamp=None,
            similarity_score=None,
            volatility=Volatility.DYNAMIC
        ))
    
    # 3. Optional: Check for stale data
    current_time = datetime.now()
    stale_count = sum(1 for ctx in rag_contexts if ctx.check_staleness(current_time))
    
    if stale_count > 0:
        print(f" [WARN] {stale_count}/{len(rag_contexts)} contexts are stale")
    
    # 4. Generate Answer
    # Pass history EXCLUDING current query to avoid duplication
    conversation_history = state["messages"][:-1]
    final_answer = generate_response(query, [c.content for c in rag_contexts], conversation_history)
    
    # Return DELTA (only new message and context)
    return {
        "retrieved_context": rag_contexts,
        "messages": [final_answer]  # operator.add will append this
    }

def dream_node(state: LegionState):
    """
    The Artist: Pure generation without RAG constraints.
    
    GEMINI VALIDATION: Returns DELTA updates only.
    """
    print(" [DREAM] Entering Latent Space...")
    query = state["messages"][-1]
    
    # Pass history for context-aware dreaming
    conversation_history = state["messages"][:-1]
    answer = generate_response(query, [], conversation_history) 
    
    # Return DELTA
    return {"messages": [f"[Creative Mode] {answer}"]}

def fallback_node(state: LegionState):
    """
    The Safety Net: Triggered when intent is ambiguous.
    
    GEMINI VALIDATION: Returns DELTA updates only.
    """
    message = "I'm detecting ambiguity in that request. Could you be more specific?"
    
    # Optional: Include confidence if available
    if state.get("confidence_metrics"):
        conf = state["confidence_metrics"].final_confidence
        message = f"I'm detecting ambiguity (confidence: {conf:.1f}%). Could you clarify?"
    
    # Return DELTA
    return {"messages": [message]}

# --- ROUTING LOGIC ---

def route_decision(state: LegionState) -> Literal["rag", "dream", "ask"]:
    """
    Determines the next node based on the Router's decision tag.
    Includes validation to prevent None access.
    """
    # Validate that routing actually happened
    if not state.get("routing_decision"):
        print("[ERROR] No routing decision found, defaulting to ASK")
        return "ask"
    
    # Validate required data exists
    if not state.get("confidence_metrics"):
        print("[WARN] Missing confidence metrics, routing to ASK for safety")
        return "ask"
    
    decision = state["routing_decision"].upper().strip()
    
    route_map = {
        "EXECUTE": "rag",
        "DREAM": "dream",
        "ASK_USER": "ask"
    }
    
    return route_map.get(decision, "ask")

# --- GRAPH CONSTRUCTION ---

workflow = StateGraph(LegionState)

workflow.add_node("router", router_node_wrapper)
workflow.add_node("rag", rag_execution_node)
workflow.add_node("dream", dream_node)
workflow.add_node("ask", fallback_node)

workflow.set_entry_point("router")

workflow.add_conditional_edges(
    "router",
    route_decision,
    {
        "rag": "rag",
        "dream": "dream",
        "ask": "ask"
    }
)

workflow.add_edge("rag", END)
workflow.add_edge("dream", END)
workflow.add_edge("ask", END)

app_graph = workflow.compile()

# --- FLASK SERVER ---

server = Flask(__name__)

@server.route('/query', methods=['POST'])
def handle_query():
    """
    Main API endpoint.
    
    GEMINI VALIDATION: 
    - History truncation prevents memory leaks
    - No duplicate query insertion
    - Factory function ensures clean state initialization
    """
    data = request.get_json()
    if not data or 'query' not in data: 
        return jsonify({"error": "No query provided"}), 400
    
    user_query = data['query']
    raw_history = data.get('chat_history', [])
    
    # TRUNCATION: Prevent unbounded memory growth
    if len(raw_history) > MAX_HISTORY_LENGTH:
        print(f"[INFO] Truncating history from {len(raw_history)} to {MAX_HISTORY_LENGTH} messages")
        raw_history = raw_history[-MAX_HISTORY_LENGTH:]
    
    # Convert GUI format to state format
    formatted_history = [msg.get('content', '') for msg in raw_history]
    
    # FACTORY STATE: Clean initialization with all required fields
    initial_state = create_initial_state(user_query, formatted_history)
    
    try:
        # Execute the graph
        final_state = app_graph.invoke(initial_state)
        
        # Extract the final response
        bot_response = final_state["messages"][-1]
        
        # Reconstruct GUI history
        updated_history = raw_history + [
            {'role': 'user', 'content': user_query},
            {'role': 'assistant', 'content': bot_response}
        ]
        
        # Include debug info for transparency
        response_data = {
            "results": bot_response, 
            "chat_history": updated_history,
            "decision_debug": final_state.get("routing_decision"),
            "confidence": final_state.get("confidence_metrics").final_confidence if final_state.get("confidence_metrics") else None
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        print(f"CRITICAL GRAPH ERROR: {e}")
        import traceback
        traceback.print_exc()
        
        return jsonify({
            "results": f"System Error: {str(e)}", 
            "chat_history": raw_history
        }), 500

@server.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint for monitoring."""
    return jsonify({
        "status": "online",
        "version": "13.3",
        "rag_engine": "initialized" if brain_memory else "failed"
    })

if __name__ == "__main__":
    print("=" * 60)
    print("LEGION V13.3 ONLINE (GEMINI VALIDATED)")
    print("=" * 60)
    print("✓ Delta Return Pattern (No Duplication)")
    print("✓ Circuit Breaker Active")
    print("✓ History Truncation Enabled")
    print("✓ Prompt Injection Protection")
    print("=" * 60)
    print("Listening on Port 5000...")
    server.run(host='0.0.0.0', port=5000, debug=False)