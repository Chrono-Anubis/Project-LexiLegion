import json
import math
import os
from typing import Optional, Dict, Any, List
from legion_state_schema import (
    LegionState, 
    RouterOutput, 
    ConfidenceMetrics, 
    RagContext, 
    Volatility
)

# --- Configuration ---
MODEL_NAME = os.getenv("LEGION_MODEL_NAME", "qwen-3-8b-int8") 
ROUTER_TEMPERATURE = 0.3 

def calculate_classification_uncertainty(logprobs_data) -> float:
    """
    Measures uncertainty by analyzing the probability distribution
    across alternative token predictions (Entropy).
    """
    if not logprobs_data or not logprobs_data.content:
        return 0.5 
        
    uncertainties = []
    
    # Look at first 5 tokens to capture the classification label stability
    for token_data in logprobs_data.content[:5]:
        if token_data.top_logprobs:
            probs = [math.exp(alt.logprob) for alt in token_data.top_logprobs]
            
            if probs:
                entropy = -sum(p * math.log2(p) for p in probs if p > 0)
                max_entropy = math.log2(len(probs)) if len(probs) > 1 else 1.0
                normalized = entropy / max_entropy if max_entropy > 0 else 0.0
                uncertainties.append(normalized)
    
    if not uncertainties:
        return 0.5
        
    return round(sum(uncertainties) / len(uncertainties), 3)

def create_fallback_state(state: LegionState, error_reason: str) -> LegionState:
    """
    Creates a safe, valid state object when the router fails.
    Prevents downstream graph crashes by populating required fields with defaults.
    """
    print(f"  ⚠️  Router Fallback Triggered: {error_reason}")
    
    fallback_state = state.copy()
    
    # Default to a safe "Ask User" state
    router_data = RouterOutput(
        intent_category="General",
        keyword_match=["error", "fallback", "unknown"],
        ambiguity_score=1.0
    )
    
    metrics = ConfidenceMetrics(
        consensus_score=0.0,
        probability_score=0.0,
        intent_score=0.0,
        final_confidence=0.0
    )
    
    fallback_state.update({
        "router_output": router_data,
        "confidence_metrics": metrics,
        "routing_decision": "ASK_USER" # Safe default
    })
    
    return fallback_state

def production_dual_stream_router(
    state: LegionState, 
    llm_client: Any 
) -> LegionState:
    """
    The Production V13 Router.
    Replaces 'mock' randomness with actual Qwen 3 inference via vLLM.
    """
    print("\n--- NODE: Production Dual-Stream Router ---")
    
    user_query = state["messages"][-1] if state["messages"] else ""
    if not user_query:
        return create_fallback_state(state, "Empty user query")

    print(f"  > Analyzing Query: {user_query[:50]}...")

    system_prompt = """You are the Legion Router. Classify user queries.
    
Output JSON:
{"intent_category": "Coding|Factual|Creative|Safety_Critical", "keyword_match": ["term1", "term2", "term3"]}"""
    
    prompt = f"Classify: {user_query}"

    try:
        response = llm_client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"}, 
            logprobs=True, 
            top_logprobs=5,
            max_tokens=150,
            temperature=ROUTER_TEMPERATURE
        )

        if not response.choices:
             return create_fallback_state(state, "No response from LLM")

        content = response.choices[0].message.content
        
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            return create_fallback_state(state, "Invalid JSON output from LLM")
        
        # Calculate Real Ambiguity
        real_ambiguity = calculate_classification_uncertainty(response.choices[0].logprobs)
        
        # Safe Keyword Handling
        raw_keywords = data.get("keyword_match", [])
        if not isinstance(raw_keywords, list): raw_keywords = []
        safe_keywords = (raw_keywords + ["general", "query", "misc"])[:3]

        router_data = RouterOutput(
            intent_category=data.get("intent_category", "General"),
            keyword_match=safe_keywords, 
            ambiguity_score=real_ambiguity
        )

        # Compute Tri-Band Metrics
        consensus = max(0.0, 0.95 - real_ambiguity)
        
        first_token_prob = 0.8
        if response.choices[0].logprobs and response.choices[0].logprobs.content:
             first_token_prob = math.exp(response.choices[0].logprobs.content[0].logprob)

        intent_map = {
            "Coding": 0.90, "Factual": 0.93, "Creative": 0.88, "Safety_Critical": 0.60
        }
        intent_score = intent_map.get(router_data.intent_category, 0.75)

        final_score_val = (
            0.4 * consensus +
            0.3 * first_token_prob +
            0.3 * intent_score
        ) * 100
        
        if final_score_val >= 75:
            decision = "EXECUTE"
        elif final_score_val >= 50:
            decision = "ASK_USER"
        else:
            decision = "DREAM"

        metrics = ConfidenceMetrics(
            consensus_score=round(consensus, 3),
            probability_score=round(first_token_prob, 3),
            intent_score=round(intent_score, 3),
            final_confidence=round(final_score_val, 2)
        )

        print(f"  > Intent: {router_data.intent_category}")
        print(f"  > Ambiguity: {real_ambiguity:.3f}")
        print(f"  > Decision: {decision}")

        updated_state = state.copy()
        updated_state.update({
            "router_output": router_data,
            "confidence_metrics": metrics,
            "routing_decision": decision
        })
        
        return updated_state

    except Exception as e:
        return create_fallback_state(state, f"Unhandled Exception: {str(e)}")