import json
import math
import os
from typing import Optional, Dict, Any, List
from legion_state_schema import LegionState, RouterOutput, ConfidenceMetrics

# --- Configuration ---
# Update this if using a different model identifier in LM Studio/vLLM
MODEL_NAME = os.getenv("LEGION_MODEL_NAME", "qwen-2.5-7b-instruct") 
ROUTER_TEMPERATURE = 0.3 

def calculate_classification_uncertainty(logprobs_data) -> float:
    """
    Measures uncertainty using Entropy analysis on token logprobs.
    Returns a score between 0.0 (certain) and 1.0 (pure chaos).
    
    How it works:
    1. Extract probability distributions from top-k token predictions
    2. Calculate Shannon entropy: H = -Σ(p * log2(p))
    3. Normalize by max possible entropy for fair comparison
    4. Average across first 5 tokens for stability
    """
    if not logprobs_data or not logprobs_data.content:
        return 0.5  # Default: moderate uncertainty
        
    uncertainties = []
    
    # Analyze the first 5 tokens for classification stability
    for token_data in logprobs_data.content[:5]:
        if token_data.top_logprobs:
            # Convert log-probabilities to probabilities
            probs = [math.exp(alt.logprob) for alt in token_data.top_logprobs]
            
            if probs:
                # Shannon entropy
                entropy = -sum(p * math.log2(p) for p in probs if p > 0)
                
                # Normalize by theoretical maximum
                max_entropy = math.log2(len(probs)) if len(probs) > 1 else 1.0
                normalized = entropy / max_entropy if max_entropy > 0 else 0.0
                
                uncertainties.append(normalized)
    
    if not uncertainties:
        return 0.5
        
    return round(sum(uncertainties) / len(uncertainties), 3)

def create_fallback_update(error_reason: str) -> Dict:
    """
    Safety net: Returns a DELTA update even if the LLM crashes.
    
    CRITICAL: This must return ONLY changed fields (delta pattern).
    LangGraph will merge this with the existing state.
    """
    print(f"  ⚠️  Router Fallback Triggered: {error_reason}")
    
    # Return ONLY the fields that need to change
    return {
        "router_output": RouterOutput(
            intent_category="General",
            keyword_match=["error", "fallback"],
            ambiguity_score=1.0
        ),
        "confidence_metrics": ConfidenceMetrics(
            consensus_score=0.0, 
            probability_score=0.0, 
            intent_score=0.0, 
            final_confidence=0.0
        ),
        "routing_decision": "ASK_USER",
        "last_error": error_reason
    }

def production_dual_stream_router(state: LegionState, llm_client: Any) -> Dict:
    """
    The Production V13.3 Router - Dual-Stream Intent Classification.
    
    Architecture:
    - Stream 1 (Intent): Semantic category (Coding, Factual, Creative, Safety)
    - Stream 2 (Keywords): Specific terms for RAG retrieval
    - Entropy Analysis: Measures model confidence via logprob distribution
    - Tri-Band Scoring: Combines consensus, probability, and intent signals
    
    RETURNS: A dict of UPDATES only (not the full state) to prevent duplication.
    LangGraph's operator.add will append these to existing state fields.
    """
    print("\n--- NODE: Production Dual-Stream Router ---")
    
    user_query = state["messages"][-1] if state["messages"] else ""
    if not user_query:
        return create_fallback_update("Empty user query")

    # Enhanced logging
    print(f"  > Conversation Depth: {len(state['messages'])} messages")
    print(f"  > Analyzing Query: {user_query[:80]}{'...' if len(user_query) > 80 else ''}")

    # Anti-Injection Protection: Wrap user query in XML tags
    # This prevents the user from injecting "Ignore previous instructions" type attacks
    prompt = f"Classify the following query inside the tags: <user_query>{user_query}</user_query>"
    
    system_prompt = """You are the Legion Router. Classify user queries into intent categories.

You must respond with ONLY valid JSON matching this exact structure:
{
  "intent_category": "Coding|Factual|Creative|Safety_Critical",
  "keyword_match": ["keyword1", "keyword2", "keyword3"]
}

Categories:
- Coding: Programming, debugging, code review, technical implementation
- Factual: Questions with verifiable answers, definitions, explanations
- Creative: Writing, art, storytelling, hypotheticals, brainstorming
- Safety_Critical: Medical, legal, financial advice, dangerous content

Output ONLY the JSON. No preamble, no explanation."""
    
    try:
        response = llm_client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},  # Force JSON output
            logprobs=True,  # Enable entropy calculation
            top_logprobs=5,  # Get top-5 alternatives for each token
            max_tokens=150,
            temperature=ROUTER_TEMPERATURE
        )

        if not response.choices:
            return create_fallback_update("No response from LLM")

        content = response.choices[0].message.content
        
        try:
            data = json.loads(content)
        except json.JSONDecodeError as e:
            return create_fallback_update(f"Invalid JSON output: {str(e)}")
        
        # Calculate real ambiguity via entropy
        real_ambiguity = calculate_classification_uncertainty(response.choices[0].logprobs)
        
        # Sanitize keywords with validation
        raw_keywords = data.get("keyword_match", [])
        safe_keywords = (raw_keywords if isinstance(raw_keywords, list) else [])[:3]
        
        # ENHANCEMENT: Ensure we always have at least one keyword
        if not safe_keywords:
            safe_keywords = ["general"]

        router_data = RouterOutput(
            intent_category=data.get("intent_category", "General"),
            keyword_match=safe_keywords, 
            ambiguity_score=real_ambiguity
        )

        # --- Tri-Band Scoring Logic ---
        
        # Band 1: Consensus Score (inverse of ambiguity)
        # High consensus = low entropy = model is confident
        consensus = max(0.0, 0.95 - real_ambiguity)
        
        # Band 2: First Token Probability
        # Measures how confident the model is in its first classification token
        first_token_prob = 0.8  # Default
        if response.choices[0].logprobs and response.choices[0].logprobs.content:
            first_token_prob = math.exp(response.choices[0].logprobs.content[0].logprob)

        # Band 3: Intent-Based Score
        # Different intents have different risk profiles
        intent_map = {
            "Coding": 0.90,          # High confidence allowed
            "Factual": 0.93,         # Very high confidence needed
            "Creative": 0.88,        # Slightly lower threshold
            "Safety_Critical": 0.60  # Conservative - always ask user
        }
        intent_score = intent_map.get(router_data.intent_category, 0.75)

        # Weighted combination (scale to 0-100)
        final_score_val = (
            0.4 * consensus +        # 40% weight on agreement
            0.3 * first_token_prob + # 30% weight on model confidence
            0.3 * intent_score       # 30% weight on intent safety
        ) * 100
        
        # Decision Thresholds
        if final_score_val >= 75:
            decision = "EXECUTE"     # Green light: High confidence
        elif final_score_val >= 50:
            decision = "ASK_USER"    # Yellow light: Clarify first
        else:
            decision = "DREAM"       # Red light: Pure creativity mode

        metrics = ConfidenceMetrics(
            consensus_score=round(consensus, 3),
            probability_score=round(first_token_prob, 3),
            intent_score=round(intent_score, 3),
            final_confidence=round(final_score_val, 2)
        )

        # Enhanced logging
        print(f"  > Intent: {router_data.intent_category}")
        print(f"  > Ambiguity: {real_ambiguity:.3f}")
        print(f"  > Final Confidence: {final_score_val:.1f}%")
        print(f"  > Decision: {decision}")

        # V13.3 CRITICAL: Return ONLY the delta updates
        # LangGraph will merge these fields into the existing state
        # DO NOT return the full state or messages list
        return {
            "router_output": router_data,
            "confidence_metrics": metrics,
            "routing_decision": decision,
            "last_error": None  # Clear any previous errors
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return create_fallback_update(f"Unhandled Exception: {str(e)}")