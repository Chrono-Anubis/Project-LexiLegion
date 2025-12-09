from typing import TypedDict, Annotated, List, Optional
from pydantic import BaseModel, Field
from datetime import datetime
import operator
from enum import Enum

# --- 1. The Sub-State Models ---

class RouterOutput(BaseModel):
    """The raw output from the Dual-Stream Router."""
    intent_category: str = Field(description="The user's underlying goal.")
    keyword_match: List[str] = Field(description="Specific keywords identified.")
    ambiguity_score: float = Field(description="0.0 to 1.0 score of confusion.")

class ConfidenceMetrics(BaseModel):
    """The calculated scores for the Tri-Band Logic."""
    consensus_score: float  
    probability_score: float 
    intent_score: float     
    final_confidence: float 

class Volatility(str, Enum):
    """Defines how fast information 'rots'."""
    STATIC = "static"       
    DYNAMIC = "dynamic"     
    EPHEMERAL = "ephemeral" 

class RagContext(BaseModel):
    """
    Represents a single retrieved chunk of information.
    Tracks freshness and verification status to prevent stale data injection.
    """
    content: str
    source_id: str
    timestamp: Optional[datetime] = None
    similarity_score: Optional[float] = None
    volatility: Volatility = Volatility.DYNAMIC
    is_verified: bool = False
    is_stale: bool = False
    
    def check_staleness(self, current_time: datetime, max_age_hours: dict = None) -> bool:
        """
        Determines if this context is stale based on volatility rules.
        """
        if self.volatility == Volatility.STATIC:
            return False  # Static data never expires
        
        if not self.timestamp:
            return True  # Missing timestamp = assume stale
        
        if not max_age_hours:
            max_age_hours = {
                Volatility.EPHEMERAL: 1,    # 1 hour
                Volatility.DYNAMIC: 168,    # 1 week
                Volatility.STATIC: float('inf')
            }
        
        age_hours = (current_time - self.timestamp).total_seconds() / 3600
        threshold = max_age_hours.get(self.volatility, 24)
        
        return age_hours > threshold

# --- 2. The Main Graph State ---

class LegionState(TypedDict):
    """
    The Central Nervous System of Legion V13.3.
    
    CRITICAL ARCHITECTURE NOTE:
    The 'messages' field uses operator.add, which means nodes must return
    DELTA updates, not full state copies. 
    
    CORRECT:   return {"messages": [new_message_only]}
    INCORRECT: return {"messages": state["messages"] + [new_message]}
    
    The second approach causes exponential duplication because LangGraph
    performs: existing_list + returned_list
    """
    # operator.add requires DELTA RETURNS (only new items)
    messages: Annotated[List[str], operator.add]
    
    router_output: Optional[RouterOutput]
    confidence_metrics: Optional[ConfidenceMetrics]
    retrieved_context: List[RagContext]
    routing_decision: str 
    dream_seed: Optional[str]
    last_error: Optional[str]

def create_initial_state(user_query: str, history: List[str] = None) -> LegionState:
    """
    Factory function to create a valid initial state with all required fields.
    Prevents silent failures from missing keys.
    """
    if history is None:
        history = []
    
    return LegionState(
        messages=history + [user_query],
        router_output=None,
        confidence_metrics=None,
        retrieved_context=[],
        routing_decision="",
        dream_seed=None,
        last_error=None
    )