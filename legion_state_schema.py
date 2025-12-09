from typing import TypedDict, Annotated, List, Optional
from pydantic import BaseModel, Field
from datetime import datetime
import operator
from enum import Enum

# --- 1. The Sub-State Models (The "Thought" Packets) ---

class RouterOutput(BaseModel):
    """The raw output from the Dual-Stream Router."""
    intent_category: str = Field(description="The user's underlying goal (e.g., 'Coding', 'Factual', 'Creative').")
    keyword_match: List[str] = Field(description="Specific keywords identified.")
    ambiguity_score: float = Field(description="0.0 to 1.0 score of how confusing the prompt is.")

class ConfidenceMetrics(BaseModel):
    """The calculated scores for the Tri-Band Logic."""
    consensus_score: float  # Agreement between Intent and Keyword streams
    probability_score: float # Raw model log-probs
    intent_score: float     # Safety/Answerability assessment
    
    # The Final Weighted Score (0-100)
    final_confidence: float 

class Volatility(str, Enum):
    """Defines how fast information 'rots'."""
    STATIC = "static"       # Never expires (Math, History, Core Rules)
    DYNAMIC = "dynamic"     # Expires slowly (Software versions, APIs)
    EPHEMERAL = "ephemeral" # Expires instantly (Weather, Stock prices, Current user mood)

class RagContext(BaseModel):
    """
    Represents a single retrieved chunk of information.
    Tracks freshness and verification status to prevent stale data injection.
    """
    content: str = Field(description="The actual text retrieved from the Vector DB.")
    source_id: str = Field(description="Where this data came from (File ID, URL).")
    timestamp: datetime = Field(description="When this data was indexed.")
    similarity_score: float = Field(description="Vector similarity score (0.0 - 1.0).")
    
    # The Fix for the "Alzheimer's" Problem
    volatility: Volatility = Field(
        default=Volatility.DYNAMIC, 
        description="How fast does this info rot? STATIC items ignore timestamps."
    )
    
    # The Oracle's Stamp of Approval
    is_verified: bool = Field(default=False, description="Has the Oracle checked this against live data?")
    is_stale: bool = Field(default=False, description="Is this data too old to be trusted (based on volatility)?")

# --- 2. The Main Graph State (The Memory) ---

class LegionState(TypedDict):
    """
    The Central Nervous System of Legion V13.
    This dict is passed between every node in the LangGraph.
    """
    # The Conversation History (Standard LangChain memory)
    messages: Annotated[List[str], operator.add]
    
    # The "Executive Cortex" Data (New V13 Logic)
    router_output: Optional[RouterOutput]
    confidence_metrics: Optional[ConfidenceMetrics]
    
    # The RAG Memory (New Quality Control Layer)
    retrieved_context: List[RagContext]
    
    # The Decision Flag (Controls the Conditional Edge)
    # Options: "EXECUTE" (Green), "ASK_USER" (Yellow), "DREAM" (Red)
    routing_decision: str 
    
    # The "Dream Seed" (Only used if routing_decision == "DREAM")
    dream_seed: Optional[str]