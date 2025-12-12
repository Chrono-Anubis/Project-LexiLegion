import os
import time
import logging
import re
import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder
from langchain.text_splitter import RecursiveCharacterTextSplitter
import ollama
from collections import defaultdict
from dotenv import load_dotenv

# --- Configuration ---
load_dotenv()
logger = logging.getLogger(__name__)

class RagConfig:
    PERSIST_DIR = os.getenv("RAG_PERSIST_DIR", "F:/lexica_db")
    KB_DIR = os.getenv("RAG_KB_DIR", "./knowledge_base")
    COLLECTION_NAME = os.getenv("RAG_COLLECTION", "lexica_memory")
    EMBED_MODEL = os.getenv("RAG_EMBED_MODEL", "all-MiniLM-L6-v2")
    RERANK_MODEL = os.getenv("RAG_RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    LLM_MODEL = os.getenv("RAG_LLM_MODEL", "phi3:mini") 
    ENABLE_QUERY_EXPANSION = os.getenv("RAG_QUERY_EXPANSION", "true").lower() == "true"

class RAG_Engine:
    """
    Production RAG Engine for Legion V13.
    Features Multi-Query, RRF Fusion, Cross-Encoder Re-ranking, and Contextual Reordering.
    """
    def __init__(self):
        try:
            self.client = chromadb.PersistentClient(path=RagConfig.PERSIST_DIR)
            self.embedding_model = SentenceTransformer(RagConfig.EMBED_MODEL)
            self.cross_encoder = CrossEncoder(RagConfig.RERANK_MODEL)
            self.collection = self.client.get_or_create_collection(name=RagConfig.COLLECTION_NAME)
            self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            
            logger.info(f"⚡ RAG Engine Online. KB: {RagConfig.KB_DIR}")
            
            if self.collection.count() == 0:
                self.initialize_knowledge_base()
                
        except Exception as e:
            logger.error(f"RAG Engine initialization failed: {e}")
            raise

    def initialize_knowledge_base(self):
        logger.info("Initializing knowledge base from disk...")
        all_chunks, all_chunk_ids = [], []
        
        if not os.path.exists(RagConfig.KB_DIR):
            logger.warning(f"Knowledge base directory '{RagConfig.KB_DIR}' not found. Creating empty KB.")
            os.makedirs(RagConfig.KB_DIR, exist_ok=True)
            return
            
        for filename in os.listdir(RagConfig.KB_DIR):
            if filename.endswith(".txt"):
                filepath = os.path.join(RagConfig.KB_DIR, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        full_text = f.read()
                        if not full_text.strip(): continue
                        
                        chunks = self.text_splitter.split_text(full_text)
                        for i, chunk in enumerate(chunks):
                            all_chunks.append(chunk)
                            all_chunk_ids.append(f"{filename}_chunk_{i}")
                except Exception as e:
                    logger.error(f"Error reading {filename}: {e}")

        if all_chunks:
            # Batch processing to prevent memory issues
            batch_size = 100
            for i in range(0, len(all_chunks), batch_size):
                batch_chunks = all_chunks[i:i+batch_size]
                batch_ids = all_chunk_ids[i:i+batch_size]
                
                embeddings = self.embedding_model.encode(batch_chunks).tolist()
                self.collection.add(embeddings=embeddings, documents=batch_chunks, ids=batch_ids)
                logger.info(f"Indexed batch {i//batch_size + 1}")
            
            logger.info(f"Successfully processed {len(all_chunks)} chunks.")

    def generate_multiple_queries(self, original_query: str, num_queries=3) -> list[str]:
        """Generates alternative search queries to improve recall."""
        prompt = f"Generate {num_queries} alternative search queries for: '{original_query}'. Output only the queries, one per line."
        try:
            response = ollama.chat(model=RagConfig.LLM_MODEL, messages=[{'role': 'user', 'content': prompt}], stream=False)
            raw_queries = response['message']['content'].strip().split('\n')
            
            # Robust cleaning: remove numbering (1., - )
            clean_queries = []
            for q in raw_queries:
                cleaned = re.sub(r'^[\d\-\*\•\.)\]]+\s*', '', q.strip())
                if cleaned and len(cleaned) > 5:
                    clean_queries.append(cleaned)
            
            # Deduplicate
            return list(dict.fromkeys([original_query] + clean_queries))[:num_queries+1]
        except Exception as e: 
            logger.warning(f"Query expansion failed: {e}")
            return [original_query]

    def reciprocal_rank_fusion(self, all_queries, retrieve_n=10, k=60):
        """
        Combines results from multiple queries using RRF algorithm.
        """
        fused_scores = defaultdict(float)
        
        for q in all_queries:
            query_embedding = self.embedding_model.encode([q]).tolist()
            results = self.collection.query(query_embeddings=query_embedding, n_results=retrieve_n)
            
            if results.get('documents') and results['documents'][0]:
                docs = results['documents'][0]
                for rank, doc in enumerate(docs):
                    fused_scores[doc] += 1.0 / (k + rank + 1) # rank is 0-indexed here
        
        return sorted(fused_scores.keys(), key=lambda x: fused_scores[x], reverse=True)

    def reorder_context_deterministic(self, context_docs):
        """
        Reorders documents to place most relevant info at start/end (Lost in the Middle fix).
        """
        if len(context_docs) <= 2: return context_docs
        
        reordered = []
        # Convert to deque or just list handling
        # Strategy: [1, 3, 5, ..., 6, 4, 2] logic
        # Simple implementation: Alternate adding to start and end
        # But for list return: [Best, 3rd, 5th ... 6th, 4th, 2nd]
        # Actually simpler: [Best, ..., Worst, ..., 2nd Best]
        # Common approach: [0, 2, 4, ... 5, 3, 1]
        
        # Let's use the explicit "Best at ends" logic
        # Input is sorted by relevance (Best -> Worst)
        # Output layout: [1, 3, 5, 7, 9, 10, 8, 6, 4, 2]
        
        layout = [None] * len(context_docs)
        left, right = 0, len(context_docs) - 1
        
        for i, doc in enumerate(context_docs):
            if i % 2 == 0:
                layout[left] = doc
                left += 1
            else:
                layout[right] = doc
                right -= 1
        
        return layout

    def query_and_process(self, query_text, retrieve_n=10, final_n=5):
        """
        Optimized retrieval pipeline.
        """
        start_time = time.time()
        
        try:
            # 1. Query Expansion
            if RagConfig.ENABLE_QUERY_EXPANSION:
                all_queries = self.generate_multiple_queries(query_text)
            else:
                all_queries = [query_text]
            
            # 2. Retrieval & RRF Fusion
            fused_docs = self.reciprocal_rank_fusion(all_queries, retrieve_n)
            if not fused_docs: return []
            
            # Take top candidates for re-ranking
            candidates = fused_docs[:retrieve_n] # RRF already sorted them

            # 3. Cross-Encoder Re-ranking
            pairs = [[query_text, doc] for doc in candidates]
            scores = self.cross_encoder.predict(pairs)
            final_scored_docs = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
            top_docs = [doc for score, doc in final_scored_docs][:final_n]
            
            # 4. Contextual Reordering
            final_docs = self.reorder_context_deterministic(top_docs)
            
            elapsed = time.time() - start_time
            logger.info(f"Retrieval complete: {len(final_docs)} docs in {elapsed:.2f}s")
            
            return final_docs

        except Exception as e:
            logger.error(f"Query processing error: {e}", exc_info=True)
            return []

# --- Generation Function (Used by main.py) ---
def generate_response(user_question, context, chat_history):
    """
    Generates the final answer.
    """
    if not context:
        return "I couldn't find relevant information in my knowledge base."

    # Format context with source citations
    numbered_context = [f"[Source {i+1}] {ctx}" for i, ctx in enumerate(context)]
    context_block = "\n\n".join(numbered_context)
    
    system_prompt = f"""You are Legion. Answer the user's question using ONLY the context below. 
    Cite your sources using [Source X] notation.
    If the context does not contain the answer, say "I don't have that information."
    
    Context:
    {context_block}
    """
    
    messages = [{'role': 'system', 'content': system_prompt}]
    
    # Robust History Parsing
    for msg in chat_history:
        if isinstance(msg, dict):
             messages.append(msg)
        elif isinstance(msg, str):
             role = 'user' if 'User:' in msg else 'assistant'
             content = msg.replace('User:', '').replace('Legion:', '').strip()
             messages.append({'role': role, 'content': content})

    messages.append({'role': 'user', 'content': user_question})

    try:
        response = ollama.chat(model=RagConfig.LLM_MODEL, messages=messages, stream=False)
        return response['message']['content']
    except Exception as e:
        logger.error(f"Ollama Generation Error: {e}")
        return "Error: Could not communicate with the local AI model."