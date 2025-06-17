# rag_engine.py (v7 with Reciprocal Rank Fusion)

from flask import Flask, request, jsonify
import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import ollama
from collections import defaultdict

# --- Constants ---
PERSIST_DIRECTORY = "F:/lexica_db"
KNOWLEDGE_BASE_DIR = "./knowledge_base"
COLLECTION_NAME = "lexica_memory"
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
CROSS_ENCODER_MODEL = 'cross-encoder/ms-marco-MiniLM-L-6-v2'

class RAG_Engine:
    """
    The Brain: Evolved with Reciprocal Rank Fusion for superior context assembly.
    """
    def __init__(self):
        # ... (init remains the same as v6) ...
        self.client = chromadb.PersistentClient(path=PERSIST_DIRECTORY)
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        self.cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)
        self.collection = self.client.get_or_create_collection(name=COLLECTION_NAME)
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        self.initialize_knowledge_base()
        print("RAG Engine initialized with Rank Fusion. The Brain is online.")

    def initialize_knowledge_base(self):
        # ... (this function remains the same as v6) ...
        if self.collection.count() > 0:
            return
        print("Knowledge base is empty. Initializing from documents...")
        all_chunks, all_chunk_ids = [], []
        if not os.path.exists(KNOWLEDGE_BASE_DIR):
            print(f"Warning: Knowledge base directory '{KNOWLEDGE_BASE_DIR}' not found.")
            return
        for filename in os.listdir(KNOWLEDGE_BASE_DIR):
            if filename.endswith(".txt"):
                filepath = os.path.join(KNOWLEDGE_BASE_DIR, filename)
                with open(filepath, 'r', encoding='utf-8') as f:
                    full_text = f.read()
                    chunks = self.text_splitter.split_text(full_text)
                    for i, chunk in enumerate(chunks):
                        all_chunks.append(chunk)
                        all_chunk_ids.append(f"{filename}_chunk_{i}")
        if all_chunks:
            embeddings = self.embedding_model.encode(all_chunks).tolist()
            self.collection.add(embeddings=embeddings, documents=all_chunks, ids=all_chunk_ids)
            print(f"Successfully split docs into {len(all_chunks)} chunks and added to The Tesseract.")
        else:
            print("Warning: No .txt files found to process in the knowledge_base directory.")
            
    def generate_multiple_queries(self, original_query: str) -> list[str]:
        # ... (this function remains the same as v6) ...
        prompt = f"""
        You are an AI language model assistant. Your task is to generate 3 different versions of the given user question to retrieve relevant documents from a vector database.
        By generating multiple perspectives on the user question, your goal is to help the user overcome some of the limitations of distance-based similarity search.
        Provide these alternative questions separated by newlines.
        Original question: {original_query}
        """
        try:
            response = ollama.chat(model='phi3:mini', messages=[{'role': 'user', 'content': prompt}], stream=False)
            generated_queries = response['message']['content'].strip().split('\n')
            all_queries = [original_query] + [q for q in generated_queries if q]
            print(f"--- Generated Queries: {all_queries} ---")
            return all_queries
        except Exception as e:
            print(f"Error generating multiple queries: {e}")
            return [original_query]

    def query_and_fuse(self, query_text, retrieve_n=10, final_n=3):
        """
        Generates multiple queries, retrieves ranked lists, fuses them with RRF,
        and then re-ranks the top results for final context.
        """
        # 1. Generate multiple queries
        all_queries = self.generate_multiple_queries(query_text)
        
        # 2. Retrieve ranked lists for each query
        all_results = []
        for q in all_queries:
            query_embedding = self.embedding_model.encode([q]).tolist()
            results = self.collection.query(query_embeddings=query_embedding, n_results=retrieve_n)
            if results.get('documents'):
                all_results.append(results['documents'][0])

        if not all_results:
            return []

        # 3. Fuse the ranks using Reciprocal Rank Fusion (RRF)
        fused_scores = defaultdict(float)
        k = 60  # Constant for RRF, balances influence of lower-ranked items
        for rank_list in all_results:
            for rank, doc in enumerate(rank_list):
                fused_scores[doc] += 1 / (k + rank)
        
        # Sort documents by their fused score
        reranked_by_fusion = sorted(fused_scores.items(), key=lambda item: item[1], reverse=True)
        
        # Get the top documents from the fused list to pass to the cross-encoder
        fused_docs = [doc for doc, score in reranked_by_fusion]
        top_fused_docs = fused_docs[:retrieve_n] # Take top N from fused list for final check

        if not top_fused_docs:
            return []

        # 4. Re-rank the top fused candidates with the powerful Cross-Encoder
        pairs = [[query_text, doc] for doc in top_fused_docs]
        scores = self.cross_encoder.predict(pairs)
        final_scored_docs = list(zip(scores, top_fused_docs))
        final_scored_docs.sort(key=lambda x: x[0], reverse=True)

        # 5. Return the final, most relevant documents
        final_docs = [doc for score, doc in final_scored_docs]
        return final_docs[:final_n]

# --- Flask routes and generation function remain the same ---
def generate_response(user_question, context, chat_history):
    prompt_with_context = f"""
    Using ONLY the following context, answer the user's question. If the context does not contain the answer, state that you do not have that information. Do not use any prior knowledge.

    Context:
    {"---".join(context)}

    Question:
    {user_question}
    """
    
    current_call_history = chat_history + [{'role': 'user', 'content': prompt_with_context}]
    print("--- Sending stateless payload to Local LLM ---")
    try:
        response = ollama.chat(model='phi3:mini', messages=current_call_history, stream=False)
        assistant_response = response['message']['content']
        return assistant_response
    except Exception as e:
        print(f"Error communicating with Ollama: {e}")
        return "Error: Could not communicate with the local AI model."

app = Flask(__name__)
rag_engine_instance = RAG_Engine()

@app.route('/query', methods=['POST'])
def handle_query():
    data = request.get_json()
    if not data or 'query' not in data:
        return jsonify({"error": "'query' field is required."}), 400

    query_text = data['query']
    client_chat_history = data.get('chat_history', [])
    
    # UPDATED to call the new fusion method
    retrieved_context = rag_engine_instance.query_and_fuse(query_text)
    
    final_answer = generate_response(query_text, retrieved_context, client_chat_history)
    
    updated_chat_history = client_chat_history + [
        {'role': 'user', 'content': query_text},
        {'role': 'assistant', 'content': final_answer}
    ]

    return jsonify({
        "results": final_answer,
        "chat_history": updated_chat_history
    })

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=False)
