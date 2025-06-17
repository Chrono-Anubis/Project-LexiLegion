# rag_engine.py (v5 with Re-ranking)

from flask import Flask, request, jsonify
import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import ollama

# --- Constants ---
PERSIST_DIRECTORY = "F:/lexica_db"
KNOWLEDGE_BASE_DIR = "./knowledge_base"
COLLECTION_NAME = "lexica_memory"
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
CROSS_ENCODER_MODEL = 'cross-encoder/ms-marco-MiniLM-L-6-v2' # The re-ranking model

class RAG_Engine:
    """
    The Brain: Upgraded with a Re-ranking stage for enhanced retrieval accuracy.
    """
    def __init__(self):
        self.client = chromadb.PersistentClient(path=PERSIST_DIRECTORY)
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        # Load the Cross-Encoder model for re-ranking
        self.cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)
        self.collection = self.client.get_or_create_collection(name=COLLECTION_NAME)
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        self.initialize_knowledge_base()
        print("RAG Engine initialized with Re-ranking. The Brain is online.")

    def initialize_knowledge_base(self):
        """Loads docs, splits them into chunks, and populates the vector store."""
        if self.collection.count() > 0:
            print("Knowledge base already populated.")
            return
        # ... (The rest of this function remains the same as v4)
        print("Knowledge base is empty. Initializing from documents...")
        all_chunks = []
        all_chunk_ids = []
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

    def query_and_rerank(self, query_text, retrieve_n=10, final_n=3):
        """
        Retrieves a broad set of documents and then re-ranks them for relevance.
        """
        # 1. Retrieve a larger set of candidate documents
        query_embedding = self.embedding_model.encode([query_text]).tolist()
        initial_results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=retrieve_n
        )
        candidate_docs = initial_results['documents'][0] if initial_results.get('documents') else []

        if not candidate_docs:
            return []

        # 2. Re-rank the candidates using the Cross-Encoder
        # The Cross-Encoder takes pairs of (query, document) and scores them
        pairs = [[query_text, doc] for doc in candidate_docs]
        scores = self.cross_encoder.predict(pairs)

        # 3. Combine docs with their new scores and sort
        scored_docs = list(zip(scores, candidate_docs))
        scored_docs.sort(key=lambda x: x[0], reverse=True)

        # 4. Return the top N documents after re-ranking
        final_docs = [doc for score, doc in scored_docs]
        return final_docs[:final_n]

# --- The rest of your Flask app code remains the same, but calls the new method ---
# ... (generate_response, app initialization, etc.) ...
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
        response = ollama.chat(
            model='phi3:mini',
            messages=current_call_history,
            stream=False
        )
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
    
    # --- UPDATED LOGIC: Call the new query_and_rerank method ---
    retrieved_context = rag_engine_instance.query_and_rerank(query_text)
    
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
