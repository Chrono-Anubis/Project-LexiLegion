# rag_engine.py (v8 with Contextual Reordering)

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
    The Brain: Now featuring Contextual Reordering to combat the 'Lost in the Middle' problem.
    """
    def __init__(self):
        # ... (init remains the same) ...
        self.client = chromadb.PersistentClient(path=PERSIST_DIRECTORY)
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        self.cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)
        self.collection = self.client.get_or_create_collection(name=COLLECTION_NAME)
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        self.initialize_knowledge_base()
        print("RAG Engine initialized with Contextual Reordering. The Brain is online.")

    def initialize_knowledge_base(self):
        # ... (this function remains the same) ...
        if self.collection.count() > 0: return
        print("Initializing knowledge base...")
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
            print(f"Successfully processed {len(all_chunks)} chunks.")

    def generate_multiple_queries(self, original_query: str) -> list[str]:
        # ... (this function remains the same) ...
        prompt = f"You are an AI assistant. Generate 3 different versions of the user question to retrieve relevant documents from a vector database. Provide these alternative questions separated by newlines.\nOriginal question: {original_query}"
        try:
            response = ollama.chat(model='phi3:mini', messages=[{'role': 'user', 'content': prompt}], stream=False)
            generated_queries = response['message']['content'].strip().split('\n')
            return [original_query] + [q for q in generated_queries if q]
        except Exception: return [original_query]

    def reorder_context(self, query_text, context_docs):
        """
        Uses an LLM to reorder the retrieved documents for optimal generation.
        """
        print("--- Performing Contextual Reordering ---")
        prompt = f"""
        Given the following user query and a list of retrieved documents, your task is to reorder the documents.
        The most relevant documents that directly answer the query should be placed at the beginning and end of the list.
        Less relevant, contextual documents should be placed in the middle.

        User Query: {query_text}

        Retrieved Documents:
        {"\n---\n".join(context_docs)}

        Return the reordered list of documents, separated by '---'.
        """
        try:
            response = ollama.chat(model='phi3:mini', messages=[{'role': 'user', 'content': prompt}], stream=False)
            reordered_text = response['message']['content']
            # Split the reordered text back into a list of documents
            reordered_docs = reordered_text.split('---')
            return [doc.strip() for doc in reordered_docs if doc.strip()]
        except Exception as e:
            print(f"Error during context reordering: {e}")
            return context_docs # Fallback to original order

    def query_and_process(self, query_text, retrieve_n=10, final_n=3):
        """
        The full pipeline: Multi-Query -> Fusion -> Re-ranking -> Contextual Reordering
        """
        # 1. & 2. Multi-Query and Retrieval
        all_queries = self.generate_multiple_queries(query_text)
        all_results = []
        for q in all_queries:
            query_embedding = self.embedding_model.encode([q]).tolist()
            results = self.collection.query(query_embeddings=query_embedding, n_results=retrieve_n)
            if results.get('documents'):
                all_results.append(results['documents'][0])
        if not all_results: return []

        # 3. Fuse the ranks using RRF
        fused_scores = defaultdict(float)
        for rank_list in all_results:
            for rank, doc in enumerate(rank_list):
                fused_scores[doc] += 1 / (60 + rank)
        reranked_by_fusion = sorted(fused_scores.items(), key=lambda item: item[1], reverse=True)
        fused_docs = [doc for doc, score in reranked_by_fusion][:retrieve_n]
        if not fused_docs: return []

        # 4. Re-rank with Cross-Encoder
        pairs = [[query_text, doc] for doc in fused_docs]
        scores = self.cross_encoder.predict(pairs)
        final_scored_docs = sorted(zip(scores, fused_docs), key=lambda x: x[0], reverse=True)
        
        # 5. Contextual Reordering on the top N results
        top_docs = [doc for score, doc in final_scored_docs][:final_n]
        reordered_docs = self.reorder_context(query_text, top_docs)

        return reordered_docs

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
    try:
        response = ollama.chat(model='phi3:mini', messages=current_call_history, stream=False)
        return response['message']['content']
    except Exception as e:
        print(f"Error communicating with Ollama: {e}")
        return "Error: Could not communicate with the local AI model."

app = Flask(__name__)
rag_engine_instance = RAG_Engine()

@app.route('/query', methods=['POST'])
def handle_query():
    data = request.get_json()
    if not data or 'query' not in data: return jsonify({"error": "'query' field is required."}), 400
    query_text = data['query']
    client_chat_history = data.get('chat_history', [])
    retrieved_context = rag_engine_instance.query_and_process(query_text)
    final_answer = generate_response(query_text, retrieved_context, client_chat_history)
    updated_chat_history = client_chat_history + [{'role': 'user', 'content': query_text}, {'role': 'assistant', 'content': final_answer}]
    return jsonify({"results": final_answer, "chat_history": updated_chat_history})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=False)
