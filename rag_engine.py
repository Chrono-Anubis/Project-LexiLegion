from flask import Flask, request, jsonify
import chromadb
from sentence_transformers import SentenceTransformer
import os
import ollama

# --- Constants ---
PERSIST_DIRECTORY = "F:/lexica_db"
KNOWLEDGE_BASE_DIR = "./knowledge_base"
COLLECTION_NAME = "lexica_memory"
EMBEDDING_MODEL = 'all-MiniLM-L6-v2' # Explicitly define the model

class RAG_Engine:
    """
    The Brain: A more robust engine that explicitly handles embeddings.
    """
    def __init__(self):
        self.client = chromadb.PersistentClient(path=PERSIST_DIRECTORY)
        # Load the sentence transformer model
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        self.collection = self.client.get_or_create_collection(name=COLLECTION_NAME)
        self.initialize_knowledge_base()
        print("RAG Engine initialized. The Brain is online.")

    def initialize_knowledge_base(self):
        """Checks if the knowledge base is empty and populates it from documents."""
        if self.collection.count() == 0:
            print("Knowledge base is empty. Initializing from documents...")
            docs = []
            doc_ids = []
            if not os.path.exists(KNOWLEDGE_BASE_DIR):
                print(f"Warning: Knowledge base directory '{KNOWLEDGE_BASE_DIR}' not found.")
                return

            for i, filename in enumerate(os.listdir(KNOWLEDGE_BASE_DIR)):
                if filename.endswith(".txt"):
                    filepath = os.path.join(KNOWLEDGE_BASE_DIR, filename)
                    with open(filepath, 'r', encoding='utf-8') as f:
                        docs.append(f.read())
                        doc_ids.append(f"doc_{i+1}")
            
            if docs:
                # Explicitly generate embeddings for the documents
                embeddings = self.embedding_model.encode(docs).tolist()
                self.collection.add(
                    embeddings=embeddings,
                    documents=docs,
                    ids=doc_ids
                )
                print(f"Successfully added {len(docs)} documents to The Tesseract.")
            else:
                print("Warning: No .txt files found in the knowledge_base directory.")

    def query(self, query_text, n_results=2):
        """Queries the vector store using an explicitly generated query embedding."""
        query_embedding = self.embedding_model.encode([query_text]).tolist()
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=n_results
        )
        return results['documents'][0] if results.get('documents') else []

def generate_response(user_question, context, chat_history):
    """
    A stateless function to generate a response using the provided history.
    """
    prompt_with_context = f"""
    Using ONLY the following context, answer the user's question. If the context does not contain the answer, state that you do not have that information. Do not use any prior knowledge.

    Context:
    {"---".join(context)}

    Question:
    {user_question}
    """
    
    # Create a temporary history for this specific call
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

# --- Flask App Initialization ---
app = Flask(__name__)
rag_engine_instance = RAG_Engine()

@app.route('/query', methods=['POST'])
def handle_query():
    """
    A stateless API endpoint that receives the entire conversation state
    from the client with each request.
    """
    data = request.get_json()
    if not data or 'query' not in data:
        return jsonify({"error": "'query' field is required."}), 400

    query_text = data['query']
    # Receive the chat history from the client, or start a new one if not provided
    client_chat_history = data.get('chat_history', [])
    
    retrieved_context = rag_engine_instance.query(query_text)
    final_answer = generate_response(query_text, retrieved_context, client_chat_history)
    
    # Update the history and send it back to the client
    updated_chat_history = client_chat_history + [
        {'role': 'user', 'content': query_text},
        {'role': 'assistant', 'content': final_answer}
    ]

    return jsonify({
        "results": final_answer,
        "chat_history": updated_chat_history
    })

# Main execution block to run the Flask server
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=False)
