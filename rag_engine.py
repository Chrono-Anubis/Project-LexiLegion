# rag_engine.py

from flask import Flask, request, jsonify
import chromadb
import os
import ollama

# --- Constants ---
PERSIST_DIRECTORY = "F:/lexica_db"  # Make sure this drive is available
KNOWLEDGE_BASE_DIR = "./knowledge_base"
COLLECTION_NAME = "lexica_memory"

# --- Global In-Memory Store for Conversational History ---
# For a production system, this would be moved to a session-based cache like Redis
chat_history = []

class RAG_Engine:
    """
    The Brain: Contains the core RAG logic, including document processing,
    embedding, and retrieval from the vector store.
    """
    def __init__(self):
        self.client = chromadb.PersistentClient(path=PERSIST_DIRECTORY)
        self.collection = self.client.get_or_create_collection(name=COLLECTION_NAME)
        self.initialize_knowledge_base()
        print("RAG Engine initialized. The Brain is online.")

    def initialize_knowledge_base(self):
        """Checks if the knowledge base needs to be populated and does so if necessary."""
        if self.collection.count() == 0:
            print("Knowledge base is empty. Initializing from documents...")
            docs = []
            doc_ids = []
            doc_id_counter = 1
            for filename in os.listdir(KNOWLEDGE_BASE_DIR):
                if filename.endswith(".txt"):
                    filepath = os.path.join(KNOWLEDGE_BASE_DIR, filename)
                    with open(filepath, 'r', encoding='utf-8') as f:
                        content = f.read()
                        docs.append(content)
                        doc_ids.append(f"doc_{doc_id_counter}")
                        doc_id_counter += 1
            
            if docs:
                self.collection.add(
                    documents=docs,
                    ids=doc_ids
                )
                print(f"Successfully added {len(docs)} documents to The Tesseract.")
            else:
                print("Warning: No .txt files found in the knowledge_base directory.")

    def query(self, query_text, n_results=2):
        """
        Queries the vector store to find the most relevant documents.
        """
        results = self.collection.query(
            query_texts=[query_text],
            n_results=n_results
        )
        return results['documents'][0] if results['documents'] else []

def generate_conversational_response(user_question, context):
    """
    Generates a response using the local Ollama model,
    maintaining conversational history.
    """
    # Construct the full prompt with context for this turn
    prompt_with_context = f"""
    Using ONLY the following context, answer the user's question.
    If the context does not contain the answer, state that you do not have that information.
    Do not use any prior knowledge.

    Context:
    {"---".join(context)}

    Question:
    {user_question}
    """
    
    # Append the user's new message to the history for the LLM
    chat_history.append({'role': 'user', 'content': prompt_with_context})
    print("--- Sending Chat History to Local LLM ---")

    try:
        response = ollama.chat(
            model='phi3:mini',
            messages=chat_history,
            stream=False
        )
        assistant_response = response['message']['content']
        # Append the assistant's response to remember it for the next turn
        chat_history.append({'role': 'assistant', 'content': assistant_response})
        return assistant_response
    except Exception as e:
        print(f"Error communicating with Ollama: {e}")
        return "Error: Could not communicate with the local AI model."

# --- Flask App Initialization ---
app = Flask(__name__)
# Instantiate a single instance of the RAG engine for the app
rag_engine_instance = RAG_Engine()

@app.route('/query', methods=['POST'])
def handle_query():
    """
    The main API endpoint. Receives a query, retrieves context from
    the RAG engine, generates a final answer, and returns it.
    """
    data = request.get_json()
    if not data or 'query' not in data:
        return jsonify({"error": "Invalid request. 'query' field is required."}), 400

    query_text = data['query']
    
    # Step 1: Retrieve context from The Tesseract
    retrieved_context = rag_engine_instance.query(query_text)
    
    # Step 2: Generate a conversational response using the context
    final_answer = generate_conversational_response(query_text, retrieved_context)

    # Step 3: Return the final, AI-generated answer
    return jsonify({"results": final_answer})

@app.route('/new_chat', methods=['POST'])
def new_chat():
    """
    Clears the chat history to start a new conversation.
    """
    global chat_history
    chat_history = []
    print("--- New Chat Started: Short-term memory cleared. ---")
    return jsonify({"status": "success", "message": "New chat session started."})

# Main execution block to run the Flask server
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=False)
