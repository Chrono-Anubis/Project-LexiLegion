from flask import Flask, request, jsonify
import chromadb
import os

# rag_engine.py
import ollama
# ... other imports

app = Flask(__name__)

# This list will store the history of the current conversation
chat_history = []
# --- Constants ---
PERSIST_DIRECTORY = "F:/lexica_db"
COLLECTION_NAME = "lexica_memory"

class RAG_Engine:
    """
    Contains the core RAG logic, designed to be used by the Flask server.
    """
    def __init__(self):
        self.client = chromadb.PersistentClient(path=PERSIST_DIRECTORY)
        self.collection = self.client.get_or_create_collection(name=COLLECTION_NAME)
        print("RAG Engine initialized. The Brain is online.")

def generate_conversational_response(user_question, context):
    """
    Generates a response using the local Ollama model,
    maintaining conversational history.
    """
    # Construct the full prompt with context for this turn
    prompt_with_context = f"""
    Using the following context, answer the user's question.
    If the context does not contain the answer, state that you don't have enough information.

    Context:
    {context}

    Question:
    {user_question}
    """
    
    # Append the user's new message to the history
    # This prepares it for the Ollama API call
    chat_history.append({'role': 'user', 'content': prompt_with_context})

    print("--- Sending Chat History to Local LLM ---")

    try:
        # Send the ENTIRE chat history to Ollama
        response = ollama.chat(
            model='phi3:mini',
            messages=chat_history, # Pass the whole history
            stream=False
        )
        
        # Extract the assistant's response
        assistant_response = response['message']['content']
        
        # Append the assistant's response to the history to remember it for next time
        chat_history.append({'role': 'assistant', 'content': assistant_response})
        
        return assistant_response

    except Exception as e:
        print(f"Error communicating with Ollama: {e}")
        return "There was an error communicating with the local AI model."

# --- Flask App Initialization ---
app = Flask(__name__)
# Instantiate a single instance of the RAG engine for the app
rag_engine = RAG_Engine()

@app.route('/query', methods=['POST'])
def handle_query():
    """
    The API endpoint to receive queries from a client,
    ask The Brain, and return the results.
    """
    data = request.get_json()
    if not data or 'query' not in data:
        return jsonify({"error": "Invalid request. 'query' field is required."}), 400

    query_text = data['query']
    retrieved_docs = rag_engine.query(query_text)

    return jsonify({"results": retrieved_docs})

# Main execution block to run the Flask server
if __name__ == "__main__":
    # Host='0.0.0.0' makes it accessible on your local network
    app.run(host='0.0.0.0', port=5000, debug=False)