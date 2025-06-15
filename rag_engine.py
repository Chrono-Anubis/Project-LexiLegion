from flask import Flask, request, jsonify
import chromadb
import os

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

    def query(self, query_text, n_results=3):
        """
        Performs a semantic search on the vector store.
        """
        print(f"Querying The Tesseract with: '{query_text}'")
        try:
            results = self.collection.query(
                query_texts=[query_text],
                n_results=n_results
            )
            return results['documents'][0] if results and results['documents'] else []
        except Exception as e:
            print(f"Error querying ChromaDB: {e}")
            return []

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