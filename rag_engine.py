import chromadb
import os
from flask import Flask, request, jsonify

# --- Constants ---
KNOWLEDGE_BASE_DIR = "knowledge_base"
PERSIST_DIRECTORY = "F:/lexica_db" # As per your project architecture
COLLECTION_NAME = "lexica_memory"

class RAG_Engine:
    """
    This class contains the core logic for the RAG system. It is now designed
    to be instantiated and used by a web server.
    """
    def __init__(self):
        # Initialize the ChromaDB client with persistence
        self.client = chromadb.PersistentClient(path=PERSIST_DIRECTORY)
        # Get or create the collection
        self.collection = self.client.get_or_create_collection(name=COLLECTION_NAME)
        print("RAG Engine initialized. Ready to query The Tesseract.")
        # We can run the update on startup to ensure the KB is fresh
        self.update_knowledge_base()

    def _load_and_chunk_file(self, file_path):
        """
        Loads a text file and splits it into chunks (paragraphs in this case).
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        # A simple chunking strategy: split by double newlines
        chunks = content.split('\n\n')
        return [chunk.strip() for chunk in chunks if chunk.strip()]

    def update_knowledge_base(self):
        """
        Scans the knowledge base directory, processes each file, and updates
        the vector store. This is simplified to run once at startup.
        """
        print("Scanning knowledge base...")
        if not os.path.exists(KNOWLEDGE_BASE_DIR):
            os.makedirs(KNOWLEDGE_BASE_DIR)
            
        files_to_process = [f for f in os.listdir(KNOWLEDGE_BASE_DIR) if f.endswith('.txt')]
        
        if not files_to_process:
            print("Knowledge base directory is empty. Skipping update.")
            return

        all_chunks = []
        metadatas = []
        ids = []
        
        for file_name in files_to_process:
            file_path = os.path.join(KNOWLEDGE_BASE_DIR, file_name)
            print(f"  - Processing {file_name}")
            chunks = self._load_and_chunk_file(file_path)
            
            for i, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                metadatas.append({'source': file_name})
                ids.append(f"{file_name}_{i}")

        if not all_chunks:
            print("No new documents found to update.")
            return

        print(f"Updating The Tesseract with {len(all_chunks)} chunks.")
        self.collection.add(
            documents=all_chunks,
            metadatas=metadatas,
            ids=ids
        )
        print("Knowledge base update complete.")

    def query(self, query_text, n_results=3):
        """
        Performs a semantic search on the vector store.
        """
        print(f"\nQuerying The Tesseract with: '{query_text}'")
        results = self.collection.query(
            query_texts=[query_text],
            n_results=n_results
        )
        return results['documents'][0] if results and results['documents'] else []

# --- Flask App Initialization ---
app = Flask(__name__)
# Instantiate our RAG engine. This will be a single instance for the entire app.
rag_engine = RAG_Engine()

@app.route('/query', methods=['POST'])
def handle_query():
    """
    The API endpoint to receive queries and return results from the RAG engine.
    """
    data = request.get_json()
    if not data or 'query' not in data:
        return jsonify({"error": "Invalid request. 'query' field is required."}), 400

    query_text = data['query']
    retrieved_docs = rag_engine.query(query_text)

    return jsonify({"results": retrieved_docs})

# --- Main execution block to run the Flask server ---
if __name__ == "__main__":
    # Host='0.0.0.0' makes it accessible on your local network
    # Port 5000 is a common default for Flask apps
    app.run(host='0.0.0.0', port=5000, debug=True)