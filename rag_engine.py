import chromadb
import os

# --- Constants ---
KNOWLEDGE_BASE_DIR = "knowledge_base"
PERSIST_DIRECTORY = "F:/lexica_db" # As per your project architecture 
COLLECTION_NAME = "lexica_memory"

class RAG_Engine:
    """
    Represents the initial RAG prototype. It can load documents from a directory,
    create a vector store, and perform semantic searches.
    """
    def __init__(self):
        # Initialize the ChromaDB client with persistence
        self.client = chromadb.PersistentClient(path=PERSIST_DIRECTORY)
        # Get or create the collection
        self.collection = self.client.get_or_create_collection(name=COLLECTION_NAME)
        print("RAG Engine initialized. Ready to query The Tesseract.")

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
        the vector store.
        """
        print("Scanning knowledge base...")
        files_to_process = [f for f in os.listdir(KNOWLEDGE_BASE_DIR) if f.endswith('.txt')]
        
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

        print(f"Adding {len(all_chunks)} new chunks to The Tesseract.")
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
        return results['documents'][0] if results else []

# --- Main execution block to demonstrate and test the prototype ---
if __name__ == "__main__":
    # Create the knowledge_base directory if it doesn't exist
    if not os.path.exists(KNOWLEDGE_BASE_DIR):
        os.makedirs(KNOWLEDGE_BASE_DIR)
        # Create a dummy file for the initial test
        with open(os.path.join(KNOWLEDGE_BASE_DIR, 'test_memory.txt'), 'w') as f:
            f.write("The first milestone was a RAG prototype.\n\nThe Tesseract is a ChromaDB vector store.")

    # 1. Initialize the engine
    engine = RAG_Engine()
    
    # 2. Update the knowledge base from the directory
    engine.update_knowledge_base()
    
    # 3. Perform a test query
    retrieved_docs = engine.query("What was the first milestone?")
    
    print("\n--- Test Query Results ---")
    if retrieved_docs:
        for doc in retrieved_docs:
            print(f"- {doc}")
    else:
        print("No relevant documents found.")
    print("--------------------------")