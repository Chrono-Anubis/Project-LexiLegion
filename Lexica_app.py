import tkinter as tk
from tkinter import scrolledtext, Entry, Button, END
import requests

# --- Constants ---
API_URL = "http://127.0.0.1:5000/query"

class LexicaClient:
    """
    The Orchestrator: a Tkinter GUI client to interact with the RAG API server.
    """
    def __init__(self, root):
        self.root = root
        self.root.title("Lexica Command Center")
        self.root.geometry("800x600")

        # --- Widgets ---
        self.conversation_display = scrolledtext.ScrolledText(root, wrap=tk.WORD, state='disabled')
        self.conversation_display.pack(padx=10, pady=10, expand=True, fill='both')

        self.input_box = Entry(root, width=80)
        self.input_box.pack(padx=10, pady=(0, 5), fill='x')
        self.input_box.bind("<Return>", self.send_query)

        self.send_button = Button(root, text="Send Query", command=self.send_query)
        self.send_button.pack(padx=10, pady=(0, 10))

        self.add_text("System: Lexica Orchestrator is online. Ready to connect to The Brain.\n")

    def add_text(self, text):
        """Adds text to the conversation display."""
        self.conversation_display.config(state='normal')
        self.conversation_display.insert(END, text + "\n")
        self.conversation_display.config(state='disabled')
        self.conversation_display.see(END)

    def send_query(self, event=None):
        """Sends the query from the input box to the RAG API."""
        query = self.input_box.get()
        if not query:
            return

        self.add_text(f"You: {query}")
        self.input_box.delete(0, END)

        try:
            response = requests.post(API_URL, json={"query": query})
            response.raise_for_status()

            data = response.json()
            results = data.get('results', [])
            
            response_text = "Lexica (from Tesseract):\n"
            if results:
                for i, doc in enumerate(results):
                    response_text += f"  - Memory {i+1}: {doc}\n"
            else:
                response_text += "  - No relevant memories were found in The Tesseract.\n"
            
            self.add_text(response_text)

        except requests.exceptions.RequestException:
            error_message = f"System Error: Connection to The Brain failed. Is rag_engine.py running?"
            self.add_text(error_message)

# --- Main execution block to run the GUI ---
if __name__ == "__main__":
    root = tk.Tk()
    app = LexicaClient(root)
    root.mainloop()