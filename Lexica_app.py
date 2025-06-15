import tkinter as tk
from tkinter import scrolledtext, Entry, Button, END
import requests
import json

# --- Constants ---
API_URL = "http://127.0.0.1:5000/query"

class LexicaClient:
    """
    A simple Tkinter GUI client to interact with the RAG API server.
    """
    def __init__(self, root):
        self.root = root
        self.root.title("Lexica Command Center")

        # --- Configure the window ---
        self.root.geometry("800x600")
        self.root.configure(bg="#2E2E2E")

        # --- Create widgets ---
        # Conversation display
        self.conversation_display = scrolledtext.ScrolledText(
            self.root, 
            wrap=tk.WORD, 
            state='disabled',
            bg="#1E1E1E", 
            fg="#DCDCDC",
            font=("Consolas", 11)
        )
        self.conversation_display.pack(padx=10, pady=10, expand=True, fill='both')

        # Input box
        self.input_box = Entry(
            self.root, 
            bg="#3C3C3C", 
            fg="#FFFFFF", 
            insertbackground="white", # cursor color
            font=("Consolas", 11),
            width=80
        )
        self.input_box.pack(padx=10, pady=(0, 5), fill='x')
        self.input_box.bind("<Return>", self.send_query) # Bind Enter key

        # Send button
        self.send_button = Button(
            self.root, 
            text="Send Query", 
            command=self.send_query,
            bg="#007ACC",
            fg="white",
            font=("Arial", 10, "bold")
        )
        self.send_button.pack(padx=10, pady=(0, 10))

        self.add_text("System: Welcome to the Lexica Command Center. The Brain API is ready.\n")

    def add_text(self, text):
        """Adds text to the conversation display."""
        self.conversation_display.config(state='normal')
        self.conversation_display.insert(END, text + "\n")
        self.conversation_display.config(state='disabled')
        self.conversation_display.see(END) # Auto-scroll

    def send_query(self, event=None):
        """Sends the query from the input box to the RAG API."""
        query = self.input_box.get()
        if not query:
            return

        self.add_text(f"You: {query}")
        self.input_box.delete(0, END)

        try:
            # --- Make the API call to The Brain ---
            response = requests.post(API_URL, json={"query": query})
            response.raise_for_status() # Raise an exception for bad status codes

            # --- Process and display the results ---
            data = response.json()
            results = data.get('results', [])
            
            response_text = "Lexica (from Tesseract):\n"
            if results:
                for i, doc in enumerate(results):
                    response_text += f"  - Retrieved Memory {i+1}: {doc}\n"
            else:
                response_text += "  - No relevant memories found in The Tesseract.\n"
            
            self.add_text(response_text)

        except requests.exceptions.RequestException as e:
            error_message = f"System Error: Could not connect to The Brain API at {API_URL}.\nIs rag_engine.py running?"
            self.add_text(error_message)

# --- Main execution block to run the GUI ---
if __name__ == "__main__":
    root = tk.Tk()
    app = LexicaClient(root)
    root.mainloop()