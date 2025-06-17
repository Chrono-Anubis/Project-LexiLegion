import tkinter as tk
from tkinter import scrolledtext, Entry, Button, END, Frame
import requests

# --- Constants ---
API_BASE_URL = "http://127.0.0.1:5000"

class LexicaClient:
    """
    The Orchestrator: A Tkinter GUI client that now manages its own conversational state.
    """
    def __init__(self, root):
        self.root = root
        self.root.title("Lexica Command Center")
        self.root.geometry("800x600")

        # --- Client-side Conversational Memory ---
        self.chat_history = []

        # --- Widgets ---
        self.conversation_display = scrolledtext.ScrolledText(root, wrap=tk.WORD, state='disabled', bg="#2b2b2b", fg="#d3d3d3", font=("Consolas", 11))
        self.conversation_display.pack(padx=10, pady=10, expand=True, fill='both')

        # Frame for input and buttons
        bottom_frame = Frame(root)
        bottom_frame.pack(padx=10, pady=(0, 10), fill='x')

        self.input_box = Entry(bottom_frame, width=70, bg="#3c3f41", fg="#d3d3d3", insertbackground='white')
        self.input_box.pack(side='left', expand=True, fill='x')
        self.input_box.bind("<Return>", self.send_query)

        self.send_button = Button(bottom_frame, text="Send", command=self.send_query, bg="#007acc", fg="white")
        self.send_button.pack(side='left', padx=(5, 0))
        
        self.new_chat_button = Button(bottom_frame, text="New Chat", command=self.new_chat, bg="#555", fg="white")
        self.new_chat_button.pack(side='left', padx=(5, 0))

        self.add_text("System: Lexica Orchestrator is online. Ready to connect to The Brain.\n")

    def add_text(self, text, tag=None):
        """Adds text to the conversation display with optional color tagging."""
        self.conversation_display.config(state='normal')
        self.conversation_display.insert(END, text + "\n", tag)
        self.conversation_display.config(state='disabled')
        self.conversation_display.see(END)
        # Define tags for colors
        self.conversation_display.tag_config("user_tag", foreground="#87ceeb") # Light blue for user
        self.conversation_display.tag_config("lexica_tag", foreground="#98fb98") # Light green for Lexica

    def send_query(self, event=None):
        """Sends the query and the entire chat history to the RAG API."""
        query = self.input_box.get()
        if not query:
            return

        self.add_text(f"You: {query}", "user_tag")
        self.input_box.delete(0, END)

        payload = {
            "query": query,
            "chat_history": self.chat_history
        }

        try:
            response = requests.post(f"{API_BASE_URL}/query", json=payload)
            response.raise_for_status()

            data = response.json()
            final_answer = data.get('results', "Error: No response from AI.")
            
            # Update the local chat history with the new state from the server
            self.chat_history = data.get('chat_history', [])
            
            self.add_text(f"\nLexica: {final_answer}\n", "lexica_tag")

        except requests.exceptions.RequestException:
            error_message = f"System Error: Connection to The Brain failed. Is rag_engine.py running?"
            self.add_text(error_message)

    def new_chat(self):
        """Starts a new chat by clearing the local history."""
        self.chat_history = []
        self.conversation_display.config(state='normal')
        self.conversation_display.delete(1.0, END)
        self.add_text("System: New chat started. Short-term memory cleared.\n")
        self.conversation_display.config(state='disabled')

# --- Main execution block to run the GUI ---
if __name__ == "__main__":
    root = tk.Tk()
    app = LexicaClient(root)
    root.mainloop()
