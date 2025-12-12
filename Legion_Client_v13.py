import tkinter as tk
from tkinter import scrolledtext, Entry, Button, END, Frame, Label, ttk, messagebox
import requests
import threading
import json
import time
import os
from datetime import datetime

# --- Constants ---
# Configurable via environment variable
API_BASE_URL = os.getenv("LEGION_API_URL", "http://127.0.0.1:5000")
CLIENT_VERSION = "13.0.1"

class LegionClient:
    """
    The V13 Command Center (Gold Master).
    Production-Hardened: Thread-Safe, Memory-Limited, Robust Error Handling.
    Includes Version Handshake and Telemetry Export.
    """
    
    # --- Configuration ---
    MAX_HISTORY = 50           
    MAX_TELEMETRY_LINES = 500  
    MAX_QUERY_LENGTH = 5000    
    REQUEST_TIMEOUT = 30       

    def __init__(self, root):
        self.root = root
        self.root.title(f"Legion V13 Command Center v{CLIENT_VERSION}")
        self.root.geometry("1100x700") 
        self.root.configure(bg="#2b2b2b")
        
        # Handle Window Close
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        # --- Client-side Memory & Concurrency ---
        self.chat_history = []
        self.history_lock = threading.Lock() 
        self.active_request = False          

        # --- Layout ---
        main_paned = ttk.PanedWindow(root, orient=tk.HORIZONTAL)
        main_paned.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Left: Conversation
        left_frame = Frame(main_paned, bg="#2b2b2b")
        main_paned.add(left_frame, weight=2)
        
        self.conversation_display = scrolledtext.ScrolledText(left_frame, wrap=tk.WORD, state='disabled', bg="#1e1e1e", fg="#d3d3d3", font=("Consolas", 12))
        self.conversation_display.pack(expand=True, fill='both')
        self.setup_tags()

        # Input Area
        input_frame = Frame(left_frame, bg="#2b2b2b")
        input_frame.pack(fill='x', pady=(10, 0))

        self.input_box = Entry(input_frame, bg="#3c3f41", fg="white", font=("Consolas", 11), insertbackground='white')
        self.input_box.pack(side='left', expand=True, fill='x', ipady=5)
        self.input_box.bind("<Return>", lambda event: self.start_query_thread())

        self.send_button = Button(input_frame, text="SEND", command=self.start_query_thread, bg="#007acc", fg="white", font=("Consolas", 10, "bold"))
        self.send_button.pack(side='left', padx=(5, 0))
        
        self.new_chat_button = Button(input_frame, text="RESET", command=self.new_chat, bg="#444", fg="white", font=("Consolas", 10))
        self.new_chat_button.pack(side='left', padx=(5, 0))

        # Right: Telemetry
        right_frame = Frame(main_paned, bg="#252526", width=350)
        main_paned.add(right_frame, weight=1)
        
        Label(right_frame, text="EXECUTIVE CORTEX", bg="#252526", fg="#00ff00", font=("Consolas", 12, "bold")).pack(pady=10)
        
        self.telemetry_display = scrolledtext.ScrolledText(right_frame, wrap=tk.WORD, state='disabled', bg="#000000", fg="#00ff00", font=("Consolas", 10))
        self.telemetry_display.pack(expand=True, fill='both', padx=5, pady=5)
        
        # Export Button
        self.export_btn = Button(right_frame, text="EXPORT LOG", command=self.export_telemetry, bg="#333", fg="#aaa", font=("Consolas", 9))
        self.export_btn.pack(pady=5)

        # Key Bindings
        self.root.bind("<Control-r>", lambda e: self.new_chat())

        self.add_text(f"System: Legion Client v{CLIENT_VERSION} online.", "system_tag")
        
        # Version Check (Async)
        self.root.after(1000, self.check_version)

    def setup_tags(self):
        self.conversation_display.tag_config("user_tag", foreground="#87ceeb") 
        self.conversation_display.tag_config("legion_tag", foreground="#98fb98") 
        self.conversation_display.tag_config("system_tag", foreground="#ff6347") 
        self.conversation_display.tag_config("thought_tag", foreground="#aaaaaa", font=("Consolas", 10, "italic")) 

    # --- Thread-Safe GUI Helpers ---
    def add_text(self, text, tag=None):
        def _update():
            self.conversation_display.config(state='normal')
            self.conversation_display.insert(END, text + "\n", tag)
            self.conversation_display.config(state='disabled')
            self.conversation_display.see(END)
        self.root.after(0, _update)

    def update_telemetry(self, text):
        def _update():
            self.telemetry_display.config(state='normal')
            self.telemetry_display.insert(END, text + "\n")
            lines = int(self.telemetry_display.index('end-1c').split('.')[0])
            if lines > self.MAX_TELEMETRY_LINES:
                 self.telemetry_display.delete('1.0', f'{lines - self.MAX_TELEMETRY_LINES}.0')
            self.telemetry_display.see(END)
            self.telemetry_display.config(state='disabled')
        self.root.after(0, _update)

    def clear_ui(self):
        def _update():
            self.conversation_display.config(state='normal')
            self.conversation_display.delete(1.0, END)
            self.telemetry_display.config(state='normal')
            self.telemetry_display.delete(1.0, END)
            self.telemetry_display.config(state='disabled')
            self.conversation_display.config(state='disabled')
        self.root.after(0, _update)

    def set_input_state(self, enabled=True):
        state = 'normal' if enabled else 'disabled'
        self.root.after(0, lambda: self.send_button.config(state=state))

    # --- Logic ---

    def check_version(self):
        """Checks if server version matches client version."""
        def _check():
            try:
                # Assuming your Flask app has a /version endpoint
                response = requests.get(f"{API_BASE_URL}/version", timeout=2)
                if response.status_code == 200:
                    server_version = response.json().get("version", "Unknown")
                    self.add_text(f"System: Connected to Server v{server_version}", "system_tag")
                    if server_version != CLIENT_VERSION:
                        self.add_text(f"System: ⚠️ Version Mismatch! (Server: {server_version} != Client: {CLIENT_VERSION})", "system_tag")
            except:
                pass # Silent fail is okay for version check
        thread = threading.Thread(target=_check, daemon=True)
        thread.start()

    def start_query_thread(self):
        if self.active_request:
            self.add_text("System: ⚠️ Please wait for current process to finish.", "system_tag")
            return

        query = self.input_box.get().strip()
        
        # Unicode Sanitation (Basic)
        try:
            query = query.encode('utf-8', errors='ignore').decode('utf-8')
        except:
            pass

        if not query: 
            self.add_text("System: ℹ️ Empty query ignored.", "thought_tag")
            return
        
        if len(query) > self.MAX_QUERY_LENGTH:
             self.add_text(f"System: ⚠️ Query too long ({len(query)} chars). Limit is {self.MAX_QUERY_LENGTH}.", "system_tag")
             return

        self.add_text(f"Douglas: {query}", "user_tag")
        self.input_box.delete(0, END)
        
        self.active_request = True
        self.set_input_state(False)
        self.add_text("[Processing...]", "thought_tag")

        thread = threading.Thread(target=self.send_query, args=(query,), daemon=True)
        thread.start()

    def send_query(self, query):
        with self.history_lock:
            recent_history = self.chat_history[-self.MAX_HISTORY:]

        payload = { "query": query, "chat_history": recent_history }

        try:
            response = requests.post(f"{API_BASE_URL}/query", json=payload, timeout=self.REQUEST_TIMEOUT)
            response.raise_for_status()
            data = response.json()
            
            router_out = data.get("router_output", {})
            metrics = data.get("confidence_metrics", {})
            decision = data.get("routing_decision", "UNKNOWN")
            
            self.update_telemetry("\n" + "=" * 30)
            self.update_telemetry(f"QUERY: {query[:20]}...")
            self.update_telemetry("-" * 30)
            self.update_telemetry(f"INTENT:   {router_out.get('intent_category', 'N/A')}")
            self.update_telemetry(f"KEYWORDS: {router_out.get('keyword_match', [])}")
            self.update_telemetry(f"AMBIGUITY: {router_out.get('ambiguity_score', 0.0):.3f}")
            self.update_telemetry("-" * 30)
            self.update_telemetry(f"CONSENSUS: {metrics.get('consensus_score', 0.0):.2f}")
            self.update_telemetry(f"PROB:      {metrics.get('probability_score', 0.0):.2f}")
            self.update_telemetry("-" * 30)
            self.update_telemetry(f"CONFIDENCE: {metrics.get('final_confidence', 0.0):.1f}%")
            self.update_telemetry(f"DECISION:   [{decision}]")
            self.update_telemetry("=" * 30)

            final_answer = data.get('answer', "No response text generated.")
            
            if decision == "DREAM":
                self.add_text("[System: Dreaming Mind Activated...]", "thought_tag")
            elif decision == "ASK_USER":
                self.add_text("[System: Ambiguity Detected. Requesting Clarification...]", "thought_tag")
                
            self.add_text(f"Legion: {final_answer}\n", "legion_tag")
            
            with self.history_lock:
                self.chat_history.append({"role": "user", "content": query})
                self.chat_history.append({"role": "assistant", "content": final_answer})

        except requests.exceptions.Timeout:
             self.add_text("System: ❌ Request Timed Out (30s). The Mind is slow to wake.", "system_tag")
             # Could add retry logic here if desired
        except requests.exceptions.ConnectionError:
            self.add_text("System: ❌ Connection Refused. Is the V13 Server running?", "system_tag")
        except Exception as e:
            self.add_text(f"System: ❌ Error: {str(e)}", "system_tag")
        finally:
            self.active_request = False
            self.set_input_state(True)

    def export_telemetry(self):
        """Saves current telemetry log to file."""
        content = self.telemetry_display.get("1.0", END)
        filename = f"legion_telemetry_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        try:
            with open(filename, 'w') as f:
                f.write(content)
            self.add_text(f"System: Telemetry saved to {filename}", "system_tag")
        except Exception as e:
            self.add_text(f"System: Failed to export telemetry: {e}", "system_tag")

    def new_chat(self):
        with self.history_lock:
            self.chat_history = []
        self.clear_ui()
        self.add_text("System: Memory Cleared. Ready for new input.\n", "system_tag")

    def on_closing(self):
        """Clean shutdown."""
        if self.active_request:
             if not messagebox.askokcancel("Quit", "A query is processing. Quit anyway?"):
                 return
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = LegionClient(root)
    root.mainloop()