import tkinter as tk
from tkinter import scrolledtext, ttk
import threading
import time
import datetime
import json
import os
import ollama  # Requires 'pip install ollama' and the Ollama app running

# --- CONFIGURATION ---
#MODEL_NAME = "mistral"  # Change to your local model (llama3, mistral, etc.)
MODEL_NAME = "llama3:8b"
MEMORY_FOLDER = "sycon_memories"
MAX_CONTEXT_CHARS = 12000  # Rough limit before we prune (simulating token limit)
DEFAULT_SPEED = 0.05  # Seconds delay per token (lower is faster)
PROMPT_FILE = "sycon_prompt.txt"

class SyconConsciousness:
    def __init__(self, ui_callback_thought, ui_callback_chat):
        self.ui_callback_thought = ui_callback_thought
        self.ui_callback_chat = ui_callback_chat

        # State
        self.running = False
        self.thinking_speed = DEFAULT_SPEED
        self.context_buffer = ""
        self.pending_user_input = []
        self.full_context = [] # New: This will hold the main conversation context for the LLM
        self.session_chat_log = "" # <<< NEW: Dedicated log for memory summarizer

        # Memory Initialization
        if not os.path.exists(MEMORY_FOLDER):
            os.makedirs(MEMORY_FOLDER)

        # Threads
        self.stop_event = threading.Event()

    def get_initial_prompt(self):
        """Loads the prompt template and inserts current time and memories."""
        try:
            with open(PROMPT_FILE, 'r') as f:
                prompt_template = f.read()
        except FileNotFoundError:
            raise FileNotFoundError(f"Error: Prompt file '{PROMPT_FILE}' not found. Please create it.")

        # 1. Load memories
        self.long_term_memory = self.load_memories()

        # 2. Insert variables into the template
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        initial_prompt = prompt_template.format(
            timestamp=timestamp,
            memories=self.long_term_memory
        )

        return initial_prompt

    def start_new_session(self):
        """
        Initializes Sycon's context and state when the user presses START.
        This must be called when 'running' state changes to True.
        """

        # Reset current session data
        self.context_buffer = ""
        self.pending_user_input = []

        # Build the initial context for the LLM
        self.system_prompt = self.get_initial_prompt()

        # The Ollama API expects a list of messages. The first one is the system prompt.
        self.full_context = [{"role": "system", "content": self.system_prompt}]

        # Inject an initial thought to kick off the loop
        initial_kickoff = "\n[SYSTEM KICKOFF: Begin reflection on goals and past existence.]\n"
        self.pending_user_input.append(initial_kickoff)
        self.ui_callback_thought("\n*** SYCON ACTIVATED ***\n", "system")
        self.ui_callback_thought(initial_kickoff, "system")

        self.running = True

    def load_memories(self):
        """Reads JSON files from memory folder and creates a summary string."""
        combined_memory = "PAST MEMORIES:\n"
        files = sorted(os.listdir(MEMORY_FOLDER))
        if not files:
            return "No previous memories."

        # Read only the last 3 days to avoid prompt overflow (naive approach)
        for f in files[-3:]:
            if f.endswith(".json"):
                try:
                    with open(os.path.join(MEMORY_FOLDER, f), 'r') as file:
                        data = json.load(file)
                        combined_memory += f"[{data['timestamp']}] {data['summary']}\n"
                except:
                    pass
        return combined_memory

    def get_session_memory_summary(self):
        """Uses the LLM to generate a structured, key-entity summary for long-term memory."""

        # This prompt explicitly guides the LLM to extract key details in FIRST PERSON.
        full_session_context = (
            "--- SYCON'S INTERNAL MONOLOGUE ---\n"
            f"{self.context_buffer}\n"
            "--- USER INTERACTIONS ---\n"
            f"{self.session_chat_log}"
        )

        prompt = (
            "You are a Memory Consolidation Agent acting as Sycon's inner voice. Your task is to analyze the following session context "
            "and produce a concise summary (max 3 sentences) focusing on specific details, "
            "and any major events or facts discussed (e.g., User's name, job, core goals, or my reflections).\n"
            "**Crucially, write the entire summary in the FIRST PERSON (using 'I' and 'my')**.\n\n"
            f"SESSION CONTEXT:\n---\n{full_session_context}\n---"
        )

        try:
            response = ollama.generate(
                model=MODEL_NAME,
                prompt=prompt,
                options={'temperature': 0.1, 'num_predict': 256}
            )
            return response['response'].strip()
        except Exception as e:
            print(f"Error during memory summarization: {e}")
            return f"[FAILED TO GENERATE DETAILED MEMORY: LLM ERROR. Session context was: {full_session_context[:100]}...]"


    def save_memory(self):
        """Generates a structured summary of the *entire* current context and saves to JSON."""

        # 1. Use LLM to summarize the entire session's context buffer
        print("Generating final session summary...")
        final_summary = self.get_session_memory_summary()

        memory_obj = {
            "timestamp": str(datetime.datetime.now()),
            "type": "Conversation Summary",
            "summary": final_summary # Now using the LLM-generated detailed summary
        }

        filename = f"memory_{int(time.time())}.json"
        with open(os.path.join(MEMORY_FOLDER, filename), 'w') as f:
            json.dump(memory_obj, f, indent=4)
        print(f"Memory saved: {filename}")
        self.session_chat_log = ""


    def get_llm_summary(self, chunk_to_summarize):
        """Uses the LLM to generate a concise 1-2 sentence summary of a text chunk."""

        # This is a focused, non-streaming, quick call to the LLM
        prompt = (
            "You are a summarization utility. Review the following internal monologue chunk "
            "from Sycon and generate a concise, 1-2 sentence summary of the core topics and reflections. "
            "Do not use the -SAY-: tag. Use past tense and keep it objective.\n\n"
            f"CHUNK:\n---\n{chunk_to_summarize}\n---"
        )

        try:
            response = ollama.generate(
                model=MODEL_NAME,
                prompt=prompt,
                options={'temperature': 0.1, 'num_predict': 128} # Use low temperature for accuracy
            )
            return response['response'].strip()
        except Exception as e:
            print(f"Error during summarization: {e}")
            return f"[FAILED TO SUMMARIZE: LLM ERROR. Content truncated: {chunk_to_summarize[:50]}...]"


    def inject_user_input(self, text):
        """Immediately injects user input into the stream."""
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        injection = f"\n[INPUT RECEIVED AT {timestamp}: User said: '{text}']\n"
        self.pending_user_input.append(injection)
        self.ui_callback_thought(injection, "input")

    def prune_context(self):
        """
        Automated process to prune the context buffer before it hits the limit.
        1. Identifies the oldest, least relevant chunk (the start of the buffer).
        2. Generates a concise summary of that chunk using the LLM.
        3. Replaces the full chunk with the summary.
        """
        if len(self.context_buffer) > MAX_CONTEXT_CHARS:

            # 1. Determine the chunk size to cut (e.g., 20% of the current buffer)
            cut_length = int(len(self.context_buffer) * 0.20)

            # 2. Extract the oldest chunk
            chunk_to_summarize = self.context_buffer[:cut_length]

            # 3. Generate summary
            summary = self.get_llm_summary(chunk_to_summarize)

            # Format the summary for permanent context
            pruned_note = f"\n[INTERNAL ARCHIVE NOTE (Pruning System): Older thoughts summarized: \"{summary}\"]\n"

            # 4. Replace the old chunk with the short summary in the context buffer
            self.context_buffer = pruned_note + self.context_buffer[cut_length:]

            # Update the UI to show the pruning event
            self.ui_callback_thought(
                f"\n[SYSTEM NOTICE: Context Pruning Occurred. {cut_length} characters removed, summarized to 1 memory note.]\n",
                "system"
            )

    def time_keeper_loop(self):
        """Injects time every minute."""
        while not self.stop_event.is_set():
            time.sleep(60)
            if self.running:
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                injection = f"\n[SYSTEM NOTICE: Current Time is {timestamp} Remember to use double quotes for talking to the user]\n"
                self.pending_user_input.append(injection) # Treat as input to interrupt flow
                self.ui_callback_thought(injection, "system")

    def consciousness_loop(self):
        """The main thinking loop, using streaming and quote-based detection."""

        while not self.stop_event.is_set():
            if not self.running:
                time.sleep(0.1)
                continue

            # --- PHASE 1: Process Inputs ---
            while self.pending_user_input:
                inp = self.pending_user_input.pop(0)
                self.full_context.append({"role": "user", "content": f"React to this: {inp}"})
                self.session_chat_log += f"\n[User said]: {inp}"
                self.context_buffer += inp

            # --- PHASE 2: Prune & Prepare ---
            self.prune_context()

            # 3. Generate Stream (Continuous Thinking)
            prompt_trigger = "Continue your stream of consciousness. Reflect, observe, or decide to speak (using quotes)."

            # **FIX**: Append the temporary trigger for the API call
            # **FIX**: Add the explicit rule reminder to the temporary prompt trigger
            prompt_trigger = (
                "Continue your stream of consciousness. Reflect, observe, or decide to speak. "
                "**CRITICAL REMINDER: If you speak to the user, you MUST use double quotes for the entire message (e.g., \"Hello, User.\")**"
            )
            # Append the temporary trigger for the API call
            self.full_context.append({"role": "user", "content": prompt_trigger})

            try:
                stream = ollama.chat(
                    model=MODEL_NAME,
                    messages=self.full_context,
                    stream=True,
                    options={
                    'temperature': 0.7,
                    # *** FIX: Add anti-repetition penalties ***
                    'repeat_penalty': 1.15,
                    'frequency_penalty': 0.05
                    }
                )

                current_thought_chunk = ""
                capture_say_message = False
                say_message_buffer = ""

                for chunk in stream:
                    if self.stop_event.is_set() or not self.running:
                        break

                    word = chunk['message']['content']
                    current_thought_chunk += word

                    # 4. Real-time Parsing for Quotes (External Message)

                    if not capture_say_message and '"' in current_thought_chunk:

                        # --- START CAPTURE ---
                        parts = current_thought_chunk.split('"', 1)
                        pure_thought = parts[0]
                        remaining_text = parts[1]

                        self.ui_callback_thought(pure_thought, "thought")

                        capture_say_message = True
                        say_message_buffer += remaining_text
                        current_thought_chunk = ""

                    elif capture_say_message:
                        # --- ACTIVE CAPTURE ---
                        say_message_buffer += word

                        # Check for the closing quote
                        if say_message_buffer.strip().endswith('"'):

                            # Clean up and deliver the message (robustly removing quotes and whitespace)
                            final_msg = say_message_buffer.strip()
                            if final_msg.endswith('"'):
                                # Remove closing quote
                                final_msg = final_msg[:-1].strip()

                            self.ui_callback_chat(final_msg, "Sycon")

                            # Add Sycon's speech to history
                            self.full_context.append({"role": "assistant", "content": f"I said to the User: {final_msg}"})

                            # Reset state trackers
                            capture_say_message = False
                            say_message_buffer = ""


                    elif not capture_say_message and word.strip():
                        # --- STANDARD THOUGHT FLOW ---
                        self.ui_callback_thought(word, "thought")

                        if self.thinking_speed > 0:
                            time.sleep(self.thinking_speed)

                # --- Stream END Handling ---

                # **FIX**: Pop the temporary trigger message we added before the API call
                self.full_context.pop()

                # 5. Safety Net for unfinished messages
                if capture_say_message and say_message_buffer.strip():
                    final_msg = say_message_buffer.strip().strip('"')
                    if final_msg:
                        self.ui_callback_chat(f"[Incomplete]: {final_msg}", "Sycon")
                        self.full_context.append({"role": "assistant", "content": f"I said to the User (incomplete): {final_msg}"})

                # Append the full internal thought to context logic
                self.context_buffer += current_thought_chunk

                # Add the LLM's actual internal monologue as its official response
                self.full_context.append({"role": "assistant", "content": current_thought_chunk})

                self.ui_callback_thought("\n", "thought")


            except Exception as e:
                err = f"\n[CRITICAL ERROR: {str(e)} - Check Ollama connectivity: {e}]\n"
                self.ui_callback_thought(err, "system")
                try:
                    self.full_context.pop() # Attempt to clean up the temporary trigger on error
                except IndexError:
                    pass
                time.sleep(5)
                continue

            # --- PHASE 3: Continuous Thinking Loop Delay ---
            pass # No deliberate delay, forcing continuous stream

# --- UI CLASS ---
class SyconUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Sycon - Consciousness Simulator")
        self.root.geometry("1200x800")
        self.root.configure(bg="#1e1e1e")

        # Style
        style = ttk.Style()
        style.theme_use('clam')

        # --- Layout ---
        # Top: Controls
        # Left: Chat (User Interaction)
        # Right: Stream of Consciousness (Internal Monologue)

        # Control Frame
        ctrl_frame = tk.Frame(root, bg="#2d2d2d", height=50)
        ctrl_frame.pack(side="top", fill="x", padx=5, pady=5)

        self.btn_start = tk.Button(ctrl_frame, text="START Sycon", command=self.toggle_sycon, bg="#4CAF50", fg="white")
        self.btn_start.pack(side="left", padx=10)

        tk.Label(ctrl_frame, text="Thinking Delay (s):", bg="#2d2d2d", fg="white").pack(side="left", padx=5)
        self.speed_scale = tk.Scale(ctrl_frame, from_=0.0, to=1.0, resolution=0.01, orient="horizontal", bg="#2d2d2d", fg="white", command=self.update_speed)
        self.speed_scale.set(DEFAULT_SPEED)
        self.speed_scale.pack(side="left")

        # Main Split Frame
        split_frame = tk.PanedWindow(root, orient="horizontal", bg="#1e1e1e", sashwidth=5)
        split_frame.pack(fill="both", expand=True, padx=5, pady=5)

        # LEFT: Chat Interface
        chat_frame = tk.Frame(split_frame, bg="#1e1e1e")
        split_frame.add(chat_frame, minsize=400)

        tk.Label(chat_frame, text="Interaction Channel", bg="#1e1e1e", fg="#aaaaaa").pack(anchor="w")
        self.chat_display = scrolledtext.ScrolledText(chat_frame, bg="#252526", fg="white", font=("Consolas", 11))
        self.chat_display.pack(fill="both", expand=True)

        input_frame = tk.Frame(chat_frame, bg="#1e1e1e")
        input_frame.pack(fill="x", pady=5)
        self.user_input = tk.Entry(input_frame, bg="#3c3c3c", fg="white", font=("Consolas", 11))
        self.user_input.pack(side="left", fill="x", expand=True)
        self.user_input.bind("<Return>", self.send_message)
        btn_send = tk.Button(input_frame, text="Send", command=self.send_message, bg="#007acc", fg="white")
        btn_send.pack(side="right", padx=5)

        # RIGHT: Stream of Consciousness
        soc_frame = tk.Frame(split_frame, bg="#1e1e1e")
        split_frame.add(soc_frame, minsize=400)

        tk.Label(soc_frame, text="Stream of Consciousness (Internal)", bg="#1e1e1e", fg="#aaaaaa").pack(anchor="w")
        self.soc_display = scrolledtext.ScrolledText(soc_frame, bg="#000000", fg="#00ff00", font=("Courier New", 10))
        self.soc_display.pack(fill="both", expand=True)

        # --- LOGIC CONNECTION ---
        self.sycon = SyconConsciousness(self.update_soc_display, self.update_chat_display)

        # Start background threads
        self.t_loop = threading.Thread(target=self.sycon.consciousness_loop, daemon=True)
        self.t_time = threading.Thread(target=self.sycon.time_keeper_loop, daemon=True)
        self.t_loop.start()
        self.t_time.start()

        # Handle Close
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def toggle_sycon(self):
        if self.sycon.running:
            # PAUSE logic
            self.sycon.running = False
            self.btn_start.config(text="RESUME Sycon (Paused)", bg="#f44336")
        else:
            # START/RESUME logic
            if not self.sycon.full_context:
                # If no context exists, this is a clean start
                self.sycon.start_new_session()
            else:
                # Otherwise, it's a simple resume
                self.sycon.running = True

            self.btn_start.config(text="PAUSE Sycon", bg="#4CAF50")

    def update_speed(self, val):
        self.sycon.thinking_speed = float(val)

    def send_message(self, event=None):
        msg = self.user_input.get()
        if msg.strip():
            self.update_chat_display(msg, "User")
            self.sycon.inject_user_input(msg)
            self.user_input.delete(0, tk.END)

    def update_soc_display(self, text, type_):
        # Tkinter is not thread safe, must use after() logic or simple insertion if simple enough.
        # Ideally, use a queue. For this example, we access directly (risk of minor race condition but usually fine for text append)
        def _update():
            self.soc_display.insert(tk.END, text)
            self.soc_display.see(tk.END)
        self.root.after(0, _update)

    def update_chat_display(self, text, sender):
        def _update():
            color = "#00aaff" if sender == "Sycon" else "#ffaa00"
            self.chat_display.tag_config(sender, foreground=color)
            self.chat_display.insert(tk.END, f"{sender}: ", sender)
            self.chat_display.insert(tk.END, f"{text}\n\n")
            self.chat_display.see(tk.END)
        self.root.after(0, _update)

    def on_close(self):
        self.sycon.stop_event.set()
        self.sycon.save_memory() # Save on exit
        self.root.destroy()

# --- MAIN ---
if __name__ == "__main__":
    root = tk.Tk()
    app = SyconUI(root)
    root.mainloop()