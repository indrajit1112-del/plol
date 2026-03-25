import os
import numpy as np
from flask import Flask, request, jsonify, render_template_string
from openai import OpenAI
import core_engine

app = Flask(__name__)

# Initialize OpenAI client (requires OPENAI_API_KEY environment variable)
# Default to empty string for safety if not set to avoid immediate crash
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))

# Load and enrich data (prefers cleaned JSON, falls back to xlsx+regex)
df = core_engine.load_cleaned_inventory()

# Pre-compute embeddings for semantic search fallback
_hardware_names = df['Name'].tolist() if not df.empty else []
_embeddings = core_engine.generate_embeddings(client, _hardware_names) if _hardware_names else np.array([])

def build_response(user_message, recent_context=""):
    """Process user message and return a structured HTML/Markdown chatbot response."""
    msg = user_message.strip().lower()

    # Greetings
    if msg in ("hi", "hello", "hey", "yo", "sup"):
        return "Hey! I'm the **Price Bot**. Ask me about any component price — CPUs, GPUs, RAM, SSDs, etc."

    # Help
    if msg in ("help", "?", "commands"):
        return (
            "Just type a component name or specific constraints and I'll find its price!\n\n"
            "**Examples:**\n"
            "• `list of amd gpus in stock`\n"
            "• `all non rgb rams with 5200 mhz`\n"
            "• `all 6000mhz ram or 5200 mhz ram`\n"
            "• `specific any gpu name 5090 tuf`\n"
            "• `ram under 10k`\n"
            "• `list all` — show everything"
        )

    # List all
    if msg in ("list all", "show all", "all", "list", "all prices", "show everything"):
        lines = [f"• **{row['Name']}** ({row['Category']}) — {core_engine.format_price(row['price'])}" for _, row in df.iterrows()]
        return "Here are all components in inventory:\n\n" + "\n".join(lines)

    # Intelligent Search with dynamic fallback
    try:
        intent_obj = core_engine.extract_search_intent(client, user_message, recent_context)

        # Tier 1: strict filters with progressive relaxation
        filtered_df, was_relaxed, relaxed_fields = core_engine.execute_with_fallback(df, intent_obj)

        # Tier 2: semantic search if relaxation still returned nothing
        if filtered_df.empty and len(_hardware_names) > 0:
            filtered_df = core_engine.semantic_search(client, user_message, df, _embeddings, top_k=10)
            if not filtered_df.empty:
                records = filtered_df.to_dict('records')
                lines = [f"🔍 No exact filter match. Here are the **top {len(records)} semantically similar** items:\n"]
                for m in records:
                    lines.append(f"• **{m['Name']}** — {core_engine.format_price(m['price'])}")
                return "\n".join(lines)

        # Tier 3: LLM apology
        if filtered_df.empty:
            fallback_prompt = f"The user asked: '{user_message}'. Parsed as: {intent_obj.model_dump_json(exclude_none=True)}. No inventory matched. Formulate conversational apology and proactively suggest dropping constraints."
            fallback_res = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "system", "content": "You are a helpful hardware assistant."}, {"role": "user", "content": fallback_prompt}],
                temperature=0.5,
                max_tokens=150
            )
            return fallback_res.choices[0].message.content.strip()

        records = filtered_df.to_dict('records')
        
        if len(records) == 1:
            m = records[0]
            return f"**{m['Name']}** ({m.get('Category', '')}) — {core_engine.format_price(m['price'])}"

        # Format multiple results
        if was_relaxed:
            relaxed_str = ", ".join(relaxed_fields)
            header = f"⚡ No exact match — relaxed {relaxed_str} and found {len(records)} items:\n"
        else:
            header = f"Found {len(records)} matches for your criteria:\n"

        lines = [header]
        for m in records:
            lines.append(f"• **{m['Name']}** — {core_engine.format_price(m['price'])}")
            
        return "\n".join(lines)

    except Exception as e:
        # Fallback if OpenAI fails or key is missing
        return f"Error connecting to intelligence engine. Ensure OPENAI_API_KEY is set. Technical details: {str(e)}"

# A simple in-memory session store (dictionary mapping IP -> context string) for simplicity
sessions = {}

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Price Bot</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }

  body {
    font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
    background: #0f0f0f;
    color: #e0e0e0;
    height: 100vh;
    display: flex;
    flex-direction: column;
  }

  .header {
    background: #1a1a2e;
    padding: 16px 24px;
    display: flex;
    align-items: center;
    gap: 12px;
    border-bottom: 1px solid #2a2a4a;
  }

  .header .dot {
    width: 10px; height: 10px;
    background: #00d26a;
    border-radius: 50%;
    animation: pulse 2s infinite;
  }

  @keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.4; }
  }

  .header h1 {
    font-size: 18px;
    font-weight: 600;
    color: #fff;
  }

  .header .subtitle {
    font-size: 12px;
    color: #888;
    margin-left: auto;
  }

  .chat-container {
    flex: 1;
    overflow-y: auto;
    padding: 20px 24px;
    display: flex;
    flex-direction: column;
    gap: 16px;
  }

  .message {
    max-width: 75%;
    padding: 12px 16px;
    border-radius: 16px;
    font-size: 14px;
    line-height: 1.6;
    word-wrap: break-word;
    animation: fadeIn 0.3s ease;
  }

  @keyframes fadeIn {
    from { opacity: 0; transform: translateY(8px); }
    to { opacity: 1; transform: translateY(0); }
  }

  .message.bot {
    background: #1e1e3a;
    border: 1px solid #2a2a4a;
    align-self: flex-start;
    border-bottom-left-radius: 4px;
  }

  .message.user {
    background: #4a3aff;
    color: #fff;
    align-self: flex-end;
    border-bottom-right-radius: 4px;
  }

  .message strong { color: #7c8aff; }
  .message.user strong { color: #c0c8ff; }

  .input-area {
    padding: 16px 24px;
    background: #1a1a2e;
    border-top: 1px solid #2a2a4a;
    display: flex;
    gap: 10px;
  }

  .input-area input {
    flex: 1;
    padding: 12px 16px;
    background: #0f0f1f;
    border: 1px solid #2a2a4a;
    border-radius: 12px;
    color: #e0e0e0;
    font-size: 14px;
    outline: none;
    transition: border-color 0.2s;
  }

  .input-area input:focus {
    border-color: #4a3aff;
  }

  .input-area input::placeholder {
    color: #555;
  }

  .input-area button {
    padding: 12px 20px;
    background: #4a3aff;
    color: #fff;
    border: none;
    border-radius: 12px;
    font-size: 14px;
    font-weight: 600;
    cursor: pointer;
    transition: background 0.2s;
  }

  .input-area button:hover { background: #3a2ae0; }
  .input-area button:disabled { opacity: 0.5; cursor: default; }

  .chat-container::-webkit-scrollbar { width: 6px; }
  .chat-container::-webkit-scrollbar-track { background: transparent; }
  .chat-container::-webkit-scrollbar-thumb { background: #333; border-radius: 3px; }
</style>
</head>
<body>

<div class="header">
  <div class="dot"></div>
  <h1>Price Bot</h1>
  <span class="subtitle">AI Hardware Search Engine</span>
</div>

<div class="chat-container" id="chat">
  <div class="message bot">Hey! I'm the <strong>Price Bot</strong>. Ask me about any component price — CPUs, GPUs, RAM, SSDs, etc.<br><br>Type <strong>help</strong> for examples.</div>
</div>

<div class="input-area">
  <input type="text" id="input" placeholder="Ask about a component price..." autocomplete="off" autofocus>
  <button id="sendBtn" onclick="send()">Send</button>
</div>

<script>
const chat = document.getElementById('chat');
const input = document.getElementById('input');
const sendBtn = document.getElementById('sendBtn');

input.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !sendBtn.disabled) send();
});

function formatMarkdown(text) {
  // Bold
  text = text.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
  // Inline code
  text = text.replace(/`(.+?)`/g, '<code style="background:#2a2a4a;padding:2px 6px;border-radius:4px;font-size:13px;">$1</code>');
  // Line breaks
  text = text.replace(/\n/g, '<br>');
  return text;
}

function addMessage(text, sender) {
  const div = document.createElement('div');
  div.className = 'message ' + sender;
  div.innerHTML = formatMarkdown(text);
  chat.appendChild(div);
  chat.scrollTop = chat.scrollHeight;
}

async function send() {
  const msg = input.value.trim();
  if (!msg) return;

  addMessage(msg, 'user');
  input.value = '';
  sendBtn.disabled = true;

  try {
    const res = await fetch('/chat', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({message: msg})
    });
    const data = await res.json();
    addMessage(data.reply, 'bot');
  } catch {
    addMessage('Something went wrong. Please try again.', 'bot');
  }

  sendBtn.disabled = false;
  input.focus();
}
</script>
</body>
</html>"""

@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    if not data or "message" not in data:
        return jsonify({"reply": "Please send a message."}), 400
    user_message = str(data["message"]).strip()[:200]
    
    ip = request.remote_addr
    recent_context = sessions.get(ip, "")
    
    reply = build_response(user_message, recent_context)
    
    # Store minimal intent for next turn if needed (simplified for flask, actual intents handled in core engine)
    sessions[ip] = f"{user_message} -> {reply[:50]}..."
    
    return jsonify({"reply": reply})

if __name__ == "__main__":
    print("Price Bot Intelligent Engine running at http://localhost:6699")
    app.run(host="127.0.0.1", port=6699, debug=False)
