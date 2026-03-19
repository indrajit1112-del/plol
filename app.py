from flask import Flask, request, jsonify, render_template_string
import openpyxl
from thefuzz import fuzz, process
import os

app = Flask(__name__)

def load_prices():
    """Load component prices from the Excel file."""
    xlsx_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prices.xlsx")
    wb = openpyxl.load_workbook(xlsx_path, read_only=True)
    ws = wb.active
    items = []
    for row in ws.iter_rows(min_row=2, values_only=True):
        name, price = row[0], row[1]
        if name and price is not None:
            items.append({"name": str(name).strip(), "price": int(price)})
    wb.close()
    return items

PRICES = load_prices()
COMPONENT_NAMES = [item["name"] for item in PRICES]

def format_price(price):
    """Format price in Indian numbering system with rupee symbol."""
    s = str(price)
    if len(s) <= 3:
        return f"\u20b9{s}"
    last3 = s[-3:]
    remaining = s[:-3]
    groups = []
    while remaining:
        groups.append(remaining[-2:])
        remaining = remaining[:-2]
    groups.reverse()
    return f"\u20b9{','.join(groups)},{last3}"

def parse_exclusions(query):
    """Split query into search terms and exclusion keywords (prefixed with -)."""
    parts = query.strip().split()
    search_parts = []
    excludes = []
    for part in parts:
        if part.startswith("-") and len(part) > 1:
            excludes.append(part[1:].lower())
        else:
            search_parts.append(part)
    return " ".join(search_parts), excludes

def apply_exclusions(items, excludes):
    """Filter out items whose name contains any exclusion keyword."""
    if not excludes:
        return items
    return [item for item in items if not any(ex in item["name"].lower() for ex in excludes)]

def search_components(query, limit=5):
    """Fuzzy search components matching the query, supporting -keyword exclusions."""
    search_query, excludes = parse_exclusions(query)
    if not search_query:
        return []

    # Try exact substring match first
    exact_matches = [item for item in PRICES if search_query.lower() in item["name"].lower()]
    if exact_matches:
        return apply_exclusions(exact_matches, excludes)[:limit]

    # Fall back to fuzzy matching
    results = process.extract(search_query, COMPONENT_NAMES, scorer=fuzz.token_set_ratio, limit=limit * 3)
    matched = []
    for result in results:
        name, score = result[0], result[1]
        if score >= 45:
            item = next(i for i in PRICES if i["name"] == name)
            matched.append(item)
    return apply_exclusions(matched, excludes)[:limit]

def build_response(user_message):
    """Process user message and return a chatbot response."""
    msg = user_message.strip().lower()

    # Greetings
    if msg in ("hi", "hello", "hey", "yo", "sup"):
        return "Hey! I'm the Price Bot. Ask me about any component price — CPUs, GPUs, RAM, SSDs, etc."

    # Help
    if msg in ("help", "?", "commands"):
        return (
            "Just type a component name and I'll find its price!\n\n"
            "**Examples:**\n"
            "• `5080`\n"
            "• `DDR5 6000`\n"
            "• `9070XT`\n"
            "• `1TB Gen4`\n"
            "• `DDR5 -rgb` — exclude RGB variants\n"
            "• `list all` — show everything"
        )

    # List all
    if msg in ("list all", "show all", "all", "list", "all prices", "show everything"):
        lines = [f"• **{item['name']}** — {format_price(item['price'])}" for item in PRICES]
        return "Here are all components:\n\n" + "\n".join(lines)

    # Search
    matches = search_components(user_message)
    if not matches:
        return f"Sorry, I couldn't find anything matching **\"{user_message}\"**. Try a different name or type `help` for examples."

    if len(matches) == 1:
        m = matches[0]
        return f"**{m['name']}** — {format_price(m['price'])}"

    lines = [f"• **{m['name']}** — {format_price(m['price'])}" for m in matches]
    return f"Found {len(matches)} matches:\n\n" + "\n".join(lines)


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
  <span class="subtitle">Component prices &bull; Fuzzy search</span>
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
    reply = build_response(user_message)
    return jsonify({"reply": reply})


if __name__ == "__main__":
    print("Price Bot running at http://localhost:6699")
    app.run(host="127.0.0.1", port=6699, debug=False)
