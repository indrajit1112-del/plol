import streamlit as st
import openpyxl
from thefuzz import fuzz, process
from openai import OpenAI
import os
import json

st.set_page_config(page_title="Price Bot", page_icon="💰", layout="centered")

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

@st.cache_data
def load_prices():
    xlsx_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prices.xlsx")
    wb = openpyxl.load_workbook(xlsx_path, read_only=True)
    ws = wb.active
    items = []
    for row in ws.iter_rows(min_row=2, values_only=True):
        category, name, price = row[0], row[1], row[2]
        if name and price is not None:
            items.append({"category": str(category).strip() if category else "Other", "name": str(name).strip(), "price": int(price)})
    wb.close()
    return items

PRICES = load_prices()
COMPONENT_NAMES = [item["name"] for item in PRICES]
CATEGORIES = sorted(set(item["category"] for item in PRICES))


def format_price(price):
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
    if not excludes:
        return items
    return [item for item in items if not any(ex in item["name"].lower() for ex in excludes)]


def search_components(query, limit=15, category_filter=None):
    search_query, excludes = parse_exclusions(query)
    if not search_query:
        return []

    pool = PRICES
    if category_filter:
        pool = [item for item in PRICES if item["category"].lower() == category_filter.lower()]

    exact_matches = [item for item in pool if search_query.lower() in item["name"].lower()]
    if exact_matches:
        return apply_exclusions(exact_matches, excludes)[:limit]

    pool_names = [item["name"] for item in pool]
    results = process.extract(search_query, pool_names, scorer=fuzz.token_set_ratio, limit=limit * 3)
    matched = []
    for result in results:
        name, score = result[0], result[1]
        if score >= 45:
            item = next(i for i in pool if i["name"] == name)
            matched.append(item)
    return apply_exclusions(matched, excludes)[:limit]


def detect_intent(user_message, recent_context=""):
    """Use ChatGPT to detect what component the user is looking for.
    Returns a dict with 'clear' (bool), 'search_term' (str), 'category' (str or null), and 'follow_up' (str)."""
    component_list = ", ".join(COMPONENT_NAMES[:50])
    category_list = ", ".join(CATEGORIES)

    context_note = ""
    if recent_context:
        context_note = f"\n\nRecent conversation for context:\n{recent_context}\nUse this context to understand follow-up replies from the user."

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a component price assistant. The user wants to look up PC component prices. "
                    "Available components include: " + component_list + "... and more.\n"
                    "Available categories: " + category_list + "\n\n"
                    "Your job is to figure out what specific component or category the user wants. "
                    "Respond ONLY with a JSON object (no markdown, no code fences):\n"
                    '{"clear": true/false, "search_term": "extracted search keyword", "category": "matching category or null", "follow_up": "question to ask if unclear"}\n\n'
                    "Rules:\n"
                    "- If the user mentions a specific model/part (e.g. '5080', 'DDR5', 'RTX 4070'), set clear=true and extract a search_term.\n"
                    "- If the user is vague (e.g. 'I need something fast', 'good GPU'), set clear=false and write a follow_up question.\n"
                    "- If the user asks about a category (e.g. 'show me GPUs', 'RAM options', 'SSDs'), set clear=true with the category as search_term and set category to the matching category name.\n"
                    "- If the query matches a category, set category to the EXACT category name from the list above.\n"
                    "- If the user is replying to a follow-up question, combine the context from the conversation to form a complete search_term.\n"
                    "- Keep search_term short — just the key words for searching.\n"
                    "- Set category to null if no specific category is detected."
                    + context_note
                )
            },
            {"role": "user", "content": user_message}
        ],
        temperature=0.3,
        max_tokens=150
    )
    text = response.choices[0].message.content.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"clear": True, "search_term": user_message, "category": None, "follow_up": ""}


def build_response(user_message):
    msg = user_message.strip().lower()

    if msg in ("hi", "hello", "hey", "yo", "sup"):
        return "Hey! I'm the **Price Bot**. Ask me about any component price — CPUs, GPUs, RAM, SSDs, etc."

    if msg in ("help", "?", "commands"):
        return (
            "Just type a component name and I'll find its price!\n\n"
            "**Examples:**\n"
            "- `5080`\n"
            "- `DDR5 6000`\n"
            "- `9070XT`\n"
            "- `1TB Gen4`\n"
            "- `DDR5 -rgb` — exclude RGB variants\n"
            "- `list all` — show everything"
        )

    if msg in ("list all", "show all", "all", "list", "all prices", "show everything"):
        lines = []
        for cat in CATEGORIES:
            cat_items = [item for item in PRICES if item["category"] == cat]
            if cat_items:
                lines.append(f"\n**{cat}:**")
                for item in cat_items:
                    lines.append(f"- {item['name']} — {format_price(item['price'])}")
        return "Here are all components:\n" + "\n".join(lines)

    # Use ChatGPT to detect intent, include recent conversation for context
    recent_context = ""
    if len(st.session_state.messages) >= 3:
        last_msgs = st.session_state.messages[-3:]
        recent_context = " | ".join(f"{m['role']}: {m['content'][:100]}" for m in last_msgs)

    intent = detect_intent(user_message, recent_context)

    if not intent.get("clear", True):
        return intent.get("follow_up", "Could you be more specific about which component you're looking for?")

    search_term = intent.get("search_term", user_message)
    category_filter = intent.get("category", None)

    # If the query is essentially a category name, list all items in that category
    if category_filter:
        cat_items = [item for item in PRICES if item["category"].lower() == category_filter.lower()]
        if cat_items and search_term.lower().replace(" ", "") in category_filter.lower().replace(" ", ""):
            lines = [f"- **{m['name']}** — {format_price(m['price'])}" for m in cat_items[:15]]
            return f"**{category_filter}** ({len(cat_items)} items):\n\n" + "\n".join(lines)

    matches = search_components(search_term, category_filter=category_filter)
    if not matches:
        # Fallback: try original message directly without category filter
        matches = search_components(user_message)
    if not matches:
        return f'Sorry, I couldn\'t find anything matching **"{user_message}"**. Try a different name or type `help` for examples.'

    if len(matches) == 1:
        m = matches[0]
        return f"**{m['name']}** ({m['category']}) — {format_price(m['price'])}"

    lines = [f"- **{m['name']}** ({m['category']}) — {format_price(m['price'])}" for m in matches]
    return f"Found {len(matches)} matches:\n\n" + "\n".join(lines)


# --- Streamlit UI ---

st.markdown("""
<style>
    .stChatMessage { border-radius: 12px; }
    div[data-testid="stChatMessageContent"] p { font-size: 15px; }
</style>
""", unsafe_allow_html=True)

st.title("💰 Price Bot")
st.caption("Component prices • Fuzzy search • Use `-keyword` to exclude")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hey! I'm the **Price Bot**. Ask me about any component price — CPUs, GPUs, RAM, SSDs, etc.\n\nType **help** for examples."}
    ]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask about a component price..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    reply = build_response(prompt)
    st.session_state.messages.append({"role": "assistant", "content": reply})
    with st.chat_message("assistant"):
        st.markdown(reply)
