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


def category_has_tag(category, tag):
    """Check if a category contains the given tag (categories use 'or' as delimiter)."""
    tags = [t.strip().lower() for t in category.split(" or ")]
    return tag.strip().lower() in tags


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


def search_components(query, category_filter=None):
    search_query, excludes = parse_exclusions(query)
    if not search_query:
        return []

    pool = PRICES
    if category_filter:
        pool = [item for item in PRICES if category_has_tag(item["category"], category_filter)]

    exact_matches = [item for item in pool if search_query.lower() in item["name"].lower()]
    if exact_matches:
        return apply_exclusions(exact_matches, excludes)

    pool_names = [item["name"] for item in pool]
    results = process.extract(search_query, pool_names, scorer=fuzz.token_set_ratio, limit=len(pool_names))
    matched = []
    for result in results:
        name, score = result[0], result[1]
        if score >= 45:
            item = next(i for i in pool if i["name"] == name)
            matched.append(item)
    return apply_exclusions(matched, excludes)


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
                    '{"clear": true/false, "search_term": "extracted search keyword", "category": "matching tag or null", "follow_up": "question to ask if unclear"}\n\n'
                    "Rules:\n"
                    "- Categories use 'or' as delimiter for tags. E.g. 'GPU or Nvidia' has tags 'GPU' and 'Nvidia'.\n"
                    "- If the user says 'gpu', set category to 'GPU' (matches 'GPU or Nvidia', 'GPU or AMD', etc.).\n"
                    "- If the user says 'nvidia', set category to 'Nvidia' (matches only 'GPU or Nvidia').\n"
                    "- If the user mentions a specific model/part (e.g. '5080', 'DDR5', 'RTX 4070'), set clear=true and extract a search_term.\n"
                    "- If the user is vague (e.g. 'I need something fast', 'good GPU'), set clear=false and write a follow_up question.\n"
                    "- If the user asks about a category (e.g. 'show me GPUs', 'RAM options', 'SSDs'), set clear=true with the category as search_term and set category to the matching tag.\n"
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


def format_grouped_results(matches):
    """Format results grouped by category. Uses ChatGPT to label groups of items
    that don't fit neatly into existing categories."""
    if len(matches) == 1:
        m = matches[0]
        return f"**{m['name']}** ({m['category']}) — {format_price(m['price'])}"

    # Group by category
    from collections import OrderedDict
    groups = OrderedDict()
    for m in matches:
        cat = m["category"]
        if cat not in groups:
            groups[cat] = []
        groups[cat].append(m)

    # If only one category, simple list
    if len(groups) == 1:
        cat = list(groups.keys())[0]
        lines = [f"- **{m['name']}** — {format_price(m['price'])}" for m in matches]
        return f"**{cat}** ({len(matches)} items):\n\n" + "\n".join(lines)

    # Multiple categories — use ChatGPT to generate a group label for mixed results
    cat_summary = ", ".join(f"{cat} ({len(items)})" for cat, items in groups.items())
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "Given categories of PC components found in a search, provide a short 2-4 word group label. Respond with ONLY the label text, nothing else."
                },
                {"role": "user", "content": f"Categories found: {cat_summary}"}
            ],
            temperature=0.3,
            max_tokens=20
        )
        group_label = response.choices[0].message.content.strip()
    except Exception:
        group_label = "Search Results"

    lines = [f"**{group_label}** — {len(matches)} items found:\n"]
    for cat, items in groups.items():
        lines.append(f"\n**{cat}:**")
        for m in items:
            lines.append(f"- {m['name']} — {format_price(m['price'])}")

    return "\n".join(lines)


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
        cat_items = [item for item in PRICES if category_has_tag(item["category"], category_filter)]
        if cat_items and search_term.lower().replace(" ", "") in category_filter.lower().replace(" ", ""):
            return format_grouped_results(cat_items)

    matches = search_components(search_term, category_filter=category_filter)
    if not matches:
        # Fallback: try original message directly without category filter
        matches = search_components(user_message)
    if not matches:
        return f'Sorry, I couldn\'t find anything matching **"{user_message}"**. Try a different name or type `help` for examples.'

    return format_grouped_results(matches)


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
