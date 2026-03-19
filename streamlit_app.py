import streamlit as st
import openpyxl
from thefuzz import fuzz, process
import os

st.set_page_config(page_title="Price Bot", page_icon="💰", layout="centered")

@st.cache_data
def load_prices():
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


def search_components(query, limit=5):
    search_query, excludes = parse_exclusions(query)
    if not search_query:
        return []

    exact_matches = [item for item in PRICES if search_query.lower() in item["name"].lower()]
    if exact_matches:
        return apply_exclusions(exact_matches, excludes)[:limit]

    results = process.extract(search_query, COMPONENT_NAMES, scorer=fuzz.token_set_ratio, limit=limit * 3)
    matched = []
    for result in results:
        name, score = result[0], result[1]
        if score >= 45:
            item = next(i for i in PRICES if i["name"] == name)
            matched.append(item)
    return apply_exclusions(matched, excludes)[:limit]


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
        lines = [f"- **{item['name']}** — {format_price(item['price'])}" for item in PRICES]
        return "Here are all components:\n\n" + "\n".join(lines)

    matches = search_components(user_message)
    if not matches:
        return f'Sorry, I couldn\'t find anything matching **"{user_message}"**. Try a different name or type `help` for examples.'

    if len(matches) == 1:
        m = matches[0]
        return f"**{m['name']}** — {format_price(m['price'])}"

    lines = [f"- **{m['name']}** — {format_price(m['price'])}" for m in matches]
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
