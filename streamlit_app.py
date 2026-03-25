import streamlit as st
from openai import OpenAI
import core_engine

st.set_page_config(page_title="Price Bot", page_icon="💰", layout="centered")

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

@st.cache_data
def get_data():
    return core_engine.load_and_enrich_data()

df = get_data()

def process_query(prompt):
    recent_context = st.session_state.get("last_intent", "")
    
    # 1. Parse Intent via LLM Structured Outputs
    intent_obj = core_engine.extract_search_intent(client, prompt, recent_context)
    
    # Store for next conversational turn
    st.session_state.last_intent = intent_obj.model_dump_json(exclude_none=True)
    
    # 2. Filter Multidimensional Pandas DataFrame
    filtered_df = core_engine.execute_pandas_filters(df, intent_obj)
    
    if filtered_df.empty:
        # Fallback secondary prompt (hidden from user)
        fallback_prompt = f"The user asked: '{prompt}'. We parsed it as: {intent_obj.model_dump_json(exclude_none=True)}. No inventory matched these exact criteria. Formulate a concise conversational apology, and proactively suggest dropping one of the specific constraints (like speed, exact latency, or trying a different brand) to find related items."
        fallback_res = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": "You are a helpful hardware assistant."}, {"role": "user", "content": fallback_prompt}],
            temperature=0.5,
            max_tokens=150
        )
        return {"type": "text", "content": fallback_res.choices[0].message.content.strip()}

    records = filtered_df.to_dict('records')
    
    # If a single item is requested/found
    if len(records) == 1:
        m = records[0]
        category = m.get('Category', '')
        msg = f"**{m['Name']}** ({category}) — {core_engine.format_price(m['price'])}"
        return {"type": "text", "content": msg}
    
    # Multiple items returned - render as interactive dataframe
    cols_to_show = ['Category', 'Name', 'price']
    for c in ['Brand', 'Speed', 'Latency', 'RGB', 'PCIe_Gen']:
        if c in filtered_df.columns and not filtered_df[c].isna().all():
            cols_to_show.append(c)
            
    display_df = filtered_df[cols_to_show].copy()
    display_df['Price (INR)'] = display_df['price'].apply(core_engine.format_price)
    # Drop original price integer column and rearrange
    display_df = display_df.drop(columns=['price'])
    # move Price (INR) to 3rd column
    cols = display_df.columns.tolist()
    cols.insert(2, cols.pop(cols.index('Price (INR)')))
    display_df = display_df[cols]
    
    preview_text = f"Found {len(records)} items matching your criteria:"
    return {"type": "dataframe", "content": display_df, "text": preview_text}


# --- Streamlit UI ---

st.markdown("""
<style>
    .stChatMessage { border-radius: 12px; }
    div[data-testid="stChatMessageContent"] p { font-size: 15px; }
</style>
""", unsafe_allow_html=True)

st.title("💰 Price Bot")
st.caption("Intelligent Hardware Search Engine • Powered by structured logic")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "type": "text", "content": "Hey! I'm the **Price Bot**. Ask me about any component price — CPUs, GPUs, RAM, SSDs, etc.\n\nTry asking: `list all 6000mhz ram with cl30` or `amd gpus in stock`"}
    ]

# Render chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message.get("type") == "text":
            st.markdown(message["content"])
        elif message.get("type") == "dataframe":
            if "text" in message:
                st.markdown(message["text"])
            st.dataframe(message["content"], use_container_width=True, hide_index=True)

# Process User Input
if prompt := st.chat_input("Ask about a component price..."):
    # Add user message to UI
    st.session_state.messages.append({"role": "user", "type": "text", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Simple keyword commands 
    if prompt.lower() in ("help", "?", "commands"):
        reply = "Try asking:\n- `all non rgb rams with 5200mhz`\n- `specific 5080 tuf`\n- `list of nvidia gpus in stock`"
        st.session_state.messages.append({"role": "assistant", "type": "text", "content": reply})
        with st.chat_message("assistant"):
            st.markdown(reply)
            
    elif prompt.lower() in ("list all", "show all", "all"):
        display_df = df[['Category', 'Name', 'price']].copy()
        display_df['Price (INR)'] = display_df['price'].apply(core_engine.format_price)
        display_df = display_df.drop(columns=['price'])
        
        reply = {"type": "dataframe", "content": display_df, "text": f"Here are all {len(df)} components in inventory:"}
        st.session_state.messages.append({"role": "assistant", **reply})
        with st.chat_message("assistant"):
            st.markdown(reply["text"])
            st.dataframe(reply["content"], use_container_width=True, hide_index=True)
            
    else:
        # Dynamic Multi-Attribute Filtering LLM Flow
        with st.spinner("Parsing intent and scanning database..."):
            try:
                reply = process_query(prompt)
            except Exception as e:
                reply = {"type": "text", "content": f"Sorry, an error occurred while processing: {str(e)}"}
            
        st.session_state.messages.append({"role": "assistant", **reply})
        
        # Display response
        with st.chat_message("assistant"):
            if reply["type"] == "text":
                st.markdown(reply["content"])
            elif reply["type"] == "dataframe":
                if "text" in reply:
                    st.markdown(reply["text"])
                st.dataframe(reply["content"], use_container_width=True, hide_index=True)
