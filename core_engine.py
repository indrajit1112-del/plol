import pandas as pd
import openpyxl
import re
import os
import json
import numpy as np
from pydantic import BaseModel, Field
from typing import List, Optional
from thefuzz import fuzz, process

# ---------------------------------------------------------------------------
# Pydantic Schema — "Antigravity" edition
# ---------------------------------------------------------------------------

class HardwareQuery(BaseModel):
    category: Optional[str] = Field(None, description="Primary hardware subset. Examples: 'GPU', 'RAM', 'CPU', 'SSD', 'Quadro'")
    gpu_brand: Optional[str] = Field(None, description="Extract 'AMD' or 'Nvidia' from conversational cues. E.g. 'Radeon' or 'RX' -> AMD. 'GeForce' or 'RTX' -> Nvidia.")
    ssd_gen: Optional[int] = Field(None, description="PCIe Generation for SSDs (e.g., 3, 4, 5). Extract 5 from 'Gen5 SSD'.")
    speed_mhz: Optional[List[int]] = Field(None, description="Array of clock speeds in MHz (e.g. [6000, 5200]). IMPORTANT: Only fill this for RAM. Never put GPU model numbers (like 5080, 5090) here.")
    max_cas_latency: Optional[int] = Field(None, description="Maximum acceptable CAS latency value (e.g. 30, 32, 36). Matches items with Latency <= this value.")
    is_rgb: Optional[bool] = Field(None, description="Strict inclusion (true), strict exclusion (false), or indifference (null) toward RGB lighting.")
    specific_model: Optional[str] = Field(None, description="Highly specific nomenclature for substring matching (e.g. '7800X 3D', '5080', '5090 TUF'). MUST EXCLUDE general terms like 'all ram', 'gpu', 'mhz', 'list'. Leave null if query is generic.")
    min_price: Optional[int] = Field(None, description="Minimum price in INR. E.g. 'above 30k' -> 30000. Leave null if no lower bound mentioned.")
    max_price: Optional[int] = Field(None, description="Maximum price in INR. E.g. 'under 50k' -> 50000, 'below 1 lakh' -> 100000. Leave null if no upper bound.")
    inferred_tier: Optional[str] = Field(None, description="Inferred price/quality tier: 'budget', 'mid-range', or 'high-end'. Derive from cues like 'cheap'/'affordable' -> budget, 'best'/'top'/'flagship' -> high-end. Leave null if ambiguous.")


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------

def load_and_enrich_data(xlsx_path=None):
    if not xlsx_path:
        xlsx_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prices.xlsx")
    
    wb = openpyxl.load_workbook(xlsx_path, read_only=True)
    ws = wb.active
    items = []
    
    for row in ws.iter_rows(min_row=2, values_only=True):
        category, name, price = row[0], row[1], row[2]
        if name and price is not None:
            items.append({
                "Category": str(category).strip() if category else "Other",
                "Name": str(name).strip(),
                "price": int(price)
            })
    wb.close()
    
    df = pd.DataFrame(items)
    if df.empty:
        return df
    
    # regex parsing for multidimensional features
    # Speed (MHz) - e.g. DDR5 6000
    df['Speed'] = df['Name'].str.extract(r'(?:DDR\d\s+|)(\d{3,4})(?=\s*MHz|\s*\()', flags=re.IGNORECASE)[0]
    # Fallback to general 4 digits if nothing found but it's RAM
    mask_ram = df['Category'].str.contains('RAM', case=False, na=False)
    # Another approach: find DDR* \d{3,4}
    alt_speed = df['Name'].str.extract(r'DDR\d.*?(\d{4})')[0]
    df.loc[mask_ram & df['Speed'].isna(), 'Speed'] = alt_speed
    df['Speed'] = pd.to_numeric(df['Speed'])

    # CAS Latency - e.g. CL30
    df['Latency'] = pd.to_numeric(df['Name'].str.extract(r'CL(\d{2})', flags=re.IGNORECASE)[0])
    
    # RGB state
    df['RGB'] = df['Name'].str.contains(r'(?i)\bRGB\b', na=False)

    # GPU Brand
    df['Brand'] = None
    # Check explicitly in Name
    df.loc[df['Name'].str.contains(r'(?i)\b(?:Nvidia|RTX|GTX|GeForce)\b', na=False), 'Brand'] = 'Nvidia'
    df.loc[df['Name'].str.contains(r'(?i)\b(?:AMD|RX|Radeon)\b', na=False), 'Brand'] = 'AMD'
    # Fallback checking Category (fixes issues where GPU name is just "5080" without RTX)
    df.loc[df['Brand'].isna() & df['Category'].str.contains(r'(?i)Nvidia', na=False), 'Brand'] = 'Nvidia'
    df.loc[df['Brand'].isna() & df['Category'].str.contains(r'(?i)AMD', na=False), 'Brand'] = 'AMD'

    # SSD PCIe Gen
    df['PCIe_Gen'] = pd.to_numeric(df['Name'].str.extract(r'(?i)Gen(\d)')[0])

    return df


def load_cleaned_inventory(json_path=None):
    """Load pre-computed cleaned_inventory.json. Falls back to xlsx+regex if JSON doesn't exist."""
    if not json_path:
        json_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cleaned_inventory.json")
    
    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        df = pd.DataFrame(data)
        # Ensure numeric columns are typed correctly
        for col in ['price', 'Speed', 'Latency', 'PCIe_Gen']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        if 'RGB' in df.columns:
            df['RGB'] = df['RGB'].astype(bool)
        return df
    
    # Fallback to original xlsx+regex pipeline
    return load_and_enrich_data()


# ---------------------------------------------------------------------------
# LLM Intent Extraction
# ---------------------------------------------------------------------------

def extract_search_intent(client, user_message, recent_context=""):
    context_note = ""
    if recent_context:
        context_note = f"\n\nRecent conversation for context:\n{recent_context}\nUse this context to infer the missing variables."

    system_prompt = f"""You are a high-precision hardware database query parser.
    Your task is to map casual vocabulary to exact database schema.
    
    === ANTIGRAVITY PRINCIPLE ===
    Your default posture is EXPANSIVE. Maximize inventory visibility by leaving constraints null
    unless the user has EXPLICITLY requested them. When in doubt, leave it null and let the
    engine show more results rather than fewer.
    
    Rules:
    - GPU Brands: 'Radeon' or 'RX' maps to gpu_brand: 'AMD'. 'GeForce' or 'RTX' maps to gpu_brand: 'Nvidia'.
    - SSD Generation: 'Gen5 SSD' maps to ssd_gen: 5. 'Gen4' maps to 4.
    - Logical Disjunctions: Handle "OR" statements. E.g. "all 6000mhz ram or 5200 mhz ram" -> speed_mhz: [6000, 5200]
    - RGB state: "Show me all RGB rams" -> is_rgb: true. "All non rgb rams" -> is_rgb: false. "All rams with cl30" (no mention of lighting) -> is_rgb: null.
    - Explicit exclusions (e.g., "no rgb", "-rgb") -> is_rgb: false.
    - specific_model: MUST ONLY contain the actual product identifier. If the user types "all ram" or "6000 ram", specific_model MUST be null. Only use for things like "5080", "5090", "7800X 3D".
    - speed_mhz: MUST NEVER contain GPU model numbers like 5080, 5090, 4070. Only use for memory speeds like 5200, 6000.
    - Do not guess constraints if they are not explicitly specified.
    
    === BUDGET / PRICE EXTRACTION ===
    - "under 50k" or "below 50000" -> max_price: 50000
    - "above 30k" or "over 30000" -> min_price: 30000
    - "between 20k and 40k" -> min_price: 20000, max_price: 40000
    - "below 1 lakh" -> max_price: 100000
    - Interpret "k" as thousands (e.g. 50k = 50000). Interpret "lakh" as 100000.
    - Leave min_price/max_price null if no budget is mentioned.
    
    === LATENCY (max_cas_latency) ===
    - "CL30" or "cl30" -> max_cas_latency: 30 (items with latency <= 30)
    - "CL36 or lower" -> max_cas_latency: 36
    - If the user says an exact CL value like "CL30 ram", set max_cas_latency: 30 (this will match CL30 and below).
    
    === TIER INFERENCE (inferred_tier) ===
    - "cheap", "affordable", "entry level" -> inferred_tier: "budget"
    - "mid range", "decent" -> inferred_tier: "mid-range"
    - "best", "top", "flagship", "premium" -> inferred_tier: "high-end"
    - Leave null if no quality/tier cue is present.
    {context_note}
    """

    response = client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ],
        response_format=HardwareQuery,
        temperature=0.1
    )
    
    return response.choices[0].message.parsed


# ---------------------------------------------------------------------------
# Pandas Filter Engine
# ---------------------------------------------------------------------------

def execute_pandas_filters(df, query_obj: HardwareQuery):
    if df.empty:
        return df

    filtered_df = df.copy()

    # Category Mask
    if query_obj.category:
        filtered_df = filtered_df[filtered_df['Category'].str.contains(query_obj.category, case=False, na=False)]

    # Brand Mask (GPU)
    if query_obj.gpu_brand:
        filtered_df = filtered_df[filtered_df['Brand'] == query_obj.gpu_brand]

    # Speed Mask (RAM)
    if query_obj.speed_mhz and len(query_obj.speed_mhz) > 0:
        filtered_df = filtered_df[filtered_df['Speed'].isin(query_obj.speed_mhz)]

    # CAS Latency Mask — range-based (<=) instead of exact match
    if query_obj.max_cas_latency is not None:
        filtered_df = filtered_df[filtered_df['Latency'] <= query_obj.max_cas_latency]

    # explicit RGB Mask
    if query_obj.is_rgb is not None:
        filtered_df = filtered_df[filtered_df['RGB'] == query_obj.is_rgb]

    # SSD PCIe Gen Mask
    if query_obj.ssd_gen:
        filtered_df = filtered_df[filtered_df['PCIe_Gen'] == query_obj.ssd_gen]

    # Price Range Masks
    if query_obj.min_price is not None:
        filtered_df = filtered_df[filtered_df['price'] >= query_obj.min_price]
    if query_obj.max_price is not None:
        filtered_df = filtered_df[filtered_df['price'] <= query_obj.max_price]

    # Specific Model Fuzzy Fallback
    if query_obj.specific_model:
        exact_mask = filtered_df['Name'].str.contains(query_obj.specific_model, case=False, regex=False, na=False)
        exact_df = filtered_df[exact_mask]
        
        if not exact_df.empty:
            filtered_df = exact_df
        else:
            pool_names = filtered_df['Name'].tolist()
            if pool_names:
                results = process.extract(query_obj.specific_model, pool_names, scorer=fuzz.token_set_ratio, limit=len(pool_names))
                matched_names = [r[0] for r in results if r[1] >= 45]
                filtered_df = filtered_df[filtered_df['Name'].isin(matched_names)]

    return filtered_df


# ---------------------------------------------------------------------------
# Dynamic Fallback — progressively relax constraints
# ---------------------------------------------------------------------------

# Constraints stripped in order of strictness (strictest first)
_RELAXATION_ORDER = ['specific_model', 'max_cas_latency', 'speed_mhz', 'min_price', 'max_price']

def execute_with_fallback(df, query_obj: HardwareQuery):
    """Run filters; if empty, progressively strip the strictest constraints and re-run.
    
    Returns:
        (result_df, was_relaxed: bool, relaxed_fields: list[str])
    """
    result = execute_pandas_filters(df, query_obj)
    if not result.empty:
        return result, False, []

    relaxed_fields = []
    relaxed_query = query_obj.model_copy()

    for field_name in _RELAXATION_ORDER:
        if getattr(relaxed_query, field_name) is not None:
            setattr(relaxed_query, field_name, None)
            relaxed_fields.append(field_name)
            result = execute_pandas_filters(df, relaxed_query)
            if not result.empty:
                return result, True, relaxed_fields

    return result, True, relaxed_fields


# ---------------------------------------------------------------------------
# Semantic Search Fallback (OpenAI Embeddings)
# ---------------------------------------------------------------------------

def generate_embeddings(client, texts: list, model="text-embedding-3-small") -> np.ndarray:
    """Generate embeddings for a list of text strings. Returns an (N, dim) numpy array."""
    # OpenAI supports up to 2048 texts per batch request
    batch_size = 2000
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        response = client.embeddings.create(input=batch, model=model)
        batch_embeddings = [item.embedding for item in response.data]
        all_embeddings.extend(batch_embeddings)
    return np.array(all_embeddings, dtype=np.float32)


def semantic_search(client, query: str, df: pd.DataFrame, embeddings: np.ndarray, top_k: int = 10) -> pd.DataFrame:
    """Embed the user query and rank the dataframe rows by cosine similarity to hardware names."""
    if df.empty or embeddings.shape[0] == 0:
        return df

    query_emb = generate_embeddings(client, [query])[0]  # shape (dim,)

    # Cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1  # avoid division by zero
    normed = embeddings / norms
    query_norm = query_emb / (np.linalg.norm(query_emb) or 1)
    similarities = normed @ query_norm  # shape (N,)

    top_indices = np.argsort(similarities)[::-1][:top_k]
    return df.iloc[top_indices].copy()


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

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
