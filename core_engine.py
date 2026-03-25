import pandas as pd
import openpyxl
import re
import os
import json
from pydantic import BaseModel, Field
from typing import List, Optional
from thefuzz import fuzz, process

class HardwareQuery(BaseModel):
    category: Optional[str] = Field(None, description="Primary hardware subset. Examples: 'GPU', 'RAM', 'CPU', 'SSD', 'Quadro'")
    gpu_brand: Optional[str] = Field(None, description="Extract 'AMD' or 'Nvidia' from conversational cues. E.g. 'Radeon' or 'RX' -> AMD. 'GeForce' or 'RTX' -> Nvidia.")
    ssd_gen: Optional[int] = Field(None, description="PCIe Generation for SSDs (e.g., 3, 4, 5). Extract 5 from 'Gen5 SSD'.")
    speed_mhz: Optional[List[int]] = Field(None, description="Array of clock speeds in MHz (e.g. [6000, 5200]). Only fill if asked explicitly.")
    cas_latency: Optional[int] = Field(None, description="Absolute numerical value of CL latency (e.g. 30, 32, 36).")
    is_rgb: Optional[bool] = Field(None, description="Strict inclusion (true), strict exclusion (false), or indifference (null) toward RGB lighting.")
    specific_model: Optional[str] = Field(None, description="Highly specific nomenclature intended for exact substring matching (e.g. '7800X 3D'). Exclude vague terms like 'gpu'.")

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

def extract_search_intent(client, user_message, recent_context=""):
    context_note = ""
    if recent_context:
        context_note = f"\n\nRecent conversation for context:\n{recent_context}\nUse this context to infer the missing variables."

    system_prompt = f"""You are a high-precision hardware database query parser.
    Your task is to map casual vocabulary to exact database schema.
    
    Rules:
    - GPU Brands: 'Radeon' or 'RX' maps to gpu_brand: 'AMD'. 'GeForce' or 'RTX' maps to gpu_brand: 'Nvidia'.
    - SSD Generation: 'Gen5 SSD' maps to ssd_gen: 5. 'Gen4' maps to 4.
    - Logical Disjunctions: Handle "OR" statements. E.g. "all 6000mhz ram or 5200 mhz ram" -> speed_mhz: [6000, 5200]
    - RGB state: "Show me all RGB rams" -> is_rgb: true. "All non rgb rams" -> is_rgb: false. "All rams with cl30" (no mention of lighting) -> is_rgb: null.
    - Explicit exclusions (e.g., "no rgb", "-rgb") -> is_rgb: false.
    - Do not guess constraints if they are not explicitly specified.
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

    # CAS Latency Mask
    if query_obj.cas_latency:
        filtered_df = filtered_df[filtered_df['Latency'] == query_obj.cas_latency]

    # explicit RGB explicit Mask
    if query_obj.is_rgb is not None:
        filtered_df = filtered_df[filtered_df['RGB'] == query_obj.is_rgb]

    # SSD PCIe Gen Mask
    if query_obj.ssd_gen:
        filtered_df = filtered_df[filtered_df['PCIe_Gen'] == query_obj.ssd_gen]

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
