"""
data_ingestion.py — Standalone script to clean and structure hardware inventory.

Usage:
    python data_ingestion.py

Requires OPENAI_API_KEY environment variable to be set.

Reads prices.xlsx, sends each product name to gpt-4o-mini for structured spec extraction,
and saves the result to cleaned_inventory.json.
"""

import os
import json
import time
import openpyxl
from pydantic import BaseModel, Field
from typing import Optional
from openai import OpenAI

# ---------------------------------------------------------------------------
# Pydantic model for structured spec extraction
# ---------------------------------------------------------------------------

class HardwareSpecs(BaseModel):
    speed_mhz: Optional[int] = Field(None, description="Clock speed in MHz for RAM (e.g. 6000, 5200). null for non-RAM items.")
    latency: Optional[int] = Field(None, description="CAS Latency value (e.g. 30, 32, 36). null if not present.")
    is_rgb: Optional[bool] = Field(None, description="True if the product name indicates RGB lighting, False if explicitly 'non-RGB', null if unclear.")
    pcie_gen: Optional[int] = Field(None, description="PCIe generation for SSDs (e.g. 3, 4, 5). null for non-SSD items.")
    brand: Optional[str] = Field(None, description="GPU brand: 'Nvidia' or 'AMD'. null for non-GPU items. Infer from keywords like RTX/GeForce -> Nvidia, RX/Radeon -> AMD.")


def load_raw_data(xlsx_path: str) -> list[dict]:
    """Load raw rows from prices.xlsx."""
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
    return items


def extract_specs_batch(client: OpenAI, items: list[dict], batch_size: int = 20) -> list[dict]:
    """Send product names to gpt-4o-mini in small batches for structured spec extraction."""
    enriched = []
    total = len(items)

    for i in range(0, total, batch_size):
        batch = items[i:i + batch_size]
        print(f"  Processing batch {i // batch_size + 1} ({i + 1}–{min(i + batch_size, total)} of {total})...")

        for item in batch:
            retries = 3
            for attempt in range(retries):
                try:
                    response = client.beta.chat.completions.parse(
                        model="gpt-4o-mini",
                        messages=[
                            {
                                "role": "system",
                                "content": (
                                    "You are a hardware spec extraction engine. Given a product name and its category, "
                                    "extract the structured specifications. Be precise:\n"
                                    "- speed_mhz: only for RAM (e.g. DDR5 6000 -> 6000)\n"
                                    "- latency: CAS Latency from CL## pattern (e.g. CL30 -> 30)\n"
                                    "- is_rgb: true if 'RGB' in name, false if 'non-RGB' or similar, null otherwise\n"
                                    "- pcie_gen: PCIe gen for SSDs (e.g. Gen4 -> 4)\n"
                                    "- brand: 'Nvidia' for RTX/GTX/GeForce, 'AMD' for RX/Radeon. Also check category.\n"
                                    "Return null for any field that doesn't apply to this product type."
                                )
                            },
                            {
                                "role": "user",
                                "content": f"Category: {item['Category']}\nProduct: {item['Name']}"
                            }
                        ],
                        response_format=HardwareSpecs,
                        temperature=0.0
                    )

                    specs = response.choices[0].message.parsed

                    enriched.append({
                        "Category": item["Category"],
                        "Name": item["Name"],
                        "price": item["price"],
                        "Speed": specs.speed_mhz,
                        "Latency": specs.latency,
                        "RGB": specs.is_rgb if specs.is_rgb is not None else False,
                        "Brand": specs.brand,
                        "PCIe_Gen": specs.pcie_gen
                    })
                    break  # success

                except Exception as e:
                    if attempt < retries - 1:
                        print(f"    Retry {attempt + 1} for '{item['Name']}': {e}")
                        time.sleep(1)
                    else:
                        print(f"    FAILED after {retries} attempts for '{item['Name']}': {e}")
                        # Fall back to raw data with null specs
                        enriched.append({
                            "Category": item["Category"],
                            "Name": item["Name"],
                            "price": item["price"],
                            "Speed": None,
                            "Latency": None,
                            "RGB": False,
                            "Brand": None,
                            "PCIe_Gen": None
                        })

        # Small delay between batches to respect rate limits
        if i + batch_size < total:
            time.sleep(0.5)

    return enriched


def main():
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        print("ERROR: OPENAI_API_KEY environment variable is not set.")
        print("Set it with: set OPENAI_API_KEY=sk-...")
        return

    client = OpenAI(api_key=api_key)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    xlsx_path = os.path.join(script_dir, "prices.xlsx")
    json_path = os.path.join(script_dir, "cleaned_inventory.json")

    if not os.path.exists(xlsx_path):
        print(f"ERROR: {xlsx_path} not found.")
        return

    print(f"Loading raw data from {xlsx_path}...")
    items = load_raw_data(xlsx_path)
    print(f"Found {len(items)} products.\n")

    print("Extracting specs via gpt-4o-mini structured outputs...")
    enriched = extract_specs_batch(client, items)

    print(f"\nSaving {len(enriched)} enriched records to {json_path}...")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(enriched, f, indent=2, ensure_ascii=False)

    print(f"Done! Cleaned inventory saved to: {json_path}")


if __name__ == "__main__":
    main()
