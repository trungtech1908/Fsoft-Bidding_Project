import json
import pandas as pd
from openpyxl.styles import Font, Alignment
import os
from dotenv import load_dotenv
load_dotenv()

# --- 1. LOAD JSON FILE FROM PREVIOUS STAGE ---
INPUT_PATH = os.getenv("COMBINED_RESULTS_FILE")

try:
    with open(INPUT_PATH, 'r', encoding='utf-8') as f:
        data_input = json.load(f)
except FileNotFoundError:
    print(f"Error: The file {INPUT_PATH} was not found.")
    exit()

# Ensure we are working with a list of tasks
if isinstance(data_input, dict):
    tasks = [data_input]
else:
    tasks = data_input

rows = []

# --- 2. PROCESSING LOGIC ---
for task_dict in tasks:
    for category, sub_categories in task_dict.items():
        if not isinstance(sub_categories, dict):
            continue

        for criteria_name, data in sub_categories.items():
            # Handle case where data might be a list or missing scored_results
            scored_results = data.get("scored_results", []) if isinstance(data, dict) else []
            best_answer=data.get("best_answer", [])
            best_entry = None
            max_combined_score = -1

            # Find the "Winner" (Highest Relevance * Confidence)
            for entry in scored_results:
                try:
                    rel = float(entry.get("relevance_score", 0))
                    conf = float(entry.get("confidence_score", 0))
                except (ValueError, TypeError):
                    rel, conf = 0, 0

                combined_score = rel * conf

                if combined_score > max_combined_score:
                    max_combined_score = combined_score
                    best_entry = entry

            # Map to final row format
            if best_entry:
                rows.append({
                    "13 Criterias": category,
                    "Expanded 36 criterias": criteria_name,
                    "Results": best_answer,
                    "Source": best_entry.get("source", "N/A"),
                    "Original Result": str(best_entry.get("result", "")),
                    "Score": best_entry.get("relevance_score"),
                    "Confidence Score": best_entry.get("confidence_score")
                })
            else:
                rows.append({
                    "13 Criterias": category,
                    "Expanded 36 criterias": criteria_name,
                    "Results": "Không tìm thấy",
                    "Source": "N/A",
                    "Original Result": "N/A",  # Added for consistency
                    "Score": 0,
                    "Confidence Score": 0
                })

# --- 3. EXPORT TO EXCEL ---
df = pd.DataFrame(rows)
output_file = os.getenv("FINAL_EXCEL_FILE")

with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
    df.to_excel(writer, index=False, sheet_name='Analysis')
    worksheet = writer.sheets['Analysis']

    # Styles
    bold_font = Font(bold=True)
    wrap_alignment = Alignment(wrap_text=True, vertical='top')

    # Apply bold to headers
    for cell in worksheet[1]:
        cell.font = bold_font

    # Column width adjustments
    dims = {'A': 25, 'B': 30, 'C': 80, 'D': 20, 'E': 40, 'F': 10, 'G': 15}
    for col_letter, width in dims.items():
        worksheet.column_dimensions[col_letter].width = width

    # Apply text wrapping to all data cells
    for row in worksheet.iter_rows(min_row=2, max_row=worksheet.max_row, min_col=1, max_col=len(df.columns)):
        for cell in row:
            cell.alignment = wrap_alignment

print(f"Success! Data processed into {output_file}")