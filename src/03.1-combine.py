import argparse
import openai
import json
from dotenv import load_dotenv
import os
from tenacity import (retry, wait_random_exponential, stop_after_attempt)

load_dotenv()

# Configuration from Environment Variables
MODEL = os.getenv("OPENAI_MODEL")
BASE_URL = os.getenv("OPENAI_BASE_URL")
API_KEY = os.getenv("OPENAI_API_KEY")
INPUT_FILE = os.getenv("SCORES_RESULTS_FILE")
OUTPUT_FILE = os.getenv("COMBINED_RESULTS_FILE")

# Initialize the global client
client = openai.OpenAI(
    base_url=BASE_URL,
    api_key=API_KEY
)


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(3))
def get_best_answer_from_llm(subcategory_name, answers_list):
    """
    Sends a list of answers for a specific subcategory to the LLM
    to synthesize into one 'Best Answer'.
    """
    if not answers_list:
        return "No data found."

    # Combine the multiple answers found for this specific subcategory
    combined_context = "\n\n".join([f"Source {i + 1}: {a}" for i, a in enumerate(answers_list)])

    print(f"Synthesizing best answer for: {subcategory_name}...")

    prompt = (
        f"""The following are multiple extracted answers for the subcategory: `{subcategory_name}`.
        Please synthesize them into a single, comprehensive 'Best Answer'. 
        Ensure all specific details (like project names, IDs, or dates) are preserved and 
        remove any duplicate information. Output only the refined answer string.
        For the answer:
        - Use only information present in the source.
        - Do not answer with information not explicitly specified in the source document.
        - Avoid redundant phrasing.
        - If the answer is satisfactory, do not explain anymore.
        - Add a '※ Source:' at the end of your answer, taken from result_text's doc_id and page_num.
        - If a reference is used, include it at the end using '※ Ref:'
        - Use plain text only.Use bullet points if possible.
        - If there is a relevant source, add all necessary information from that source
        - Do not answer if not sure.
        - If there is no answer, say 'I don't know' and do not explain further
        - If the query is for 'description', the answer is usually just one sentence.
        """
    )

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system",
             "content": "You are an expert analyst. Your goal is to merge multiple overlapping information sources into one clear, accurate best answer."},
            {"role": "user", "content": f"{prompt}\n\nContext:\n{combined_context}"}
        ],
        temperature=0
    )
    return response.choices[0].message.content.strip()


def process_results(raw_data):
    """
    Iterates through categories and subcategories, preserving all original fields
    while adding the LLM-generated 'best_answer'.
    """
    final_output = {}

    for category, subcategories in raw_data.items():
        if not isinstance(subcategories, dict):
            final_output[category] = subcategories  # Preserve non-dict top-level items
            continue

        final_output[category] = {}

        for sub_name, sub_data in subcategories.items():
            # 1. Start by copying ALL existing fields (query_used, scored_results, etc.)
            sub_entry = sub_data.copy() if isinstance(sub_data, dict) else sub_data

            if isinstance(sub_data, dict) and "scored_results" in sub_data:
                # 2. Extract answers for synthesis
                answers = [
                    res.get("answer", "")
                    for res in sub_data["scored_results"]
                    if res.get("answer")
                ]

                # 3. Generate the best answer
                if answers:
                    best_val = get_best_answer_from_llm(sub_name, answers)
                    sub_entry["best_answer"] = best_val
                else:
                    sub_entry["best_answer"] = "No answer found in source results."

            final_output[category][sub_name] = sub_entry

    return final_output


# --- Main Execution ---
if __name__ == "__main__":
    try:
        if not INPUT_FILE:
            raise ValueError("Environment variable 'SCORES_RESULTS_FILE' is not set.")

        with open(INPUT_FILE, 'r') as file:
            raw_data = json.load(file)

        print(f"Starting processing... Reading from: {INPUT_FILE}")

        # Process data
        processed_json = process_results(raw_data)

        if not OUTPUT_FILE:
            raise ValueError("Environment variable 'COMBINED_RESULTS_FILE' is not set.")

        # Save to file
        with open(OUTPUT_FILE, 'w') as out_file:
            json.dump(processed_json, out_file, indent=4)

        print(f"Success: Processed results (including all original fields) saved to {OUTPUT_FILE}")

    except FileNotFoundError:
        print(f"Error: The file {INPUT_FILE} was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")