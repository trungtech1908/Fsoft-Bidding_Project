import argparse
import openai
import json
from dotenv import load_dotenv
load_dotenv()
import os
from tenacity import (retry, retry_base, wait,
                      wait_random_exponential, stop, stop_after_attempt)

MODEL = os.getenv("OPENAI_MODEL")
BASE_URL = os.getenv("OPENAI_BASE_URL")
API_KEY = os.getenv("OPENAI_API_KEY")

client = openai.OpenAI(
     base_url=BASE_URL,
     api_key=API_KEY
)

@retry( # Retry, wait, and stop to prevent rate limits
    wait=wait_random_exponential(min=1, max=60),
    stop=stop_after_attempt(3)
)
def call_agent(query, source):

    system_prompt = """
        You are a bidding agent. You must answer concisely and precisely the user's query using the given source text and give a relevance score.
        Do not describe the user query. Do not describe the source.
        Output in JSON format using the keys: 'relevance_score', 'confidence_score', 'answer', 'source'.
        The relevance score and confidence score range from 0 to 10. Use up to 2 points decimal

        For the answer:
        - Use only information present in the source.
        - Do not answer with information not explicitly specified in the source document.
        - Avoid redundant phrasing.
        - If the answer is satisfactory, do not explain anymore.
        - Add a '※ Source:' at the end of your answer, taken from result_text's doc_id and page_num.
        - If a reference is used, include it at the end using '※ Ref:'
        - Use plain text only. Use bullet points if possible.
        - If there is a relevant source, add all necessary information from that source
        - Do not answer if not sure.
        - If there is no answer, say 'I don't know' and do not explain further
        - If the query is for 'description', the answer is usually just one sentence.
    """

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": f"Query: {query}\nSource: {source}"
            }
        ],
        response_format={"type": "json_object"}
    )
    print(response.choices[0].message.content)
    return json.loads(response.choices[0].message.content)

# 1. Initialize the OpenAI client
def process_multiple_results(input_path, output_path):
    # Load the original JSON data
    with open(input_path, 'r') as f:
        data = json.load(f)

    final_output = {} # This is the output file. Will be converted to JSON.
    
    _sub_cat_index = 1
    for category, sub_categories in data.items():
        print(f"--- Processing Category: {_sub_cat_index}. {category} ---")
        _sub_cat_index += 1
        final_output[category] = {}
        
        for sub_cat_name, content in sub_categories.items():
            query_used = content.get("query_used", "")

            # Expecting reranked_results to be a list now
            results_list = content.get("reranked_results", [])
            result_texts = [result['text'] for result in results_list]

            print(f"  Scoring and answering {len(result_texts)} results for [{sub_cat_name}]...")
            
            scored_entries = []

            for result_text in result_texts:
                try:
                    # 2. API Call for each individual result
                    # Parse the JSON from the AI response
                    scores_and_answer = call_agent(query=query_used, source=result_text)

                    # 3. Create a combined entry for this specific result
                    scored_entries.append({
                        "result": result_text,
                        "relevance_score": scores_and_answer.get("relevance_score"),
                        "confidence_score": scores_and_answer.get("confidence_score"),
                        "answer": scores_and_answer.get("answer"),
                        "source": scores_and_answer.get("source")
                    })
                except Exception as e:
                    print(f"    Error scoring a result in {sub_cat_name}: {e}")
                    scored_entries.append({
                        "result": result_text,
                        "error": "Failed to process"
                    })

            # Store the list of results and the query under the sub-category
            final_output[category][sub_cat_name] = {
                "query_used": query_used,
                "scored_results": scored_entries
            }

    # 4. Save the final JSON
    with open(output_path, 'w') as f:
        json.dump(final_output, f, indent=4)

    print("*" * 60)
    print(f"\nCompleted! Results saved to: {output_path}")
    print("*" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Hybrid BM25 + Vector reranker"
    )
    parser.add_argument("--input", help="Input JSON file")
    parser.add_argument("--output", help="Output JSON file")
    args = parser.parse_args()
    input_path = args.input or os.getenv("RERANK_RESULTS_FILE", "../docs/results/02.1-rerank.json")
    output_path = args.output or os.getenv("SCORES_RESULTS_FILE", "../docs/results/03-scores.json")

    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found.")
    else:
        process_multiple_results(input_path, output_path)