import json
import argparse
import sys
import re
import math
from concurrent.futures import ThreadPoolExecutor

from dotenv import load_dotenv
import os

load_dotenv()
# File config
INPUT_PATH = os.getenv("CRITERIA_MAPPING_PATH")
OUTPUT_PATH = os.getenv("SEARCH_RESULTS_FILE")

COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME") # Qdrant collection to search

SCORE_THRESHOLD = 0
RESULTS_LIMIT_VECTOR = 5 # Number of results to search for both vector and BM25 text search
RESULTS_LIMIT_BM25 = 10

# Embedding and storage
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")

# Qdrant config
QDRANT_HOST = os.getenv("QDRANT_HOST")
QDRANT_PORT = os.getenv("QDRANT_PORT")

def _get_word_set(text):
    """Utility to extract a set of unique words for comparison."""
    return set(re.findall(r'\w+', text.lower()))

def _deduplicate_results(vector_res, bm25_res):
    """
    Combines results and removes duplicates or near-duplicates (80%+ overlap).
    Priority: Vector results (semantic) > BM25 results (lexical).
    """
    vector_list = vector_res if isinstance(vector_res, list) else []
    bm25_list = bm25_res if isinstance(bm25_res, list) else []
    
    # Combined pool: concatenate both lists into one
    combined_pool = vector_list + bm25_list
    unique_list = []

    for item in combined_pool:
        new_text = item.get("text", "")
        new_words = _get_word_set(new_text)
        
        if not new_words:
            continue

        is_duplicate = False
        for existing_item in unique_list:
            existing_words = _get_word_set(existing_item.get("text", ""))
            
            # Calculate Overlap Ratio
            # (Intersection / size of the smaller set)
            intersection = new_words.intersection(existing_words)
            overlap_ratio = len(intersection) / min(len(new_words), len(existing_words))
            
            # If 80% or more of the words are the same, consider it a duplicate
            if overlap_ratio > 0.5:
                is_duplicate = True
                # Optional: If the new chunk is actually LONGER than the existing one, 
                # you could replace it here, but keeping the first (Vector) is safer.
                break
        
        if not is_duplicate:
            unique_list.append(item)
            
    return unique_list if unique_list else "Not found"

def _vector_task(pipeline, query): # Vector/Semantic Search
    print(f"Collection {COLLECTION_NAME} -> Query {query}")
    results = pipeline.search(query=query, collection_name=COLLECTION_NAME, limit=RESULTS_LIMIT_VECTOR)
    filtered = [res for res in results if res.get('score', 0) > SCORE_THRESHOLD]

    normalized = _normalize_vector_scores(filtered)
    return normalized if normalized else "Not found"

import math

def _normalize_bm25_scores(results):
    """
    Min–max normalize BM25 scores to [0, 1] per query.
    """
    if not results or results == "Not found":
        return results

    scores = [res.get("score", 0.0) for res in results]
    min_s = min(scores)
    max_s = max(scores)

    # Avoid division by zero (all scores equal)
    if max_s == min_s:
        for res in results:
            res["score_norm"] = 1.0
        return results

    for res in results:
        res["score_norm"] = (res["score"] - min_s) / (max_s - min_s)

    return results



def _normalize_vector_scores(results):
    if not results or results == "Not found":
        return results

    scores = [res.get("score", 0.0) for res in results]
    min_s, max_s = min(scores), max(scores)

    if max_s == min_s:
        for res in results:
            res["score_norm"] = 1.0
        return results

    for res in results:
        res["score_norm"] = (res["score"] - min_s) / (max_s - min_s)

    return results


def _bm25_task(pipeline, query):  # BM25 Search
    results = pipeline.search_bm25(
        query=query,
        collection_name=COLLECTION_NAME,
        limit=RESULTS_LIMIT_BM25,
    )

    filtered = [res for res in results if res.get('score', 0) > SCORE_THRESHOLD]

    normalized = _normalize_bm25_scores(filtered)

    return normalized if normalized else "Not found"


def _run_parallel_search(name, keywords, pipeline): # Multithreaded run for faster performance
    search_query = f"{name} " + " ".join(keywords)
    
    with ThreadPoolExecutor(max_workers=2) as executor:
        v_future = executor.submit(_vector_task, pipeline, search_query)
        b_future = executor.submit(_bm25_task, pipeline, search_query)
        
        v_res = v_future.result()
        b_res = b_future.result()
        
        # --- DEDUPLICATION ---
        merged_unique = _deduplicate_results(v_res, b_res)
        
        return {
            "query_used": search_query,
            "vector_results": v_res,
            "bm25_results": b_res,
            "unique_results": merged_unique # Combined list without duplicates
        }

from utils._pdf_full_pipeline import PDFToQdrantPipeline

def main():
    parser = argparse.ArgumentParser(description="Hybrid Parallel Search with Deduplication")
    parser.add_argument('criteria', type=str, nargs='?', default=None)
    args = parser.parse_args()

    try:
        with open(INPUT_PATH, 'r', encoding='utf-8') as f:
            mapping = json.load(f)
    except FileNotFoundError:
        print(f"Error: {INPUT_PATH} not found.");
        sys.exit(1)

    final_results = {}

    # Initialize pipeline
    pipeline = PDFToQdrantPipeline(
        embedding_model=EMBEDDING_MODEL,
        qdrant_host=QDRANT_HOST,
        qdrant_port=QDRANT_PORT
    )
    # If no criteria is provided, search for all criterias (either 13 or 36)
    if args.criteria is None:
        print("Processing 36 criteria (Vector + BM25 + Deduplication)...")
        for cat, items in mapping.items():
            final_results[cat] = {k: _run_parallel_search(k, v, pipeline) for k, v in items.items()}
    else:
        target = args.criteria.strip().lower()
        match = next(((cat, k, v) for cat, items in mapping.items() for k, v in items.items() if k.lower() == target),
                     None)
        if match:
            print(f"Executing hybrid search for: {match[1]}")
            final_results[match[0]] = {match[1]: _run_parallel_search(match[1], match[2], pipeline)}
        else:
            print(f"Error: {target} not found in mapping.");
            sys.exit(1)

    # Export to file
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, indent=4, ensure_ascii=False)

    print(f"Success! Data with unique_results saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
