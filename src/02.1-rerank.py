import json
import math
import argparse
import os
from typing import Dict, List
from dotenv import load_dotenv
load_dotenv()

RERANK_THRESHOLD = os.getenv("RERANK_THRESHOLD")
RERANK_THRESHOLD = float(RERANK_THRESHOLD)

class HybridReranker:
    def __init__(
        self,
        alpha: float = 0.6,      # vector weight
        bm25_k: float = 1.0,     # sigmoid steepness
        bm25_x0: float = 8.0     # sigmoid midpoint
    ):
        self.alpha = alpha
        self.bm25_k = bm25_k
        self.bm25_x0 = bm25_x0

    # ----------------------------
    # Normalization
    # ----------------------------

    def _sigmoid_bm25(self, score: float) -> float:
        return 1.0 / (1.0 + math.exp(-self.bm25_k * (score - self.bm25_x0)))

    def _minmax_vector(self, results: List[Dict]) -> Dict[str, float]:
        scores = [r.get("score", 0.0) for r in results]
        if not scores:
            return {}

        min_s, max_s = min(scores), max(scores)
        if min_s == max_s:
            return {
                r["metadata"]["chunk_id"]: 1.0
                for r in results
            }

        return {
            r["metadata"]["chunk_id"]:
                (r["score"] - min_s) / (max_s - min_s)
            for r in results
        }

    # ----------------------------
    # Core logic
    # ----------------------------

    def rerank_entry(self, entry: Dict) -> Dict:
        """
        Reranks one query result block.
        """
        vector_results = entry.get("vector_results", [])
        bm25_results = entry.get("bm25_results", [])
        unique_results = entry.get("unique_results", [])

        if unique_results == "Not found" or not unique_results:
            entry["reranked_results"] = unique_results
            return entry

        vector_map = self._minmax_vector(vector_results)

        bm25_map = {
            r["metadata"]["chunk_id"]:
                self._sigmoid_bm25(r.get("score", 0.0))
            for r in bm25_results
        }

        for r in unique_results:
            cid = r.get("metadata", {}).get("chunk_id")

            v = vector_map.get(cid, 0.0)
            b = bm25_map.get(cid, 0.0)

            r["vector_score_norm"] = v
            r["bm25_score_norm"] = b
            r["hybrid_score"] = self.alpha * v + (1.0 - self.alpha) * b

        # Prune low-relevance results
        unique_results = [
            r for r in unique_results
            if r.get("hybrid_score", 0.0) >= RERANK_THRESHOLD
        ]

        unique_results.sort(
            key=lambda x: x.get("hybrid_score", 0.0),
            reverse=True
        )

        entry["reranked_results"] = unique_results
        return entry


# ============================================================
# File-level processing
# ============================================================

def rerank_file(
    input_path: str,
    output_path: str,
    alpha: float = 0.6,
    bm25_k: float = 1.0,
    bm25_x0: float = 8.0
):
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    reranker = HybridReranker(alpha, bm25_k, bm25_x0)

    for category, criteria in data.items():
        for crit_name, crit_data in criteria.items():
            data[category][crit_name] = reranker.rerank_entry(crit_data)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

    print(f"✅ Reranked file written to: {output_path}")


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    print(os.getenv("SEARCH_RESULTS_FILE"))
    parser = argparse.ArgumentParser(
        description="Hybrid BM25 + Vector reranker"
    )
    parser.add_argument("--input", help="Input JSON file")
    parser.add_argument("--output", help="Output JSON file")
    parser.add_argument("--alpha", type=float, default=0.6)
    parser.add_argument("--bm25_k", type=float, default=1.0)
    parser.add_argument("--bm25_x0", type=float, default=8.0)

    args = parser.parse_args()

    rerank_file(
        input_path=os.getenv("SEARCH_RESULTS_FILE"),
        output_path=os.getenv("RERANK_RESULTS_FILE"),
        alpha=args.alpha,
        bm25_k=args.bm25_k,
        bm25_x0=args.bm25_x0
    )
