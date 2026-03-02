# src/rag_engine/hybrid_search.py
import sys
import os
import json
import re
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from rank_bm25 import BM25Okapi
from src.rag_engine.vector_store import MSMEVectorStore
from google import genai

class HybridSearchEngine:
    """
    Combines BM25 (keyword) + Semantic (meaning) search.
    BM25 is great for exact terms like 'Section 44AD' or 'MUDRA'.
    Semantic is great for meaning like 'how to reduce tax burden'.
    Together they cover everything.
    """

    def __init__(self, vector_store: MSMEVectorStore):
        self.vector_store = vector_store
        self.bm25_index = {}
        self.chunk_registry = {}
        self._build_bm25_index()

    def _tokenize(self, text: str) -> list:
        """Simple tokenizer — lowercase, remove punctuation"""
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        return [t for t in text.split() if len(t) > 2]

    def _build_bm25_index(self):
        """
        Builds BM25 index from knowledge_base_chunks.json.
        BM25 works on exact keyword matching — great for
        financial terms, section numbers, scheme names.
        """
        print("Building BM25 index...")

        with open("data/processed/knowledge_base_chunks.json", "r", encoding="utf-8") as f:
            chunks = json.load(f)

        # Group by category
        category_chunks = {}
        for chunk in chunks:
            cat = chunk["metadata"]["category"]
            if cat not in category_chunks:
                category_chunks[cat] = []
            category_chunks[cat].append(chunk)

        # Build BM25 for each category
        for category, cat_chunks in category_chunks.items():
            tokenized = [self._tokenize(c["text"]) for c in cat_chunks]
            self.bm25_index[category] = BM25Okapi(tokenized)
            self.chunk_registry[category] = cat_chunks

        # Also build a global index across all chunks
        all_tokenized = [self._tokenize(c["text"]) for c in chunks]
        self.bm25_index["all"] = BM25Okapi(all_tokenized)
        self.chunk_registry["all"] = chunks

        print(f"✅ BM25 index built for {len(category_chunks)} categories + global")

    def hybrid_search(self, query: str, category: str = None,
                      n_results: int = 5, 
                      semantic_weight: float = 0.6,
                      bm25_weight: float = 0.4) -> list:
        """
        Hybrid search combining BM25 + Semantic scores.
        
        Weights: 60% semantic + 40% BM25 by default.
        For exact financial terms (GST section numbers etc),
        increase bm25_weight. For conceptual queries, 
        increase semantic_weight.
        """
        # ── Semantic Search ────────────────────────────────────────
        semantic_results = self.vector_store.semantic_search(
            query=query,
            category=category,
            n_results=n_results * 2  # Get more then filter
        )

        # ── BM25 Search ────────────────────────────────────────────
        index_key = category if category in self.bm25_index else "all"
        bm25 = self.bm25_index[index_key]
        chunks = self.chunk_registry[index_key]

        query_tokens = self._tokenize(query)
        bm25_scores = bm25.get_scores(query_tokens)

        # Get top BM25 results
        top_bm25_indices = sorted(
            range(len(bm25_scores)),
            key=lambda i: bm25_scores[i],
            reverse=True
        )[:n_results * 2]

        # Normalize BM25 scores to 0-1 range
        max_bm25 = max(bm25_scores) if max(bm25_scores) > 0 else 1
        bm25_results = {}
        for idx in top_bm25_indices:
            chunk = chunks[idx]
            normalized_score = bm25_scores[idx] / max_bm25
            bm25_results[chunk["chunk_id"]] = {
                "text": chunk["text"],
                "metadata": chunk["metadata"],
                "bm25_score": normalized_score
            }

        # ── Combine Scores ─────────────────────────────────────────
        combined = {}

        # Add semantic results
        for r in semantic_results:
            chunk_id = r["metadata"].get("source_file", "") + r["text"][:30]
            combined[chunk_id] = {
                "text": r["text"],
                "metadata": r["metadata"],
                "semantic_score": max(0, r["score"]),
                "bm25_score": 0.0
            }

        # Merge BM25 scores
        for chunk_id, bm25_data in bm25_results.items():
            match_key = bm25_data["metadata"].get("source_file", "") + bm25_data["text"][:30]
            if match_key in combined:
                combined[match_key]["bm25_score"] = bm25_data["bm25_score"]
            else:
                combined[match_key] = {
                    "text": bm25_data["text"],
                    "metadata": bm25_data["metadata"],
                    "semantic_score": 0.0,
                    "bm25_score": bm25_data["bm25_score"]
                }

        # Calculate final hybrid score
        results = []
        for key, data in combined.items():
            hybrid_score = (
                semantic_weight * data["semantic_score"] +
                bm25_weight * data["bm25_score"]
            )
            results.append({
                "text": data["text"],
                "metadata": data["metadata"],
                "hybrid_score": round(hybrid_score, 4),
                "semantic_score": round(data["semantic_score"], 4),
                "bm25_score": round(data["bm25_score"], 4)
            })

        # Sort by hybrid score
        results.sort(key=lambda x: x["hybrid_score"], reverse=True)
        return results[:n_results]


# ── Test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("="*55)
    print("Testing Hybrid Search Engine")
    print("="*55)

    store = MSMEVectorStore()
    engine = HybridSearchEngine(store)

    test_queries = [
        ("What is GST input tax credit and how to claim it?", "tax"),
        ("MUDRA loan eligibility and application process", "scheme"),
        ("loan restructuring options for MSME stress", "regulatory"),
        ("What is Section 44AD presumptive taxation?", "tax"),
        ("How to register my business under Udyam?", "scheme"),
    ]

    for query, category in test_queries:
        print(f"\n🔍 Query: '{query}'")
        print(f"   Category: {category}")
        results = engine.hybrid_search(query, category=category, n_results=2)
        for i, r in enumerate(results):
            print(f"\n   Result {i+1}:")
            print(f"   Hybrid: {r['hybrid_score']} | Semantic: {r['semantic_score']} | BM25: {r['bm25_score']}")
            print(f"   Source: {r['metadata'].get('source_file', 'N/A')}")
            print(f"   Text: {r['text'][:180]}...")
        print()