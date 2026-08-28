import os
import sys
from typing import List, Dict, Any

from ingestion.hybrid_retriever import HybridRetriever
from ingestion.reranker import BgeReranker
from ingestion.generator import GeminiGenerator

class AskPipeline:
    """
    RAG Orchestrator (AskPipeline).
    Glues together HybridRetriever (Stage 1 Search), BgeReranker (Stage 2 Reranking),
    and GeminiGenerator (Stage 3 Response Generation with structured citations).
    """
    def __init__(
        self,
        qdrant_path: str = "data/qdrant_db",
        collection_name: str = "repomind_collection",
        gemini_model: str = "gemini-3.6-flash"


    ):
        print("[RUNNING] Initializing AskPipeline components...")
        self.retriever = HybridRetriever(qdrant_path=qdrant_path, collection_name=collection_name)
        self.reranker = BgeReranker()
        self.generator = GeminiGenerator(model_name=gemini_model)
        print("[SUCCESS] AskPipeline components initialized and ready!")

    def close(self):
        try:
            if hasattr(self, "retriever") and self.retriever:
                self.retriever.close()
        except Exception:
            pass


    def ask(self, query: str, repo: str = None) -> Dict[str, Any]:
        """
        Executes the end-to-end RAG answer pipeline:
        1. Hybrid Search (Vector + BM25 via RRF) -> Top-20 candidates
        2. Reranking candidates (Cross-Encoder) -> Top-5 candidates
        3. Structured Generation (Gemini with Citations) -> Answer + Citations
        
        Returns:
          A dictionary containing:
            - "answer": Cohesive text answer from Gemini
            - "cited_chunk_ids": List of chunk IDs cited by the model
            - "source_chunks": Details of the top-5 source chunks retrieved
        """
        print(f"\n[ASK] Query received: \"{query}\" (Repo: {repo or 'All'})")
        
        # 1. Perform Hybrid Search to get top-20 candidates
        print("[STEP 1/3] Performing Hybrid Search (Vector + BM25 via RRF)...")
        candidates = self.retriever.search(query=query, top_k=20, method="rrf", repo=repo)
        print(f" -> Retrieved {len(candidates)} hybrid candidate chunks.")

        if not candidates:
            print("[INFO] No candidates retrieved. Returning empty response.")
            return {
                "answer": "No relevant document chunks could be found in the repository.",
                "cited_chunk_ids": [],
                "source_chunks": []
            }

        # Flatten metadata and map keys cleanly before passing to reranker / generator
        flattened_candidates = []
        for cand in candidates:
            payload = cand.get("payload") or {}
            flattened_candidates.append({
                "chunk_id": payload.get("chunk_id") or cand.get("chunk_id") or cand.get("id") or "N/A",
                "doc_id": cand.get("doc_id") or payload.get("doc_id") or "N/A",
                "doc_type": cand.get("doc_type") or payload.get("doc_type") or "N/A",
                "title": payload.get("title") or cand.get("title") or "N/A",
                "url": payload.get("url") or cand.get("url") or "N/A",
                "text": cand.get("text") or payload.get("text") or "",
                "vector_score": cand.get("vector_score") or 0.0,
                "bm25_score": cand.get("bm25_score") or 0.0,
                "rrf_score": cand.get("rrf_score") or 0.0,
            })

        # 2. Pass top-20 to the reranker and get the top-5
        print("[STEP 2/3] Reranking candidates using Cross-Encoder...")
        top_5 = self.reranker.rerank(query=query, candidates=flattened_candidates, top_k=5)
        print(" -> Reranking complete. Selected top-5 precision chunks.")

        # 3. Pass top-5 and query to GeminiGenerator to get the structured answer
        print("[STEP 3/3] Querying Gemini for structured answer and citations...")
        try:
            generator_result = self.generator.generate(query=query, candidates=top_5)
            answer = generator_result.answer
            cited_chunk_ids = generator_result.cited_chunk_ids
        except Exception as e:
            print(f"[WARNING] Failed to generate answer using Gemini: {e}")
            answer = "An error occurred while generating the answer. However, relevant sources are listed below."
            cited_chunk_ids = []

        # 4. Form final return payload
        # Clean up source chunks representation in response
        formatted_sources = []
        for rank, item in enumerate(top_5, 1):
            formatted_sources.append({
                "rank": rank,
                "chunk_id": item["chunk_id"],
                "doc_id": item["doc_id"],
                "doc_type": item["doc_type"],
                "title": item["title"],
                "url": item["url"],
                "snippet": item["text"][:150] + ("..." if len(item["text"]) > 150 else ""),
                "text": item["text"],
                "rerank_score": item.get("rerank_score"),
                "rerank_prob": item.get("rerank_prob")
            })


        return {
            "answer": answer,
            "cited_chunk_ids": cited_chunk_ids,
            "source_chunks": formatted_sources
        }

if __name__ == "__main__":
    # Test script block to verify the AskPipeline end-to-end flow locally
    # Safe print helper to prevent Windows console Unicode crashes
    def safe_print_dict(data: dict):
        try:
            print(json.dumps(data, indent=2))
        except Exception:
            # Fallback
            import sys
            encoding = sys.stdout.encoding or 'utf-8'
            dumped = json.dumps(data, indent=2, default=str)
            print(dumped.encode(encoding, errors='replace').decode(encoding))

    import json
    test_query = "How do I update mcp dependency version?"
    
    try:
        pipeline = AskPipeline()
        result = pipeline.ask(test_query)
        print("\n================ RAG Pipeline Result ================")
        safe_print_dict(result)
        print("=====================================================")
    except Exception as e:
        print(f"[ERROR] AskPipeline execution failed: {e}")
