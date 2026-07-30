import os
import sys
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient

from ingestion.embedder import TextEmbedder
from ingestion.bm25_retriever import BM25Retriever

class HybridRetriever:
    """
    Hybrid Search Engine (Dense Vector + Sparse BM25 Fusion).
    Combines semantic vector embeddings from Qdrant with BM25 lexical keyword matching
    using Reciprocal Rank Fusion (RRF) or Min-Max Normalized Weighted Sum.
    """
    def __init__(
        self,
        qdrant_path: str = "data/qdrant_db",
        collection_name: str = "repomind_collection",
        bm25_retriever: Optional[BM25Retriever] = None
    ):
        self.qdrant_path = qdrant_path
        self.collection_name = collection_name

        from ingestion.qdrant_store import get_qdrant_client
        self.client = get_qdrant_client(qdrant_path)

        self.embedder = TextEmbedder()
        if bm25_retriever:
            self.bm25_retriever = bm25_retriever
        else:
            self.bm25_retriever = BM25Retriever(qdrant_path=qdrant_path, collection_name=collection_name)

    def close(self):
        # Global client is managed at the application process level, do not close here
        pass

    def _get_vector_candidates(self, query: str, top_k: int = 20, client: Optional[QdrantClient] = None, repo: str = None) -> List[Dict[str, Any]]:
        """
        Retrieves top_k candidate chunks using Dense Vector Search (Qdrant).
        """
        query_vec = self.embedder.embed_text(query)
        if client is None:
            client = self.client

        from qdrant_client.models import Filter, FieldCondition, MatchValue
        query_filter = Filter(must=[FieldCondition(key="repo", match=MatchValue(value=repo))]) if repo else None

        try:
            results = client.search(collection_name=self.collection_name, query_vector=query_vec, query_filter=query_filter, limit=top_k)
        except Exception:
            results = client.query_points(collection_name=self.collection_name, query=query_vec, query_filter=query_filter, limit=top_k).points


        candidates = []
        for point in results:
            payload = point.payload or {}
            candidates.append({
                "doc_id": payload.get("doc_id", "N/A"),
                "doc_type": payload.get("doc_type", "N/A"),
                "text": payload.get("text", ""),
                "vector_score": float(point.score),
                "payload": payload
            })
        return candidates

    def _reciprocal_rank_fusion(
        self,
        vector_candidates: List[Dict[str, Any]],
        bm25_candidates: List[Dict[str, Any]],
        k: int = 60
    ) -> List[Dict[str, Any]]:
        """
        Merges vector and BM25 candidate lists using Reciprocal Rank Fusion (RRF).
        RRF Score(d) = 1/(k + rank_vector) + 1/(k + rank_bm25)
        """
        fused_scores: Dict[str, Dict[str, Any]] = {}

        # Process Vector search rankings
        for rank, cand in enumerate(vector_candidates, 1):
            doc_key = f"{cand['doc_id']}||{cand['text'][:50]}"
            if doc_key not in fused_scores:
                fused_scores[doc_key] = {
                    "doc_id": cand["doc_id"],
                    "doc_type": cand["doc_type"],
                    "text": cand["text"],
                    "vector_score": cand["vector_score"],
                    "bm25_score": 0.0,
                    "vector_rank": rank,
                    "bm25_rank": None,
                    "rrf_score": 0.0,
                    "payload": cand.get("payload", {})
                }
            else:
                fused_scores[doc_key]["vector_rank"] = rank
                fused_scores[doc_key]["vector_score"] = cand["vector_score"]

            fused_scores[doc_key]["rrf_score"] += 1.0 / (k + rank)

        # Process BM25 search rankings
        for rank, cand in enumerate(bm25_candidates, 1):
            doc_key = f"{cand['doc_id']}||{cand['text'][:50]}"
            if doc_key not in fused_scores:
                fused_scores[doc_key] = {
                    "doc_id": cand["doc_id"],
                    "doc_type": cand["doc_type"],
                    "text": cand["text"],
                    "vector_score": 0.0,
                    "bm25_score": cand["bm25_score"],
                    "vector_rank": None,
                    "bm25_rank": rank,
                    "rrf_score": 0.0,
                    "payload": cand.get("payload", {})
                }
            else:
                fused_scores[doc_key]["bm25_rank"] = rank
                fused_scores[doc_key]["bm25_score"] = cand["bm25_score"]

            fused_scores[doc_key]["rrf_score"] += 1.0 / (k + rank)

        # Convert to list and sort by RRF score descending
        fused_list = list(fused_scores.values())
        fused_list.sort(key=lambda x: x["rrf_score"], reverse=True)
        return fused_list

    def _min_max_fusion(
        self,
        vector_candidates: List[Dict[str, Any]],
        bm25_candidates: List[Dict[str, Any]],
        alpha: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Merges vector and BM25 candidates by min-max scaling raw scores to [0, 1]
        and performing weighted summation: Score = alpha * V_norm + (1 - alpha) * BM25_norm
        """
        fused_map: Dict[str, Dict[str, Any]] = {}

        # 1. Normalize Vector scores
        v_scores = [c["vector_score"] for c in vector_candidates]
        v_min, v_max = (min(v_scores), max(v_scores)) if v_scores else (0.0, 1.0)
        v_denom = (v_max - v_min) if (v_max - v_min) > 0 else 1.0

        for cand in vector_candidates:
            doc_key = f"{cand['doc_id']}||{cand['text'][:50]}"
            norm_v = (cand["vector_score"] - v_min) / v_denom
            fused_map[doc_key] = {
                "doc_id": cand["doc_id"],
                "doc_type": cand["doc_type"],
                "text": cand["text"],
                "vector_score": cand["vector_score"],
                "bm25_score": 0.0,
                "vector_norm": norm_v,
                "bm25_norm": 0.0,
                "hybrid_score": alpha * norm_v,
                "payload": cand.get("payload", {})
            }

        # 2. Normalize BM25 scores
        b_scores = [c["bm25_score"] for c in bm25_candidates]
        b_min, b_max = (min(b_scores), max(b_scores)) if b_scores else (0.0, 1.0)
        b_denom = (b_max - b_min) if (b_max - b_min) > 0 else 1.0

        for cand in bm25_candidates:
            doc_key = f"{cand['doc_id']}||{cand['text'][:50]}"
            norm_b = (cand["bm25_score"] - b_min) / b_denom

            if doc_key in fused_map:
                fused_map[doc_key]["bm25_score"] = cand["bm25_score"]
                fused_map[doc_key]["bm25_norm"] = norm_b
                fused_map[doc_key]["hybrid_score"] += (1 - alpha) * norm_b
            else:
                fused_map[doc_key] = {
                    "doc_id": cand["doc_id"],
                    "doc_type": cand["doc_type"],
                    "text": cand["text"],
                    "vector_score": 0.0,
                    "bm25_score": cand["bm25_score"],
                    "vector_norm": 0.0,
                    "bm25_norm": norm_b,
                    "hybrid_score": (1 - alpha) * norm_b,
                    "payload": cand.get("payload", {})
                }

        fused_list = list(fused_map.values())
        fused_list.sort(key=lambda x: x["hybrid_score"], reverse=True)
        return fused_list

    def search(
        self,
        query: str,
        top_k: int = 20,
        candidate_k: int = 20,
        method: str = "rrf",
        alpha: float = 0.5,
        client: Optional[QdrantClient] = None,
        repo: str = None
    ) -> List[Dict[str, Any]]:
        """
        Main entry point for Hybrid Retrieval.
        Retrieves candidate_k vector matches and candidate_k BM25 matches,
        combines them using method ('rrf' or 'minmax'), and returns top_k fused results.
        """
        vector_candidates = self._get_vector_candidates(query=query, top_k=candidate_k, client=client, repo=repo)
        bm25_candidates = self.bm25_retriever.search(query=query, top_k=candidate_k, repo=repo)

        if method.lower() == "minmax":
            fused = self._min_max_fusion(vector_candidates, bm25_candidates, alpha=alpha)
        else:
            fused = self._reciprocal_rank_fusion(vector_candidates, bm25_candidates, k=60)

        return fused[:top_k]
