import math
from typing import List, Dict, Any

_RERANKER_CACHE = {}

class BgeReranker:
    """
    Stage 2 Precision Reranker Wrapper.
    Uses BAAI/bge-reranker-base Cross-Encoder to re-score and re-order
    candidate document chunks retrieved from Stage 1 Vector Search.
    Falls back gracefully when PyTorch / CrossEncoder is unavailable (e.g. Vercel serverless).
    """
    def __init__(self, model_name: str = "BAAI/bge-reranker-base"):
        self.model_name = model_name
        self.use_cross_encoder = False
        
        if model_name not in _RERANKER_CACHE:
            try:
                import torch
                torch.set_num_threads(1)
                from sentence_transformers import CrossEncoder
                print(f"[RUNNING] Loading Reranker model: '{model_name}'...")
                _RERANKER_CACHE[model_name] = CrossEncoder(model_name)
                print(f"[SUCCESS] Reranker model '{model_name}' ready!")
            except Exception as e:
                print(f"[INFO] CrossEncoder unavailable ({e}). Using stage-1 hybrid fallback ordering.")
                _RERANKER_CACHE[model_name] = "lightweight_mode"

        model_ref = _RERANKER_CACHE[model_name]
        if model_ref != "lightweight_mode":
            self.model = model_ref
            self.use_cross_encoder = True

    @staticmethod
    def _logit_to_sigmoid(logit: float) -> float:
        """
        Converts raw cross-encoder logit into a normalized [0, 1] probability score.
        """
        return 1.0 / (1.0 + math.exp(-float(logit)))

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Reranks candidate chunks. Uses CrossEncoder if available, otherwise hybrid fallback.
        """
        if not candidates:
            return []

        if self.use_cross_encoder:
            try:
                pairs = []
                for cand in candidates:
                    chunk_text = cand.get("text") or cand.get("content") or ""
                    pairs.append([query, chunk_text])

                raw_scores = self.model.predict(pairs)

                reranked_results = []
                for cand, score in zip(candidates, raw_scores):
                    score_float = float(score)
                    cand_copy = dict(cand)
                    cand_copy["rerank_score"] = score_float
                    cand_copy["rerank_prob"] = self._logit_to_sigmoid(score_float)
                    reranked_results.append(cand_copy)

                reranked_results.sort(key=lambda x: x["rerank_score"], reverse=True)
                return reranked_results[:top_k]
            except Exception as e:
                print(f"[WARNING] CrossEncoder rerank failed: {e}. Falling back to hybrid score.")

        # Fallback scoring when CrossEncoder is not loaded
        reranked_results = []
        for idx, cand in enumerate(candidates):
            cand_copy = dict(cand)
            score_val = cand.get("rrf_score") or cand.get("vector_score") or (1.0 / (idx + 1))
            cand_copy["rerank_score"] = float(score_val)
            cand_copy["rerank_prob"] = min(1.0, max(0.0, float(score_val)))
            reranked_results.append(cand_copy)

        reranked_results.sort(key=lambda x: x["rerank_score"], reverse=True)
        return reranked_results[:top_k]

