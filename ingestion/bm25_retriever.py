import os
import re
import sys
from typing import List, Dict, Any
from qdrant_client import QdrantClient
from rank_bm25 import BM25Okapi

class BM25Retriever:
    """
    BM25 Lexical Keyword Search Engine.
    Loads chunk corpus from Qdrant/SQLite, tokenizes text into word tokens,
    builds in-memory BM25Okapi index, and retrieves exact keyword matching chunks.
    """
    def __init__(
        self,
        qdrant_path: str = "data/qdrant_db",
        collection_name: str = "repomind_collection"
    ):
        self.qdrant_path = qdrant_path
        self.collection_name = collection_name
        self.corpus_chunks: List[Dict[str, Any]] = []
        self.tokenized_corpus: List[List[str]] = []
        self.bm25: BM25Okapi = None

        self._build_index()

    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenizes input text into lowercase word tokens.
        Handles punctuation, identifiers, and special characters cleanly.
        """
        if not text:
            return []
        # Extracts alphanumeric word tokens and converts to lowercase
        return re.findall(r'\w+', text.lower())

    def _build_index(self):
        """
        Scrolls all stored points from Qdrant local database,
        extracts text payloads, tokenizes them, and initializes BM25Okapi.
        """
        from ingestion.qdrant_store import get_qdrant_client
        client = get_qdrant_client(self.qdrant_path)
        
        # Retrieve up to 10,000 points from Qdrant collection
        try:
            points, _ = client.scroll(
                collection_name=self.collection_name,
                limit=10000,
                with_payload=True,
                with_vectors=False
            )
        except Exception as e:
            raise RuntimeError(f"Failed to scroll points from Qdrant collection '{self.collection_name}': {e}")

        self.corpus_chunks = []
        self.tokenized_corpus = []

        for point in points:
            payload = point.payload or {}
            text = payload.get("text", "")
            doc_id = payload.get("doc_id", "N/A")
            doc_type = payload.get("doc_type", "N/A")

            tokenized_text = self._tokenize(text)
            
            self.corpus_chunks.append({
                "point_id": point.id,
                "doc_id": doc_id,
                "doc_type": doc_type,
                "text": text,
                "payload": payload
            })
            self.tokenized_corpus.append(tokenized_text)

        if self.tokenized_corpus:
            self.bm25 = BM25Okapi(self.tokenized_corpus)

    def search(self, query: str, top_k: int = 20, repo: str = None) -> List[Dict[str, Any]]:
        """
        Searches the BM25 index for the input query string.
        Returns top_k most relevant candidate chunks with their BM25 relevance scores.
        """
        if not self.bm25 or not self.corpus_chunks:
            return []

        tokenized_query = self._tokenize(query)
        if not tokenized_query:
            return []

        # Get raw BM25 relevance scores for all documents
        scores = self.bm25.get_scores(tokenized_query)

        # Pair candidates with scores and sort descending
        scored_candidates = []
        for idx, score in enumerate(scores):
            if score > 0:  # Only consider documents with non-zero BM25 match
                candidate = dict(self.corpus_chunks[idx])
                if repo and candidate.get("payload", {}).get("repo") != repo:
                    continue
                candidate["bm25_score"] = float(score)
                scored_candidates.append(candidate)

        # Sort by BM25 score descending
        scored_candidates.sort(key=lambda x: x["bm25_score"], reverse=True)

        return scored_candidates[:top_k]
