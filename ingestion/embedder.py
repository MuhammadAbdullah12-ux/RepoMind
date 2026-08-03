import os
import sys

_MODEL_CACHE = {}

class TextEmbedder:
    """
    Utility wrapper for text embeddings with automatic fallback for serverless/Vercel environments.
    Tries BAAI/bge-small-en-v1.5 via sentence_transformers locally.
    Falls back to Gemini API or lightweight vector generator on Vercel serverless.
    """
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        self.dimension = 384
        self.use_st = False
        self.use_gemini = False

        if model_name not in _MODEL_CACHE:
            try:
                import torch
                torch.set_num_threads(1)
                from sentence_transformers import SentenceTransformer
                print(f"[RUNNING] Loading local embedding model: {model_name}...")
                _MODEL_CACHE[model_name] = SentenceTransformer(model_name)
                print(f"[SUCCESS] Model loaded successfully.")
            except Exception as e:
                print(f"[INFO] Local SentenceTransformer unavailable ({e}). Using lightweight serverless embedder mode.")
                _MODEL_CACHE[model_name] = "lightweight_mode"

        model_ref = _MODEL_CACHE[model_name]
        if model_ref != "lightweight_mode":
            self.model = model_ref
            self.use_st = True
            try:
                self.dimension = self.model.get_sentence_embedding_dimension()
            except Exception:
                self.dimension = 384
        else:
            self.use_st = False
            gemini_key = os.getenv("GEMINI_API_KEY")
            if gemini_key:
                try:
                    from google import genai
                    self.genai_client = genai.Client(api_key=gemini_key)
                    self.use_gemini = True
                except Exception:
                    self.use_gemini = False

    def _gemini_embed(self, text: str) -> list[float]:
        try:
            res = self.genai_client.models.embed_content(
                model="text-embedding-004",
                contents=text
            )
            vec = res.embedding.values
            if len(vec) > 384:
                vec = vec[:384]
            elif len(vec) < 384:
                vec = vec + [0.0] * (384 - len(vec))
            norm = sum(x*x for x in vec) ** 0.5
            if norm > 0:
                vec = [x / norm for x in vec]
            return vec
        except Exception:
            return self._hash_embed(text)

    def _hash_embed(self, text: str) -> list[float]:
        import hashlib
        import math
        vec = [0.0] * self.dimension
        words = text.lower().split()
        if not words:
            return vec
        for i, word in enumerate(words):
            h = int(hashlib.md5(word.encode("utf-8")).hexdigest(), 16)
            idx = h % self.dimension
            sign = 1.0 if ((h >> 4) & 1) == 1 else -1.0
            vec[idx] += sign
        norm = math.sqrt(sum(x * x for x in vec))
        if norm > 0:
            vec = [x / norm for x in vec]
        return vec

    def embed_text(self, text: str) -> list[float]:
        """
        Encodes a single text string into a normalized list of floats.
        """
        if not text or not text.strip():
            return [0.0] * self.dimension
        if self.use_st:
            try:
                embedding = self.model.encode(text, convert_to_numpy=False, normalize_embeddings=True)
                return list(embedding)
            except Exception:
                pass
        if self.use_gemini:
            return self._gemini_embed(text)
        return self._hash_embed(text)

    def embed_chunks(self, chunks: list[str]) -> list[list[float]]:
        """
        Encodes a batch list of string chunks into normalized vectors.
        """
        if not chunks:
            return []
        cleaned_chunks = [c if c.strip() else " " for c in chunks]
        if self.use_st:
            try:
                embeddings = self.model.encode(cleaned_chunks, convert_to_numpy=False, normalize_embeddings=True)
                return [list(emb) for emb in embeddings]
            except Exception:
                pass
        return [self.embed_text(c) for c in cleaned_chunks]

