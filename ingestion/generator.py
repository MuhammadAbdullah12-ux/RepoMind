import os
import json
from typing import List, Dict, Any
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

# Ensure env variables are loaded
load_dotenv()

class RAGResponseSchema(BaseModel):
    """
    Structured response schema for the repository question answering system.
    """
    answer: str = Field(description="A factual, concise answer generated using only the provided context chunks.")
    cited_chunk_ids: List[str] = Field(description="List of unique chunk IDs of the chunks that support the claims in the answer.")

class GeminiGenerator:
    """
    Gemini Answer Generator for RAG.
    Takes retrieved/reranked candidate document chunks and queries the Gemini LLM
    to produce a structured answer citing exact sources.
    """
    def __init__(self, model_name: str = "gemini-2.5-flash"):
        self.model_name = model_name
        
        api_key = os.getenv("GEMINI_API_KEY")
        if api_key:
            api_key = api_key.strip().strip("'").strip('"')
            
        if not api_key or api_key == "your_key_here":
            print("[WARNING] GEMINI_API_KEY is not configured in environment or .env. Gemini calls will fail.")
            
        print(f"[RUNNING] Initializing Gemini client with model '{self.model_name}'...")
        try:
            if api_key and api_key != "your_key_here":
                self.client = genai.Client(api_key=api_key)
            else:
                self.client = genai.Client()
            print("[SUCCESS] Gemini client initialized successfully.")
        except Exception as e:
            print(f"[WARNING] Failed to initialize genai.Client: {e}")
            self.client = None

    def construct_prompt(self, query: str, candidates: List[Dict[str, Any]]) -> str:
        """
        Formats the query and retrieved context chunks into a structured prompt.
        """
        chunks_str_list = []
        for idx, cand in enumerate(candidates, 1):
            chunk_id = cand.get("chunk_id") or cand.get("id") or "N/A"
            doc_type = cand.get("doc_type") or cand.get("payload", {}).get("doc_type", "N/A")
            title = cand.get("title") or cand.get("payload", {}).get("title", "N/A")
            url = cand.get("url") or cand.get("payload", {}).get("url", "N/A")
            text = cand.get("text") or cand.get("content") or cand.get("payload", {}).get("text", "")
            
            chunk_format = (
                f"--- Chunk #{idx} (ID: {chunk_id}) ---\n"
                f"Document Type: {doc_type}\n"
                f"Document Title: {title}\n"
                f"URL: {url}\n"
                f"Content Snippet:\n{text.strip()}\n"
            )
            chunks_str_list.append(chunk_format)
            
        context_block = "\n".join(chunks_str_list)
        
        prompt = (
            "You are a precise repository search assistant. Answer the user's question using ONLY the retrieved document chunks provided below.\n"
            "Do not make assumptions, extrapolate, or use outside knowledge. If the provided context does not contain enough information to answer the question, state that clearly.\n\n"
            "Here are the retrieved document chunks:\n"
            "=========================================\n"
            f"{context_block}\n"
            "=========================================\n\n"
            f"User Question: {query}\n"
        )
        return prompt

    def generate(self, query: str, candidates: List[Dict[str, Any]]) -> RAGResponseSchema:
        """
        Calls Gemini to generate a structured answer based on candidate chunks.
        """
        if not candidates:
            return RAGResponseSchema(
                answer="No context chunks were provided to answer the question.",
                cited_chunk_ids=[]
            )

        if not self.client:
            summary = "\n".join([f"• {c.get('title', 'Doc')}: {c.get('text', '')[:120]}..." for c in candidates[:3]])
            return RAGResponseSchema(
                answer=f"Gemini API client is unavailable. Top retrieved context chunks:\n\n{summary}",
                cited_chunk_ids=[c.get("chunk_id", "") for c in candidates[:3] if c.get("chunk_id")]
            )

        prompt = self.construct_prompt(query, candidates)
        
        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=RAGResponseSchema,
            temperature=0.0
        )
        
        # Try candidate model names in order of reliability
        models_to_try = [self.model_name, "gemini-2.5-flash", "gemini-1.5-flash", "gemini-flash-latest"]
        last_error = None

        for m_name in models_to_try:
            try:
                print(f"[RUNNING] Generating answer using Gemini ({m_name})...")
                response = self.client.models.generate_content(
                    model=m_name,
                    contents=prompt,
                    config=config
                )
                response_data = json.loads(response.text)
                validated_response = RAGResponseSchema(**response_data)
                print("[SUCCESS] Answer generated and verified.")
                return validated_response
            except Exception as e:
                last_error = e
                print(f"[WARNING] Model '{m_name}' failed: {e}. Trying fallback...")
                
        # If all LLM attempts fail, return a helpful summary from retrieved chunks instead of crashing
        summary = "\n".join([f"• [{c.get('title', 'Doc')}] {c.get('text', '')[:120]}..." for c in candidates[:3]])
        return RAGResponseSchema(
            answer=f"Could not reach Gemini LLM ({last_error}). Retrieved context summary:\n\n{summary}",
            cited_chunk_ids=[c.get("chunk_id", "") for c in candidates[:3] if c.get("chunk_id")]
        )

if __name__ == "__main__":
    # Quick module test/demonstration
    import sys
    
    # Simple test data
    test_query = "How do I upgrade the mcp dependency?"
    test_chunks = [
        {
            "chunk_id": "pr-16018-chunk-0",
            "doc_type": "pr",
            "title": "Bump mcp from 1.26.0 to 1.28.1",
            "url": "https://github.com/fastapi/fastapi/pull/16018",
            "text": "Bumps mcp version dependency requirement in pyproject.toml from 1.26.0 to 1.28.1."
        },
        {
            "chunk_id": "readme-fastapi-chunk-12",
            "doc_type": "readme",
            "title": "README.md",
            "url": "https://github.com/fastapi/fastapi/blob/main/README.md",
            "text": "FastAPI is a modern, fast (high-performance), web framework for building APIs with Python 3.8+."
        }
    ]
    
    # Attempt to load generator
    try:
        generator = GeminiGenerator()
        print("\n--- Test Prompt Construction ---")
        prompt = generator.construct_prompt(test_query, test_chunks)
        print(prompt)
        
        # Only run generation if key is present and not the default placeholder
        api_key = os.getenv("GEMINI_API_KEY")
        if api_key and api_key != "your_key_here":
            print("--- Running Generation (API Call) ---")
            result = generator.generate(test_query, test_chunks)
            print(f"\nAnswer: {result.answer}")
            print(f"Citations: {result.cited_chunk_ids}")
        else:
            print("[INFO] Skipping API call because GEMINI_API_KEY is not configured with a valid key.")
    except Exception as e:
        print(f"[ERROR] Module verification failed: {e}")
