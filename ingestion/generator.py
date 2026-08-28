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
    def __init__(self, model_name: str = "gemini-3.6-flash"):
        self.model_name = model_name
        self.has_key = False
        self.client = None
        
        api_key = os.getenv("GEMINI_API_KEY")
        if api_key:
            api_key = api_key.strip().strip("'").strip('"').replace(" ", "").replace("\t", "")
            
        print(f"[RUNNING] Initializing Gemini client with model '{self.model_name}'...")
        try:
            if api_key and api_key != "your_key_here" and len(api_key) > 5:
                self.client = genai.Client(api_key=api_key)
                self.has_key = True
                print("[SUCCESS] Gemini client initialized with API Key.")
            else:
                print("[INFO] GEMINI_API_KEY environment variable is not configured.")
                self.has_key = False
                self.client = None
        except Exception as e:
            print(f"[WARNING] Failed to initialize genai.Client: {e}")
            self.has_key = False
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

        summary_blocks = []
        for cand in candidates[:3]:
            t = cand.get('title')
            if not t or t == "N/A":
                t = cand.get('doc_id') or 'Document'
            doc_type = cand.get('doc_type', 'doc')
            snippet = cand.get('text', '').strip()[:250]
            summary_blocks.append(f"• **{t}** (`{doc_type}`):\n{snippet}...")
        context_summary = "\n\n".join(summary_blocks)

        if not self.has_key or not self.client:
            api_info = "⚠️ **Action Required**: `GEMINI_API_KEY` is not configured in Vercel. Please add `GEMINI_API_KEY` in **Vercel Project Settings -> Environment Variables** and click **Redeploy**."
            return RAGResponseSchema(
                answer=f"{api_info}\n\n### Top Retrieved Context Chunks\n\n{context_summary}",
                cited_chunk_ids=[c.get("chunk_id", "") for c in candidates[:3] if c.get("chunk_id")]
            )

        prompt = self.construct_prompt(query, candidates)
        json_prompt = prompt + '\n\nIMPORTANT: Return a JSON object with two fields:\n- "answer": string containing your concise factual answer\n- "cited_chunk_ids": array of string IDs cited'

        config_plain = types.GenerateContentConfig(
            temperature=0.0
        )

        models_to_try = [
            "gemini-3.6-flash",
            "gemini-2.5-flash",
            "gemini-flash-latest"
        ]
        last_error = None

        for m_name in models_to_try:
            # Attempt 1: Fast plain text JSON generation
            try:
                print(f"[RUNNING] Generating answer using Gemini ({m_name})...")
                response = self.client.models.generate_content(
                    model=m_name,
                    contents=json_prompt,
                    config=config_plain
                )
                txt = response.text.strip()
                if "```json" in txt:
                    txt = txt.split("```json")[1].split("```")[0].strip()
                elif "```" in txt:
                    txt = txt.split("```")[1].split("```")[0].strip()
                
                try:
                    data = json.loads(txt)
                    ans = data.get("answer") or txt
                    c_ids = data.get("cited_chunk_ids") or []
                except Exception:
                    ans = txt
                    c_ids = []
                    
                return RAGResponseSchema(
                    answer=ans,
                    cited_chunk_ids=c_ids
                )
            except Exception as e:
                last_error = e
                print(f"[WARNING] Model '{m_name}' failed: {e}.")

        # Formulate clean, user-friendly fallback context summary
        err_msg = str(last_error)
        clean_err = err_msg.replace("'", "").replace('"', '').strip()
        api_info = f"⚠️ **Gemini LLM Notice** ({clean_err[:120]}...): Please verify that `GEMINI_API_KEY` is added to your Vercel Project Environment Variables and redeployed."

        summary_blocks = []
        for cand in candidates[:3]:
            t = cand.get('title')
            if not t or t == "N/A":
                t = cand.get('doc_id') or 'Document'
            doc_type = cand.get('doc_type', 'doc')
            snippet = cand.get('text', '').strip()[:250]
            summary_blocks.append(f"• **{t}** (`{doc_type}`):\n{snippet}...")

        context_summary = "\n\n".join(summary_blocks)

        return RAGResponseSchema(
            answer=f"{api_info}\n\n### Top Retrieved Context Chunks\n\n{context_summary}",
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
