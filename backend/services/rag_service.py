import os
import re
from typing import Dict, Optional

from groq import Groq
from dotenv import load_dotenv

from services.vector_store import (
    get_document_metadata,
    get_latest_document_metadata,
    search_chunks,
)

# Load GROQ_API_KEY from the root .env (two levels up from this file)
_ENV_PATH = os.path.join(os.path.dirname(__file__), "..", "..", ".env")
load_dotenv(dotenv_path=_ENV_PATH)

GROQ_MODEL = "llama-3.3-70b-versatile"

_groq_client: Optional[Groq] = None


def _get_groq_client() -> Groq:
    global _groq_client
    if _groq_client is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise EnvironmentError("GROQ_API_KEY is not set in the environment.")
        _groq_client = Groq(api_key=api_key)
    return _groq_client


def _build_prompt(question: str, context_chunks: list, is_broad_question: bool) -> str:
    """Construct a grounded prompt with better behavior for broad questions."""
    context_text = "\n\n---\n\n".join(
        [f"[Source {i+1} | {c['filename']} | heading: {c.get('heading', '') or 'n/a'}]\n{c['text']}"
         for i, c in enumerate(context_chunks)]
    )

    broad_instruction = (
        "The question is broad. Provide a concise summary of what the document is about, then key points."
        if is_broad_question
        else "Answer the question directly and concisely."
    )

    return f"""You are a grounded document assistant.
Answer naturally using only the provided context.
{broad_instruction}

Rules:
1) Prioritize exact evidence from context.
2) If context is partial but useful, give the best grounded answer and mention uncertainty briefly.
3) Only reply with "Not found in document." if the context truly has no relevant evidence.
4) Do not invent facts.

=== CONTEXT ===
{context_text}

=== QUESTION ===
{question}

=== ANSWER ==="""


def answer_question(question: str, document_id: str = None, top_k: int = 9) -> Dict:
    """
    Full RAG pipeline:
      1. Retrieve top_k relevant chunks from ChromaDB (optionally filtered by document_id).
      2. Build a grounded prompt.
      3. Query Groq LLM.
      4. Return structured response with sources.
    """
    metadata_answer = _maybe_answer_from_metadata(question=question, document_id=document_id)
    if metadata_answer is not None:
        return {
            "success": True,
            "question": question,
            "answer": metadata_answer,
            "sources": [],
        }

    broad_question = _is_broad_question(question)
    retrieval_query = _build_retrieval_query(question, broad_question)
    chunks = search_chunks(query=retrieval_query, top_k=top_k, document_id=document_id)
    if not chunks:
        return {
            "success": False,
            "question": question,
            "answer": "No relevant documents found.",
            "sources": [],
        }
    prompt = _build_prompt(question, chunks, broad_question)
    client = _get_groq_client()
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=1024,
    )
    answer = response.choices[0].message.content.strip()
    sources = [
        {
            "filename": c["filename"],
            "chunk_index": c["chunk_index"],
            "document_id": c.get("document_id"),
            "heading": c.get("heading", ""),
            "text": c["text"][:200] + ("..." if len(c["text"]) > 200 else ""),
        }
        for c in chunks
    ]
    return {
        "success": True,
        "question": question,
        "answer": answer,
        "sources": sources,
    }

def summarize_document(document_id: str, top_k: int = 8) -> Dict:
    """
    Summarize a document by fetching top representative chunks and asking Groq for summary, topics, and entities.
    """
    # Use the first N chunks as a simple heuristic for summary
    chunks = search_chunks(query="summary", top_k=top_k, document_id=document_id)
    if not chunks:
        return {"success": False, "summary": "No content found for this document."}
    context_text = "\n\n---\n\n".join(
        [f"[Chunk {c['chunk_index']}]\n{c['text']}" for c in chunks]
    )
    prompt = (
        f"You are a document summarization assistant.\n"
        f"Given the following document chunks, provide:\n"
        f"- A concise summary\n- Key topics\n- Important entities and dates (if any)\n\n=== CHUNKS ===\n{context_text}\n\n=== SUMMARY ==="
    )
    client = _get_groq_client()
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=512,
    )
    summary = response.choices[0].message.content.strip()
    return {"success": True, "summary": summary}


def _maybe_answer_from_metadata(question: str, document_id: Optional[str]) -> Optional[str]:
    question_lower = (question or "").strip().lower()
    if not question_lower:
        return None

    metadata = get_document_metadata(document_id) or get_latest_document_metadata()
    if not metadata:
        return None

    page_patterns = [
        r"\bhow many pages\b",
        r"\bpage count\b",
        r"\bnumber of pages\b",
        r"\btotal pages\b",
    ]
    filename_patterns = [
        r"\bfile name\b",
        r"\bfilename\b",
        r"\bname of (the )?(file|document)\b",
    ]
    active_doc_patterns = [
        r"\bactive document\b",
        r"\bcurrent document\b",
        r"\bwhich document\b",
    ]

    if _matches_any(question_lower, page_patterns):
        pages = metadata.get("pages", 0)
        if pages:
            return f"This PDF has {pages} pages."
        return "Page count is not available for this document."

    if _matches_any(question_lower, filename_patterns):
        return f"The file name is {metadata.get('filename', 'unknown')}."

    if _matches_any(question_lower, active_doc_patterns):
        return f"Current active document is {metadata.get('filename', 'unknown')}."

    return None


def _matches_any(text: str, patterns: list) -> bool:
    return any(re.search(pattern, text) for pattern in patterns)


def _is_broad_question(question: str) -> bool:
    q = (question or "").strip().lower()
    broad_phrases = [
        "what is this",
        "summarize",
        "summary",
        "what is the file about",
        "explain this document",
        "give an overview",
        "key points",
        "skills",
        "requirements",
    ]
    return any(phrase in q for phrase in broad_phrases)


def _build_retrieval_query(question: str, is_broad_question: bool) -> str:
    if not is_broad_question:
        return question
    return f"{question} summary overview key points important details"
