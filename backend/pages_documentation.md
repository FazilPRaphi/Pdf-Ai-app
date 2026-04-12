# Backend Pages Documentation

This document provides an overview of the main backend pages (modules) that have been created and contain code in the project. Each section includes a brief description and the main code for reference.

---

## 1. app.py
**Location:** backend/app.py

This is the main FastAPI application file. It sets up the API, configures CORS, handles PDF uploads, and uses utility functions to extract text from PDFs.

```python
import os
import shutil
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

from utils.pdf_utils import extract_text_from_pdf

app = FastAPI(title="Multimodal RAG Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.get("/")
def home():
    return {
        "message": "Backend running successfully 🚀"
    }

@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        # Only allow PDF
        if not file.filename.lower().endswith(".pdf"):
            return {
                "success": False,
                "message": "Only PDF files are allowed."
            }

        file_path = os.path.join(UPLOAD_DIR, file.filename)

        # Save uploaded file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Extract text
        result = extract_text_from_pdf(file_path)

        if not result["success"]:
            return result

        return {
            "success": True,
            "filename": file.filename,
            "pages": result["pages"],
            "text_preview": result["text"][:2000]  # first 2k chars
        }

    except Exception as e:
        return {
            "success": False,
            "message": str(e)
        }
```

---

## 2. services/rag_service.py
**Location:** backend/services/rag_service.py

Handles the Retrieval-Augmented Generation (RAG) pipeline, including prompt building and querying the Groq LLM.

```python
import os
from typing import Dict, Optional

from groq import Groq
from dotenv import load_dotenv

from services.vector_store import search_chunks

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

def _build_prompt(question: str, context_chunks: list) -> str:
    """Construct a grounded-answering prompt from retrieved chunks."""
    context_text = "\n\n---\n\n".join(
        [f"[Source {i+1} | {c['filename']}]\n{c['text']}"
         for i, c in enumerate(context_chunks)]
    )

    return f"""You are a precise document-answering assistant.
Use ONLY the context below to answer the question.
If the answer is not present in the context, reply exactly with: "Not found in document."

=== CONTEXT ===
{context_text}

=== QUESTION ===
{question}

=== ANSWER ==="""

def answer_question(question: str, top_k: int = 5) -> Dict:
    """
    Full RAG pipeline:
      1. Retrieve top_k relevant chunks from ChromaDB.
      2. Build a grounded prompt.
      3. Query Groq LLM.
      4. Return structured response.
    """
    # Retrieve
    chunks = search_chunks(query=question, top_k=top_k)

    if not chunks:
        return {}
```

---

## 3. services/vector_store.py
**Location:** backend/services/vector_store.py

Manages the vector store using ChromaDB and Sentence Transformers for embedding and retrieval.

```python
import os
from pathlib import Path
from typing import List, Dict, Optional

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Paths — resolve relative to this file so the server can be launched from
# any working directory.
# ---------------------------------------------------------------------------
_BACKEND_DIR = Path(__file__).resolve().parent.parent
CHROMA_PERSIST_DIR = str(_BACKEND_DIR.parent / "chroma_db")

COLLECTION_NAME = "pdf_chunks"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# ---------------------------------------------------------------------------
# Singletons — initialised once per process lifetime
# ---------------------------------------------------------------------------
_embedder: Optional[SentenceTransformer] = None
_chroma_client: Optional[chromadb.PersistentClient] = None
_collection = None

def _get_embedder() -> SentenceTransformer:  # type: ignore[return]
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBEDDING_MODEL)
    return _embedder

def _get_collection():
    global _chroma_client, _collection
    if _collection is None:
        os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)
        _chroma_client = chromadb.PersistentClient(
            path=CHROMA_PERSIST_DIR,
            settings=Settings(anonymized_telemetry=False),
        )
        _collection = _chroma_client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
    return _collection

def embed_texts(texts: List[str]) -> List[List[float]]:
    """Return embeddings for a list of strings."""
    embedder = _get_embedder()
    return embedder.encode(texts, show_progress_bar=False).tolist()

def store_chunks(chunks: List[Dict]) -> int:
    """
    ...
```

---

## 4. utils/chunker.py
**Location:** backend/utils/chunker.py

Splits extracted PDF text into overlapping semantic chunks for storage and retrieval.

```python
import re
from typing import List, Dict

def chunk_text(
    text: str,
    filename: str,
    chunk_size: int = 700,
    overlap: int = 120,
) -> List[Dict]:
    """
    Split extracted PDF text into overlapping semantic chunks.

    Strategy:
    1. Split on paragraph boundaries (double newlines) first.
    2. If a paragraph exceeds chunk_size, split further at sentence boundaries.
    3. Merge small paragraphs until chunk_size is approached.
    4. Apply overlap by prepending the tail of the previous chunk.

    Returns a list of dicts with keys: text, filename, chunk_index.
    """

    # Normalise line endings and collapse excessive blank lines
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]

    # --- Step 1: break oversized paragraphs at sentence boundaries ---
    sentence_end = re.compile(r"(?<=[.!?])\s+")
    raw_segments: List[str] = []

    for para in paragraphs:
        if len(para) <= chunk_size:
            raw_segments.append(para)
        else:
            sentences = sentence_end.split(para)
            current = ""
            for sent in sentences:
                if len(current) + len(sent) + 1 <= chunk_size:
                    current = (current + " " + sent).strip()
                else:
                    if current:
                        raw_segments.append(current)
                    current = sent
            if current:
                raw_segments.append(current)

    # --- Step 2: merge small segments into chunks approaching chunk_size ---
    merged: List[str] = []
    buffer = ""

    for seg in raw_segments:
        if not buffer:
            buffer = seg
        elif len(buffer) + len(seg) + 1 <= chunk_size:
            buffer = buffer + " " + seg
        else:
            merged.append(buffer)
            buffer = seg

    if buffer:
        merged.append(buffer)
    # ...
```

---

## 5. utils/pdf_utils.py
**Location:** backend/utils/pdf_utils.py

Extracts text and page count from PDF files using PyMuPDF (fitz).

```python
import fitz  

def extract_text_from_pdf(file_path: str):
    """
    Extract text from PDF and return:
    - full text
    - total pages
    """
    text = ""
    page_count = 0

    try:
        doc = fitz.open(file_path)
        page_count = len(doc)

        for page in doc:
            text += page.get_text() + "\n"

        doc.close()

        return {
            "success": True,
            "text": text.strip(),
            "pages": page_count
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }
```

---

> **Note:** This documentation covers only the backend pages with implemented code. For additional modules or updates, please regenerate this file.
