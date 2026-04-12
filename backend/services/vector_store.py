import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

_BACKEND_DIR = Path(__file__).resolve().parent.parent
CHROMA_PERSIST_DIR = str(_BACKEND_DIR.parent / "chroma_db")

COLLECTION_NAME = "pdf_chunks"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

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
    embedder = _get_embedder()
    return embedder.encode(texts, show_progress_bar=False).tolist()


def store_chunks(chunks: List[Dict], batch_size: int = 24) -> int:
    if not chunks:
        return 0

    collection = _get_collection()
    total_stored = 0
    safe_batch_size = max(20, min(30, int(batch_size)))

    for start in range(0, len(chunks), safe_batch_size):
        batch = chunks[start : start + safe_batch_size]
        texts = [chunk["text"] for chunk in batch]
        embeddings = embed_texts(texts)
        ids = [
            f"{chunk.get('document_id', 'unknown')}_{chunk.get('filename', 'unknown')}_chunk_{chunk.get('chunk_index', start + i)}"
            for i, chunk in enumerate(batch)
        ]
        metadatas = [
            {
                "filename": chunk.get("filename", "unknown"),
                "chunk_index": chunk.get("chunk_index", start + i),
                "document_id": chunk.get("document_id", ""),
                "heading": chunk.get("heading", ""),
            }
            for i, chunk in enumerate(batch)
        ]
        collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )
        total_stored += len(batch)

    return total_stored


def search_chunks(query: str, top_k: int = 8, document_id: Optional[str] = None) -> List[Dict]:
    collection = _get_collection()
    if collection.count() == 0:
        return []

    query_embedding = embed_texts([query])[0]
    semantic_limit = min(max(top_k * 4, 24), collection.count())
    query_kwargs = {
        "query_embeddings": [query_embedding],
        "n_results": semantic_limit,
        "include": ["documents", "metadatas", "distances"],
    }
    if document_id:
        query_kwargs["where"] = {"document_id": document_id}

    results = collection.query(**query_kwargs)
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    ranked = []
    for doc, meta, dist in zip(documents, metadatas, distances):
        semantic_score = 1.0 - float(dist)
        keyword_score = _keyword_overlap_score(query, doc, meta.get("heading", ""))
        heading_bonus = _heading_relevance_bonus(query, meta.get("heading", ""))
        hybrid_score = (semantic_score * 0.7) + (keyword_score * 0.25) + heading_bonus
        ranked.append(
            {
                "text": doc,
                "filename": meta.get("filename", "unknown"),
                "chunk_index": meta.get("chunk_index", -1),
                "document_id": meta.get("document_id"),
                "heading": meta.get("heading", ""),
                "distance": round(float(dist), 4),
                "score": round(hybrid_score, 4),
            }
        )

    ranked.sort(key=lambda item: item["score"], reverse=True)
    return ranked[: max(1, top_k)]


_doc_metadata: Dict[str, Dict] = {}


def add_document_metadata(
    document_id: str,
    filename: str,
    chunks: int,
    pages: int = 0,
    upload_date: Optional[str] = None,
):
    _doc_metadata[document_id] = {
        "filename": filename,
        "chunks": chunks,
        "pages": int(pages or 0),
        "upload_date": upload_date or _now_iso(),
    }


def list_documents():
    return [
        {
            "document_id": doc_id,
            "filename": meta["filename"],
            "chunks": meta["chunks"],
            "pages": meta.get("pages", 0),
            "upload_date": meta.get("upload_date"),
        }
        for doc_id, meta in _doc_metadata.items()
    ]


def get_document_metadata(document_id: Optional[str]) -> Optional[Dict]:
    if not document_id:
        return None
    metadata = _doc_metadata.get(document_id)
    if not metadata:
        return None
    return {"document_id": document_id, **metadata}


def get_latest_document_metadata() -> Optional[Dict]:
    if not _doc_metadata:
        return None
    latest_id = max(
        _doc_metadata.keys(),
        key=lambda doc_id: _doc_metadata[doc_id].get("upload_date") or "",
    )
    return {"document_id": latest_id, **_doc_metadata[latest_id]}


def delete_document(document_id: str):
    _doc_metadata.pop(document_id, None)
    collection = _get_collection()
    results = collection.get(include=["metadatas"])
    ids_to_delete = [
        chunk_id
        for chunk_id, meta in zip(results["ids"], results["metadatas"])
        if meta.get("document_id") == document_id
    ]
    if ids_to_delete:
        collection.delete(ids=ids_to_delete)


def _keyword_overlap_score(query: str, text: str, heading: str = "") -> float:
    query_terms = set(_tokenize(query))
    if not query_terms:
        return 0.0

    content_terms = set(_tokenize(text))
    heading_terms = set(_tokenize(heading))
    overlap_content = len(query_terms.intersection(content_terms))
    overlap_heading = len(query_terms.intersection(heading_terms))
    overlap_total = overlap_content + (overlap_heading * 1.4)
    return min(1.0, overlap_total / max(1, len(query_terms)))


def _heading_relevance_bonus(query: str, heading: str) -> float:
    if not heading:
        return 0.0
    query_terms = set(_tokenize(query))
    heading_terms = set(_tokenize(heading))
    if not query_terms or not heading_terms:
        return 0.0
    if query_terms.intersection(heading_terms):
        return 0.05
    return 0.0


def _tokenize(text: str) -> List[str]:
    return [token for token in re.findall(r"[a-zA-Z0-9]{2,}", (text or "").lower())]


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()
