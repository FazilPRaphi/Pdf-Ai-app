from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from typing import Optional
from services import rag_service, vector_store
from utils import pdf_utils, chunker
import os
import shutil
import uuid

router = APIRouter()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@router.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    # Sanitize filename
    filename = os.path.basename(file.filename)
    if not filename.lower().endswith(".pdf"):
        return {"success": False, "message": "Only PDF files are allowed."}
    file_path = os.path.join(UPLOAD_DIR, filename)
    # Avoid duplicate uploads
    if os.path.exists(file_path):
        return {"success": False, "message": "File already exists."}
    # Save file
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    # Extract text
    result = pdf_utils.extract_text_from_pdf(file_path)
    if not result["success"]:
        return result
    # Assign document_id
    document_id = str(uuid.uuid4())
    # Chunk and store
    chunks = chunker.chunk_text(result["text"], filename, document_id=document_id)
    stored = vector_store.store_chunks(chunks)
    # Store metadata
    vector_store.add_document_metadata(document_id, filename, stored)
    return {
        "success": True,
        "filename": filename,
        "document_id": document_id,
        "pages": result["pages"],
        "total_chunks_stored": stored,
    }

@router.post("/ask")
async def ask_question(payload: dict):
    question = payload.get("question")
    document_id = payload.get("document_id")
    if not question:
        raise HTTPException(status_code=400, detail="Missing question.")
    return rag_service.answer_question(question, document_id=document_id)

@router.post("/summarize")
async def summarize(payload: dict):
    document_id = payload.get("document_id")
    if not document_id:
        raise HTTPException(status_code=400, detail="Missing document_id.")
    return rag_service.summarize_document(document_id)

@router.get("/documents")
async def list_documents():
    return vector_store.list_documents()

@router.delete("/documents/{document_id}")
async def delete_document(document_id: str):
    vector_store.delete_document(document_id)
    return {"success": True}
