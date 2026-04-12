from pathlib import Path
import os
import shutil
import tempfile
import uuid
from typing import Optional

from fastapi.concurrency import run_in_threadpool
from fastapi import FastAPI, File, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from services.rag_service import answer_question
from services.vector_store import add_document_metadata, get_document_metadata, store_chunks
from utils.chunker import chunk_text
from utils.pdf_utils import extract_text_from_pdf

app = FastAPI(title="Multimodal RAG Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
STATIC_DIR = BASE_DIR / "static"
TEMPLATES_DIR = BASE_DIR / "templates"
MAX_UPLOAD_SIZE_BYTES = 100 * 1024 * 1024
UPLOAD_STREAM_CHUNK_SIZE = 1024 * 1024

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
STATIC_DIR.mkdir(parents=True, exist_ok=True)
TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@app.get("/")
def home():
    return {"message": "Backend running successfully"}


@app.get("/ui", response_class=HTMLResponse)
def ui(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"title": "Multimodal Document Intelligence"},
    )


# ---------------------------------------------------------------------------
# PDF Upload + Ingestion
# ---------------------------------------------------------------------------

@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    temp_file_path: Optional[Path] = None
    final_file_path: Optional[Path] = None
    try:
        filename = os.path.basename(file.filename or "").strip()
        if not filename.lower().endswith(".pdf"):
            return {"success": False, "message": "Only PDF files are allowed."}

        temp_file_path = await _save_upload_to_temp(file)
        document_id = str(uuid.uuid4())
        final_file_path = UPLOAD_DIR / f"{document_id}_{filename}"
        shutil.move(str(temp_file_path), str(final_file_path))

        # Extract full text (page-level, via PyMuPDF) in threadpool to avoid blocking event loop.
        result = await run_in_threadpool(extract_text_from_pdf, str(final_file_path))
        if not result["success"]:
            if final_file_path.exists():
                final_file_path.unlink()
            return {
                "success": False,
                "error": result.get("error") or "Failed to process PDF. Please retry.",
            }

        if not result.get("text", "").strip():
            if final_file_path.exists():
                final_file_path.unlink()
            return {
                "success": False,
                "error": "No readable text found. The PDF may be scanned or image-only.",
            }

        # Chunk the extracted text with document_id attached
        chunks = await run_in_threadpool(
            chunk_text,
            result["text"],
            filename,
            document_id,
            1050,
            240,
        )

        # Embed and store in ChromaDB
        stored = await run_in_threadpool(store_chunks, chunks, 24)
        if stored == 0:
            return {"success": False, "error": "Failed to process PDF. Please retry."}

        add_document_metadata(
            document_id=document_id,
            filename=filename,
            chunks=stored,
            pages=result.get("pages", 0),
        )
        metadata = get_document_metadata(document_id) or {}

        return {
            "success": True,
            "filename": filename,
            "document_id": document_id,
            "pages": result["pages"],
            "total_chunks_stored": stored,
            "upload_date": metadata.get("upload_date"),
            "message": "Large PDF processing may take 1-2 mins.",
        }

    except ValueError as exc:
        if final_file_path and final_file_path.exists():
            final_file_path.unlink()
        return {"success": False, "error": str(exc)}
    except Exception:
        if final_file_path and final_file_path.exists():
            final_file_path.unlink()
        return {"success": False, "error": "Failed to process PDF. Please retry."}
    finally:
        try:
            await file.close()
        except Exception:
            pass
        if temp_file_path and temp_file_path.exists():
            temp_file_path.unlink()


async def _save_upload_to_temp(file: UploadFile) -> Path:
    temp_handle = tempfile.NamedTemporaryFile(
        mode="wb",
        delete=False,
        suffix=".pdf",
        dir=str(UPLOAD_DIR),
    )
    total_bytes = 0
    temp_file_path = Path(temp_handle.name)

    try:
        while True:
            chunk = await file.read(UPLOAD_STREAM_CHUNK_SIZE)
            if not chunk:
                break
            total_bytes += len(chunk)
            if total_bytes > MAX_UPLOAD_SIZE_BYTES:
                raise ValueError("File is too large. Maximum supported size is 100 MB.")
            temp_handle.write(chunk)
        temp_handle.flush()
        return temp_file_path
    finally:
        temp_handle.close()


# ---------------------------------------------------------------------------
# RAG Question-Answering
# ---------------------------------------------------------------------------

class AskRequest(BaseModel):
    question: str
    document_id: Optional[str] = None


@app.post("/ask")
async def ask(request: AskRequest):
    try:
        question = request.question.strip()
        if not question:
            return {"success": False, "error": "Question cannot be empty."}

        result = answer_question(
            question=question,
            document_id=request.document_id,
            top_k=9,
        )
        return result

    except Exception as exc:
        return {"success": False, "error": str(exc)}
