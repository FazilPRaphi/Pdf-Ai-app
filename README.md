# Multimodal Document Intelligence System

An end-to-end AI-powered document chat application that lets users upload PDFs and ask grounded questions with source-aware responses. Built with a production-ready Retrieval-Augmented Generation (RAG) pipeline using FastAPI, Chroma, and Groq.

## Features

* **PDF Upload & Parsing**
  Upload PDFs of any size and extract structured text using PyMuPDF.

* **Advanced RAG Pipeline**
  Semantic chunking with overlap for better context retention.

* **Grounded Q&A**
  Answers are generated strictly from document context to reduce hallucinations.

* **Source-Aware Responses**
  Retrieved context chunks are linked to each answer (collapsible source view).

* **Metadata-Aware Responses**
  Directly answers:

  * page count
  * active filename
  * document metadata

* **Large PDF Support**
  Batch embedding and vector insertion for robust handling of long documents.

* **Modern Full-Stack UI**
  Dark-themed glassmorphism interface with:

  * drag & drop upload
  * real-time upload progress
  * grounded chat experience

## Tech Stack

### Backend

* FastAPI
* Jinja
* PyMuPDF
* sentence-transformers
* Chroma
* Groq

### Frontend

* HTML + CSS + minimal JavaScript
* Jinja2 templating

## Project Structure

```bash
DOC-READER/
├── backend/
│   ├── routes/              # API routes
│   ├── services/            # RAG + vector DB services
│   ├── static/              # CSS / JS
│   ├── templates/           # Jinja templates
│   ├── uploads/             # runtime uploads (ignored)
│   ├── utils/               # PDF parsing + chunking
│   ├── app.py               # main FastAPI app
│   └── requirements.txt
├── chroma_db/               # vector DB (ignored)
├── .env
├── .gitignore
└── README.md
```

## Setup Instructions

### 1. Clone the repository

```bash
git clone <your_repo_url>
cd DOC-READER
```

### 2. Create and activate virtual environment

```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r backend/requirements.txt
```

### 4. Add environment variables

Create a `.env` file in root:

```env
GROQ_API_KEY=your_groq_api_key_here
```

### 5. Run the app

```bash
cd backend
uvicorn app:app --reload
```

### 6. Open in browser

```text
http://127.0.0.1:8000/ui
```

## Example Use Cases

* Resume / CV analysis
* Legal document summarization
* Research paper Q&A
* Notes / textbook assistance
* Enterprise document search

## Future Improvements

* OCR for scanned PDFs
* support for images / DOCX
* user authentication
* document history
* cloud deployment

## License

This project is for educational and portfolio purposes.
