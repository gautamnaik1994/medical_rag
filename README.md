# Retrieval-Augmented Generation (RAG) API with FastAPI and LangChain

This repository demonstrates how to build a Retrieval-Augmented Generation (RAG) system using FastAPI and LangChain. The system allows you to upload PDF documents (including scanned/image-based PDFs), process them with OCR, and query them for information using OpenAI's language models.

## Features

- **PDF Upload:** Upload one or more PDF files (text-based or scanned).
- **OCR Support:** Uses `pytesseract` to extract text from image-based PDFs.
- **Indexing:** Documents are indexed for efficient retrieval.
- **RAG Query:** Ask questions about your documents and get answers with source references.
- **API-first:** Interact via REST endpoints or Swagger UI.

## Technologies Used

- FastAPI
- LangChain
- Tesseract OCR
- Docker
- PyMuPDF
- OpenAI API

## Getting Started

### 1. Prerequisites

- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) installed and available in your system path.
- [Docker](https://www.docker.com/)
- OpenAI API key

### 2. Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/gautamnaik1994/medical_rag.git
cd medical_rag
pip install -r requirements.txt
```

### 3. Environment Variables

Create a `.env` file in the project root with your OpenAI API key:

```plaintext
OPENAI_API_KEY=your_openai_api_key
```

### 4. Running the Application

#### With Docker

If you have Docker installed, you can build and run the app with:

```bash
docker compose up --build
```

#### Without Docker

Run the FastAPI app directly:

```bash
uvicorn main:app --reload
```

> **Note:** If you encounter issues with `pytesseract`, ensure Tesseract OCR is installed and accessible from your system's PATH.

## API Usage

Once running, access the API at [http://localhost:8000](http://localhost:8000).

- **Swagger UI:** [http://localhost:8000/docs](http://localhost:8000/docs)

### Endpoints

#### 1. Upload PDF and Build Index

**Example using `curl`:**

```bash
curl -X 'POST' \
  'http://localhost:8000/upload_pdf/' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'files=@input_data2.pdf;type=application/pdf' \
  -F 'files=@input_data.pdf;type=application/pdf'
```

#### 2. Query Endpoint

**Request Body:**

```json
{
  "question": "What is this document about?",
  "doc_name": "input_data.pdf"
}
```

The `doc_name` should match one of the uploaded PDF files. This is required to retrieve the correct document for querying.

**Example using `curl`:**

```bash
curl -X 'POST' \
  'http://localhost:8000/query_rag/' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "question": "What is this document about?",
  "doc_name": "input_data.pdf"
}'
```

**Response:**

```json
{
  "result": "This document is about ...",
  "source_documents": [
    {"page": 1, "source": "input_data1.pdf"},
    ...
  ]
}
```

## Troubleshooting

- **Tesseract Not Found:** Make sure Tesseract is installed and available in your system's PATH.
- **OpenAI API Errors:** Ensure your API key is correct and you have sufficient quota.
- **Vector Store Not Found:** You must upload and index a PDF before querying it.
