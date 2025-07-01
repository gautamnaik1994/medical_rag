from typing import List
from fastapi.middleware.cors import CORSMiddleware
import os
from fastapi import FastAPI, File, Request, UploadFile
from fastapi.responses import JSONResponse
from modules.rag import RagPipeline, QueryRAG
from pydantic import BaseModel
from modules.ocr import OCRPipeline
import logging
from modules.logger import logger
app = FastAPI()

logger = logging.getLogger("app")

env = os.getenv("ENVIRONMENT", "production")


allowed_origins = [
]

if env == "development":
    allowed_origins.append("http://localhost:3000")
    allowed_origins.append("http://127.0.0.1:3000")

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return JSONResponse(content={"message": "Welcome to the RAG API!"})


@app.post("/upload_pdf/", summary="Upload PDF files for processing", description="This endpoint allows you to upload one or more PDF files. The files will be processed using OCR to extract text, and then indexed for RAG (Retrieval-Augmented Generation) purposes. The processed files will be stored in the 'uploaded_files' directory.")
async def upload_pdf(request: Request, files: List[UploadFile] = File(...)):
    try:
        for file in files:
            content = await file.read()
            if os.path.exists("uploaded_files") is False:
                os.makedirs("uploaded_files")
            file_path = f"uploaded_files/{file.filename}"
            with open(file_path, "wb") as f:
                f.write(content)
            ocr_pipeline = OCRPipeline(file_path)
            text_content = ocr_pipeline.run()
            logger.info(f"Extracted text from {file.filename}")
            rag_pipeline = RagPipeline(text_content, doc_name=file.filename)
            rag_pipeline.build_index()
            logger.info(f"Indexed file: {file.filename}")
        return JSONResponse(content={"message": "PDF files processed successfully."})
    except Exception as e:
        logger.error(f"Error processing PDF files: {str(e)}")
        return JSONResponse(content={"error": str(e)}, status_code=500)


class QueryRequest(BaseModel):
    question: str
    doc_name: str = "input_data.pdf"


@app.post("/query_rag/", summary="Query the document information", description="This endpoint allows you to query the RAG system with a question and a document name. It returns the answer along with the source documents used to generate the answer.")
async def query_rag(request: QueryRequest):
    try:
        query_rag = QueryRAG(doc_name=request.doc_name)
        result, source_documents = query_rag.query(request.question)
        logger.debug(f"Query: {request.question} | Result: {result}")
        return JSONResponse(content={"result": result, "source_documents": [doc.metadata for doc in source_documents]})
    except Exception as e:
        logger.error(f"Error querying RAG: {str(e)}")
        return JSONResponse(content={"error": str(e)}, status_code=500)
