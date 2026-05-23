from fastapi import FastAPI, UploadFile
import shutil
import os

from ingestion import extract_text_from_pdf
from embedding import chunk_text, create_embeddings
from vector_store import add_to_index, load_existing_data
from retrieval import search
from llm import generate_answer

app = FastAPI()

UPLOAD_DIR = "../data/pdfs"

# Create upload folder if not exists
os.makedirs(UPLOAD_DIR, exist_ok=True)


@app.post("/upload")
async def upload_pdf(file: UploadFile):

    # Save uploaded PDF
    file_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Extract text
    pages = extract_text_from_pdf(file_path)

    # Create chunks with metadata
    chunks = chunk_text(pages, file.filename)

    # Generate embeddings
    embeddings = create_embeddings(chunks)

    # Load existing vector DB
    load_existing_data()

    # Append new PDF embeddings/chunks
    add_to_index(embeddings, chunks)

    return {
        "message": f"{file.filename} processed successfully",
        "chunks": len(chunks)
    }


@app.get("/ask")
def ask_question(query: str):

    # Load vector DB
    load_existing_data()

    # Retrieve relevant chunks
    retrieved_chunks = search(query)

    # Generate answer
    answer = generate_answer(query, retrieved_chunks)

    # Citations
    citations = [
        {
            "filename": chunk["filename"],
            "page": chunk["page"]
        }
        for chunk in retrieved_chunks
    ]

    return {
        "answer": answer,
        "sources": citations,
        "retrieved_chunks": retrieved_chunks
    }