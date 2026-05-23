from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("BAAI/bge-small-en-v1.5")

def chunk_text(pages, pdf_name):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=100
    )

    chunks = []

    for page in pages:
        split_chunks = splitter.split_text(page["text"])

        for chunk in split_chunks:
            chunks.append({
                "text": chunk,
                "page": page["page"],
                "filename": pdf_name
            })

    return chunks


def create_embeddings(chunks):
    texts = [chunk["text"] for chunk in chunks]

    embeddings = model.encode(texts)

    return embeddings