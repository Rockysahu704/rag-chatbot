from sentence_transformers import SentenceTransformer
import numpy as np
import vector_store

# Load embedding model
model = SentenceTransformer("BAAI/bge-small-en-v1.5")


def search(query, top_k=5):

    # Create embedding for user query
    query_embedding = model.encode([query])

    # Search FAISS vector database
    distances, indices = vector_store.index.search(
        np.array(query_embedding).astype("float32"),
        top_k
    )

    results = []

    # Collect retrieved chunks
    for i, idx in enumerate(indices[0]):

        # Skip invalid indexes
        if idx == -1:
            continue

        chunk = vector_store.stored_chunks[idx]

        # Add similarity score
        chunk["score"] = float(distances[0][i])

        # Safe filename fallback
        chunk["filename"] = chunk.get(
            "filename",
            "Unknown Document"
        )

        results.append(chunk)

    return results