from sentence_transformers import SentenceTransformer
import numpy as np
import vector_store

model = SentenceTransformer("BAAI/bge-small-en-v1.5")


def search(query, top_k=10):

    # Create query embedding
    query_embedding = model.encode([query])

    # Search FAISS index
    distances, indices = vector_store.index.search(
        np.array(query_embedding).astype("float32"),
        top_k
    )

    results = []

    # Track used files
    used_files = set()

    for i, idx in enumerate(indices[0]):

        chunk = vector_store.stored_chunks[idx]

        # Safe filename handling
        filename = chunk.get("filename", "Unknown Document")

        # Diversify retrieval across PDFs
        if filename not in used_files:

            chunk["score"] = float(distances[0][i])

            chunk["filename"] = filename

            results.append(chunk)

            used_files.add(filename)

        # Final limit
        if len(results) >= 5:
            break

    return results