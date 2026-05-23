import faiss
import numpy as np
import pickle
import os

# Base directory of current file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Vector DB directory
VECTOR_DB_DIR = os.path.join(BASE_DIR, "../vector_db")

# Create vector_db folder if not exists
os.makedirs(VECTOR_DB_DIR, exist_ok=True)

# File paths
INDEX_PATH = os.path.join(VECTOR_DB_DIR, "faiss.index")
CHUNKS_PATH = os.path.join(VECTOR_DB_DIR, "chunks.pkl")

# Global variables
index = None
stored_chunks = []


# Load existing vector DB if available
def load_existing_data():

    global index
    global stored_chunks

    # Load FAISS index
    if os.path.exists(INDEX_PATH):

        index = faiss.read_index(INDEX_PATH)

    # Load stored chunks
    if os.path.exists(CHUNKS_PATH):

        with open(CHUNKS_PATH, "rb") as f:

            stored_chunks = pickle.load(f)


# Save vector DB
def save_data():

    global index
    global stored_chunks

    # Save FAISS index
    faiss.write_index(index, INDEX_PATH)

    # Save chunk metadata
    with open(CHUNKS_PATH, "wb") as f:

        pickle.dump(stored_chunks, f)


# Add new embeddings + chunks
def add_to_index(embeddings, chunks):

    global index
    global stored_chunks

    embeddings_np = np.array(embeddings).astype("float32")

    # First upload
    if index is None:

        dimension = len(embeddings[0])

        index = faiss.IndexFlatL2(dimension)

    # Append embeddings
    index.add(embeddings_np)

    # Append chunks
    stored_chunks.extend(chunks)

    # Save updated DB
    save_data()