from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")

chunks_store = []
embedding_store = []
faiss_index = None

def index_chunks(chunks):
    global faiss_index, chunks_store, embedding_store

    embeddings = model.encode(chunks, convert_to_numpy=True)

    chunks_store = chunks
    embedding_store = embeddings

    dim = embeddings.shape[1]
    faiss_index = faiss.IndexFlatL2(dim)
    faiss_index.add(embeddings)