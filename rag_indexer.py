import chromadb
from openai import OpenAI

client = OpenAI()
embedding_model = "text-embedding-3-small"  # or 3-large

chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection(name="pdf_chunks")


def index_chunks(chunks):
    for i, chunk in enumerate(chunks):
        embedding = (
            client.embeddings.create(input=chunk, model=embedding_model)
            .data[0]
            .embedding
        )

        collection.add(documents=[chunk], embeddings=[embedding], ids=[str(i)])