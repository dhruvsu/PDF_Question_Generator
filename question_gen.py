import os
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import rag_indexer as rindex

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=api_key)
model = rindex.model


def retrieve_context(question, k=3):
    query_vec = model.encode([question], convert_to_numpy=True)
    D, I = rindex.faiss_index.search(query_vec, k)

    top_chunks = [rindex.chunks_store[i] for i in I[0]]
    return "\n".join(top_chunks)


def generate_answer(context, question):
    prompt = f"""
Use the context below to answer the question. Be specific and only answer based on the document.

Context:
{context}

Question:
{question}
"""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant who answers questions based on context.",
            },
            {"role": "user", "content": prompt},
        ],
    )
    return response.choices[0].message.content


def run_rag_pipeline(question):
    context = retrieve_context(question)
    return generate_answer(context, question)


def generate_questions(text, num_questions=5):
    prompt = f"""
  You are a helpful assistant. Generate {num_questions} study questions based on the following text:
  
  {text[:3000]}  # Truncate to stay within token limits
  """

    responses = client.chat.completions.create(
        model="gpt-4o-mini",
        store=True,
        messages=[
            {
                "role": "system",
                "content": "You are a tutor generating exam-style questions.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.7,
    )
    return responses.choices[0].message.content
