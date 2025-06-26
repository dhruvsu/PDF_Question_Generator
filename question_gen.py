import os
from dotenv import load_dotenv
from openai import OpenAI
import chromadb

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=api_key)

chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection(name="pdf_chunks")


def embed_question(question: str, model="text-embedding-3-small"):
    embedding = client.embeddings.create(input=question, model=model).data[0].embedding
    return embedding


def retrieve_context(question_embedding, k=3):
    results = collection.query(query_embeddings=[question_embedding], n_results=k)
    documents = results["documents"][0]
    return "\n".join(documents)


def generate_answer(context, question, model="gpt-4o"):
    prompt = f"""
Use the context below to answer the question. If the context is insufficient, say "I'm not sure based on the document."

Context:
{context}

Question:
{question}
"""
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant who answers based on provided context.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0,
    )

    return response.choices[0].message.content

def run_rag_pipeline(question: str):
    question_embedding = embed_question(question)
    context = retrieve_context(question_embedding)
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
