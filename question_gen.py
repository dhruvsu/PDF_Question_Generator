import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=api_key)

def generate_questions(text, num_questions=5):
    prompt = f"""
  You are a helpful assistant. Generate {num_questions} study questions based on the following text:
  
  {text[:3000]}  # Truncate to stay within token limits
  """

    responses = client.chat.completions.create(
        model="gpt-4o-mini",
        store=True,
        messages=[
            {"role": "system", "content": "You are a tutor generating exam-style questions."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7
    )
    return responses.choices[0].message.content