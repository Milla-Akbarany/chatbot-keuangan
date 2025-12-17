# llm_client.py
import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL")

groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

def call_llm(prompt: str) -> str:
    if not groq_client:
        return "❌ LLM tidak tersedia"
    resp = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}]
    )
    return resp.choices[0].message.content
