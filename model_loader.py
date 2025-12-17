# model_loader.py
from sentence_transformers import SentenceTransformer

print("⏳ Loading embedding model...")
model = SentenceTransformer("sentence-transformers/paraphrase-MiniLM-L6-v2")
print("✅ Model siap dipakai")
