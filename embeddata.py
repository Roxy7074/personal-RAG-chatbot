import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load data directly from files
with open("resume.txt", "r") as f:
    resume_data = f.read()

with open("personal.txt", "r") as f:
    personal_data = f.read()

# Split resume by double newlines (paragraphs/sections)
resume_chunks = [chunk.strip() for chunk in resume_data.split("\n\n") if chunk.strip()]

# Keep personal as one chunk
chunks = resume_chunks + [personal_data]

# Create embeddings
embeddings = model.encode(chunks)
embeddings = np.array(embeddings).astype('float32')

# Create and populate FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# Save only the FAISS index
faiss.write_index(index, "faiss_index.bin")

print(f"Created index with {index.ntotal} vectors of dimension {dimension}")