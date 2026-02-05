"""
embedding script for the personal chatbot
creates a faiss index from the resume and personal data files
run this once to generate the faiss_index.bin file
"""

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# load the sentence transformer model for creating embeddings
# all-MiniLM-L6-v2 produces 384 dimensional vectors and is fast
model = SentenceTransformer('all-MiniLM-L6-v2')

# load the personal data files
with open("resume.txt", "r") as f:
    resume_data = f.read()

with open("personal.txt", "r") as f:
    personal_data = f.read()

# Split resume by double newlines (paragraphs/sections)
resume_chunks = [chunk.strip() for chunk in resume_data.split("\n\n") if chunk.strip()]

# keep personal data as one chunk since its usually shorter
chunks = resume_chunks + [personal_data]

# create embeddings for all the chunks
embeddings = model.encode(chunks)
embeddings = np.array(embeddings).astype('float32')

# create a faiss index using l2 distance
# IndexFlatL2 is simple but works great for small datasets
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# save the index to disk
faiss.write_index(index, "faiss_index.bin")

print(f"created index with {index.ntotal} vectors of dimension {dimension}")
