import streamlit as st
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Load model and index
@st.cache_resource
def load_resources():
    model = SentenceTransformer('all-MiniLM-L6-v2')
    index = faiss.read_index("faiss_index.bin")
    
    # Load and chunk the same way as embeddata.py
    with open("resume.txt", "r") as f:
        resume_data = f.read()
    with open("personal.txt", "r") as f:
        personal_data = f.read()
    
    resume_chunks = [chunk.strip() for chunk in resume_data.split("\n\n") if chunk.strip()]
    chunks = resume_chunks + [personal_data]
    
    return model, index, chunks

model, index, chunks = load_resources()

# Streamlit UI
st.title("Roxy's personal chatbot 💬")

query = st.text_input("Ask me anything about Roxy:")

if query:
    # Embed the query
    query_embedding = model.encode([query]).astype('float32')
    
    # Search FAISS for relevant chunks
    k = 2
    distances, indices = index.search(query_embedding, k)
    relevant_chunks = [chunks[i] for i in indices[0]]
    context = "\n".join(relevant_chunks)
    
    # Send to LLM
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant answering questions about Roxy Story based on her resume and personal information. Only use the provided context to answer. Be conversational and friendly."
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {query}"
            }
        ]
    )
    
    answer = response.choices[0].message.content
    
    # Display answer
    st.write(answer)
    
    # Show retrieved context (for debugging)
    with st.expander("See retrieved context"):
        st.write(context)