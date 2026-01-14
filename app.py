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

# Custom CSS for dark theme
st.markdown("""
<style>
    /* Main app background - solid black */
    .stApp {
        background: #000000;
        min-height: 100vh;
    }
    
    /* Style the main container */
    .main .block-container {
        background: rgba(20, 20, 20, 0.95);
        border-radius: 20px;
        padding: 2rem;
        margin-top: 1rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.5);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Title styling */
    h1 {
        color: #ffffff !important;
        background: none;
        -webkit-text-fill-color: #ffffff;
        text-align: center;
        font-size: 2.5rem !important;
    }
    
    /* Input field wrapper for gradient border */
    .stTextInput > div > div {
        background: linear-gradient(135deg, #8b2457 0%, #6b3fa0 50%, #1e3a8a 100%) !important;
        padding: 2px !important;
        border-radius: 17px !important;
    }
    
    /* Input field styling */
    .stTextInput > div > div > input {
        background: #1a1a1a !important;
        border: none !important;
        border-radius: 15px !important;
        color: #ffffff !important;
        padding: 12px;
        font-size: 1.1rem;
    }
    
    .stTextInput > div > div > input:focus {
        box-shadow: 0 0 20px rgba(107, 63, 160, 0.4);
    }
    
    .stTextInput > div > div > input::placeholder {
        color: rgba(255, 255, 255, 0.4);
    }
    
    /* Label styling */
    .stTextInput label {
        color: #cccccc !important;
        font-size: 1.1rem !important;
        font-weight: 500;
    }
    
    /* Response text styling */
    .stMarkdown p, .stWrite p {
        color: #e0e0e0;
        font-size: 1.05rem;
        line-height: 1.6;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: #1a1a1a;
        border-radius: 10px;
        color: #cccccc !important;
    }
    
    .streamlit-expanderContent {
        background: rgba(30, 30, 30, 0.9);
        border-radius: 0 0 10px 10px;
        border: 1px solid #333333;
    }
    
    /* Spinner styling */
    .stSpinner > div {
        border-top-color: #ffffff !important;
    }
</style>
""", unsafe_allow_html=True)

# Streamlit UI
st.title("📄 Roxy's Personal Chatbot")

# Simple divider
st.markdown("""
<div style="text-align: center; margin: 10px 0 20px 0;">
    <span style="color: #444444; font-size: 1.5rem;">━━━━━━━━━</span>
</div>
""", unsafe_allow_html=True)

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
    
    # Display answer with dark themed box
    st.markdown(f"""
    <div style="
        background: rgba(30, 30, 30, 0.9);
        border: 1px solid #333333;
        border-radius: 15px;
        padding: 20px;
        margin: 15px 0;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
    ">
        <div style="text-align: center; margin-bottom: 10px;">
            <span style="color: #cccccc; font-weight: bold; font-size: 1.1rem;">Response</span>
        </div>
        <p style="color: #e0e0e0; font-size: 1.05rem; line-height: 1.7; margin: 0;">
            {answer}
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Show retrieved context (for debugging)
    with st.expander("See retrieved context"):
        st.markdown(f"""
        <p style="color: #cccccc; font-size: 0.95rem; line-height: 1.5;">
            {context}
        </p>
        """, unsafe_allow_html=True)
