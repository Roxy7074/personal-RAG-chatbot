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

# Custom CSS for sunset theme
st.markdown("""
<style>
    /* Main app background with sunset gradient */
    .stApp {
        background: linear-gradient(180deg,
            #1a1a2e 0%,
            #4a1942 15%,
            #8b2f5f 30%,
            #d4587a 45%,
            #f5a962 65%,
            #ffd89b 85%,
            #fff1c1 100%);
        min-height: 100vh;
    }
    
    /* Style the main container */
    .main .block-container {
        background: rgba(26, 26, 46, 0.85);
        border-radius: 20px;
        padding: 2rem;
        margin-top: 1rem;
        box-shadow: 0 8px 32px rgba(212, 88, 122, 0.3);
        border: 1px solid rgba(245, 169, 98, 0.3);
    }
    
    /* Title styling */
    h1 {
        background: linear-gradient(90deg, #ffd89b 0%, #f5a962 30%, #ff8c69 50%, #f5a962 70%, #ffd89b 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        font-size: 2.5rem !important;
    }
    
    /* Input field styling */
    .stTextInput > div > div > input {
        background: #fff8e7 !important;
        border: 2px solid #f5a962 !important;
        border-radius: 15px !important;
        color: #333333 !important;
        padding: 12px;
        font-size: 1.1rem;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #d4587a;
        box-shadow: 0 0 15px rgba(212, 88, 122, 0.5);
    }
    
    .stTextInput > div > div > input::placeholder {
        color: rgba(255, 241, 193, 0.6);
    }
    
    /* Label styling */
    .stTextInput label {
        color: #ffd89b !important;
        font-size: 1.1rem !important;
        font-weight: 500;
    }
    
    /* Response text styling */
    .stMarkdown p, .stWrite p {
        color: #fff1c1;
        font-size: 1.05rem;
        line-height: 1.6;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: linear-gradient(90deg, #8b2f5f, #d4587a);
        border-radius: 10px;
        color: #fff1c1 !important;
    }
    
    .streamlit-expanderContent {
        background: rgba(139, 47, 95, 0.3);
        border-radius: 0 0 10px 10px;
        border: 1px solid #d4587a;
    }
    
    /* Spinner styling */
    .stSpinner > div {
        border-top-color: #f5a962 !important;
    }
    
    /* Sun rays animation */
    @keyframes sunPulse {
        0%, 100% { transform: scale(1); opacity: 1; }
        50% { transform: scale(1.1); opacity: 0.8; }
    }
    
    .sun-decoration {
        animation: sunPulse 3s ease-in-out infinite;
    }
</style>
""", unsafe_allow_html=True)

# Sun decorations header
st.markdown("""
<div style="text-align: center; margin-bottom: -20px;">
    <span style="font-size: 3rem;" class="sun-decoration">🌅</span>
    <span style="font-size: 2rem;">✨</span>
    <span style="font-size: 2.5rem;" class="sun-decoration">☀️</span>
    <span style="font-size: 2rem;">✨</span>
    <span style="font-size: 3rem;" class="sun-decoration">🌇</span>
</div>
""", unsafe_allow_html=True)

# Streamlit UI
st.title("🌞 Roxy's Personal Chatbot 🌞")

# Decorative sun divider
st.markdown("""
<div style="text-align: center; margin: 10px 0 20px 0;">
    <span style="color: #f5a962; font-size: 1.5rem;">━━━</span>
    <span style="font-size: 1.5rem;">🔆</span>
    <span style="color: #f5a962; font-size: 1.5rem;">━━━</span>
</div>
""", unsafe_allow_html=True)

query = st.text_input("☀️ Ask me anything about Roxy:")

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
    
    # Display answer with sunset styled box
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, rgba(139, 47, 95, 0.4) 0%, rgba(212, 88, 122, 0.3) 50%, rgba(245, 169, 98, 0.2) 100%);
        border: 2px solid #f5a962;
        border-radius: 15px;
        padding: 20px;
        margin: 15px 0;
        box-shadow: 0 4px 15px rgba(245, 169, 98, 0.3);
    ">
        <div style="text-align: center; margin-bottom: 10px;">
            <span style="font-size: 1.5rem;">🌅</span>
            <span style="color: #ffd89b; font-weight: bold; font-size: 1.1rem;"> Response </span>
            <span style="font-size: 1.5rem;">🌅</span>
        </div>
        <p style="color: #fff1c1; font-size: 1.05rem; line-height: 1.7; margin: 0;">
            {answer}
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Show retrieved context (for debugging) with sunset themed expander
    with st.expander("🔆 See retrieved context"):
        st.markdown(f"""
        <p style="color: #ffd89b; font-size: 0.95rem; line-height: 1.5;">
            {context}
        </p>
        """, unsafe_allow_html=True)
