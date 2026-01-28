"""
Personal RAG Chatbot with Resume Analysis capabilities.
Supports uploading up to 20 resumes, cross-resume queries, and conversation memory.
"""
import streamlit as st
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from dotenv import load_dotenv
import os
from resume_manager import ResumeManager

# Load environment variables
load_dotenv()

# Check for OpenAI API key
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("⚠️ OpenAI API key not found!")
    st.markdown("""
    **To set up your API key:**
    
    1. Create a `.env` file in the project root
    2. Add the line: `OPENAI_API_KEY=your_api_key_here`
    3. Restart the application
    
    Or set the environment variable directly:
    ```bash
    export OPENAI_API_KEY=your_api_key_here
    ```
    """)
    st.stop()

# Initialize OpenAI client
client = OpenAI(api_key=api_key)

# Load model and index for personal chatbot mode
@st.cache_resource
def load_resources():
    """Load the embedding model and FAISS index for the personal chatbot."""
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

# Initialize resume manager in session state
def get_resume_manager():
    """Get or create the resume manager from session state."""
    if 'resume_manager' not in st.session_state:
        st.session_state.resume_manager = ResumeManager()
    return st.session_state.resume_manager

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
    
    h2, h3 {
        color: #ffffff !important;
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
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #1a1a1a;
        border-radius: 10px;
        color: #cccccc;
        padding: 10px 20px;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #8b2457 0%, #6b3fa0 100%);
        color: white;
    }
    
    /* File uploader styling */
    .stFileUploader {
        background: #1a1a1a;
        border-radius: 10px;
        padding: 10px;
    }
    
    /* Sidebar styling */
    .css-1d391kg, [data-testid="stSidebar"] {
        background: #0a0a0a;
    }
    
    /* Card styling for resume metadata */
    .resume-card {
        background: rgba(30, 30, 30, 0.9);
        border: 1px solid #333333;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
    
    /* Chat message styling */
    .chat-message {
        background: rgba(30, 30, 30, 0.9);
        border: 1px solid #333333;
        border-radius: 15px;
        padding: 15px;
        margin: 10px 0;
    }
    
    .user-message {
        background: rgba(107, 63, 160, 0.2);
        border-color: rgba(107, 63, 160, 0.5);
    }
    
    .assistant-message {
        background: rgba(30, 30, 30, 0.9);
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

# Create tabs for different modes
tab1, tab2 = st.tabs(["💬 Personal Chat", "📋 Resume Analyzer"])

# Tab 1: Original Personal Chatbot
with tab1:
    model, index, chunks = load_resources()
    
    query = st.text_input("Ask me anything about Roxy:", key="personal_query")
    
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

# Tab 2: Resume Analyzer
with tab2:
    resume_manager = get_resume_manager()
    
    # Create columns for layout
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("📤 Upload Resumes")
        st.caption(f"Upload up to {resume_manager.MAX_RESUMES} resumes (PDF, DOCX, or TXT)")
        
        # File uploader
        uploaded_files = st.file_uploader(
            "Choose resume files",
            type=['pdf', 'docx', 'txt'],
            accept_multiple_files=True,
            key="resume_uploader"
        )
        
        # Process uploaded files
        if uploaded_files:
            current_count = resume_manager.get_resume_count()
            remaining_slots = resume_manager.MAX_RESUMES - current_count
            
            if len(uploaded_files) > remaining_slots:
                st.warning(f"⚠️ Only {remaining_slots} slots available. First {remaining_slots} files will be processed.")
                uploaded_files = uploaded_files[:remaining_slots]
            
            if st.button("📥 Process Resumes", type="primary"):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for i, uploaded_file in enumerate(uploaded_files):
                    try:
                        status_text.text(f"Processing: {uploaded_file.name}")
                        file_bytes = uploaded_file.read()
                        resume_id, metadata = resume_manager.add_resume(file_bytes, uploaded_file.name)
                        st.success(f"✅ Added: {metadata['candidate_name']}")
                    except Exception as e:
                        st.error(f"❌ Failed to process {uploaded_file.name}: {str(e)}")
                    
                    progress_bar.progress((i + 1) / len(uploaded_files))
                
                status_text.text("Processing complete!")
                st.rerun()
        
        # Display current resume count
        st.markdown("---")
        count = resume_manager.get_resume_count()
        st.metric("Resumes Loaded", f"{count} / {resume_manager.MAX_RESUMES}")
        
        # Clear buttons
        if count > 0:
            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("🗑️ Clear All", type="secondary"):
                    resume_manager.clear_all_resumes()
                    st.rerun()
            with col_b:
                if st.button("🔄 New Chat", type="secondary"):
                    resume_manager.clear_conversation()
                    if 'chat_history' in st.session_state:
                        st.session_state.chat_history = []
                    st.rerun()
    
    with col2:
        st.subheader("💬 Ask Questions")
        
        if resume_manager.get_resume_count() == 0:
            st.info("👈 Upload resumes to start asking questions about candidates.")
        else:
            # Initialize chat history in session state
            if 'chat_history' not in st.session_state:
                st.session_state.chat_history = []
            
            # Display chat history
            chat_container = st.container()
            with chat_container:
                for msg in st.session_state.chat_history:
                    if msg["role"] == "user":
                        st.markdown(f"""
                        <div class="chat-message user-message">
                            <strong>🧑 You:</strong><br>
                            {msg["content"]}
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="chat-message assistant-message">
                            <strong>🤖 Assistant:</strong><br>
                            {msg["content"]}
                        </div>
                        """, unsafe_allow_html=True)
            
            # Query input
            st.markdown("---")
            resume_query = st.text_input(
                "Ask about the resumes:",
                placeholder="e.g., 'Who has experience with AWS?' or 'Summarize John's background'",
                key="resume_query"
            )
            
            # Example queries
            st.caption("💡 Try: 'Who would be best for a data science role?' | 'Compare candidates' Python skills' | 'What industries has [name] worked in?'")
            
            if resume_query:
                with st.spinner("Analyzing resumes..."):
                    response = resume_manager.query(resume_query)
                
                # Add to chat history
                st.session_state.chat_history.append({"role": "user", "content": resume_query})
                st.session_state.chat_history.append({"role": "assistant", "content": response})
                
                # Display response
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
                        <span style="color: #cccccc; font-weight: bold; font-size: 1.1rem;">🤖 Response</span>
                    </div>
                    <p style="color: #e0e0e0; font-size: 1.05rem; line-height: 1.7; margin: 0; white-space: pre-wrap;">
                        {response}
                    </p>
                </div>
                """, unsafe_allow_html=True)
    
    # Resume metadata section (below the main content)
    if resume_manager.get_resume_count() > 0:
        st.markdown("---")
        st.subheader("📊 Resume Database")
        
        all_metadata = resume_manager.get_all_metadata()
        
        # Create tabs for each resume
        resume_tabs = st.tabs([meta["candidate_name"] for meta in all_metadata])
        
        for i, tab in enumerate(resume_tabs):
            with tab:
                meta = all_metadata[i]
                
                col_left, col_right = st.columns(2)
                
                with col_left:
                    st.markdown(f"""
                    <div class="resume-card">
                        <h4 style="color: #ffffff; margin-top: 0;">👤 {meta['candidate_name']}</h4>
                        <p style="color: #cccccc; margin: 5px 0;"><strong>📧 Email:</strong> {meta['email']}</p>
                        <p style="color: #cccccc; margin: 5px 0;"><strong>📱 Phone:</strong> {meta['phone']}</p>
                        <p style="color: #cccccc; margin: 5px 0;"><strong>💼 Role:</strong> {meta['current_role']}</p>
                        <p style="color: #cccccc; margin: 5px 0;"><strong>📅 Experience:</strong> {meta['experience_years']} years</p>
                        <p style="color: #cccccc; margin: 5px 0;"><strong>🎓 Education:</strong> {meta['education']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col_right:
                    st.markdown(f"""
                    <div class="resume-card">
                        <h4 style="color: #ffffff; margin-top: 0;">📝 Summary</h4>
                        <p style="color: #cccccc; font-size: 0.95rem; line-height: 1.5;">
                            {meta['summary']}
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Skills and industries
                st.markdown(f"""
                <div class="resume-card">
                    <h4 style="color: #ffffff; margin-top: 0;">🛠️ Key Skills</h4>
                    <p style="color: #cccccc;">
                        {', '.join(meta['key_skills'][:15]) if meta['key_skills'] else 'Not extracted'}
                    </p>
                    <h4 style="color: #ffffff; margin-top: 15px;">🏢 Industries</h4>
                    <p style="color: #cccccc;">
                        {', '.join(meta['industries']) if meta['industries'] else 'Not extracted'}
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                # Get detailed summary button
                if st.button(f"📄 Generate Detailed Summary", key=f"summary_{meta['resume_id']}"):
                    with st.spinner("Generating detailed summary..."):
                        detailed_summary = resume_manager.summarize_resume(meta['resume_id'])
                    st.markdown(f"""
                    <div style="
                        background: rgba(40, 40, 40, 0.9);
                        border: 1px solid #444444;
                        border-radius: 10px;
                        padding: 20px;
                        margin: 10px 0;
                    ">
                        <h4 style="color: #ffffff;">📋 Detailed Summary</h4>
                        <p style="color: #e0e0e0; white-space: pre-wrap; line-height: 1.6;">
                            {detailed_summary}
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Remove resume button
                if st.button(f"🗑️ Remove Resume", key=f"remove_{meta['resume_id']}"):
                    resume_manager.remove_resume(meta['resume_id'])
                    st.rerun()

# Footer
st.markdown("""
<div style="text-align: center; margin-top: 40px; padding: 20px;">
    <span style="color: #444444; font-size: 0.9rem;">
        Powered by RAG • FAISS • OpenAI
    </span>
</div>
""", unsafe_allow_html=True)
