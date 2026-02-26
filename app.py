"""
personal rag chatbot with resume analysis capabilities
supports uploading resumes, cross-resume queries, and conversation memory
built with streamlit, faiss, and openai
"""

import json
from datetime import datetime
import streamlit as st
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from dotenv import load_dotenv
import os
from resume_manager import ResumeManager
from tools import init_resources, get_openai_tools, run_tool

# load environment variables from the env file
load_dotenv()

# configure the streamlit page with wide layout for better spacing
st.set_page_config(
    page_title="roxy's chatbot",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# check if the openai api key exists before proceeding
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("openai api key not found")
    st.markdown("""
    **to set up your api key:**
    
    1. create a `.env` file in the project root
    2. add the line: `OPENAI_API_KEY=your_api_key_here`
    3. restart the application
    """)
    st.stop()

# create the openai client for making api calls
client = OpenAI(api_key=api_key)


@st.cache_resource
def load_resources():
    """
    loads the sentence transformer model and faiss index for semantic search
    also loads and chunks the personal data files for retrieval
    cached so it only runs once per session
    """
    model = SentenceTransformer('all-MiniLM-L6-v2')
    index = faiss.read_index("faiss_index.bin")
    
    # load the text files containing personal info
    with open("resume.txt", "r") as f:
        resume_data = f.read()
    with open("personal.txt", "r") as f:
        personal_data = f.read()
    
    # split resume into chunks by paragraph for better retrieval
    resume_chunks = [chunk.strip() for chunk in resume_data.split("\n\n") if chunk.strip()]
    chunks = resume_chunks + [personal_data]
    
    return model, index, chunks


def get_resume_manager():
    """
    gets or creates the resume manager from session state
    keeps the manager persistent across reruns
    """
    if 'resume_manager' not in st.session_state:
        st.session_state.resume_manager = ResumeManager()
    return st.session_state.resume_manager


def init_conversation_context():
    """
    initializes conversation context tracking for better multi-turn awareness
    tracks which candidates have been mentioned and the conversation flow
    """
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    if 'conversation_context' not in st.session_state:
        st.session_state.conversation_context = {
            'last_candidate': None,
            'last_query_type': None,
            'mentioned_candidates': set()
        }


def update_conversation_context(query: str, resume_manager):
    """
    updates conversation context based on the query and available candidates
    helps the LLM understand who is being discussed in follow-up questions
    """
    context = st.session_state.conversation_context
    
    # extract candidate names mentioned in the query
    if hasattr(resume_manager, 'resumes'):
        for candidate_id, resume_data in resume_manager.resumes.items():
            candidate_name = resume_data['metadata']['candidate_name']
            # check if name or parts of name are mentioned
            name_parts = candidate_name.lower().split()
            query_lower = query.lower()
            if candidate_name.lower() in query_lower or any(part in query_lower for part in name_parts if len(part) > 2):
                context['mentioned_candidates'].add(candidate_name)
                context['last_candidate'] = candidate_name
    
    # detect query type for better context handling
    query_lower = query.lower()
    if any(word in query_lower for word in ['compare', 'versus', 'vs', 'both', 'difference', 'between']):
        context['last_query_type'] = 'comparison'
    elif any(word in query_lower for word in ['who', 'which candidate', 'find', 'anyone', 'best', 'most']):
        context['last_query_type'] = 'search'
    elif any(word in query_lower for word in ['tell me more', 'what about', 'their', 'them', 'summarize', 'elaborate']):
        context['last_query_type'] = 'follow_up'
    else:
        context['last_query_type'] = 'general'


def get_all_candidates_summary(resume_manager) -> str:
    """
    creates a concise summary of all uploaded candidates for context
    helps the LLM keep track of who is in the database
    """
    if not hasattr(resume_manager, 'resumes') or not resume_manager.resumes:
        return "No candidates uploaded yet."
    
    summaries = []
    for candidate_id, resume_data in resume_manager.resumes.items():
        meta = resume_data['metadata']
        skills = meta.get('key_skills', [])[:4]
        skills_str = ', '.join(skills) if skills else 'Not specified'
        summary = (
            f"• {meta['candidate_name']}: "
            f"{meta.get('current_role', 'Unknown role')} | "
            f"{meta.get('experience_years', 'Unknown')} years exp | "
            f"Skills: {skills_str}"
        )
        summaries.append(summary)
    
    return '\n'.join(summaries) if summaries else "No candidates uploaded yet."


def build_context_aware_system_prompt(resume_manager, retrieved_docs: str = "") -> str:
    """
    builds a system prompt that includes conversation context for better multi-turn awareness
    helps the LLM understand pronouns and follow-up references correctly
    """
    context = st.session_state.conversation_context
    candidates_summary = get_all_candidates_summary(resume_manager)
    
    last_candidate = context.get('last_candidate') or 'None yet'
    mentioned = ', '.join(context.get('mentioned_candidates', set())) or 'None yet'
    query_type = context.get('last_query_type') or 'initial query'
    
    system_prompt = f"""You are an expert HR assistant analyzing resumes for hiring decisions.

=== CANDIDATES IN DATABASE ===
{candidates_summary}

=== CONVERSATION CONTEXT ===
- Last discussed candidate: {last_candidate}
- Previously mentioned candidates: {mentioned}
- Current query type: {query_type}

=== CRITICAL INSTRUCTIONS FOR CONTEXT AWARENESS ===
0. When Roxy is among the candidates, prioritize or highlight her when relevant; other candidates are secondary. Focus comparisons on Roxy when the recruiter asks about "me" or the primary candidate.
1. When the user says "tell me more", "what about them", "their background", or uses pronouns like "he/she/they", they are referring to: {last_candidate}
2. When comparing candidates, ALWAYS state each candidate's FULL NAME before discussing their details
3. In EVERY sentence about a candidate, specify WHICH candidate you're discussing by name
4. If the user's question is ambiguous about which candidate, ask for clarification: "Are you asking about [Candidate A] or [Candidate B]?"
5. When a candidate is first mentioned in a response, include their role: "John Smith (Senior Software Engineer)"
6. Track pronouns carefully - if you use "he/she/they", make sure it's clear who you mean

=== RETRIEVED RESUME INFORMATION ===
{retrieved_docs}

Answer the user's question using the information above. Be specific, professional, and ALWAYS clarify who you're talking about."""

    return system_prompt


# custom css for dark theme with cyan/purple gradient accents
# inspired by modern dashboard design with glass morphism
st.markdown("""
<style>
    /* import clean font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Playfair+Display:wght@500;600;700&display=swap');
    
    /* css variables for consistent theming */
    :root {
        --bg-primary: #0a0f1a;
        --bg-secondary: #111827;
        --bg-card: #1a2332;
        --bg-card-hover: #1f2a3d;
        --accent-cyan: #22d3ee;
        --accent-teal: #2dd4bf;
        --accent-purple: #a855f7;
        --accent-magenta: #ec4899;
        --accent-blue: #3b82f6;
        --text-primary: #f1f5f9;
        --text-secondary: #94a3b8;
        --text-muted: #64748b;
        --border-subtle: rgba(255, 255, 255, 0.1);
        --border-accent: rgba(34, 211, 238, 0.4);
        --gradient-primary: linear-gradient(135deg, #22d3ee 0%, #a855f7 100%);
        --gradient-secondary: linear-gradient(135deg, #a855f7 0%, #ec4899 100%);
        --gradient-button: linear-gradient(135deg, #06b6d4 0%, #8b5cf6 100%);
        --shadow-glow-cyan: 0 0 40px rgba(34, 211, 238, 0.3);
        --shadow-glow-purple: 0 0 40px rgba(168, 85, 247, 0.3);
        --shadow-glow-mixed: 0 0 50px rgba(34, 211, 238, 0.2), 0 0 80px rgba(168, 85, 247, 0.15);
    }
    
    /* main app background - deep dark blue with subtle gradient */
    .stApp {
        background: linear-gradient(180deg, #0a0f1a 0%, #0d1321 50%, #111827 100%);
        font-family: 'Inter', sans-serif;
        min-height: 100vh;
    }
    
    /* main content container */
    .main .block-container {
        background: transparent;
        padding: 2rem 3rem;
        max-width: 100%;
    }
    
    /* sidebar styling - dark with accent glow - ensure visibility */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1321 0%, #111827 100%) !important;
        border-right: 1px solid var(--border-subtle) !important;
        box-shadow: 4px 0 30px rgba(0, 0, 0, 0.3) !important;
        display: flex !important;
        visibility: visible !important;
    }
    
    [data-testid="stSidebar"] > div:first-child {
        background: transparent !important;
        padding: 1.5rem 1rem !important;
    }
    
    /* Ensure sidebar content is visible */
    [data-testid="stSidebarContent"] {
        display: block !important;
        visibility: visible !important;
    }
    
    /* PAGE TITLES - Large and prominent - using !important everywhere to override Streamlit */
    .page-title {
        font-size: 3rem !important;
        font-weight: 800 !important;
        background: linear-gradient(135deg, #22d3ee 0%, #a855f7 100%) !important;
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
        background-clip: text !important;
        margin: 0 0 0.5rem 0 !important;
        padding: 0 !important;
        letter-spacing: -0.03em !important;
        line-height: 1.1 !important;
        display: block !important;
        border: none !important;
    }
    
    .page-subtitle {
        color: #94a3b8 !important;
        font-size: 1.2rem !important;
        font-weight: 400 !important;
        margin: 0 0 2rem 0 !important;
        padding: 0 !important;
    }
    
    /* Section headers */
    .section-header {
        font-size: 1.4rem !important;
        font-weight: 700 !important;
        color: var(--text-primary) !important;
        margin-bottom: 1rem !important;
        padding-bottom: 0.75rem;
        border-bottom: 2px solid var(--border-subtle);
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .section-header-icon {
        display: inline-block;
        width: 8px;
        height: 8px;
        background: var(--gradient-primary);
        border-radius: 50%;
        box-shadow: 0 0 10px var(--accent-cyan);
    }
    
    /* all headings - light text with gradient option */
    h1, h2, h3, h4, h5, h6 {
        color: var(--text-primary) !important;
        font-weight: 600 !important;
        letter-spacing: -0.02em;
    }
    
    h1 {
        font-size: 2.5rem !important;
        margin-bottom: 0.5rem !important;
    }
    
    h2 {
        font-size: 1.75rem !important;
        color: var(--text-primary) !important;
    }
    
    h3 {
        font-size: 1.35rem !important;
        color: var(--text-primary) !important;
    }
    
    h4 {
        font-size: 1.1rem !important;
        color: var(--accent-cyan) !important;
    }
    
    /* paragraph and general text */
    p, span, li {
        color: var(--text-secondary);
        font-size: 0.95rem;
        line-height: 1.7;
    }
    
    /* labels and form text */
    label, .stTextInput label, .stSelectbox label, .stFileUploader label {
        color: var(--text-secondary) !important;
        font-weight: 500 !important;
        font-size: 0.9rem !important;
    }
    
    /* input fields - dark glass style with glow */
    .stTextInput > div > div {
        background: transparent !important;
        padding: 0 !important;
    }
    
    .stTextInput > div > div > input {
        background: var(--bg-card) !important;
        border: 1px solid var(--border-subtle) !important;
        border-radius: 14px !important;
        color: var(--text-primary) !important;
        padding: 16px 20px !important;
        font-size: 1rem !important;
        transition: all 0.3s ease;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: var(--accent-cyan) !important;
        box-shadow: 0 0 0 3px rgba(34, 211, 238, 0.15), var(--shadow-glow-cyan) !important;
    }
    
    .stTextInput > div > div > input::placeholder {
        color: var(--text-muted) !important;
    }
    
    /* buttons - gradient style with glow - ENSURE WHITE TEXT */
    .stButton > button {
        background: var(--gradient-button) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.8rem 2rem !important;
        font-weight: 700 !important;
        font-size: 1rem !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 20px rgba(34, 211, 238, 0.3) !important;
        text-shadow: 0 1px 2px rgba(0, 0, 0, 0.3) !important;
    }
    
    /* Force all text inside buttons to be white */
    .stButton > button * {
        color: #ffffff !important;
    }
    
    .stButton > button p,
    .stButton > button span,
    .stButton > button div {
        color: #ffffff !important;
        font-weight: 700 !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px) !important;
        box-shadow: var(--shadow-glow-mixed) !important;
    }
    
    /* secondary/outline buttons */
    .stButton > button[kind="secondary"] {
        background: transparent !important;
        color: var(--accent-cyan) !important;
        border: 2px solid var(--accent-cyan) !important;
        box-shadow: none !important;
        text-shadow: none !important;
    }
    
    .stButton > button[kind="secondary"] * {
        color: var(--accent-cyan) !important;
    }
    
    .stButton > button[kind="secondary"]:hover {
        background: rgba(34, 211, 238, 0.15) !important;
        box-shadow: var(--shadow-glow-cyan) !important;
    }
    
    /* file uploader - responsive and properly contained */
    .stFileUploader {
        background: var(--bg-card) !important;
        border-radius: 16px !important;
        padding: 1.25rem !important;
        border: 2px dashed rgba(34, 211, 238, 0.4) !important;
        transition: all 0.3s ease !important;
        min-width: 0 !important;
        overflow: hidden !important;
    }
    
    .stFileUploader:hover {
        border-color: var(--accent-cyan) !important;
        box-shadow: var(--shadow-glow-cyan) !important;
    }
    
    .stFileUploader label {
        color: var(--text-secondary) !important;
    }
    
    /* File uploader inner content - prevent text overflow */
    [data-testid="stFileUploader"] {
        min-width: 0 !important;
    }
    
    [data-testid="stFileUploader"] section {
        background: transparent !important;
        border: none !important;
        min-width: 0 !important;
    }
    
    [data-testid="stFileUploader"] section > div {
        min-width: 0 !important;
    }
    
    /* File uploader drop zone text */
    [data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"] {
        min-width: 0 !important;
        padding: 1rem !important;
    }
    
    [data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"] > div {
        display: flex !important;
        flex-direction: column !important;
        align-items: center !important;
        text-align: center !important;
        min-width: 0 !important;
    }
    
    [data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"] span,
    [data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"] p,
    [data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"] small {
        white-space: normal !important;
        word-wrap: break-word !important;
        overflow-wrap: break-word !important;
        text-align: center !important;
        max-width: 100% !important;
    }
    
    /* File uploader button */
    [data-testid="stFileUploader"] button {
        background: var(--gradient-button) !important;
        color: #ffffff !important;
        font-weight: 600 !important;
        white-space: nowrap !important;
    }
    
    [data-testid="stFileUploader"] button * {
        color: #ffffff !important;
    }
    
    /* File list in uploader */
    [data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] {
        background: rgba(34, 211, 238, 0.1) !important;
        border-radius: 8px !important;
        padding: 0.5rem !important;
        margin: 0.25rem 0 !important;
        min-width: 0 !important;
    }
    
    [data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] > div {
        min-width: 0 !important;
        overflow: hidden !important;
    }
    
    [data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] span {
        overflow: hidden !important;
        text-overflow: ellipsis !important;
        white-space: nowrap !important;
        max-width: 100% !important;
        display: block !important;
    }
    
    /* metrics */
    [data-testid="stMetricValue"] {
        color: var(--accent-cyan) !important;
        font-weight: 700 !important;
        font-size: 2.5rem !important;
    }
    
    [data-testid="stMetricLabel"] {
        color: var(--text-muted) !important;
    }
    
    /* expander */
    .streamlit-expanderHeader {
        background: var(--bg-card) !important;
        border-radius: 12px !important;
        color: var(--text-primary) !important;
        font-weight: 500 !important;
        border: 1px solid var(--border-subtle) !important;
    }
    
    .streamlit-expanderContent {
        background: var(--bg-card);
        border-radius: 0 0 12px 12px;
        border: 1px solid var(--border-subtle);
        border-top: none;
    }
    
    /* info and warning boxes */
    .stAlert {
        background: var(--bg-card) !important;
        border-radius: 14px !important;
        color: var(--text-secondary) !important;
        border: 1px solid var(--border-subtle) !important;
    }
    
    /* success message */
    div[data-testid="stAlert"][data-baseweb="notification"] {
        background: rgba(34, 197, 94, 0.15) !important;
        border: 1px solid rgba(34, 197, 94, 0.4) !important;
    }
    
    /* error message */
    .stError, .element-container .stAlert:has([data-testid="stAlertContentError"]) {
        background: rgba(239, 68, 68, 0.15) !important;
        border: 1px solid rgba(239, 68, 68, 0.4) !important;
    }
    
    /* caption text */
    .stCaption, [data-testid="stCaptionContainer"] {
        color: var(--text-muted) !important;
        font-size: 0.9rem !important;
    }
    
    /* custom dark card class with glow effect */
    .dark-card {
        background: var(--bg-card);
        border-radius: 20px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid var(--border-subtle);
        transition: all 0.3s ease;
    }
    
    .dark-card:hover {
        border-color: var(--border-accent);
        box-shadow: var(--shadow-glow-cyan);
        transform: translateY(-2px);
    }
    
    .dark-card h4 {
        color: var(--accent-cyan) !important;
        margin-top: 0 !important;
        margin-bottom: 0.75rem !important;
        font-size: 1.1rem !important;
        font-weight: 600 !important;
    }
    
    .dark-card p {
        color: var(--text-secondary) !important;
        margin: 0.5rem 0 !important;
        font-size: 0.95rem !important;
    }
    
    .dark-card strong {
        color: var(--text-primary) !important;
    }
    
    /* purple glow variant */
    .dark-card-purple:hover {
        border-color: rgba(168, 85, 247, 0.4);
        box-shadow: var(--shadow-glow-purple);
    }
    
    /* response card styling with enhanced glow */
    .response-card {
        background: linear-gradient(135deg, rgba(34, 211, 238, 0.1) 0%, rgba(168, 85, 247, 0.1) 100%);
        border-radius: 20px;
        padding: 1.75rem;
        margin: 1.5rem 0;
        border: 1px solid rgba(34, 211, 238, 0.3);
        box-shadow: var(--shadow-glow-mixed);
    }
    
    .response-card p {
        color: var(--text-primary) !important;
        font-size: 1rem !important;
        line-height: 1.8 !important;
    }
    
    /* chat messages with colored accents */
    .chat-user {
        background: linear-gradient(135deg, rgba(168, 85, 247, 0.15) 0%, rgba(236, 72, 153, 0.1) 100%);
        border-radius: 18px;
        padding: 1.25rem 1.5rem;
        margin: 1rem 0;
        border: 1px solid rgba(168, 85, 247, 0.3);
        border-left: 4px solid var(--accent-purple);
    }
    
    .chat-assistant {
        background: linear-gradient(135deg, rgba(34, 211, 238, 0.12) 0%, rgba(45, 212, 191, 0.08) 100%);
        border-radius: 18px;
        padding: 1.25rem 1.5rem;
        margin: 1rem 0;
        border: 1px solid rgba(34, 211, 238, 0.3);
        border-left: 4px solid var(--accent-cyan);
    }
    
    .chat-user p, .chat-assistant p {
        color: var(--text-primary) !important;
        margin: 0 !important;
        font-size: 0.95rem !important;
    }
    
    .chat-label {
        font-weight: 700 !important;
        color: var(--text-muted) !important;
        font-size: 0.75rem !important;
        margin-bottom: 0.5rem !important;
        text-transform: uppercase;
        letter-spacing: 0.1em;
    }
    
    /* sidebar brand header */
    .sidebar-header {
        padding: 1.5rem 0.5rem;
        margin-bottom: 2rem;
        text-align: center;
    }
    
    .sidebar-brand {
        font-size: 1.6rem !important;
        font-weight: 800 !important;
        background: var(--gradient-primary);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 0 !important;
        letter-spacing: -0.02em;
    }
    
    .sidebar-tagline {
        color: var(--text-muted) !important;
        font-size: 0.8rem !important;
        margin-top: 0.25rem !important;
    }
    
    /* sidebar nav title - plain text, not a button */
    .nav-title {
        color: var(--text-muted) !important;
        font-size: 0.75rem !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 0.15em;
        margin-bottom: 0.75rem !important;
        padding-left: 0.25rem;
    }
    
    /* hide default streamlit elements - but NOT the sidebar controls */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    /* Note: Don't hide header as it contains sidebar toggle controls */
    
    /* RADIO BUTTON STYLING - Hide the label completely */
    [data-testid="stSidebar"] .stRadio > div:first-child {
        display: none !important;
    }
    
    .stRadio > div {
        display: flex;
        flex-direction: column;
        gap: 0.5rem;
    }
    
    .stRadio > label {
        display: none !important;
    }
    
    .stRadio [role="radiogroup"] {
        gap: 0.5rem;
    }
    
    .stRadio [role="radiogroup"] > label {
        background: var(--bg-card) !important;
        border-radius: 14px !important;
        padding: 1rem 1.25rem !important;
        margin: 0 !important;
        cursor: pointer !important;
        transition: all 0.3s ease !important;
        border: 1px solid var(--border-subtle) !important;
        display: flex !important;
        align-items: center !important;
        color: var(--text-secondary) !important;
        font-weight: 500 !important;
        font-size: 0.95rem !important;
    }
    
    .stRadio [role="radiogroup"] > label:hover {
        background: var(--bg-card-hover) !important;
        border-color: var(--accent-cyan) !important;
        color: var(--text-primary) !important;
        box-shadow: var(--shadow-glow-cyan) !important;
        transform: translateX(4px);
    }
    
    .stRadio [role="radiogroup"] > label[data-checked="true"],
    .stRadio [role="radiogroup"] > label:has(input:checked) {
        background: linear-gradient(135deg, rgba(34, 211, 238, 0.2) 0%, rgba(168, 85, 247, 0.2) 100%) !important;
        border-color: var(--accent-cyan) !important;
        color: var(--text-primary) !important;
        box-shadow: var(--shadow-glow-cyan) !important;
    }
    
    /* divider with gradient */
    hr {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, var(--border-subtle), transparent);
        margin: 2rem 0;
    }
    
    /* spinner */
    .stSpinner > div {
        border-top-color: var(--accent-cyan) !important;
    }
    
    /* progress bar with gradient */
    .stProgress > div > div {
        background: var(--gradient-primary) !important;
        border-radius: 10px !important;
    }
    
    .stProgress > div {
        background: var(--bg-card) !important;
        border-radius: 10px !important;
    }
    
    /* tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.75rem;
        background: transparent;
        border-bottom: 2px solid var(--border-subtle);
        padding-bottom: 0;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 10px 10px 0 0;
        color: var(--text-muted);
        padding: 0.85rem 1.75rem;
        font-weight: 500;
        border: none;
        border-bottom: 3px solid transparent;
        transition: all 0.3s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        color: var(--text-primary);
        background: rgba(34, 211, 238, 0.05);
    }
    
    .stTabs [aria-selected="true"] {
        background: transparent !important;
        color: var(--accent-cyan) !important;
        border-bottom: 3px solid var(--accent-cyan) !important;
    }
    
    .stTabs [data-baseweb="tab-panel"] {
        padding-top: 2rem;
    }
    
    /* stat card for metrics with glow */
    .stat-card {
        background: var(--bg-card);
        border-radius: 20px;
        padding: 1.5rem;
        text-align: center;
        border: 1px solid var(--border-subtle);
        transition: all 0.3s ease;
    }
    
    .stat-card:hover {
        border-color: var(--accent-cyan);
        box-shadow: var(--shadow-glow-cyan);
        transform: translateY(-3px);
    }
    
    .stat-value {
        font-size: 2.5rem;
        font-weight: 800;
        background: var(--gradient-primary);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .stat-label {
        color: var(--text-muted);
        font-size: 0.9rem;
        font-weight: 500;
        margin-top: 0.5rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* section title styling */
    .section-title {
        color: var(--text-muted) !important;
        font-size: 0.75rem !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 0.15em;
        margin-bottom: 0.75rem !important;
        padding-left: 0.25rem;
    }
    
    /* feature card in sidebar */
    .feature-card {
        background: var(--bg-card);
        border-radius: 14px;
        padding: 1rem 1.25rem;
        margin: 0.5rem 0;
        border: 1px solid var(--border-subtle);
        transition: all 0.3s ease;
    }
    
    .feature-card:hover {
        border-color: var(--border-accent);
        box-shadow: var(--shadow-glow-cyan);
        transform: translateY(-2px);
    }
    
    .feature-card-purple:hover {
        border-color: rgba(168, 85, 247, 0.4);
        box-shadow: var(--shadow-glow-purple);
    }
    
    .feature-title {
        font-size: 0.95rem !important;
        font-weight: 600 !important;
        margin-bottom: 0.25rem !important;
    }
    
    .feature-desc {
        font-size: 0.8rem !important;
        color: var(--text-muted) !important;
        margin: 0 !important;
        line-height: 1.5 !important;
    }
    
    /* content container card */
    .content-card {
        background: var(--bg-card);
        border-radius: 24px;
        padding: 2rem;
        margin: 1.5rem 0;
        border: 1px solid var(--border-subtle);
        transition: all 0.3s ease;
    }
    
    .content-card:hover {
        border-color: rgba(34, 211, 238, 0.2);
    }
    
    .content-card-header {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        margin-bottom: 1rem;
        padding-bottom: 1rem;
        border-bottom: 1px solid var(--border-subtle);
    }
    
    .content-card-icon {
        width: 40px;
        height: 40px;
        background: var(--gradient-primary);
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.25rem;
        box-shadow: 0 0 20px rgba(34, 211, 238, 0.3);
    }
    
    .content-card-title {
        font-size: 1.2rem !important;
        font-weight: 600 !important;
        color: var(--text-primary) !important;
        margin: 0 !important;
    }
    
    /* accessibility - focus states */
    *:focus-visible {
        outline: 2px solid var(--accent-cyan) !important;
        outline-offset: 2px !important;
    }
    
    /* scrollbar styling */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: var(--bg-primary);
    }
    
    ::-webkit-scrollbar-thumb {
        background: var(--bg-card);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: var(--bg-card-hover);
    }
    
    /* reduced motion preference */
    @media (prefers-reduced-motion: reduce) {
        * {
            transition: none !important;
            animation: none !important;
        }
    }
    
    /* RESPONSIVE DESIGN - Handle narrow screens */
    @media (max-width: 768px) {
        .page-title {
            font-size: 2.5rem !important;
        }
        
        .page-subtitle {
            font-size: 1rem !important;
        }
        
        .content-card {
            padding: 1.25rem !important;
        }
        
        .content-card-title {
            font-size: 1rem !important;
        }
        
        .stFileUploader {
            padding: 1rem !important;
        }
        
        [data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"] {
            padding: 0.75rem !important;
        }
        
        [data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"] span {
            font-size: 0.85rem !important;
        }
        
        .stButton > button {
            padding: 0.6rem 1rem !important;
            font-size: 0.9rem !important;
        }
    }
    
    @media (max-width: 500px) {
        .page-title {
            font-size: 2rem !important;
        }
        
        .main .block-container {
            padding: 1rem !important;
        }
        
        .content-card {
            padding: 1rem !important;
            border-radius: 16px !important;
        }
        
        .dark-card {
            padding: 1rem !important;
            border-radius: 14px !important;
        }
    }
    
    /* SIDEBAR TOGGLE BUTTONS - keep visible but minimal */
    [data-testid="collapsedControl"] {
        display: flex !important;
        visibility: visible !important;
        opacity: 1 !important;
        position: fixed !important;
        left: 0.75rem !important;
        top: 0.75rem !important;
        z-index: 999999 !important;
    }

    [data-testid="collapsedControl"] > button,
    [data-testid="baseButton-header"] {
        background: #ffffff !important;
        border: 1px solid #cfd8e8 !important;
        border-radius: 10px !important;
        color: #1f2a44 !important;
        width: 42px !important;
        height: 42px !important;
        box-shadow: 0 2px 8px rgba(16, 24, 40, 0.08) !important;
    }

    [data-testid="collapsedControl"] > button:hover,
    [data-testid="baseButton-header"]:hover {
        background: #f8fbff !important;
        border-color: #2f6fec !important;
        color: #14213d !important;
        transform: none !important;
        box-shadow: 0 4px 12px rgba(47, 111, 236, 0.16) !important;
    }

    /* FINAL OVERRIDES: clean, light, bold, simplified UI */
    .stApp {
        background: linear-gradient(180deg, #f6f9ff 0%, #f3f7ff 100%) !important;
    }

    .main .block-container {
        padding: 1.5rem 2rem !important;
        max-width: 1220px !important;
    }

    [data-testid="stSidebar"] {
        background: #eef3fb !important;
        border-right: 1px solid #d3deef !important;
        box-shadow: none !important;
    }

    .sidebar-brand {
        color: #16233b !important;
        -webkit-text-fill-color: #16233b !important;
        background: none !important;
        font-size: 1.25rem !important;
        font-weight: 800 !important;
    }

    .sidebar-tagline {
        color: #4f6385 !important;
    }

    .nav-title {
        color: #334768 !important;
        font-size: 0.72rem !important;
        letter-spacing: 0.1em !important;
    }

    .feature-card,
    .dark-card,
    .content-card,
    .response-card,
    .stat-card {
        background: #ffffff !important;
        border: 1px solid #d8e2f0 !important;
        border-radius: 14px !important;
        box-shadow: 0 4px 14px rgba(18, 36, 76, 0.06) !important;
    }

    .feature-card:hover,
    .dark-card:hover,
    .content-card:hover,
    .stat-card:hover {
        transform: none !important;
        border-color: #bccbe4 !important;
        box-shadow: 0 6px 16px rgba(18, 36, 76, 0.09) !important;
    }

    .feature-title,
    .content-card-title,
    .chat-label {
        color: #14213d !important;
    }

    .feature-desc,
    .page-subtitle,
    .stCaption,
    [data-testid="stCaptionContainer"] {
        color: #4f6385 !important;
    }

    .page-title {
        color: #0f1d35 !important;
        background: none !important;
        -webkit-text-fill-color: #0f1d35 !important;
        font-size: 2.7rem !important;
        font-weight: 800 !important;
        letter-spacing: -0.01em !important;
    }

    .content-card-icon {
        background: #e8f0ff !important;
        color: #2d5fd5 !important;
        box-shadow: none !important;
    }

    .chat-user {
        background: #eef4ff !important;
        border: 1px solid #c9dafb !important;
        border-left: 4px solid #3f7cf0 !important;
    }

    .chat-assistant {
        background: #f3f6fb !important;
        border: 1px solid #d7e0ed !important;
        border-left: 4px solid #0f766e !important;
    }

    .chat-user p,
    .chat-assistant p,
    p,
    span,
    li {
        color: #243555 !important;
    }

    h1, h2, h3, h4, h5, h6,
    label, .stMarkdown, .stText, .stAlert, .stInfo, .stSuccess, .stWarning {
        color: #14213d !important;
    }

    h4 {
        color: #1d4ed8 !important;
    }

    .stTextInput > div > div > input,
    [data-testid="stFileUploader"],
    .streamlit-expanderHeader,
    .streamlit-expanderContent {
        background: #ffffff !important;
        border: 1px solid #ccd7ea !important;
        color: #14213d !important;
    }

    .stTextInput > div > div > input::placeholder {
        color: #7285a4 !important;
    }

    .stButton > button {
        background: #1d4ed8 !important;
        color: #ffffff !important;
        border: 1px solid #1e40af !important;
        border-radius: 10px !important;
        box-shadow: 0 4px 10px rgba(29, 78, 216, 0.2) !important;
        text-shadow: none !important;
        font-weight: 700 !important;
    }

    .stButton > button:hover {
        background: #1e40af !important;
        transform: none !important;
        box-shadow: 0 6px 12px rgba(29, 78, 216, 0.24) !important;
    }

    .stButton > button[kind="secondary"] {
        background: #ffffff !important;
        color: #1d4ed8 !important;
        border: 1px solid #b9c9e8 !important;
        box-shadow: none !important;
    }

    .stButton > button[kind="secondary"]:hover {
        background: #f8fbff !important;
        border-color: #1d4ed8 !important;
    }

    .stRadio > div {
        background: #ffffff !important;
        border: 1px solid #d3deef !important;
        border-radius: 12px !important;
        padding: 0.35rem !important;
    }

    .stRadio [role="radiogroup"] label {
        border-radius: 10px !important;
        padding: 0.35rem 0.5rem !important;
    }

    .stRadio [aria-checked="true"] {
        background: #e8f0ff !important;
    }

    .stTabs [data-baseweb="tab-list"] {
        border-bottom: 1px solid #ccd7ea !important;
    }

    .stTabs [data-baseweb="tab"] {
        color: #5a6f90 !important;
    }

    .stTabs [aria-selected="true"] {
        color: #14213d !important;
        border-bottom-color: #1d4ed8 !important;
    }

    .sidebar-chat-history {
        color: #263a5c !important;
    }

    [data-testid="stSidebar"] .stButton > button {
        background: #ffffff !important;
        border: 1px solid #ccd7ea !important;
        color: #1f2a44 !important;
        box-shadow: none !important;
    }

    [data-testid="stSidebar"] .stButton > button:hover {
        background: #f8fbff !important;
        border-color: #1d4ed8 !important;
        color: #14213d !important;
        box-shadow: none !important;
    }

    [data-testid="stSidebar"] .stButton > button[kind="primary"] {
        background: #1d4ed8 !important;
        border: 1px solid #1e40af !important;
        color: #ffffff !important;
    }

    .stRadio [role="radiogroup"] > label {
        background: #ffffff !important;
        border: 1px solid #ccd7ea !important;
        color: #263a5c !important;
        box-shadow: none !important;
    }

    .stRadio [role="radiogroup"] > label:hover {
        background: #f8fbff !important;
        border-color: #9ab4e7 !important;
        color: #14213d !important;
        transform: none !important;
        box-shadow: none !important;
    }

    .stRadio [role="radiogroup"] > label[data-checked="true"],
    .stRadio [role="radiogroup"] > label:has(input:checked) {
        background: #e8f0ff !important;
        border-color: #2f6fec !important;
        color: #14213d !important;
        box-shadow: none !important;
    }

    hr {
        background: linear-gradient(90deg, transparent, #cfd9ea, transparent) !important;
    }

    /* UX refresh: larger scale, pastel palette, quieter motion */
    :root {
        --pastel-bg: #f6f7fe;
        --pastel-surface: #ffffff;
        --pastel-surface-soft: #f1f4ff;
        --pastel-border: #d8def1;
        --pastel-text: #1f2940;
        --pastel-muted: #62718f;
        --pastel-primary: #7a8fe8;
        --pastel-primary-strong: #647ddf;
        --pastel-accent: #8bcfc3;
        --pastel-shadow: 0 10px 26px rgba(79, 96, 148, 0.10);
        --ease-out-smooth: cubic-bezier(0.22, 1, 0.36, 1);
    }

    .stApp {
        font-family: 'Plus Jakarta Sans', 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif !important;
        background: linear-gradient(180deg, #f6f7fe 0%, #f3f5ff 100%) !important;
        color: var(--pastel-text) !important;
    }

    .main .block-container {
        max-width: 1320px !important;
        padding: 2rem 2.25rem 2.25rem !important;
    }

    [data-testid="stSidebar"] {
        background: #edf1fb !important;
        border-right: 1px solid var(--pastel-border) !important;
        min-width: 300px !important;
        max-width: 340px !important;
    }

    .page-title {
        font-size: clamp(2.25rem, 3.4vw, 3.3rem) !important;
        font-weight: 800 !important;
        color: #222d46 !important;
        margin-bottom: 0.65rem !important;
        animation: fadeSlideIn 420ms var(--ease-out-smooth);
    }

    .page-subtitle {
        font-size: 1.08rem !important;
        line-height: 1.65 !important;
        color: var(--pastel-muted) !important;
        margin-bottom: 2rem !important;
    }

    .content-card,
    .feature-card,
    .dark-card,
    .response-card,
    .stat-card {
        background: var(--pastel-surface) !important;
        border: 1px solid var(--pastel-border) !important;
        border-radius: 16px !important;
        box-shadow: var(--pastel-shadow) !important;
        transition: transform 220ms var(--ease-out-smooth), box-shadow 220ms var(--ease-out-smooth), border-color 220ms var(--ease-out-smooth);
    }

    .content-card:hover,
    .feature-card:hover,
    .dark-card:hover,
    .response-card:hover,
    .stat-card:hover {
        transform: translateY(-2px) !important;
        border-color: #c6d0ee !important;
        box-shadow: 0 12px 30px rgba(79, 96, 148, 0.14) !important;
    }

    .content-card-title,
    .feature-title,
    .chat-label {
        font-size: 1.1rem !important;
        font-weight: 700 !important;
        color: var(--pastel-text) !important;
    }

    .feature-desc,
    p, span, li {
        color: var(--pastel-muted) !important;
        font-size: 1rem !important;
        line-height: 1.65 !important;
    }

    .stButton > button {
        background: linear-gradient(135deg, #8094ea 0%, #92a3ee 100%) !important;
        border: 1px solid #748add !important;
        color: #ffffff !important;
        font-size: 1rem !important;
        font-weight: 700 !important;
        min-height: 48px !important;
        padding: 0.75rem 1.05rem !important;
        border-radius: 12px !important;
        box-shadow: 0 8px 18px rgba(101, 124, 209, 0.24) !important;
        transition: transform 170ms var(--ease-out-smooth), box-shadow 170ms var(--ease-out-smooth), background 170ms var(--ease-out-smooth) !important;
    }

    .stButton > button:hover {
        transform: translateY(-1px) !important;
        background: linear-gradient(135deg, #7289e4 0%, #869aeb 100%) !important;
        box-shadow: 0 10px 22px rgba(101, 124, 209, 0.30) !important;
    }

    .stButton > button:active {
        transform: translateY(0) scale(0.99) !important;
    }

    .stTextInput > div > div > input {
        background: #ffffff !important;
        border: 1px solid #cfd8f0 !important;
        border-radius: 12px !important;
        min-height: 50px !important;
        font-size: 1rem !important;
        color: #1f2940 !important;
        transition: border-color 180ms var(--ease-out-smooth), box-shadow 180ms var(--ease-out-smooth), transform 180ms var(--ease-out-smooth) !important;
    }

    .stTextInput > div > div > input:focus {
        border-color: #8ea0ea !important;
        box-shadow: 0 0 0 4px rgba(122, 143, 232, 0.16) !important;
        transform: translateY(-1px);
    }

    [data-testid="stSidebar"] .stButton > button {
        background: #ffffff !important;
        border: 1px solid var(--pastel-border) !important;
        color: #30405f !important;
        min-height: 42px !important;
    }

    [data-testid="stSidebar"] .stButton > button:hover {
        background: #f6f8ff !important;
        border-color: #b7c5ee !important;
        color: #243451 !important;
        transform: translateY(-1px) !important;
    }

    [data-testid="stSidebar"] .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #8094ea 0%, #92a3ee 100%) !important;
        border: 1px solid #748add !important;
        color: #ffffff !important;
    }

    .chat-user {
        background: #edf2ff !important;
        border: 1px solid #cad5f3 !important;
        border-left: 4px solid #7b90e8 !important;
    }

    .chat-assistant {
        background: #f2f7f6 !important;
        border: 1px solid #d7e7e3 !important;
        border-left: 4px solid #85cabc !important;
    }

    .tools-used-pill {
        display: inline-flex;
        flex-wrap: wrap;
        gap: 0.35rem;
        margin-top: 0.6rem;
        font-size: 0.75rem;
        color: var(--pastel-muted);
    }
    .tools-used-pill span {
        background: #eef2ff;
        border: 1px solid #c8d2f1;
        border-radius: 999px;
        padding: 0.2rem 0.6rem;
        font-weight: 500;
        color: #4f6385;
    }

    [data-testid="stFileUploader"] {
        background: #ffffff !important;
        border: 1px dashed #b8c6ea !important;
        border-radius: 14px !important;
        padding: 0.85rem !important;
        min-height: 170px !important;
        transition: border-color 200ms var(--ease-out-smooth), box-shadow 200ms var(--ease-out-smooth) !important;
    }

    [data-testid="stFileUploader"]:hover {
        border-color: #8fa2e9 !important;
        box-shadow: 0 8px 18px rgba(122, 143, 232, 0.15) !important;
    }

    [data-testid="stFileUploaderDropzone"] {
        min-height: 140px !important;
        align-items: center !important;
        justify-content: center !important;
        text-align: center !important;
        padding: 0.75rem !important;
    }

    [data-testid="stFileUploaderDropzone"] * {
        white-space: normal !important;
        overflow-wrap: anywhere !important;
        word-break: break-word !important;
        font-size: 0.98rem !important;
    }

    .onboard-panel {
        background: var(--pastel-surface-soft);
        border: 1px solid #cfdaef;
        border-radius: 14px;
        padding: 0.95rem 1rem;
        margin-bottom: 0.85rem;
    }

    .onboard-title {
        font-size: 0.95rem;
        font-weight: 700;
        color: #2b3a57;
        margin: 0 0 0.5rem 0;
    }

    .onboard-step {
        display: flex;
        gap: 0.5rem;
        margin: 0.35rem 0;
        color: #4f6286;
        font-size: 0.9rem;
    }

    .history-empty-note {
        color: #6b7da1 !important;
        font-size: 0.83rem !important;
        margin: 0.4rem 0 0 0 !important;
        padding: 0.25rem 0.2rem 0 0.2rem;
    }

    .chat-history-active {
        background: #edf2ff !important;
        border-color: #c3d1f4 !important;
    }

    .sidebar-footer {
        text-align: center;
        padding: 1rem 0;
    }

    .sidebar-footer-label {
        color: #5a6f90;
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin: 0;
    }

    .sidebar-footer-value {
        color: #1f2a44;
        font-size: 0.85rem;
        font-weight: 600;
        margin: 0.5rem 0 0 0;
    }

    .spacer-xs {
        height: 0.5rem;
    }

    .spacer-sm {
        height: 1rem;
    }

    .spacer-md {
        height: 1.5rem;
    }

    .chat-history-card {
        padding: 0.72rem 0.88rem;
        margin: 0.3rem 0;
    }

    .chat-history-title {
        margin: 0;
        font-size: 0.93rem;
        color: #1f2940;
        font-weight: 700;
    }

    .chat-history-date {
        margin: 0.25rem 0 0 0;
        font-size: 0.74rem;
        color: #6b7da1;
    }

    .text-muted-sm {
        color: #4f6385;
        margin: 0;
        font-size: 0.95rem;
    }

    .text-muted-xs {
        color: #4f6385;
        margin: 0;
        font-size: 0.9rem;
    }

    .pre-wrap {
        white-space: pre-wrap;
    }

    .no-margin-card {
        margin: 0;
    }

    .content-card-spaced {
        margin-bottom: 1.5rem;
    }

    .content-card-icon-search {
        background: #eaf7f5;
        color: #0f766e;
    }

    .context-pill {
        padding: 0.75rem 1rem;
        margin-bottom: 1rem;
        background: #eef4ff;
        border-color: #c9d9f6;
    }

    .context-pill-text {
        margin: 0;
        font-size: 0.8rem;
        color: #4f6385;
    }

    .context-pill-strong-primary {
        color: #1d4ed8;
    }

    .context-pill-strong-accent {
        color: #0f766e;
    }

    .section-header-wrap {
        margin-bottom: 1.5rem;
    }

    .section-subtitle {
        color: #4f6385;
        font-size: 0.9rem;
        margin-top: 0.5rem;
    }

    .heading-inline-top {
        margin-top: 1rem;
    }

    /* FINAL POLISH: balanced sizing, elegant typography, quieter palette */
    html, body, [data-testid="stAppViewContainer"], .stApp {
        font-size: 16px !important;
    }

    .stApp {
        background: linear-gradient(180deg, #fbfaff 0%, #f7f8ff 55%, #f8fbff 100%) !important;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f3f0fb 0%, #edf1fd 100%) !important;
        border-right: 1px solid #d9dcef !important;
    }

    [data-testid="stSidebar"][aria-expanded="true"] {
        min-width: 290px !important;
        max-width: 290px !important;
    }

    [data-testid="stSidebar"][aria-expanded="false"] {
        min-width: 0 !important;
        max-width: 0 !important;
    }

    .sidebar-brand {
        font-family: 'Playfair Display', Georgia, serif !important;
        font-size: 1.85rem !important;
        font-weight: 600 !important;
        color: #1f2540 !important;
        -webkit-text-fill-color: #1f2540 !important;
        letter-spacing: -0.01em !important;
    }

    .sidebar-tagline {
        color: #6b7397 !important;
        font-size: 0.85rem !important;
    }

    .main .block-container {
        max-width: 1260px !important;
        padding: 1.9rem 2.15rem 2.35rem !important;
    }

    .page-title {
        font-family: 'Playfair Display', Georgia, serif !important;
        font-size: clamp(2.35rem, 3.5vw, 3.25rem) !important;
        font-weight: 600 !important;
        letter-spacing: -0.01em !important;
        color: #1f2540 !important;
    }

    .page-subtitle {
        font-size: 1.08rem !important;
        color: #5f688f !important;
    }

    .nav-title {
        color: #2c3555 !important;
        font-weight: 800 !important;
        letter-spacing: 0.12em !important;
    }

    .content-card,
    .feature-card,
    .dark-card,
    .response-card,
    .stat-card {
        border-radius: 16px !important;
        border: 1px solid #d9def3 !important;
        background: #ffffff !important;
    }

    .content-card-header {
        margin-bottom: 1rem !important;
        padding-bottom: 0.9rem !important;
    }

    .content-card-title,
    .feature-title {
        font-size: 1.14rem !important;
        font-weight: 700 !important;
        color: #26304d !important;
    }

    .feature-desc,
    .text-muted-sm,
    .text-muted-xs,
    p, span, li {
        font-size: 0.98rem !important;
        line-height: 1.65 !important;
        color: #434b64 !important;
    }

    .stButton > button {
        min-height: 48px !important;
        border-radius: 12px !important;
        font-size: 0.98rem !important;
        background: #101726 !important;
        border: 1px solid #101726 !important;
        color: #ffffff !important;
        box-shadow: 0 8px 20px rgba(16, 23, 38, 0.18) !important;
        transition: transform 170ms var(--ease-out-smooth), box-shadow 170ms var(--ease-out-smooth), background 170ms var(--ease-out-smooth) !important;
    }

    .stButton > button * {
        color: #ffffff !important;
    }

    .stButton > button:hover {
        transform: translateY(-1px) !important;
        background: #0b1220 !important;
        box-shadow: 0 10px 24px rgba(16, 23, 38, 0.24) !important;
        filter: none !important;
    }

    .stButton > button[kind="secondary"] {
        background: #ffffff !important;
        color: #1f2540 !important;
        border: 1px solid #ccd4eb !important;
        box-shadow: 0 4px 12px rgba(31, 37, 64, 0.08) !important;
    }

    .stButton > button[kind="secondary"] * {
        color: #1f2540 !important;
    }

    .stButton > button[kind="secondary"]:hover {
        background: #f7f9ff !important;
        border-color: #b8c4e5 !important;
    }

    /* COLORIZE OVERRIDE: dark pink accent system */
    :root {
        --accent-pink-700: #97165b;
        --accent-pink-600: #ad1d6a;
        --accent-pink-500: #c02a78;
        --accent-pink-200: #efc7dc;
        --accent-pink-100: #f7e3ee;
    }

    .home-kicker {
        color: var(--accent-pink-600) !important;
    }

    .stButton > button {
        background: var(--accent-pink-600) !important;
        border-color: var(--accent-pink-600) !important;
        color: #ffffff !important;
        box-shadow: 0 8px 18px rgba(173, 29, 106, 0.22) !important;
    }

    .stButton > button * {
        color: #ffffff !important;
    }

    .stButton > button:hover {
        background: var(--accent-pink-700) !important;
        border-color: var(--accent-pink-700) !important;
        box-shadow: 0 10px 22px rgba(151, 22, 91, 0.28) !important;
    }

    .stButton > button[kind="secondary"] {
        background: #ffffff !important;
        color: var(--accent-pink-700) !important;
        border-color: var(--accent-pink-200) !important;
        box-shadow: 0 4px 12px rgba(173, 29, 106, 0.08) !important;
    }

    .stButton > button[kind="secondary"] * {
        color: var(--accent-pink-700) !important;
    }

    .stButton > button[kind="secondary"]:hover {
        background: #fff8fc !important;
        border-color: var(--accent-pink-500) !important;
    }

    .stTextInput > div > div > input:focus,
    [data-testid="stTextInputRootElement"] input:focus {
        border-color: var(--accent-pink-500) !important;
        box-shadow: 0 0 0 4px rgba(192, 42, 120, 0.16) !important;
    }

    [data-testid="stSidebar"] .stRadio [role="radiogroup"] > label[data-checked="true"],
    [data-testid="stSidebar"] .stRadio [role="radiogroup"] > label:has(input:checked) {
        background: var(--accent-pink-700) !important;
        border-color: var(--accent-pink-700) !important;
        color: #ffffff !important;
        box-shadow: 0 10px 20px rgba(151, 22, 91, 0.24) !important;
    }

    [data-testid="stSidebar"] .stRadio [role="radiogroup"] > label:hover {
        border-color: var(--accent-pink-500) !important;
    }

    .content-card-icon,
    .content-card-icon-search {
        background: var(--accent-pink-100) !important;
        color: var(--accent-pink-700) !important;
    }

    .chat-user {
        border-left-color: var(--accent-pink-600) !important;
        background: #fbf0f7 !important;
        border-color: var(--accent-pink-200) !important;
    }

    .context-pill {
        background: #fdf3f8 !important;
        border-color: var(--accent-pink-200) !important;
    }

    .context-pill-strong-primary {
        color: var(--accent-pink-700) !important;
    }

    .stat-value {
        color: var(--accent-pink-700) !important;
    }

    .section-header-icon {
        background: var(--accent-pink-600) !important;
        box-shadow: 0 0 0 4px rgba(173, 29, 106, 0.16) !important;
    }

    /* bolder select mode controls in sidebar */
    [data-testid="stSidebar"] .stRadio > div {
        background: transparent !important;
        border: none !important;
        padding: 0 !important;
    }

    [data-testid="stSidebar"] .stRadio [role="radiogroup"] {
        gap: 0.6rem !important;
    }

    [data-testid="stSidebar"] .stRadio [role="radiogroup"] > label {
        background: #eef2ff !important;
        border: 1px solid #c8d2f1 !important;
        border-radius: 12px !important;
        min-height: 46px !important;
        padding: 0.7rem 0.85rem !important;
        color: #2f385a !important;
        font-weight: 700 !important;
        transition: transform 170ms var(--ease-out-smooth), background 170ms var(--ease-out-smooth), color 170ms var(--ease-out-smooth), box-shadow 170ms var(--ease-out-smooth) !important;
    }

    [data-testid="stSidebar"] .stRadio [role="radiogroup"] > label input {
        display: none !important;
        opacity: 0 !important;
        width: 0 !important;
        height: 0 !important;
        margin: 0 !important;
    }

    [data-testid="stSidebar"] .stRadio [role="radiogroup"] > label > div:first-child {
        display: none !important;
    }

    [data-testid="stSidebar"] .stRadio [role="radiogroup"] > label:hover {
        background: #f7e8f1 !important;
        border-color: #e3b3cb !important;
        transform: translateX(2px) !important;
        box-shadow: 0 6px 16px rgba(151, 22, 91, 0.10) !important;
    }

    [data-testid="stSidebar"] .stRadio [role="radiogroup"] > label[data-checked="true"],
    [data-testid="stSidebar"] .stRadio [role="radiogroup"] > label:has(input:checked) {
        background: #f6dce8 !important;
        border-color: #d888b1 !important;
        color: #79124a !important;
        box-shadow: 0 8px 18px rgba(151, 22, 91, 0.14) !important;
    }

    [data-testid="stSidebar"] .stRadio [role="radiogroup"] > label[data-checked="true"] *,
    [data-testid="stSidebar"] .stRadio [role="radiogroup"] > label:has(input:checked) * {
        color: #79124a !important;
    }

    .stTextInput > div > div > input {
        min-height: 58px !important;
        border-radius: 12px !important;
        font-size: 1rem !important;
        line-height: 58px !important;
        padding: 0 1rem !important;
        border: 1px solid #ccd3ee !important;
        background: #ffffff !important;
        box-sizing: border-box !important;
    }

    .stTextInput > div > div > input:focus {
        box-shadow: 0 0 0 4px rgba(161, 154, 237, 0.16) !important;
        border-color: #a6a0ee !important;
    }

    /* keep all query bars white and vertically centered */
    [data-testid="stTextInputRootElement"] input {
        background: #ffffff !important;
        color: #25314e !important;
        height: 58px !important;
        min-height: 58px !important;
        line-height: 58px !important;
        padding: 0 1rem !important;
        box-sizing: border-box !important;
        margin: 0 !important;
    }

    [data-testid="stTextInputRootElement"] input::placeholder {
        color: #556083 !important;
        line-height: 58px !important;
    }

    /* remove default form box so composer feels clean */
    [data-testid="stForm"] {
        border: none !important;
        padding: 0 !important;
        background: transparent !important;
    }

    .chat-user,
    .chat-assistant {
        border-radius: 14px !important;
        padding: 0.95rem 1rem !important;
        margin-bottom: 0.72rem !important;
    }

    .chat-label {
        font-size: 0.9rem !important;
        margin-bottom: 0.3rem !important;
    }

    .stat-value {
        color: #97165b !important;
        background: none !important;
        -webkit-text-fill-color: #97165b !important;
        background-clip: border-box !important;
        text-shadow: none !important;
        font-weight: 800 !important;
    }

    .feature-card,
    .content-card,
    .response-card {
        animation: fadeSlideIn 300ms var(--ease-out-smooth);
    }

    .feature-card {
        border-color: #d2d9f1 !important;
    }

    .feature-card:hover {
        border-color: #1f2744 !important;
        box-shadow: 0 12px 26px rgba(31, 39, 68, 0.15) !important;
    }

    [data-testid="stSidebar"] .stButton > button {
        min-height: 44px !important;
        font-size: 0.92rem !important;
    }

    .personal-composer-wrap {
        max-width: 860px;
        margin: 0 auto;
    }

    .personal-composer-actions {
        max-width: 860px;
        margin: 0.7rem auto 0 auto;
    }

    .personal-composer-actions .stButton > button {
        min-height: 50px !important;
        font-weight: 700 !important;
    }

    .resume-composer-wrap {
        max-width: 860px;
        margin: 0 auto;
    }

    /* AUTHORITATIVE INPUT ALIGNMENT FIX (final) */
    [data-testid="stTextInputRootElement"] > div,
    [data-baseweb="input"] {
        display: flex !important;
        align-items: center !important;
        min-height: 58px !important;
    }

    .stTextInput input,
    [data-baseweb="input"] input,
    [data-testid="stTextInputRootElement"] input {
        height: 58px !important;
        min-height: 58px !important;
        line-height: 58px !important;
        padding: 0 1rem !important;
        margin: 0 !important;
        box-sizing: border-box !important;
    }

    .stTextInput input::placeholder,
    [data-baseweb="input"] input::placeholder,
    [data-testid="stTextInputRootElement"] input::placeholder {
        line-height: 58px !important;
    }

    .resume-composer-note {
        text-align: center;
        color: #4a5679;
        font-size: 0.9rem;
        margin: 0.45rem 0 0.15rem 0;
    }

    .empty-state-card {
        max-width: 860px;
        margin: 0 auto 0.6rem auto;
        background: #ffffff;
        border: 1px solid #d9def3;
        border-radius: 14px;
        padding: 1rem 1.05rem;
        color: #5f688f;
    }

    .home-hero {
        max-width: 980px;
        margin: 0 auto 1.8rem auto;
        padding: 1.25rem 0 0.3rem;
        animation: fadeSlideIn 340ms var(--ease-out-smooth);
    }

    .home-kicker {
        color: #8d5db8;
        font-size: 0.86rem;
        font-weight: 700;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        margin-bottom: 0.5rem;
    }

    .home-title {
        font-family: 'Playfair Display', Georgia, serif;
        color: #1f2540;
        font-size: clamp(2.1rem, 4.2vw, 3.1rem);
        margin: 0 0 0.65rem 0;
    }

    .home-subtitle {
        color: #495472;
        font-size: 1.04rem;
        line-height: 1.7;
        margin: 0 0 1.2rem 0;
        max-width: 62ch;
    }

    .home-actions {
        max-width: 980px;
        margin: 0 auto 0.8rem auto;
        animation: fadeSlideIn 420ms var(--ease-out-smooth);
    }

    .home-delight {
        max-width: 980px;
        margin: 0 auto 1rem auto;
        color: #5b6587;
        font-size: 0.92rem;
    }

    .tab-enter,
    .personal-composer-wrap,
    .resume-composer-wrap,
    .onboard-panel,
    .empty-state-card,
    .chat-user,
    .chat-assistant,
    [data-baseweb="tab-panel"] {
        animation: fadeSlideIn 300ms var(--ease-out-smooth);
    }

    [data-testid="stFileUploader"] {
        min-height: 190px !important;
        border-style: dashed !important;
        border-width: 2px !important;
        border-color: #b9c3ea !important;
        background: #fbfcff !important;
    }

    /* remove gradient look from file uploader button */
    [data-testid="stFileUploaderDropzone"] button {
        background: #101726 !important;
        color: #ffffff !important;
        border: 1px solid #101726 !important;
        border-radius: 11px !important;
        min-height: 42px !important;
        box-shadow: 0 6px 14px rgba(16, 23, 38, 0.18) !important;
    }

    [data-testid="stFileUploaderDropzone"] button:hover {
        background: #0b1220 !important;
        color: #ffffff !important;
        border-color: #0b1220 !important;
    }

    @media (max-width: 1100px) {
        .main .block-container {
            padding: 1.3rem 1rem 1.6rem !important;
        }

        .page-title {
            font-size: clamp(2.05rem, 7.2vw, 2.75rem) !important;
        }
    }

    @keyframes fadeSlideIn {
        from { opacity: 0; transform: translateY(8px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .content-card,
    .chat-user,
    .chat-assistant,
    .response-card {
        animation: fadeSlideIn 320ms var(--ease-out-smooth);
    }

    @media (max-width: 900px) {
        .main .block-container {
            padding: 1.15rem 1rem 1.6rem !important;
        }

        .page-title {
            font-size: clamp(2rem, 8vw, 2.5rem) !important;
        }

        .content-card-title,
        .feature-title {
            font-size: 1.02rem !important;
        }

        .feature-desc,
        p, span, li {
            font-size: 0.96rem !important;
        }
    }
</style>
""", unsafe_allow_html=True)

# sidebar navigation with dark theme
# handle home CTA navigation targets BEFORE radio widget is created
if "nav_mode" not in st.session_state:
    st.session_state.nav_mode = "Home"
if "nav_target" in st.session_state:
    st.session_state.nav_mode = st.session_state.nav_target
    del st.session_state["nav_target"]

with st.sidebar:
    # brand header with gradient
    st.markdown("""
    <div class="sidebar-header">
        <p class="sidebar-brand">Chat & Analyze!</p>
        <p class="sidebar-tagline">Personal chat and resume analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
    # navigation section - plain text title, not a button
    st.markdown('<p class="nav-title">Select Mode</p>', unsafe_allow_html=True)
    
    # navigation using radio buttons for accessibility
    page = st.radio(
        "nav",
        ["Home", "Personal Chat", "Resume Analyzer"],
        key="nav_mode",
        label_visibility="hidden"
    )
    
    st.markdown("---")
    
    # features section
    st.markdown('<p class="nav-title">Features</p>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="feature-card">
        <p class="feature-title">Personal Chat</p>
        <p class="feature-desc">Ask questions about Roxy's background, skills, and experience</p>
    </div>
    <div class="feature-card feature-card-purple">
        <p class="feature-title">Resume Analyzer</p>
        <p class="feature-desc">Upload and analyze resumes with AI-powered insights</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # footer info
    st.markdown("""
    <div class="sidebar-footer">
        <p class="sidebar-footer-label">
            Powered by
        </p>
        <p class="sidebar-footer-value">
            FAISS • OpenAI • Streamlit
        </p>
    </div>
    """, unsafe_allow_html=True)

# main content area - dynamic title based on page with large styled titles
# using <div> instead of <h1> to avoid Streamlit's default h1 styling override
if page == "Home":
    st.markdown("""
    <div class="page-title">Welcome</div>
    <div class="page-subtitle">Choose a mode to talk to Roxy's personal chatbot or analyze candidate resumes.</div>
    """, unsafe_allow_html=True)
elif page == "Personal Chat":
    st.markdown("""
    <div class="page-title">Personal Chat</div>
    <div class="page-subtitle">Ask me anything about Roxy's background, skills, and experience</div>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <div class="page-title">Resume Analyzer</div>
    <div class="page-subtitle">Upload resumes and get AI-powered insights and comparisons</div>
    """, unsafe_allow_html=True)

st.markdown("---")

# home page
if page == "Home":
    st.markdown("""
    <div class="home-hero tab-enter">
        <p class="home-kicker">Smart Assistant</p>
        <h2 class="home-title">One workspace for personal chat and resume analysis</h2>
        <p class="home-subtitle">
            Use Personal Chat to talk to Roxy's personal chatbot, or open Resume Analyzer to upload candidate resumes and compare insights.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col_home_a, col_home_b = st.columns(2, gap="large")
    with col_home_a:
        if st.button("Open Personal Chat", key="home_to_personal", use_container_width=True):
            st.session_state.nav_target = "Personal Chat"
            st.rerun()
    with col_home_b:
        if st.button("Open Resume Analyzer", key="home_to_resume", use_container_width=True):
            st.session_state.nav_target = "Resume Analyzer"
            st.rerun()

    st.markdown("""
    <p class="home-delight">Tip: Start with Personal Chat for quick profile Q&A, then switch to Resume Analyzer when you need candidate comparisons.</p>
    """, unsafe_allow_html=True)

# personal chat page
elif page == "Personal Chat":
    # load the resources for semantic search and inject into tools
    model, index, chunks = load_resources()
    init_resources(model, index, chunks)

    # initialize session state for personal chat history
    if 'personal_chats' not in st.session_state:
        st.session_state.personal_chats = {}  # {chat_id: {messages: [], title: str, created: str}}
    if 'personal_current_chat_id' not in st.session_state:
        st.session_state.personal_current_chat_id = None
    if 'personal_input_key' not in st.session_state:
        st.session_state.personal_input_key = 0
    
    # helper function to create a new chat
    def create_new_personal_chat(force_new: bool = False):
        import datetime

        # avoid creating duplicate empty placeholder chats unless explicitly requested
        current_id = st.session_state.personal_current_chat_id
        if not force_new and current_id and current_id in st.session_state.personal_chats:
            existing = st.session_state.personal_chats[current_id]
            if not existing.get('messages'):
                return current_id

        chat_id = f"chat_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        st.session_state.personal_chats[chat_id] = {
            'messages': [],
            'title': 'Untitled Chat',
            'created': datetime.datetime.now().strftime('%b %d, %I:%M %p')
        }
        st.session_state.personal_current_chat_id = chat_id
        st.session_state.personal_input_key += 1
        return chat_id
    
    # create first chat if none exists
    if not st.session_state.personal_current_chat_id:
        create_new_personal_chat()
    
    # get current chat
    current_chat_id = st.session_state.personal_current_chat_id
    current_chat = st.session_state.personal_chats.get(current_chat_id, {'messages': [], 'title': 'Untitled Chat'})
    
    # sidebar section for chat history
    with st.sidebar:
        st.markdown("---")
        st.markdown('<p class="nav-title">Chat History</p>', unsafe_allow_html=True)
        
        # new chat button
        if st.button("+ New Chat", key="new_personal_chat", use_container_width=True):
            create_new_personal_chat(force_new=True)
            st.rerun()
        
        st.markdown("<div class='spacer-xs'></div>", unsafe_allow_html=True)
        
        # list of past chats
        if st.session_state.personal_chats:
            visible_chats = []
            for chat_id, chat_data in st.session_state.personal_chats.items():
                if chat_id == current_chat_id or chat_data.get('messages'):
                    visible_chats.append((chat_id, chat_data))

            for chat_id, chat_data in reversed(visible_chats):
                is_current = chat_id == current_chat_id
                # prefer meaningful titles; avoid repeating generic "new chat" items
                raw_title = chat_data.get('title', 'Untitled Chat')
                if raw_title.strip().lower() in {'new chat', 'untitled chat'} and chat_data.get('messages'):
                    first_user_msg = next((m['content'] for m in chat_data['messages'] if m.get('role') == 'user'), '')
                    raw_title = first_user_msg[:35] if first_user_msg else 'Untitled Chat'

                title = raw_title[:32]
                if len(raw_title) > 32:
                    title += '...'

                if is_current:
                    st.markdown(f"""
                    <div class="feature-card chat-history-active chat-history-card">
                        <p class="chat-history-title">{title}</p>
                        <p class="chat-history-date">{chat_data.get('created', '')}</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    if st.button(f"💬 {title}", key=f"switch_{chat_id}", use_container_width=True):
                        st.session_state.personal_current_chat_id = chat_id
                        st.session_state.personal_input_key += 1
                        st.rerun()
        else:
            st.markdown('<p class="history-empty-note">Your previous chats will appear here.</p>', unsafe_allow_html=True)
    
    # display chat messages
    if current_chat['messages']:
        for msg in current_chat['messages']:
            if msg['role'] == 'user':
                st.markdown(f"""
                <div class="chat-user">
                    <div class="chat-label">You</div>
                    <p>{msg['content']}</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                tools_used = msg.get("tools_used") or []
                tool_labels = {"semantic_search_personal": "my info", "get_weather": "weather", "web_search": "web search", "github_search": "GitHub"}
                pill_spans = "".join([f'<span>{tool_labels.get(t, t)}</span>' for t in tools_used])
                pill_html = f'<div class="tools-used-pill">Used: {pill_spans}</div>' if pill_spans else ""
                st.markdown(f"""
                <div class="chat-assistant">
                    <div class="chat-label">Roxy</div>
                    <p class="pre-wrap">{msg['content']}</p>
                    {pill_html}
                </div>
                """, unsafe_allow_html=True)
                # show context if available (legacy)
                if msg.get('context'):
                    with st.expander("View Retrieved Context"):
                        st.markdown(f"""
                        <div class="dark-card no-margin-card">
                            <p class="text-muted-xs">{msg['context']}</p>
                        </div>
                        """, unsafe_allow_html=True)
        
        st.markdown("---")
    
    # centered composer with explicit Ask button (Enter still works)
    with st.container():
        st.markdown("<div class='personal-composer-wrap'>", unsafe_allow_html=True)
        with st.form(key=f"personal_form_{st.session_state.personal_input_key}", clear_on_submit=False):
            query = st.text_input(
                "Your Question",
                placeholder="What are Roxy's skills? What's her background?",
                key=f"personal_query_{st.session_state.personal_input_key}",
                label_visibility="collapsed"
            )
            submitted = st.form_submit_button("Ask", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # process query only on submit (with tool-calling loop)
    if submitted and query.strip():
        today_str = datetime.now().strftime("%A, %B %d, %Y")
        PERSONAL_SYSTEM_PROMPT = f"""You are Roxy. You are in a conversation with a recruiter. Answer in first person from your own experience and background. Be conversational and friendly.

Today's date is {today_str}. Use this when answering questions about "this year", "2025", "upcoming", "current", or time-sensitive topics so your answers are accurate.

Use the semantic_search_personal tool to look up your resume and personal info whenever you need to answer questions about your skills, experience, education, or background. You can call it multiple times with different queries if needed.

Use other tools only when relevant:
- get_weather: for small talk about weather (e.g. "What's the weather like today?").
- web_search: for current events or general facts outside your resume.
- github_search: for your GitHub or anyone's—use query "user:USERNAME" to find that person's repos (e.g. user:roxystory for yours). Only claim a repo as yours if the owner in the result is actually you.

If a tool is unavailable or fails, answer from memory or say you're not sure. Always respond as Roxy in first person."""

        messages = [{"role": "system", "content": PERSONAL_SYSTEM_PROMPT}]
        for m in current_chat["messages"]:
            messages.append({"role": m["role"], "content": m["content"]})
        messages.append({"role": "user", "content": query.strip()})

        tools = get_openai_tools()
        tools_used_this_turn = []

        with st.spinner("Thinking..."):
            while True:
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=messages,
                    tools=tools,
                    tool_choice="auto",
                )
                msg = response.choices[0].message
                if not getattr(msg, "tool_calls", None) or not msg.tool_calls:
                    answer = msg.content or ""
                    break
                # one assistant message with all tool_calls
                tool_calls_for_api = [
                    {"id": tc.id, "type": "function", "function": {"name": tc.function.name, "arguments": tc.function.arguments or "{}"}}
                    for tc in msg.tool_calls
                ]
                messages.append({"role": "assistant", "content": msg.content or None, "tool_calls": tool_calls_for_api})
                for tc in msg.tool_calls:
                    name = tc.function.name
                    try:
                        args = json.loads(tc.function.arguments) if tc.function.arguments else {}
                    except json.JSONDecodeError:
                        args = {}
                    tools_used_this_turn.append(name)
                    result = run_tool(name, args)
                    messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})

        # add to chat history
        st.session_state.personal_chats[current_chat_id]["messages"].append({"role": "user", "content": query.strip()})
        st.session_state.personal_chats[current_chat_id]["messages"].append({
            "role": "assistant",
            "content": answer,
            "tools_used": tools_used_this_turn,
        })

        # update chat title based on first question
        if st.session_state.personal_chats[current_chat_id]["title"] in {"New Chat", "Untitled Chat"}:
            st.session_state.personal_chats[current_chat_id]["title"] = query[:30] + ("..." if len(query) > 30 else "")

        st.session_state.personal_input_key += 1
        st.rerun()

    # clear chat action moved to the bottom of personal chat page
    st.markdown("<div class='personal-composer-actions'>", unsafe_allow_html=True)
    if st.button("🗑️ Clear Chat", key="clear_personal_chat_bottom", use_container_width=True):
        if current_chat_id in st.session_state.personal_chats:
            st.session_state.personal_chats[current_chat_id]['messages'] = []
            st.session_state.personal_chats[current_chat_id]['title'] = 'Untitled Chat'
        st.session_state.personal_input_key += 1
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

# resume analyzer page
elif page == "Resume Analyzer":
    resume_manager = get_resume_manager()
    
    # initialize conversation context for tracking candidates
    init_conversation_context()
    if 'resume_input_key' not in st.session_state:
        st.session_state.resume_input_key = 0
    
    # two column layout for upload and chat
    col_upload, col_chat = st.columns([1.05, 1.95], gap="large")
    
    with col_upload:
        st.markdown("""
        <div class="content-card content-card-spaced">
            <div class="content-card-header">
                <div class="content-card-icon">📄</div>
                <p class="content-card-title">Upload Resumes</p>
            </div>
            <p class="text-muted-xs">
                Supported formats: PDF, DOCX, TXT
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="onboard-panel">
            <p class="onboard-title">Quick start</p>
            <div class="onboard-step"><span>1.</span><span>Upload one or more resume files.</span></div>
            <div class="onboard-step"><span>2.</span><span>Click <strong>Process Files</strong> to validate and index them.</span></div>
            <div class="onboard-step"><span>3.</span><span>Ask comparison questions in the right panel.</span></div>
        </div>
        """, unsafe_allow_html=True)
        
        # file uploader widget
        uploaded_files = st.file_uploader(
            "Choose Files",
            type=['pdf', 'docx', 'txt'],
            accept_multiple_files=True,
            key="resume_uploader",
            label_visibility="collapsed"
        )
        
        # process button appears when files are selected
        if uploaded_files:
            if st.button("Process Files", type="primary"):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # import the processor for validation
                from resume_processor import ResumeProcessor
                validator = ResumeProcessor()
                
                accepted = 0
                rejected = 0
                
                for i, uploaded_file in enumerate(uploaded_files):
                    try:
                        # extract text from the file
                        status_text.text(f"Validating: {uploaded_file.name}")
                        file_bytes = uploaded_file.read()
                        extracted_text = validator.extract_text(file_bytes, uploaded_file.name)
                        
                        # check if its actually a resume
                        is_valid, reason = validator.validate_is_resume(extracted_text)
                        
                        if not is_valid:
                            st.error(f"Rejected: {uploaded_file.name} - {reason}")
                            rejected += 1
                        else:
                            # process valid resumes
                            status_text.text(f"Processing: {uploaded_file.name}")
                            resume_id, metadata = resume_manager.add_resume(file_bytes, uploaded_file.name)
                            st.success(f"Added: {metadata['candidate_name']}")
                            accepted += 1
                            
                    except Exception as e:
                        st.error(f"Failed: {uploaded_file.name} - {str(e)}")
                        rejected += 1
                    
                    progress_bar.progress((i + 1) / len(uploaded_files))
                
                status_text.text(f"Done: {accepted} added, {rejected} rejected")
                
                if accepted > 0:
                    st.rerun()
        
        st.markdown("---")
        
        # show how many resumes are loaded with styled metric
        count = resume_manager.get_resume_count()
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-value">{count}</div>
            <div class="stat-label">Resumes Loaded</div>
        </div>
        """, unsafe_allow_html=True)
        
        # action buttons when resumes exist
        if count > 0:
            st.markdown("<div class='spacer-sm'></div>", unsafe_allow_html=True)
            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("Clear All", use_container_width=True):
                    resume_manager.clear_all_resumes()
                    # also clear conversation context
                    st.session_state.chat_history = []
                    st.session_state.conversation_context = {
                        'last_candidate': None,
                        'last_query_type': None,
                        'mentioned_candidates': set()
                    }
                    st.rerun()
            with col_b:
                if st.button("New Chat", use_container_width=True):
                    resume_manager.clear_conversation()
                    # clear chat history and conversation context
                    st.session_state.chat_history = []
                    st.session_state.conversation_context = {
                        'last_candidate': None,
                        'last_query_type': None,
                        'mentioned_candidates': set()
                    }
                    st.rerun()
    
    with col_chat:
        if resume_manager.get_resume_count() == 0:
            st.markdown(
                '<div class="empty-state-card">Upload some resumes to start asking questions about candidates.</div>',
                unsafe_allow_html=True
            )
        else:
            # About Roxy: show primary candidate summary so recruiters see "me" first
            try:
                with open("resume.txt", "r") as f:
                    roxy_blurb = f.read().strip()[:500]
                if roxy_blurb:
                    roxy_display = roxy_blurb + ("…" if len(roxy_blurb) >= 500 else "")
                    st.markdown(f"""
                    <div class="dark-card about-roxy-card">
                        <h4>About Roxy</h4>
                        <p class="text-muted-xs">{roxy_display}</p>
                    </div>
                    """, unsafe_allow_html=True)
            except FileNotFoundError:
                pass

            # initialize chat history if needed
            if 'chat_history' not in st.session_state:
                st.session_state.chat_history = []
            
            # show conversation context indicator if there's active context
            context = st.session_state.get('conversation_context', {})
            if context.get('mentioned_candidates'):
                mentioned = ', '.join(list(context['mentioned_candidates'])[:3])
                last = context.get('last_candidate', 'None')
                st.markdown(f"""
                <div class="dark-card context-pill">
                    <p class="context-pill-text">
                        <strong class="context-pill-strong-primary">Context:</strong> Tracking {mentioned} | 
                        <strong class="context-pill-strong-accent">Last discussed:</strong> {last}
                    </p>
                </div>
                """, unsafe_allow_html=True)
            
            # display previous messages in a chat container
            if st.session_state.chat_history:
                for msg in st.session_state.chat_history:
                    if msg["role"] == "user":
                        st.markdown(f"""
                        <div class="chat-user">
                            <div class="chat-label">You</div>
                            <p>{msg["content"]}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="chat-assistant">
                            <div class="chat-label">Assistant</div>
                            <p class="pre-wrap">{msg["content"]}</p>
                        </div>
                        """, unsafe_allow_html=True)
                
                st.markdown("---")
            
            # centered composer with explicit Ask button for consistency with personal chat
            st.markdown("<div class='resume-composer-wrap'>", unsafe_allow_html=True)
            with st.form(key=f"resume_form_{st.session_state.resume_input_key}", clear_on_submit=False):
                resume_query = st.text_input(
                    "Your Question",
                    placeholder="Who has AWS experience? Compare their Python skills...",
                    key=f"resume_query_{st.session_state.resume_input_key}",
                    label_visibility="collapsed"
                )
                resume_submitted = st.form_submit_button("Ask", use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown(
                '<p class="resume-composer-note">Try: Compare candidates, find specific skills, or ask follow-up questions.</p>',
                unsafe_allow_html=True
            )
            
            if resume_submitted and resume_query.strip():
                # update conversation context before querying
                update_conversation_context(resume_query, resume_manager)
                
                # build context-aware system prompt
                search_results = resume_manager.search_resumes_with_metadata(resume_query, k=6)
                retrieved_docs = "\n\n".join([r['formatted_text'] for r in search_results])
                
                # track mentioned candidates from search results
                for result in search_results:
                    if result['candidate_name'] != 'Unknown':
                        st.session_state.conversation_context['mentioned_candidates'].add(result['candidate_name'])
                        if not st.session_state.conversation_context['last_candidate']:
                            st.session_state.conversation_context['last_candidate'] = result['candidate_name']
                
                # create enhanced system prompt with context
                system_prompt = build_context_aware_system_prompt(resume_manager, retrieved_docs)
                
                with st.spinner("Analyzing..."):
                    response = resume_manager.query(resume_query, system_prompt=system_prompt)
                
                # update last candidate based on response (simple heuristic)
                for candidate_name in st.session_state.conversation_context['mentioned_candidates']:
                    if candidate_name.lower() in response.lower():
                        st.session_state.conversation_context['last_candidate'] = candidate_name
                
                # add to chat history
                st.session_state.chat_history.append({"role": "user", "content": resume_query})
                st.session_state.chat_history.append({"role": "assistant", "content": response})
                
                # show the response
                st.markdown(f"""
                <div class="response-card">
                    <p class="pre-wrap">{response}</p>
                </div>
                """, unsafe_allow_html=True)
                st.session_state.resume_input_key += 1
                st.rerun()
    
    # resume database section shows when resumes are loaded
    if resume_manager.get_resume_count() > 0:
        st.markdown("---")
        st.markdown("""
        <div class="section-header-wrap">
            <div class="section-header">
                <span class="section-header-icon"></span>
                Resume Database
            </div>
            <p class="section-subtitle">View and manage uploaded candidate profiles</p>
        </div>
        """, unsafe_allow_html=True)
        
        all_metadata = resume_manager.get_all_metadata()
        
        # create tabs for each candidate
        resume_tabs = st.tabs([meta["candidate_name"] for meta in all_metadata])
        
        for i, tab in enumerate(resume_tabs):
            with tab:
                meta = all_metadata[i]
                
                col_info, col_summary = st.columns(2)
                
                with col_info:
                    st.markdown(f"""
                    <div class="dark-card">
                        <h4>Contact Info</h4>
                        <p><strong>Email:</strong> {meta['email']}</p>
                        <p><strong>Phone:</strong> {meta['phone']}</p>
                        <p><strong>Role:</strong> {meta['current_role']}</p>
                        <p><strong>Experience:</strong> {meta['experience_years']} years</p>
                        <p><strong>Education:</strong> {meta['education']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col_summary:
                    st.markdown(f"""
                    <div class="dark-card">
                        <h4>Summary</h4>
                        <p>{meta['summary']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # skills section
                skills_text = ', '.join(meta['key_skills'][:15]) if meta['key_skills'] else 'Not extracted'
                industries_text = ', '.join(meta['industries']) if meta['industries'] else 'Not extracted'
                
                st.markdown(f"""
                <div class="dark-card">
                    <h4>Skills</h4>
                    <p>{skills_text}</p>
                    <h4 class="heading-inline-top">Industries</h4>
                    <p>{industries_text}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # action buttons for this resume
                col_btn1, col_btn2, col_spacer = st.columns([1, 1, 2])
                
                with col_btn1:
                    if st.button("Detailed Summary", key=f"summary_{meta['resume_id']}"):
                        with st.spinner("Generating summary..."):
                            detailed = resume_manager.summarize_resume(meta['resume_id'])
                        st.markdown(f"""
                        <div class="response-card">
                            <p class="pre-wrap">{detailed}</p>
                        </div>
                        """, unsafe_allow_html=True)
                
                with col_btn2:
                    if st.button("Remove", key=f"remove_{meta['resume_id']}"):
                        resume_manager.remove_resume(meta['resume_id'])
                        st.rerun()
