"""
personal rag chatbot with resume analysis capabilities
supports uploading resumes, cross-resume queries, and conversation memory
built with streamlit, faiss, and openai
"""

import streamlit as st
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from dotenv import load_dotenv
import os
from resume_manager import ResumeManager

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


# custom css for dark theme with cyan/purple gradient accents
# inspired by modern dashboard design with glass morphism
st.markdown("""
<style>
    /* import clean font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
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
    
    /* SIDEBAR TOGGLE BUTTONS - Ensure expand button is visible */
    /* Collapse button inside sidebar */
    [data-testid="stSidebar"] button[kind="header"] {
        background: var(--bg-card) !important;
        border: 1px solid var(--border-subtle) !important;
        border-radius: 8px !important;
        color: var(--text-secondary) !important;
        transition: all 0.3s ease !important;
    }
    
    [data-testid="stSidebar"] button[kind="header"]:hover {
        background: var(--bg-card-hover) !important;
        border-color: var(--accent-cyan) !important;
        color: var(--accent-cyan) !important;
    }
    
    /* Expand button when sidebar is collapsed - this is the important one */
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
        background: linear-gradient(135deg, #1a2332 0%, #111827 100%) !important;
        border: 1px solid rgba(34, 211, 238, 0.3) !important;
        border-radius: 12px !important;
        color: var(--accent-cyan) !important;
        width: 44px !important;
        height: 44px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        cursor: pointer !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4), 0 0 20px rgba(34, 211, 238, 0.15) !important;
    }
    
    [data-testid="collapsedControl"] > button:hover,
    [data-testid="baseButton-header"]:hover {
        background: linear-gradient(135deg, #22d3ee 0%, #a855f7 100%) !important;
        border-color: var(--accent-cyan) !important;
        color: white !important;
        transform: scale(1.05) !important;
        box-shadow: 0 6px 25px rgba(0, 0, 0, 0.5), 0 0 30px rgba(34, 211, 238, 0.3) !important;
    }
    
    [data-testid="collapsedControl"] svg,
    [data-testid="baseButton-header"] svg {
        width: 20px !important;
        height: 20px !important;
        stroke: currentColor !important;
    }
    
    /* Make sure the button is always clickable */
    [data-testid="collapsedControl"] * {
        pointer-events: auto !important;
    }
</style>
""", unsafe_allow_html=True)

# sidebar navigation with dark theme
with st.sidebar:
    # brand header with gradient
    st.markdown("""
    <div class="sidebar-header">
        <p class="sidebar-brand">Roxy's AI Hub</p>
        <p class="sidebar-tagline">Powered by AI</p>
    </div>
    """, unsafe_allow_html=True)
    
    # navigation section - plain text title, not a button
    st.markdown('<p class="nav-title">Select Mode</p>', unsafe_allow_html=True)
    
    # navigation using radio buttons for accessibility
    page = st.radio(
        "nav",
        ["Personal Chat", "Resume Analyzer"],
        label_visibility="hidden"
    )
    
    st.markdown("---")
    
    # features section
    st.markdown('<p class="nav-title">Features</p>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="feature-card">
        <p class="feature-title" style="color: #22d3ee;">Personal Chat</p>
        <p class="feature-desc">Ask questions about Roxy's background, skills, and experience</p>
    </div>
    <div class="feature-card feature-card-purple">
        <p class="feature-title" style="color: #a855f7;">Resume Analyzer</p>
        <p class="feature-desc">Upload and analyze resumes with AI-powered insights</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # footer info
    st.markdown("""
    <div style="text-align: center; padding: 1rem 0;">
        <p style="color: #64748b; font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.1em; margin: 0;">
            Powered by
        </p>
        <p style="background: linear-gradient(135deg, #22d3ee, #a855f7); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-size: 0.85rem; font-weight: 600; margin: 0.5rem 0 0 0;">
            FAISS • OpenAI • Streamlit
        </p>
    </div>
    """, unsafe_allow_html=True)

# main content area - dynamic title based on page with large styled titles
# using <div> instead of <h1> to avoid Streamlit's default h1 styling override
if page == "Personal Chat":
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

# personal chat page
if page == "Personal Chat":
    # load the resources for semantic search
    model, index, chunks = load_resources()
    
    # create a nice input section with proper card container
    st.markdown("""
    <div class="content-card">
        <div class="content-card-header">
            <div class="content-card-icon">💬</div>
            <p class="content-card-title">Ask a Question</p>
        </div>
        <p style="color: #94a3b8; margin: 0 0 1rem 0; font-size: 0.95rem;">
            Get answers about Roxy's background, skills, projects, and more
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # input field for the query
    query = st.text_input(
        "Your Question",
        placeholder="What are Roxy's skills? What's her background?",
        key="personal_query",
        label_visibility="collapsed"
    )
    
    if query:
        # convert the query to an embedding vector
        query_embedding = model.encode([query]).astype('float32')
        
        # search faiss for the most relevant chunks
        k = 2
        distances, indices = index.search(query_embedding, k)
        relevant_chunks = [chunks[i] for i in indices[0]]
        context = "\n".join(relevant_chunks)
        
        # send to openai with the context
        with st.spinner("Thinking..."):
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "you are a helpful assistant answering questions about roxy story based on her resume and personal information. only use the provided context to answer. respond as if you are roxy (in first-person). be conversational and friendly"
                    },
                    {
                        "role": "user",
                        "content": f"context:\n{context}\n\nquestion: {query}"
                    }
                ]
            )
        
        answer = response.choices[0].message.content
        
        # display the response in a nice card
        st.markdown(f"""
        <div class="response-card">
            <p style="white-space: pre-wrap;">{answer}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # expandable section to see what context was retrieved
        with st.expander("View Retrieved Context"):
            st.markdown(f"""
            <div class="dark-card" style="margin: 0;">
                <p style="font-size: 0.9rem; color: #94a3b8;">{context}</p>
            </div>
            """, unsafe_allow_html=True)

# resume analyzer page
elif page == "Resume Analyzer":
    resume_manager = get_resume_manager()
    
    # two column layout for upload and chat
    col_upload, col_chat = st.columns([1, 2])
    
    with col_upload:
        st.markdown("""
        <div class="content-card" style="margin-bottom: 1.5rem;">
            <div class="content-card-header">
                <div class="content-card-icon">📄</div>
                <p class="content-card-title">Upload Resumes</p>
            </div>
            <p style="color: #94a3b8; margin: 0; font-size: 0.9rem;">
                Supported formats: PDF, DOCX, TXT
            </p>
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
            st.markdown("<div style='height: 1rem'></div>", unsafe_allow_html=True)
            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("Clear All", use_container_width=True):
                    resume_manager.clear_all_resumes()
                    st.rerun()
            with col_b:
                if st.button("New Chat", use_container_width=True):
                    resume_manager.clear_conversation()
                    if 'chat_history' in st.session_state:
                        st.session_state.chat_history = []
                    st.rerun()
    
    with col_chat:
        st.markdown("""
        <div class="content-card" style="margin-bottom: 1.5rem;">
            <div class="content-card-header">
                <div class="content-card-icon" style="background: linear-gradient(135deg, #a855f7 0%, #ec4899 100%); box-shadow: 0 0 20px rgba(168, 85, 247, 0.3);">🔍</div>
                <p class="content-card-title">Ask Questions</p>
            </div>
            <p style="color: #94a3b8; margin: 0; font-size: 0.9rem;">
                Query candidates, compare skills, get AI-powered insights
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        if resume_manager.get_resume_count() == 0:
            st.info("Upload some resumes to start asking questions about candidates")
        else:
            # initialize chat history if needed
            if 'chat_history' not in st.session_state:
                st.session_state.chat_history = []
            
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
                            <p style="white-space: pre-wrap;">{msg["content"]}</p>
                        </div>
                        """, unsafe_allow_html=True)
                
                st.markdown("---")
            
            # query input
            resume_query = st.text_input(
                "Your Question",
                placeholder="Who has AWS experience? Compare their Python skills...",
                key="resume_query",
                label_visibility="collapsed"
            )
            
            st.caption("Try: Compare candidates, find specific skills, request summaries")
            
            if resume_query:
                with st.spinner("Analyzing..."):
                    response = resume_manager.query(resume_query)
                
                # add to chat history
                st.session_state.chat_history.append({"role": "user", "content": resume_query})
                st.session_state.chat_history.append({"role": "assistant", "content": response})
                
                # show the response
                st.markdown(f"""
                <div class="response-card">
                    <p style="white-space: pre-wrap;">{response}</p>
                </div>
                """, unsafe_allow_html=True)
    
    # resume database section shows when resumes are loaded
    if resume_manager.get_resume_count() > 0:
        st.markdown("---")
        st.markdown("""
        <div style="margin-bottom: 1.5rem;">
            <div class="section-header">
                <span class="section-header-icon"></span>
                Resume Database
            </div>
            <p style="color: #64748b; font-size: 0.9rem; margin-top: 0.5rem;">View and manage uploaded candidate profiles</p>
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
                    <h4 style="margin-top: 1rem;">Industries</h4>
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
                            <p style="white-space: pre-wrap;">{detailed}</p>
                        </div>
                        """, unsafe_allow_html=True)
                
                with col_btn2:
                    if st.button("Remove", key=f"remove_{meta['resume_id']}"):
                        resume_manager.remove_resume(meta['resume_id'])
                        st.rerun()
