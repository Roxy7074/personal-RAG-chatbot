# Personal Retrieval Augmented Generation Chatbot

A powerful RAG-based chatbot with dual functionality: personal information queries and multi-resume analysis with cross-candidate queries.

## Features

### 💬 Personal Chat Mode
- Query personal information and resume data
- Uses semantic search with FAISS for accurate retrieval
- Conversational AI powered by OpenAI GPT-4o-mini

### 📋 Resume Analyzer Mode
- **Multi-Resume Upload**: Upload up to 20 resumes (PDF, DOCX, or TXT)
- **Automatic Metadata Extraction**: 
  - Candidate name
  - Contact information (email, phone)
  - Professional summary
  - Key skills and technologies
  - Years of experience
  - Current/recent role
  - Education
  - Industry experience
- **Cross-Resume Queries**: Ask questions across all uploaded resumes
  - "Who has experience with AWS?"
  - "Which candidate would be best for a data science role?"
  - "Compare the Python skills of all candidates"
- **Context-Aware Conversations**: Memory-enabled chat that remembers previous questions
- **Individual Summaries**: Generate detailed summaries for any candidate
- **Semantic Search**: FAISS-powered vector search for accurate information retrieval

## Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure Environment
Create a `.env` file with your OpenAI API key:
```
OPENAI_API_KEY=your_api_key_here
```

### 3. Initialize Personal Data Index
```bash
python embeddata.py
```

### 4. Run the Application
```bash
streamlit run app.py
```

## Project Structure

```
├── app.py                 # Main Streamlit application with dual-mode UI
├── embeddata.py           # Script to create FAISS index for personal data
├── resume_processor.py    # PDF/DOCX text extraction and metadata generation
├── resume_manager.py      # Multi-resume management with FAISS indexing
├── requirements.txt       # Python dependencies
├── faiss_index.bin        # Pre-built FAISS index for personal data
├── resume.txt             # Personal resume data
├── personal.txt           # Personal information data
└── README.md              # This file
```

## Module Documentation

### embeddata.py
- Loads text from `resume.txt` and `personal.txt`
- Chunks the resume by paragraphs, keeps personal data as one chunk
- Uses the `all-MiniLM-L6-v2` model to convert each chunk into a 384-dimensional embedding vector
- Stores all vectors in a FAISS index (`faiss_index.bin`) for fast similarity search

### app.py
- Dual-tab interface: Personal Chat and Resume Analyzer
- Loads the embedding model and FAISS index
- Takes user questions and embeds them into the same vector space
- Searches FAISS for relevant chunks
- Sends context to OpenAI LLM for response generation

### resume_processor.py
- `ResumeProcessor` class handles file parsing:
  - PDF extraction using pdfplumber (with PyPDF2 fallback)
  - DOCX extraction using python-docx
  - TXT file handling with encoding detection
- Automatic metadata generation using LLM analysis

### resume_manager.py
- `ResumeManager` class provides:
  - Multi-resume storage and indexing (up to 20 resumes)
  - FAISS-based semantic search across all resumes
  - Cross-resume query detection and handling
  - Conversation memory for context-aware responses
- `ConversationMemory` class manages chat history with sliding window

## Usage Examples

### Personal Chat
```
Q: "What programming languages does Roxy know?"
Q: "Tell me about Roxy's work experience"
Q: "What are Roxy's hobbies?"
```

### Resume Analyzer
```
Q: "Who in this group has AWS experience?"
Q: "Summarize John Smith's background"
Q: "Which candidates have 5+ years of experience?"
Q: "Compare the technical skills of all candidates"
Q: "Who would be best for a machine learning role?"
```

## Technical Details

- **Embedding Model**: `all-MiniLM-L6-v2` (384 dimensions)
- **Vector Store**: FAISS IndexFlatL2
- **LLM**: OpenAI GPT-4o-mini
- **UI Framework**: Streamlit
- **Supported File Types**: PDF, DOCX, TXT

## Dependencies

- faiss-cpu
- sentence-transformers
- streamlit
- openai
- python-dotenv
- PyPDF2
- pdfplumber
- python-docx
