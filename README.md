# personal rag chatbot

a rag-based chatbot with dual functionality: personal information queries and multi-resume analysis with cross-candidate queries

## deployment

**live demo:** [coming soon]

## tech stack

| technology | purpose |
|------------|---------|
| python 3.10+ | core language |
| streamlit | web interface and ui framework |
| faiss | vector similarity search for semantic retrieval |
| sentence-transformers | text embeddings using all-MiniLM-L6-v2 |
| openai gpt-4o-mini | language model for generating responses |
| pdfplumber / pypdf2 | pdf text extraction |
| python-docx | word document parsing |

## features

### personal chat mode
- Roxy persona: answers as you in first person (e.g. in front of a recruiter)
- semantic search over your resume and personal data (callable as a tool; model can run it multiple times per turn)
- tools: weather (Open-Meteo), web search (Tavily or DuckDuckGo), GitHub search (all of GitHub; use "user:USERNAME" to find someone's repos)
- conversational ai powered by openai with function/tool calling

### resume analyzer mode
- upload unlimited resumes (pdf, docx, or txt)
- automatic validation rejects non-resume documents
- automatic metadata extraction including:
  - candidate name and contact info
  - professional summary
  - key skills and technologies
  - years of experience
  - education and industries
- cross-resume queries like "who has aws experience"
- conversation memory for context-aware follow-ups
- detailed summary generation for any candidate

## setup

### 1 install dependencies
```bash
pip install -r requirements.txt
```

### 2 configure environment
create a `.env` file with your openai api key:
```
OPENAI_API_KEY=your_api_key_here
```

**Optional (Personal Chat tools):** Tools degrade gracefully if not set.
| variable | purpose |
|----------|---------|
| `TAVILY_API_KEY` | Prefer Tavily for web search (otherwise DuckDuckGo is used, no key). Install `tavily-python` if using. |
| `GITHUB_TOKEN` | Higher GitHub search rate limits (optional). |
| Open-Meteo | Weather tool uses [Open-Meteo](https://open-meteo.com) — no API key required. |

### 3 initialize personal data index
```bash
python embeddata.py
```

### 4 run the application
```bash
streamlit run app.py
```

## how to deploy

### deploy to streamlit cloud (recommended)

1. push your code to github
2. go to [share.streamlit.io](https://share.streamlit.io)
3. connect your github account
4. select your repository
5. set the main file path to `app.py`
6. add your `OPENAI_API_KEY` in the secrets section
7. click deploy

### deploy to railway

1. create an account at [railway.app](https://railway.app)
2. connect your github repository
3. add environment variable `OPENAI_API_KEY`
4. railway will auto-detect the streamlit app
5. your app will be live at a railway url

### deploy to render

1. create a `render.yaml` file:
```yaml
services:
  - type: web
    name: rag-chatbot
    env: python
    buildCommand: pip install -r requirements.txt
    startCommand: streamlit run app.py --server.port $PORT --server.address 0.0.0.0
    envVars:
      - key: OPENAI_API_KEY
        sync: false
```
2. connect your github repo on [render.com](https://render.com)
3. add your api key in the environment variables

### deploy with docker

1. create a `Dockerfile`:
```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```
2. build and run:
```bash
docker build -t rag-chatbot .
docker run -p 8501:8501 -e OPENAI_API_KEY=your_key rag-chatbot
```

## project structure

```
├── app.py                 # main streamlit application
├── tools.py               # Personal Chat tools (semantic search, weather, web, GitHub)
├── embeddata.py           # script to create faiss index
├── resume_processor.py    # pdf/docx text extraction and validation
├── resume_manager.py      # multi-resume management with faiss
├── requirements.txt       # python dependencies
├── faiss_index.bin        # pre-built faiss index for personal data
├── resume.txt             # personal resume data
├── personal.txt           # personal information data
├── .streamlit/            # streamlit configuration
│   └── config.toml
└── README.md
```

## testing the features

Use these example questions in **Personal Chat** to verify each feature. After a reply, check for the "Used: …" pill under Roxy's message to confirm which tool ran.

| Feature | Example questions to ask |
|--------|---------------------------|
| **Current date** | "What's today's date?" or "What year is it?" |
| **Semantic search (your resume/personal)** | "What are my skills?" / "What's my background?" / "Where did I go to school?" (Roxy answers from your data; may show "Used: my info".) |
| **Weather** | "What's the weather in San Francisco today?" / "Is it going to rain in NYC?" (Uses Open-Meteo; no key needed.) |
| **Web search** | "What's the best movie of 2025 according to Google?" / "Latest news about [topic]." (Uses Tavily if `TAVILY_API_KEY` set, else DuckDuckGo.) |
| **GitHub search** | "What repos do I have on GitHub?" (Use your real username in `resume.txt` or ask with your username.) / "Find repos by user octocat." (Searches all of GitHub; use "user:USERNAME" when the bot calls the tool.) |
| **Roxy persona** | "Tell me about yourself." / "Why should we hire you?" (Answers in first person as Roxy.) |

**Resume Analyzer:** Upload 1–2 PDF/DOCX resumes, click **Process Files**, then ask e.g. "Who has Python experience?", "Compare these candidates," or "Summarize [candidate name]." The "About Roxy" blurb at the top uses `resume.txt` when present.

## accessibility

the ui is designed with accessibility in mind:
- wcag compliant color contrast ratios
- keyboard navigation support
- screen reader friendly labels
- respects reduced motion preferences
- clear focus indicators

## license

mit
