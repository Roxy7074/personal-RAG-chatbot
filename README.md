Personal Retreival Augmented Generation Chatbot!

- Use requirements.txt to ensure downloads
- embeddata.py
  * Loads text from resume.txt and personal.txt 
  * Chunks the resume by paragraphs, keeps personal data as one chunk 
  * Uses the all-MiniLM-L6-v2 model to convert each chunk into a 384-dimensional embedding vector 
  * Stores all vectors in a FAISS index (faiss_index.bin) for fast similarity search
- app.py
  * Loads the same embedding model and FAISS index 
  * Takes a user question and embeds it into the same vector space 
  * Searches FAISS for the 2 most similar chunks (k=2)
  * Sends those chunks as context to the LLM  

 
