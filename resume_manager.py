"""
resume manager module for handling multiple resumes with faiss indexing
supports conversation memory and cross-resume query capabilities
this is the main brain of the resume analyzer feature
"""

import faiss
import numpy as np
import json
import os
from typing import Dict, List, Optional, Tuple
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from dotenv import load_dotenv
from resume_processor import ResumeProcessor

# load environment variables
load_dotenv()


class ConversationMemory:
    """
    manages conversation history for context-aware responses
    keeps a sliding window of recent messages so we dont exceed token limits
    """

    def __init__(self, max_messages: int = 20):
        """
        sets up the conversation memory
        max_messages controls how many messages we keep around
        """
        self.messages = []
        self.max_messages = max_messages

    def add_message(self, role: str, content: str):
        """
        adds a new message to the conversation history
        role should be either user or assistant
        automatically trims old messages if we exceed the max
        """
        self.messages.append({"role": role, "content": content})
        # keep only the most recent messages
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages:]

    def get_context(self, last_n: int = 6) -> List[Dict]:
        """
        grabs the most recent messages for context
        returns up to last_n messages
        """
        return self.messages[-last_n:] if self.messages else []

    def clear(self):
        """wipes all conversation history"""
        self.messages = []

    def get_summary_for_context(self) -> str:
        """
        creates a text summary of recent conversation
        useful for injecting context into prompts
        """
        if not self.messages:
            return ""
        recent = self.get_context(4)
        summary_parts = []
        for msg in recent:
            prefix = "user" if msg["role"] == "user" else "assistant"
            summary_parts.append(f"{prefix}: {msg['content'][:200]}")
        return "\n".join(summary_parts)


class ResumeManager:
    """
    manages multiple resumes with faiss indexing for semantic search
    supports cross-resume queries and remembers conversation context
    this is the core of the resume analysis feature
    """

    MAX_RESUMES = None  # no limit on number of resumes

    def __init__(self):
        """
        initializes the manager with embedding model and storage
        sets up faiss index and conversation memory
        """
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set")
        self.client = OpenAI(api_key=api_key)
        self.processor = ResumeProcessor()
        
        # storage for all the resumes
        self.resumes = {}  # resume_id -> {"text": str, "metadata": dict, "chunks": list}
        self.all_chunks = []  # flat list of all chunks for faiss
        self.chunk_to_resume = []  # maps chunk index back to resume_id
        
        # faiss index gets rebuilt when resumes change
        self.index = None
        self.dimension = 384  # dimension of all-MiniLM-L6-v2 embeddings
        
        # conversation memory for context-aware responses
        self.conversation = ConversationMemory()

    def _generate_resume_id(self, filename: str) -> str:
        """
        creates a unique id for a resume based on the filename
        handles duplicates by adding a counter suffix
        """
        base_id = filename.replace(' ', '_').replace('.', '_')
        counter = 1
        resume_id = base_id
        while resume_id in self.resumes:
            resume_id = f"{base_id}_{counter}"
            counter += 1
        return resume_id

    def _chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """
        splits text into overlapping chunks for better retrieval
        chunks are sized to fit well in embeddings while maintaining context
        overlap helps ensure we dont miss info at chunk boundaries
        """
        # first try splitting by sections (double newlines)
        sections = [s.strip() for s in text.split("\n\n") if s.strip()]
        
        chunks = []
        for section in sections:
            if len(section) <= chunk_size:
                chunks.append(section)
            else:
                # split long sections with overlap
                start = 0
                while start < len(section):
                    end = start + chunk_size
                    chunk = section[start:end]
                    chunks.append(chunk)
                    start = end - overlap
        
        return chunks

    def add_resume(self, file_bytes: bytes, filename: str) -> Tuple[str, Dict]:
        """
        adds a new resume to the manager
        processes the file, extracts metadata, and rebuilds the search index
        
        takes the raw file bytes and original filename
        returns tuple of (resume_id, metadata)
        raises an error if processing fails
        """
        # no limit on number of resumes
        
        # process the resume to get text and metadata
        text, metadata = self.processor.process_resume(file_bytes, filename)
        
        # create a unique id for this resume
        resume_id = self._generate_resume_id(filename)
        
        # chunk the text for better retrieval
        chunks = self._chunk_text(text)
        
        # add the candidate name to each chunk for context
        enriched_chunks = []
        for chunk in chunks:
            enriched = f"[resume: {metadata['candidate_name']}]\n{chunk}"
            enriched_chunks.append(enriched)
        
        # store everything
        self.resumes[resume_id] = {
            "text": text,
            "metadata": metadata,
            "chunks": enriched_chunks
        }
        
        # rebuild the faiss index with the new resume
        self._rebuild_index()
        
        return resume_id, metadata

    def remove_resume(self, resume_id: str) -> bool:
        """
        removes a resume from the manager
        returns true if successful, false if resume wasnt found
        """
        if resume_id not in self.resumes:
            return False
        
        del self.resumes[resume_id]
        self._rebuild_index()
        return True

    def _rebuild_index(self):
        """
        rebuilds the faiss index from all stored resumes
        called automatically when resumes are added or removed
        """
        self.all_chunks = []
        self.chunk_to_resume = []
        
        # collect all chunks from all resumes
        for resume_id, data in self.resumes.items():
            for chunk in data["chunks"]:
                self.all_chunks.append(chunk)
                self.chunk_to_resume.append(resume_id)
        
        # create embeddings and build the index
        if self.all_chunks:
            embeddings = self.model.encode(self.all_chunks)
            embeddings = np.array(embeddings).astype('float32')
            
            self.index = faiss.IndexFlatL2(self.dimension)
            self.index.add(embeddings)
        else:
            self.index = None

    def get_all_metadata(self) -> List[Dict]:
        """
        returns metadata for all stored resumes
        useful for displaying the resume database
        """
        return [
            {**data["metadata"], "resume_id": resume_id}
            for resume_id, data in self.resumes.items()
        ]

    def get_resume_metadata(self, resume_id: str) -> Optional[Dict]:
        """
        gets metadata for a specific resume by id
        returns none if the resume isnt found
        """
        if resume_id in self.resumes:
            return {**self.resumes[resume_id]["metadata"], "resume_id": resume_id}
        return None

    def search_resumes(self, query: str, k: int = 5) -> List[Tuple[str, str, float]]:
        """
        searches across all resumes for content relevant to the query
        uses faiss for fast similarity search
        
        returns list of (resume_id, chunk_text, distance) tuples
        """
        if not self.index or self.index.ntotal == 0:
            return []
        
        # embed the query and search
        query_embedding = self.model.encode([query]).astype('float32')
        k = min(k, self.index.ntotal)
        distances, indices = self.index.search(query_embedding, k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.all_chunks):
                resume_id = self.chunk_to_resume[idx]
                chunk = self.all_chunks[idx]
                distance = float(distances[0][i])
                results.append((resume_id, chunk, distance))
        
        return results

    def search_resumes_with_metadata(self, query: str, k: int = 6) -> List[Dict]:
        """
        enhanced search that returns chunks with full candidate metadata
        includes formatted text with candidate headers for better context
        
        returns list of dicts with text, candidate info, and formatted version
        """
        if not self.index or self.index.ntotal == 0:
            return []
        
        query_embedding = self.model.encode([query]).astype('float32')
        k = min(k, self.index.ntotal)
        distances, indices = self.index.search(query_embedding, k)
        
        results = []
        seen_candidates = set()
        
        for i, idx in enumerate(indices[0]):
            if idx < len(self.all_chunks):
                resume_id = self.chunk_to_resume[idx]
                chunk = self.all_chunks[idx]
                
                # get candidate metadata
                if resume_id in self.resumes:
                    meta = self.resumes[resume_id]['metadata']
                    candidate_name = meta['candidate_name']
                    
                    result = {
                        'text': chunk,
                        'candidate_name': candidate_name,
                        'current_role': meta.get('current_role', 'Unknown'),
                        'experience_years': meta.get('experience_years', 0),
                        'key_skills': meta.get('key_skills', [])[:5],
                        'distance': float(distances[0][i])
                    }
                    
                    # add candidate header if first chunk from this candidate
                    if candidate_name not in seen_candidates:
                        result['formatted_text'] = f"\n=== {candidate_name} ({meta.get('current_role', 'Unknown Role')}) ===\n{chunk}"
                        seen_candidates.add(candidate_name)
                    else:
                        result['formatted_text'] = f"[{candidate_name}]: {chunk}"
                    
                    results.append(result)
                else:
                    results.append({
                        'text': chunk,
                        'formatted_text': chunk,
                        'candidate_name': 'Unknown',
                        'current_role': 'Unknown',
                        'experience_years': 0,
                        'key_skills': [],
                        'distance': float(distances[0][i])
                    })
        
        return results

    def _build_cross_resume_context(self, query: str) -> str:
        """
        builds context for queries that span multiple resumes
        includes an overview of all candidates plus relevant search results
        """
        # get summaries of all candidates
        all_meta = self.get_all_metadata()
        
        candidates_overview = "=== resume database overview ===\n"
        for meta in all_meta:
            candidates_overview += f"\n{meta['candidate_name']}\n"
            candidates_overview += f"   role: {meta['current_role']}\n"
            candidates_overview += f"   experience: {meta['experience_years']} years\n"
            candidates_overview += f"   skills: {', '.join(meta['key_skills'][:10])}\n"
            candidates_overview += f"   industries: {', '.join(meta['industries'][:5])}\n"
        
        # also do semantic search for specific relevant chunks
        search_results = self.search_resumes(query, k=6)
        
        relevant_content = "\n=== relevant resume sections ===\n"
        seen_resumes = set()
        for resume_id, chunk, distance in search_results:
            if resume_id not in seen_resumes:
                relevant_content += f"\n{chunk}\n"
                seen_resumes.add(resume_id)
        
        return candidates_overview + relevant_content

    def query(self, user_query: str, use_memory: bool = True, system_prompt: str = None) -> str:
        """
        processes a user query and generates a response using rag
        automatically detects if the query is about multiple resumes
        
        takes the users question and optional custom system prompt
        returns the generated response
        """
        if not self.resumes:
            return "no resumes have been uploaded yet - please upload some resumes first to start querying"
        
        # detect if this is asking about multiple candidates
        cross_resume_keywords = [
            "who", "which candidate", "compare", "all", "everyone", "anyone",
            "best", "most", "candidates", "people", "resumes", "group",
            "among", "between", "across"
        ]
        is_cross_resume = any(kw in user_query.lower() for kw in cross_resume_keywords)
        
        # detect follow-up queries that need context
        follow_up_keywords = ["tell me more", "what about", "their", "them", "he ", "she ", "they ", 
                              "his ", "her ", "elaborate", "continue", "and what", "also"]
        is_follow_up = any(kw in user_query.lower() for kw in follow_up_keywords)
        
        # build the appropriate context with enhanced metadata
        if is_cross_resume:
            context = self._build_cross_resume_context(user_query)
        else:
            # use enhanced search with metadata for better context
            search_results = self.search_resumes_with_metadata(user_query, k=6)
            context = "\n\n".join([r['formatted_text'] for r in search_results])
        
        # use custom system prompt if provided, otherwise use default
        if system_prompt:
            system_content = system_prompt
        else:
            # build candidates overview for context
            candidates_overview = self._get_candidates_overview()
            system_content = f"""You are an expert HR assistant helping analyze a collection of resumes.

=== CANDIDATES IN DATABASE ===
{candidates_overview}

=== INSTRUCTIONS ===
- When discussing a candidate, ALWAYS include their full name
- When comparing candidates, clearly separate information about each person
- If using pronouns (he/she/they), make sure it's clear who you're referring to
- When asked follow-up questions, refer back to previously discussed candidates
- Be objective and cite specific information from their resumes
- If information is not available, clearly state that

=== RESUME CONTEXT ===
{context}"""
        
        # set up the messages for the llm
        messages = [{"role": "system", "content": system_content}]
        
        # add conversation history for context awareness (increased from 4 to 8 for better follow-up)
        if use_memory:
            conv_context = self.conversation.get_context(8)
            for msg in conv_context:
                messages.append(msg)
        
        # add the current query
        if is_follow_up and conv_context:
            # for follow-ups, remind the LLM about the context
            user_message = f"""Follow-up question (refer to previously discussed candidates): {user_query}

Please continue the conversation and answer based on the context above."""
        else:
            user_message = f"""Question: {user_query}

Please provide a helpful, accurate answer. Always specify which candidate you're discussing."""
        
        messages.append({"role": "user", "content": user_message})
        
        # generate the response
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=0.3
            )
            answer = response.choices[0].message.content
            
            # save to conversation memory for future context
            self.conversation.add_message("user", user_query)
            self.conversation.add_message("assistant", answer)
            
            return answer
            
        except Exception as e:
            return f"error generating response: {str(e)}"

    def _get_candidates_overview(self) -> str:
        """
        creates a quick reference of all candidates for the system prompt
        """
        if not self.resumes:
            return "No candidates in database."
        
        lines = []
        for resume_id, data in self.resumes.items():
            meta = data['metadata']
            skills = ', '.join(meta.get('key_skills', [])[:4]) or 'Not specified'
            lines.append(
                f"• {meta['candidate_name']}: {meta.get('current_role', 'Unknown')} | "
                f"{meta.get('experience_years', 'Unknown')} years | Skills: {skills}"
            )
        return '\n'.join(lines)

    def summarize_resume(self, resume_id: str) -> str:
        """
        generates a detailed professional summary for a specific resume
        
        takes the resume id
        returns a comprehensive summary string
        """
        if resume_id not in self.resumes:
            return f"resume with id '{resume_id}' not found"
        
        data = self.resumes[resume_id]
        metadata = data["metadata"]
        text = data["text"]
        
        prompt = f"""please provide a comprehensive professional summary for this candidate:

name: {metadata['candidate_name']}
current role: {metadata['current_role']}
experience: {metadata['experience_years']} years
education: {metadata['education']}
key skills: {', '.join(metadata['key_skills'])}

full resume text:
{text[:6000]}

provide a 3-4 paragraph summary covering:
1. professional background and expertise
2. key achievements and notable experience
3. technical skills and competencies
4. overall assessment and potential fit for technical roles"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "you are an expert hr professional creating comprehensive candidate summaries"
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"error generating summary: {str(e)}"

    def find_candidates_with_skill(self, skill: str) -> List[Dict]:
        """
        finds all candidates who have a specific skill
        searches both the extracted skills list and full resume text
        
        takes the skill to search for
        returns list of matching candidate metadata
        """
        skill_lower = skill.lower()
        matching = []
        
        for resume_id, data in self.resumes.items():
            metadata = data["metadata"]
            text_lower = data["text"].lower()
            skills_lower = [s.lower() for s in metadata["key_skills"]]
            
            # check extracted skills and full text
            if any(skill_lower in s for s in skills_lower) or skill_lower in text_lower:
                matching.append({
                    **metadata,
                    "resume_id": resume_id
                })
        
        return matching

    def clear_conversation(self):
        """clears conversation history to start fresh"""
        self.conversation.clear()

    def clear_all_resumes(self):
        """removes all resumes and resets everything"""
        self.resumes = {}
        self.all_chunks = []
        self.chunk_to_resume = []
        self.index = None
        self.conversation.clear()

    def get_resume_count(self) -> int:
        """returns how many resumes are currently stored"""
        return len(self.resumes)
