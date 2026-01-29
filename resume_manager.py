"""
Resume manager module for handling multiple resumes with FAISS indexing,
conversation memory, and cross-resume query capabilities.
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

# Load environment variables
load_dotenv()


class ConversationMemory:
    """
    Manages conversation history for context-aware responses.
    Maintains a sliding window of recent messages for memory efficiency.
    """

    def __init__(self, max_messages: int = 20):
        """
        Initialize conversation memory.

        Args:
            max_messages: Maximum number of messages to retain in memory.
        """
        self.messages = []
        self.max_messages = max_messages

    def add_message(self, role: str, content: str):
        """
        Add a message to conversation history.

        Args:
            role: Either 'user' or 'assistant'.
            content: The message content.
        """
        self.messages.append({"role": role, "content": content})
        # Keep only the last max_messages
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages:]

    def get_context(self, last_n: int = 6) -> List[Dict]:
        """
        Get recent conversation context.

        Args:
            last_n: Number of recent messages to retrieve.

        Returns:
            List of message dictionaries.
        """
        return self.messages[-last_n:] if self.messages else []

    def clear(self):
        """Clear all conversation history."""
        self.messages = []

    def get_summary_for_context(self) -> str:
        """
        Get a text summary of recent conversation for context injection.

        Returns:
            A formatted string of recent conversation.
        """
        if not self.messages:
            return ""
        recent = self.get_context(4)
        summary_parts = []
        for msg in recent:
            prefix = "User" if msg["role"] == "user" else "Assistant"
            summary_parts.append(f"{prefix}: {msg['content'][:200]}")
        return "\n".join(summary_parts)


class ResumeManager:
    """
    Manages multiple resumes with FAISS indexing for semantic search,
    supporting cross-resume queries and context-aware conversations.
    """

    MAX_RESUMES = None  # No limit on number of resumes

    def __init__(self):
        """Initialize the ResumeManager with embedding model and storage structures."""
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set")
        self.client = OpenAI(api_key=api_key)
        self.processor = ResumeProcessor()
        
        # Storage for resumes
        self.resumes = {}  # resume_id -> {"text": str, "metadata": dict, "chunks": list}
        self.all_chunks = []  # list of all chunks with resume_id reference
        self.chunk_to_resume = []  # maps chunk index to resume_id
        
        # FAISS index (will be rebuilt when resumes change)
        self.index = None
        self.dimension = 384  # dimension of all-MiniLM-L6-v2 embeddings
        
        # Conversation memory
        self.conversation = ConversationMemory()

    def _generate_resume_id(self, filename: str) -> str:
        """
        Generate a unique ID for a resume based on filename.

        Args:
            filename: The original filename.

        Returns:
            A unique string identifier.
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
        Split text into overlapping chunks for better retrieval.

        Args:
            text: The text to chunk.
            chunk_size: Target size of each chunk in characters.
            overlap: Number of characters to overlap between chunks.

        Returns:
            List of text chunks.
        """
        # First, try splitting by double newlines (sections)
        sections = [s.strip() for s in text.split("\n\n") if s.strip()]
        
        chunks = []
        for section in sections:
            if len(section) <= chunk_size:
                chunks.append(section)
            else:
                # Split long sections into smaller chunks with overlap
                start = 0
                while start < len(section):
                    end = start + chunk_size
                    chunk = section[start:end]
                    chunks.append(chunk)
                    start = end - overlap
        
        return chunks

    def add_resume(self, file_bytes: bytes, filename: str) -> Tuple[str, Dict]:
        """
        Add a new resume to the manager.

        Args:
            file_bytes: Raw bytes of the resume file.
            filename: Original filename.

        Returns:
            Tuple of (resume_id, metadata).

        Raises:
            ValueError: If file processing fails.
        """
        # No limit on number of resumes
        
        # Process the resume
        text, metadata = self.processor.process_resume(file_bytes, filename)
        
        # Generate unique ID
        resume_id = self._generate_resume_id(filename)
        
        # Chunk the text
        chunks = self._chunk_text(text)
        
        # Add metadata context to each chunk for better retrieval
        enriched_chunks = []
        for chunk in chunks:
            enriched = f"[Resume: {metadata['candidate_name']}]\n{chunk}"
            enriched_chunks.append(enriched)
        
        # Store resume data
        self.resumes[resume_id] = {
            "text": text,
            "metadata": metadata,
            "chunks": enriched_chunks
        }
        
        # Rebuild the index
        self._rebuild_index()
        
        return resume_id, metadata

    def remove_resume(self, resume_id: str) -> bool:
        """
        Remove a resume from the manager.

        Args:
            resume_id: The ID of the resume to remove.

        Returns:
            True if successful, False if resume not found.
        """
        if resume_id not in self.resumes:
            return False
        
        del self.resumes[resume_id]
        self._rebuild_index()
        return True

    def _rebuild_index(self):
        """Rebuild the FAISS index from all stored resumes."""
        self.all_chunks = []
        self.chunk_to_resume = []
        
        for resume_id, data in self.resumes.items():
            for chunk in data["chunks"]:
                self.all_chunks.append(chunk)
                self.chunk_to_resume.append(resume_id)
        
        if self.all_chunks:
            embeddings = self.model.encode(self.all_chunks)
            embeddings = np.array(embeddings).astype('float32')
            
            self.index = faiss.IndexFlatL2(self.dimension)
            self.index.add(embeddings)
        else:
            self.index = None

    def get_all_metadata(self) -> List[Dict]:
        """
        Get metadata for all stored resumes.

        Returns:
            List of metadata dictionaries.
        """
        return [
            {**data["metadata"], "resume_id": resume_id}
            for resume_id, data in self.resumes.items()
        ]

    def get_resume_metadata(self, resume_id: str) -> Optional[Dict]:
        """
        Get metadata for a specific resume.

        Args:
            resume_id: The ID of the resume.

        Returns:
            Metadata dictionary or None if not found.
        """
        if resume_id in self.resumes:
            return {**self.resumes[resume_id]["metadata"], "resume_id": resume_id}
        return None

    def search_resumes(self, query: str, k: int = 5) -> List[Tuple[str, str, float]]:
        """
        Search across all resumes for relevant content.

        Args:
            query: The search query.
            k: Number of results to return.

        Returns:
            List of tuples (resume_id, chunk_text, distance).
        """
        if not self.index or self.index.ntotal == 0:
            return []
        
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

    def _build_cross_resume_context(self, query: str) -> str:
        """
        Build context for cross-resume queries by including relevant info from multiple resumes.

        Args:
            query: The user's query.

        Returns:
            A formatted context string.
        """
        # Get all metadata summaries
        all_meta = self.get_all_metadata()
        
        # Create a summary of all candidates
        candidates_overview = "=== Resume Database Overview ===\n"
        for meta in all_meta:
            candidates_overview += f"\n📄 {meta['candidate_name']}\n"
            candidates_overview += f"   Role: {meta['current_role']}\n"
            candidates_overview += f"   Experience: {meta['experience_years']} years\n"
            candidates_overview += f"   Skills: {', '.join(meta['key_skills'][:10])}\n"
            candidates_overview += f"   Industries: {', '.join(meta['industries'][:5])}\n"
        
        # Also do semantic search for relevant chunks
        search_results = self.search_resumes(query, k=6)
        
        relevant_content = "\n=== Relevant Resume Sections ===\n"
        seen_resumes = set()
        for resume_id, chunk, distance in search_results:
            if resume_id not in seen_resumes:
                relevant_content += f"\n{chunk}\n"
                seen_resumes.add(resume_id)
        
        return candidates_overview + relevant_content

    def query(self, user_query: str, use_memory: bool = True) -> str:
        """
        Process a user query and generate a response using RAG.

        Args:
            user_query: The user's question.
            use_memory: Whether to include conversation history for context.

        Returns:
            The generated response.
        """
        if not self.resumes:
            return "No resumes have been uploaded yet. Please upload some resumes first to start querying."
        
        # Detect if this is a cross-resume query
        cross_resume_keywords = [
            "who", "which candidate", "compare", "all", "everyone", "anyone",
            "best", "most", "candidates", "people", "resumes", "group",
            "among", "between", "across"
        ]
        is_cross_resume = any(kw in user_query.lower() for kw in cross_resume_keywords)
        
        # Build context based on query type
        if is_cross_resume:
            context = self._build_cross_resume_context(user_query)
        else:
            # Standard semantic search
            search_results = self.search_resumes(user_query, k=4)
            context = "\n\n".join([chunk for _, chunk, _ in search_results])
        
        # Build messages for the LLM
        messages = [
            {
                "role": "system",
                "content": """You are an expert HR assistant helping to analyze and answer questions about a collection of resumes.
You have access to detailed resume information including skills, experience, education, and work history.
When comparing candidates, be objective and cite specific information from their resumes.
When asked about specific individuals, provide detailed and accurate information.
Always base your answers on the provided context and indicate if information is not available."""
            }
        ]
        
        # Add conversation history for context awareness
        if use_memory:
            conv_context = self.conversation.get_context(4)
            for msg in conv_context:
                messages.append(msg)
        
        # Add the current query with context
        user_message = f"""Context from resumes:
{context}

Current question: {user_query}

Please provide a helpful, accurate answer based on the resume information provided."""
        
        messages.append({"role": "user", "content": user_message})
        
        # Generate response
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=0.3
            )
            answer = response.choices[0].message.content
            
            # Store in conversation memory
            self.conversation.add_message("user", user_query)
            self.conversation.add_message("assistant", answer)
            
            return answer
            
        except Exception as e:
            return f"Error generating response: {str(e)}"

    def summarize_resume(self, resume_id: str) -> str:
        """
        Generate a comprehensive summary of a specific resume.

        Args:
            resume_id: The ID of the resume to summarize.

        Returns:
            A detailed summary string.
        """
        if resume_id not in self.resumes:
            return f"Resume with ID '{resume_id}' not found."
        
        data = self.resumes[resume_id]
        metadata = data["metadata"]
        text = data["text"]
        
        prompt = f"""Please provide a comprehensive professional summary for this candidate:

Name: {metadata['candidate_name']}
Current Role: {metadata['current_role']}
Experience: {metadata['experience_years']} years
Education: {metadata['education']}
Key Skills: {', '.join(metadata['key_skills'])}

Full Resume Text:
{text[:6000]}

Provide a 3-4 paragraph summary covering:
1. Professional background and expertise
2. Key achievements and notable experience
3. Technical skills and competencies
4. Overall assessment and potential fit for technical roles"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert HR professional creating comprehensive candidate summaries."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error generating summary: {str(e)}"

    def find_candidates_with_skill(self, skill: str) -> List[Dict]:
        """
        Find all candidates who have a specific skill.

        Args:
            skill: The skill to search for.

        Returns:
            List of matching candidate metadata.
        """
        skill_lower = skill.lower()
        matching = []
        
        for resume_id, data in self.resumes.items():
            metadata = data["metadata"]
            text_lower = data["text"].lower()
            skills_lower = [s.lower() for s in metadata["key_skills"]]
            
            # Check if skill is in extracted skills or in full text
            if any(skill_lower in s for s in skills_lower) or skill_lower in text_lower:
                matching.append({
                    **metadata,
                    "resume_id": resume_id
                })
        
        return matching

    def clear_conversation(self):
        """Clear conversation history to start fresh."""
        self.conversation.clear()

    def clear_all_resumes(self):
        """Remove all resumes and reset the manager."""
        self.resumes = {}
        self.all_chunks = []
        self.chunk_to_resume = []
        self.index = None
        self.conversation.clear()

    def get_resume_count(self) -> int:
        """
        Get the current number of stored resumes.

        Returns:
            Number of resumes.
        """
        return len(self.resumes)
