"""
Resume processor module for extracting text from PDF and DOCX files,
and generating metadata for each resume.
"""
import re
import io
from typing import Dict, Optional, Tuple
from openai import OpenAI
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()


class ResumeProcessor:
    """
    Handles the extraction of text from resume files (PDF, DOCX, TXT)
    and generates metadata including candidate name, summary, and skills.
    """

    def __init__(self):
        """Initialize the ResumeProcessor with OpenAI client for metadata generation."""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set")
        self.client = OpenAI(api_key=api_key)

    def extract_text_from_pdf(self, file_bytes: bytes) -> str:
        """
        Extract text content from a PDF file.

        Args:
            file_bytes: The raw bytes of the PDF file.

        Returns:
            The extracted text content as a string.
        """
        try:
            import pdfplumber
            text_parts = []
            with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text_parts.append(page_text)
            return "\n\n".join(text_parts)
        except Exception as e:
            # fallback to PyPDF2 if pdfplumber fails
            try:
                from PyPDF2 import PdfReader
                reader = PdfReader(io.BytesIO(file_bytes))
                text_parts = []
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text_parts.append(page_text)
                return "\n\n".join(text_parts)
            except Exception as e2:
                raise ValueError(f"Failed to extract text from PDF: {e}, {e2}")

    def extract_text_from_docx(self, file_bytes: bytes) -> str:
        """
        Extract text content from a DOCX file.

        Args:
            file_bytes: The raw bytes of the DOCX file.

        Returns:
            The extracted text content as a string.
        """
        try:
            from docx import Document
            doc = Document(io.BytesIO(file_bytes))
            text_parts = []
            for paragraph in doc.paragraphs:
                if paragraph.text.strip():
                    text_parts.append(paragraph.text.strip())
            return "\n\n".join(text_parts)
        except Exception as e:
            raise ValueError(f"Failed to extract text from DOCX: {e}")

    def extract_text_from_txt(self, file_bytes: bytes) -> str:
        """
        Extract text content from a TXT file.

        Args:
            file_bytes: The raw bytes of the TXT file.

        Returns:
            The extracted text content as a string.
        """
        try:
            return file_bytes.decode('utf-8')
        except UnicodeDecodeError:
            try:
                return file_bytes.decode('latin-1')
            except Exception as e:
                raise ValueError(f"Failed to decode text file: {e}")

    def extract_text(self, file_bytes: bytes, filename: str) -> str:
        """
        Extract text from a file based on its extension.

        Args:
            file_bytes: The raw bytes of the file.
            filename: The name of the file including extension.

        Returns:
            The extracted text content as a string.

        Raises:
            ValueError: If the file type is not supported.
        """
        filename_lower = filename.lower()
        if filename_lower.endswith('.pdf'):
            return self.extract_text_from_pdf(file_bytes)
        elif filename_lower.endswith('.docx'):
            return self.extract_text_from_docx(file_bytes)
        elif filename_lower.endswith('.txt'):
            return self.extract_text_from_txt(file_bytes)
        else:
            raise ValueError(f"Unsupported file type: {filename}. Supported types: PDF, DOCX, TXT")

    def generate_metadata(self, resume_text: str, filename: str) -> Dict:
        """
        Generate metadata for a resume using LLM analysis.
        Extracts candidate name, summary, key skills, experience level, and education.

        Args:
            resume_text: The full text content of the resume.
            filename: The original filename of the resume.

        Returns:
            A dictionary containing extracted metadata.
        """
        prompt = f"""Analyze this resume and extract the following information in a structured format.
Be concise but thorough.

Resume text:
{resume_text[:8000]}

Please provide:
1. CANDIDATE_NAME: The full name of the candidate (if not found, use "Unknown")
2. EMAIL: Email address if present (if not found, use "Not provided")
3. PHONE: Phone number if present (if not found, use "Not provided")
4. SUMMARY: A 2-3 sentence professional summary of this candidate
5. KEY_SKILLS: A comma-separated list of technical skills, tools, and technologies mentioned
6. EXPERIENCE_YEARS: Estimated years of professional experience (number only)
7. CURRENT_ROLE: Current or most recent job title
8. EDUCATION: Highest degree and institution
9. INDUSTRIES: Industries or domains they have experience in (comma-separated)

Format your response exactly like this:
CANDIDATE_NAME: [name]
EMAIL: [email]
PHONE: [phone]
SUMMARY: [summary]
KEY_SKILLS: [skills]
EXPERIENCE_YEARS: [years]
CURRENT_ROLE: [role]
EDUCATION: [education]
INDUSTRIES: [industries]"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert HR assistant that analyzes resumes and extracts key information accurately."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.1
            )
            
            response_text = response.choices[0].message.content
            metadata = self._parse_metadata_response(response_text, filename)
            return metadata
            
        except Exception as e:
            # Return basic metadata if LLM fails
            return {
                "candidate_name": "Unknown",
                "email": "Not provided",
                "phone": "Not provided",
                "summary": "Resume uploaded but metadata extraction failed.",
                "key_skills": [],
                "experience_years": 0,
                "current_role": "Unknown",
                "education": "Unknown",
                "industries": [],
                "filename": filename,
                "error": str(e)
            }

    def _parse_metadata_response(self, response_text: str, filename: str) -> Dict:
        """
        Parse the LLM response into a structured metadata dictionary.

        Args:
            response_text: The raw response from the LLM.
            filename: The original filename.

        Returns:
            A dictionary containing parsed metadata.
        """
        metadata = {
            "candidate_name": "Unknown",
            "email": "Not provided",
            "phone": "Not provided",
            "summary": "",
            "key_skills": [],
            "experience_years": 0,
            "current_role": "Unknown",
            "education": "Unknown",
            "industries": [],
            "filename": filename
        }

        lines = response_text.strip().split('\n')
        for line in lines:
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip().upper()
                value = value.strip()
                
                if key == "CANDIDATE_NAME":
                    metadata["candidate_name"] = value if value else "Unknown"
                elif key == "EMAIL":
                    metadata["email"] = value if value else "Not provided"
                elif key == "PHONE":
                    metadata["phone"] = value if value else "Not provided"
                elif key == "SUMMARY":
                    metadata["summary"] = value
                elif key == "KEY_SKILLS":
                    skills = [s.strip() for s in value.split(',') if s.strip()]
                    metadata["key_skills"] = skills
                elif key == "EXPERIENCE_YEARS":
                    try:
                        # Extract just the number
                        years_match = re.search(r'(\d+)', value)
                        if years_match:
                            metadata["experience_years"] = int(years_match.group(1))
                    except ValueError:
                        metadata["experience_years"] = 0
                elif key == "CURRENT_ROLE":
                    metadata["current_role"] = value if value else "Unknown"
                elif key == "EDUCATION":
                    metadata["education"] = value if value else "Unknown"
                elif key == "INDUSTRIES":
                    industries = [i.strip() for i in value.split(',') if i.strip()]
                    metadata["industries"] = industries

        return metadata

    def process_resume(self, file_bytes: bytes, filename: str) -> Tuple[str, Dict]:
        """
        Process a resume file: extract text and generate metadata.

        Args:
            file_bytes: The raw bytes of the resume file.
            filename: The name of the file.

        Returns:
            A tuple of (extracted_text, metadata_dict).
        """
        text = self.extract_text(file_bytes, filename)
        metadata = self.generate_metadata(text, filename)
        return text, metadata
