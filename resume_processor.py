"""
resume processor module for extracting text from pdf and docx files
also generates metadata for each resume using llm analysis
handles validation to ensure only actual resumes are processed
"""

import re
import io
from typing import Dict, Optional, Tuple
from openai import OpenAI
from dotenv import load_dotenv
import os

# load env vars so we can access the api key
load_dotenv()


class ResumeProcessor:
    """
    handles the extraction of text from resume files like pdf, docx, and txt
    also generates metadata including candidate name, summary, and skills
    includes validation to reject non-resume documents
    """

    def __init__(self):
        """
        sets up the processor with an openai client for metadata generation
        raises an error if the api key isnt configured
        """
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set")
        self.client = OpenAI(api_key=api_key)

    def extract_text_from_pdf(self, file_bytes: bytes) -> str:
        """
        pulls text content out of a pdf file
        tries pdfplumber first since it usually gives better results
        falls back to pypdf2 if pdfplumber has issues
        
        takes the raw bytes of the pdf file
        returns the extracted text as a string
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
            # pdfplumber failed so try pypdf2 as backup
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
                raise ValueError(f"failed to extract text from pdf: {e}, {e2}")

    def extract_text_from_docx(self, file_bytes: bytes) -> str:
        """
        pulls text content out of a docx word document
        reads each paragraph and joins them together
        
        takes the raw bytes of the docx file
        returns the extracted text as a string
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
            raise ValueError(f"failed to extract text from docx: {e}")

    def extract_text_from_txt(self, file_bytes: bytes) -> str:
        """
        reads text content from a plain text file
        tries utf-8 encoding first then falls back to latin-1
        
        takes the raw bytes of the txt file
        returns the text content as a string
        """
        try:
            return file_bytes.decode('utf-8')
        except UnicodeDecodeError:
            try:
                return file_bytes.decode('latin-1')
            except Exception as e:
                raise ValueError(f"failed to decode text file: {e}")

    def extract_text(self, file_bytes: bytes, filename: str) -> str:
        """
        extracts text from a file based on its extension
        automatically detects the file type and uses the right method
        
        takes the raw bytes and filename
        returns the extracted text
        raises an error if the file type isnt supported
        """
        filename_lower = filename.lower()
        if filename_lower.endswith('.pdf'):
            return self.extract_text_from_pdf(file_bytes)
        elif filename_lower.endswith('.docx'):
            return self.extract_text_from_docx(file_bytes)
        elif filename_lower.endswith('.txt'):
            return self.extract_text_from_txt(file_bytes)
        else:
            raise ValueError(f"unsupported file type: {filename} - supported types are pdf, docx, txt")

    def validate_is_resume(self, text: str) -> Tuple[bool, str]:
        """
        checks whether a document is actually a resume or cv
        acts as a guardrail to reject random documents
        uses both heuristic checks and llm validation
        
        takes the extracted text content
        returns a tuple of (is_valid, reason) explaining the decision
        """
        # quick check first - resumes usually have certain keywords
        text_lower = text.lower()
        resume_indicators = [
            'experience', 'education', 'skills', 'employment', 'work history',
            'professional', 'resume', 'curriculum vitae', 'cv', 'objective',
            'summary', 'qualifications', 'references', 'certifications'
        ]
        indicator_count = sum(1 for indicator in resume_indicators if indicator in text_lower)
        
        # if almost no indicators found its probably not a resume
        if indicator_count < 2:
            return False, "document doesnt appear to be a resume - missing typical sections like experience, education, or skills"

        # use the llm for a more accurate check
        prompt = f"""analyze this document and determine if it is a resume or cv

document text (first 2500 characters):
{text[:2500]}

a resume must contain:
- a persons name (usually at the top)
- work experience or education history
- contact information or skills section

documents that are not resumes include:
- articles, essays, or research papers
- business documents, contracts, or reports
- cover letters (these are separate from resumes)
- random text or unrelated content

respond with exactly this format:
IS_RESUME: YES or NO
REASON: one sentence explanation"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "you are a document classifier - determine if documents are resumes or cvs - be strict and only accept actual resumes"
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.1,
                max_tokens=100
            )
            
            response_text = response.choices[0].message.content.strip()
            lines = response_text.split('\n')
            
            is_resume = False
            reason = "could not determine document type"
            
            for line in lines:
                if ':' in line:
                    key, value = line.split(':', 1)
                    key = key.strip().upper()
                    value = value.strip()
                    
                    if key == "IS_RESUME":
                        is_resume = value.upper() == "YES"
                    elif key == "REASON":
                        reason = value
            
            return is_resume, reason
            
        except Exception as e:
            # if llm fails fall back to the heuristic result
            if indicator_count >= 3:
                return True, "validated by heuristic check"
            return False, f"validation failed: {str(e)}"

    def generate_metadata(self, resume_text: str, filename: str) -> Dict:
        """
        generates structured metadata for a resume using llm analysis
        extracts things like name, summary, skills, experience level
        
        takes the full resume text and original filename
        returns a dictionary with all the extracted metadata
        """
        prompt = f"""analyze this resume and extract the following information in a structured format
be concise but thorough

resume text:
{resume_text[:8000]}

please provide:
1. CANDIDATE_NAME: the full name of the candidate (if not found use "unknown")
2. EMAIL: email address if present (if not found use "not provided")
3. PHONE: phone number if present (if not found use "not provided")
4. SUMMARY: a 2-3 sentence professional summary of this candidate
5. KEY_SKILLS: a comma-separated list of technical skills, tools, and technologies mentioned
6. EXPERIENCE_YEARS: estimated years of professional experience (number only)
7. CURRENT_ROLE: current or most recent job title
8. EDUCATION: highest degree and institution
9. INDUSTRIES: industries or domains they have experience in (comma-separated)

format your response exactly like this:
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
                        "content": "you are an expert hr assistant that analyzes resumes and extracts key information accurately"
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
            # return basic metadata if the llm call fails
            return {
                "candidate_name": "unknown",
                "email": "not provided",
                "phone": "not provided",
                "summary": "resume uploaded but metadata extraction failed",
                "key_skills": [],
                "experience_years": 0,
                "current_role": "unknown",
                "education": "unknown",
                "industries": [],
                "filename": filename,
                "error": str(e)
            }

    def _parse_metadata_response(self, response_text: str, filename: str) -> Dict:
        """
        parses the llm response into a structured metadata dictionary
        handles the key: value format from the llm output
        
        takes the raw llm response and original filename
        returns a clean metadata dictionary
        """
        metadata = {
            "candidate_name": "unknown",
            "email": "not provided",
            "phone": "not provided",
            "summary": "",
            "key_skills": [],
            "experience_years": 0,
            "current_role": "unknown",
            "education": "unknown",
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
                    metadata["candidate_name"] = value if value else "unknown"
                elif key == "EMAIL":
                    metadata["email"] = value if value else "not provided"
                elif key == "PHONE":
                    metadata["phone"] = value if value else "not provided"
                elif key == "SUMMARY":
                    metadata["summary"] = value
                elif key == "KEY_SKILLS":
                    skills = [s.strip() for s in value.split(',') if s.strip()]
                    metadata["key_skills"] = skills
                elif key == "EXPERIENCE_YEARS":
                    try:
                        # pull out just the number
                        years_match = re.search(r'(\d+)', value)
                        if years_match:
                            metadata["experience_years"] = int(years_match.group(1))
                    except ValueError:
                        metadata["experience_years"] = 0
                elif key == "CURRENT_ROLE":
                    metadata["current_role"] = value if value else "unknown"
                elif key == "EDUCATION":
                    metadata["education"] = value if value else "unknown"
                elif key == "INDUSTRIES":
                    industries = [i.strip() for i in value.split(',') if i.strip()]
                    metadata["industries"] = industries

        return metadata

    def process_resume(self, file_bytes: bytes, filename: str) -> Tuple[str, Dict]:
        """
        main method to process a resume file end to end
        extracts the text and then generates all the metadata
        
        takes the raw file bytes and filename
        returns a tuple of (extracted_text, metadata_dict)
        """
        text = self.extract_text(file_bytes, filename)
        metadata = self.generate_metadata(text, filename)
        return text, metadata
