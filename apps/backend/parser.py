import re
import pandas as pd
from typing import Dict, Any, List, Optional, Set, Tuple
import os
import json
from io import BytesIO

class ResumeParser:
    """
    Parser to extract key information from resumes.
    In a production system, we would use more sophisticated libraries like
    PyPDF2, textract, or even OCR solutions for different file types.
    """
    
    def __init__(self):
        # Common skills to detect (this would be more comprehensive in production)
        self.skills_keywords = {
            "programming": {"python", "java", "javascript", "typescript", "c++", "c#", "go", "rust", "ruby", "php"},
            "web": {"react", "angular", "vue", "node", "django", "flask", "fastapi", "html", "css", "bootstrap"},
            "data": {"sql", "nosql", "mongodb", "postgresql", "mysql", "data analysis", "data science", "machine learning", "ml", "ai"},
            "cloud": {"aws", "azure", "gcp", "docker", "kubernetes", "terraform", "serverless"},
            "tools": {"git", "jenkins", "ci/cd", "agile", "scrum", "jira"}
        }
        
        # Education keywords
        self.education_keywords = {"bachelor", "master", "phd", "degree", "bs", "ms", "ba", "computer science", "engineering"}
        
        # Experience level patterns
        self.experience_patterns = [
            r'(\d+)\s*\+?\s*years?\s+(?:of\s+)?experience',
            r'experience\s*:\s*(\d+)\s*\+?\s*years?',
            r'(\d+)\s*\+?\s*years?\s+(?:of\s+)?work'
        ]
        
    def parse(self, content: bytes, filename: str) -> Dict[str, Any]:
        """
        Parse resume content and extract key information
        
        Args:
            content: Resume file content as bytes
            filename: Original filename
            
        Returns:
            Dict with structured resume information
        """
        try:
            # In a real implementation, we would handle different file types
            # For now, assume text content for simplicity
            text = content.decode('utf-8', errors='ignore').lower()
            
            # Basic resume information
            parsed_data = {
                "filename": filename,
                "skills": self._extract_skills(text),
                "education": self._extract_education(text),
                "experience_years": self._extract_experience(text),
                "location": self._extract_location(text),
                "summary": self._generate_summary(text),
                "original_text": text[:5000]  # Truncate for storage efficiency
            }
            
            return parsed_data
            
        except Exception as e:
            # Log the error and return minimal information
            return {
                "filename": filename,
                "error": str(e),
                "skills": [],
                "education": "",
                "experience_years": 0,
                "location": "",
                "summary": "",
                "original_text": ""
            }
    
    def _extract_skills(self, text: str) -> List[str]:
        """Extract skills from resume text"""
        found_skills = []
        
        # Flatten skills dictionary for matching
        all_skills = set()
        for category_skills in self.skills_keywords.values():
            all_skills.update(category_skills)
        
        # Find skills in text
        for skill in all_skills:
            # Use word boundary search for better matching
            if re.search(r'\b' + re.escape(skill) + r'\b', text):
                found_skills.append(skill)
                
        return found_skills
    
    def _extract_education(self, text: str) -> str:
        """Extract education information as a simple string"""
        education_info = []
        
        for keyword in self.education_keywords:
            if keyword in text:
                # Find sentences containing education keywords
                sentences = re.findall(r'[^.!?]*' + re.escape(keyword) + r'[^.!?]*[.!?]', text)
                for sentence in sentences:
                    education_info.append(sentence.strip())
        
        # Return the first found education info or empty string
        return education_info[0] if education_info else ""
    
    def _extract_experience(self, text: str) -> int:
        """Extract years of experience"""
        for pattern in self.experience_patterns:
            matches = re.search(pattern, text)
            if matches:
                try:
                    return int(matches.group(1))
                except (ValueError, IndexError):
                    pass
        
        return 0  # Default if no experience found
    
    def _extract_location(self, text: str) -> str:
        """Extract location information (simple implementation)"""
        # In a real system, we would use NER or a location database
        # This is a placeholder implementation
        
        # Common location indicators
        location_patterns = [
            r'location\s*:\s*([A-Za-z\s,]+)',
            r'city\s*:\s*([A-Za-z\s,]+)',
            r'(?:lives|residing) in\s+([A-Za-z\s,]+)',
            r'remote|onsite|hybrid'
        ]
        
        for pattern in location_patterns:
            matches = re.search(pattern, text)
            if matches:
                try:
                    return matches.group(1).strip()
                except (IndexError):
                    if 'remote' in text:
                        return 'remote'
                    elif 'onsite' in text:
                        return 'onsite'
                    elif 'hybrid' in text:
                        return 'hybrid'
        
        return ""  # Default if no location found
    
    def _generate_summary(self, text: str) -> str:
        """
        Generate a brief summary of the resume
        In a production system, this would use an LLM or sophisticated summarization
        """
        # Simple implementation - just take the first 200 characters
        summary = text[:200] + "..." if len(text) > 200 else text
        return summary
    
    def bulk_parse(self, resume_files_dir: str) -> pd.DataFrame:
        """
        Parse multiple resumes in a directory and return as a DataFrame
        
        Args:
            resume_files_dir: Directory containing resume files
            
        Returns:
            DataFrame with parsed resumes
        """
        parsed_data = []
        
        for filename in os.listdir(resume_files_dir):
            file_path = os.path.join(resume_files_dir, filename)
            
            try:
                with open(file_path, 'rb') as f:
                    content = f.read()
                parsed_resume = self.parse(content, filename)
                parsed_data.append(parsed_resume)
            except Exception as e:
                print(f"Error parsing {filename}: {str(e)}")
        
        return pd.DataFrame(parsed_data)
    
    def save_to_csv(self, parsed_data: List[Dict[str, Any]], output_file: str) -> str:
        """Save parsed resume data to CSV file"""
        df = pd.DataFrame(parsed_data)
        df.to_csv(output_file, index=False)
        return output_file 