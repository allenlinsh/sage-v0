import re
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Union, Tuple
from rank_bm25 import BM25Okapi
import json

class Ranker:
    """
    Ranker class for ranking candidates based on job description using BM25 algorithm.
    """
    
    def __init__(self):
        # Weightage for different resume components
        self.weights = {
            "skills": 0.4,
            "education": 0.2,
            "experience": 0.3,
            "location": 0.1
        }
        
        # Experience scoring thresholds
        self.experience_thresholds = {
            "entry": 2,      # 0-2 years
            "mid": 5,        # 3-5 years
            "senior": 8,     # 6-8 years
            "lead": float('inf')  # 9+ years
        }
    
    def rank(self, 
             candidates: List[Dict[str, Any]], 
             job_description: str,
             filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Rank candidates based on job description and filtering criteria
        
        Args:
            candidates: List of parsed candidate resumes
            job_description: Job description text
            filters: Optional filtering criteria (e.g., min_experience, location)
            
        Returns:
            List of candidates with ranking scores
        """
        # Apply filters first if provided
        if filters:
            filtered_candidates = self._apply_filters(candidates, filters)
            
            # If all candidates were filtered out but there were original candidates,
            # log a warning and use the original candidates
            if not filtered_candidates and candidates:
                print("Warning: All candidates were filtered out. Using original candidates.")
                filtered_candidates = candidates
        else:
            filtered_candidates = candidates
        
        if not filtered_candidates:
            return []
        
        # Process job description
        job_keywords = self._extract_keywords(job_description)
        
        # Compute scores for each candidate
        ranked_candidates = []
        for candidate in filtered_candidates:
            # Compute component scores
            skills_score = self._compute_skills_score(candidate, job_keywords)
            education_score = self._compute_education_score(candidate, job_description)
            experience_score = self._compute_experience_score(candidate, job_description)
            location_score = self._compute_location_score(candidate, filters.get('location', '')) if filters else 0.5
            
            # Calculate composite score
            composite_score = (
                skills_score * self.weights["skills"] + 
                education_score * self.weights["education"] + 
                experience_score * self.weights["experience"] + 
                location_score * self.weights["location"]
            )
            
            # Add BM25 score
            bm25_score = self._compute_bm25_score(candidate, job_description)
            
            # Create ranked candidate entry
            ranked_candidate = candidate.copy()
            ranked_candidate.update({
                "skills_score": round(skills_score, 2),
                "education_score": round(education_score, 2),
                "experience_score": round(experience_score, 2),
                "location_score": round(location_score, 2),
                "bm25_score": round(bm25_score, 2),
                "composite_score": round(composite_score, 2),
                "final_score": round((composite_score + bm25_score) / 2, 2)  # Average of composite and BM25
            })
            
            ranked_candidates.append(ranked_candidate)
        
        # Sort by final_score in descending order
        ranked_candidates.sort(key=lambda x: x["final_score"], reverse=True)
        
        # Add rank
        for i, candidate in enumerate(ranked_candidates):
            candidate["rank"] = i + 1
        
        return ranked_candidates
    
    def _apply_filters(self, candidates: List[Dict[str, Any]], filters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Apply filters to candidates"""
        filtered = candidates.copy()
        
        # Skip filtering if no valid filters or no candidates
        if not filters or not filtered:
            return filtered
            
        # Remove None or empty values from filters
        effective_filters = {k: v for k, v in filters.items() if v is not None and v != ""}
        
        if not effective_filters:
            return filtered
        
        # Keep track of original count
        original_count = len(filtered)
        
        # Filter by minimum experience
        if 'min_experience' in effective_filters:
            min_exp = effective_filters['min_experience']
            filtered = [c for c in filtered if c.get('experience_years', 0) >= min_exp]
            
            # If all candidates were filtered out, relax the filter
            if not filtered and original_count > 0:
                print(f"All candidates filtered out by min_experience={min_exp}. Relaxing filter.")
                filtered = candidates.copy()
        
        # Filter by maximum experience
        if 'max_experience' in effective_filters:
            max_exp = effective_filters['max_experience']
            filtered = [c for c in filtered if c.get('experience_years', 0) <= max_exp]
            
            # If all candidates were filtered out, relax the filter
            if not filtered and original_count > 0:
                print(f"All candidates filtered out by max_experience={max_exp}. Relaxing filter.")
                filtered = candidates.copy()
        
        # Filter by location
        if 'location' in effective_filters and effective_filters['location']:
            loc = effective_filters['location'].lower()
            if loc in ['remote', 'onsite', 'hybrid']:
                # Exact match for work arrangement
                location_filtered = [c for c in filtered if loc in c.get('location', '').lower()]
                # Only apply if we don't filter out everything
                if location_filtered or len(filtered) == 0:
                    filtered = location_filtered
            else:
                # Partial match for geographical location
                location_filtered = [c for c in filtered if loc in c.get('location', '').lower()]
                # Only apply if we don't filter out everything
                if location_filtered or len(filtered) == 0:
                    filtered = location_filtered
        
        # Filter by required skills
        if 'required_skills' in effective_filters and effective_filters['required_skills']:
            req_skills = set(s.lower() for s in effective_filters['required_skills'])
            skills_filtered = [
                c for c in filtered 
                if req_skills.issubset(set(s.lower() for s in c.get('skills', [])))
            ]
            
            # Only apply if we don't filter out everything
            if skills_filtered or len(filtered) == 0:
                filtered = skills_filtered
        
        return filtered
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract meaningful keywords from text"""
        # Remove common stopwords and punctuation
        text = text.lower()
        # Simple preprocessing - in production we'd use NLP libraries
        words = re.findall(r'\b[a-z]{3,}\b', text)
        return words
    
    def _compute_skills_score(self, candidate: Dict[str, Any], job_keywords: List[str]) -> float:
        """Compute skills match score"""
        candidate_skills = set(s.lower() for s in candidate.get('skills', []))
        
        if not candidate_skills:
            return 0.0
        
        # Count matching skills in job keywords
        matches = sum(1 for skill in candidate_skills if skill in job_keywords)
        
        # Calculate score as percentage of matching skills
        return min(1.0, matches / max(1, len(candidate_skills)))
    
    def _compute_education_score(self, candidate: Dict[str, Any], job_description: str) -> float:
        """Compute education match score"""
        education = candidate.get('education', '').lower()
        
        if not education:
            return 0.2  # Base score for unknown education
        
        # Check for degree mentions in education
        degree_score = 0.0
        if 'phd' in education or 'doctorate' in education:
            degree_score = 1.0
        elif 'master' in education or 'ms' in education or 'ma' in education:
            degree_score = 0.8
        elif 'bachelor' in education or 'bs' in education or 'ba' in education:
            degree_score = 0.6
        else:
            degree_score = 0.4  # Some education but degree not specified
        
        # Check if education field matches job description
        field_score = 0.0
        job_desc_lower = job_description.lower()
        
        relevant_fields = ['computer science', 'engineering', 'information technology', 
                         'data science', 'mathematics', 'statistics']
        
        for field in relevant_fields:
            if field in education:
                if field in job_desc_lower:
                    field_score = 1.0
                else:
                    field_score = 0.7
                break
        
        # Return weighted combination of degree and field scores
        return 0.6 * degree_score + 0.4 * field_score
    
    def _compute_experience_score(self, candidate: Dict[str, Any], job_description: str) -> float:
        """Compute experience match score"""
        years = candidate.get('experience_years', 0)
        
        # Extract expected experience from job description
        expected_years = self._extract_expected_experience(job_description)
        
        if expected_years == 0:
            # If expected experience not specified, use a bell curve with peak at 5 years
            if years <= 2:
                return 0.5  # Entry level
            elif years <= 5:
                return 0.9  # Mid level (preferred)
            elif years <= 8:
                return 0.7  # Senior level
            else:
                return 0.5  # Very senior might be overqualified
        else:
            # Score based on how close candidate's experience is to expected
            difference = abs(years - expected_years)
            if difference == 0:
                return 1.0
            elif difference <= 2:
                return 0.8
            elif difference <= 4:
                return 0.6
            else:
                return 0.4
    
    def _extract_expected_experience(self, job_description: str) -> int:
        """Extract expected years of experience from job description"""
        job_desc_lower = job_description.lower()
        
        # Common patterns for experience requirements
        patterns = [
            r'(\d+)\s*\+?\s*years?\s+(?:of\s+)?experience',
            r'experience\s*:\s*(\d+)\s*\+?\s*years?',
            r'minimum\s+(?:of\s+)?(\d+)\s*\+?\s*years?'
        ]
        
        for pattern in patterns:
            matches = re.search(pattern, job_desc_lower)
            if matches:
                try:
                    return int(matches.group(1))
                except (ValueError, IndexError):
                    pass
        
        # Check for experience level keywords
        if 'entry level' in job_desc_lower or 'junior' in job_desc_lower:
            return 1
        elif 'mid level' in job_desc_lower:
            return 3
        elif 'senior' in job_desc_lower:
            return 5
        elif 'lead' in job_desc_lower or 'principal' in job_desc_lower:
            return 8
        
        return 0  # Default if no experience requirement found
    
    def _compute_location_score(self, candidate: Dict[str, Any], preferred_location: str) -> float:
        """Compute location match score"""
        candidate_location = candidate.get('location', '').lower()
        
        if not candidate_location or not preferred_location:
            return 0.5  # Neutral score if location info missing
        
        preferred_location = preferred_location.lower()
        
        # Check for exact match
        if candidate_location == preferred_location:
            return 1.0
        
        # Check for work arrangement match (remote, onsite, hybrid)
        work_arrangements = ['remote', 'onsite', 'hybrid']
        if preferred_location in work_arrangements and preferred_location in candidate_location:
            return 1.0
        
        # Partial match (e.g., candidate in "New York City" and preferred is "New York")
        if preferred_location in candidate_location or candidate_location in preferred_location:
            return 0.8
        
        return 0.3  # Location mismatch
    
    def _compute_bm25_score(self, candidate: Dict[str, Any], job_description: str) -> float:
        """Compute BM25 relevance score"""
        # Create a simple document from candidate data
        candidate_text = " ".join([
            candidate.get('original_text', ''),
            " ".join(candidate.get('skills', [])),
            candidate.get('education', ''),
            candidate.get('location', '')
        ]).lower()
        
        # Tokenize
        candidate_tokens = self._extract_keywords(candidate_text)
        job_tokens = self._extract_keywords(job_description)
        
        if not candidate_tokens or not job_tokens:
            return 0.0
        
        # Create BM25 instance with a single document (the candidate)
        bm25 = BM25Okapi([candidate_tokens])
        
        # Get BM25 score
        score = bm25.get_scores(job_tokens)[0]
        
        # Normalize score to 0-1 range
        # BM25 can be arbitrarily large, so we use a sigmoid-like normalization
        normalized_score = min(1.0, score / 20.0)  # Assuming most scores are < 20
        
        return normalized_score
    
    def save_rankings(self, ranked_candidates: List[Dict[str, Any]], output_file: str) -> str:
        """Save rankings to a CSV file"""
        # Convert to DataFrame for easier saving
        df = pd.DataFrame(ranked_candidates)
        
        # Select relevant columns
        columns = [
            'rank', 'filename', 'final_score', 'composite_score', 'bm25_score',
            'skills_score', 'education_score', 'experience_score', 'location_score',
            'skills', 'education', 'experience_years', 'location', 'summary'
        ]
        
        # Filter to only columns that exist
        existing_columns = [col for col in columns if col in df.columns]
        df = df[existing_columns]
        
        # Save to CSV
        df.to_csv(output_file, index=False)
        return output_file 