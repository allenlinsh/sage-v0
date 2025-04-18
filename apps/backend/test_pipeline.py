#!/usr/bin/env python3

"""
Test script for the resume screening and candidate ranking pipeline.
This script creates sample resume data and runs it through the full pipeline.
"""

import os
import json
import tempfile
import argparse
from typing import List, Dict, Any
from dotenv import load_dotenv

# Import our modules
from parser import ResumeParser
from ranker import Ranker
from reranker import Reranker
from evaluator import Evaluator
from helpers import ensure_dir_exists, save_to_json, get_env_var

# Load environment variables
load_dotenv()

def create_sample_resumes(output_dir: str, count: int = 5) -> List[str]:
    """
    Create sample resume text files for testing
    
    Args:
        output_dir: Directory to save sample resumes
        count: Number of sample resumes to create
        
    Returns:
        List of file paths to the created resumes
    """
    ensure_dir_exists(output_dir)
    
    sample_resumes = [
        # Strong candidate for software engineering
        {
            "name": "Alex Johnson",
            "title": "Senior Software Engineer",
            "skills": ["python", "javascript", "react", "django", "aws", "docker", "kubernetes", "ci/cd", "git"],
            "education": "Master's Degree in Computer Science from Stanford University",
            "experience": "8 years of experience in full-stack development",
            "location": "San Francisco, CA (Remote)"
        },
        # Mid-level candidate
        {
            "name": "Jamie Smith",
            "title": "Full Stack Developer",
            "skills": ["javascript", "react", "node", "express", "mongodb", "git"],
            "education": "Bachelor's Degree in Computer Engineering",
            "experience": "4 years of experience in web development",
            "location": "Chicago, IL (Hybrid)"
        },
        # Junior candidate
        {
            "name": "Taylor Wilson",
            "title": "Junior Developer",
            "skills": ["python", "html", "css", "javascript", "flask"],
            "education": "Bachelor's Degree in Information Technology",
            "experience": "1 year of experience in web development",
            "location": "Remote"
        },
        # Data science candidate
        {
            "name": "Morgan Lee",
            "title": "Data Scientist",
            "skills": ["python", "sql", "pandas", "scikit-learn", "tensorflow", "data analysis", "machine learning"],
            "education": "PhD in Statistics from MIT",
            "experience": "5 years of experience in data science and analytics",
            "location": "Boston, MA"
        },
        # DevOps candidate
        {
            "name": "Casey Brown",
            "title": "DevOps Engineer",
            "skills": ["aws", "docker", "kubernetes", "terraform", "jenkins", "python", "linux", "bash"],
            "education": "Bachelor's in Systems Engineering",
            "experience": "6 years of experience in cloud infrastructure and automation",
            "location": "Austin, TX (Onsite)"
        }
    ]
    
    # Create more candidates if needed by varying the templates
    while len(sample_resumes) < count:
        base = sample_resumes[len(sample_resumes) % 5]
        variation = base.copy()
        variation["name"] = f"Variant {len(sample_resumes) + 1}"
        variation["skills"] = base["skills"][:3] + ["ruby"] + base["skills"][3:]
        variation["experience"] = f"{(len(sample_resumes) % 10) + 1} years of experience in software development"
        sample_resumes.append(variation)
    
    # Save resumes as text files
    resume_files = []
    for i, resume in enumerate(sample_resumes[:count]):
        # Format as a simple text resume
        resume_text = f"""
        {resume['name']}
        {resume['title']}
        
        SKILLS:
        {', '.join(resume['skills'])}
        
        EDUCATION:
        {resume['education']}
        
        EXPERIENCE:
        {resume['experience']}
        
        LOCATION:
        {resume['location']}
        """
        
        # Save to file
        file_path = os.path.join(output_dir, f"resume_{i+1}.txt")
        with open(file_path, 'w') as f:
            f.write(resume_text)
        
        resume_files.append(file_path)
    
    return resume_files

def create_sample_job_description() -> str:
    """Create a sample job description for testing"""
    return """
    Senior Software Engineer
    
    About the Role:
    We're looking for a Senior Software Engineer to join our growing team. You'll be responsible for designing, developing, and maintaining our core backend services and APIs.
    
    Requirements:
    - 5+ years of experience in software development
    - Strong proficiency in Python and JavaScript
    - Experience with web frameworks like Django or Flask
    - Knowledge of front-end technologies (React preferred)
    - Experience with cloud services (AWS, GCP, or Azure)
    - Experience with containerization (Docker, Kubernetes)
    - Strong understanding of CI/CD pipelines
    - Excellent problem-solving and communication skills
    
    Nice to Have:
    - Experience with microservices architecture
    - Knowledge of database design and optimization
    - Experience with GraphQL
    - Open-source contributions
    
    Location:
    Remote (US-based) or hybrid option available in our San Francisco office.
    """

def main(args):
    print("Resume Screening and Candidate Ranking Pipeline Test")
    print("=" * 60)
    
    # Create a temporary directory for test files
    temp_dir = args.output_dir or os.path.join(os.path.dirname(__file__), "output")
    ensure_dir_exists(temp_dir)
    
    print(f"Using directory: {temp_dir}")
    
    # 1. Create sample resumes
    print("\n1. Creating sample resumes...")
    resume_files = create_sample_resumes(temp_dir, args.count)
    print(f"Created {len(resume_files)} sample resumes")
    
    # 2. Create job description
    print("\n2. Creating sample job description...")
    job_description = create_sample_job_description()
    job_desc_path = os.path.join(temp_dir, "job_description.txt")
    with open(job_desc_path, 'w') as f:
        f.write(job_description)
    print(f"Job description saved to {job_desc_path}")
    
    # 3. Parse resumes
    print("\n3. Parsing resumes...")
    parser = ResumeParser()
    parsed_resumes = []
    
    for file_path in resume_files:
        print(f"  Parsing {os.path.basename(file_path)}...")
        with open(file_path, 'rb') as f:
            content = f.read()
        parsed_resume = parser.parse(content, os.path.basename(file_path))
        parsed_resumes.append(parsed_resume)
    
    # Save parsed resumes
    parsed_resumes_path = os.path.join(temp_dir, "parsed_resumes.json")
    save_to_json(parsed_resumes, parsed_resumes_path)
    print(f"Parsed resumes saved to {parsed_resumes_path}")
    
    # 4. Rank candidates
    print("\n4. Ranking candidates...")
    ranker = Ranker()
    
    # Define filters if needed
    filters = {
        "min_experience": 2 if args.filter else None,
        "location": "remote" if args.filter else None
    }
    
    # Apply filter only if requested
    ranking_filters = filters if args.filter else None
    ranked_candidates = ranker.rank(parsed_resumes, job_description, ranking_filters)
    
    # Save ranked candidates
    ranked_path = os.path.join(temp_dir, "ranked_candidates.json")
    save_to_json(ranked_candidates, ranked_path)
    print(f"Ranked candidates saved to {ranked_path}")
    
    # Print top 3 ranked candidates
    print("\nTop 3 candidates from initial ranking:")
    for i, candidate in enumerate(ranked_candidates[:3]):
        print(f"  {i+1}. {candidate.get('filename', 'Unknown')} - Score: {candidate.get('final_score', 0):.2f}")
        print(f"     Skills: {', '.join(candidate.get('skills', []))[:80]}...")
    
    # 5. Re-rank candidates with LLM if enabled
    if args.rerank:
        if not os.environ.get("OPENAI_API_KEY"):
            print("\n⚠️ OPENAI_API_KEY environment variable not set. Skipping re-ranking.")
        else:
            print("\n5. Re-ranking candidates with LLM...")
            reranker = Reranker()
            reranked_candidates = reranker.rerank(ranked_candidates, job_description, top_k=args.count)
            
            # Save re-ranked candidates
            reranked_path = os.path.join(temp_dir, "reranked_candidates.json")
            save_to_json(reranked_candidates, reranked_path)
            print(f"Re-ranked candidates saved to {reranked_path}")
            
            # Print top 3 re-ranked candidates
            print("\nTop 3 candidates after LLM re-ranking:")
            for i, candidate in enumerate(reranked_candidates[:3]):
                print(f"  {i+1}. {candidate.get('filename', 'Unknown')} - Score: {candidate.get('llm_score', 0)}")
                if "llm_assessment" in candidate:
                    print(f"     Assessment: {candidate['llm_assessment'][:80]}...")
    
    # 6. Evaluate results if enabled
    if args.evaluate:
        if not os.environ.get("OPENAI_API_KEY"):
            print("\n⚠️ OPENAI_API_KEY environment variable not set. Skipping evaluation.")
        else:
            print("\n6. Evaluating ranking results...")
            evaluator = Evaluator()
            
            # Use re-ranked candidates if available, otherwise use ranked candidates
            candidates_to_evaluate = reranked_candidates if args.rerank else ranked_candidates
            
            evaluation_results = evaluator.evaluate(candidates_to_evaluate, job_description)
            
            # Save evaluation results
            eval_path = os.path.join(temp_dir, "evaluation_results.json")
            save_to_json(evaluation_results, eval_path)
            print(f"Evaluation results saved to {eval_path}")
            
            # Print some evaluation metrics
            print("\nEvaluation Metrics:")
            if "ranking_quality" in evaluation_results:
                quality = evaluation_results["ranking_quality"]
                print(f"  Alignment Score: {quality.get('alignment_score', 'N/A')}")
                print(f"  Effectiveness Score: {quality.get('effectiveness_score', 'N/A')}")
            
            # Print recommendations
            if "recommended_improvements" in evaluation_results:
                print("\nRecommended Improvements:")
                for i, rec in enumerate(evaluation_results["recommended_improvements"][:3]):
                    print(f"  {i+1}. {rec}")
    
    print("\nPipeline test completed successfully!")
    print(f"All output files are available in: {temp_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the resume screening pipeline")
    parser.add_argument("--count", type=int, default=5, help="Number of sample resumes to create")
    parser.add_argument("--output-dir", type=str, help="Directory to save output files")
    parser.add_argument("--filter", action="store_true", help="Apply filters to ranking")
    parser.add_argument("--rerank", action="store_true", help="Re-rank candidates with LLM")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate ranking results")
    args = parser.parse_args()
    
    main(args) 