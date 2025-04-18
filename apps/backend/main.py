from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from typing import List, Dict, Any, Optional
import json
import os
from dotenv import load_dotenv

# Import our modules
from parser import ResumeParser
from ranker import Ranker
from reranker import Reranker
from evaluator import Evaluator
from helpers import ensure_dir_exists, save_to_json, load_from_json, save_uploaded_file, get_env_var

# Load environment variables
load_dotenv()

app = FastAPI(title="Resume Screening and Candidate Ranking API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Resume Screening and Candidate Ranking API"}

@app.post("/parse-resumes")
async def parse_resumes(resumes: List[UploadFile] = File(...)):
    """Parse uploaded resumes and extract key information"""
    parser = ResumeParser()
    parsed_data = []
    
    for resume in resumes:
        content = await resume.read()
        # Save file temporarily or process directly
        parsed_resume = parser.parse(content, resume.filename)
        parsed_data.append(parsed_resume)
    
    return {"parsed_resumes": parsed_data}

@app.post("/rank-candidates")
async def rank_candidates(
    parsed_resumes: List[Dict[str, Any]],
    job_description: str,
    filters: Optional[Dict[str, Any]] = None
):
    """Rank candidates based on job description and filters"""
    ranker = Ranker()
    ranked_candidates = ranker.rank(parsed_resumes, job_description, filters)
    return {"ranked_candidates": ranked_candidates}

@app.post("/rerank-candidates")
async def rerank_candidates(
    candidates: List[Dict[str, Any]],
    job_description: str,
    top_k: int = 10
):
    """Re-rank top candidates using LLM"""
    reranker = Reranker()
    reranked_candidates = reranker.rerank(candidates, job_description, top_k)
    return {"reranked_candidates": reranked_candidates}

@app.post("/evaluate-ranking")
async def evaluate_ranking(
    candidates: List[Dict[str, Any]],
    job_description: str,
    ground_truth: Optional[List[Dict[str, Any]]] = None
):
    """Evaluate the ranking results"""
    evaluator = Evaluator()
    evaluation_results = evaluator.evaluate(candidates, job_description, ground_truth)
    return {"evaluation_results": evaluation_results}

@app.post("/complete-pipeline")
async def complete_pipeline(
    resumes: List[UploadFile] = File(...),
    job_description: str = Form(...),
    filters: str = Form("{}"),
    top_k: int = Form(10)
):
    """Run the complete pipeline from parsing to evaluation"""
    try:
        # Parse filters
        filters_dict = json.loads(filters)
        
        # 1. Parse resumes
        parser = ResumeParser()
        parsed_resumes = []
        
        for resume in resumes:
            content = await resume.read()
            parsed_resume = parser.parse(content, resume.filename)
            parsed_resumes.append(parsed_resume)
        
        # 2. Rank candidates
        ranker = Ranker()
        ranked_candidates = ranker.rank(parsed_resumes, job_description, filters_dict)
        
        # 3. Re-rank top candidates
        reranker = Reranker()
        reranked_candidates = reranker.rerank(ranked_candidates, job_description, top_k)
        
        # 4. Evaluate results
        evaluator = Evaluator()
        evaluation_results = evaluator.evaluate(reranked_candidates, job_description)
        
        return {
            "parsed_resumes": parsed_resumes,
            "ranked_candidates": ranked_candidates,
            "reranked_candidates": reranked_candidates,
            "evaluation_results": evaluation_results
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(e)}")

if __name__ == "__main__":
    port = int(get_env_var("PORT", "8000"))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True) 