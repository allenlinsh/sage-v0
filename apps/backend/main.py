from classes import Resume
from fastapi import FastAPI, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
from dotenv import load_dotenv

from ranker import BM25Ranker
from reranker import LLMReranker
from evaluator import Evaluator
from helpers import get_env_var


load_dotenv()

app = FastAPI(title="Resume Screening and Candidate Ranking API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def load_resumes() -> List[Resume]:
    try:
        df = pd.read_csv("./anonymized_resumes.csv")
        df = df.dropna(subset=["education", "location", "skills"])
        df = df[(df["education"] != "") & (df["location"] != "") & (df["skills"] != "")]
        df = df[(df["education"] != "[]") & (df["skills"] != "[]")]
        parsed_data = df.to_dict(orient="records")

        resumes = [Resume.from_csv_row(row) for row in parsed_data]

        return resumes
    except Exception as e:
        raise Exception(f"Error loading resumes: {str(e)}")


def rank_candidates(
    resumes: List[Resume],
    job_description: str,
) -> List[Tuple[Resume, float]]:
    try:
        ranker = BM25Ranker(resumes)

        ranked_candidates = ranker.rank(job_description)

        return ranked_candidates
    except Exception as e:
        raise Exception(f"Error ranking candidates: {str(e)}")


def rerank_candidates(
    ranked_candidates: List[Tuple[Resume, float]], job_description: str, top_k: int = 10
) -> List[Tuple[Resume, float]]:
    try:
        reranker_model = get_env_var("RERANKER_MODEL", "gpt-4.1-nano")
        reranker = LLMReranker(ranked_candidates, reranker_model)

        reranked_candidates = reranker.rerank(job_description, top_k)

        return reranked_candidates
    except Exception as e:
        raise Exception(f"Error reranking candidates: {str(e)}")


def evaluate_ranking(
    candidates: List[Tuple[Resume, float]],
    job_description: str,
    top_k: int = 10,
) -> Tuple[float, float]:
    try:
        evaluator = Evaluator(top_k)

        evaluation_results = evaluator.evaluate(candidates, job_description)

        return evaluation_results
    except Exception as e:
        raise Exception(f"Error evaluating ranking: {str(e)}")


@app.post("/run")
async def complete_pipeline(job_description: str = Form(...), top_k: int = Form(10)):
    try:
        candidates = load_resumes()

        print("\nRanking Candidates...")

        ranked_candidates = rank_candidates(candidates, job_description)

        print("\nBM25 Ranked Candidates\n")
        for candidate, score in ranked_candidates:
            print(f"{candidate.resume_id}: {score}")

        print("\nReranking Candidates...")

        reranked_candidates = rerank_candidates(
            ranked_candidates, job_description, top_k
        )

        print("\nLLM Reranked Candidates\n")
        for candidate, score in reranked_candidates:
            print(f"{candidate.resume_id}: {score}")

        print("\nEvaluating Candidates...")

        map_at_10, precision_at_10 = evaluate_ranking(
            reranked_candidates, job_description
        )

        print(f"\nMAP@10: {map_at_10}")
        print(f"\nPrecision@10: {precision_at_10}")

        return {
            "ranked_candidates": [(c[0].__dict__, c[1]) for c in ranked_candidates],
            "reranked_candidates": [(c[0].__dict__, c[1]) for c in reranked_candidates],
            "map_at_10": map_at_10,
            "precision_at_10": precision_at_10,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(e)}")


if __name__ == "__main__":
    port = int(get_env_var("PORT", "8000"))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
