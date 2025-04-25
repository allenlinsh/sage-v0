import pytest
import json
import pandas as pd
import os
from typing import List, Tuple

from classes import Resume, Education
from ranker import BM25Ranker
from reranker import LLMReranker
from evaluator import Evaluator
from helpers import get_env_var
from main import load_resumes, rank_candidates, evaluate_ranking, rerank_candidates


def load_job_descriptions() -> dict[str, str]:
    job_descriptions = {}
    job_dir = "example_job_descriptions"

    for filename in os.listdir(job_dir):
        if filename.endswith(".txt"):
            file_path = os.path.join(job_dir, filename)
            with open(file_path, "r") as file:
                job_descriptions[filename.split(".")[0]] = file.read()
    return job_descriptions


def test_pipeline():
    resumes = load_resumes()

    job_descriptions = load_job_descriptions()

    topk = 10
    for job_name, job_description in job_descriptions.items():
        print(f"Testing job description {job_name}\n")

        ranked_candidates = rank_candidates(resumes, job_description)
        print("All candidates and scores:")
        for candidate, score in ranked_candidates:
            print(f"ID: {candidate.resume_id}, Score: {score:.4f}")

        top_candidates = ranked_candidates[:topk]
        mapa10, pa10 = evaluate_ranking(top_candidates, job_description, topk)
        print("Pre-reranking:")
        print(f"MAP@{topk}: {mapa10}, Precision@{topk}: {pa10}\n\n")

        reranked_candidates = rerank_candidates(top_candidates, job_description)
        print(f"Top {topk} reranked candidates and scores:")
        for candidate, score in reranked_candidates:
            print(f"ID: {candidate.resume_id}, Score: {score:.4f}")

        mapa10, pa10 = evaluate_ranking(reranked_candidates, job_description, topk)
        print(f"MAP@{topk}: {mapa10}, Precision@{topk}: {pa10}")


if __name__ == "__main__":
    test_pipeline()
