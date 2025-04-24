from typing import List, Tuple
from reranker import LLMReranker
from helpers import get_env_var
from dotenv import load_dotenv
import litellm
from classes import Resume

load_dotenv()


"""
evaluator flow:
1. generate rubric with o3
2. rerank candidates with o3 to generate a ground truth list of (candidate, score)
3. judge relevance of each candidate. if score >= 50, set relevance to true, otherwise false. output list of (candidate, is_relevant)
4. calculate map@10 using the ground truth list
"""


class Evaluator:
    def __init__(self):
        self.model_name = get_env_var("EVALUATOR_MODEL", "o3")
        self.top_k = 10

        litellm.set_verbose = False

    def evaluate(
        self,
        candidates: List[Tuple[Resume, float]],
        job_description: str,
    ) -> float:
        try:
            better_reranker = LLMReranker(candidates, self.model_name)
            ground_truth: List[Tuple[Resume, float]] = better_reranker.rerank(
                job_description, self.top_k
            )
            map_at_10 = self.calculate_map_at_k(candidates, ground_truth, self.top_k)

            return map_at_10

        except Exception as e:
            raise e

    def calculate_map_at_k(
        self,
        candidates: List[Tuple[Resume, float]],
        ground_truth: List[Tuple[Resume, float]],
    ) -> float:
        if not candidates or not ground_truth:
            return 0.0

        relevant_candidates = {
            candidate.resume_id for candidate, score in ground_truth if score >= 50
        }

        if not relevant_candidates:
            return 0.0

        candidate_ids = [
            candidate.resume_id for candidate, _ in candidates[: self.top_k]
        ]

        precisions = []
        num_relevant_found = 0

        for i, candidate_id in enumerate(candidate_ids):
            if candidate_id in relevant_candidates:
                num_relevant_found += 1
                precision_at_i = num_relevant_found / (i + 1)
                precisions.append(precision_at_i)

        if not precisions:
            return 0.0
        return sum(precisions) / len(relevant_candidates)
