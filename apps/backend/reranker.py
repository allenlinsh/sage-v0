import json
from classes import EvaluationResult, EvaluationRubric, Resume
import litellm
import re
from typing import Dict, Any, List, Tuple
import time
from dotenv import load_dotenv

load_dotenv()


class LLMReranker:
    def __init__(
        self,
        ranked_candidates: List[Tuple[Resume, float]],
        model_name: str = "gpt-4.1-nano",
    ):
        self.ranked_candidates = ranked_candidates
        self.model_name = model_name

        litellm.set_verbose = False

    def rerank(
        self, job_description: str, top_k: int = 10
    ) -> List[Tuple[Resume, float]]:
        top_candidates = self.ranked_candidates[
            : min(top_k, len(self.ranked_candidates))
        ]

        if not top_candidates:
            return []

        reranked_candidates = self.batch_rerank(top_candidates, job_description)

        return sorted(reranked_candidates, key=lambda x: x[1], reverse=True)

    def batch_rerank(
        self, top_candidates: List[Tuple[Resume, float]], job_description: str
    ) -> List[Tuple[Resume, float]]:
        reranked_candidates = []

        rubric = self.generate_evaluation_rubric(job_description)

        reranked_candidates = self.evaluate_candidates(
            next(zip(*top_candidates)), job_description, rubric
        )

        return reranked_candidates

    def generate_evaluation_rubric(self, job_description: str) -> EvaluationRubric:
        system_prompt = f"""
        You are an expert hiring manager assistant. Based on the following job description, create a structured evaluation rubric that can be used to score candidates.
        The rubric should include 3-5 key criteria that are most important for this role.
        
        For each criterion, specify:
        1. The name of the criterion
        2. Why it's important for this role
        3. What would constitute a score range of 80-100 out of 100
        4. What would constitute a score range of 60-79 out of 100
        5. What would constitute a score range of 40-59 out of 100
        6. What would constitute a score range of 20-39 out of 100
        7. What would constitute a score range of 0-19 out of 100
        
        Job Description:
        {job_description}
        
        Return the rubric as a structured JSON object with each criterion and its details.
        """

        try:
            response = litellm.completion(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": "Generate the rubric."},
                ],
                temperature=0.2,
                response_format=EvaluationRubric,
            )

            response_message = response.choices[0].message.content

            return EvaluationRubric.model_validate_json(response_message)

        except Exception as e:
            raise e

    def evaluate_candidates(
        self,
        candidates: List[Resume],
        job_description: str,
        rubric: EvaluationRubric,
    ) -> List[Tuple[Resume, float]]:
        rubric_text = self.prepare_rubric(rubric)

        try:
            responses = litellm.batch_completion(
                model=self.model_name,
                messages=[
                    [
                        {
                            "role": "system",
                            "content": self.prepare_system_prompt(
                                job_description, candidate, rubric_text
                            ),
                        },
                        {
                            "role": "user",
                            "content": "Return the numerical score as a float value.",
                        },
                    ]
                    for candidate in candidates
                ],
                temperature=0.2,
                response_format=EvaluationResult,
            )

            return [
                (
                    candidate,
                    EvaluationResult.model_validate_json(
                        response.choices[0].message.content
                    ).score,
                )
                for candidate, response in zip(candidates, responses)
            ]

        except Exception as e:
            raise e

    def prepare_system_prompt(
        self, job_description: str, candidate: Resume, rubric_text: str
    ):
        return f"""
        You are an expert hiring manager assistant. Evaluate the following candidate for a job using the provided rubric.
        
        ## Job Description:
        {job_description}
        
        ## Candidate Resume:
        {candidate.resume_text}
        
        ## Evaluation Rubric:
        {rubric_text}
        
        Based on the candidate's resume and the evaluation rubric, provide a single numerical score between 0 and 100 that represents how well the candidate matches the job description.
        """

    def prepare_rubric(self, rubric: EvaluationRubric) -> str:
        rubric_text = "EVALUATION CRITERIA:\n\n"

        for criterion in rubric.criteria:
            rubric_text += (
                f"Criterion: {criterion.name} (Importance: {criterion.importance})\n"
            )
            rubric_text += f"- Excellent (80-100): {criterion.score_80_100}\n"
            rubric_text += f"- Good (60-79): {criterion.score_60_79}\n"
            rubric_text += f"- Average (40-59): {criterion.score_40_59}\n"
            rubric_text += f"- Below Average (20-39): {criterion.score_20_39}\n"
            rubric_text += f"- Poor (0-19): {criterion.score_0_19}\n\n"

        return rubric_text
