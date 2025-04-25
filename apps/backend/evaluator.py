from typing import List, Tuple, Dict, Set
from reranker import LLMReranker
from helpers import get_env_var
from dotenv import load_dotenv
import litellm
from classes import Resume, BaseModel

load_dotenv()


"""
evaluator flow:
1. generate rubric with LLM
2. use LLM to make binary judgments about each candidate (relevant/not relevant)
3. calculate map@10 using the binary relevance judgments
"""


class BinaryRelevanceResult(BaseModel):
    """Model for the binary relevance judgment from the LLM"""

    reasoning: str
    is_relevant: bool


class Evaluator:
    def __init__(self, top_k: int = 10):
        self.model_name = get_env_var("EVALUATOR_MODEL", "o3")
        self.top_k = top_k

        litellm.set_verbose = False
        litellm.drop_params = True

    def evaluate(
        self,
        candidates: List[Tuple[Resume, float]],
        job_description: str,
    ) -> Tuple[float, float]:
        """Get MAP@k and precision@k for a given job description and candidates."""
        try:
            relevant_candidates = self.get_binary_relevance_judgments(
                candidates, job_description, self.top_k
            )

            map_at_k = self.map_at_k(candidates, relevant_candidates)

            precision_at_k = self.precision_at_k(candidates, relevant_candidates)

            return map_at_k, precision_at_k

        except Exception as e:
            raise e

    def get_binary_relevance_judgments(
        self, candidates: List[Tuple[Resume, float]], job_description: str, top_k: int
    ) -> Set[str]:
        """
        Use LLM to make binary relevance judgments for each candidate
        """
        top_candidates = candidates[: min(top_k, len(candidates))]

        if not top_candidates:
            return set()

        candidate_resumes = [candidate for candidate, _ in top_candidates]

        messages = [
            [
                {
                    "role": "system",
                    "content": self._create_relevance_prompt(
                        job_description, candidate
                    ),
                },
                {
                    "role": "user",
                    "content": "Return a JSON with the binary relevance judgment (true/false).",
                },
            ]
            for candidate in candidate_resumes
        ]

        responses = litellm.batch_completion(
            model=self.model_name,
            messages=messages,
            temperature=0.2,
            response_format=BinaryRelevanceResult,
        )

        relevant_ids = {
            candidate.resume_id
            for candidate, response in zip(candidate_resumes, responses)
            if BinaryRelevanceResult.model_validate_json(
                response.choices[0].message.content
            ).is_relevant
        }

        return relevant_ids

    def _create_relevance_prompt(self, job_description: str, candidate: Resume) -> str:
        """Create a prompt for binary relevance judgment"""
        return f"""
        You are an expert hiring manager assistant tasked with judging whether a candidate is relevant for a job.
        
        ## Job Description:
        {job_description}
        
        ## Candidate Resume:
        {candidate.resume_text}
        
        Based on the job description and the candidate's resume, determine if this candidate is relevant for the position.
        A candidate is RELEVANT if they meet the core requirements for the role and would reasonably be considered for an interview.
        A candidate is NOT RELEVANT if they clearly lack the necessary skills, experience, or qualifications for the role.
        
        Make a binary judgment (relevant or not relevant) and provide brief reasoning.
        """

    def precision_at_k(
        self,
        candidates: List[Tuple[Resume, float]],
        relevant_candidates: Set[str],
    ) -> float:
        if not candidates or not relevant_candidates:
            return 0.0

        top_k_ids = [candidate.resume_id for candidate, _ in candidates[: self.top_k]]

        relevant_in_top_k = sum(
            1 for candidate_id in top_k_ids if candidate_id in relevant_candidates
        )

        return relevant_in_top_k / self.top_k

    def map_at_k(
        self,
        candidates: List[Tuple[Resume, float]],
        relevant_candidates: Set[str],
    ) -> float:

        if not candidates or not relevant_candidates:
            return 0.0

        candidate_ids = [
            candidate.resume_id for candidate, _ in candidates[: self.top_k]
        ]

        all_precision_values = []

        average_precision = 0.0

        num_relevant_found = 0

        for i in range(len(candidate_ids)):
            position = i + 1
            candidate_id = candidate_ids[i]

            if candidate_id in relevant_candidates:
                num_relevant_found += 1

                relevant_count = 0
                for j in range(i + 1):
                    if candidate_ids[j] in relevant_candidates:
                        relevant_count += 1

                precision_at_i = relevant_count / position

                average_precision += precision_at_i

                all_precision_values.append((candidate_id, position, precision_at_i))

        # print(f"Precision values for relevant documents: {all_precision_values}")
        # print(
        #     f"Found {num_relevant_found} relevant documents out of {len(relevant_candidates)}"
        # )
        # print(f"Sum of precisions: {average_precision}")

        if num_relevant_found == 0:
            return 0.0

        map_score = average_precision / len(relevant_candidates)
        # print(f"MAP@{self.top_k}: {map_score}")

        return map_score
