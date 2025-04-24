from typing import List, Tuple, Dict, Any, Optional

import numpy as np

from classes import Resume


class BM25Ranker:
    def __init__(self, candidates: List[Resume], k1: float = 1.5, b: float = 0.75):
        self.candidates = candidates

        self.idf_index: Dict[str, float] = {}

        appearance_index: Dict[str, int] = {}

        self.avg_doc_length = 0

        self.k1 = k1
        self.b = b

        # index the IDFs of each token across all resumes
        for candidate in self.candidates:
            tokens = candidate.tokens
            for token in set(tokens):
                appearance_index[token] = appearance_index.get(token, 0) + 1

            self.avg_doc_length += len(tokens)

        self.avg_doc_length /= len(self.candidates)

        for token, appearance_count in appearance_index.items():
            self.idf_index[token] = np.log(len(self.candidates) / appearance_count)

    def rank(self, job_description: str) -> List[Tuple[Resume, float]]:
        """
        Rank candidates using BM25 scoring.

        Args:
            candidates: List of Resume objects to rank
            job_description: JobDescription object containing job requirements
        Returns:
            List of tuples containing Resume and BM25 score
        """

        scored_candidates = [
            (candidate, self.bm25_score(candidate, job_description))
            for candidate in self.candidates
        ]

        return sorted(scored_candidates, key=lambda x: x[1], reverse=True)

    def bm25_score(self, candidate: Resume, job_description: str) -> float:
        score = 0
        for token in set(job_description.split()):
            score += (
                self.idf_index.get(token, 0)
                * (candidate.term_frequencies.get(token, 0) * (self.k1 + 1))
                / (
                    candidate.term_frequencies.get(token, 0)
                    + self.k1
                    * (
                        1
                        - self.b
                        + self.b * len(candidate.tokens) / self.avg_doc_length
                    )
                )
            )

        return score
