from typing import List, Tuple, Dict

import numpy as np

from classes import Resume
from text_cleaner import clean


class BM25Ranker:
    def __init__(
        self,
        candidates: List[Resume],
        k1: float = 1.5,
        k2: float = 1.69,
        b: float = 0.75,
    ):
        self.candidates = candidates
        self.k1 = k1
        self.k2 = k2
        self.b = b

        self.idf_index: Dict[str, float] = {}
        self.avg_doc_length = 0

        # index the IDFs of each token across all resumes
        appearance_index: Dict[str, int] = (
            {}
        )  # the number of times token appears across documents
        for candidate in self.candidates:
            tokens = candidate.tokens
            for token in set(tokens):
                appearance_index[token] = appearance_index.get(token, 0) + 1

            self.avg_doc_length += len(tokens)
        self.avg_doc_length /= len(self.candidates)

        # normalized idfs with log((N - n_i + 0.5) / (n_i + 0.5))
        N = len(self.candidates)
        for token, n_i in appearance_index.items():
            self.idf_index[token] = np.log((N - n_i + 0.5) / (n_i + 0.5))

        # index doc lenth normalization factors (the K in bm25)
        # K = k1 * (1 - b + b * |d|/avg_doc_length)
        self.K = {
            c.resume_id: k1 * ((1 - b) + b * (len(c.tokens) / self.avg_doc_length))
            for c in candidates
        }

    def rank(self, job_description: str) -> List[Tuple[Resume, float]]:
        """
        Rank candidates using BM25 scoring.

        Args:
            job_description: Job description text used for query
        Returns:
            List of tuples containing Resume and BM25 score
        """
        processed_description = clean(job_description)
        return sorted(
            ((c, self.bm25_score(c, processed_description)) for c in self.candidates),
            key=lambda x: x[1],
            reverse=True,
        )

    def bm25_score(self, candidate: Resume, job_description_tokens: List[str]) -> float:
        score = 0.0

        for token in set(job_description_tokens):
            qf = job_description_tokens.count(token) if self.k2 > 0 else 1

            f = candidate.term_frequencies.get(token, 0)

            if f == 0:
                continue

            idf = self.idf_index.get(token, 0.0)
            K = self.K[candidate.resume_id]

            term_score = idf * (f * (self.k1 + 1)) / (f + K)

            if self.k2 > 0:
                term_score *= ((self.k2 + 1) * qf) / (self.k2 + qf)

            score += term_score

        return score
