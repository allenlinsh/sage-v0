import json
from classes import Resume
import litellm
import re
from typing import Dict, Any, List, Tuple
import time
from dotenv import load_dotenv

load_dotenv()


class LLMReranker:
    def __init__(self, ranked_candidates: List[Tuple[Resume, float]], model_name: str = "gpt-4.1-mini"):
        self.ranked_candidates = ranked_candidates
        self.model_name = model_name

        litellm.set_verbose = False

    def rerank(
        self, job_description: str, top_k: int = 10
    ) -> List[Tuple[Resume, float]]:
        top_candidates = self.ranked_candidates[: min(top_k, len(self.ranked_candidates))]

        if not top_candidates:
            return []

        reranked_candidates = self.batch_rerank(top_candidates, job_description)

        return sorted(reranked_candidates, key=lambda x: x.llm_score, reverse=True)

    def batch_rerank(
        self, top_candidates: List[Tuple[Resume, float]], job_description: str
    ) -> List[Tuple[Resume, float]]:
        reranked_candidates = []

        rubric = self.generate_evaluation_rubric(job_description)

        for candidate, score in top_candidates:
            time.sleep(0.1)

            evaluated_candidate, evaluated_score = self.evaluate_candidate(
                candidate, score, job_description, rubric
            )
            
            reranked_candidates.append((evaluated_candidate, evaluated_score))

        return reranked_candidates

    def generate_evaluation_rubric(self, job_description: str) -> Dict[str, Any]:
        """Generate an evaluation rubric based on the job description"""
        prompt = f"""
        You are an expert hiring manager assistant. Based on the following job description, 
        create a structured evaluation rubric that can be used to score candidates.
        The rubric should include 3-5 key criteria that are most important for this role.
        
        Job Description:
        {job_description}
        
        For each criterion, specify:
        1. The name of the criterion
        2. Why it's important for this role
        3. What would constitute a high score (5/5)
        4. What would constitute a medium score (3/5)
        5. What would constitute a low score (1/5)
        
        Return the rubric as a structured JSON object with each criterion and its details.
        """

        try:
            response = litellm.completion(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=1000,
            )

            content = response.choices[0].message.content

            # Try to extract JSON from the response
            try:
                # Look for JSON in the response
                json_content = self._extract_json(content)
                return json.loads(json_content)
            except:
                # If JSON extraction fails, return a simple structured rubric
                return {
                    "criteria": [
                        {
                            "name": "Technical Skills Match",
                            "importance": "Technical skills directly impact performance in the role",
                            "high_score": "Possesses all key technical skills mentioned in job description",
                            "medium_score": "Has most technical skills but missing some key requirements",
                            "low_score": "Missing many key technical skills required for the role",
                        },
                        {
                            "name": "Experience Level",
                            "importance": "Right experience level ensures candidate can handle job complexity",
                            "high_score": "Experience level aligns perfectly with job requirements",
                            "medium_score": "Has experience but slightly under/over qualified",
                            "low_score": "Significantly under or overqualified for the position",
                        },
                        {
                            "name": "Education and Certifications",
                            "importance": "Demonstrates formal knowledge and training in relevant areas",
                            "high_score": "Education/certifications exceed requirements",
                            "medium_score": "Has minimum required education/certifications",
                            "low_score": "Missing required education/certifications",
                        },
                    ]
                }

        except Exception as e:
            print(f"Error generating rubric: {str(e)}")
            # Return a default rubric on error
            return {
                "criteria": [
                    {
                        "name": "Overall Job Fit",
                        "importance": "Measures how well the candidate matches the job requirements",
                        "high_score": "Excellent match for key requirements",
                        "medium_score": "Satisfactory match for requirements",
                        "low_score": "Poor match for requirements",
                    }
                ]
            }

    def evaluate_candidate(
        self, candidate: Resume, score: float, job_description: str, rubric: Dict[str, Any]
    ) -> Tuple[Resume, float]:
        candidate_summary = self.prepare_candidate_summary(candidate)

        # Prepare the evaluation prompt with the rubric
        rubric_text = json.dumps(rubric, indent=2)

        prompt = f"""
        You are an expert hiring manager assistant. Evaluate the following candidate for a job using the provided rubric.
        
        ## Job Description:
        {job_description}
        
        ## Candidate Information:
        {candidate_summary}
        
        ## Evaluation Rubric:
        {rubric_text}
        
        For each criterion in the rubric, score the candidate on a scale of 1-5 and provide specific justification based on the candidate's profile.
        
        Then provide:
        1. An overall score on a scale of 0-100
        2. A brief explanation (2-3 sentences) of your overall assessment
        3. Key strengths (up to 3)
        4. Key weaknesses (up to 3)
        
        Return your evaluation in a structured JSON format with the following fields:
        - criterion_scores: Object with criterion names as keys and scores (1-5) as values
        - justifications: Object with criterion names as keys and justification texts as values
        - overall_score: Numeric score between 0-100
        - overall_assessment: Text explanation of overall assessment
        - strengths: Array of strengths
        - weaknesses: Array of weaknesses
        """

        try:
            response = litellm.completion(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=1500,
            )

            content = response.choices[0].message.content

            # Extract JSON from the response
            try:
                evaluation_json = self._extract_json(content)
                evaluation = json.loads(evaluation_json)

                # Update the candidate with the LLM evaluation
                result = candidate.copy()
                result["llm_score"] = evaluation.get("overall_score", 0)
                result["llm_assessment"] = evaluation.get("overall_assessment", "")
                result["llm_strengths"] = evaluation.get("strengths", [])
                result["llm_weaknesses"] = evaluation.get("weaknesses", [])
                result["llm_criterion_scores"] = evaluation.get("criterion_scores", {})
                result["llm_justifications"] = evaluation.get("justifications", {})

                return result
            except Exception as json_error:
                print(f"Error parsing LLM response as JSON: {str(json_error)}")
                # Try to extract the overall score from the text
                score_match = re.search(
                    r"overall\s+score\s*:?\s*(\d+)", content.lower()
                )
                overall_score = int(score_match.group(1)) if score_match else 50

                # Update with extracted information
                result = candidate.copy()
                result["llm_score"] = overall_score
                result["llm_assessment"] = (
                    "Error parsing full evaluation: " + content[:100] + "..."
                )
                result["llm_strengths"] = []
                result["llm_weaknesses"] = []

                return result

        except Exception as e:
            print(f"Error during LLM evaluation: {str(e)}")
            # Return original candidate with error noted
            result = candidate.copy()
            result["llm_score"] = candidate.get("final_score", 0) * 20  # Scale to 0-100
            result["llm_assessment"] = f"LLM evaluation failed: {str(e)}"
            result["llm_strengths"] = []
            result["llm_weaknesses"] = []

            return result

    def prepare_candidate_summary(self, candidate: Dict[str, Any]) -> str:
        """Prepare a text summary of the candidate for the LLM prompt"""
        # Extract key information
        skills = ", ".join(candidate.get("skills", []))
        education = candidate.get("education", "Not specified")
        experience_years = candidate.get("experience_years", 0)
        location = candidate.get("location", "Not specified")
        summary = candidate.get("summary", "")

        # Format as text
        candidate_summary = f"""
        Skills: {skills}
        Education: {education}
        Years of Experience: {experience_years}
        Location: {location}
        Initial Ranking Score: {candidate.get("final_score", 0):.2f}
        
        Summary:
        {summary}
        """

        return candidate_summary

    def _extract_json(self, text: str) -> str:
        """Extract JSON content from text that may have markdown or other formatting"""
        # Look for JSON content between triple backticks
        json_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
        if json_match:
            return json_match.group(1)

        # Look for content that appears to be JSON (starting with { and ending with })
        json_match = re.search(r"(\{[\s\S]*\})", text)
        if json_match:
            return json_match.group(1)

        # If no JSON-like content found, return the original text
        return text
