import json
import os
from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
from dotenv import load_dotenv
import litellm
from classes import Resume, EvaluationResult, EvaluationCriterion, EvaluationRubric

load_dotenv()

class Evaluator:
    """
    Enhanced evaluator that uses GPT models to evaluate the quality of candidate 
    rankings and calculate metrics like Mean Average Precision (MAP@K).
    """
    
    def __init__(self, model_name: str = "gpt-o3"):
        """
        Initialize the evaluator with the specified model.
        
        Args:
            model_name: Name of the LLM model to use (default: gpt-o3)
        """
        self.model_name = model_name
        
        # Configure litellm
        litellm.set_verbose = False
    
    def evaluate(self, 
                candidates: List[Tuple[Resume, float]], 
                job_description: str,
                ground_truth: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Evaluate the quality of candidate rankings and calculate metrics
        
        Args:
            candidates: List of (resume, score) tuples
            job_description: Job description text
            ground_truth: Optional list of resume_ids considered relevant
            
        Returns:
            Dictionary with evaluation metrics and insights
        """
        # Initialize evaluation results dictionary
        evaluation_results = {
            "metrics": {},
            "relevance_scores": [],
            "candidate_evaluations": []
        }
        
        try:
            # 1. Generate evaluation rubric
            rubric = self._generate_evaluation_rubric(job_description)
            evaluation_results["rubric"] = rubric.model_dump()
            
            # 2. Evaluate relevance of each candidate
            relevance_results = self._evaluate_candidate_relevance(
                [c[0] for c in candidates[:10]], job_description, rubric
            )
            
            # Store relevance scores
            evaluation_results["relevance_scores"] = [
                {"resume_id": c[0].resume_id, "relevance_score": r.score, "reasoning": r.reason}
                for c, r in zip(candidates[:10], relevance_results)
            ]
            
            # 3. Calculate MAP@10 if ground truth is provided
            if ground_truth:
                map_score = self._calculate_map_at_k(
                    [c[0].resume_id for c in candidates], 
                    ground_truth, 
                    k=10
                )
                evaluation_results["metrics"]["map_at_10"] = map_score
            
            # Add metadata
            evaluation_results["metadata"] = {
                "model_used": self.model_name,
                "candidates_evaluated": len(candidates[:10]),
                "job_description_length": len(job_description)
            }
            
            return evaluation_results
            
        except Exception as e:
            # Return error information
            return {
                "error": str(e),
                "metrics": {},
                "metadata": {
                    "model_used": self.model_name,
                    "error_occurred": True
                }
            }
    
    def _generate_evaluation_rubric(self, job_description: str) -> EvaluationRubric:
        """
        Create a comprehensive evaluation rubric for the given job description
        """
        system_prompt = f"""
        You are an expert hiring manager. Based on the following job description, create a structured evaluation rubric 
        that can be used to score candidates.
        
        The rubric should include 3-5 key criteria that are most important for this role.
        
        For each criterion, specify:
        1. The name of the criterion
        2. Why it's important for this role (importance)
        3. What would constitute a score range of 80-100 out of 100
        4. What would constitute a score range of 60-79 out of 100
        5. What would constitute a score range of 40-59 out of 100
        6. What would constitute a score range of 20-39 out of 100
        7. What would constitute a score range of 0-19 out of 100
        
        Job Description:
        {job_description}
        
        Return the rubric as a structured JSON object with a list of criteria.
        """
        
        try:
            response = litellm.completion(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": "Generate the evaluation rubric."}
                ],
                temperature=0.2,
                response_format={"type": "json_object"}
            )
            
            content = response.choices[0].message.content
            
            # Parse the JSON response
            try:
                rubric_data = json.loads(content)
                return EvaluationRubric.model_validate(rubric_data)
            except Exception as json_err:
                # If JSON validation fails, create a basic rubric
                print(f"Error parsing rubric JSON: {str(json_err)}")
                return self._create_default_rubric()
                
        except Exception as e:
            print(f"Error generating rubric: {str(e)}")
            return self._create_default_rubric()
    
    def _create_default_rubric(self) -> EvaluationRubric:
        """Create a default evaluation rubric"""
        return EvaluationRubric(
            criteria=[
                EvaluationCriterion(
                    name="Technical Skills",
                    importance="Core technical abilities needed for the role",
                    score_80_100="Exceptional mastery of all required technical skills",
                    score_60_79="Strong in most required technical skills",
                    score_40_59="Sufficient technical skills but room for improvement",
                    score_20_39="Lacking in several key technical areas",
                    score_0_19="Minimal or no relevant technical skills"
                ),
                EvaluationCriterion(
                    name="Experience",
                    importance="Practical application of skills in relevant contexts",
                    score_80_100="Extensive experience in directly relevant roles",
                    score_60_79="Solid experience in similar roles",
                    score_40_59="Some relevant experience but limited",
                    score_20_39="Minimal relevant experience",
                    score_0_19="No relevant experience"
                ),
                EvaluationCriterion(
                    name="Overall Fit",
                    importance="Alignment with job requirements",
                    score_80_100="Perfect match for the role",
                    score_60_79="Good match with minor gaps",
                    score_40_59="Acceptable match with some gaps",
                    score_20_39="Poor match with major gaps",
                    score_0_19="Not a match for this role"
                )
            ]
        )
    
    def _evaluate_candidate_relevance(
        self, 
        candidates: List[Resume], 
        job_description: str,
        rubric: EvaluationRubric
    ) -> List[EvaluationResult]:
        """
        Evaluate the relevance of each candidate to the job description
        """
        rubric_text = self._format_rubric_for_prompt(rubric)
        
        try:
            # Create batch of messages for evaluation
            message_batches = []
            for candidate in candidates:
                # Prepare education and skills text
                education_text = "\n".join([str(edu) for edu in candidate.education])
                skills_text = ", ".join(candidate.skills)
                
                # Create message batch for this candidate
                messages = [
                    {
                        "role": "system",
                        "content": f"""
                        You are an expert hiring manager evaluating a candidate for a job.
                        
                        ## Job Description:
                        {job_description}
                        
                        ## Evaluation Rubric:
                        {rubric_text}
                        
                        ## Candidate Information:
                        Resume Text: {candidate.resume_text[:2000]}
                        
                        Education: {education_text}
                        
                        Skills: {skills_text}
                        
                        Location: {candidate.location}
                        
                        Evaluate this candidate based on the rubric. Provide a single numerical score between 0 and 100 
                        that represents how relevant the candidate is to the job description.
                        
                        Format your response as a JSON object with 'score' (float) and 'reason' (string explaining the score).
                        """
                    },
                    {
                        "role": "user",
                        "content": "Evaluate this candidate's relevance to the job description."
                    }
                ]
                message_batches.append(messages)
            
            # Get batch of evaluations
            responses = litellm.batch_completion(
                model=self.model_name,
                messages=message_batches,
                temperature=0.2,
                response_format={"type": "json_object"}
            )
            
            # Process responses
            evaluation_results = []
            for response in responses:
                try:
                    content = response.choices[0].message.content
                    result_data = json.loads(content)
                    evaluation_results.append(
                        EvaluationResult(
                            score=float(result_data.get("score", 50)),
                            reason=result_data.get("reason", "No reason provided")
                        )
                    )
                except Exception as e:
                    # If parsing fails for a response, add a default result
                    print(f"Error parsing evaluation result: {str(e)}")
                    evaluation_results.append(
                        EvaluationResult(
                            score=50.0,
                            reason=f"Error evaluating candidate: {str(e)}"
                        )
                    )
            
            return evaluation_results
                
        except Exception as e:
            print(f"Error in batch evaluation: {str(e)}")
            # Return default results
            return [
                EvaluationResult(
                    score=50.0,
                    reason=f"Evaluation failed: {str(e)}"
                )
                for _ in candidates
            ]
    
    def _format_rubric_for_prompt(self, rubric: EvaluationRubric) -> str:
        """Format the rubric for inclusion in a prompt"""
        formatted_text = "EVALUATION CRITERIA:\n\n"
        
        for criterion in rubric.criteria:
            formatted_text += f"Criterion: {criterion.name} (Importance: {criterion.importance})\n"
            formatted_text += f"- Excellent (80-100): {criterion.score_80_100}\n"
            formatted_text += f"- Good (60-79): {criterion.score_60_79}\n"
            formatted_text += f"- Average (40-59): {criterion.score_40_59}\n"
            formatted_text += f"- Below Average (20-39): {criterion.score_20_39}\n"
            formatted_text += f"- Poor (0-19): {criterion.score_0_19}\n\n"
        
        return formatted_text
    
    def _calculate_map_at_k(
        self, 
        candidate_ids: List[str], 
        relevant_ids: List[str], 
        k: int = 10
    ) -> float:
        """
        Calculate Mean Average Precision at K
        
        Args:
            candidate_ids: Ordered list of candidate IDs
            relevant_ids: List of relevant candidate IDs (ground truth)
            k: Number of results to consider
            
        Returns:
            MAP@K score
        """
        if not relevant_ids or not candidate_ids:
            return 0.0
        
        # Convert relevant_ids to set for faster lookups
        relevant_set = set(relevant_ids)
        
        # Calculate precision at each relevant result position
        precisions = []
        num_relevant_found = 0
        
        for i, candidate_id in enumerate(candidate_ids[:k]):
            if candidate_id in relevant_set:
                num_relevant_found += 1
                precision_at_i = num_relevant_found / (i + 1)
                precisions.append(precision_at_i)
        
        # Return MAP (average of precisions)
        if not precisions:
            return 0.0
        return sum(precisions) / len(relevant_ids)