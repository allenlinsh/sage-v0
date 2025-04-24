import json
from classes import Resume
import litellm
import re
from typing import Dict, Any, List, Optional, Union, Tuple
from dotenv import load_dotenv

load_dotenv()

class Evaluator:
    """
    Evaluator that uses a more advanced LLM (e.g., GPT-o3) to evaluate 
    the quality of candidate rankings and provide insights.
    """
    
    def __init__(self, model_name: str = "gpt-4o"):
        """
        Initialize the evaluator with the specified model.
        
        Args:
            model_name: Name of the LLM model to use (default: gpt-4o)
        """
        self.model_name = model_name
        
        # Initialize litellm
        litellm.set_verbose = False
        
        # Track revisions needed for metrics
        self.revision_count = 0
        self.max_revisions = 3
    
    def evaluate(self, 
                candidates: List[Resume], 
                job_description: str,
                ground_truth: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Evaluate the quality of candidate rankings
        
        Args:
            candidates: List of ranked candidates
            job_description: Job description text
            ground_truth: Optional ground truth rankings for evaluation
            
        Returns:
            Dictionary with evaluation metrics and insights
        """
        # Reset revision count
        self.revision_count = 0
        
        # Initialize evaluation results
        evaluation_results = {
            "metrics": {},
            "insights": [],
            "recommended_improvements": [],
            "top_candidates_analysis": []
        }
        
        try:
            # Generate evaluation rubric
            rubric = self._generate_evaluation_rubric(job_description)
            
            # 1. Evaluate ranking quality
            ranking_quality = self._evaluate_ranking_quality(candidates, job_description, rubric)
            evaluation_results["ranking_quality"] = ranking_quality
            
            # 2. Calculate precision at K
            if ground_truth:
                precision_at_k = self._calculate_precision_at_k(candidates, ground_truth)
                evaluation_results["metrics"].update(precision_at_k)
            
            # 3. Analyze top candidates
            top_candidates_analysis = self._analyze_top_candidates(candidates[:10], job_description)
            evaluation_results["top_candidates_analysis"] = top_candidates_analysis
            
            # 4. Generate improvement recommendations
            recommendations = self._generate_recommendations(candidates, job_description, ranking_quality)
            evaluation_results["recommended_improvements"] = recommendations
            
            # Add metadata
            evaluation_results["metadata"] = {
                "model_used": self.model_name,
                "revision_count": self.revision_count,
                "candidates_evaluated": len(candidates),
                "job_description_length": len(job_description)
            }
            
            return evaluation_results
            
        except Exception as e:
            # Return error information
            return {
                "error": str(e),
                "metrics": {},
                "insights": ["Evaluation failed due to error"],
                "recommended_improvements": ["Fix evaluation system errors"],
                "metadata": {
                    "model_used": self.model_name,
                    "revision_count": self.revision_count,
                    "error_occurred": True
                }
            }
    
    def _generate_evaluation_rubric(self, job_description: str) -> Dict[str, Any]:
        """
        Create a comprehensive evaluation rubric for the given job
        """
        # This is the agent's first revision/iteration
        self.revision_count = 1
        
        prompt = f"""
        You are an expert hiring manager with deep experience in technical recruitment.
        Create a comprehensive evaluation rubric for assessing candidates for the following job:
        
        ---
        {job_description}
        ---
        
        The rubric should:
        1. Identify 4-6 key competencies required for this role
        2. For each competency, explain why it's important
        3. For each competency, define clear criteria for different levels of proficiency (Excellent, Good, Average, Below Average)
        4. Define clear scoring criteria for overall candidate evaluation
        
        Make this rubric extremely specific to the job description, not generic.
        Your rubric will be used to evaluate the quality of our candidate ranking system.
        
        Return the rubric as a structured JSON object.
        """
        
        try:
            response = litellm.completion(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=2000
            )
            
            content = response.choices[0].message.content
            
            # Try to extract JSON from the response
            try:
                json_content = self._extract_json(content)
                rubric = json.loads(json_content)
                return rubric
            except:
                # If JSON extraction fails, use the raw content
                self.revision_count += 1  # Count as a revision
                
                # Try again with a more structured prompt
                return self._retry_generate_rubric(job_description)
                
        except Exception as e:
            print(f"Error generating rubric: {str(e)}")
            # Return a simple default rubric
            return {
                "competencies": [
                    {
                        "name": "Technical Skills",
                        "importance": "Core technical abilities needed for the role",
                        "levels": {
                            "Excellent": "Exceptional mastery of all required technical skills",
                            "Good": "Strong in most required technical skills",
                            "Average": "Sufficient technical skills but room for improvement",
                            "Below Average": "Lacking in several key technical areas"
                        }
                    },
                    {
                        "name": "Experience",
                        "importance": "Practical application of skills in relevant contexts",
                        "levels": {
                            "Excellent": "Extensive experience in directly relevant roles",
                            "Good": "Solid experience in similar roles",
                            "Average": "Some relevant experience but limited",
                            "Below Average": "Minimal relevant experience"
                        }
                    }
                ],
                "overall_scoring": {
                    "90-100": "Exceptional candidate, perfect fit",
                    "70-89": "Strong candidate, good fit",
                    "50-69": "Potential candidate with some gaps",
                    "Below 50": "Not recommended for this role"
                }
            }
    
    def _retry_generate_rubric(self, job_description: str) -> Dict[str, Any]:
        """Try to generate rubric again with more structured prompt"""
        prompt = f"""
        Create an evaluation rubric for this job:
        
        {job_description}
        
        Format the rubric EXACTLY as this JSON structure:
        {{
          "competencies": [
            {{
              "name": "Technical Skills",
              "importance": "Explanation of why this matters",
              "levels": {{
                "Excellent": "Description of excellent level",
                "Good": "Description of good level",
                "Average": "Description of average level",
                "Below Average": "Description of below average level"
              }}
            }},
            // Add 3-5 more competencies specific to this job
          ],
          "overall_scoring": {{
            "90-100": "Exceptional candidate, perfect fit",
            "70-89": "Strong candidate, good fit",
            "50-69": "Potential candidate with some gaps",
            "Below 50": "Not recommended for this role"
          }}
        }}
        
        Ensure your response is ONLY the JSON with no additional text.
        """
        
        try:
            response = litellm.completion(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=2000
            )
            
            content = response.choices[0].message.content
            
            # Try to extract JSON from the response
            json_content = self._extract_json(content)
            return json.loads(json_content)
        except:
            # If JSON extraction fails again, return a simple default rubric
            return {
                "competencies": [
                    {
                        "name": "Job Fit",
                        "importance": "Overall alignment with job requirements",
                        "levels": {
                            "Excellent": "Perfect match for the job",
                            "Good": "Good match with minor gaps",
                            "Average": "Acceptable match with some gaps",
                            "Below Average": "Poor match with major gaps"
                        }
                    }
                ],
                "overall_scoring": {
                    "90-100": "Excellent fit",
                    "70-89": "Good fit",
                    "50-69": "Potential fit",
                    "Below 50": "Not recommended"
                }
            }
    
    def _evaluate_ranking_quality(self, candidates: List[Dict[str, Any]], job_description: str, rubric: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate the quality of candidate rankings using the LLM agent
        """
        # Prepare data for evaluation
        # Take only top 10 for quality evaluation
        top_candidates = candidates[:min(10, len(candidates))]
        
        # Prepare candidate summaries
        candidate_summaries = []
        for i, candidate in enumerate(top_candidates):
            summary = f"Candidate {i+1} (Rank {candidate.get('rank', i+1)}):\n"
            summary += f"- Skills: {', '.join(candidate.get('skills', []))}\n"
            summary += f"- Education: {candidate.get('education', 'Not specified')}\n"
            summary += f"- Experience: {candidate.get('experience_years', 0)} years\n"
            summary += f"- Location: {candidate.get('location', 'Not specified')}\n"
            summary += f"- Final Score: {candidate.get('final_score', 0):.2f}\n"
            
            # Add LLM scores if available
            if "llm_score" in candidate:
                summary += f"- LLM Score: {candidate.get('llm_score', 0)}\n"
                summary += f"- LLM Assessment: {candidate.get('llm_assessment', '')}\n"
            
            candidate_summaries.append(summary)
        
        # Join all candidate summaries
        all_candidates = "\n".join(candidate_summaries)
        
        # Prepare rubric as string
        rubric_str = json.dumps(rubric, indent=2)
        
        prompt = f"""
        You are an expert evaluator assessing a candidate ranking system for job applicants.
        
        ## Job Description:
        {job_description}
        
        ## Evaluation Rubric:
        {rubric_str}
        
        ## Top Ranked Candidates:
        {all_candidates}
        
        Please analyze the quality of these rankings using the evaluation rubric:
        
        1. Alignment Score (0-100): How well do the rankings align with the job requirements?
        2. Ranking Effectiveness (0-100): How effective is the ranking in placing the best candidates at the top?
        3. Diversity of Skills: Are the top candidates showing diversity in relevant skills or are they too similar?
        4. Depth Analysis: Analyze the depth of experience and qualifications in the top candidates.
        5. Critical Issues: Identify any critical issues with the current ranking system.
        
        For each of these aspects, provide:
        - A numerical score where applicable
        - A 2-3 sentence justification
        - A specific example from the candidate data
        
        Return your evaluation as structured JSON.
        """
        
        try:
            response = litellm.completion(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=2000
            )
            
            content = response.choices[0].message.content
            
            # Try to extract JSON from the response
            try:
                json_content = self._extract_json(content)
                evaluation = json.loads(json_content)
                return evaluation
            except:
                # If JSON extraction fails, try to create a structured response from the text
                self.revision_count += 1  # Count as a revision
                
                # Extract scores using regex
                alignment_score = self._extract_score(content, r'alignment\s*score\s*(?:\(0-100\))?\s*:?\s*(\d+)')
                effectiveness_score = self._extract_score(content, r'ranking\s*effectiveness\s*(?:\(0-100\))?\s*:?\s*(\d+)')
                
                # Extract analysis sections
                diversity_analysis = self._extract_section(content, r'diversity of skills:?\s*(.*?)(?:\n\n|\n[A-Z])')
                depth_analysis = self._extract_section(content, r'depth analysis:?\s*(.*?)(?:\n\n|\n[A-Z])')
                critical_issues = self._extract_section(content, r'critical issues:?\s*(.*?)(?:\n\n|$)')
                
                # Create structured evaluation
                return {
                    "alignment_score": alignment_score,
                    "effectiveness_score": effectiveness_score,
                    "diversity_analysis": diversity_analysis,
                    "depth_analysis": depth_analysis,
                    "critical_issues": critical_issues,
                    "raw_evaluation": content[:500] + "..." if len(content) > 500 else content
                }
                
        except Exception as e:
            print(f"Error during ranking quality evaluation: {str(e)}")
            # Return a basic evaluation
            return {
                "alignment_score": 50,
                "effectiveness_score": 50,
                "diversity_analysis": "Analysis failed due to error",
                "depth_analysis": "Analysis failed due to error",
                "critical_issues": f"Evaluation system error: {str(e)}",
                "error": str(e)
            }
    
    def _calculate_precision_at_k(self, ranked_candidates: List[Dict[str, Any]], ground_truth: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Calculate precision at different K values (1, 5, 10)
        
        Args:
            ranked_candidates: List of ranked candidates
            ground_truth: List of candidates that are considered relevant/good
            
        Returns:
            Dictionary with P@1, P@5, P@10 metrics
        """
        # Extract IDs or filenames from ground truth for comparison
        ground_truth_ids = set(item.get('filename', '') for item in ground_truth)
        
        # Calculate precision at different K values
        metrics = {}
        
        # P@1
        if len(ranked_candidates) >= 1:
            relevant_at_1 = 1 if ranked_candidates[0].get('filename', '') in ground_truth_ids else 0
            metrics["precision_at_1"] = relevant_at_1
        
        # P@5
        if len(ranked_candidates) >= 5:
            relevant_at_5 = sum(1 for c in ranked_candidates[:5] if c.get('filename', '') in ground_truth_ids)
            metrics["precision_at_5"] = relevant_at_5 / 5
        
        # P@10
        if len(ranked_candidates) >= 10:
            relevant_at_10 = sum(1 for c in ranked_candidates[:10] if c.get('filename', '') in ground_truth_ids)
            metrics["precision_at_10"] = relevant_at_10 / 10
        
        return metrics
    
    def _analyze_top_candidates(self, top_candidates: List[Dict[str, Any]], job_description: str) -> List[Dict[str, Any]]:
        """
        Analyze the strengths and weaknesses of top candidates
        """
        analysis_results = []
        
        for candidate in top_candidates[:5]:  # Analyze only top 5 for efficiency
            try:
                analysis = self._analyze_candidate_fit(candidate, job_description)
                analysis_results.append({
                    "filename": candidate.get("filename", "Unknown"),
                    "rank": candidate.get("rank", 0),
                    "fit_score": analysis.get("fit_score", 0),
                    "strengths": analysis.get("strengths", []),
                    "weaknesses": analysis.get("weaknesses", []),
                    "recommendation": analysis.get("recommendation", "")
                })
            except Exception as e:
                # Add error information for this candidate
                analysis_results.append({
                    "filename": candidate.get("filename", "Unknown"),
                    "rank": candidate.get("rank", 0),
                    "error": str(e)
                })
        
        return analysis_results
    
    def _analyze_candidate_fit(self, candidate: Dict[str, Any], job_description: str) -> Dict[str, Any]:
        """
        Analyze how well a candidate fits the job description
        """
        # Prepare candidate summary
        skills = ", ".join(candidate.get("skills", []))
        education = candidate.get("education", "Not specified")
        experience_years = candidate.get("experience_years", 0)
        location = candidate.get("location", "Not specified")
        summary = candidate.get("summary", "")
        
        prompt = f"""
        Analyze how well this candidate fits the job description:
        
        ## Job Description:
        {job_description}
        
        ## Candidate Information:
        - Skills: {skills}
        - Education: {education}
        - Experience: {experience_years} years
        - Location: {location}
        - Summary: {summary}
        
        Provide the following in your analysis:
        1. Fit Score (0-100): How well the candidate fits the role
        2. Strengths: Three key strengths of this candidate for this role
        3. Weaknesses: Three potential weaknesses or gaps
        4. Recommendation: Whether to interview, consider, or reject
        
        Return the analysis as JSON.
        """
        
        try:
            response = litellm.completion(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=1000
            )
            
            content = response.choices[0].message.content
            
            # Try to extract JSON from the response
            try:
                json_content = self._extract_json(content)
                analysis = json.loads(json_content)
                return analysis
            except:
                # If JSON extraction fails, extract information using regex
                fit_score = self._extract_score(content, r'fit\s*score\s*(?:\(0-100\))?\s*:?\s*(\d+)')
                
                # Extract lists with regex
                strengths = re.findall(r'(?:Strengths|STRENGTHS):\s*(?:\d\.\s*)?([^\n\d]+)(?:\n|$)', content)
                weaknesses = re.findall(r'(?:Weaknesses|WEAKNESSES):\s*(?:\d\.\s*)?([^\n\d]+)(?:\n|$)', content)
                
                # Extract recommendation
                recommendation_match = re.search(r'(?:Recommendation|RECOMMENDATION):\s*([^\n]+)', content)
                recommendation = recommendation_match.group(1).strip() if recommendation_match else "No recommendation provided"
                
                return {
                    "fit_score": fit_score,
                    "strengths": strengths[:3],  # Take at most 3
                    "weaknesses": weaknesses[:3],  # Take at most 3
                    "recommendation": recommendation
                }
                
        except Exception as e:
            print(f"Error analyzing candidate fit: {str(e)}")
            # Return basic analysis
            return {
                "fit_score": 50,
                "strengths": ["Could not determine strengths"],
                "weaknesses": ["Could not determine weaknesses"],
                "recommendation": "Analysis failed due to error",
                "error": str(e)
            }
    
    def _generate_recommendations(self, candidates: List[Dict[str, Any]], job_description: str, ranking_quality: Dict[str, Any]) -> List[str]:
        """
        Generate recommendations for improving the ranking system
        """
        # Use ranking quality results to inform recommendations
        alignment_score = ranking_quality.get("alignment_score", 50)
        effectiveness_score = ranking_quality.get("effectiveness_score", 50)
        critical_issues = ranking_quality.get("critical_issues", "No critical issues identified")
        
        # Prepare summary of ranking performance
        ranking_summary = f"""
        Alignment Score: {alignment_score}
        Effectiveness Score: {effectiveness_score}
        Critical Issues: {critical_issues}
        Number of Candidates: {len(candidates)}
        """
        
        prompt = f"""
        As an expert in recruitment and candidate evaluation, provide specific recommendations 
        to improve our resume screening and candidate ranking system.
        
        ## Job Description:
        {job_description}
        
        ## Current Ranking Performance:
        {ranking_summary}
        
        Provide 3-5 specific, actionable recommendations to improve:
        1. The quality of our candidate rankings
        2. The assessment of candidate fit
        3. The overall effectiveness of our hiring pipeline
        
        Each recommendation should be concrete and implementable.
        Return the recommendations as a JSON array of strings.
        """
        
        try:
            response = litellm.completion(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.4,
                max_tokens=1500
            )
            
            content = response.choices[0].message.content
            
            # Try to extract JSON array from the response
            try:
                json_content = self._extract_json(content)
                recommendations = json.loads(json_content)
                
                # Ensure it's a list of strings
                if isinstance(recommendations, list):
                    return [str(r) for r in recommendations]
                else:
                    # If it's not a list, try to extract recommendations as list items
                    return self._extract_list_items(content)
                    
            except:
                # Extract recommendations as list items
                return self._extract_list_items(content)
                
        except Exception as e:
            print(f"Error generating recommendations: {str(e)}")
            # Return basic recommendations
            return [
                "Ensure the job description clearly defines required skills and experience",
                "Consider adding more weight to technical skills matching in ranking algorithm",
                "Implement a feedback loop to improve ranking based on hiring outcomes"
            ]
    
    def _extract_json(self, text: str) -> str:
        """Extract JSON content from text"""
        # Look for JSON content between triple backticks
        json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text)
        if json_match:
            return json_match.group(1)
        
        # Look for content that appears to be JSON (starting with { and ending with })
        json_match = re.search(r'(\{[\s\S]*\}|\[[\s\S]*\])', text)
        if json_match:
            return json_match.group(1)
        
        # If no JSON-like content found, return the original text
        return text
    
    def _extract_score(self, text: str, pattern: str) -> int:
        """Extract a numeric score from text using regex"""
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            try:
                return int(match.group(1))
            except (ValueError, IndexError):
                pass
        return 50  # Default score
    
    def _extract_section(self, text: str, pattern: str) -> str:
        """Extract a section of text using regex"""
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            try:
                return match.group(1).strip()
            except (IndexError):
                pass
        return "No information available"
    
    def _extract_list_items(self, text: str) -> List[str]:
        """Extract numbered or bulleted list items from text"""
        # Look for numbered items
        items = re.findall(r'(?:^|\n)\s*\d+\.\s*(.*?)(?:\n|$)', text)
        
        # If no numbered items, look for bulleted items
        if not items:
            items = re.findall(r'(?:^|\n)\s*[-*•]\s*(.*?)(?:\n|$)', text)
        
        # If still no items found, split by newlines and filter non-empty lines
        if not items:
            items = [line.strip() for line in text.split('\n') if line.strip()]
        
        # Return up to 5 items, making sure they're not too long
        return [item[:150] + '...' if len(item) > 150 else item for item in items[:5]] 