from typing import List, Dict, Any
from pydantic import BaseModel
from typing import List

from text_cleaner import clean


class Education:
    """
    Represents an education entry in a resume.
    Based on the schema: school, degree, year
    """

    def __init__(self, school: str, degree: str, year: str = ""):
        self.school = school
        self.degree = degree
        self.year = year

    @classmethod
    def from_dict(cls, data: Dict[str, str]) -> "Education":
        """Create an Education instance from a dictionary."""
        return cls(
            school=data.get("school", ""),
            degree=data.get("degree", ""),
            year=data.get("year", ""),
        )

    def __str__(self) -> str:
        """Return a string representation of the education entry."""
        if self.year:
            return f"{self.school} - {self.degree} ({self.year})"
        else:
            return f"{self.school} - {self.degree}"


class Resume:
    """
    Represents a parsed resume with all relevant candidate information.
    Based on the CSV schema: resume_id, resume_text, education, location, skills
    """

    def __init__(
        self,
        resume_id: str,
        resume_text: str,
        education: List[Education] = None,
        location: str = "",
        skills: List[str] = None,
    ):
        self.resume_id = resume_id
        self.resume_text = resume_text
        self.education = education or []
        self.location = location
        self.skills = skills or []

        self.tokens = clean(self.resume_text)

        self.term_frequencies: Dict[str, int] = {}

        for token in self.tokens:
            self.term_frequencies[token] = self.term_frequencies.get(token, 0) + 1

    @classmethod
    def from_csv_row(cls, row: Dict[str, Any]) -> "Resume":
        """
        Create a Resume instance from a row in the CSV.

        Args:
            row: Dictionary representing a row from the CSV

        Returns:
            Resume instance
        """
        # Handle different types of education data (string or list of dicts)
        education_data = row.get("education", [])
        if isinstance(education_data, str):
            try:
                import json

                education_data = json.loads(education_data)
            except:
                education_data = []

        # Convert education dictionaries to Education objects
        education_objects = [Education.from_dict(edu) for edu in education_data]

        # Handle skills data
        skills_data = row.get("skills", [])
        if isinstance(skills_data, str):
            try:
                import json

                skills_data = json.loads(skills_data)
            except:
                skills_data = []

        return cls(
            resume_id=row.get("resume_id", ""),
            resume_text=row.get("anonResumeText", ""),
            education=education_objects,
            location=row.get("location", ""),
            skills=skills_data,
        )


class EvaluationCriterion(BaseModel):
    name: str
    importance: str
    score_80_100: str
    score_60_79: str
    score_40_59: str
    score_20_39: str
    score_0_19: str


class EvaluationRubric(BaseModel):
    criteria: List[EvaluationCriterion]


class EvaluationResult(BaseModel):
    reason: str
    score: float
