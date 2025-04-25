import pandas as pd
import json
from typing import List

from classes import Resume, Education


def load_resumes_from_csv(
    csv_file_path: str = "anonymized_resumes.csv",
) -> List[Resume]:
    """
    Load resumes from a CSV file and convert them to Resume objects.

    Args:
        csv_file_path: Path to the CSV file containing resume data

    Returns:
        List of Resume objects
    """
    df = pd.read_csv(csv_file_path)

    resumes = []
    for _, row in df.iterrows():
        education_data = row.get("education", "[]")
        if isinstance(education_data, str):
            try:
                education_data = json.loads(education_data)
            except:
                education_data = []

        skills_data = row.get("skills", "[]")
        if isinstance(skills_data, str):
            try:
                skills_data = json.loads(skills_data)
            except:
                skills_data = []

        location = row.get("location", "")
        if isinstance(location, str):
            location = location.strip('"')

        resume = Resume(
            resume_id=row.get("resume_id", ""),
            resume_text=row.get("anonResumeText", ""),
            education=[Education.from_dict(edu) for edu in education_data],
            location=location,
            skills=skills_data,
        )
        resumes.append(resume)

    return resumes
