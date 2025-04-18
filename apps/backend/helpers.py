import os
import json
import pandas as pd
from typing import Dict, Any, List, Optional, Union, Tuple
import tempfile
import shutil
import re
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def ensure_dir_exists(directory: str) -> None:
    """Ensure a directory exists, create it if it doesn't"""
    if not os.path.exists(directory):
        os.makedirs(directory)

def save_to_json(data: Any, file_path: str) -> str:
    """Save data to a JSON file"""
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)
    return file_path

def load_from_json(file_path: str) -> Any:
    """Load data from a JSON file"""
    with open(file_path, 'r') as f:
        return json.load(f)

def csv_to_json(csv_file: str) -> List[Dict[str, Any]]:
    """Convert CSV file to JSON"""
    df = pd.read_csv(csv_file)
    return df.to_dict(orient='records')

def save_uploaded_file(uploaded_file: bytes, filename: str, directory: str) -> str:
    """
    Save an uploaded file to the specified directory
    
    Args:
        uploaded_file: File content as bytes
        filename: Original filename
        directory: Directory to save the file to
        
    Returns:
        Path to the saved file
    """
    ensure_dir_exists(directory)
    file_path = os.path.join(directory, filename)
    
    with open(file_path, 'wb') as f:
        f.write(uploaded_file)
    
    return file_path

def clean_text(text: str) -> str:
    """Clean text by removing extra whitespace, special chars, etc."""
    # Remove multiple whitespaces
    text = re.sub(r'\s+', ' ', text)
    # Remove special characters that are not useful
    text = re.sub(r'[^\w\s.,;:?!@#$%^&*()-]', '', text)
    return text.strip()

def extract_keywords(text: str, min_length: int = 3) -> List[str]:
    """Extract keywords from text"""
    # Simple implementation - in production would use NLP libraries
    words = re.findall(r'\b[a-zA-Z]{' + str(min_length) + r',}\b', text.lower())
    return list(set(words))  # Remove duplicates

def get_temp_directory() -> str:
    """Get a temporary directory for file processing"""
    temp_dir = tempfile.mkdtemp()
    return temp_dir

def cleanup_temp_directory(directory: str) -> None:
    """Clean up a temporary directory"""
    if os.path.exists(directory):
        shutil.rmtree(directory)

def calculate_similarity(text1: str, text2: str) -> float:
    """
    Calculate simple similarity between two texts
    In production, we would use more sophisticated methods
    """
    # Extract keywords
    keywords1 = set(extract_keywords(text1))
    keywords2 = set(extract_keywords(text2))
    
    # Calculate Jaccard similarity
    if not keywords1 or not keywords2:
        return 0.0
    
    intersection = keywords1.intersection(keywords2)
    union = keywords1.union(keywords2)
    
    return len(intersection) / len(union)

def format_api_response(data: Any, success: bool = True, message: str = "") -> Dict[str, Any]:
    """Format a consistent API response"""
    return {
        "success": success,
        "message": message,
        "data": data
    }

def get_env_var(name: str, default: str = "") -> str:
    """Get an environment variable with a default value"""
    return os.environ.get(name, default) 