import os
import json
from typing import Any
from dotenv import load_dotenv

load_dotenv()


def ensure_dir_exists(directory: str) -> None:
    if not os.path.exists(directory):
        os.makedirs(directory)


def save_to_json(data: Any, file_path: str) -> str:
    with open(file_path, "w") as f:
        json.dump(data, f, indent=2)
    return file_path


def get_env_var(name: str, default: str = "") -> str:
    return os.environ.get(name, default)
