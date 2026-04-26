# evaluations/datasets/data_loader.py
import json
import os
from pathlib import Path

def load_golden_dataset(filename="golden_dataset.json"):
    """
    Loads the golden dataset for Pytest parameterization.
    Resolves the path relative to this script's location.
    """
    # Assuming golden_dataset.json is at the root of the project (two levels up from this file)
    current_dir = Path(__file__).resolve().parent
    file_path = current_dir.parent.parent / filename
    
    if not file_path.exists():
        raise FileNotFoundError(f"Could not find golden dataset at: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    # Extract only the fields necessary for DeepEval execution to keep parameters clean
    return [
        (
            item["id"],
            item["query"],
            item["source"],
            item["expected_snippet"],
            item["expected_keywords"],
            item["reasoning_path"]
        ) for item in data
    ]