# evaluations/datasets/data_loader.py
import json
import os
from pathlib import Path
from evaluations.eval_config import eval_settings


def load_golden_dataset(filename="golden_dataset_v2.json"):
    """
    Loads the golden dataset for Pytest parameterization.
    Resolves the path relative to this script's location.
    """
    # The datasets are stored in the same directory as this script
    current_dir = Path(__file__).resolve().parent
    file_path = current_dir / filename

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
            item.get("expected_output", f"Keywords to include: {', '.join(item.get('expected_keywords', []))}. Logic: {item.get('reasoning_path', '')}"),
        )
        for item in data[eval_settings.start_index : eval_settings.end_index]
    ]
