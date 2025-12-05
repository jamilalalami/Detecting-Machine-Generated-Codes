import json
from pathlib import Path

def read_json(path):
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def save_json(obj, path):
    path = Path(path)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)