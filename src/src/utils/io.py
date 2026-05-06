from __future__ import annotations
from pathlib import Path
from typing import Dict, List
import json
import pandas as pd

def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p

def append_row_csv(csv_path: str | Path, row: Dict) -> None:
    csv_path = Path(csv_path)
    df = pd.DataFrame([row])
    header = not csv_path.exists()
    df.to_csv(csv_path, mode="a", index=False, header=header)

def append_rows_csv(csv_path: str | Path, rows: List[Dict]) -> None:
    if not rows:
        return
    csv_path = Path(csv_path)
    df = pd.DataFrame(rows)
    header = not csv_path.exists()
    df.to_csv(csv_path, mode="a", index=False, header=header)

def save_json(path: str | Path, obj: Dict) -> None:
    Path(path).write_text(json.dumps(obj, indent=2), encoding="utf-8")
