from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict

class FileSnapshotStore:
    def __init__(self, root: str):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def put(self, run_id: str, symbol: str, payload: Dict[str, Any]) -> str:
        p = self.root / run_id
        p.mkdir(parents=True, exist_ok=True)
        fp = p / f"{symbol.replace('/','_')}.json"
        fp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        return str(fp)
