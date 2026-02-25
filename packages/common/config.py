from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict
import yaml

@dataclass
class Config:
    raw: Dict[str, Any]

    @staticmethod
    def load(path: str = "config.yaml") -> "Config":
        p = Path(path)
        data = yaml.safe_load(p.read_text(encoding="utf-8"))
        return Config(raw=data)
