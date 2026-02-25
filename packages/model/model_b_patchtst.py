from __future__ import annotations
import os, time, json
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

@dataclass
class ModelBState:
    loaded: bool = False
    hf_id: str = ""
    cache_dir: str = ""
    version: str = ""
    last_error: str = ""

class ModelBPatchTST:
    """PatchTST backbone + lightweight head (to be trained offline/online later).

    For Iteration-3 we implement:
    - HF cache download + load backbone
    - embedding extraction (mean pooled)
    - logistic head placeholder (returns prob=0.5 until trained)
    """
    def __init__(self, hf_id: str, cache_dir: str = "data/hf_cache", device: str = "cpu"):
        self.hf_id=hf_id
        self.cache_dir=cache_dir
        self.device=device
        self.state=ModelBState(loaded=False, hf_id=hf_id, cache_dir=cache_dir, version=hf_id)
        self.backbone=None
        self._head_w=None
        self._head_b=0.0

    def load(self) -> ModelBState:
        try:
            os.makedirs(self.cache_dir, exist_ok=True)
            # lazy imports to avoid hard dependency failures
            from huggingface_hub import snapshot_download
            from transformers import AutoModel

            local_dir = snapshot_download(repo_id=self.hf_id, cache_dir=self.cache_dir, local_files_only=False)
            self.backbone = AutoModel.from_pretrained(local_dir)
            self.backbone.eval()
            self.state.loaded=True
            self.state.last_error=""
        except Exception as e:
            self.state.loaded=False
            self.state.last_error=str(e)
        return self.state

    def infer(self, X: np.ndarray) -> Dict[str, Any]:
        """X shape [C, L] -> prob."""
        if not self.state.loaded or self.backbone is None:
            return {"prob": 0.5, "status": "NOT_LOADED"}
        try:
            import torch
            # PatchTST expects [batch, seq, features] or model-specific; AutoModel outputs last_hidden_state.
            # We'll map [C,L] -> [1, L, C]
            inp=torch.tensor(X.T[None,:,:], dtype=torch.float32)
            with torch.no_grad():
                out=self.backbone(inputs_embeds=inp) if "inputs_embeds" in self.backbone.forward.__code__.co_varnames else self.backbone(inp)
            h=getattr(out, "last_hidden_state", None)
            if h is None:
                # fallback: first element
                h=out[0]
            emb=h.mean(dim=1).cpu().numpy()[0]  # [D]
            # logistic head placeholder
            z=float(np.dot(self._head_w, emb) + self._head_b) if self._head_w is not None else 0.0
            prob=1.0/(1.0+np.exp(-z))
            return {"prob": float(prob), "status": "OK"}
        except Exception as e:
            return {"prob": 0.5, "status": "ERROR", "err": str(e)}
