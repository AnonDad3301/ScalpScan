from __future__ import annotations
import json, time, uuid, os
from dataclasses import dataclass
from typing import Any, Dict, Optional

def now_ms() -> int:
    return int(time.time()*1000)

def new_span_id() -> str:
    return str(uuid.uuid4())

@dataclass
class Span:
    trace_id: str
    span_id: str
    name: str
    symbol: str
    start_ms: int
    meta: Dict[str, Any]

class Telemetry:
    def __init__(self, path: str, service: str, es=None, base_event: Optional[Dict[str, Any]]=None):
        self.path = path
        self.service = service
        self.es = es
        self.base_event = base_event or {}
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    def log(self, level: str, msg: str, **fields: Any) -> None:
        rec = {"ts": now_ms(), "service": self.service, "level": level, "msg": msg, **fields}
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def span_start(self, name: str, trace_id: str, symbol: str="*", **meta: Any) -> Span:
        sp = Span(trace_id=trace_id, span_id=new_span_id(), name=name, symbol=symbol, start_ms=now_ms(), meta=meta)
        self.log("INFO", "SPAN_START", trace_id=sp.trace_id, span_id=sp.span_id, span=name, symbol=symbol, meta=meta)
        if self.es:
            self.es.append({**self.base_event, "event_id": str(uuid.uuid4()), "ts": now_ms(), "run_id": trace_id,
                            "service": self.service, "symbol": symbol, "stage": "SPAN_START", "level": "INFO",
                            "payload": {"span": name, "span_id": sp.span_id, "meta": meta}})
        return sp

    def span_end(self, sp: Span, status: str="OK", **meta: Any) -> None:
        dur = now_ms() - sp.start_ms
        dur = int((time.time()*1000) - sp.start_ms)
        meta = dict(sp.meta) if sp.meta else {}
        payload = {"status": status, "dur_ms": dur, **meta}
        self.log("INFO", "SPAN_END", trace_id=sp.trace_id, span_id=sp.span_id, span=sp.name, symbol=sp.symbol, **payload)
        if self.es:
            self.es.append({**self.base_event, "event_id": str(uuid.uuid4()), "ts": now_ms(), "run_id": sp.trace_id,
                            "service": self.service, "symbol": sp.symbol, "stage": "SPAN_END", "level": "INFO",
                            "payload": payload})
