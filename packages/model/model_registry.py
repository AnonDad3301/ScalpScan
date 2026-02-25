from __future__ import annotations
import os, json, time
from dataclasses import dataclass
from typing import Any, Dict, Optional, List
import joblib

@dataclass
class Info:
    name:str
    version:str
    path:str
    created_ms:int
    metrics:Dict[str,Any]

class Registry:
    def __init__(self, root:str):
        self.root=root
        os.makedirs(self.root, exist_ok=True)

    def save(self, name:str, obj:Any, metrics:Dict[str,Any]) -> Info:
        ver=str(int(time.time()*1000))
        p=os.path.join(self.root,f"{name}_{ver}.joblib")
        joblib.dump(obj,p)
        info=Info(name=name,version=ver,path=p,created_ms=int(time.time()*1000),metrics=metrics)
        with open(p+".json","w",encoding="utf-8") as f:
            json.dump(info.__dict__,f,ensure_ascii=False,indent=2)
        return info

    def promote(self, info:Info)->None:
        with open(os.path.join(self.root,f"{info.name}_CURRENT.json"),"w",encoding="utf-8") as f:
            json.dump(info.__dict__,f,ensure_ascii=False,indent=2)

    def load_current(self, name:str)->Optional[Any]:
        ptr=os.path.join(self.root,f"{name}_CURRENT.json")
        if not os.path.exists(ptr):
            return None
        try:
            meta=json.load(open(ptr,"r",encoding="utf-8"))
            return joblib.load(meta["path"])
        except Exception:
            return None

    def rollback(self, name:str)->bool:
        metas=[]
        for fn in os.listdir(self.root):
            if fn.startswith(f"{name}_") and fn.endswith(".joblib.json"):
                try:
                    metas.append(json.load(open(os.path.join(self.root,fn),"r",encoding="utf-8")))
                except Exception:
                    pass
        metas.sort(key=lambda m:int(m.get("created_ms",0)), reverse=True)
        if len(metas)<2:
            return False
        with open(os.path.join(self.root,f"{name}_CURRENT.json"),"w",encoding="utf-8") as f:
            json.dump(metas[1],f,ensure_ascii=False,indent=2)
        return True
