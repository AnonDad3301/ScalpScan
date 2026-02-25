from __future__ import annotations
from typing import Any, Dict, List


def decide(features: Dict[str, float], inv_results: List[Dict[str, Any]], model_out: Dict[str, Any], model_health: Dict[str, Any], cfg: Dict[str, Any]) -> Dict[str, Any]:
    prof_name = cfg.get("profile", "scalp")
    prof = cfg.get("profiles", {}).get(prof_name, {})
    entry_th = float(prof.get("entry_threshold", 0.06))
    conf_min = float(prof.get("confidence_min", 0.52))
    adx_min = float(prof.get("adx_min", 8))
    spread_max = float(prof.get("spread_max", 1.5))
    volz_max = float(prof.get("volz_max", 3.5))

    rules = []
    reasons = []

    def add(name, passed, value=None, threshold=None, note=None):
        rules.append({"name": name, "pass": bool(passed), "value": value, "threshold": threshold, "note": note})
        if not passed:
            reasons.append(name)

    add("INVARIANTS_OK", all(bool(r.get("pass")) for r in inv_results))

    samples = int(model_health.get("samples", 0))
    min_samples = int(cfg.get("min_samples", 50))
    add("MODEL_SAMPLES_MIN", samples >= min_samples, samples, min_samples)

    # AUC warmup: don't block signals until enough samples
    auc = float(model_health.get("auc", 0.0))
    warmup = int(cfg.get("auc_warmup_samples", 600))
    min_auc = float(cfg.get("min_auc", 0.52))
    if samples < warmup:
        add("MODEL_AUC_WARMUP", True, auc, f"warmup<{warmup}", note="AUC check skipped during warmup")
    else:
        add("MODEL_AUC_MIN", auc >= min_auc, auc, min_auc)

    conf = float(model_out.get("confidence", 0.5))
    add("CONFIDENCE_MIN", conf >= conf_min, conf, conf_min)

    pred = float(model_out.get("pred", 0.0))
    add("ENTRY_THRESHOLD", abs(pred) >= entry_th, abs(pred), entry_th)

    adx = float(features.get("adx", 0.0))
    add("ADX_MIN", adx >= adx_min, adx, adx_min)

    spread = float(features.get("spread_bps", 0.0))
    add("SPREAD_MAX", spread <= spread_max, spread, spread_max)

    volz = float(features.get("volz", 0.0))
    add("VOLZ_MAX", volz <= volz_max, volz, volz_max)

    decision = "PASS" if all(r["pass"] for r in rules) else "FAIL"
    return {"decision": decision, "rules": rules, "reasons": reasons}
