from __future__ import annotations
from typing import Any, Dict, List


def decide(features: Dict[str, float], inv_results: List[Dict[str, Any]], model_out: Dict[str, Any], model_health: Dict[str, Any], cfg: Dict[str, Any]) -> Dict[str, Any]:
    prof_name = cfg.get("profile", "scalp")
    prof = cfg.get("profiles", {}).get(prof_name, {})
    entry_th = float(prof.get("entry_threshold", 0.06))
    conf_min = float(prof.get("confidence_min", 0.52))
    adx_min = float(prof.get("adx_min", 8))
    spread_max = float(prof.get("spread_max", 12.0))
    volz_max = float(prof.get("volz_max", 3.5))
    uncertainty_max = float(prof.get("uncertainty_max", 0.02))
    rr_min = float(prof.get("rr_min", 0.9))
    trend_strength_min = float(prof.get("trend_strength_min", 0.05))
    ev_min = float(prof.get("ev_min", 0.0))
    high_vol_conf_boost = float(prof.get("high_vol_conf_boost", 0.08))
    cooldown_after_sl = int(prof.get("cooldown_after_sl", 3))
    spread_rank_max = float(prof.get("spread_rank_max", 0.95))
    latency_ms_max = float(prof.get("latency_ms_max", 1200.0))
    high_conf_min = float(prof.get("high_confidence_min", 0.7))
    enforce_auc_min = bool(prof.get("enforce_auc_min", False))
    enforce_directional_agreement = bool(prof.get("enforce_directional_agreement", True))
    enforce_high_confidence = bool(prof.get("enforce_high_confidence", False))

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

    auc = float(model_health.get("auc", 0.0))
    warmup = int(cfg.get("auc_warmup_samples", 600))
    min_auc = float(cfg.get("min_auc", 0.52))
    conf = float(model_out.get("confidence", 0.5))
    auc_override_conf = float(cfg.get("auc_override_confidence_min", 0.70))
    if samples < warmup:
        add("MODEL_AUC_WARMUP", True, auc, f"warmup<{warmup}", note="AUC check skipped during warmup")
    else:
        auc_ok = auc >= min_auc
        if not auc_ok and conf >= auc_override_conf:
            add("MODEL_AUC_MIN", True, auc, min_auc, note=f"override by confidence>={auc_override_conf}")
        elif not auc_ok and not enforce_auc_min:
            add("MODEL_AUC_MIN", True, auc, min_auc, note="soft-check: enforce_auc_min=false")
        else:
            add("MODEL_AUC_MIN", auc_ok, auc, min_auc)

    win_rate = float(features.get("win_rate", 0.5))
    sl_rate = float(features.get("sl_rate", 0.0))
    if sl_rate > 0.55:
        conf_min = min(0.99, conf_min + 0.05)
        rr_min += 0.10
    elif win_rate > 0.62:
        conf_min = max(0.50, conf_min - 0.02)

    regime = str(features.get("regime", "unknown"))
    if regime == "high_volatility":
        conf_min = min(0.99, conf_min + high_vol_conf_boost)
        rr_min += 0.1
    if regime == "mean_reversion":
        rr_min = max(0.6, rr_min - 0.15)
    add("CONFIDENCE_MIN", conf >= conf_min, conf, conf_min)

    pred = float(model_out.get("pred", 0.0))
    add("ENTRY_THRESHOLD", abs(pred) >= entry_th, abs(pred), entry_th)

    adx = float(features.get("adx", 0.0))
    add("ADX_MIN", adx >= adx_min, adx, adx_min)

    spread_bps = float(features.get("spread_bps", features.get("spread", 0.0)))
    add("SPREAD_MAX", spread_bps <= spread_max, spread_bps, spread_max)

    spread_rank = float(features.get("spread_pct_rank", 0.0))
    add("SPREAD_RANK_MAX", spread_rank <= spread_rank_max, spread_rank, spread_rank_max)

    latency_ms = float(features.get("latency_ms", 0.0))
    add("LATENCY_MS_MAX", latency_ms <= latency_ms_max, latency_ms, latency_ms_max)

    volz = float(features.get("volz", abs(features.get("vol_z_last", 0.0))))
    add("VOLZ_MAX", volz <= volz_max, volz, volz_max)

    unc = float(features.get("forecast_uncertainty", 1.0))
    add("UNCERTAINTY_MAX", unc <= uncertainty_max, unc, uncertainty_max)

    rr = float(features.get("rr_ratio", 0.0))
    add("RR_MIN", rr >= rr_min, rr, rr_min)

    trend_strength = float(features.get("trend_strength", 0.0))
    add("TREND_STRENGTH_MIN", trend_strength >= trend_strength_min, trend_strength, trend_strength_min)

    add("REGIME_NOT_LOW_LIQ", regime != "low_liquidity", regime, "!=low_liquidity")

    model_a_direction = str(model_out.get("model_a_direction", ""))
    model_b_direction = str(model_out.get("model_b_direction", ""))
    if (not enforce_directional_agreement) or (model_b_direction in ("", "NEUTRAL")):
        agree = True
    else:
        agree = bool(model_a_direction) and (model_a_direction == model_b_direction)
    add("DIRECTIONAL_AGREEMENT", agree, f"{model_a_direction}/{model_b_direction}", "same")

    costs = float(features.get("costs_bps", 0.0)) / 10000.0
    tp = float(features.get("tp_return", 0.0))
    sl = float(features.get("sl_return", 0.0))
    p = float(model_out.get("p_tp_first", 0.5))
    ev = p * tp - (1.0 - p) * sl - costs
    add("EV_POSITIVE", ev > ev_min, ev, ev_min)

    sl_streak = int(features.get("sl_streak", 0))
    add("COOLDOWN_AFTER_SL", sl_streak < cooldown_after_sl, sl_streak, cooldown_after_sl)

    high_conf = float(model_out.get("confidence", 0.0)) >= high_conf_min
    add("HIGH_CONFIDENCE", (high_conf or (not enforce_high_confidence)), float(model_out.get("confidence", 0.0)), high_conf_min, note="optional" if not enforce_high_confidence else None)

    decision = "PASS" if all(r["pass"] for r in rules) else "FAIL"
    return {
        "decision": decision,
        "rules": rules,
        "reasons": reasons,
        "spread_bps": spread_bps,
        "volz": volz,
        "ret_last": float(features.get("ret_last", 0.0)),
        "range_last": float(features.get("range_last", 0.0)),
        "vol_z_last": float(features.get("vol_z_last", 0.0)),
        "imb": float(features.get("ob_imb", 0.0)),
        "wall_dist_bps": float(features.get("wall_dist_bps", 0.0)),
        "wall_age": float(features.get("wall_age", 0.0)),
        "wall_touches": float(features.get("wall_touches", 0.0)),
        "forecast_uncertainty": unc,
        "rr_ratio": rr,
        "trend_strength": trend_strength,
        "regime": regime,
        "expected_value": ev,
        "sl_streak": sl_streak,
    }
