from __future__ import annotations

from typing import Dict


class ScenarioSimulationEngine:
    def simulate(self, similarity_summary: Dict[str, float], forecast_4m: Dict[str, float]) -> Dict[str, object]:
        p_up = float(similarity_summary.get("p_up_4m", 0.0))
        p_down = float(similarity_summary.get("p_down_4m", 0.0))
        p_flat = max(0.0, 1.0 - p_up - p_down)

        ret = float(forecast_4m.get("ret", 0.0))
        unc = max(1e-9, float(forecast_4m.get("uncertainty", 1.0)))
        tilt = max(-0.25, min(0.25, ret / unc * 0.1))

        bullish = min(1.0, max(0.0, p_up + max(0.0, tilt)))
        bearish = min(1.0, max(0.0, p_down + max(0.0, -tilt)))
        neutral = min(1.0, max(0.0, 1.0 - bullish - bearish))
        reversal = min(bullish, bearish) * 0.5 + min(0.15, p_flat)

        confidence = max(0.0, min(1.0, 1.0 - unc * 10.0))
        direction = "neutral"
        if bullish > bearish and bullish >= neutral:
            direction = "bullish"
        elif bearish > bullish and bearish >= neutral:
            direction = "bearish"

        return {
            "direction": direction,
            "confidence": confidence,
            "expected_range": {
                "min_ret": float(similarity_summary.get("avg_ret_4m", 0.0) - unc),
                "max_ret": float(similarity_summary.get("avg_ret_4m", 0.0) + unc),
            },
            "scenarios": {
                "bullish": bullish,
                "bearish": bearish,
                "neutral": neutral,
                "false_breakout_reversal": min(1.0, reversal),
            },
            "explanation": "Blend of historical similarity and uncertainty-adjusted 4m trend projection.",
        }
