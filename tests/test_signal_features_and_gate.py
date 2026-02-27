import unittest

from packages.feature.pipeline import compute
from packages.decision.gate import decide


class SignalFeatureGateTests(unittest.TestCase):
    def test_pipeline_computes_nonzero_micro_features(self):
        ohlcv = []
        p = 100.0
        for i in range(40):
            p += 0.2
            ohlcv.append((i * 60000, p - 0.1, p + 0.3, p - 0.3, p, 100 + i))
        ob = {
            "spread": 0.2,
            "spread_bps": 20.0,
            "imbalance": 0.15,
        }
        feats = compute(ohlcv, ob)
        self.assertGreater(feats["spread_bps"], 0)
        self.assertNotEqual(feats["ret_last"], 0.0)
        self.assertNotEqual(feats["range_last"], 0.0)
        self.assertIn("vol_z_last", feats)

    def test_gate_exposes_quality_filters(self):
        features = {
            "adx": 30,
            "spread_bps": 5,
            "volz": 0.8,
            "ret_last": 0.001,
            "range_last": 0.002,
            "vol_z_last": 0.3,
            "ob_imb": 0.1,
            "forecast_uncertainty": 0.005,
            "rr_ratio": 1.8,
            "trend_strength": 0.4,
            "regime": "trend",
            "tp_return": 0.01,
            "sl_return": 0.006,
            "costs_bps": 3.0,
            "sl_streak": 0,
            "spread_pct_rank": 0.2,
            "latency_ms": 200,
        }
        out = decide(
            features,
            inv_results=[{"pass": True}],
            model_out={"confidence": 0.8, "pred": 0.2, "p_tp_first": 0.62, "model_a_direction": "LONG", "model_b_direction": "LONG"},
            model_health={"samples": 700, "auc": 0.65},
            cfg={
                "profile": "scalp",
                "profiles": {
                    "scalp": {
                        "entry_threshold": 0.06,
                        "confidence_min": 0.52,
                        "adx_min": 8,
                        "spread_max": 20,
                        "volz_max": 3.5,
                        "uncertainty_max": 0.02,
                        "rr_min": 1.2,
                        "trend_strength_min": 0.05,
                        "high_confidence_min": 0.70,
                    }
                },
                "min_samples": 50,
                "auc_warmup_samples": 600,
                "min_auc": 0.52,
            },
        )
        self.assertEqual(out["decision"], "PASS")
        self.assertIn("forecast_uncertainty", out)
        self.assertIn("rr_ratio", out)
        self.assertGreater(out.get("expected_value", -1), 0.0)

    def test_gate_auc_override_by_high_confidence(self):
        features = {
            "adx": 20,
            "spread_bps": 5,
            "volz": 0.5,
            "forecast_uncertainty": 0.005,
            "rr_ratio": 1.1,
            "trend_strength": 0.2,
            "regime": "trend",
            "tp_return": 0.01,
            "sl_return": 0.005,
            "costs_bps": 3.0,
            "sl_streak": 0,
            "spread_pct_rank": 0.2,
            "latency_ms": 200,
        }
        out = decide(
            features,
            inv_results=[{"pass": True}],
            model_out={"confidence": 0.75, "pred": 0.2, "p_tp_first": 0.65, "model_a_direction": "LONG", "model_b_direction": "LONG"},
            model_health={"samples": 800, "auc": 0.45},
            cfg={
                "profile": "scalp",
                "profiles": {
                    "scalp": {
                        "entry_threshold": 0.06,
                        "confidence_min": 0.52,
                        "adx_min": 8,
                        "spread_max": 20,
                        "volz_max": 3.5,
                        "uncertainty_max": 0.02,
                        "rr_min": 0.9,
                        "trend_strength_min": 0.05,
                        "high_confidence_min": 0.70,
                    }
                },
                "min_samples": 50,
                "auc_warmup_samples": 600,
                "min_auc": 0.52,
                "auc_override_confidence_min": 0.70,
            },
        )
        self.assertEqual(out["decision"], "PASS")


    def test_gate_soft_allows_low_samples_when_not_enforced(self):
        features = {
            "adx": 30, "spread_bps": 5, "spread_pct_rank": 0.2, "latency_ms": 100,
            "volz": 0.4, "forecast_uncertainty": 0.003, "rr_ratio": 1.4, "trend_strength": 0.3,
            "regime": "trend", "tp_return": 0.012, "sl_return": 0.006, "costs_bps": 3.0, "sl_streak": 0,
        }
        out = decide(
            features,
            inv_results=[{"pass": True}],
            model_out={"confidence": 0.7, "pred": 0.2, "p_tp_first": 0.62, "model_a_direction": "LONG", "model_b_direction": "LONG"},
            model_health={"samples": 5, "auc": 0.45},
            cfg={
                "profile": "scalp",
                "profiles": {"scalp": {
                    "entry_threshold": 0.06, "confidence_min": 0.52, "adx_min": 8, "spread_max": 20,
                    "volz_max": 3.5, "uncertainty_max": 0.02, "rr_min": 0.9, "trend_strength_min": 0.05,
                    "enforce_auc_min": False, "enforce_directional_agreement": False, "enforce_high_confidence": False,
                    "enforce_model_samples_min": False,
                }},
                "min_samples": 50, "auc_warmup_samples": 600, "min_auc": 0.52,
            },
        )
        self.assertEqual(out["decision"], "PASS")

    def test_gate_blocks_extreme_spread_window(self):
        features = {
            "adx": 25, "spread_bps": 3, "spread_pct_rank": 0.99, "latency_ms": 1500,
            "volz": 0.3, "forecast_uncertainty": 0.004, "rr_ratio": 1.3, "trend_strength": 0.3,
            "regime": "trend", "tp_return": 0.01, "sl_return": 0.005, "costs_bps": 3.0, "sl_streak": 0,
        }
        out = decide(
            features,
            inv_results=[{"pass": True}],
            model_out={"confidence": 0.85, "pred": 0.2, "p_tp_first": 0.65, "model_a_direction": "LONG", "model_b_direction": "LONG"},
            model_health={"samples": 800, "auc": 0.6},
            cfg={
                "profile": "scalp",
                "profiles": {"scalp": {"entry_threshold": 0.06, "confidence_min": 0.52, "adx_min": 8, "spread_max": 20, "volz_max": 3.5, "uncertainty_max": 0.02, "rr_min": 0.9, "trend_strength_min": 0.05}},
                "min_samples": 50, "auc_warmup_samples": 600, "min_auc": 0.52,
            },
        )
        self.assertEqual(out["decision"], "FAIL")
        self.assertIn("SPREAD_RANK_MAX", out["reasons"])

    def test_gate_allows_neutral_model_b_direction(self):
        features = {
            "adx": 30, "spread_bps": 5, "spread_pct_rank": 0.2, "latency_ms": 100,
            "volz": 0.4, "forecast_uncertainty": 0.003, "rr_ratio": 1.4, "trend_strength": 0.3,
            "regime": "trend", "tp_return": 0.012, "sl_return": 0.006, "costs_bps": 3.0, "sl_streak": 0,
        }
        out = decide(
            features,
            inv_results=[{"pass": True}],
            model_out={"confidence": 0.6, "pred": 0.2, "p_tp_first": 0.62, "model_a_direction": "LONG", "model_b_direction": "NEUTRAL"},
            model_health={"samples": 900, "auc": 0.45},
            cfg={
                "profile": "scalp",
                "profiles": {"scalp": {
                    "entry_threshold": 0.06, "confidence_min": 0.52, "adx_min": 8, "spread_max": 20,
                    "volz_max": 3.5, "uncertainty_max": 0.02, "rr_min": 0.9, "trend_strength_min": 0.05,
                    "enforce_auc_min": False, "enforce_directional_agreement": True, "enforce_high_confidence": False,
                }},
                "min_samples": 50, "auc_warmup_samples": 600, "min_auc": 0.52,
            },
        )
        self.assertEqual(out["decision"], "PASS")


if __name__ == "__main__":
    unittest.main()
