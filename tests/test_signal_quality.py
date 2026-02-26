import unittest

from packages.feature.pipeline import compute
from packages.decision.gate import decide


class SignalQualityTests(unittest.TestCase):
    def test_compute_has_spread_bps_and_volz(self):
        ohlcv=[]
        p=100.0
        for i in range(40):
            p+=0.2
            ohlcv.append((i*60000,p-0.3,p+0.4,p-0.4,p,100+i))
        feats=compute(ohlcv,{"spread":0.1,"imbalance":0.2})
        self.assertIn("spread_bps", feats)
        self.assertIn("volz", feats)
        self.assertGreater(feats["spread_bps"],0)

    def test_gate_exports_model_b_features(self):
        features={
            "adx":20,"spread_bps":0.5,"volz":0.2,
            "ret_last":0.001,"range_last":0.01,"vol_z_last":0.2,
            "ob_imb":0.1,"wall_dist_bps":12.0,"wall_age":3.0,"wall_touches":2.0,
            "forecast_uncertainty_3m":0.01,"rr_ratio":1.6,"trend_strength":0.3,
        }
        out=decide(features,[{"pass":True}],{"confidence":0.8,"pred":0.2},{"samples":1000,"auc":0.6},{
            "profile":"scalp",
            "profiles":{"scalp":{"entry_threshold":0.06,"confidence_min":0.5,"adx_min":8,"spread_max":2.0,"volz_max":4.0}},
            "min_samples":50,
            "auc_warmup_samples":600,
            "min_auc":0.52,
        })
        self.assertIn("ret_last", out)
        self.assertIn("wall_age", out)
        self.assertIn("spread_bps", out)


if __name__ == "__main__":
    unittest.main()
