import tempfile
import unittest

from packages.model.model_b_trainer import Trainer
from packages.model.model_registry import Registry


class ModelBTrainerTests(unittest.TestCase):
    def test_retrain_interval_sec_is_honored(self):
        with tempfile.TemporaryDirectory() as tmp:
            trainer = Trainer(
                dataset_path=f"{tmp}/dataset.csv",
                registry=Registry(root=f"{tmp}/registry"),
                retrain_interval_sec=12.5,
            )
            self.assertEqual(trainer.retrain_ms, 12500)

    def test_head_is_updated_after_retrain_even_without_promotion(self):
        import csv
        import random
        with tempfile.TemporaryDirectory() as tmp:
            ds = f"{tmp}/dataset.csv"
            with open(ds, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["ts","symbol","label","ret","bar_ts","decision_ts","price_source","market_type","ret_last","range_last","vol_z_last","imb","spread_bps","wall_dist_bps","wall_age","wall_touches"])
                random.seed(1)
                for i in range(260):
                    y = 1 if (i % 3 != 0) else 0
                    base = 1.0 if y == 1 else -1.0
                    feats = [base + random.uniform(-0.1,0.1) for _ in range(8)]
                    w.writerow([i, "BTC/USDT", y, 0.001 if y else -0.001, i, i, "last", "perp", *feats])
            trainer = Trainer(dataset_path=ds, registry=Registry(root=f"{tmp}/registry"), retrain_interval_sec=0.0)
            m, _ = trainer.maybe_retrain(now_ms=1, force=True)
            self.assertIsNotNone(m)
            self.assertTrue(getattr(trainer.head, "fitted", False))
            p = trainer.infer_prob({"ret_last": 1, "range_last": 1, "vol_z_last": 1, "imb": 1, "spread_bps": 1, "wall_dist_bps": 1, "wall_age": 1, "wall_touches": 1})
            self.assertNotEqual(float(p), 0.5)


if __name__ == "__main__":
    unittest.main()
