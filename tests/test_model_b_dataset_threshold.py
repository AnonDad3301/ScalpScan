import csv
import tempfile
import unittest

from packages.model.dataset_builder import DatasetBuilder


class ModelBDatasetThresholdTests(unittest.TestCase):
    def test_append_shadow_skips_micro_returns_under_threshold(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = f"{tmp}/samples.csv"
            ds = DatasetBuilder(path=path, label_ret_threshold=0.001)
            ok = ds.append_shadow(
                ts=1,
                symbol="BTC/USDT:USDT",
                bar_ts=1,
                decision_ts=1,
                price_source="last",
                market_type="perp",
                feats={},
                ret=0.0005,
            )
            self.assertFalse(ok)
            with open(path, "r", encoding="utf-8") as f:
                rows = list(csv.reader(f))
            self.assertEqual(len(rows), 1)  # header only


if __name__ == "__main__":
    unittest.main()
