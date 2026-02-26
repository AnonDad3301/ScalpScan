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


if __name__ == "__main__":
    unittest.main()
