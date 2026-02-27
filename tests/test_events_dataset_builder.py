import csv
import tempfile
import unittest

from packages.infra.event_store.sqlite_store import SQLiteEventStore
from packages.model.events_dataset_builder import EventsDatasetBuilder


class EventsDatasetBuilderTests(unittest.TestCase):
    def test_build_model_b_from_events(self):
        with tempfile.TemporaryDirectory() as tmp:
            events_db = f"{tmp}/events.sqlite"
            out_csv = f"{tmp}/out.csv"
            es = SQLiteEventStore(events_db)
            es.append({
                "ts": 1, "run_id": "r", "service": "s", "stage": "MODEL_B_INFERRED", "level": "INFO", "event_id": "m1", "symbol": "BTC/USDT",
                "payload": {"ret_last": 0.1, "range_last": 0.2, "vol_z_last": 0.3, "imb": 0.1, "spread_bps": 5, "wall_dist_bps": 12, "wall_age": 3, "wall_touches": 1}
            })
            es.append({
                "ts": 2, "run_id": "r", "service": "s", "stage": "TRADE_CLOSED", "level": "INFO", "event_id": "t1", "symbol": "BTC/USDT",
                "payload": {"pnl": 1.0}
            })
            b = EventsDatasetBuilder(events_db, out_csv)
            n = b.build_model_b(limit=100)
            self.assertEqual(n, 1)
            with open(out_csv, "r", encoding="utf-8") as f:
                rows = list(csv.reader(f))
            self.assertEqual(len(rows), 2)


if __name__ == "__main__":
    unittest.main()
