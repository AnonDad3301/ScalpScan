import sqlite3
import tempfile
import unittest

from packages.infra.event_store.sqlite_store import SQLiteEventStore
from packages.infra.timeline_store.sqlite_timeline import SQLiteTimelineStore


class EventSchemaAndTimelineTests(unittest.TestCase):
    def test_event_store_migrates_to_schema_v2(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = f"{tmp}/events.sqlite"
            conn = sqlite3.connect(db)
            conn.execute("CREATE TABLE events (ts INTEGER, run_id TEXT, service TEXT, exchange TEXT, market TEXT, symbol TEXT, timeframe TEXT, stage TEXT, level TEXT, event_id TEXT PRIMARY KEY, payload_json TEXT)")
            conn.commit(); conn.close()

            es = SQLiteEventStore(db)
            es.append({"ts": 1, "run_id": "r", "service": "s", "stage": "X", "level": "INFO", "event_id": "e1", "payload": {"a": 1}})
            e = es.tail(1)[0]
            self.assertEqual(e["schema_version"], 2)
            self.assertIn("source", e)
            self.assertIn("tags", e)

    def test_timeline_store_append_tail(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = f"{tmp}/timeline.sqlite"
            tl = SQLiteTimelineStore(db)
            tl.append(1000, "pass_rate", 67.5, {"tf": "3m"})
            row = tl.tail(1)[0]
            self.assertEqual(row["metric"], "pass_rate")
            self.assertAlmostEqual(float(row["value"]), 67.5, places=6)
            self.assertEqual(row["tags"].get("tf"), "3m")

    def test_unified_log_file_is_written(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = f"{tmp}/events.sqlite"
            es = SQLiteEventStore(db)
            es.append({"ts": 2, "run_id": "r", "service": "s", "stage": "Y", "level": "INFO", "event_id": "e2", "payload": {"x": 2}})
            rows = es.unified_log_tail(5)
            self.assertTrue(len(rows) >= 1)
            self.assertEqual(rows[-1].get("event_id"), "e2")
            self.assertEqual(rows[-1].get("stage"), "Y")


if __name__ == "__main__":
    unittest.main()
