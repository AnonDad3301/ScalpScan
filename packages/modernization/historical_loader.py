from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple


OHLCV = Tuple[int, float, float, float, float, float]


@dataclass
class BackfillReport:
    requested: int
    inserted: int
    duplicates: int
    gaps: int


class HistoricalDataLoader:
    """Exchange-agnostic historical loader with dedup + integrity checks."""

    def backfill(
        self,
        market,
        symbol: str,
        timeframe: str,
        limit: int,
        existing: Sequence[OHLCV],
    ) -> Tuple[List[OHLCV], BackfillReport]:
        raw = market.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        old_ts = {int(x[0]) for x in existing}
        merged = list(existing)
        inserted = 0
        duplicates = 0

        for row in raw:
            ts = int(row[0])
            if ts in old_ts:
                duplicates += 1
                continue
            merged.append((
                ts,
                float(row[1]),
                float(row[2]),
                float(row[3]),
                float(row[4]),
                float(row[5]),
            ))
            old_ts.add(ts)
            inserted += 1

        merged.sort(key=lambda x: x[0])
        gaps = self._count_gaps(merged)
        return merged, BackfillReport(requested=len(raw), inserted=inserted, duplicates=duplicates, gaps=gaps)

    @staticmethod
    def _count_gaps(rows: Sequence[OHLCV]) -> int:
        if len(rows) < 3:
            return 0
        intervals = [rows[i][0] - rows[i - 1][0] for i in range(1, len(rows))]
        step = min([x for x in intervals if x > 0], default=0)
        if step <= 0:
            return 0
        return sum(1 for x in intervals if x > step * 2)
