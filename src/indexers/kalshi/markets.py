"""Indexer for Kalshi markets data."""

from pathlib import Path
from typing import Optional

from src.common.indexer import Indexer
from src.common.storage import ParquetStorage
from src.indexers.kalshi.client import KalshiClient

DATA_DIR = Path("data/kalshi/markets")
CURSOR_FILE = Path("data/kalshi/.backfill_cursor")


class KalshiMarketsIndexer(Indexer):
    """Fetches and stores Kalshi markets data."""

    def __init__(
        self,
        min_close_ts: Optional[int] = None,
        max_close_ts: Optional[int] = None,
        status: str = "settled",
    ):
        super().__init__(
            name="kalshi_markets",
            description="Backfills Kalshi markets data to parquet files",
        )
        self._min_close_ts = min_close_ts
        self._max_close_ts = max_close_ts
        # Only backfill markets that have fully settled (status="finalized" in the
        # API response) — i.e. trading has stopped AND a final result is available.
        # Pass status=None to fall back to the API default (all statuses).
        self._status = status

    def run(self) -> None:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        CURSOR_FILE.parent.mkdir(parents=True, exist_ok=True)

        client = KalshiClient()
        storage = ParquetStorage(data_dir=DATA_DIR)

        cursor = None
        if CURSOR_FILE.exists():
            cursor = CURSOR_FILE.read_text().strip() or None
            if cursor:
                print(f"Resuming from cursor: {cursor[:20]}...")

        total = 0
        try:
            for markets, next_cursor in client.iter_markets(
                limit=1000,
                cursor=cursor,
                min_close_ts=self._min_close_ts,
                max_close_ts=self._max_close_ts,
                status=self._status,
            ):
                if markets:
                    total_stored = storage.append_markets(markets)
                    total += len(markets)
                    print(f"Fetched {len(markets)} markets (total: {total}, stored: {total_stored})")

                if next_cursor:
                    CURSOR_FILE.write_text(next_cursor)
                else:
                    if CURSOR_FILE.exists():
                        CURSOR_FILE.unlink()
                    break
        finally:
            # Persist any buffered records that didn't reach a full chunk yet,
            # so a crash/interrupt (or hitting the end of the results) never
            # loses markets that were already fetched but not yet flushed.
            storage.flush()

        print(f"\nBackfill complete: {total} markets fetched")
