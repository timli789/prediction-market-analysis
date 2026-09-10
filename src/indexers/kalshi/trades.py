"""Indexer for Kalshi trades data."""

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import duckdb
import pandas as pd
from tqdm import tqdm

from src.common.indexer import Indexer
from src.indexers.kalshi.client import KalshiClient

DATA_DIR = Path("data/kalshi/trades")
MARKETS_DIR = Path("data/kalshi/markets")
CURSOR_FILE = Path("data/kalshi/.backfill_trades_cursor")


class KalshiTradesIndexer(Indexer):
    """Fetches and stores Kalshi trades data."""

    def __init__(
        self,
        min_ts: Optional[int] = None,
        max_ts: Optional[int] = None,
        max_workers: int = 10,
    ):
        super().__init__(
            name="kalshi_trades",
            description="Backfills Kalshi trades data to parquet files",
        )
        self._min_ts = min_ts
        self._max_ts = max_ts
        self._max_workers = max_workers

    def run(self) -> None:
        BATCH_SIZE = 10000
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        CURSOR_FILE.parent.mkdir(parents=True, exist_ok=True)

        # Apply resource limits so DuckDB spills to disk instead of getting OOM-killed
        # on memory-constrained devices (e.g. Raspberry Pi).
        duckdb.sql("PRAGMA max_memory='3GB'")
        duckdb.sql("PRAGMA threads=2")
        duckdb.sql(f"PRAGMA temp_directory='{DATA_DIR.parent / '.duckdb_tmp'}'")

        # Compute markets not yet processed via a DuckDB set difference instead of
        # materializing every existing trade_id into Python sets (was OOM-killing
        # this on memory-constrained devices once enough parquet files piled up).
        # Skipping is done at the ticker level: a ticker is only re-fetched if it
        # has no trades file at all yet, so partially-fetched tickers aren't
        # deduped trade-by-trade — this trades fine-grained resumability for
        # bounded memory use, which is fine since fetches are per-ticker anyway.
        parquet_files = list(DATA_DIR.glob("trades_*.parquet"))

        total_markets = duckdb.sql(f"""
            SELECT COUNT(DISTINCT ticker) FROM '{MARKETS_DIR}/markets_*_*.parquet'
            WHERE volume >= 100 AND status = 'finalized'
        """).fetchone()[0]
        print(f"Found {total_markets} unique markets")

        if parquet_files:
            print("Computing markets not yet processed...")
            tickers_to_process = [
                row[0]
                for row in duckdb.sql(f"""
                    SELECT DISTINCT ticker FROM '{MARKETS_DIR}/markets_*_*.parquet'
                    WHERE volume >= 100 AND status = 'finalized'
                    EXCEPT
                    SELECT DISTINCT ticker FROM '{DATA_DIR}/trades_*.parquet'
                    ORDER BY ticker
                """).fetchall()
            ]
        else:
            tickers_to_process = [
                row[0]
                for row in duckdb.sql(f"""
                    SELECT DISTINCT ticker FROM '{MARKETS_DIR}/markets_*_*.parquet'
                    WHERE volume >= 100 AND status = 'finalized'
                    ORDER BY ticker
                """).fetchall()
            ]

        print(
            f"Skipped {total_markets - len(tickers_to_process)} already processed, "
            f"{len(tickers_to_process)} to fetch"
        )

        if not tickers_to_process:
            print("Nothing to process")
            return

        all_trades: list[dict] = []
        total_trades_saved = 0
        next_chunk_idx = 0

        # Calculate next chunk index
        if parquet_files:
            indices = []
            for f in parquet_files:
                parts = f.stem.split("_")
                if len(parts) >= 2:
                    try:
                        indices.append(int(parts[1]))
                    except ValueError:
                        pass
            if indices:
                next_chunk_idx = max(indices) + BATCH_SIZE

        def save_batch(trades_batch: list[dict]) -> int:
            nonlocal next_chunk_idx
            if not trades_batch:
                return 0
            chunk_path = DATA_DIR / f"trades_{next_chunk_idx}_{next_chunk_idx + BATCH_SIZE}.parquet"
            df = pd.DataFrame(trades_batch)
            df.to_parquet(chunk_path)
            next_chunk_idx += BATCH_SIZE
            return len(trades_batch)

        def fetch_ticker_trades(ticker: str) -> list[dict]:
            """Fetch trades for a single ticker."""
            client = KalshiClient()
            try:
                trades = client.get_market_trades(
                    ticker,
                    verbose=False,
                    min_ts=self._min_ts,
                    max_ts=self._max_ts,
                )
                if not trades:
                    return []
                fetched_at = datetime.utcnow()
                return [{**asdict(t), "_fetched_at": fetched_at} for t in trades]
            finally:
                client.close()

        # Concurrent fetching, submitted in bounded chunks so we never hold
        # millions of pending Future objects in memory at once (was the other
        # major contributor to the OOM kill).
        SUBMIT_CHUNK_SIZE = 2000
        pbar = tqdm(total=len(tickers_to_process), desc="Fetching trades")
        with ThreadPoolExecutor(max_workers=self._max_workers) as executor:
            for chunk_start in range(0, len(tickers_to_process), SUBMIT_CHUNK_SIZE):
                chunk = tickers_to_process[chunk_start:chunk_start + SUBMIT_CHUNK_SIZE]
                futures = {executor.submit(fetch_ticker_trades, ticker): ticker for ticker in chunk}

                for future in as_completed(futures):
                    ticker = futures[future]
                    try:
                        trades_data = future.result()
                        if trades_data:
                            all_trades.extend(trades_data)

                        pbar.update(1)
                        pbar.set_postfix(buffer=len(all_trades), saved=total_trades_saved, last=ticker[-20:])

                        # Save in batches
                        while len(all_trades) >= BATCH_SIZE:
                            saved = save_batch(all_trades[:BATCH_SIZE])
                            total_trades_saved += saved
                            all_trades = all_trades[BATCH_SIZE:]

                    except Exception as e:
                        pbar.update(1)
                        tqdm.write(f"Error fetching {ticker}: {e}")

        pbar.close()

        # Save remaining
        if all_trades:
            total_trades_saved += save_batch(all_trades)

        print(
            f"\nBackfill trades complete: {len(tickers_to_process)} markets processed, "
            f"{total_trades_saved} trades saved"
        )
