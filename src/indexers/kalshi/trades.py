"""Indexer for Kalshi trades data."""

import threading
from collections import deque
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
MANIFEST_FILE = Path("data/kalshi/.trades_manifest")


class KalshiTradesIndexer(Indexer):
    """Fetches and stores Kalshi trades data."""

    def __init__(
        self,
        min_ts: Optional[int] = None,
        max_ts: Optional[int] = None,
        max_workers: int = 4,
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
        MANIFEST_FILE.parent.mkdir(parents=True, exist_ok=True)

        # Apply resource limits so DuckDB spills to disk instead of getting OOM-killed
        # on memory-constrained devices (e.g. Raspberry Pi).
        duckdb.sql("PRAGMA max_memory='3GB'")
        duckdb.sql("PRAGMA threads=2")
        duckdb.sql(f"PRAGMA temp_directory='{DATA_DIR.parent / '.duckdb_tmp'}'")

        parquet_files = list(DATA_DIR.glob("trades_*.parquet"))

        # Which tickers are already done is tracked in a flat append-only manifest
        # file instead of re-deriving it from every trades_*.parquet file on each
        # run. Globbing tens of thousands of small parquet files just to compute
        # a ticker set difference got slower (and more memory-hungry, blowing past
        # the max_memory pragma above) as the trades directory grew, which is what
        # made backfills look like they'd frozen for minutes before doing any work.
        if not MANIFEST_FILE.exists():
            if parquet_files:
                print("No manifest found but trade files exist — bootstrapping ticker manifest (one-time)...")
                # Written via DuckDB's own COPY instead of fetchall()-ing into a Python
                # list: with tens of millions of tickers, that list (and its per-object
                # Python overhead) stuck around for the rest of run() and was what
                # actually blew past PRAGMA max_memory and got the process OOM-killed —
                # the pragma only bounds DuckDB's buffers, not Python-side objects.
                duckdb.sql(f"""
                    COPY (SELECT DISTINCT ticker FROM '{DATA_DIR}/trades_*.parquet' ORDER BY ticker)
                    TO '{MANIFEST_FILE}' (FORMAT CSV, HEADER false)
                """)
                manifest_ticker_count = duckdb.sql(
                    f"SELECT COUNT(*) FROM read_csv('{MANIFEST_FILE}', header=false, columns={{'ticker': 'VARCHAR'}})"
                ).fetchone()[0]
                print(f"Manifest bootstrapped with {manifest_ticker_count} tickers")
            else:
                MANIFEST_FILE.touch()

        total_markets = duckdb.sql(f"""
            SELECT COUNT(DISTINCT ticker) FROM '{MARKETS_DIR}/markets_*_*.parquet'
            WHERE volume >= 100 AND status = 'finalized'
        """).fetchone()[0]
        print(f"Found {total_markets} unique markets")

        if MANIFEST_FILE.stat().st_size > 0:
            tickers_to_process = [
                row[0]
                for row in duckdb.sql(f"""
                    SELECT DISTINCT ticker FROM '{MARKETS_DIR}/markets_*_*.parquet'
                    WHERE volume >= 100 AND status = 'finalized'
                    EXCEPT
                    SELECT ticker FROM read_csv('{MANIFEST_FILE}', header=false, columns={{'ticker': 'VARCHAR'}})
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
            f"Skipped {total_markets - len(tickers_to_process)} already processed, {len(tickers_to_process)} to fetch"
        )

        if not tickers_to_process:
            print("Nothing to process")
            return

        # (ticker, trades) groups waiting to be flushed. A ticker only ever lands
        # here once its own fetch is fully complete — high-volume tickers flush
        # their own BATCH_SIZE-sized chunks directly as they're fetched (see
        # fetch_ticker_trades) and only pass their small leftover remainder
        # through here — so once a group is written to a parquet chunk, its
        # ticker can be safely appended to the manifest as done.
        pending: deque[tuple[str, list[dict]]] = deque()
        pending_count = 0
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

        # Guards next_chunk_idx and total_trades_saved, and serializes parquet
        # writes themselves: worker threads now flush high-volume tickers
        # directly (see fetch_ticker_trades) instead of only ever flushing from
        # the main thread via flush_ready, so both paths must share this lock.
        write_lock = threading.Lock()

        def save_batch(trades_batch: list[dict]) -> int:
            nonlocal next_chunk_idx, total_trades_saved
            if not trades_batch:
                return 0
            chunk_path = DATA_DIR / f"trades_{next_chunk_idx}_{next_chunk_idx + BATCH_SIZE}.parquet"
            df = pd.DataFrame(trades_batch)
            df.to_parquet(chunk_path)
            next_chunk_idx += BATCH_SIZE
            total_trades_saved += len(trades_batch)
            return len(trades_batch)

        def mark_done(tickers: list[str]) -> None:
            if not tickers:
                return
            with open(MANIFEST_FILE, "a") as f:
                for ticker in tickers:
                    f.write(f"{ticker}\n")

        def flush_ready() -> None:
            """Flush whole (ticker, trades) groups once enough trades have piled up."""
            nonlocal pending_count
            while pending and pending_count >= BATCH_SIZE:
                batch_trades: list[dict] = []
                batch_tickers: list[str] = []
                count = 0
                while pending and count < BATCH_SIZE:
                    ticker, trades = pending.popleft()
                    batch_tickers.append(ticker)
                    batch_trades.extend(trades)
                    count += len(trades)
                pending_count -= count
                with write_lock:
                    save_batch(batch_trades)
                mark_done(batch_tickers)

        # Each worker thread reuses a single KalshiClient (and its underlying
        # httpx connection pool) instead of creating and tearing down a brand
        # new one per ticker. With millions of tickers to fetch, per-ticker
        # client churn was itself a source of slow, steadily growing memory use.
        thread_local = threading.local()
        clients: list[KalshiClient] = []
        clients_lock = threading.Lock()

        def get_client() -> KalshiClient:
            client = getattr(thread_local, "client", None)
            if client is None:
                client = KalshiClient()
                thread_local.client = client
                with clients_lock:
                    clients.append(client)
            return client

        def fetch_ticker_trades(ticker: str) -> list[dict]:
            """Fetch trades for a single ticker using this thread's cached client.

            Streams pages in via iter_market_trades and flushes directly to disk
            every BATCH_SIZE trades instead of buffering the whole ticker, since
            some tickers (e.g. high-volume World Cup markets) have 1M+ trades —
            holding even one of those fully in memory, times however many of them
            max_workers fetches concurrently, is what was OOM-killing this on the
            Pi. Only the leftover remainder (<BATCH_SIZE) is returned, to be
            merged with other tickers' leftovers via the normal pending/
            flush_ready path so ordinary (small) tickers still end up batched
            together into shared parquet files instead of one file each.
            If a ticker fails partway through, any chunks already flushed for it
            are not retroactively undone — the ticker is simply never marked
            done, so a retry re-fetches it from scratch and duplicates those
            chunks. That's the same coarse, ticker-granularity resumability
            trade-off already made elsewhere in this indexer.
            """
            client = get_client()
            fetched_at = datetime.utcnow()
            buffer: list[dict] = []
            for page in client.iter_market_trades(ticker, min_ts=self._min_ts, max_ts=self._max_ts):
                buffer.extend({**asdict(t), "_fetched_at": fetched_at} for t in page)
                while len(buffer) >= BATCH_SIZE:
                    with write_lock:
                        save_batch(buffer[:BATCH_SIZE])
                    buffer = buffer[BATCH_SIZE:]
            return buffer

        # Concurrent fetching, submitted in bounded chunks so we never hold
        # millions of pending Future objects in memory at once (was the other
        # major contributor to the OOM kill).
        SUBMIT_CHUNK_SIZE = 2000
        pbar = tqdm(total=len(tickers_to_process), desc="Fetching trades")
        try:
            with ThreadPoolExecutor(max_workers=self._max_workers) as executor:
                for chunk_start in range(0, len(tickers_to_process), SUBMIT_CHUNK_SIZE):
                    chunk = tickers_to_process[chunk_start : chunk_start + SUBMIT_CHUNK_SIZE]
                    futures = {executor.submit(fetch_ticker_trades, ticker): ticker for ticker in chunk}

                    for future in as_completed(futures):
                        ticker = futures[future]
                        try:
                            trades_data = future.result()
                            pending.append((ticker, trades_data))
                            pending_count += len(trades_data)

                            pbar.update(1)
                            pbar.set_postfix(buffer=pending_count, saved=total_trades_saved, last=ticker[-20:])

                            flush_ready()

                        except Exception as e:
                            pbar.update(1)
                            tqdm.write(f"Error fetching {ticker}: {e}")
        finally:
            pbar.close()
            for client in clients:
                client.close()

        # Save remaining
        if pending:
            batch_trades = []
            batch_tickers = []
            for ticker, trades in pending:
                batch_tickers.append(ticker)
                batch_trades.extend(trades)
            with write_lock:
                save_batch(batch_trades)
            mark_done(batch_tickers)

        print(
            f"\nBackfill trades complete: {len(tickers_to_process)} markets processed, "
            f"{total_trades_saved} trades saved"
        )
