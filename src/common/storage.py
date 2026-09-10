from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Union

import duckdb
import pandas as pd


class ParquetStorage:
    CHUNK_SIZE = 10000

    def __init__(self, data_dir: Union[Path, str] = "data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._buffer: list[dict] = []
        self._next_start: int | None = None
        self._existing_loaded = False
        self._existing_count = 0

        # Kalshi alone now has 70M+ market tickers on disk. Dedup used to load
        # every ticker into a Python `set` -- tens of millions of Python str
        # objects is itself several GB of interpreter overhead, which is what
        # was actually OOM-killing this on memory-constrained devices (e.g.
        # Raspberry Pi), not just slow. Keeping the dedup table inside DuckDB
        # instead lets it spill to disk under these limits rather than
        # exhausting RAM.
        duckdb.sql("PRAGMA max_memory='2GB'")
        duckdb.sql("PRAGMA threads=2")
        duckdb.sql(f"PRAGMA temp_directory='{self.data_dir.parent / '.duckdb_tmp'}'")

    def _get_market_chunks(self) -> list[Path]:
        """Get all market chunk files sorted by start index."""
        chunks = list(self.data_dir.glob("markets_*_*.parquet"))
        chunks.sort(key=lambda p: int(p.stem.split("_")[1]))
        return chunks

    def _chunk_path(self, start: int, end: int) -> Path:
        return self.data_dir / f"markets_{start}_{end}.parquet"

    def _ensure_existing_tickers_loaded(self) -> None:
        """Materialize existing tickers into a DuckDB temp table (once).

        This replaces a `SELECT DISTINCT ... -> Python set` that doesn't scale
        past tens of millions of rows. The temp table lives in DuckDB, bound
        by the memory/temp_directory pragmas set in __init__.
        """
        if self._existing_loaded:
            return
        chunks = self._get_market_chunks()
        if chunks:
            duckdb.sql(f"""
                CREATE OR REPLACE TEMP TABLE existing_tickers AS
                SELECT DISTINCT ticker FROM '{self.data_dir}/markets_*.parquet'
            """)
        else:
            duckdb.sql("CREATE OR REPLACE TEMP TABLE existing_tickers (ticker VARCHAR)")
        self._existing_count = duckdb.sql("SELECT COUNT(*) FROM existing_tickers").fetchone()[0]
        self._existing_loaded = True

    def _next_chunk_start(self) -> int:
        """Where the next new chunk file should start numbering from."""
        if self._next_start is None:
            chunks = self._get_market_chunks()
            self._next_start = max((int(p.stem.split("_")[2]) for p in chunks), default=0)
        return self._next_start

    def _write_chunk(self, df: pd.DataFrame) -> None:
        start = self._next_chunk_start()
        end = start + len(df)
        df.to_parquet(self._chunk_path(start, end))
        self._next_start = end

    def _dedup_and_flush(self, records: list[dict]) -> int:
        """Drop records whose ticker is already known, write the rest, return count added."""
        self._ensure_existing_tickers_loaded()
        candidate = pd.DataFrame(records).drop_duplicates(subset="ticker")  # noqa: F841 (referenced by name in SQL below)
        deduped = duckdb.sql("""
            SELECT c.* FROM candidate c
            ANTI JOIN existing_tickers e ON c.ticker = e.ticker
        """).df()
        if deduped.empty:
            return 0
        self._write_chunk(deduped)
        duckdb.sql("INSERT INTO existing_tickers SELECT ticker FROM deduped")
        return len(deduped)

    def append_markets(self, markets: list) -> int:
        """Buffer and persist new markets, deduplicated by ticker.

        New records are only ever written to brand-new chunk files, never
        merged into an existing one -- the previous implementation read the
        entire current chunk back into memory and rewrote it on every call,
        which turned into a severe memory/IO blowup once a chunk grew into
        the hundreds of thousands of rows. Call `flush()` once done fetching
        to persist any leftover partial batch.
        """
        fetched_at = datetime.utcnow()
        for market in markets:
            record = asdict(market)
            record["_fetched_at"] = fetched_at
            self._buffer.append(record)

        while len(self._buffer) >= self.CHUNK_SIZE:
            self._existing_count += self._dedup_and_flush(self._buffer[: self.CHUNK_SIZE])
            self._buffer = self._buffer[self.CHUNK_SIZE :]

        return self._existing_count

    def flush(self) -> None:
        """Persist any buffered records smaller than a full chunk."""
        if self._buffer:
            self._existing_count += self._dedup_and_flush(self._buffer)
            self._buffer = []
