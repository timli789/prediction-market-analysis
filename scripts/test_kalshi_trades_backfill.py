"""
Runs the real Kalshi trades backfill end to end (manifest bootstrap, streaming
fetch, batch flush, parquet write) against a single currently-active ticker,
capped to roughly MAX_TRADES trades, writing to an isolated test location
instead of data/kalshi/ — so it's safe to run anytime without touching
production data or risking a repeat of the OOM crash this was built to catch.

Usage:
    uv run python scripts/test_kalshi_trades_backfill.py
"""

import resource
import shutil
import sys
from pathlib import Path
from typing import Optional

import duckdb

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import src.indexers.kalshi.trades as trades_mod
from src.indexers.kalshi.client import KalshiClient
from src.indexers.kalshi.trades import KalshiTradesIndexer

MAX_TRADES = 1000
TEST_DIR = Path("data/kalshi_test")


def pick_active_ticker(client: KalshiClient) -> str:
    """Pick whichever ticker is most active right now, so this script never
    depends on a hardcoded ticker that might have closed (and 404) by the
    time it's next run."""
    trades = client.get_recent_trades(limit=1000)
    counts: dict[str, int] = {}
    for t in trades:
        counts[t.ticker] = counts.get(t.ticker, 0) + 1
    return max(counts, key=counts.get)


def find_cutoff(client: KalshiClient, ticker: str, max_trades: int) -> Optional[int]:
    """Return the created_time (unix ts) of the max_trades'th-most-recent trade
    for ticker, or None if it has fewer than max_trades trades total (nothing
    to cap). iter_market_trades pages newest-first, so this is a floor to pass
    as min_ts — not max_ts, which would instead bound to everything OLDER than
    the cutoff (i.e. almost the entire rest of the market's history)."""
    count = 0
    for page in client.iter_market_trades(ticker):
        for t in page:
            count += 1
            if count == max_trades:
                return int(t.created_time.timestamp())
    return None


def build_markets_fixture(ticker: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    duckdb.sql(f"""
        COPY (
            SELECT '{ticker}' AS ticker, 'finalized' AS status, {MAX_TRADES} AS volume
        ) TO '{path}' (FORMAT PARQUET)
    """)


def main() -> None:
    if TEST_DIR.exists():
        shutil.rmtree(TEST_DIR)

    client = KalshiClient()
    ticker = pick_active_ticker(client)
    print(f"Testing against currently active ticker: {ticker}")

    min_ts = find_cutoff(client, ticker, MAX_TRADES)
    client.close()

    build_markets_fixture(ticker, TEST_DIR / "markets" / "markets_0_1.parquet")

    trades_mod.DATA_DIR = TEST_DIR / "trades"
    trades_mod.MARKETS_DIR = TEST_DIR / "markets"
    trades_mod.MANIFEST_FILE = TEST_DIR / ".trades_manifest"

    indexer = KalshiTradesIndexer(max_workers=1, min_ts=min_ts)
    indexer.run()

    peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    print(f"\nTest data written to: {TEST_DIR}/")
    print(f"Peak RSS: {peak_mb:.1f} MB")


if __name__ == "__main__":
    main()
