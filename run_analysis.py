"""Download public samples if needed, then produce evaluation CSVs."""
from pathlib import Path
import json
from urllib.request import urlretrieve
from zipfile import ZipFile
import pandas as pd
from execution_sim.core import TICKERS, SOURCE, run_one_ticker

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "raw"
RESULTS = ROOT / "results"


def main() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    frames, metadata = [], []
    for ticker in TICKERS:
        path = DATA / f"{ticker}_1min_firstratedata.csv"
        if not path.exists():
            archive = DATA / f"{ticker}.zip"
            urlretrieve(SOURCE.format(TICKER=ticker), archive)
            with ZipFile(archive) as zf:
                path.write_bytes(zf.read(f"{ticker}_1min_firstratedata.csv"))
            archive.unlink()
        result, details = run_one_ticker(path, ticker)
        frames.append(result)
        metadata.append(details)
    raw = pd.concat(frames, ignore_index=True)
    raw.to_csv(RESULTS / "trials.csv", index=False)
    summary = raw.groupby(["ticker", "method", "order_fraction", "participation_limit", "cost_scenario"], as_index=False).agg(
        sessions=("date", "nunique"),
        mean_completion=("completion_rate", "mean"),
        mean_shortfall_bps=("implementation_shortfall_bps", "mean"),
        mean_vwap_slippage_bps=("vwap_slippage_bps", "mean"),
    )
    summary.to_csv(RESULTS / "summary.csv", index=False)
    (RESULTS / "sample_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    print(f"\nSaved {len(raw)} evaluations across {len(metadata)} equities to {RESULTS}")


if __name__ == "__main__":
    main()
