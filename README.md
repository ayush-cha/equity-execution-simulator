# Equity execution simulator

A reproducible research project comparing three ways to execute a **buy** order: time-weighted average price (TWAP), a historical-volume-curve VWAP schedule, and participation of volume (POV). It evaluates five liquid US equities (AAPL, MSFT, AMZN, META, TSLA), three participation caps (5%, 10%, 20%) and three order sizes (1%, 3%, 5% of the median training-day volume).

## Run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
PYTHONPATH=src python run_analysis.py
PYTHONPATH=src python -m unittest discover -s tests -v
```

The script downloads publicly available one-minute OHLCV samples to `data/raw/` on the first run. It saves individual evaluations, grouped results and date ranges under `results/`. Raw samples are excluded from this repository.

## Design

- Parse New York local timestamps and keep regular trading hours (09:30–15:59). Drop duplicate timestamps, invalid prices/volumes and days with fewer than 300 observed minutes.
- Use the latest 50 clean sessions per ticker, split chronologically: first 60% for training, last 40% for evaluation. The VWAP curve and reference daily volume come **only** from training sessions.
- TWAP seeks a uniform cumulative fill; VWAP uses the earlier sessions' mean minute-volume curve; POV targets the specified fraction of volume observed during each bar. Every method has the same participation cap.
- Model a buy fill at the minute close plus a half-spread and square-root market impact. Low/base/high assumptions use 1/2/4 bps half-spread and 5/10/20 bps impact at full participation. These are illustrative assumptions, not measured transaction costs. The OHLCV file has no bid/ask quotes or order book.
- Report completion rate, average fill price, implementation shortfall against the day's first open, and slippage against the full-day market VWAP. Shortfall marks unfilled shares at the final close. The full-day VWAP is an **ex-post benchmark**, not information supplied to the schedules.

This is a bar-level simulation, not an executable trading system. A bar close is a price proxy and may not be available for the simulated order. The model omits queue position, spread variation, fees, hidden liquidity, and the feedback of an order on future prices. Compare methods within the same assumptions; do not treat small differences as forecasts of live performance.

## Data source

The data come from [FirstRate Data's free intraday samples](https://firstratedata.com/free-intraday-data) for AAPL, MSFT, AMZN, META and TSLA. The provider states stock and ETF volume is measured in shares. These samples contain 2022–2023 observations. Data are downloaded at run time and are not redistributed here. Review the provider's terms before using them outside personal research.

## Sample run

On the 2023 sample, the latest 50 clean sessions per ticker yielded 30 training and 20 evaluation sessions for each of five equities: 100 distinct evaluation sessions and 8,100 method/size/cap/cost scenario evaluations. The base-cost means across all sizes and caps were:

| Method | Mean completion | Mean implementation shortfall (bps) | Mean VWAP slippage (bps) |
|---|---:|---:|---:|
| TWAP | 0.99 | -0.70 | 4.63 |
| Historical-curve VWAP | 0.99 | -0.44 | 5.04 |
| POV | 0.99 | 7.92 | 14.07 |

These are descriptive results from a fixed-cost model on this particular sample. A negative shortfall can occur when prices fall after the arrival benchmark; it does not prove a strategy predicts prices. The [full grouped output](results/summary.csv) shows ticker, order-size, cap and cost-scenario detail. `sample_metadata.json` records the exact training/evaluation date ranges.
