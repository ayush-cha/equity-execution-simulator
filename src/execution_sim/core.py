"""Transparent, bar-based equity execution simulation for educational research."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd

TICKERS = ("AAPL", "MSFT", "AMZN", "META", "TSLA")
METHODS = ("TWAP", "VWAP", "POV")
PARTICIPATION_LIMITS = (0.05, 0.10, 0.20)
ORDER_FRACTIONS = (0.01, 0.03, 0.05)
COST_SCENARIOS = {"low": (1.0, 5.0), "base": (2.0, 10.0), "high": (4.0, 20.0)}
SOURCE = "https://frd001.s3-us-east-2.amazonaws.com/{TICKER}_1min_sample_firstratedata.zip"


@dataclass(frozen=True)
class Assumptions:
    half_spread_bps: float = 2.0
    impact_bps_at_full_participation: float = 10.0
    min_bars_per_day: int = 300


def load_sessions(path: Path, min_bars_per_day: int = 300) -> list[pd.DataFrame]:
    """Return regular-hours sessions, rejecting malformed and sparse days."""
    df = pd.read_csv(path, usecols=["timestamp", "open", "high", "low", "close", "volume"])
    # FirstRate timestamps are New York local time, including daylight-saving changes.
    df["datetime"] = pd.to_datetime(df["timestamp"], errors="raise").dt.tz_localize(
        "America/New_York", ambiguous="NaT", nonexistent="NaT")
    df = df.dropna(subset=["datetime"])
    df = df.sort_values("datetime").drop_duplicates("datetime", keep="last")
    df["local"] = df["datetime"]
    mins = df["local"].dt.hour * 60 + df["local"].dt.minute
    df = df[(mins >= 570) & (mins < 960)].copy()
    df["minute"] = mins.loc[df.index].to_numpy() - 570
    valid = (df[["open", "high", "low", "close"]] > 0).all(axis=1)
    valid &= (df["volume"] > 0) & (df["low"] <= df["high"])
    valid &= (df["low"] <= df["open"]) & (df["open"] <= df["high"])
    valid &= (df["low"] <= df["close"]) & (df["close"] <= df["high"])
    df = df[valid].copy()
    df["date"] = df["local"].dt.date
    sessions = []
    for _, day in df.groupby("date", sort=True):
        if len(day) >= min_bars_per_day and day["minute"].is_unique:
            sessions.append(day.reset_index(drop=True))
    if len(sessions) < 6:
        raise ValueError(f"Need at least six clean sessions; found {len(sessions)} in {path}")
    return sessions


def split_sessions(sessions: list[pd.DataFrame]) -> tuple[list[pd.DataFrame], list[pd.DataFrame]]:
    cut = max(3, int(len(sessions) * 0.6))
    return sessions[:cut], sessions[cut:]


def volume_profile(training: list[pd.DataFrame]) -> np.ndarray:
    """Mean minute share of daily volume, estimated only on earlier sessions."""
    curves = []
    for day in training:
        profile = np.zeros(390, dtype=float)
        vol = day["volume"].to_numpy(dtype=float)
        profile[day["minute"].to_numpy(dtype=int)] = vol / vol.sum()
        curves.append(profile)
    result = np.mean(curves, axis=0)
    return result / result.sum()


def simulate(day: pd.DataFrame, method: str, order_shares: float,
             participation_limit: float, profile: np.ndarray,
             assumptions: Assumptions = Assumptions()) -> dict:
    """Simulate a buy order. No future test-session volume enters TWAP/VWAP targets."""
    if method not in METHODS:
        raise ValueError(method)
    if order_shares <= 0 or not 0 < participation_limit <= 1:
        raise ValueError("Order shares and participation limit must be positive")
    vol = day["volume"].to_numpy(dtype=float)
    close = day["close"].to_numpy(dtype=float)
    minute = day["minute"].to_numpy(dtype=int)
    arrival = float(day["open"].iloc[0])
    market_vwap = float(np.dot(vol, close) / vol.sum())
    cumulative_profile = np.cumsum(profile)
    filled = 0.0
    spent = 0.0
    for i, (v, price, m) in enumerate(zip(vol, close, minute)):
        if method == "TWAP":
            target = order_shares * (m + 1) / 390
        elif method == "VWAP":
            target = order_shares * cumulative_profile[m]
        else:
            target = filled + participation_limit * v
        requested = max(0.0, target - filled)
        qty = min(requested, participation_limit * v, order_shares - filled)
        if qty <= 0:
            continue
        participation = qty / v
        cost_bps = (assumptions.half_spread_bps
                    + assumptions.impact_bps_at_full_participation * np.sqrt(participation))
        spent += qty * price * (1 + cost_bps / 10_000)
        filled += qty
    unfilled = max(0.0, order_shares - filled)
    avg_price = spent / filled if filled > 0 else np.nan
    # Unfilled quantity is marked at the final close, representing opportunity cost.
    is_bps = (spent + unfilled * close[-1] - order_shares * arrival) / (order_shares * arrival) * 10_000
    slippage_bps = (avg_price / market_vwap - 1) * 10_000 if filled > 0 else np.nan
    return {"date": str(day["date"].iloc[0]), "method": method,
            "participation_limit": participation_limit, "order_shares": order_shares,
            "filled_shares": filled, "completion_rate": filled / order_shares,
            "arrival_price": arrival, "market_vwap": market_vwap,
            "average_fill_price": avg_price, "implementation_shortfall_bps": is_bps,
            "vwap_slippage_bps": slippage_bps}


def run_one_ticker(path: Path, ticker: str, assumptions: Assumptions = Assumptions()) -> tuple[pd.DataFrame, dict]:
    sessions = load_sessions(path, assumptions.min_bars_per_day)[-50:]
    training, evaluation = split_sessions(sessions)
    profile = volume_profile(training)
    median_training_volume = float(np.median([d["volume"].sum() for d in training]))
    records = []
    for day in evaluation:
        for order_fraction in ORDER_FRACTIONS:
            size = median_training_volume * order_fraction
            for limit in PARTICIPATION_LIMITS:
                for method in METHODS:
                    for name, (half_spread, impact) in COST_SCENARIOS.items():
                        scenario = Assumptions(half_spread, impact, assumptions.min_bars_per_day)
                        result = simulate(day, method, size, limit, profile, scenario)
                        result.update(ticker=ticker, order_fraction=order_fraction, cost_scenario=name)
                        records.append(result)
    metadata = {"ticker": ticker, "train_sessions": len(training),
                "evaluation_sessions": len(evaluation),
                "train_start": str(training[0]["date"].iloc[0]),
                "train_end": str(training[-1]["date"].iloc[0]),
                "evaluation_start": str(evaluation[0]["date"].iloc[0]),
                "evaluation_end": str(evaluation[-1]["date"].iloc[0]),
                "median_training_volume": median_training_volume}
    return pd.DataFrame.from_records(records), metadata
