"""Single-asset mean-reversion research prototype.

Signals are calculated at the close and positions are entered on the next bar,
which avoids using information before it was available.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class StrategyConfig:
    window: int = 20
    entry_z: float = 2.0
    exit_z: float = 0.5
    allow_short: bool = True
    transaction_cost_bps: float = 5.0
    signal: str = "return_z"


def download_prices(ticker: str, start: str, end: str | None = None) -> pd.Series:
    """Download adjusted daily closes from Yahoo Finance."""
    import yfinance as yf

    frame = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if frame.empty:
        raise RuntimeError(
            f"Yahoo Finance returned no data for {ticker!r}; it may be rate-limiting requests. "
            "Wait a few minutes and retry."
        )
    close = frame["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    return close.rename("close").dropna().astype(float)


def add_indicators(close: pd.Series, window: int = 20) -> pd.DataFrame:
    """Calculate daily returns and rolling price/return z-scores."""
    if window < 2:
        raise ValueError("window must be at least 2")

    data = close.rename("close").to_frame().sort_index()
    data["return"] = data["close"].pct_change()

    price_mean = data["close"].rolling(window).mean()
    price_std = data["close"].rolling(window).std(ddof=1)
    return_mean = data["return"].rolling(window).mean()
    return_std = data["return"].rolling(window).std(ddof=1)

    data["price_mean_20"] = price_mean
    data["price_std_20"] = price_std
    data["return_mean_20"] = return_mean
    data["return_std_20"] = return_std
    data["price_z"] = (data["close"] - price_mean) / price_std.replace(0, np.nan)
    data["return_z"] = (data["return"] - return_mean) / return_std.replace(0, np.nan)
    return data


def build_positions(zscore: pd.Series, entry_z: float, exit_z: float, allow_short: bool) -> pd.Series:
    """Create stateful positions: buy oversold, short overbought, exit near zero."""
    if not 0 <= exit_z < entry_z:
        raise ValueError("thresholds must satisfy 0 <= exit_z < entry_z")

    position = 0
    output: list[int] = []
    for z in zscore:
        if pd.isna(z):
            position = 0
        elif position == 0:
            if z <= -entry_z:
                position = 1
            elif allow_short and z >= entry_z:
                position = -1
        elif position == 1 and z >= -exit_z:
            position = 0
        elif position == -1 and z <= exit_z:
            position = 0
        output.append(position)
    return pd.Series(output, index=zscore.index, name="signal_position", dtype=int)


def backtest(close: pd.Series, config: StrategyConfig = StrategyConfig()) -> pd.DataFrame:
    """Run a close-to-close backtest with next-bar execution and trading costs."""
    data = add_indicators(close, config.window)
    if config.signal not in {"return_z", "price_z"}:
        raise ValueError("signal must be 'return_z' or 'price_z'")

    data["signal_position"] = build_positions(
        data[config.signal], config.entry_z, config.exit_z, config.allow_short
    )
    data["position"] = data["signal_position"].shift(1).fillna(0).astype(int)
    data["turnover"] = data["position"].diff().abs().fillna(data["position"].abs())
    data["strategy_return_gross"] = data["position"] * data["return"]
    cost = data["turnover"] * config.transaction_cost_bps / 10_000
    data["strategy_return"] = data["strategy_return_gross"] - cost
    data["equity"] = (1 + data["strategy_return"].fillna(0)).cumprod()
    data["buy_hold_equity"] = (1 + data["return"].fillna(0)).cumprod()
    return data


def performance_summary(results: pd.DataFrame) -> dict[str, float]:
    returns = results["strategy_return"].dropna()
    if returns.empty:
        return {"total_return": 0.0, "annualized_return": 0.0, "annualized_volatility": 0.0,
                "sharpe": 0.0, "max_drawdown": 0.0, "trades": 0.0}
    years = len(returns) / 252
    total = float((1 + returns).prod() - 1)
    annual = float((1 + total) ** (1 / years) - 1) if years > 0 and total > -1 else -1.0
    volatility = float(returns.std(ddof=1) * np.sqrt(252))
    sharpe = float(returns.mean() / returns.std(ddof=1) * np.sqrt(252)) if returns.std(ddof=1) else 0.0
    equity = (1 + returns).cumprod()
    drawdown = equity / equity.cummax() - 1
    trades = float((results["turnover"] > 0).sum())
    return {"total_return": total, "annualized_return": annual,
            "annualized_volatility": volatility, "sharpe": sharpe,
            "max_drawdown": float(drawdown.min()), "trades": trades}


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest a one-asset mean-reversion strategy")
    parser.add_argument("--ticker", default="NVDA")
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end")
    parser.add_argument("--window", type=int, default=20)
    parser.add_argument("--entry-z", type=float, default=2.0)
    parser.add_argument("--exit-z", type=float, default=0.5)
    parser.add_argument("--signal", choices=["return_z", "price_z"], default="return_z")
    parser.add_argument("--long-only", action="store_true")
    parser.add_argument("--cost-bps", type=float, default=5.0)
    parser.add_argument("--output", default="nvda_backtest.csv")
    args = parser.parse_args()

    config = StrategyConfig(args.window, args.entry_z, args.exit_z, not args.long_only,
                            args.cost_bps, args.signal)
    results = backtest(download_prices(args.ticker, args.start, args.end), config)
    results.to_csv(args.output)
    latest = results.dropna(subset=[args.signal]).iloc[-1]
    summary = performance_summary(results)
    print(f"{args.ticker} latest close: {latest['close']:.2f}")
    print(f"Latest {args.signal}: {latest[args.signal]:.3f}")
    print(f"Next-bar target: {int(latest['signal_position']):+d} (-1 short, 0 flat, +1 long)")
    for key, value in summary.items():
        print(f"{key}: {value:.2%}" if key != "trades" else f"{key}: {int(value)}")
    print(f"Saved daily results to {args.output}")


if __name__ == "__main__":
    main()
