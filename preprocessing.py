"""
Preprocessing utilities for stock data.
Downloads stock data via yfinance and adds technical indicators.
"""

import yfinance as yf
import pandas as pd
import ta


def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten MultiIndex columns that yfinance sometimes returns."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def add_technical_indicators(stock_data: pd.DataFrame) -> pd.DataFrame:
    """
    Add technical indicators to stock OHLCV data and create prediction target.

    Indicators added:
        - SMA (14-day Simple Moving Average)
        - RSI (14-day Relative Strength Index)
        - OBV (On-Balance Volume)
        - ATR_14 (14-day Average True Range)
        - CCI_20 (20-day Commodity Channel Index)

    Also adds:
        - NextDayClose: next trading day's close price
        - Target: binary label (1 if next close > current close)
    """
    stock_data = flatten_columns(stock_data.copy())

    stock_data["SMA"] = ta.trend.sma_indicator(stock_data["Close"], window=14)
    stock_data["RSI"] = ta.momentum.rsi(stock_data["Close"], window=14)
    stock_data["OBV"] = ta.volume.on_balance_volume(
        stock_data["Close"], stock_data["Volume"]
    )
    stock_data["ATR_14"] = ta.volatility.average_true_range(
        stock_data["High"], stock_data["Low"], stock_data["Close"], window=14
    )
    stock_data["CCI_20"] = ta.trend.cci(
        stock_data["High"], stock_data["Low"], stock_data["Close"], window=20
    )
    stock_data["NextDayClose"] = stock_data["Close"].shift(-1)
    stock_data["Target"] = (stock_data["NextDayClose"] > stock_data["Close"]).astype(
        int
    )
    stock_data.dropna(inplace=True)
    return stock_data


def download_stock(symbol: str, start: str, end: str) -> pd.DataFrame:
    """Download stock data and add technical indicators."""
    raw = yf.download(symbol, start=start, end=end, progress=False)
    return add_technical_indicators(raw)


# ---------------------------------------------------------------------------
# Stock universe (based on NASDAQ-100 sector classification)
# ---------------------------------------------------------------------------

CONSUMER_DISCRETIONARY = ["AMZN", "CPRT", "SBUX", "PAYX", "MNST"]
CONSUMER_STAPLES = ["PEP", "KHC", "WBA", "CCEP", "MDLZ"]
HEALTH_CARE = ["AMGN", "VRTX", "ISRG", "MRNA", "ILMN"]
INDUSTRIAL = ["CSX", "BKR", "AAPL", "ROP", "HON"]
TECHNOLOGY = ["QCOM", "MSFT", "INTC", "MDB", "GOOG"]
TELECOMMUNICATIONS = ["CMCSA", "WBD", "CSCO", "TMUS", "AEP"]
UTILITIES = ["XEL", "EXC", "PCG", "SRE", "OGE"]

ALL_STOCKS = (
    CONSUMER_DISCRETIONARY
    + CONSUMER_STAPLES
    + HEALTH_CARE
    + INDUSTRIAL
    + TECHNOLOGY
    + TELECOMMUNICATIONS
    + UTILITIES
)
