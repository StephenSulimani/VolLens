from datetime import datetime
from typing import TypedDict

import pandas as pd
import yfinance as yf


class OptionPrice(TypedDict):
    AskPrice: float
    BidPrice: float
    MidPrice: float
    Spread: float


class Option(TypedDict):
    Price: OptionPrice
    Type: str
    Expiry: datetime
    ImpliedVol: float
    Volume: float
    Strike: float
    Delta: float
    Gamma: float
    Theta: float
    Vega: float


class YahooOptions:
    def __init__(self):
        self._ticker_cache: dict[str, yf.Ticker] = {}

    def _ticker(self, symbol: str) -> yf.Ticker:
        if symbol not in self._ticker_cache:
            self._ticker_cache[symbol] = yf.Ticker(symbol)
        return self._ticker_cache[symbol]

    def options_chain(self, symbol: str, limit: int = 20000) -> list[Option]:
        ticker = self._ticker(symbol)
        expiries = ticker.options
        if not expiries:
            return []

        found: list[Option] = []
        for expiry in expiries:
            expiry_dt = pd.to_datetime(expiry)
            chain = ticker.option_chain(expiry)
            for side, opt_type in ((chain.calls, "C"), (chain.puts, "P")):
                if side is None or side.empty:
                    continue
                for _, row in side.iterrows():
                    bid = float(row.get("bid", 0.0) or 0.0)
                    ask = float(row.get("ask", 0.0) or 0.0)
                    last = float(row.get("lastPrice", 0.0) or 0.0)
                    iv = float(row.get("impliedVolatility", 0.0) or 0.0)
                    volume = float(row.get("volume", 0.0) or 0.0)
                    strike = float(row.get("strike", 0.0) or 0.0)

                    # When bid/ask are missing, fallback to last trade.
                    if bid > 0 and ask > 0:
                        mid = 0.5 * (bid + ask)
                        spread = max(0.0, ask - bid)
                    else:
                        mid = max(0.0, last)
                        spread = 0.0

                    option: Option = {
                        "Price": {
                            "AskPrice": ask,
                            "BidPrice": bid,
                            "MidPrice": mid,
                            "Spread": spread,
                        },
                        "Type": opt_type,
                        "Expiry": expiry_dt,
                        "ImpliedVol": iv,
                        "Volume": volume,
                        "Strike": strike,
                        "Delta": 0.0,
                        "Gamma": 0.0,
                        "Theta": 0.0,
                        "Vega": 0.0,
                    }
                    found.append(option)
                    if len(found) >= limit:
                        return found
        return found
