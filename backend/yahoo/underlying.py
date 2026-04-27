import yfinance as yf


def get_dividend_yield(symbol: str) -> float:
    """
    Get the dividend yield of a stock

    Args:
        symbol: The symbol of the stock

    Returns:
        The dividend yield of the stock

    """
    ticker = yf.Ticker(symbol)
    return float(ticker.info.get("dividendYield", 0))  # Use 0 if not found


def get_price(symbol: str) -> float:
    """
    Get the price of a stock

    Args:
        symbol: The symbol of the stock

    Returns:
        The price of the stock
    """
    ticker = yf.Ticker(symbol)
    price = float(ticker.info.get("regularMarketPrice", 0))
    if price == 0:
        # Fallback to last close if regularMarketPrice is not available
        hist = ticker.history(period='1d')
        if not hist.empty:
            price = float(hist['Close'].iloc[-1])
    return price
