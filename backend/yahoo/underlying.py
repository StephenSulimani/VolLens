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
    return float(ticker.info.get("currentPrice", 0))
