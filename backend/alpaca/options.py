from enum import Enum
from .alpaca import Alpaca
from typing import TypedDict
from datetime import datetime
import re


class OptionType(Enum):
    CALL = "C"
    PUT = "P"


class OptionPrice(TypedDict):
    AskPrice: float
    BidPrice: float
    MidPrice: float
    Spread: float


class Option(TypedDict):
    Price: OptionPrice
    Type: OptionType
    Expiry: datetime
    ImpliedVol: float
    Volume: float
    Strike: float
    Delta: float
    Gamma: float
    Theta: float
    Vega: float


class AlpacaOptions(Alpaca):
    def __init__(self, api_key, api_secret):
        super().__init__(api_key, api_secret)

    def _parse_osi(self, osi) -> dict:
        """
        Internal function to parse OSI (Option Symbol Identifier)

        Args:
            osi (str): Option Symbol Identifier

        Returns:
            dict: Parsed OSI

        Raises:
            ValueError: If the OSI is invalid
        """
        osi_pattern = r"([A-Z]+)(\d+)([CP])(\d+)"
        match = re.match(osi_pattern, osi)

        if not match:
            raise ValueError(f"Invalid OSI: {osi}")

        expiry = match.group(2)

        parsed_expiry = datetime.strptime(expiry, "%y%m%d")

        parsed_strike = float(match.group(4)) / 1000

        return {
            "symbol": match.group(1),
            "expiry": parsed_expiry,
            "type": OptionType(match.group(3)),
            "strike": parsed_strike,
        }

    def options_quote(self, osi: str | list[str]) -> OptionPrice | list[OptionPrice]:
        """
        Fetches the quote for a single option or a list of options.

        Args:
            osi (str | list[str]): Option Symbol Identifier

        Returns:
            OptionPrice | list[OptionPrice]: Option quote

        Raises:
            Exception: If the request fails
        """
        if isinstance(osi, list):
            osi = ",".join(osi)

        option_prices: list[OptionPrice] = []

        response = self._send_request(
            "GET",
            f"https://data.alpaca.markets/v1beta1/options/quotes/latest?symbols={osi}&feed=indicative",
        )

        if response.status_code != 200:
            raise Exception(f"Error fetching options quote for {osi}: {response.text}")

        response_json = response.json()

        for osi, option in response_json["quotes"].items():
            try:
                price: OptionPrice = {
                    "AskPrice": option["ap"],
                    "BidPrice": option["bp"],
                    "Spread": option["ap"] - option["bp"],
                    "MidPrice": (option["ap"] + option["bp"]) / 2,
                }
                option_prices.append(price)
            except KeyError:
                pass

        return option_prices[0] if len(option_prices) == 1 else option_prices

    def options_chain(self, symbol, limit=1000) -> list[Option]:
        """
        Fetches the options chain for a given symbol

        Args:
            symbol (str): Symbol
            limit (int, optional): Maximum number of options to fetch. Defaults to 1000.

        Returns:
            list[Option]: List of options


        Raises:
            Exception: If the request fails
        """
        found_options: list[Option] = []
        next_page_token = None
        while len(found_options) < limit or next_page_token is not None:
            url = f"options/snapshots/{symbol}?feed=indicative&limit={1000 if limit > 1000 else limit}"
            if next_page_token is not None:
                url += f"&page_token={next_page_token}"

            response = self._send_request("GET", url)

            if response.status_code != 200:
                raise Exception(
                    f"Error fetching options chain for {symbol}: {response.text}"
                )

            response_json = response.json()

            next_page_token = response_json["next_page_token"]

            for osi, option in response_json["snapshots"].items():
                parsed_osi = self._parse_osi(osi)

                try:
                    option_price: OptionPrice = {
                        "AskPrice": option["latestQuote"]["ap"],
                        "BidPrice": option["latestQuote"]["bp"],
                        "Spread": option["latestQuote"]["ap"]
                        - option["latestQuote"]["bp"],
                        "MidPrice": (
                            option["latestQuote"]["ap"] + option["latestQuote"]["bp"]
                        )
                        / 2,
                    }

                    parsed_option: Option = {
                        "Price": option_price,
                        "Type": parsed_osi["type"],
                        "Expiry": parsed_osi["expiry"],
                        "ImpliedVol": option["impliedVolatility"],
                        "Volume": option["dailyBar"]["v"],
                        "Strike": parsed_osi["strike"],
                        "Delta": option["greeks"]["delta"],
                        "Gamma": option["greeks"]["gamma"],
                        "Theta": option["greeks"]["theta"],
                        "Vega": option["greeks"]["vega"],
                    }

                    found_options.append(parsed_option)
                except KeyError:
                    pass
            if next_page_token is None:
                break
        return found_options
