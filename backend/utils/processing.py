from datetime import datetime
import numpy as np
import pandas as pd
from typing import Mapping, Any


def process_options_chain(
    raw_data: list[Mapping[str, Any]], spot_price: float, r: float, q: float
) -> pd.DataFrame:
    """
    Processes the options chain and creates a dataframe

    Args:
        raw_data (list[Mapping[str, Any]]): Raw options data from a provider
        spot_price (float): The current price of a given stock
        r (float): The risk-free rate
        q (float): The divident yield for a given stock

    Returns:

    """
    df_columns = [
        "expiry_date",
        "strike",
        "T",
        "forward",
        "mid_price",
        "mkt_iv",
        "volume",
        "is_call",
        "spread_pct",
    ]
    df_objs = []

    for option in raw_data:
        # Time to maturity in years
        expiry_dt = pd.to_datetime(option["Expiry"])
        T = (expiry_dt - datetime.now()).days / 365.0

        # Financial Constants
        forward = spot_price * np.exp((r - q) * T)
        mid = option["Price"]["MidPrice"]

        # Avoid division by zero
        spread_pct = option["Price"]["Spread"] / mid if mid > 0 else 0

        df_objs.append(
            [
                expiry_dt,
                option["Strike"],
                T,
                forward,
                mid,
                option["ImpliedVol"],
                option["Volume"],
                True if option["Type"] == "C" else False,
                spread_pct,
            ]
        )

    df = pd.DataFrame(df_objs, columns=df_columns)

    # --- ENHANCED FILTERS ---

    # 1. Hard Volatility Filter: Remove negative or extremely low IV
    df = df[df["mkt_iv"] > 0.005]  # 0.5% minimum IV
    df = df[df["mid_price"] > 0.05]

    # 2. Spread Filter: High spreads = unreliable IVs (Arb signal noise)
    # Exclude if the bid-ask spread is > 15% of the mid price
    df = df[df["spread_pct"] < 0.15]

    # 3. Moneyness Filter: Tighten the range for stability
    # Deep OTM options have "gamma" issues that break the Hagan approximation
    df = df[(df["strike"] > spot_price * 0.75) & (df["strike"] < spot_price * 1.25)]

    # 4. Maturity Filter: Remove noise from options expiring too soon
    df = df[df["T"] > (7 / 365)]

    # 5. Keep one high-quality quote per strike/expiry (avoid call+put double-counting).
    df["quality_rank"] = np.sqrt(df["volume"] + 1.0) / np.maximum(df["spread_pct"], 0.005)
    df = (
        df.sort_values(["expiry_date", "strike", "quality_rank"], ascending=[True, True, False])
        .groupby(["expiry_date", "strike"], as_index=False)
        .first()
    )
    df = df.drop(columns=["quality_rank"])

    # 6. Group Validity Filter: Ensure we have enough points to fit a smile
    # A SABR smile needs at least 5 points to be meaningful
    df = df.groupby("expiry_date").filter(lambda x: len(x) >= 5)

    return df
