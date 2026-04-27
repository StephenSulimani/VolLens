import pandas as pd
import numpy as np
from pysabr import Hagan2002LognormalSABR
from scipy.optimize import minimize


def calibrate_sabr(df: pd.DataFrame, beta=0.5):
    """
    Calibrates the SABR model parameters for each unique expiration date in the dataset.

    This function iterates through the provided DataFrame, grouping by maturity. For each
    expiry, it utilizes the At-the-Money (ATM) implied volatility to anchor the model and
    optimizes the Rho (correlation) and VolVol (nu) parameters to minimize the sum of
    squared errors between the SABR-predicted volatilities and market-observed volatilities.

    Args:
        df (pd.DataFrame): A cleaned DataFrame containing the processed options chain.
                           Must include 'expiry_date', 'forward', 'T', 'strike',
                           and 'mkt_iv' columns.
        beta (float, optional): The CEV exponent parameter, typically fixed between 0 and 1.
                                Defaults to 0.5 for equity markets to represent the
                                leverage effect.

    Returns:
        dict: A dictionary where keys are expiration dates and values are dictionaries
              containing the calibrated 'rho', 'volvol', and 'atm_vol' for that maturity.
    """
    calibrated_params = {}

    for expiry, smile in df.groupby("expiry_date"):
        f = smile["forward"].iloc[0]
        t = smile["T"].iloc[0]
        strikes = smile["strike"].values
        market_vols = smile["mkt_iv"].values

        atm_idx = np.argmin(np.abs(strikes - f))
        atm_vol = market_vols[atm_idx]

        def objective(params):
            rho, volvol = params

            if not (-1 < rho < 1) or volvol <= 0:
                return 1e10

            sabr = Hagan2002LognormalSABR(
                f=f, t=t, shift=0, beta=beta, v_atm_n=atm_vol, rho=rho, volvol=volvol
            )
            model_vols = np.array([sabr.lognormal_vol(k) for k in strikes])

            weights = np.sqrt(smile["volume"].values)

            return np.sum(weights * (model_vols - market_vols) ** 2)

        initial_guess = [-0.5, 0.2]

        res = minimize(
            objective,
            initial_guess,
            method="L-BFGS-B",
            bounds=[(-0.999, 0.999), (0.001, 2.5)],
        )

        if res.success:
            rho, volvol = res.x
            calibrated_params[expiry] = {
                "rho": rho,
                "volvol": volvol,
                "atm_vol": atm_vol,
            }
        else:
            print(f"Calibration failed for {expiry}")

    return calibrated_params


def get_theoretical_smile(f, t, atm_vol, rho, volvol, beta=0.5, n_points=50):
    """
    Generates a smooth array of strikes and corresponding SABR implied volatilities for visualization.

    This function creates a range of strike prices around the current forward price and
    calculates the theoretical lognormal (Black) implied volatility for each strike using
    the calibrated SABR parameters. This is used to render the continuous "smile" curve
    on the volatility surface.

    Args:
        f (float): The forward price of the underlying asset for the given expiry.
        t (float): Time to maturity in years.
        atm_vol (float): The At-the-Money (ATM) implied volatility used to imply the SABR alpha.
        rho (float): The calibrated correlation parameter between the asset price and volatility.
        volvol (float): The calibrated "volatility of volatility" parameter (nu).
        beta (float, optional): The exponent for the forward price. Defaults to 0.5.
        n_points (int, optional): The number of points to generate for the smooth curve. Defaults to 50.

    Returns:
        list[dict]: A list of dictionaries containing 'strike' and 'vol' keys,
                    formatted for easy JSON serialization and frontend plotting.
    """
    strikes = np.linspace(f * 0.8, f * 1.2, n_points)

    sabr = Hagan2002LognormalSABR(
        f=f, shift=0, t=t, v_atm_n=atm_vol, beta=beta, rho=rho, volvol=volvol
    )

    theoretical_vols = [sabr.lognormal_vol(k) for k in strikes]

    return [{"strike": k, "vol": v} for k, v in zip(strikes, theoretical_vols)]
