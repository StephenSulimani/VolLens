import pandas as pd
import numpy as np
from pysabr import Hagan2002LognormalSABR
from scipy.optimize import minimize
from pysabr.models.hagan_2002_lognormal_sabr import lognormal_vol as hagan_lognormal_vol


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
        smile = smile[
            np.isfinite(smile["forward"])
            & np.isfinite(smile["T"])
            & np.isfinite(smile["strike"])
            & np.isfinite(smile["mkt_iv"])
            & (smile["forward"] > 0)
            & (smile["T"] > 0)
            & (smile["strike"] > 0)
            & (smile["mkt_iv"] > 0)
        ]
        if len(smile) < 3:
            continue

        f = float(smile["forward"].iloc[0])
        t = float(smile["T"].iloc[0])
        smile["log_mny"] = np.abs(np.log(smile["strike"] / f))
        smile = smile[smile["log_mny"] < 0.23]  # roughly [0.79F, 1.26F]
        if len(smile) < 5:
            continue
        # Use a stable core subset around forward to reduce short-dated noise.
        smile = smile.sort_values("log_mny").head(min(len(smile), 11)).copy()
        # SABR needs a two-sided smile around forward; one-sided data collapses to bounds.
        below = int((smile["strike"] < f).sum())
        above = int((smile["strike"] > f).sum())
        if below < 2 or above < 2:
            continue

        strikes = smile["strike"].to_numpy()
        market_vols = smile["mkt_iv"].to_numpy()

        atm_idx = np.argmin(np.abs(strikes - f))
        atm_vol = market_vols[atm_idx]
        alpha_anchor = max(1e-4, atm_vol * (f ** (1.0 - beta)))

        weights = np.sqrt(smile["volume"].to_numpy() + 1.0)
        moneyness = np.abs(np.log(np.maximum(strikes, 1e-8) / f))
        core_weights = np.exp(-3.0 * moneyness)
        weights = weights * core_weights

        def objective(
            params,
            f=f,
            t=t,
            strikes=strikes,
            market_vols=market_vols,
            weights=weights,
            alpha_anchor=alpha_anchor,
        ):
            alpha, rho, volvol = params

            if alpha <= 0 or not (-1 < rho < 1) or volvol <= 0:
                return 1e10

            model_vols = np.array(
                [hagan_lognormal_vol(float(k), f, t, alpha, beta, rho, volvol) for k in strikes]
            )
            if np.any(~np.isfinite(model_vols)):
                return 1e10

            fit_error = np.sum(weights * (model_vols - market_vols) ** 2)
            # Anchor alpha to ATM-implied scale to avoid degenerate boundary fits.
            alpha_penalty = 0.15 * ((alpha - alpha_anchor) / max(alpha_anchor, 1e-6)) ** 2
            volvol_penalty = 0.08 * (volvol**2)
            rho_penalty = 0.01 * (rho**2)
            boundary_barrier = 0.0
            if volvol < 0.05:
                boundary_barrier += (0.05 - volvol) * 2.0
            if volvol > 1.00:
                boundary_barrier += (volvol - 1.00) * 4.0
            if abs(rho) > 0.90:
                boundary_barrier += (abs(rho) - 0.90) * 6.0
            nu_t = volvol * np.sqrt(max(t, 1e-6))
            if nu_t > 0.75:
                boundary_barrier += 8.0 * (nu_t - 0.75) ** 2
            return fit_error + alpha_penalty + volvol_penalty + rho_penalty + boundary_barrier

        # First pass: use pysabr native fit (alpha, rho, volvol), then bounded polish.
        fit_guesses = [
            [max(0.01, atm_vol), -0.30, 0.35],
            [max(0.01, atm_vol), -0.60, 0.80],
            [max(0.01, atm_vol), 0.00, 0.25],
            [max(0.01, atm_vol), 0.30, 0.60],
        ]
        best_tuple = None
        best_err = np.inf
        for guess in fit_guesses:
            try:
                sabr_fit = Hagan2002LognormalSABR(
                    f=f, t=t, shift=0, beta=beta, v_atm_n=atm_vol, rho=0.0, volvol=0.2
                )
                alpha, rho_fit, volvol_fit = sabr_fit.fit(
                    strikes.astype(float), market_vols.astype(float) * 100.0, initial_guess=guess
                )
                if not np.isfinite(alpha) or not np.isfinite(rho_fit) or not np.isfinite(volvol_fit):
                    continue
                # Re-score on our weighted objective.
                err = objective([float(rho_fit), float(volvol_fit)])
                if np.isfinite(err) and err < best_err:
                    best_err = err
                    best_tuple = (float(alpha), float(rho_fit), float(volvol_fit))
            except Exception:
                continue

        bounds = [(1e-4, 25.0), (-0.999, 0.999), (0.02, 1.2)]
        if best_tuple is None:
            # Fallback to full alpha/rho/volvol optimization if fit fails.
            starts = [
                np.array([alpha_anchor, -0.3, 0.4]),
                np.array([alpha_anchor, -0.6, 0.8]),
                np.array([alpha_anchor, 0.0, 0.25]),
                np.array([alpha_anchor, 0.3, 0.6]),
            ]
            best_res = None
            for initial_guess in starts:
                res = minimize(
                    objective,
                    initial_guess,
                    method="L-BFGS-B",
                    bounds=bounds,
                    options={"maxiter": 120, "ftol": 1e-10},
                )
                if best_res is None or res.fun < best_res.fun:
                    best_res = res
            if best_res is None or (not best_res.success):
                print(f"Calibration failed for {expiry}")
                continue
            alpha, rho, volvol = best_res.x
            final_obj = float(best_res.fun)
        else:
            alpha, rho, volvol = best_tuple
            alpha = float(np.clip(alpha, bounds[0][0], bounds[0][1]))
            rho = float(np.clip(rho, bounds[1][0], bounds[1][1]))
            volvol = float(np.clip(volvol, bounds[2][0], bounds[2][1]))
            refine = minimize(
                objective,
                np.array([alpha, rho, volvol]),
                method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": 100, "ftol": 1e-10},
            )
            if refine.success and np.isfinite(refine.fun):
                alpha, rho, volvol = float(refine.x[0]), float(refine.x[1]), float(refine.x[2])
                final_obj = float(refine.fun)
            else:
                final_obj = float(best_err)

        if np.isfinite(alpha) and np.isfinite(rho) and np.isfinite(volvol):
            boundary_hit = (
                np.isclose(volvol, bounds[2][1], atol=1e-3)
                or np.isclose(volvol, bounds[2][0], atol=1e-4)
                or abs(rho) > 0.97
            )
            model_vols = np.array(
                [hagan_lognormal_vol(float(k), f, t, alpha, beta, rho, volvol) for k in strikes]
            )
            rmse = float(np.sqrt(np.mean((model_vols - market_vols) ** 2)))
            calibrated_params[expiry] = {
                "rho": float(rho),
                "volvol": float(volvol),
                "atm_vol": float(atm_vol),
                "alpha": float(alpha),
                "status": "boundary_hit" if boundary_hit else "ok",
                "rmse": rmse,
                "objective": final_obj,
            }

    return calibrated_params


def get_theoretical_smile(f, t, atm_vol, rho, volvol, beta=0.5, n_points=50, alpha=None):
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

    if alpha is not None and np.isfinite(alpha) and alpha > 0:
        theoretical_vols = [
            hagan_lognormal_vol(float(k), f, t, float(alpha), beta, rho, volvol)
            for k in strikes
        ]
    else:
        # Fallback for backward compatibility when alpha is unavailable.
        sabr = Hagan2002LognormalSABR(
            f=f, shift=0, t=t, v_atm_n=atm_vol, beta=beta, rho=rho, volvol=volvol
        )
        theoretical_vols = [sabr.lognormal_vol(float(k)) for k in strikes]

    smile = []
    for k, v in zip(strikes, theoretical_vols):
        if np.isfinite(v) and v > 0:
            smile.append({"strike": float(k), "vol": float(v)})

    return smile
