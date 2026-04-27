from __future__ import annotations

import numpy as np
import pandas as pd
from pysabr import Hagan2002LognormalSABR
from pysabr.models.hagan_2002_lognormal_sabr import lognormal_vol as hagan_lognormal_vol


def calculate_sabr_vols(
    df: pd.DataFrame,
    calibrated_params: dict,
    beta: float = 0.5,
) -> np.ndarray:
    """Compute SABR model IV for each row in df using expiry-level params."""
    model_vols = np.full(len(df), np.nan, dtype=float)

    for i, (_, row) in enumerate(df.iterrows()):
        expiry = row["expiry_date"]
        params = calibrated_params.get(expiry)
        if not params:
            continue

        f = float(row["forward"])
        t = float(row["T"])
        k = float(row["strike"])
        rho = float(params["rho"])
        volvol = float(params["volvol"])
        alpha = params.get("alpha")

        try:
            if alpha is not None and np.isfinite(alpha) and alpha > 0:
                vol = hagan_lognormal_vol(k, f, t, float(alpha), beta, rho, volvol)
            else:
                sabr = Hagan2002LognormalSABR(
                    f=f,
                    shift=0,
                    t=t,
                    v_atm_n=float(params["atm_vol"]),
                    beta=beta,
                    rho=rho,
                    volvol=volvol,
                )
                vol = sabr.lognormal_vol(k)
            if np.isfinite(vol) and vol > 0:
                model_vols[i] = float(vol)
        except (RuntimeError, OverflowError, ZeroDivisionError, ValueError):
            continue

    return model_vols


def find_vol_arbitrage_opportunities(
    df: pd.DataFrame,
    model_vols: np.ndarray,
    model_name: str,
    min_abs_spread: float = 0.01,
    zscore_threshold: float = 1.25,
    model_quality_by_expiry: dict | None = None,
    max_fit_rmse: float | None = None,
    allowed_statuses: set[str] | None = None,
    min_volume: float = 10.0,
    max_spread_pct: float = 0.12,
) -> pd.DataFrame:
    """
    Build a ranked discrepancy table from market IV and model IV.

    Positive spread means model vol > market vol (market may be cheap vol).
    Negative spread means model vol < market vol (market may be rich vol).
    """
    if len(df) != len(model_vols):
        raise ValueError("Length mismatch between df and model_vols.")

    out = df.copy().reset_index(drop=True)
    out["model_name"] = model_name
    out["model_iv"] = model_vols.astype(float)
    out["iv_spread"] = out["model_iv"] - out["mkt_iv"]
    out["abs_iv_spread"] = np.abs(out["iv_spread"])

    valid = np.isfinite(out["model_iv"]) & (out["model_iv"] > 0)
    out = out[valid].copy()
    if out.empty:
        return out

    # Liquidity gates
    if "volume" in out.columns:
        out = out[out["volume"] >= min_volume].copy()
    if "spread_pct" in out.columns:
        out = out[out["spread_pct"] <= max_spread_pct].copy()
    if out.empty:
        return out

    # Optional per-expiry model quality gates.
    if model_quality_by_expiry is not None:
        quality = pd.Series(model_quality_by_expiry, name="quality")
        out = out.merge(quality, left_on="expiry_date", right_index=True, how="left")
        out = out[out["quality"].notna()].copy()
        if out.empty:
            return out
        out["fit_rmse"] = out["quality"].apply(
            lambda q: q.get("rmse") if isinstance(q, dict) else np.nan
        )
        out["fit_status"] = out["quality"].apply(
            lambda q: q.get("status") if isinstance(q, dict) else None
        )

        if max_fit_rmse is not None:
            out = out[
                np.isfinite(out["fit_rmse"]) & (out["fit_rmse"] <= max_fit_rmse)
            ].copy()
        if allowed_statuses is not None:
            out = out[out["fit_status"].isin(allowed_statuses)].copy()
        out = out.drop(columns=["quality"])
        if out.empty:
            return out

    # Normalize spread by expiry to compare opportunities across maturities.
    out["spread_zscore"] = out.groupby("expiry_date")["iv_spread"].transform(
        lambda s: (s - s.mean()) / max(s.std(ddof=0), 1e-6)
    )
    out["vol_arb_side"] = np.where(
        out["iv_spread"] > 0, "buy_vol", "sell_vol"
    )
    out["signal_strength"] = np.sqrt(
        np.maximum(out["abs_iv_spread"], 0.0) * np.maximum(np.abs(out["spread_zscore"]), 0.0)
    )

    out = out[
        (out["abs_iv_spread"] >= min_abs_spread)
        & (np.abs(out["spread_zscore"]) >= zscore_threshold)
    ].copy()

    columns = [
        "model_name",
        "expiry_date",
        "T",
        "strike",
        "is_call",
        "mkt_iv",
        "model_iv",
        "iv_spread",
        "abs_iv_spread",
        "spread_zscore",
        "signal_strength",
        "vol_arb_side",
        "volume",
        "fit_rmse",
        "fit_status",
    ]
    columns = [c for c in columns if c in out.columns]
    return out[columns].sort_values(
        ["signal_strength", "abs_iv_spread", "volume"],
        ascending=[False, False, False],
    )
