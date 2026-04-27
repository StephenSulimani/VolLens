import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.optimize import minimize
from .black_scholes import implied_vol, black_scholes_price


def heston_characteristic_function(u, T, r, q, S0, v0, kappa, theta, sigma, rho):
    """The 'Little Heston Trap' stable formulation (Albrecher et al., 2007)."""
    # Use float64 for better precision
    x = np.log(S0)

    # Define intermediate terms
    # We use the (kappa - rho*sigma*u*1j) form for stability
    d = np.sqrt((rho * sigma * u * 1j - kappa) ** 2 + sigma**2 * (u * 1j + u**2))
    g = (kappa - rho * sigma * u * 1j - d) / (kappa - rho * sigma * u * 1j + d)

    # Stable formulation avoiding branch cuts
    exp_dt = np.exp(-d * T)

    C = (r - q) * u * 1j * T + (kappa * theta / sigma**2) * (
        (kappa - rho * sigma * u * 1j - d) * T - 2 * np.log((1 - g * exp_dt) / (1 - g))
    )
    D = ((kappa - rho * sigma * u * 1j - d) / sigma**2) * (
        (1 - exp_dt) / (1 - g * exp_dt)
    )

    return np.exp(C + D * v0 + 1j * u * x)


def heston_price(
    S0,
    K,
    T,
    r,
    q,
    v0,
    kappa,
    theta,
    sigma,
    rho,
    option_type="C",
    u_max=300,
    quad_limit=120,
    epsabs=1e-7,
    epsrel=1e-7,
):
    """Prices an option using the standard Heston P1/P2 formulation."""
    # Near-expiry options are better handled by discounted intrinsic value.
    if T <= 1e-8:
        if option_type == "P":
            return max(1e-8, K - S0)
        return max(1e-8, S0 - K)

    lnK = np.log(K)
    cf_minus_i = heston_characteristic_function(
        -1j, T, r, q, S0, v0, kappa, theta, sigma, rho
    )
    if not np.isfinite(cf_minus_i):
        cf_minus_i = 1.0 + 0j

    def p2_integrand(u):
        iu = 1j * u
        cf_u = heston_characteristic_function(
            u, T, r, q, S0, v0, kappa, theta, sigma, rho
        )
        val = np.exp(-1j * u * lnK) * cf_u / iu
        return np.real(val)

    def p1_integrand(u):
        iu = 1j * u
        cf_u = heston_characteristic_function(
            u - 1j, T, r, q, S0, v0, kappa, theta, sigma, rho
        )
        val = np.exp(-1j * u * lnK) * cf_u / (iu * cf_minus_i)
        return np.real(val)

    p1_int, _ = quad(
        p1_integrand,
        1e-8,
        u_max,
        limit=quad_limit,
        epsabs=epsabs,
        epsrel=epsrel,
    )
    p2_int, _ = quad(
        p2_integrand,
        1e-8,
        u_max,
        limit=quad_limit,
        epsabs=epsabs,
        epsrel=epsrel,
    )
    p1 = 0.5 + (1.0 / np.pi) * p1_int
    p2 = 0.5 + (1.0 / np.pi) * p2_int

    call_price = S0 * np.exp(-q * T) * p1 - K * np.exp(-r * T) * p2
    price = call_price if option_type == "C" else (
        call_price - S0 * np.exp(-q * T) + K * np.exp(-r * T)
    )

    # Enforce no-arbitrage bounds for numerical robustness.
    if option_type == "P":
        lower = max(0.0, K * np.exp(-r * T) - S0 * np.exp(-q * T))
        upper = K * np.exp(-r * T)
    else:
        lower = max(0.0, S0 * np.exp(-q * T) - K * np.exp(-r * T))
        upper = S0 * np.exp(-q * T)

    return float(np.clip(price, max(1e-8, lower), upper))


def calculate_heston_vols(
    df: pd.DataFrame,
    spot_price: float,
    r: float,
    q: float,
    params: np.ndarray,
    debug=False,
):
    """Vectorized calculation of Heston implied volatilities."""
    kappa, theta, sigma, rho, v0 = params
    model_vols = []

    for _, row in df.iterrows():
        h_price = heston_price(
            S0=spot_price,
            K=row["strike"],
            T=row["T"],
            r=r,
            q=q,
            v0=v0,
            kappa=kappa,
            theta=theta,
            sigma=sigma,
            rho=rho,
            option_type="C" if row["is_call"] else "P",
        )

        if debug:
            print(f"Strike: {row['strike']}, Price: {h_price:.6f}")

        try:
            iv = implied_vol(
                h_price, spot_price, row["strike"], row["T"], r, q, row["is_call"]
            )
            model_vols.append(iv)
        except Exception:
            model_vols.append(0.0)

    return np.array(model_vols)


def calibrate_heston(df: pd.DataFrame, spot_price: float, r: float, q: float):
    # CRITICAL: Filter for 'The Meat' (0.85 to 1.15 moneyness)
    # This prevents the optimizer from getting stuck on zero-price tails
    df_core = df[
        (df["strike"] > spot_price * 0.85) & (df["strike"] < spot_price * 1.15)
    ]

    # Fallback if the core is too small
    if len(df_core) < 20:
        df_core = df

    # Favor liquid and near-ATM contracts for more stable calibration.
    df_core = df_core.copy()
    df_core["moneyness"] = np.abs(np.log(df_core["strike"] / spot_price))
    df_core["liq_rank"] = np.sqrt(df_core["volume"] + 1.0) * np.exp(-2.5 * df_core["moneyness"])
    df_sampled = df_core.nlargest(min(len(df_core), 36), "liq_rank").copy()
    df_sampled["option_type"] = np.where(df_sampled["is_call"], "C", "P")
    df_sampled["mkt_price"] = df_sampled.apply(
        lambda row: black_scholes_price(
            spot_price,
            row["strike"],
            row["T"],
            r,
            q,
            row["mkt_iv"],
            row["option_type"],
        ),
        axis=1,
    )
    # Very low-priced options are dominated by microstructure noise.
    df_sampled = df_sampled[df_sampled["mkt_price"] > 0.20].copy()
    if len(df_sampled) < 12:
        df_sampled = df_core.nlargest(min(len(df_core), 24), "liq_rank").copy()
        df_sampled["option_type"] = np.where(df_sampled["is_call"], "C", "P")
        df_sampled["mkt_price"] = df_sampled.apply(
            lambda row: black_scholes_price(
                spot_price,
                row["strike"],
                row["T"],
                r,
                q,
                row["mkt_iv"],
                row["option_type"],
            ),
            axis=1,
        )

    # Better Initial Guesses based on ATM volatility
    avg_iv = df["mkt_iv"].median()
    init_v0 = avg_iv**2

    # [kappa, theta, sigma, rho, v0]
    initial_guess = [2.0, init_v0, 0.5, -0.6, init_v0]

    def objective(params):
        # Unpack only parameters used directly in the penalties.
        kappa, theta, sigma = params[0], params[1], params[2]

        # Soft penalty for Feller Condition: 2*kappa*theta > sigma^2
        feller_penalty = 0
        if 2 * kappa * theta < sigma**2:
            feller_penalty = 10 * (sigma**2 - 2 * kappa * theta)

        model_prices = np.array(
            [
                heston_price(
                    S0=spot_price,
                    K=strikes[i],
                    T=maturities[i],
                    r=r,
                    q=q,
                    v0=params[4],
                    kappa=params[0],
                    theta=params[1],
                    sigma=params[2],
                    rho=params[3],
                    option_type=option_types[i],
                    # Coarser integration for calibration speed.
                    u_max=150,
                    quad_limit=90,
                    epsabs=1e-5,
                    epsrel=1e-5,
                )
                for i in range(len(strikes))
            ]
        )

        valid_mask = np.isfinite(model_prices) & (model_prices > 0)
        if not np.any(valid_mask):
            return 1e10

        mkt_prices = df_sampled["mkt_price"].values
        rel_err = (model_prices - mkt_prices) / np.maximum(mkt_prices, 0.5)
        # Huber-style robust loss to reduce outlier influence.
        delta = 0.75
        abs_err = np.abs(rel_err)
        robust_loss = np.where(abs_err <= delta, 0.5 * rel_err**2, delta * (abs_err - 0.5 * delta))

        weights = np.sqrt(df_sampled["volume"].values + 1.0)
        sse_price = np.sum(
            weights[valid_mask]
            * robust_loss[valid_mask]
        )

        # Penalty for failed integrations to guide optimizer away from bad params
        failure_penalty = (len(model_prices) - np.sum(valid_mask)) * 5.0
        # Keep v0 and theta in a realistic annualized variance range.
        regularization = 0.25 * (params[1] ** 2 + params[4] ** 2)
        # Discourage hugging parameter boundaries.
        boundary_penalty = 0.0
        for p, (lo, hi) in zip(params, bounds):
            span = hi - lo
            dist = min((p - lo) / span, (hi - p) / span)
            if dist < 0.03:
                boundary_penalty += (0.03 - dist) * 25.0

        return sse_price + feller_penalty + failure_penalty + regularization + boundary_penalty

    strikes = df_sampled["strike"].to_numpy(dtype=float)
    maturities = df_sampled["T"].to_numpy(dtype=float)
    option_types = df_sampled["option_type"].to_numpy()

    bounds = [(0.1, 8.0), (0.001, 1.0), (0.01, 1.5), (-0.95, -0.1), (0.001, 1.0)]
    low = np.array([b[0] for b in bounds], dtype=float)
    high = np.array([b[1] for b in bounds], dtype=float)

    # Multi-start helps avoid noisy/flat regions from numerical integration.
    rng = np.random.default_rng(42)
    starts = [np.array(initial_guess, dtype=float)]
    for _ in range(4):
        starts.append(low + (high - low) * rng.random(len(bounds)))
    starts.extend(
        [
            np.array([1.2, max(0.5 * init_v0, 0.001), 0.25, -0.4, max(0.7 * init_v0, 0.001)]),
            np.array([3.5, max(1.3 * init_v0, 0.001), 0.9, -0.8, max(1.1 * init_v0, 0.001)]),
        ]
    )

    best_x = np.array(starts[0], dtype=float)
    best_fun = np.inf
    for start in starts:
        res = minimize(
            objective,
            start,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 150, "ftol": 1e-9},
        )
        candidate_x = np.array(res.x if np.all(np.isfinite(res.x)) else start, dtype=float)
        candidate_fun = objective(candidate_x)
        if np.isfinite(candidate_fun) and candidate_fun < best_fun:
            best_fun = float(candidate_fun)
            best_x = candidate_x

    # If optimization landscape is pathological, pick the least-bad sampled start.
    if not np.isfinite(best_fun):
        scored = [(objective(np.array(s, dtype=float)), np.array(s, dtype=float)) for s in starts]
        scored = [item for item in scored if np.isfinite(item[0])]
        if scored:
            best_fun, best_x = min(scored, key=lambda x: x[0])

    return best_x
