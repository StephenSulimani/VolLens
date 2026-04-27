import numpy as np
from scipy.optimize import brentq, newton
from scipy.stats import norm


def black_scholes_price(S, K, T, r, q, sigma, option_type="C"):
    """Standard Black-Scholes-Merton formula."""
    sigma = np.clip(sigma, 1e-4, 10.0)
    T = max(T, 1e-6)

    vol_sqrt_t = sigma * np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / vol_sqrt_t
    d2 = d1 - vol_sqrt_t

    if option_type == "C":
        return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)


def black_scholes_vega(S, K, T, r, q, sigma):
    """Vega (dPrice/dVol)."""
    sigma = np.clip(sigma, 1e-4, 10.0)
    T = max(T, 1e-6)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T)


def implied_vol(target_price, S, K, T, r, q, is_call):
    """Find implied volatility via Newton fallback to Brent."""
    option_type = "C" if is_call else "P"

    def f(sigma):
        return black_scholes_price(S, K, T, r, q, sigma, option_type) - target_price

    def f_prime(sigma):
        return black_scholes_vega(S, K, T, r, q, sigma)

    try:
        iv = newton(f, x0=0.3, fprime=f_prime, tol=1e-5, maxiter=50)
        return max(0.0, float(iv))
    except (RuntimeError, OverflowError, ZeroDivisionError, ValueError):
        try:
            return float(brentq(f, 1e-6, 4.0))
        except ValueError:
            return 0.0
