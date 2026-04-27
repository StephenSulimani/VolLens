from __future__ import annotations

import json
import queue
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np

from models import (
    calculate_heston_vols,
    calculate_sabr_vols,
    calibrate_heston,
    calibrate_sabr,
    find_vol_arbitrage_opportunities,
    get_theoretical_smile,
)
from utils import get_risk_free_rate, process_options_chain
from yahoo import YahooOptions, get_dividend_yield, get_price


@dataclass
class JobState:
    id: str
    ticker: str
    status: str = "queued"
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    result: dict[str, Any] | None = None
    error: str | None = None
    events: "queue.Queue[dict[str, Any]]" = field(default_factory=queue.Queue)


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, tuple):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except Exception:
            return str(value)
    return value


class VolatilityAnalysisService:
    def __init__(self):
        self._options_client = YahooOptions()
        self._jobs: dict[str, JobState] = {}
        self._lock = threading.Lock()

    def start_job(self, ticker: str, sabr_beta: float = 0.5) -> str:
        ticker = ticker.strip().upper()
        if not ticker:
            raise ValueError("Ticker is required.")

        job_id = str(uuid.uuid4())
        job = JobState(id=job_id, ticker=ticker)
        with self._lock:
            self._jobs[job_id] = job

        thread = threading.Thread(
            target=self._run_job,
            args=(job_id, ticker, sabr_beta),
            daemon=True,
        )
        thread.start()
        return job_id

    def get_job(self, job_id: str) -> JobState | None:
        with self._lock:
            return self._jobs.get(job_id)

    def _push_event(self, job: JobState, event_type: str, payload: dict[str, Any]):
        job.updated_at = time.time()
        event = {"type": event_type, "ts": datetime.utcnow().isoformat() + "Z", **payload}
        job.events.put(event)

    def _run_job(self, job_id: str, ticker: str, sabr_beta: float):
        job = self.get_job(job_id)
        if job is None:
            return
        try:
            job.status = "running"
            self._push_event(job, "status", {"status": "running", "message": "Starting analysis"})

            spot = get_price(ticker)
            r_rate = get_risk_free_rate()
            q_yield = get_dividend_yield(ticker)
            self._push_event(job, "progress", {"step": "market_data", "spot": spot})

            options = self._options_client.options_chain(ticker, limit=20000)
            self._push_event(job, "progress", {"step": "options_fetched", "contracts": len(options)})

            df = process_options_chain(options, spot, r_rate, q_yield)
            self._push_event(
                job,
                "progress",
                {"step": "options_processed", "rows": len(df), "expiries": int(df["expiry_date"].nunique())},
            )

            sabr_params = calibrate_sabr(df, beta=sabr_beta)
            self._push_event(job, "progress", {"step": "sabr_calibrated", "expiries": len(sabr_params)})

            # Build SABR surfaces by expiry.
            sabr_surfaces = {}
            for expiry, params in sabr_params.items():
                sub = df[df["expiry_date"] == expiry]
                if sub.empty:
                    continue
                fwd = float(sub["forward"].iloc[0])
                ttm = float(sub["T"].iloc[0])
                sabr_surfaces[str(expiry)] = get_theoretical_smile(
                    fwd,
                    ttm,
                    float(params["atm_vol"]),
                    float(params["rho"]),
                    float(params["volvol"]),
                    beta=sabr_beta,
                    alpha=params.get("alpha"),
                )

            heston_params = calibrate_heston(df, spot, r_rate, q_yield)
            self._push_event(job, "progress", {"step": "heston_calibrated"})

            # Heston surfaces by expiry using same strikes as market chain.
            heston_surfaces = {}
            heston_params_arr = np.array(heston_params, dtype=float)
            for expiry, sub in df.groupby("expiry_date"):
                model_vols = calculate_heston_vols(sub, spot, r_rate, q_yield, heston_params_arr)
                heston_surfaces[str(expiry)] = [
                    {
                        "strike": float(k),
                        "vol": float(v),
                    }
                    for k, v in zip(sub["strike"].to_numpy(), model_vols)
                    if np.isfinite(v) and v > 0
                ]

            sabr_model_vols = calculate_sabr_vols(df, sabr_params, beta=sabr_beta)
            heston_model_vols = calculate_heston_vols(df, spot, r_rate, q_yield, heston_params_arr)

            sabr_quality_by_expiry = {
                expiry: {"rmse": p.get("rmse", np.nan), "status": p.get("status", "unknown")}
                for expiry, p in sabr_params.items()
            }
            heston_eval = df.copy()
            heston_eval["model_iv"] = heston_model_vols
            heston_eval = heston_eval[np.isfinite(heston_eval["model_iv"]) & (heston_eval["model_iv"] > 0)]
            heston_quality_by_expiry = {}
            for expiry, g in heston_eval.groupby("expiry_date"):
                rmse = float(np.sqrt(np.mean((g["model_iv"] - g["mkt_iv"]) ** 2)))
                heston_quality_by_expiry[expiry] = {
                    "rmse": rmse,
                    "status": "ok" if rmse <= 0.08 else "weak_fit",
                }

            sabr_opps = find_vol_arbitrage_opportunities(
                df,
                sabr_model_vols,
                "SABR",
                min_abs_spread=0.015,
                zscore_threshold=1.4,
                model_quality_by_expiry=sabr_quality_by_expiry,
                max_fit_rmse=0.06,
                allowed_statuses={"ok"},
                min_volume=30,
                max_spread_pct=0.10,
            )
            heston_opps = find_vol_arbitrage_opportunities(
                df,
                heston_model_vols,
                "Heston",
                min_abs_spread=0.015,
                zscore_threshold=1.4,
                model_quality_by_expiry=heston_quality_by_expiry,
                max_fit_rmse=0.08,
                allowed_statuses={"ok"},
                min_volume=30,
                max_spread_pct=0.10,
            )

            result = {
                "ticker": ticker,
                "spot": float(spot),
                "rates": {"risk_free": float(r_rate), "dividend_yield": float(q_yield)},
                "meta": {"contracts": int(len(options)), "processed_rows": int(len(df))},
                "sabr": {
                    "beta": float(sabr_beta),
                    "params_by_expiry": _to_jsonable(sabr_params),
                    "surface_by_expiry": _to_jsonable(sabr_surfaces),
                    "opportunities": _to_jsonable(
                        sabr_opps.head(100).to_dict(orient="records")
                    ),
                },
                "heston": {
                    "params": _to_jsonable(
                        {
                            "kappa": float(heston_params_arr[0]),
                            "theta": float(heston_params_arr[1]),
                            "sigma": float(heston_params_arr[2]),
                            "rho": float(heston_params_arr[3]),
                            "v0": float(heston_params_arr[4]),
                        }
                    ),
                    "surface_by_expiry": _to_jsonable(heston_surfaces),
                    "opportunities": _to_jsonable(
                        heston_opps.head(100).to_dict(orient="records")
                    ),
                },
            }
            job.result = result
            job.status = "completed"
            self._push_event(
                job,
                "completed",
                {
                    "status": "completed",
                    "summary": {
                        "sabr_opportunities": int(len(sabr_opps)),
                        "heston_opportunities": int(len(heston_opps)),
                    },
                },
            )
        except Exception as exc:
            job.status = "failed"
            job.error = str(exc)
            self._push_event(
                job,
                "error",
                {"status": "failed", "message": str(exc)},
            )

    @staticmethod
    def event_to_sse(event: dict[str, Any]) -> str:
        payload = json.dumps(_to_jsonable(event))
        return f"event: {event.get('type', 'message')}\ndata: {payload}\n\n"
