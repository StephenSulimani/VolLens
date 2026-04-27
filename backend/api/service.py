from __future__ import annotations

import json
import logging
import queue
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np

from models import (
    build_consensus_signals,
    calculate_heston_vols,
    calculate_sabr_vols,
    calibrate_heston,
    calibrate_sabr,
    find_vol_arbitrage_opportunities,
    get_theoretical_smile,
)
from utils import get_risk_free_rate, process_options_chain
from yahoo import YahooOptions, get_dividend_yield, get_price

logger = logging.getLogger("vollens.api.service")


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
        except (AttributeError, TypeError, ValueError):
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
        logger.info(
            "job_created job_id=%s ticker=%s sabr_beta=%.3f",
            job_id,
            ticker,
            sabr_beta,
        )

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
        t0 = time.perf_counter()
        try:
            job.status = "running"
            logger.info("job_started job_id=%s ticker=%s", job_id, ticker)
            self._push_event(job, "status", {"status": "running", "message": "Starting analysis"})

            spot = get_price(ticker)
            r_rate = get_risk_free_rate()
            q_yield = get_dividend_yield(ticker)
            logger.info(
                "market_data_ready job_id=%s spot=%.4f r=%.5f q=%.5f",
                job_id,
                float(spot),
                float(r_rate),
                float(q_yield),
            )
            self._push_event(job, "progress", {"step": "market_data", "spot": spot})

            options = self._options_client.options_chain(ticker, limit=20000)
            logger.info(
                "options_fetched job_id=%s contracts=%d",
                job_id,
                len(options),
            )
            self._push_event(job, "progress", {"step": "options_fetched", "contracts": len(options)})

            df = process_options_chain(options, spot, r_rate, q_yield)
            logger.info(
                "options_processed job_id=%s rows=%d expiries=%d",
                job_id,
                len(df),
                int(df["expiry_date"].nunique()),
            )
            self._push_event(
                job,
                "progress",
                {"step": "options_processed", "rows": len(df), "expiries": int(df["expiry_date"].nunique())},
            )

            def compute_sabr():
                sabr_params = calibrate_sabr(df, beta=sabr_beta)
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
                sabr_model_vols = calculate_sabr_vols(df, sabr_params, beta=sabr_beta)
                sabr_quality_by_expiry = {
                    expiry: {"rmse": p.get("rmse", np.nan), "status": p.get("status", "unknown")}
                    for expiry, p in sabr_params.items()
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
                return sabr_params, sabr_surfaces, sabr_opps

            def compute_heston():
                heston_params = calibrate_heston(df, spot, r_rate, q_yield)
                heston_params_arr = np.array(heston_params, dtype=float)
                heston_surfaces = {}
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
                heston_model_vols = calculate_heston_vols(df, spot, r_rate, q_yield, heston_params_arr)
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
                return heston_params_arr, heston_surfaces, heston_opps

            sabr_params = {}
            sabr_surfaces = {}
            sabr_opps = None
            heston_params_arr = np.array([np.nan, np.nan, np.nan, np.nan, np.nan], dtype=float)
            heston_surfaces = {}
            heston_opps = None

            with ThreadPoolExecutor(max_workers=2) as pool:
                future_map = {
                    pool.submit(compute_sabr): "sabr",
                    pool.submit(compute_heston): "heston",
                }

                for future in as_completed(future_map):
                    model_name = future_map[future]
                    if model_name == "sabr":
                        sabr_params, sabr_surfaces, sabr_opps = future.result()
                        logger.info(
                            "sabr_ready job_id=%s expiries=%d opps=%d",
                            job_id,
                            len(sabr_params),
                            len(sabr_opps),
                        )
                        self._push_event(
                            job,
                            "sabr_ready",
                            {
                                "step": "sabr_ready",
                                "sabr": _to_jsonable(
                                    {
                                        "beta": float(sabr_beta),
                                        "params_by_expiry": sabr_params,
                                        "surface_by_expiry": sabr_surfaces,
                                        "opportunities": sabr_opps.head(100).to_dict(orient="records"),
                                    }
                                ),
                            },
                        )
                    else:
                        heston_params_arr, heston_surfaces, heston_opps = future.result()
                        logger.info(
                            "heston_ready job_id=%s expiries=%d opps=%d params=[%.4f,%.4f,%.4f,%.4f,%.4f]",
                            job_id,
                            len(heston_surfaces),
                            len(heston_opps),
                            float(heston_params_arr[0]),
                            float(heston_params_arr[1]),
                            float(heston_params_arr[2]),
                            float(heston_params_arr[3]),
                            float(heston_params_arr[4]),
                        )
                        self._push_event(
                            job,
                            "heston_ready",
                            {
                                "step": "heston_ready",
                                "heston": _to_jsonable(
                                    {
                                        "params": {
                                            "kappa": float(heston_params_arr[0]),
                                            "theta": float(heston_params_arr[1]),
                                            "sigma": float(heston_params_arr[2]),
                                            "rho": float(heston_params_arr[3]),
                                            "v0": float(heston_params_arr[4]),
                                        },
                                        "surface_by_expiry": heston_surfaces,
                                        "opportunities": heston_opps.head(100).to_dict(orient="records"),
                                    }
                                ),
                            },
                        )

            if sabr_opps is None or heston_opps is None:
                raise RuntimeError("Model computation did not complete for both SABR and Heston.")
            consensus = build_consensus_signals(sabr_opps, heston_opps)
            logger.info(
                "consensus_ready job_id=%s signals=%d",
                job_id,
                len(consensus),
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
                "consensus": {
                    "signals": _to_jsonable(consensus.head(100).to_dict(orient="records")),
                },
            }
            job.result = result
            job.status = "completed"
            elapsed = time.perf_counter() - t0
            logger.info(
                "job_completed job_id=%s ticker=%s elapsed_s=%.2f sabr_opps=%d heston_opps=%d consensus=%d",
                job_id,
                ticker,
                elapsed,
                int(len(sabr_opps)),
                int(len(heston_opps)),
                int(len(consensus)),
            )
            self._push_event(
                job,
                "completed",
                {
                    "status": "completed",
                    "summary": {
                        "sabr_opportunities": int(len(sabr_opps)),
                        "heston_opportunities": int(len(heston_opps)),
                        "consensus_signals": int(len(consensus)),
                    },
                },
            )
        except (RuntimeError, ValueError, TypeError) as exc:
            job.status = "failed"
            job.error = str(exc)
            elapsed = time.perf_counter() - t0
            logger.exception(
                "job_failed job_id=%s ticker=%s elapsed_s=%.2f error=%s",
                job_id,
                ticker,
                elapsed,
                str(exc),
            )
            self._push_event(
                job,
                "error",
                {"status": "failed", "message": str(exc)},
            )

    @staticmethod
    def event_to_sse(event: dict[str, Any]) -> str:
        payload = json.dumps(_to_jsonable(event))
        return f"event: {event.get('type', 'message')}\ndata: {payload}\n\n"
