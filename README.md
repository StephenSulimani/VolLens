# VolLens

VolLens is an educational web application for studying how **implied volatility** behaves across **strikes** and **expirations** for a listed equity or ETF. You enter a ticker; the system loads an option chain, calibrates two widely used models (**SABR** and **Heston**), and returns **three-dimensional volatility surfaces** so you can compare model shapes to each other and to the market. It also ranks **model-versus-market** discrepancies and builds **consensus** views where both models point in a similar direction, which is useful for classroom discussion of smile dynamics, calibration risk, and the limits of parametric models.

The UI walks a user from **raw chain data** through **calibration progress** (streamed in real time), then through **interactive charts and tables**: surfaces for each model, ranked opportunity lists per model, a consensus table, and a drill-down panel with an approximate **payoff at expiration**, a **volatility smile slice** at the chosen expiry, and a simplified **delta-hedged Monte Carlo** illustration. The project is meant to **support learning and visualization**, not production workflows.

---

## Disclaimer: not for real-world trading

**VolLens is not intended for use in live trading, investment decisions, portfolio management, or any commercial or personal financial activity.** Outputs are model-based, simplified, and may omit market frictions, liquidity, borrow costs, taxes, corporate events, and other factors that matter in practice. Nothing here is investment, legal, or tax advice. The authors and Georgia Institute of Technology make no representation that any signal, chart, or number is accurate, complete, or suitable for any purpose beyond academic discussion. **If you trade real capital, rely on your own diligence and professional advice, not this software.**

---

## Academic context

This work is submitted in connection with **Dr. Satyajit Karnik’s** course **MGT6081 – Derivatives Securities** at the **Georgia Institute of Technology**, as part of the **Master of Science in Quantitative and Computational Finance** program.

It is a course project: scope, correctness, and maintenance are aligned with **pedagogy and demonstration**, not with regulated or institutional systems.

---

## Authors

Developed by **Jacob Labkovski**, **Elijah Miller**, **Thomas Noe**, and **Stephen Sulimani**.

---

## Architecture

Two services in `docker-compose.yml`, each with its own `Dockerfile`: computation in the backend, UI in the frontend.

| Component | Summary |
|-----------|---------|
| **Docker Compose** | Builds and runs both containers; maps ports **8080** (API) and **5173** (UI); sets **`VITE_API_PROXY_TARGET`** so the frontend can reach the backend by service name. |
| **Backend container (Flask)** | Python app on **8080**; REST JSON plus **SSE** job streams; pulls chain data, calibrates SABR/Heston, returns structured **`result`** JSON. No server-rendered HTML. |
| **Frontend container (React + Vite)** | Bun-built SPA served on **5173**; browser loads assets from this origin; **`/api`** and **`/health`** are proxied to the backend (same origin for the client). |
| **Typical client flow** | `POST /api/analyze` → `job_id` → **`EventSource`** on `/api/jobs/<id>/stream` for progress and partial model payloads → **`GET /api/jobs/<id>/result`** for the final snapshot used by charts and tables. |

---

## Backend container (Flask)

The backend image is built from `backend/Dockerfile` (Python slim base, dependencies from `backend/requirements.txt`, entrypoint `python -m api.app` listening on `0.0.0.0:8080`). The code is organized into layers: HTTP surface, a **service** that owns jobs and threads, **Yahoo** adapters for market data, **utilities** for rates and chain hygiene, and **models** for pricing and calibration.

### `api/app.py` (Flask application)

Defines the Flask app and wires routes to a single **`VolatilityAnalysisService`** instance.

- **`GET /health`** – Liveness check for orchestrators or the frontend proxy.
- **`POST /api/analyze`** – Accepts JSON (`ticker`, optional `sabr_beta`). Validates input, starts an asynchronous **analysis job** in a background thread, returns **202** with a **`job_id`**.
- **`GET /api/jobs/<job_id>`** – Returns job metadata: status, timestamps, error message if failed.
- **`GET /api/jobs/<job_id>/result`** – Returns the full **`result`** object once the job has finished successfully; otherwise appropriate HTTP status and a short message.
- **`GET /api/jobs/<job_id>/stream`** – **Server-Sent Events (SSE)** stream. The client receives named events (`connected`, `status`, `progress`, `sabr_ready`, `heston_ready`, `completed`, `final`, etc.) with JSON payloads so the UI can show progress and can **hydrate partial results** (for example, the SABR surface as soon as SABR finishes, before Heston completes). Keep-alive comments are sent so intermediaries do not close idle connections.

### `api/service.py` (`VolatilityAnalysisService`)

The orchestration layer: **job lifecycle**, **threading**, and **coordination** of data pull, preprocessing, and both models.

- **`JobState`** – In-memory record per job: id, ticker, status, timestamps, optional error, **`result`** dict when done, and a **`queue.Queue`** of events consumed by the SSE generator.
- **`start_job`** – Allocates a UUID, stores `JobState`, spawns a **daemon thread** that runs the full pipeline, returns the id immediately so the client can subscribe to the stream.
- **`_run_job`** – High-level sequence: fetch options and spot, obtain **risk-free** and **dividend yield**, build a processed chain (`process_options_chain`), then runs **SABR** and **Heston** work. Model fitting and surface construction are dispatched in parallel (**`ThreadPoolExecutor`**, `as_completed`) so whichever model finishes first can be pushed over SSE first; the final payload merges both sides plus **consensus** rows.
- **SSE helpers** – Events are JSON-serialized; numpy and pandas types are normalized via **`_to_jsonable`** so the client always receives plain JSON.

### `yahoo/` (market data)

- **`YahooOptions`** (in `yahoo/options.py`) – Uses **yfinance** to download option chains and related fields needed for calibration and implied vol work.
- **`underlying.py`** – **Spot price** and **dividend yield** for the symbol via yfinance `Ticker` metadata, used as inputs to the pricing stack.

### `utils/` (environment and chain preparation)

- **`treasury.py`** – **`get_risk_free_rate`**: pulls a short-term **U.S. Treasury** benchmark (annualized rate as a decimal) from the Fiscal Data Treasury API, used consistently for discounting and model inputs.
- **`processing.py`** – **`process_options_chain`** and related helpers: filters, aligns, and cleans raw chain rows into the tabular form expected by SABR/Heston calibration and by the vol-arbitrage scoring logic (handles bad quotes, missing fields, and expiry alignment as implemented in code).

### `models/` (quantitative core)

- **`sabr.py`** – **SABR** calibration per expiry (parameters such as ATM vol, skew, smile controls depending on implementation), construction of **model implied volatilities** across strikes, and integration with the surface builder used by the API.
- **`heston.py`** – **Heston** stochastic volatility model: calibration to the chain (parameters such as mean reversion, long-run variance, vol of vol, correlation, initial variance), then generation of **model IV** over the same strike/expiry grid used for visualization.
- **`black_scholes.py`** – Shared **Black–Scholes** machinery (prices, implied vol inversion where needed) used as a building block inside the pipeline.
- **`arbitrage.py`** – **`calculate_sabr_vols` / `calculate_heston_vols`**, **`find_vol_arbitrage_opportunities`** (per-model ranked table: where model IV differs from market IV by strike/expiry, with spreads and heuristics such as z-scores and suggested “side”), **`build_consensus_signals`** (merges SABR and Heston disagreement into a smaller set of rows where both models agree in direction), and helpers such as **`get_theoretical_smile`** used when building **surface_by_expiry** slices for the JSON payload.

The **`result`** object returned to the client aggregates **spot**, **`rates`** (risk-free and dividend yield), **SABR** block (beta, `params_by_expiry`, `surface_by_expiry`, `opportunities`), **Heston** block (parameters dict, surfaces, opportunities), and **consensus** `signals`, plus light **meta** (counts). That structure is what the frontend maps directly into charts and tables.

---

## Frontend container (React)

The frontend image is multi-stage in `frontend/Dockerfile`: install dependencies with **Bun**, run **`bun run build`** (Vite production build), then serve with **`bun run preview`** on `0.0.0.0:5173` with the same **Vite proxy** rules as dev, so `/api` still reaches the backend container when `VITE_API_PROXY_TARGET` is set.

### Stack and layout

- **React** with **TypeScript** (`.tsx`) for components and state.
- **Vite** as bundler and dev/preview server; **Tailwind CSS** (via `@tailwindcss/vite`) for layout and styling.
- **Plotly** (via `react-plotly.js` factory + `plotly.js-dist-min`) for **3D surface** plots and **2D** payoff, smile, histogram, and dual-axis hedge charts.

### Main application behavior (`src/App.tsx`)

- **Ticker form** – Submits analysis, clears prior state, opens **EventSource** on the stream URL, listens for named SSE events, and merges **`sabr_ready` / `heston_ready`** partial payloads into React state so surfaces can appear incrementally.
- **Completion** – On `completed` / `final`, fetches **`/api/jobs/<id>/result`** and replaces state with the full **`result`**.
- **Surfaces** – Builds interpolated grids from **`surface_by_expiry`** for SABR and Heston and renders two **surface** traces with axis titles and legends oriented to teaching.
- **Consensus table** – Renders merged signals; row click opens the detail experience with a synthetic “opportunity” row when needed.
- **Opportunity tables** – SABR-only and Heston-only ranked rows; click sets **selected opportunity** and opens the modal.
- **Opportunity modal** – Shows suggested **structure** (single option vs vertical spread), **payoff at expiry** (illustrative BS-style premiums), **smile slice** at that expiry (model curves vs market IV at strike), **delta-hedged simulation** (discrete hedge paths, histogram of terminal slippage) using rates from the job when available.
- **Stream log** – Recent raw events for debugging and for demonstrating the streaming protocol in class.

Together, the frontend is a **thin client**: it does not re-implement calibration; it visualizes and explains the backend’s numbers.

---

## Quick start (Docker)

From the repository root:

```bash
docker compose up --build
```

- Backend API: `http://localhost:8080`
- Frontend: `http://localhost:5173`

The frontend proxies `/api` and `/health` to the backend using `VITE_API_PROXY_TARGET` (see `docker-compose.yml` and `frontend/vite.config.js`). Change published ports in Compose if they conflict with your machine.

For local development without Docker, use the `backend/` tree with `backend/requirements.txt` and the `frontend/` tree with Bun and the same Vite proxy defaults pointing at a locally running Flask app on port 8080 unless you override `VITE_API_PROXY_TARGET`.

---

## License

Released under the [MIT License](LICENSE).