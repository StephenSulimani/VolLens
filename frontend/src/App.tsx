import { useEffect, useMemo, useState } from "react";
import createPlotlyComponentModule from "react-plotly.js/factory";
import * as PlotlyModule from "plotly.js-dist-min";

const createPlotlyComponent =
  (
    createPlotlyComponentModule as unknown as {
      default?: (plotly: unknown) => unknown;
    }
  ).default ??
  (createPlotlyComponentModule as unknown as (plotly: unknown) => unknown);
const Plotly =
  (PlotlyModule as unknown as { default?: unknown }).default ??
  (PlotlyModule as unknown);
const Plot = createPlotlyComponent(
  Plotly as never,
) as unknown as React.ComponentType<Record<string, unknown>>;

type SurfacePoint = {
  strike: number;
  vol: number;
};

type Opportunity = {
  expiry_date: string;
  strike: number;
  is_call: boolean;
  mkt_iv: number;
  model_iv: number;
  iv_spread: number;
  spread_zscore: number;
  vol_arb_side: string;
  T?: number;
};

type ConsensusSignal = {
  expiry_date: string;
  strike: number;
  is_call: boolean;
  mkt_iv: number;
  sabr_iv_spread: number;
  heston_iv_spread: number;
  consensus_spread: number;
  agreement_score: number;
  consensus_side: string;
};

type SabrExpiryParams = {
  rmse?: number;
  status?: string;
  rho?: number;
  volvol?: number;
  atm_vol?: number;
};

type AnalysisResult = {
  spot?: number;
  rates?: {
    risk_free?: number;
    dividend_yield?: number;
  };
  sabr?: {
    beta?: number;
    params_by_expiry?: Record<string, SabrExpiryParams>;
    surface_by_expiry?: Record<string, SurfacePoint[]>;
    opportunities?: Opportunity[];
  };
  heston?: {
    surface_by_expiry?: Record<string, SurfacePoint[]>;
    opportunities?: Opportunity[];
  };
  consensus?: {
    signals?: ConsensusSignal[];
  };
};

type SurfaceGrid = {
  x: number[];
  y: number[];
  yLabels: string[];
  z: number[][];
};
type StreamEvent = Record<string, unknown> & {
  type?: string;
  ts?: string;
  status?: string;
  message?: string;
  step?: string;
  sabr?: AnalysisResult["sabr"];
  heston?: AnalysisResult["heston"];
};

type StrategyMode = "single" | "vertical";

function buildSurface(
  surfaceByExpiry?: Record<string, SurfacePoint[]>,
): SurfaceGrid | null {
  const entries = Object.entries(surfaceByExpiry || {});
  if (entries.length === 0) {
    return null;
  }

  const expiries = entries
    .map(([exp]) => exp)
    .sort((a, b) => new Date(a).getTime() - new Date(b).getTime());
  const byExpiry = new Map<string, SurfacePoint[]>(
    entries.map(([exp, pts]) => [
      exp,
      (pts || [])
        .map((p) => ({ strike: Number(p.strike), vol: Number(p.vol) }))
        .filter((p) => Number.isFinite(p.strike) && Number.isFinite(p.vol))
        .sort((a, b) => a.strike - b.strike),
    ]),
  );

  const allStrikes = Array.from(
    byExpiry.values().flatMap((pts) => pts.map((p) => p.strike)),
  ).sort((a, b) => a - b);
  if (allStrikes.length < 2) {
    return null;
  }

  const minK = allStrikes[Math.floor(allStrikes.length * 0.03)];
  const maxK = allStrikes[Math.ceil(allStrikes.length * 0.97) - 1];
  if (!Number.isFinite(minK) || !Number.isFinite(maxK) || minK >= maxK) {
    return null;
  }

  const strikeSteps = 60;
  const x = Array.from(
    { length: strikeSteps },
    (_, i) => minK + ((maxK - minK) * i) / (strikeSteps - 1),
  );

  function interpolate(points: SurfacePoint[], k: number): number {
    if (points.length === 0) return NaN;
    if (k <= points[0].strike) return points[0].vol;
    if (k >= points[points.length - 1].strike)
      return points[points.length - 1].vol;
    for (let i = 1; i < points.length; i += 1) {
      const p0 = points[i - 1];
      const p1 = points[i];
      if (k <= p1.strike) {
        const w = (k - p0.strike) / Math.max(p1.strike - p0.strike, 1e-9);
        return p0.vol + w * (p1.vol - p0.vol);
      }
    }
    return points[points.length - 1].vol;
  }

  const y = expiries.map((_, i) => i);
  const z = expiries.map((exp) => {
    const row = byExpiry.get(exp) || [];
    return x.map((k) => interpolate(row, k));
  });

  return { x, y, yLabels: expiries, z };
}

function oppRowKey(
  r: Pick<Opportunity, "expiry_date" | "strike" | "is_call" | "model_iv">,
) {
  return `${r.expiry_date}-${r.strike}-${r.is_call}-${r.model_iv}`;
}

function expiryKeyMatch(
  mapKeys: string[],
  expiryDate: string,
): string | undefined {
  const d = String(expiryDate).slice(0, 10);
  return mapKeys.find((k) => String(k).slice(0, 10) === d);
}

function findRowT(
  result: AnalysisResult | null,
  expiry: string,
  strike: number,
  isCall: boolean,
): number | undefined {
  if (!result) return undefined;
  const same = (o: Opportunity) =>
    String(o.expiry_date).slice(0, 10) === String(expiry).slice(0, 10) &&
    Number(o.strike) === Number(strike) &&
    o.is_call === isCall;
  return (
    result.sabr?.opportunities?.find(same)?.T ??
    result.heston?.opportunities?.find(same)?.T
  );
}

function sabrFitForExpiry(
  result: AnalysisResult | null,
  expiryDate: string,
): SabrExpiryParams | null {
  const p = result?.sabr?.params_by_expiry;
  if (!p) return null;
  const k = expiryKeyMatch(Object.keys(p), expiryDate);
  if (!k) return null;
  return p[k] ?? null;
}

function sliceSmileForExpiry(
  surfaceByExpiry: Record<string, SurfacePoint[]> | undefined,
  expiryDate: string,
): SurfacePoint[] {
  if (!surfaceByExpiry) return [];
  const k = expiryKeyMatch(Object.keys(surfaceByExpiry), expiryDate);
  if (!k) return [];
  const pts = surfaceByExpiry[k] || [];
  return pts
    .map((p) => ({ strike: Number(p.strike), vol: Number(p.vol) }))
    .filter((p) => Number.isFinite(p.strike) && Number.isFinite(p.vol))
    .sort((a, b) => a.strike - b.strike);
}

function OppTable({
  rows,
  title,
  caption,
  onSelect,
  selectedKey,
}: {
  rows: Opportunity[];
  title: string;
  caption?: string;
  onSelect: (row: Opportunity) => void;
  selectedKey: string;
}) {
  const headers: Array<{ key: string; label: string; tip: string }> = [
    { key: "expiry", label: "Expiry", tip: "Option expiration date." },
    { key: "strike", label: "Strike", tip: "Option strike price." },
    {
      key: "type",
      label: "Type",
      tip: "Call (C) profits if the stock finishes above the strike at expiry; put (P) profits if it finishes below.",
    },
    {
      key: "spread",
      label: "Spread",
      tip: "Model implied vol minus market implied vol. Negative usually means the market is pricing higher volatility than this model.",
    },
    {
      key: "z",
      label: "Z",
      tip: "How extreme this spread is compared with other strikes in the same expiration (like a rough signal strength).",
    },
    {
      key: "side",
      label: "Side",
      tip: "buy_vol suggests options look cheap vs the model; sell_vol suggests they look rich (educational labels, not investment advice).",
    },
  ];

  return (
    <section className="rounded-xl border border-zinc-800 bg-zinc-900/60 p-4">
      <h3 className={`text-lg font-semibold ${caption ? "mb-1" : "mb-3"}`}>
        {title}
      </h3>
      {caption ? (
        <p className="mb-3 text-xs leading-relaxed text-zinc-500">{caption}</p>
      ) : null}
      {rows.length === 0 ? (
        <p className="text-sm text-zinc-400">No opportunities found.</p>
      ) : (
        <div className="overflow-auto">
          <table className="min-w-full text-left text-sm">
            <thead className="text-zinc-400">
              <tr>
                {headers.map((h) => (
                  <th key={h.key} className="px-2 py-1">
                    <span
                      className="cursor-help decoration-dotted underline-offset-2 hover:underline"
                      title={h.tip}
                    >
                      {h.label}
                    </span>
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {rows.slice(0, 12).map((r, i) => (
                <tr
                  key={`${r.expiry_date}-${r.strike}-${i}`}
                  className={`cursor-pointer border-t border-zinc-800 hover:bg-zinc-800/60 ${
                    selectedKey && selectedKey === oppRowKey(r)
                      ? "bg-zinc-800/70"
                      : ""
                  }`}
                  onClick={() => onSelect(r)}
                >
                  <td className="px-2 py-1">
                    {String(r.expiry_date).slice(0, 10)}
                  </td>
                  <td className="px-2 py-1">{Number(r.strike).toFixed(2)}</td>
                  <td className="px-2 py-1">{r.is_call ? "C" : "P"}</td>
                  <td className="px-2 py-1">
                    {Number(r.iv_spread).toFixed(4)}
                  </td>
                  <td className="px-2 py-1">
                    {Number(r.spread_zscore).toFixed(2)}
                  </td>
                  <td className="px-2 py-1">{r.vol_arb_side}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
}

function LoadingPanel({ label }: { label: string }) {
  return (
    <div className="flex h-[420px] flex-col items-center justify-center gap-3 text-zinc-300">
      <div className="h-8 w-8 animate-spin rounded-full border-2 border-zinc-700 border-t-indigo-400" />
      <p className="text-sm">{label}</p>
    </div>
  );
}

function normCdf(x: number): number {
  // Abramowitz-Stegun approximation
  const sign = x < 0 ? -1 : 1;
  const z = Math.abs(x) / Math.sqrt(2);
  const t = 1 / (1 + 0.3275911 * z);
  const a1 = 0.254829592;
  const a2 = -0.284496736;
  const a3 = 1.421413741;
  const a4 = -1.453152027;
  const a5 = 1.061405429;
  const erf =
    1 - ((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t * Math.exp(-z * z);
  return 0.5 * (1 + sign * erf);
}

function approxOptionPremium(
  S: number,
  K: number,
  T: number,
  iv: number,
  isCall: boolean,
): number {
  const tau = Math.max(T, 1e-4);
  const sigma = Math.max(iv, 1e-4);
  const d1 =
    (Math.log(Math.max(S, 1e-6) / Math.max(K, 1e-6)) +
      0.5 * sigma * sigma * tau) /
    (sigma * Math.sqrt(tau));
  const d2 = d1 - sigma * Math.sqrt(tau);
  if (isCall) return S * normCdf(d1) - K * normCdf(d2);
  return K * normCdf(-d2) - S * normCdf(-d1);
}

/** Only if the job payload has no `rates.risk_free` yet (e.g. partial stream). */
const RISK_FREE_FALLBACK = 0.015;

function bsPrice(
  S: number,
  K: number,
  tau: number,
  sigma: number,
  isCall: boolean,
  r: number,
): number {
  const t = Math.max(tau, 1e-10);
  const sig = Math.max(sigma, 1e-6);
  const sqrtT = Math.sqrt(t);
  const d1 =
    (Math.log(Math.max(S, 1e-9) / Math.max(K, 1e-9)) +
      (r + 0.5 * sig * sig) * t) /
    (sig * sqrtT);
  const d2 = d1 - sig * sqrtT;
  const df = Math.exp(-r * t);
  if (isCall) return S * normCdf(d1) - K * df * normCdf(d2);
  return K * df * normCdf(-d2) - S * normCdf(-d1);
}

function bsDelta(
  S: number,
  K: number,
  tau: number,
  sigma: number,
  isCall: boolean,
  r: number,
): number {
  const t = Math.max(tau, 1e-10);
  const sig = Math.max(sigma, 1e-6);
  const sqrtT = Math.sqrt(t);
  const d1 =
    (Math.log(Math.max(S, 1e-9) / Math.max(K, 1e-9)) +
      (r + 0.5 * sig * sig) * t) /
    (sig * sqrtT);
  if (isCall) return normCdf(d1);
  return normCdf(d1) - 1;
}

function portIntrinsic(
  S: number,
  K: number,
  K2: number | null,
  isCall: boolean,
  vertical: boolean,
): number {
  const leg = (k: number) => (isCall ? Math.max(0, S - k) : Math.max(0, k - S));
  const p1 = leg(K);
  if (!vertical || K2 == null) return p1;
  return p1 - leg(K2);
}

function portValue(
  S: number,
  tau: number,
  K: number,
  K2: number | null,
  sigma: number,
  isCall: boolean,
  r: number,
  vertical: boolean,
): number {
  const v1 = bsPrice(S, K, tau, sigma, isCall, r);
  if (!vertical || K2 == null) return v1;
  return v1 - bsPrice(S, K2, tau, sigma, isCall, r);
}

function portDelta(
  S: number,
  tau: number,
  K: number,
  K2: number | null,
  sigma: number,
  isCall: boolean,
  r: number,
  vertical: boolean,
): number {
  const d1 = bsDelta(S, K, tau, sigma, isCall, r);
  if (!vertical || K2 == null) return d1;
  return d1 - bsDelta(S, K2, tau, sigma, isCall, r);
}

function randn(rng: () => number): number {
  const u = Math.max(rng(), 1e-12);
  const v = Math.max(rng(), 1e-12);
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a += 0x6d2b79f5;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function hashInt(s: string): number {
  let h = 2166136261 >>> 0;
  for (let i = 0; i < s.length; i += 1) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return h >>> 0;
}

type HedgePathResult = {
  times: number[];
  spotPath: number[];
  markToMarket: number[];
  terminalError: number;
};

function simulateDeltaHedgePath(args: {
  S0: number;
  K: number;
  K2: number | null;
  T: number;
  sigma: number;
  isCall: boolean;
  r: number;
  vertical: boolean;
  optionSign: number;
  nSteps: number;
  rng: () => number;
}): HedgePathResult {
  const { S0, K, K2, T, sigma, isCall, r, vertical, optionSign, nSteps, rng } =
    args;
  const dt = T / Math.max(nSteps, 1);
  let S = S0;
  const V0 = portValue(S0, T, K, K2, sigma, isCall, r, vertical);
  let C = -optionSign * V0;
  let n = 0;
  const d0 = optionSign * portDelta(S0, T, K, K2, sigma, isCall, r, vertical);
  const n0 = -d0;
  C -= (n0 - n) * S;
  n = n0;

  const times: number[] = [0];
  const spotPath: number[] = [S];
  const markToMarket: number[] = [
    C + n * S + optionSign * portValue(S, T, K, K2, sigma, isCall, r, vertical),
  ];

  for (let i = 0; i < nSteps; i += 1) {
    const tauAfter = Math.max(T - (i + 1) * dt, 0);
    const z = randn(rng);
    const drift = (r - 0.5 * sigma * sigma) * dt;
    const diff = sigma * Math.sqrt(dt) * z;
    const S_next = S * Math.exp(drift + diff);

    C += n * (S_next - S);

    const deltaNext =
      optionSign *
      portDelta(S_next, tauAfter, K, K2, sigma, isCall, r, vertical);
    const nNext = -deltaNext;
    C -= (nNext - n) * S_next;
    n = nNext;
    S = S_next;

    const Vmark =
      optionSign * portValue(S, tauAfter, K, K2, sigma, isCall, r, vertical);
    markToMarket.push(C + n * S + Vmark);
    times.push((i + 1) * dt);
    spotPath.push(S);
  }

  const I = portIntrinsic(S, K, K2, isCall, vertical);
  const terminalError = C + n * S + optionSign * I;

  return { times, spotPath, markToMarket, terminalError };
}

function OpportunityModal({
  opp,
  result,
  strategyMode,
  setStrategyMode,
  onClose,
}: {
  opp: Opportunity;
  result: AnalysisResult;
  strategyMode: StrategyMode;
  setStrategyMode: (m: StrategyMode) => void;
  onClose: () => void;
}) {
  const [hedgeSteps, setHedgeSteps] = useState(52);
  const [hedgeRun, setHedgeRun] = useState(0);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  const spot = Number(result.spot);
  const K = Number(opp.strike);
  const T =
    Number(opp.T) ||
    findRowT(result, opp.expiry_date, K, !!opp.is_call) ||
    0.25;
  const iv = Number(opp.mkt_iv);
  const isCall = !!opp.is_call;
  const side = opp.vol_arb_side === "buy_vol" ? "LONG_VOL" : "SHORT_VOL";
  const premium = Math.max(0.01, approxOptionPremium(spot, K, T, iv, isCall));
  const spreadWidth = Math.max(1, K * 0.05);
  const K2 = isCall ? K + spreadWidth : K - spreadWidth;
  const premium2 = Math.max(0.01, approxOptionPremium(spot, K2, T, iv, isCall));
  const sMin = Math.max(1, Math.min(spot, K) * 0.65);
  const sMax = Math.max(spot, K) * 1.35;
  const x = Array.from(
    { length: 80 },
    (_, i) => sMin + ((sMax - sMin) * i) / 79,
  );
  const singleY = x.map((s) => {
    const intrinsic = isCall ? Math.max(0, s - K) : Math.max(0, K - s);
    const raw = intrinsic - premium;
    return side === "LONG_VOL" ? raw : -raw;
  });
  const verticalY = x.map((s) => {
    const intrinsic1 = isCall ? Math.max(0, s - K) : Math.max(0, K - s);
    const intrinsic2 = isCall ? Math.max(0, s - K2) : Math.max(0, K2 - s);
    const netDebit = premium - premium2;
    const rawLongVertical = intrinsic1 - intrinsic2 - netDebit;
    return side === "LONG_VOL" ? rawLongVertical : -rawLongVertical;
  });
  const y = strategyMode === "single" ? singleY : verticalY;

  const strategyLabel =
    strategyMode === "single"
      ? `${side === "LONG_VOL" ? "LONG" : "SHORT"} ${isCall ? "CALL" : "PUT"}`
      : `${side === "LONG_VOL" ? "LONG" : "SHORT"} ${isCall ? "CALL" : "PUT"} VERTICAL`;

  const breakEven =
    strategyMode === "single"
      ? isCall
        ? K + premium
        : K - premium
      : side === "LONG_VOL"
        ? isCall
          ? K + (premium - premium2)
          : K - (premium - premium2)
        : isCall
          ? K + (premium - premium2)
          : K - (premium - premium2);

  const sabrSmile = sliceSmileForExpiry(
    result.sabr?.surface_by_expiry,
    opp.expiry_date,
  );
  const hestonSmile = sliceSmileForExpiry(
    result.heston?.surface_by_expiry,
    opp.expiry_date,
  );
  const sabrFit = sabrFitForExpiry(result, opp.expiry_date);
  const hasSmile = sabrSmile.length > 0 || hestonSmile.length > 0;
  const smileChartTop = Math.max(
    iv,
    ...sabrSmile.map((p) => p.vol),
    ...hestonSmile.map((p) => p.vol),
    0.05,
  );

  const verticalHedge = strategyMode === "vertical";
  const K2ForHedge = verticalHedge ? K2 : null;
  const optionSign = side === "LONG_VOL" ? 1 : -1;
  const sigmaHedge = Math.max(iv, 0.05);
  const apiRiskFree = result.rates?.risk_free;
  const hedgeR =
    typeof apiRiskFree === "number" &&
    Number.isFinite(apiRiskFree) &&
    apiRiskFree >= 0
      ? apiRiskFree
      : RISK_FREE_FALLBACK;
  const hedgeRIsFromJob =
    typeof apiRiskFree === "number" &&
    Number.isFinite(apiRiskFree) &&
    apiRiskFree >= 0;

  const hedgeSim = useMemo(() => {
    if (!Number.isFinite(spot) || spot <= 0 || !Number.isFinite(T) || T <= 0)
      return null;
    const nSteps = Math.max(8, Math.min(120, hedgeSteps));
    const nPaths = 80;
    const base =
      (hashInt(`${opp.expiry_date}-${K}-${hedgeRun}`) ^ (nSteps * 7919)) >>> 0;
    const terminalErrors: number[] = [];
    let demo: HedgePathResult | null = null;
    for (let p = 0; p < nPaths; p += 1) {
      const rng = mulberry32(base + p * 100003);
      const path = simulateDeltaHedgePath({
        S0: spot,
        K,
        K2: K2ForHedge,
        T,
        sigma: sigmaHedge,
        isCall,
        r: hedgeR,
        vertical: verticalHedge,
        optionSign,
        nSteps,
        rng,
      });
      terminalErrors.push(path.terminalError);
      if (p === 0) demo = path;
    }
    const mean =
      terminalErrors.reduce((a, b) => a + b, 0) / terminalErrors.length;
    const variance =
      terminalErrors.reduce((s, e) => s + (e - mean) ** 2, 0) /
      Math.max(terminalErrors.length - 1, 1);
    const std = Math.sqrt(variance);
    const sorted = [...terminalErrors].sort((a, b) => a - b);
    const q = (pct: number) =>
      sorted[
        Math.max(
          0,
          Math.min(sorted.length - 1, Math.floor(pct * (sorted.length - 1))),
        )
      ];
    return {
      demo,
      terminalErrors,
      mean,
      std,
      p5: q(0.05),
      p95: q(0.95),
      nSteps,
      nPaths,
    };
  }, [
    spot,
    K,
    K2ForHedge,
    T,
    iv,
    isCall,
    verticalHedge,
    optionSign,
    hedgeSteps,
    hedgeRun,
    opp.expiry_date,
    opp.strike,
    opp.is_call,
    opp.vol_arb_side,
    sigmaHedge,
    hedgeR,
  ]);

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 p-4"
      role="dialog"
      aria-modal="true"
      aria-labelledby="opp-modal-title"
      onMouseDown={(e) => {
        if (e.target === e.currentTarget) onClose();
      }}
    >
      <div className="max-h-[92vh] w-full max-w-4xl overflow-y-auto rounded-xl border border-zinc-700 bg-zinc-900 p-5 shadow-2xl">
        <div className="mb-4 flex items-start justify-between gap-3">
          <div>
            <h2
              id="opp-modal-title"
              className="text-xl font-semibold text-zinc-100"
            >
              Opportunity detail
            </h2>
            <p className="mt-1 text-sm text-zinc-400">
              {String(opp.expiry_date).slice(0, 10)} · K {K.toFixed(2)} ·{" "}
              {isCall ? "Call" : "Put"} · mkt IV {iv.toFixed(4)} · model IV{" "}
              {Number(opp.model_iv).toFixed(4)} · {opp.vol_arb_side}
            </p>
          </div>
          <button
            type="button"
            onClick={onClose}
            className="shrink-0 rounded-md border border-zinc-600 px-3 py-1.5 text-sm text-zinc-200 hover:bg-zinc-800"
          >
            Close
          </button>
        </div>

        <div className="mb-4 space-y-2 text-sm text-zinc-300">
          <div>
            Suggested play:{" "}
            <span className="font-semibold text-zinc-100">{strategyLabel}</span>
            <span className="ml-2 text-zinc-400">
              (premium est. {premium.toFixed(2)}, T ≈ {T.toFixed(4)} y)
            </span>
          </div>
          <div className="flex flex-wrap items-center gap-3 text-xs">
            <label className="flex items-center gap-2 text-zinc-300">
              Structure
              <select
                value={strategyMode}
                onChange={(e) =>
                  setStrategyMode(e.target.value as StrategyMode)
                }
                className="rounded border border-zinc-700 bg-zinc-950 px-2 py-1 text-zinc-200"
              >
                <option value="vertical">Vertical spread</option>
                <option value="single">Single option</option>
              </select>
            </label>
            {sabrFit ? (
              <span className="rounded bg-zinc-800 px-2 py-1 text-zinc-400">
                SABR fit: RMSE{" "}
                {typeof sabrFit.rmse === "number"
                  ? sabrFit.rmse.toFixed(5)
                  : "—"}
                {sabrFit.status ? ` · ${sabrFit.status}` : ""}
              </span>
            ) : null}
          </div>
        </div>

        <div className="space-y-2">
          <h3 className="text-sm font-medium text-zinc-200">
            Payoff at expiration
          </h3>
          <p className="text-xs leading-relaxed text-zinc-500">
            This is a <span className="text-zinc-400">what-if at maturity</span>
            : estimated profit or loss (vertical axis) if the stock finishes at
            the price on the horizontal axis. It is a simplified illustration
            using Black–Scholes-style premiums, not live bid/ask. Dotted lines
            mark today’s stock price, the strike, and a rough break-even level.
          </p>
          <Plot
            data={[
              {
                type: "scatter",
                mode: "lines",
                x,
                y,
                line: { color: "#818cf8", width: 3 },
                name: "Est. P/L at expiry (1 contract notional)",
                hovertemplate:
                  "If stock ends at %{x:.2f}: est. P/L %{y:.2f}<extra></extra>",
              },
            ]}
            layout={{
              autosize: true,
              margin: { l: 52, r: 16, b: 52, t: 72 },
              paper_bgcolor: "rgba(0,0,0,0)",
              plot_bgcolor: "rgba(0,0,0,0)",
              title: {
                text: "Payoff diagram — who wins if the stock lands here?",
                font: { size: 13, color: "#e4e4e7" },
                x: 0,
                xanchor: "left",
              },
              xaxis: {
                title: { text: "Stock price at expiration (US$)" },
                gridcolor: "#27272a",
              },
              yaxis: {
                title: {
                  text: "Estimated profit / loss (US$, per-contract style)",
                },
                zeroline: true,
                zerolinewidth: 2,
                gridcolor: "#27272a",
              },
              showlegend: true,
              legend: {
                orientation: "h",
                y: 1.12,
                x: 0,
                font: { size: 11, color: "#a1a1aa" },
              },
              shapes: [
                {
                  type: "line",
                  x0: spot,
                  x1: spot,
                  y0: Math.min(...y),
                  y1: Math.max(...y),
                  line: { color: "#a1a1aa", width: 1, dash: "dot" },
                },
                {
                  type: "line",
                  x0: K,
                  x1: K,
                  y0: Math.min(...y),
                  y1: Math.max(...y),
                  line: { color: "#818cf8", width: 1, dash: "dot" },
                },
                {
                  type: "line",
                  x0: breakEven,
                  x1: breakEven,
                  y0: Math.min(...y),
                  y1: Math.max(...y),
                  line: { color: "#34d399", width: 1, dash: "dot" },
                },
              ],
              annotations: [
                {
                  x: spot,
                  y: Math.max(...y),
                  text: "Spot today",
                  showarrow: false,
                  yshift: 10,
                  font: { size: 11, color: "#d4d4d8" },
                },
                {
                  x: K,
                  y: Math.max(...y),
                  text: "Strike K",
                  showarrow: false,
                  yshift: 10,
                  font: { size: 11, color: "#c7d2fe" },
                },
                {
                  x: breakEven,
                  y: Math.min(...y),
                  text: "Break-even (approx.)",
                  showarrow: false,
                  yshift: -12,
                  font: { size: 11, color: "#6ee7b7" },
                },
              ],
            }}
            style={{ width: "100%", height: "340px" }}
            useResizeHandler
            config={{ displaylogo: false }}
          />
        </div>

        <div className="mt-6 space-y-2">
          <h3 className="text-sm font-medium text-zinc-200">
            Volatility smile (this expiration only)
          </h3>
          <p className="text-xs leading-relaxed text-zinc-500">
            For <span className="text-zinc-400">one expiration date</span>, each
            curve shows how much volatility the model implies across strikes
            (the “smile” or “skew”). The yellow marker is the option chain’s
            market implied vol at your strike; the dashed line shows that strike
            on the axis. When model curves sit below the marker, the market is
            pricing relatively richer volatility than the model at that point.
          </p>
          {!hasSmile ? (
            <p className="text-sm text-zinc-500">
              No smile slice available for this expiry yet.
            </p>
          ) : (
            <Plot
              data={[
                ...(sabrSmile.length
                  ? [
                      {
                        type: "scatter",
                        mode: "lines",
                        x: sabrSmile.map((p) => p.strike),
                        y: sabrSmile.map((p) => p.vol),
                        name: "SABR model IV by strike",
                        line: { color: "#22d3ee", width: 2 },
                        hovertemplate:
                          "SABR<br>Strike %{x:.2f}<br>IV %{y:.4f}<extra></extra>",
                      },
                    ]
                  : []),
                ...(hestonSmile.length
                  ? [
                      {
                        type: "scatter",
                        mode: "lines",
                        x: hestonSmile.map((p) => p.strike),
                        y: hestonSmile.map((p) => p.vol),
                        name: "Heston model IV by strike",
                        line: { color: "#f472b6", width: 2 },
                        hovertemplate:
                          "Heston<br>Strike %{x:.2f}<br>IV %{y:.4f}<extra></extra>",
                      },
                    ]
                  : []),
                {
                  type: "scatter",
                  mode: "markers",
                  x: [K],
                  y: [iv],
                  name: "Market IV at this strike (chain)",
                  marker: { size: 10, color: "#fbbf24", symbol: "diamond" },
                  hovertemplate:
                    "Market IV<br>Strike %{x:.2f}<br>IV %{y:.4f}<extra></extra>",
                },
              ]}
              layout={{
                autosize: true,
                margin: { l: 52, r: 12, b: 52, t: 64 },
                paper_bgcolor: "rgba(0,0,0,0)",
                plot_bgcolor: "rgba(0,0,0,0)",
                title: {
                  text: "Implied volatility vs strike — same expiry as the selected row",
                  font: { size: 13, color: "#e4e4e7" },
                  x: 0,
                  xanchor: "left",
                },
                legend: {
                  orientation: "h",
                  y: 1.18,
                  x: 0,
                  font: { color: "#a1a1aa", size: 11 },
                },
                xaxis: {
                  title: { text: "Strike price (US$)" },
                  gridcolor: "#27272a",
                },
                yaxis: {
                  title: {
                    text: "Implied volatility σ (annualized; higher = pricier options)",
                  },
                  gridcolor: "#27272a",
                },
                annotations: [
                  {
                    x: K,
                    y: smileChartTop,
                    text: "Selected strike (this contract)",
                    showarrow: false,
                    yshift: 16,
                    font: { size: 11, color: "#d8b4fe" },
                  },
                ],
                shapes: [
                  {
                    type: "line",
                    x0: K,
                    x1: K,
                    y0: 0,
                    y1: 1,
                    yref: "paper",
                    line: { color: "#a78bfa", width: 1, dash: "dash" },
                  },
                ],
              }}
              style={{ width: "100%", height: "320px" }}
              useResizeHandler
              config={{ displaylogo: false }}
            />
          )}
        </div>

        <div className="mt-8 space-y-3 border-t border-zinc-800 pt-6">
          <h3 className="text-sm font-medium text-zinc-200">
            Delta-hedged simulation
          </h3>
          <p className="text-xs text-zinc-500">
            Black–Scholes delta hedge with flat σ = mkt IV and r ={" "}
            {(hedgeR * 100).toFixed(2)}%
            {hedgeRIsFromJob
              ? " — the same annualized risk-free rate shipped with this analysis result (aligned with how the backend prices options)."
              : " — using a small fixed fallback until the completed result includes a risk-free rate (for example while the stream is still partial)."}{" "}
            GBM paths use that r and σ; each step you rebalance stock to the net
            Black–Scholes delta for your structure (single or vertical, long or
            short vol). With continuous hedging the mark-to-market book stays
            near zero; coarser steps show gamma and rebalance slippage, so
            terminal cash plus stock plus option payoff clusters near zero.
          </p>
          <div className="flex flex-wrap items-center gap-3 text-xs text-zinc-300">
            <label className="flex items-center gap-2">
              Steps to expiry
              <select
                value={hedgeSteps}
                onChange={(e) => setHedgeSteps(Number(e.target.value))}
                className="rounded border border-zinc-700 bg-zinc-950 px-2 py-1 text-zinc-200"
              >
                <option value={24}>24 (coarser)</option>
                <option value={52}>52 (~weekly)</option>
                <option value={96}>96 (finer)</option>
              </select>
            </label>
            <button
              type="button"
              onClick={() => setHedgeRun((n) => n + 1)}
              className="rounded border border-zinc-600 px-2 py-1 text-zinc-200 hover:bg-zinc-800"
            >
              New paths
            </button>
            {hedgeSim ? (
              <span className="text-zinc-500">
                {hedgeSim.nPaths} paths · {hedgeSim.nSteps} steps · mean
                terminal error{" "}
                <span className="font-mono text-zinc-300">
                  {hedgeSim.mean.toFixed(3)}
                </span>{" "}
                (stdev {hedgeSim.std.toFixed(3)}, P5–P95{" "}
                {hedgeSim.p5.toFixed(2)} … {hedgeSim.p95.toFixed(2)})
              </span>
            ) : null}
          </div>
          {!hedgeSim?.demo ? (
            <p className="text-sm text-zinc-500">
              Not enough data to run hedge simulation.
            </p>
          ) : (
            <div className="space-y-4">
              <Plot
                data={[
                  {
                    type: "scatter",
                    mode: "lines",
                    x: hedgeSim.demo.times,
                    y: hedgeSim.demo.markToMarket,
                    name: "Hedged portfolio (cash + shares + option mark)",
                    line: { color: "#a5b4fc", width: 2 },
                    hovertemplate:
                      "Time %{x:.3f} y<br>Hedged book value %{y:.3f}<extra></extra>",
                  },
                  {
                    type: "scatter",
                    mode: "lines",
                    x: hedgeSim.demo.times,
                    y: hedgeSim.demo.spotPath,
                    name: "Stock price path (same simulation)",
                    line: { color: "#fbbf24", width: 1.5 },
                    yaxis: "y2",
                    hovertemplate:
                      "Time %{x:.3f} y<br>Spot %{y:.2f}<extra></extra>",
                  },
                ]}
                layout={{
                  autosize: true,
                  margin: { l: 56, r: 64, b: 52, t: 80 },
                  paper_bgcolor: "rgba(0,0,0,0)",
                  plot_bgcolor: "rgba(0,0,0,0)",
                  title: {
                    text: "One random path: does delta-hedging keep you “flat”?",
                    font: { size: 13, color: "#e4e4e7" },
                    x: 0,
                    xanchor: "left",
                  },
                  xaxis: {
                    title: {
                      text: `Elapsed time (years; axis spans 0 → T, here T ≈ ${T.toFixed(3)})`,
                    },
                    gridcolor: "#27272a",
                  },
                  yaxis: {
                    title: {
                      text: "Mark-to-market hedged book (US$; ~0 if hedging were continuous)",
                    },
                    zeroline: true,
                    zerolinewidth: 2,
                    gridcolor: "#27272a",
                  },
                  yaxis2: {
                    title: { text: "Stock price in this path (US$)" },
                    overlaying: "y",
                    side: "right",
                    showgrid: false,
                  },
                  legend: {
                    orientation: "h",
                    y: 1.2,
                    x: 0,
                    font: { size: 10, color: "#a1a1aa" },
                  },
                }}
                style={{ width: "100%", height: "340px" }}
                useResizeHandler
                config={{ displaylogo: false }}
              />
              <Plot
                data={[
                  {
                    type: "histogram",
                    x: hedgeSim.terminalErrors,
                    nbinsx: 22,
                    name: "How far off was the hedge at expiry?",
                    marker: {
                      color: "#64748b",
                      line: { color: "#475569", width: 1 },
                    },
                  },
                ]}
                layout={{
                  autosize: true,
                  margin: { l: 52, r: 16, b: 56, t: 88 },
                  paper_bgcolor: "rgba(0,0,0,0)",
                  plot_bgcolor: "rgba(0,0,0,0)",
                  title: {
                    text: "Many simulated paths: leftover cash at expiry should cluster near zero",
                    font: { size: 13, color: "#e4e4e7" },
                    x: 0,
                    xanchor: "left",
                  },
                  xaxis: {
                    title: {
                      text: "Terminal hedge imbalance (US$; 0 = perfect discrete hedge vs theory)",
                    },
                    zeroline: true,
                    zerolinewidth: 2,
                    gridcolor: "#27272a",
                  },
                  yaxis: {
                    title: { text: "Number of paths in each bucket" },
                    gridcolor: "#27272a",
                  },
                  shapes: [
                    {
                      type: "line",
                      x0: 0,
                      x1: 0,
                      y0: 0,
                      y1: 1,
                      yref: "paper",
                      line: { color: "#34d399", width: 1, dash: "dot" },
                    },
                  ],
                  annotations: [
                    {
                      xref: "paper",
                      yref: "paper",
                      x: 0,
                      y: 1.06,
                      xanchor: "left",
                      text: "Green line: zero leftover — where a perfect discrete hedge would finish.",
                      showarrow: false,
                      font: { size: 10, color: "#94a3b8" },
                    },
                  ],
                }}
                style={{ width: "100%", height: "300px" }}
                useResizeHandler
                config={{ displaylogo: false }}
              />
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default function App() {
  const [ticker, setTicker] = useState("AAPL");
  const [jobId, setJobId] = useState("");
  const [status, setStatus] = useState("idle");
  const [events, setEvents] = useState<StreamEvent[]>([]);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [error, setError] = useState("");
  const [selectedOpp, setSelectedOpp] = useState<Opportunity | null>(null);
  const [strategyMode, setStrategyMode] = useState<StrategyMode>("vertical");

  const sabrSurface = useMemo(
    () => buildSurface(result?.sabr?.surface_by_expiry),
    [result],
  );
  const hestonSurface = useMemo(
    () => buildSurface(result?.heston?.surface_by_expiry),
    [result],
  );
  const isRunning = status === "running" || status === "starting";
  const selectedKey = selectedOpp ? oppRowKey(selectedOpp) : "";

  async function startAnalysis(e: React.FormEvent<HTMLFormElement>) {
    e.preventDefault();
    setError("");
    setResult(null);
    setSelectedOpp(null);
    setEvents([]);
    setStatus("starting");

    const res = await fetch("/api/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ticker }),
    });
    const body = await res.json();
    if (!res.ok) {
      setStatus("failed");
      setError(body.error || "Failed to start analysis");
      return;
    }

    const id = body.job_id as string;
    setJobId(id);
    setStatus("running");
    let streamClosedNormally = false;

    const es = new EventSource(`/api/jobs/${id}/stream`);

    const appendEvent = (evt: MessageEvent<string>) => {
      try {
        const parsed = JSON.parse(evt.data) as StreamEvent;
        setEvents((prev) => [parsed, ...prev].slice(0, 30));
        // Stream partial model payloads as soon as each finishes.
        if (parsed.type === "sabr_ready" && parsed.sabr) {
          setResult((prev) => ({
            ...(prev || {}),
            sabr: parsed.sabr as AnalysisResult["sabr"],
            spot: prev?.spot,
            rates: prev?.rates,
            heston: prev?.heston,
            consensus: prev?.consensus,
          }));
        }
        if (parsed.type === "heston_ready" && parsed.heston) {
          setResult((prev) => ({
            ...(prev || {}),
            heston: parsed.heston as AnalysisResult["heston"],
            spot: prev?.spot,
            rates: prev?.rates,
            sabr: prev?.sabr,
            consensus: prev?.consensus,
          }));
        }
      } catch {
        // Ignore unparsable keep-alive messages.
      }
    };

    // Backend emits named SSE events; subscribe explicitly.
    es.addEventListener("connected", appendEvent);
    es.addEventListener("status", appendEvent);
    es.addEventListener("progress", appendEvent);
    es.addEventListener("sabr_ready", appendEvent);
    es.addEventListener("heston_ready", appendEvent);
    es.addEventListener("completed", appendEvent);
    es.addEventListener("final", appendEvent);

    // Keep fallback in case server emits unnamed "message" events.
    es.onmessage = appendEvent;

    es.addEventListener("completed", async () => {
      streamClosedNormally = true;
      const finalRes = await fetch(`/api/jobs/${id}/result`);
      const finalJson = await finalRes.json();
      if (finalRes.ok && finalJson.result) {
        setResult(finalJson.result as AnalysisResult);
        setStatus("completed");
      } else {
        setStatus("failed");
        setError("Result endpoint did not return final payload.");
      }
      es.close();
    });
    es.addEventListener("final", async () => {
      streamClosedNormally = true;
      const finalRes = await fetch(`/api/jobs/${id}/result`);
      const finalJson = await finalRes.json();
      if (finalRes.ok && finalJson.result) {
        setResult(finalJson.result as AnalysisResult);
        setStatus("completed");
      }
      es.close();
    });
    es.addEventListener("error", async (evt) => {
      if (streamClosedNormally) {
        es.close();
        return;
      }
      appendEvent(evt as MessageEvent<string>);
      const statusRes = await fetch(`/api/jobs/${id}`);
      const statusJson = await statusRes.json();
      if (statusJson.status === "completed") {
        setStatus("completed");
        setError("");
        es.close();
        return;
      }
      setStatus((statusJson.status as string) || "failed");
      setError((statusJson.error as string) || "Stream error");
      es.close();
    });
  }

  return (
    <main className="min-h-screen bg-zinc-950 p-6 text-zinc-100">
      <div className="mx-auto max-w-7xl space-y-6">
        <header>
          <h1 className="text-3xl font-bold">VolLens</h1>
          <p className="text-sm text-zinc-400">
            Streamed calibration of two option pricing models (SABR and Heston),
            shown as{" "}
            <span className="text-zinc-300">3D volatility surfaces</span> (how
            expensive options are across strikes and dates), plus tables of
            where each model disagrees with market prices.
          </p>
        </header>

        <form
          onSubmit={startAnalysis}
          className="flex flex-wrap items-end gap-3 rounded-xl border border-zinc-800 bg-zinc-900/60 p-4"
        >
          <label className="flex flex-col gap-1 text-sm">
            <span className="text-zinc-300">Ticker</span>
            <input
              value={ticker}
              onChange={(e) => setTicker(e.target.value.toUpperCase())}
              className="rounded-md border border-zinc-700 bg-zinc-950 px-3 py-2 outline-none focus:border-indigo-400"
              placeholder="AAPL"
              maxLength={8}
            />
          </label>
          <button
            type="submit"
            className="rounded-md bg-indigo-600 px-4 py-2 font-medium hover:bg-indigo-500 disabled:opacity-50"
            disabled={status === "running" || ticker.trim() === ""}
          >
            {status === "running" ? "Running..." : "Run Analysis"}
          </button>
          <div className="text-sm text-zinc-400">
            Status: <span className="text-zinc-200">{status}</span>
            {jobId ? (
              <span className="ml-2">Job: {jobId.slice(0, 8)}...</span>
            ) : null}
          </div>
        </form>

        {error ? (
          <div className="rounded-lg border border-red-800 bg-red-950/40 p-3 text-sm text-red-200">
            {error}
          </div>
        ) : null}

        <section className="grid gap-4 md:grid-cols-2">
          <div className="rounded-xl border border-zinc-800 bg-zinc-900/60 p-4">
            <h2 className="text-lg font-semibold">SABR surface</h2>
            <p className="mb-2 mt-1 text-xs leading-relaxed text-zinc-500">
              SABR is a standard model for how implied volatility varies by
              strike and expiry. Height and color show{" "}
              <span className="text-zinc-400">model implied volatility</span>{" "}
              (annualized): higher means the model prices larger expected
              swings. Compare with Heston to see where the two disagree.
            </p>
            {isRunning && !sabrSurface ? (
              <LoadingPanel label="Generating SABR surface..." />
            ) : sabrSurface ? (
              <Plot
                data={[
                  {
                    type: "surface",
                    x: sabrSurface.x,
                    y: sabrSurface.y,
                    z: sabrSurface.z,
                    colorscale: "Viridis",
                    name: "Model IV",
                    hovertemplate:
                      "Strike: %{x:.2f}<br>Expiry index: %{y}<br>Implied vol (model): %{z:.4f}<extra></extra>",
                  },
                ]}
                layout={{
                  autosize: true,
                  margin: { l: 0, r: 0, b: 0, t: 28 },
                  paper_bgcolor: "rgba(0,0,0,0)",
                  plot_bgcolor: "rgba(0,0,0,0)",
                  title: {
                    text: "SABR: implied volatility by strike and expiration",
                    font: { size: 13, color: "#d4d4d8" },
                    x: 0.02,
                    xanchor: "left",
                  },
                  scene: {
                    xaxis: {
                      title: { text: "Strike price (US$)" },
                      backgroundcolor: "rgba(24,24,27,0.4)",
                      gridcolor: "#3f3f46",
                      showbackground: true,
                    },
                    yaxis: {
                      title: {
                        text: "Expiration (each tick is a different expiry date)",
                      },
                      tickmode: "array",
                      tickvals: sabrSurface.y,
                      ticktext: sabrSurface.yLabels.map((d) =>
                        String(d).slice(0, 10),
                      ),
                      backgroundcolor: "rgba(24,24,27,0.4)",
                      gridcolor: "#3f3f46",
                      showbackground: true,
                    },
                    zaxis: {
                      title: {
                        text: "Implied volatility σ (annualized, model)",
                      },
                      backgroundcolor: "rgba(24,24,27,0.4)",
                      gridcolor: "#3f3f46",
                      showbackground: true,
                    },
                    camera: { eye: { x: 1.35, y: -1.6, z: 0.9 } },
                  },
                }}
                style={{ width: "100%", height: "420px" }}
                useResizeHandler
                config={{ displaylogo: false }}
              />
            ) : (
              <p className="text-sm text-zinc-400">No SABR surface yet.</p>
            )}
          </div>
          <div className="rounded-xl border border-zinc-800 bg-zinc-900/60 p-4">
            <h2 className="text-lg font-semibold">Heston surface</h2>
            <p className="mb-2 mt-1 text-xs leading-relaxed text-zinc-500">
              Heston allows volatility itself to move randomly (stochastic
              volatility), which often fits equity options better than a single
              flat vol. The surface shows the model’s{" "}
              <span className="text-zinc-400">implied volatility</span> after
              calibration.
            </p>
            {isRunning && !hestonSurface ? (
              <LoadingPanel label="Generating Heston surface..." />
            ) : hestonSurface ? (
              <Plot
                data={[
                  {
                    type: "surface",
                    x: hestonSurface.x,
                    y: hestonSurface.y,
                    z: hestonSurface.z,
                    colorscale: "Plasma",
                    name: "Model IV",
                    hovertemplate:
                      "Strike: %{x:.2f}<br>Expiry index: %{y}<br>Implied vol (model): %{z:.4f}<extra></extra>",
                  },
                ]}
                layout={{
                  autosize: true,
                  margin: { l: 0, r: 0, b: 0, t: 28 },
                  paper_bgcolor: "rgba(0,0,0,0)",
                  plot_bgcolor: "rgba(0,0,0,0)",
                  title: {
                    text: "Heston: implied volatility by strike and expiration",
                    font: { size: 13, color: "#d4d4d8" },
                    x: 0.02,
                    xanchor: "left",
                  },
                  scene: {
                    xaxis: {
                      title: { text: "Strike price (US$)" },
                      backgroundcolor: "rgba(24,24,27,0.4)",
                      gridcolor: "#3f3f46",
                      showbackground: true,
                    },
                    yaxis: {
                      title: {
                        text: "Expiration (each tick is a different expiry date)",
                      },
                      tickmode: "array",
                      tickvals: hestonSurface.y,
                      ticktext: hestonSurface.yLabels.map((d) =>
                        String(d).slice(0, 10),
                      ),
                      backgroundcolor: "rgba(24,24,27,0.4)",
                      gridcolor: "#3f3f46",
                      showbackground: true,
                    },
                    zaxis: {
                      title: {
                        text: "Implied volatility σ (annualized, model)",
                      },
                      backgroundcolor: "rgba(24,24,27,0.4)",
                      gridcolor: "#3f3f46",
                      showbackground: true,
                    },
                    camera: { eye: { x: 1.35, y: -1.6, z: 0.9 } },
                  },
                }}
                style={{ width: "100%", height: "420px" }}
                useResizeHandler
                config={{ displaylogo: false }}
              />
            ) : (
              <p className="text-sm text-zinc-400">No Heston surface yet.</p>
            )}
          </div>
        </section>

        <section className="rounded-xl border border-zinc-800 bg-zinc-900/60 p-4">
          <h3 className="text-lg font-semibold">
            Consensus signals (SABR + Heston)
          </h3>
          <p className="mb-3 mt-1 text-xs leading-relaxed text-zinc-500">
            Rows where <span className="text-zinc-400">both</span> models
            disagree with the market in a similar direction. Consensus spread is
            roughly the average of each model’s IV gap; agreement score
            summarizes how aligned the two models are. Click a row for payoff,
            smile slice, and hedge toy examples.
          </p>
          {isRunning && !result?.consensus?.signals ? (
            <div className="flex h-40 flex-col items-center justify-center gap-3 text-zinc-300">
              <div className="h-7 w-7 animate-spin rounded-full border-2 border-zinc-700 border-t-indigo-400" />
              <p className="text-sm">Computing model agreement...</p>
            </div>
          ) : (result?.consensus?.signals?.length || 0) === 0 ? (
            <p className="text-sm text-zinc-400">
              No consensus signals found yet.
            </p>
          ) : (
            <div className="overflow-auto">
              <table className="min-w-full text-left text-sm">
                <thead className="text-zinc-400">
                  <tr>
                    <th className="px-2 py-1">
                      <span
                        className="cursor-help decoration-dotted underline-offset-2 hover:underline"
                        title="Option expiration date."
                      >
                        Expiry
                      </span>
                    </th>
                    <th className="px-2 py-1">
                      <span
                        className="cursor-help decoration-dotted underline-offset-2 hover:underline"
                        title="Strike price."
                      >
                        Strike
                      </span>
                    </th>
                    <th className="px-2 py-1">Type</th>
                    <th className="px-2 py-1">
                      <span
                        className="cursor-help decoration-dotted underline-offset-2 hover:underline"
                        title="Average of SABR and Heston IV spreads."
                      >
                        Consensus Spread
                      </span>
                    </th>
                    <th className="px-2 py-1">
                      <span
                        className="cursor-help decoration-dotted underline-offset-2 hover:underline"
                        title="Higher means both models strongly agree on dislocation."
                      >
                        Agreement Score
                      </span>
                    </th>
                    <th className="px-2 py-1">
                      <span
                        className="cursor-help decoration-dotted underline-offset-2 hover:underline"
                        title="Rough directional read from both models: buy_vol means models see options as cheap vs market; sell_vol means rich."
                      >
                        Side
                      </span>
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {(result?.consensus?.signals || [])
                    .slice(0, 12)
                    .map((r, i) => (
                      <tr
                        key={`${r.expiry_date}-${r.strike}-${i}`}
                        className="cursor-pointer border-t border-zinc-800 hover:bg-zinc-800/60"
                        onClick={() => {
                          const T = findRowT(
                            result,
                            r.expiry_date,
                            r.strike,
                            r.is_call,
                          );
                          setSelectedOpp({
                            expiry_date: r.expiry_date,
                            strike: r.strike,
                            is_call: r.is_call,
                            mkt_iv: r.mkt_iv,
                            model_iv: r.mkt_iv + r.consensus_spread,
                            iv_spread: r.consensus_spread,
                            spread_zscore: r.agreement_score,
                            vol_arb_side: r.consensus_side,
                            ...(T != null ? { T } : {}),
                          });
                        }}
                      >
                        <td className="px-2 py-1">
                          {String(r.expiry_date).slice(0, 10)}
                        </td>
                        <td className="px-2 py-1">
                          {Number(r.strike).toFixed(2)}
                        </td>
                        <td className="px-2 py-1">{r.is_call ? "C" : "P"}</td>
                        <td className="px-2 py-1">
                          {Number(r.consensus_spread).toFixed(4)}
                        </td>
                        <td className="px-2 py-1">
                          {Number(r.agreement_score).toFixed(3)}
                        </td>
                        <td className="px-2 py-1">{r.consensus_side}</td>
                      </tr>
                    ))}
                </tbody>
              </table>
            </div>
          )}
        </section>

        <section className="grid gap-4 md:grid-cols-2">
          {isRunning && !result?.sabr?.opportunities ? (
            <section className="rounded-xl border border-zinc-800 bg-zinc-900/60 p-4">
              <h3 className="mb-3 text-lg font-semibold">SABR opportunities</h3>
              <div className="flex h-48 flex-col items-center justify-center gap-3 text-zinc-300">
                <div className="h-7 w-7 animate-spin rounded-full border-2 border-zinc-700 border-t-indigo-400" />
                <p className="text-sm">Scoring SABR opportunities...</p>
              </div>
            </section>
          ) : (
            <OppTable
              title="SABR opportunities"
              caption="Single-model view: largest disagreements between the SABR smile and market implied vols. Use as one lens; compare with Heston and consensus."
              rows={result?.sabr?.opportunities || []}
              onSelect={setSelectedOpp}
              selectedKey={selectedKey}
            />
          )}
          {isRunning && !result?.heston?.opportunities ? (
            <section className="rounded-xl border border-zinc-800 bg-zinc-900/60 p-4">
              <h3 className="mb-3 text-lg font-semibold">
                Heston opportunities
              </h3>
              <div className="flex h-48 flex-col items-center justify-center gap-3 text-zinc-300">
                <div className="h-7 w-7 animate-spin rounded-full border-2 border-zinc-700 border-t-indigo-400" />
                <p className="text-sm">Scoring Heston opportunities...</p>
              </div>
            </section>
          ) : (
            <OppTable
              title="Heston opportunities"
              caption="Same idea as SABR but using the stochastic-volatility Heston model. Where SABR and Heston both point the same way, the consensus table above is more informative."
              rows={result?.heston?.opportunities || []}
              onSelect={setSelectedOpp}
              selectedKey={selectedKey}
            />
          )}
        </section>

        <p className="mx-auto max-w-2xl text-center text-xs leading-relaxed text-zinc-500">
          Tip: click any row above to open a detail window with a{" "}
          <span className="text-zinc-400">payoff sketch</span> (what you might
          make at expiration if the stock finishes at different prices), a{" "}
          <span className="text-zinc-400">volatility smile slice</span> for that
          expiration, and a simple{" "}
          <span className="text-zinc-400">delta-hedge simulation</span> (how
          stock trading against the option can offset moves in the short run).
        </p>

        {selectedOpp && result?.spot != null ? (
          <OpportunityModal
            opp={selectedOpp}
            result={result}
            strategyMode={strategyMode}
            setStrategyMode={setStrategyMode}
            onClose={() => setSelectedOpp(null)}
          />
        ) : null}

        <section className="rounded-xl border border-zinc-800 bg-zinc-900/60 p-4">
          <h3 className="mb-2 text-lg font-semibold">Recent Stream Events</h3>
          <div className="max-h-80 space-y-2 overflow-auto rounded bg-zinc-950 p-3 text-sm text-zinc-200">
            {events.length === 0 ? (
              <p className="text-zinc-500">No events yet.</p>
            ) : (
              events.map((evt, i) => {
                const tone =
                  evt.type === "error"
                    ? "border-red-800/70 bg-red-950/30"
                    : evt.type === "completed" || evt.type === "final"
                      ? "border-emerald-800/70 bg-emerald-950/20"
                      : "border-zinc-800 bg-zinc-900/60";
                return (
                  <div key={i} className={`rounded-lg border p-3 ${tone}`}>
                    <div className="mb-1 flex items-center justify-between gap-3">
                      <div className="flex items-center gap-2">
                        <span className="rounded bg-zinc-800 px-2 py-0.5 text-xs uppercase tracking-wide text-zinc-300">
                          {evt.type || "message"}
                        </span>
                        {typeof evt.status === "string" ? (
                          <span className="text-xs text-zinc-400">
                            status: {evt.status}
                          </span>
                        ) : null}
                      </div>
                      <span className="text-xs text-zinc-500">
                        {typeof evt.ts === "string"
                          ? new Date(evt.ts).toLocaleTimeString()
                          : ""}
                      </span>
                    </div>

                    {typeof evt.message === "string" ? (
                      <p className="mb-1 text-zinc-200">{evt.message}</p>
                    ) : null}
                    {typeof evt.step === "string" ? (
                      <p className="mb-1 text-xs text-indigo-300">
                        step: {evt.step}
                      </p>
                    ) : null}

                    <details className="mt-2">
                      <summary className="cursor-pointer text-xs text-zinc-400 hover:text-zinc-200">
                        details
                      </summary>
                      <pre className="mt-2 overflow-auto rounded bg-zinc-950 p-2 font-mono text-xs text-zinc-300">
                        {JSON.stringify(evt, null, 2)}
                      </pre>
                    </details>
                  </div>
                );
              })
            )}
          </div>
        </section>
      </div>
    </main>
  );
}
