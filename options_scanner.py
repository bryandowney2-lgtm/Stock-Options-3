#!/usr/bin/env python3
"""
options_scanner.py  —  Directional options screener (calls + puts)

WHAT THIS DOES
  Pulls live option chains via yfinance and ranks contracts by a blended score
  built from several factors. It surfaces candidates worth a closer look. It is
  a RESEARCH AID, not a prediction. High score = high leverage + decent liquidity,
  NOT high probability of profit. You can lose 100% of an option's premium fast.

USAGE
  python options_scanner.py AAPL MSFT NVDA
  python options_scanner.py AAPL --side call --max-dte 45 --top 15
  python options_scanner.py SPY  --side put  --min-dte 7 --max-dte 30

  Requires:  pip install yfinance pandas numpy scipy
"""

import argparse
import sys
import warnings
from datetime import datetime, timezone

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    import yfinance as yf
except ImportError:
    sys.exit("Missing dependency. Run: pip install yfinance pandas numpy scipy")

from scipy.stats import norm


# ----------------------------------------------------------------------
# Black-Scholes greeks (used to estimate delta/gamma when scoring leverage)
# ----------------------------------------------------------------------
def bs_greeks(S, K, T, r, sigma, kind):
    """Return greeks dict for a European option. T in years, sigma annualized.

    Keys: delta, gamma, theta (per day), vega (per 1 vol point), pop (rough
    probability of finishing ITM, from delta).
    """
    nan = {"delta": np.nan, "gamma": np.nan, "theta": np.nan,
           "vega": np.nan, "pop": np.nan}
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return nan
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    pdf = norm.pdf(d1)

    if kind == "call":
        delta = norm.cdf(d1)
        theta = (-(S * pdf * sigma) / (2 * np.sqrt(T))
                 - r * K * np.exp(-r * T) * norm.cdf(d2))
    else:
        delta = norm.cdf(d1) - 1.0
        theta = (-(S * pdf * sigma) / (2 * np.sqrt(T))
                 + r * K * np.exp(-r * T) * norm.cdf(-d2))

    gamma = pdf / (S * sigma * np.sqrt(T))
    vega = S * pdf * np.sqrt(T) / 100.0      # per 1 percentage-point IV move
    theta = theta / 365.0                     # per calendar day
    # rough probability of finishing in-the-money
    pop = norm.cdf(d2) if kind == "call" else norm.cdf(-d2)
    return {"delta": delta, "gamma": gamma, "theta": theta, "vega": vega, "pop": pop}


def compute_direction(hist):
    """Blend trend + momentum into a score in [-1, +1].

    +1 strongly bullish, -1 strongly bearish, 0 neutral. Uses moving-average
    stack, price vs MAs, and RSI. Designed for short-term directional bias.
    """
    close = hist["Close"].dropna()
    if len(close) < 30:
        return 0.0

    sma10 = close.rolling(10).mean().iloc[-1]
    sma20 = close.rolling(20).mean().iloc[-1]
    sma50 = close.rolling(50).mean().iloc[-1] if len(close) >= 50 else sma20
    price = close.iloc[-1]

    signals = []
    # 1) price above/below short MA
    signals.append(1.0 if price > sma10 else -1.0)
    # 2) MA stack (10 over 20 over 50 = uptrend)
    if sma10 > sma20 > sma50:
        signals.append(1.0)
    elif sma10 < sma20 < sma50:
        signals.append(-1.0)
    else:
        signals.append(0.0)
    # 3) 10-day rate of change
    if len(close) >= 11:
        roc = (price - close.iloc[-11]) / close.iloc[-11]
        signals.append(float(np.clip(roc * 10, -1, 1)))
    # 4) RSI(14), recentered so 50 = neutral
    delta_c = close.diff()
    gain = delta_c.clip(lower=0).rolling(14).mean().iloc[-1]
    loss = (-delta_c.clip(upper=0)).rolling(14).mean().iloc[-1]
    if loss and loss > 0:
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        signals.append(float(np.clip((rsi - 50) / 50, -1, 1)))

    return float(np.clip(np.mean(signals), -1, 1))


# ----------------------------------------------------------------------
# Per-ticker scan
# ----------------------------------------------------------------------
def scan_ticker(symbol, side, min_dte, max_dte, max_bid=None, r=0.045):
    tk = yf.Ticker(symbol)

    # current price + recent realized volatility (for IV-vs-RV comparison).
    # Wrapped because yfinance can raise (not just return empty) when Yahoo
    # blocks or rate-limits the request — common on shared CI IPs.
    try:
        hist = tk.history(period="3mo")
    except Exception as e:
        print(f"  [skip] {symbol}: price fetch failed ({type(e).__name__})")
        return pd.DataFrame()
    if hist is None or hist.empty:
        print(f"  [skip] {symbol}: no price history")
        return pd.DataFrame()
    S = float(hist["Close"].iloc[-1])
    log_ret = np.log(hist["Close"] / hist["Close"].shift(1)).dropna()
    realized_vol = float(log_ret.std() * np.sqrt(252)) if len(log_ret) > 5 else np.nan

    # ---- directional signal: blend of trend + momentum, range -1 (bearish) .. +1 (bullish) ----
    direction = compute_direction(hist)

    try:
        expirations = tk.options
    except Exception as e:
        print(f"  [skip] {symbol}: options fetch failed ({type(e).__name__})")
        return pd.DataFrame()
    if not expirations:
        print(f"  [skip] {symbol}: no options listed")
        return pd.DataFrame()

    rows = []
    now = datetime.now(timezone.utc)

    for exp in expirations:
        exp_dt = datetime.strptime(exp, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        dte = (exp_dt - now).days
        if dte < min_dte or dte > max_dte:
            continue
        T = max(dte, 1) / 365.0

        try:
            chain = tk.option_chain(exp)
        except Exception:
            continue

        sides = []
        if side in ("call", "both"):
            sides.append(("call", chain.calls))
        if side in ("put", "both"):
            sides.append(("put", chain.puts))

        for kind, df in sides:
            if df is None or df.empty:
                continue
            df = df.copy()

            def num(val, default=0.0):
                """Coerce to float, treating NaN/None/blank as the default."""
                try:
                    f = float(val)
                except (TypeError, ValueError):
                    return default
                return default if np.isnan(f) else f

            for _, opt in df.iterrows():
                K = num(opt.get("strike"), np.nan)
                bid = num(opt.get("bid"))
                ask = num(opt.get("ask"))
                last = num(opt.get("lastPrice"))
                vol = num(opt.get("volume"))
                oi = num(opt.get("openInterest"))
                iv = num(opt.get("impliedVolatility"))

                mid = (bid + ask) / 2 if (bid > 0 and ask > 0) else last
                if mid <= 0 or K <= 0 or np.isnan(K) or np.isnan(mid):
                    continue

                # ---- cheap-premium filter: skip contracts whose bid exceeds the cap ----
                if max_bid is not None and bid > max_bid:
                    continue

                # ---- factor 1: leverage (delta * S / premium) ----
                g = bs_greeks(S, K, T, r, iv if iv > 0 else realized_vol, kind)
                delta, gamma = g["delta"], g["gamma"]
                theta, vega, pop = g["theta"], g["vega"], g["pop"]
                if np.isnan(delta):
                    continue
                leverage = abs(delta) * S / (mid * 100) if mid > 0 else 0

                # ---- factor 2: liquidity (volume + open interest, tight spread) ----
                spread_pct = (ask - bid) / mid if (ask > 0 and bid > 0 and mid > 0) else 1.0
                liquidity_raw = np.log1p(vol) + 0.5 * np.log1p(oi)
                tightness = max(0.0, 1.0 - min(spread_pct, 1.0))  # 1 = tight, 0 = wide

                # ---- factor 3: unusual activity (vol relative to OI) ----
                vol_oi = vol / oi if oi > 0 else 0.0

                # ---- factor 4: IV value (IV cheap vs realized = better) ----
                iv_ratio = (iv / realized_vol) if (realized_vol and realized_vol > 0 and iv > 0) else np.nan
                # reward IV below realized, penalize very rich IV
                iv_value = np.clip(1.5 - iv_ratio, -1, 1) if not np.isnan(iv_ratio) else 0.0

                # ---- factor 5: moneyness (favor slightly OTM for convexity) ----
                if kind == "call":
                    otm = (K - S) / S
                else:
                    otm = (S - K) / S
                # peak reward around 2-8% OTM
                moneyness_score = np.exp(-((otm - 0.05) ** 2) / (2 * 0.05 ** 2))

                # ---- factor 6: directional alignment ----
                # calls reward bullish direction, puts reward bearish.
                # direction in [-1,+1]; align in [0,1] where 1 = strongly with the trend.
                dir_signed = direction if kind == "call" else -direction
                align = (dir_signed + 1.0) / 2.0

                rows.append({
                    "symbol": symbol, "type": kind, "expiry": exp, "dte": dte,
                    "strike": K, "spot": round(S, 2), "mid": round(mid, 2),
                    "bid": bid, "ask": ask,
                    "volume": int(vol) if not np.isnan(vol) else 0,
                    "open_int": int(oi) if not np.isnan(oi) else 0,
                    "iv": round(iv, 3), "rv": round(realized_vol, 3) if realized_vol else None,
                    "delta": round(delta, 3), "gamma": round(gamma, 4),
                    "theta": round(theta, 4) if not np.isnan(theta) else None,
                    "vega": round(vega, 4) if not np.isnan(vega) else None,
                    "pop": round(pop, 3) if not np.isnan(pop) else None,
                    "direction": round(direction, 2),
                    "leverage": leverage, "spread_pct": round(spread_pct, 3),
                    "vol_oi": round(vol_oi, 2),
                    "_liq": liquidity_raw, "_tight": tightness,
                    "_ivval": iv_value, "_money": moneyness_score,
                    "_unusual": vol_oi, "_align": align,
                })

    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# Scoring: normalize each factor across the candidate pool, then blend
# ----------------------------------------------------------------------
WEIGHTS = {
    "direction": 0.30,   # trend/momentum alignment (calls↔bullish, puts↔bearish)
    "leverage":  0.22,   # bang for buck
    "liquidity": 0.20,   # can you actually get in/out
    "unusual":   0.12,   # smart-money / flow signal
    "iv_value":  0.08,   # not overpaying for vol
    "moneyness": 0.08,   # convexity sweet spot
}


def zscale(s):
    s = s.astype(float)
    if s.std(ddof=0) == 0 or s.isna().all():
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - s.mean()) / s.std(ddof=0)


def minmax(s):
    s = s.astype(float)
    lo, hi = s.min(), s.max()
    if hi == lo:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - lo) / (hi - lo)


def score(df):
    if df.empty:
        return df
    # liquidity floor: drop totally dead contracts
    df = df[(df["open_int"] >= 50) | (df["volume"] >= 25)].copy()
    if df.empty:
        return df

    df["f_leverage"]  = minmax(df["leverage"])
    df["f_liquidity"] = minmax(df["_liq"] * (0.5 + 0.5 * df["_tight"]))
    df["f_unusual"]   = minmax(np.log1p(df["_unusual"]))
    df["f_iv_value"]  = minmax(df["_ivval"])
    df["f_moneyness"] = minmax(df["_money"])
    df["f_direction"] = minmax(df["_align"])

    df["score"] = (
        WEIGHTS["direction"] * df["f_direction"] +
        WEIGHTS["leverage"]  * df["f_leverage"] +
        WEIGHTS["liquidity"] * df["f_liquidity"] +
        WEIGHTS["unusual"]   * df["f_unusual"] +
        WEIGHTS["iv_value"]  * df["f_iv_value"] +
        WEIGHTS["moneyness"] * df["f_moneyness"]
    ) * 100

    return df.sort_values("score", ascending=False)


def load_watchlist(path):
    """Read tickers from a file: one per line, or comma-separated. '#' = comment."""
    syms = []
    with open(path) as f:
        for line in f:
            line = line.split("#")[0].strip()
            if not line:
                continue
            for tok in line.replace(",", " ").split():
                syms.append(tok.upper())
    # de-dupe, preserve order
    seen, out = set(), []
    for s in syms:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def main():
    p = argparse.ArgumentParser(description="Directional options screener (calls + puts).")
    p.add_argument("tickers", nargs="*", help="Ticker symbols, e.g. AAPL MSFT NVDA")
    p.add_argument("--watchlist", metavar="PATH", default=None,
                   help="Read tickers from a file (one per line or comma-separated)")
    p.add_argument("--side", choices=["call", "put", "both"], default="both")
    p.add_argument("--min-dte", type=int, default=7, help="Min days to expiry")
    p.add_argument("--max-dte", type=int, default=45, help="Max days to expiry")
    p.add_argument("--max-bid", type=float, default=None,
                   help="Only include contracts with bid at or below this (e.g. 1.00)")
    p.add_argument("--top", type=int, default=10, help="How many to show")
    p.add_argument("--csv", metavar="PATH", default=None,
                   help="Also write the top results to this CSV file")
    p.add_argument("--summary", metavar="PATH", default=None,
                   help="Write a human-readable summary (markdown) to this file")
    args = p.parse_args()

    # assemble ticker universe from args and/or watchlist file
    tickers = list(args.tickers)
    if args.watchlist:
        try:
            tickers += load_watchlist(args.watchlist)
        except Exception as e:
            print(f"Could not read watchlist {args.watchlist}: {e}")
    # de-dupe preserving order
    seen = set()
    deduped = []
    for x in tickers:
        t = x.upper()
        if t not in seen:
            seen.add(t)
            deduped.append(t)
    tickers = deduped
    if not tickers:
        p.error("No tickers given. Pass them as arguments or use --watchlist FILE.")

    print(f"\nScanning {len(tickers)} ticker(s) | side={args.side} | "
          f"dte {args.min_dte}-{args.max_dte}\n" + "-" * 60)

    frames = []
    for sym in tickers:
        print(f"  fetching {sym} ...")
        try:
            frames.append(scan_ticker(sym, args.side, args.min_dte, args.max_dte, args.max_bid))
        except Exception as e:
            print(f"  [skip] {sym}: unexpected error ({type(e).__name__}: {e})")
            frames.append(pd.DataFrame())

    allopts = pd.concat([f for f in frames if not f.empty], ignore_index=True) \
        if any(not f.empty for f in frames) else pd.DataFrame()

    cols = ["score", "symbol", "type", "expiry", "dte", "strike", "spot",
            "mid", "direction", "delta", "pop", "theta", "vega", "leverage",
            "iv", "rv", "volume", "open_int", "vol_oi", "spread_pct"]

    if allopts.empty:
        print("\nNo qualifying contracts found (data source may be unavailable or "
              "rate-limited). Try widening --max-dte, different tickers, or rerun.")
        if args.csv:
            pd.DataFrame(columns=["scan_utc"] + cols).to_csv(args.csv, index=False)
            print(f"Wrote empty results file to {args.csv}")
        if args.summary:
            with open(args.summary, "w") as f:
                f.write(f"# Options Scan — {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC\n\n"
                        "No qualifying contracts found this run (data source may have been "
                        "unavailable or rate-limited).\n")
        return

    ranked = score(allopts)
    if ranked.empty:
        print("\nContracts found but all failed the liquidity floor.")
        if args.csv:
            pd.DataFrame(columns=["scan_utc"] + cols).to_csv(args.csv, index=False)
            print(f"Wrote empty results file to {args.csv}")
        return

    out = ranked[cols].head(args.top).copy()
    out["score"] = out["score"].round(1)
    out["leverage"] = out["leverage"].round(2)

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 220)
    print("\n" + "=" * 60)
    print(f"TOP {min(args.top, len(out))} CANDIDATES")
    print("=" * 60)
    print(out.to_string(index=False))

    if args.csv:
        out_to_save = out.copy()
        out_to_save.insert(0, "scan_utc", datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"))
        out_to_save.to_csv(args.csv, index=False)
        print(f"\nSaved top {len(out_to_save)} results to {args.csv}")

    if args.summary:
        write_summary(args.summary, out, ranked, tickers)
        print(f"Wrote readable summary to {args.summary}")

    print("\n" + "-" * 60)
    print("Score blends: direction, leverage, liquidity, unusual vol, IV value, moneyness.")
    print("HIGH SCORE = aligned w/ trend + high leverage + tradeable, NOT a win guarantee.")
    print("'direction' -1..+1 = bearish..bullish trend. 'pop' = rough prob. of finishing ITM.")
    print("Options can expire worthless. This is a research aid, not advice.")
    print("-" * 60)


def write_summary(path, out, ranked, tickers):
    """Write a human-readable markdown digest of the top ideas."""
    now = datetime.now(timezone.utc)
    lines = [f"# Options Scan — {now:%Y-%m-%d %H:%M} UTC", ""]
    lines.append(f"Scanned **{len(tickers)}** tickers: {', '.join(tickers)}  ")
    lines.append(f"Found **{len(ranked)}** qualifying contracts. "
                 f"Showing top **{len(out)}**.\n")

    def bias_word(d):
        if d >= 0.4:  return "strongly bullish"
        if d >= 0.1:  return "leaning bullish"
        if d <= -0.4: return "strongly bearish"
        if d <= -0.1: return "leaning bearish"
        return "neutral"

    lines.append("## Top ideas\n")
    for i, (_, row) in enumerate(out.iterrows(), 1):
        rv_txt = f"{row['rv']:.0%}" if pd.notna(row["rv"]) else "n/a"
        pop_txt = f"{row['pop']*100:.0f}%" if pd.notna(row["pop"]) else "n/a"
        lines.append(
            f"**{i}. {row['symbol']} {row['type'].upper()} "
            f"${row['strike']:g} exp {row['expiry']}** (score {row['score']:.1f})  \n"
            f"- Underlying ${row['spot']:g}, trend {bias_word(row['direction'])} "
            f"({row['direction']:+.2f}); this {row['type']} is aligned with it.  \n"
            f"- Premium ~${row['mid']:g}/contract, delta {row['delta']:+.2f}, "
            f"~{pop_txt} chance of finishing ITM.  \n"
            f"- Leverage {row['leverage']:.1f}x, IV {row['iv']:.0%} vs realized "
            f"{rv_txt}, {int(row['volume'])} vol / {int(row['open_int'])} OI.\n"
        )

    # per-direction tally
    n_calls = int((out["type"] == "call").sum())
    n_puts = int((out["type"] == "put").sum())
    lines.append("## Read\n")
    lines.append(f"The top list skews **{n_calls} calls / {n_puts} puts** — "
                 "a quick gauge of where the scanner sees the cleaner directional setups today.\n")
    lines.append("> Score blends trend alignment, leverage, liquidity, unusual volume, "
                 "IV value, and moneyness. A high score is an *idea to research*, not a "
                 "prediction. Options can expire worthless.\n")

    with open(path, "w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
