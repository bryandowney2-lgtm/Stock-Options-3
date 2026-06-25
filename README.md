# Options Scanner

A directional options screener (calls + puts) that pulls live option chains via
[yfinance](https://github.com/ranaroussi/yfinance), scores every contract on a
blend of factors — including a trend/momentum signal — and ranks the best ideas.
Outputs both a CSV and a plain-English summary.

> **This is a research aid, not financial advice.** A high score means a contract
> aligned with the recent trend, with high leverage, that is actually tradeable —
> **not** one likely to profit. Options can expire worthless. Verify all quotes
> against your broker before trading.

## Install

```bash
pip install -r requirements.txt
```

## Usage

```bash
# scan the names in your watchlist file
python options_scanner.py --watchlist watchlist.txt --csv out.csv --summary out.md

# quick ad-hoc scan
python options_scanner.py NVDA AMD --side call
python options_scanner.py SPY --side put --min-dte 7 --max-dte 21 --top 15
```

### Options

| Flag          | Default | Description                                   |
|---------------|---------|-----------------------------------------------|
| `--watchlist` | —       | Read tickers from a file (one per line or CSV)|
| `--side`      | `both`  | `call`, `put`, or `both`                      |
| `--min-dte`   | `7`     | Minimum days to expiry                        |
| `--max-dte`   | `45`    | Maximum days to expiry                        |
| `--max-bid`   | —       | Only include contracts with bid ≤ this (e.g. `1.00`) |
| `--top`       | `10`    | How many candidates to display                |
| `--csv`       | —       | Also write top results to this CSV            |
| `--summary`   | —       | Write a human-readable markdown summary       |

Tickers can be passed as arguments, via `--watchlist`, or both (deduped).

## The watchlist

`watchlist.txt` holds the tickers to scan — one per line or comma-separated,
`#` for comments. Edit it to change what gets scanned; no code changes needed.
The scanner only ranks within the names you give it; it does not roam the whole
market.

## How scoring works

Each contract is scored on six normalized factors:

- **Direction** (30%) — trend/momentum alignment. Calls rewarded when the
  underlying trends up, puts when it trends down. Built from a moving-average
  stack, price position, rate of change, and RSI.
- **Leverage** (22%) — delta-adjusted bang per dollar
- **Liquidity** (20%) — volume, open interest, spread tightness
- **Unusual volume** (12%) — volume relative to open interest
- **IV value** (8%) — implied vol vs realized vol
- **Moneyness** (8%) — convexity sweet spot (~2–8% OTM)

Each candidate also reports fuller greeks (theta, vega) and `pop`, a rough
probability of finishing in-the-money derived from delta.

Edit the `WEIGHTS` dict in `options_scanner.py` to change the blend — raise
`direction` for stronger directional conviction, lower it for more balance.

## Automated runs (GitHub Actions)

`.github/workflows/scan.yml` runs the scanner every weekday morning (UTC),
scans `watchlist.txt`, and uploads a dated CSV + summary as a downloadable
artifact on each run (no commits to the repo). It can also be triggered manually
from the Actions tab. The scheduled run applies `--max-bid 1.00`, so it only
surfaces contracts with a bid at or below $1.00.

Note: yfinance scrapes Yahoo, whose servers sometimes rate-limit GitHub's shared
IPs, so a scheduled run will occasionally return an empty file. The script
handles this gracefully. For rock-solid data, swap the source for a broker API
(e.g. Tradier, Polygon).

## License

MIT
