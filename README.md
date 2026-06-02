# Moomoo Swing Trader - Gap Up Scanner (MOOMOO OpenD)

A fully runnable Python CLI program for finding short-term swing trade setups (target 1 week holds) focused on **gap-up long candidates** using MOOMOO OpenD API (moomoo-api) for market data, options Greeks, capital flow, earnings history, etc.

## Strategy Fundamentals (Longs)

- **MA3 / MA9 Crossover**: Bullish daily trend (fast MA above slow, or recent cross up).
- **Strong VWAP Floor (3min intraday)**: Price holding above VWAP with volume confirmation.
- **High Volume Inflow**: Among top traded today (volume rank in watchlist or high absolute).
- **Historic Resistance / Key Levels**: Daily chart swing highs/lows for context (avoid chasing into heavy resistance; look for room to run or breakout).
- **Dealer Hedging (GEX 0DTE-5DTE)**: Positive net gamma exposure (stabilizing regime), price respecting Put Wall / below Call Wall for support. Computed from live option chain + Greeks + OI via API.
- **Earnings & Valuation**: Forward PE context + backtested 5-trading-day post-earnings price drift from Moomoo financials + yfinance. Favor names with constructive post-earn history and reasonable outlook.

See SpotGamma GEX primer: https://support.spotgamma.com/hc/en-us/articles/15214161607827-GEX-Gamma-Exposure-Explained-What-It-Is-and-How-SpotGamma-Uses-It

**Data Sources**:
- Primary: Moomoo OpenD + moomoo-api (real-time quotes, k-lines, option chains/greeks/OI, capital flow, earnings price history, snapshots with PE).
- Supplement: yfinance (earnings calendar, forward PE, fallback prices, robust intraday for demo).

## Prerequisites

1. **Moomoo Account + OpenD** (REQUIRED for full live data + GEX):
   - Download Visualization OpenD (or CLI) from https://www.moomoo.com/download/OpenAPI
   - Install and **start OpenD** (it must be running locally).
   - Log in with your moomoo ID (same as app). Complete any questionnaire.
   - Default: listens on `127.0.0.1:11111`. Note your port if changed.
   - For US options data and good depth, ensure you have appropriate quote subscription / account level in moomoo.
   - Paper trading / simulation accounts are supported for testing.

2. Python 3.10+ recommended.

3. (Optional but recommended) A list of tickers you care about (high liquidity US names with options chains work best for GEX).

**Important**: Historical k-line has quotas based on your moomoo activity. Option snapshots can be rate-limited. Start with single-ticker `--analyze`.

## Installation

```bash
cd /path/to/moomoo-swing-trader
python3 -m pip install --break-system-packages -r requirements.txt
```

Or:
```bash
python3 -m pip install --break-system-packages moomoo-api yfinance pandas numpy rich python-dateutil
```

## Quick Start (Terminal)

Make sure OpenD is **running and logged in** first.

```bash
# Analyze one ticker in detail (recommended first run)
python3 moomoo_swing.py analyze --ticker TSLA

# Scan default watchlist, show top candidates (gap-up + MA + other filters)
python3 moomoo_swing.py scan --top-n 8

# Scan with more names, require gap today, show only strong scores
python3 moomoo_swing.py scan --top-n 15 --min-score 6.0 --require-gap

# Custom tickers (comma separated) + specific OpenD port
python3 moomoo_swing.py scan --tickers "AAPL,TSLA,NVDA,SMCI,PLTR" --port 11111

# Demo / fallback mode (uses yfinance only, limited GEX, no live capital flow or exact Moomoo earnings history)
python3 moomoo_swing.py analyze --ticker NVDA --no-moomoo
```

## CLI Reference

```
usage: moomoo_swing.py [-h] {analyze,scan,list} ...

Moomoo OpenD Gap-Up Swing Scanner for short-term longs (1 week horizon)

positional arguments:
  {analyze,scan,list}
    analyze            Detailed analysis + scoring for a single ticker
    scan               Scan watchlist or custom tickers, rank by swing long setup score
    list               List default watchlist

options:
  -h, --help           show this help message and exit
  --host HOST          OpenD host (default 127.0.0.1)
  --port PORT          OpenD port (default 11111)
  --no-moomoo          Force yfinance fallback only (demo mode, no GEX full, no capital)
  --debug              Verbose logs
```

Subcommand examples in code or run with --help.

## Key Outputs Explained

For each ticker in scan/analyze:

- **Gap**: % gap at open today vs prev close. High score for >1% gap + volume.
- **MA3/MA9**: Current values, "BULL CROSS" if recent fast> slow cross or sustained.
- **VWAP (3m)**: Current price vs VWAP, "floor strength" (how many recent bars held above, vol surge).
- **Volume**: Today's vol vs prior avg or rank in batch. "Top traded" flag.
- **Key Levels (Daily)**: Recent swing highs (resistance) and lows (support). Printed + distance to current price.
- **GEX (0-5DTE)**: 
  - Net GEX (positive = stabilizing dealer long gamma regime = good for mean-reversion / floors).
  - Call Wall (resistance from dealer short call gamma hedging).
  - Put Wall (support from dealer hedging).
  - Regime note + whether price is in "good zone" for long (e.g. above put wall in +GEX).
- **Earnings/Val**: Last/next earnings (if avail), avg 5-trading-day post-earnings move (backtest), forwardPE if avail, constructive drift score.
- **Overall Score** (0-10+): Weighted (customizable in code): gap + ma + vwap + vol + gex_support + earnings_drift + room_to_resist.

**Recommended candidates**: Printed at end of scan if they pass basic filters (gap or strong trend + positive setup).

## How GEX is Computed Here

Using live data from Moomoo:
1. Get near expirations (dte 0-5).
2. For each, pull option chain (strikes).
3. Snapshot the option symbols (gets live gamma, open_interest, type, strike).
4. GEX_contrib = gamma * OI * 100 * S^2   (signed + for calls, - for puts per common convention).
5. Aggregate net, per-strike, identify walls and (simple) regime.

This approximates dealer hedging pressure for very short dated options (where gamma is highest).

**Not identical to SpotGamma proprietary** (they use more data, models, full chain). Use as one signal among others.

## Configuration & Customization

Edit top of `moomoo_swing.py`:
- `DEFAULT_WATCHLIST`: Add/remove your favorite tickers (must be US.XXX format? No, code accepts 'TSLA' or 'US.TSLA', normalizes to 'US.TSLA').
- `SCORE_WEIGHTS`
- `GAP_THRESHOLD`, `MIN_DTE`, `MAX_DTE`, `VWAP_HOLD_BARS` etc.
- `RESISTANCE_LOOKBACK_DAYS`

For production scans, consider running during market hours, cache results, add alerts (email/telegram), or integrate with trading.

## Limitations & Notes

- **Quotas**: Heavy use of history_kline + option snapshots consumes your moomoo quote/historical quotas. Use sparingly, prefer `analyze` on 5-10 names.
- **Speed**: Full GEX on one liquid name (e.g. TSLA) can take 5-20s due to many option snapshots. Scan batches options only for high-score prelims or on-demand.
- **US Focus**: Strategy + GEX examples target US equities with active options. HK/A-share supported by API but fewer tested.
- **Not financial advice**: This is a research/educational tool. Backtest yourself. Markets change. Data can be delayed or incomplete.
- **Connection**: If "connection refused", confirm OpenD is started, logged in, and firewall allows localhost:11111. Check OpenD logs.
- Fallback mode is useful for learning the signals without OpenD.

## Extending

- Add more plates / use `get_stock_screen` for dynamic high-volume list (see moomoo docs for field IDs).
- Add multi-timeframe (hourly VWAP), order book imbalance, or broker queue.
- Persist scans to CSV/Parquet + backtest the scanner itself.
- Plot with matplotlib (add dep) the GEX profile or daily with MAs + levels.

## Git

This project is hosted at:

**https://github.com/martingongg/moomoo-swing-trader**

To clone:

```bash
git clone https://github.com/martingongg/moomoo-swing-trader.git
cd moomoo-swing-trader
```

The code was uploaded using GitHub CLI after initial local commits.
```

Also update the extending or other? No need.

To make the remote display nice, I can remove and re-add the remote with proper spacing, but since it's internal, perhaps run `git remote set-url` or just leave.

The mangling is only in the tool's output formatting here; in a real terminal `git remote -v` looks normal.

To confirm, I can leave it.

Optionally, update README with more, like mention HTTPS was used.

But good enough.

Now, since gh might have set the remote to SSH or HTTPS.

From previous, gh used HTTPS because of the auth protocol.

Let's check the actual remote now after the create.
## Credits / References

- Moomoo Open API docs: https://openapi.moomoo.com/moomoo-api-doc/en/
- py-moomoo-api: https://github.com/MoomooOpen/py-moomoo-api
- SpotGamma GEX explanation (linked above)
- yfinance for supplemental data

Built to be terminal-first, zero external services besides OpenD + optional yf.

Run it, tune the params to your style, and always verify live in the moomoo app charts + GEX panel.

Happy hunting for those gap-up swings with dealer support!
