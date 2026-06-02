#!/usr/bin/env python3
"""
Moomoo OpenD Gap-Up Swing Scanner
---------------------------------
Terminal program to find short-term (swing within ~1 week) long candidates focused on gap-ups
using MOOMOO OpenD API for real data + yfinance supplement.

Core Long Filters / Signals:
- Daily MA3 > MA9 (or recent bullish crossover)
- Intraday strong VWAP floor on 3min + good volume inflow (top traded today)
- Room vs historic daily resistance/support levels (swing highs)
- Positive/short-term GEX (0-5 DTE) from live option Greeks + OI (dealer hedging support)
- Constructive earnings history (5-trading-day post-earn drift backtest) + forward PE context

Improvements based on analysis of user's Past Trades.rtf (winning realized trades in AAPL, NVDA, MRVL, TXN, ADBE, NOW, HOOD, SOFI, QCOM etc.):
- Expanded DEFAULT_WATCHLIST with names that appeared in profitable swings/options trades.
- Volume ranking and "top traded" now prioritize dollar turnover (not just share volume) -- critical for high-priced tech names that dominated the win list.
- Added explicit high $ turnover bonus in scoring.

Fully self-contained CLI. Requires running moomoo OpenD for full features.

Usage examples:
  python3 moomoo_swing.py analyze --ticker TSLA
  python3 moomoo_swing.py scan --top-n 10 --require-gap
  python3 moomoo_swing.py scan --tickers "NVDA,SMCI,AAPL,PLTR" --min-score 5.5
  python3 moomoo_swing.py analyze --ticker AAPL --no-moomoo   # demo fallback

See README.md for setup (OpenD download/login first!).
"""

import argparse
import sys
import time
import warnings
from datetime import datetime, timedelta, date
from typing import List, Dict, Optional, Tuple, Any

import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import track
from rich import print as rprint
import yfinance as yf

# moomoo may not be present or OpenD not running; handle gracefully
try:
    import moomoo as ft
    MOOMOO_AVAILABLE = True
except ImportError:
    MOOMOO_AVAILABLE = False
    ft = None  # type: ignore

warnings.filterwarnings("ignore", category=FutureWarning)
console = Console()

# ========================== CONFIG / TUNABLES ==========================

DEFAULT_WATCHLIST = [
    "AAPL", "NVDA", "AMD", "META", "MSFT", "GOOGL", "AVGO", "QCOM", "MRVL", "TXN", "ADBE", "NOW",
    "TSLA", "SMCI", "PLTR", "COIN", "MSTR", "ARM", "CRM", "INTC", "MU", "SNOW",
    "HOOD", "SOFI", "UBER", "ABNB", "SHOP", "RBLX", "DKNG", "LCID", "RIVN", "F",
    "BAC", "JPM", "XOM", "CVX", "UNH", "LLY", "JNJ", "PFE", "MRNA", "BABA",
]

# Score weights (sum ~10-12 target). Tune to taste.
SCORE_WEIGHTS = {
    "gap": 2.0,
    "ma": 1.8,
    "vwap": 1.5,
    "volume": 1.2,
    "gex": 2.0,
    "earnings": 1.3,
    "levels": 0.8,  # room to resistance (more room or breakout = better for swing)
}

# Thresholds
GAP_UP_MIN_PCT = 0.008          # 0.8%+ gap considered interesting
MIN_ABS_VOLUME_TODAY = 5_000_000  # for "good volume" filter in scan (share volume)
MIN_ABS_TURNOVER_TODAY = 100_000_000  # $100M+ dollar volume today for "top traded" (better for high-priced names like AAPL/NVDA/ADBE from past winners)
VWAP_HOLD_BARS = 6              # how many recent 3m bars we want price >= VWAP for "floor"
MA_CROSS_WINDOW = 3             # look back N bars for crossover signal
RESISTANCE_LOOKBACK = 120       # trading days for swing high detection
GEX_MAX_DTE = 5
GEX_MIN_DTE = 0
TOP_VOLUME_PCT_FOR_FLAG = 0.25  # if vol rank in top 25% of scanned batch -> "top traded"

# Earnings backtest
POST_EARN_DAYS = 5
MIN_EARN_HISTORY = 3

# Connection
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 11111

# ========================== UTILS ==========================

def normalize_code(ticker: str) -> str:
    """Normalize to moomoo US format e.g. TSLA -> US.TSLA. Accepts US.TSLA or TSLA."""
    t = ticker.strip().upper()
    if t.startswith("US."):
        return t
    if "." in t:
        return t  # let other markets pass through if user knows
    return f"US.{t}"

def denormalize(ticker: str) -> str:
    """For display / yfinance: US.TSLA -> TSLA"""
    return ticker.split(".")[-1].upper()

def safe_get(d: Dict, key: str, default: Any = None) -> Any:
    return d.get(key, default) if isinstance(d, dict) else default

def pct(x: float) -> str:
    if pd.isna(x) or x is None:
        return "N/A"
    return f"{x*100:+.2f}%"

def fmt(x: Any, nd: int = 2) -> str:
    if pd.isna(x) or x is None:
        return "N/A"
    if isinstance(x, (int, float)):
        return f"{x:,.{nd}f}" if abs(x) >= 1000 else f"{x:.{nd}f}"
    return str(x)

# ========================== DATA FETCH: MOOMOO (primary) + YF FALLBACK ==========================

class MoomooClient:
    """Thin wrapper around OpenQuoteContext. Lazy connect, context friendly."""
    def __init__(self, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT):
        self.host = host
        self.port = port
        self.ctx = None
        self.connected = False

    def connect(self) -> bool:
        if not MOOMOO_AVAILABLE:
            return False
        try:
            self.ctx = ft.OpenQuoteContext(host=self.host, port=self.port)
            # quick sanity
            ret, data = self.ctx.get_market_snapshot(["US.AAPL"])
            if ret == ft.RET_OK:
                self.connected = True
                return True
            console.print(f"[yellow]OpenD snapshot test failed: {data}[/yellow]")
            self.ctx.close()
            self.ctx = None
            return False
        except Exception as e:
            console.print(f"[red]Failed to connect to OpenD at {self.host}:{self.port}: {e}[/red]")
            console.print("[yellow]Make sure OpenD is running and you are logged in.[/yellow]")
            self.ctx = None
            self.connected = False
            return False

    def close(self):
        if self.ctx:
            try:
                self.ctx.close()
            except Exception:
                pass
        self.connected = False
        self.ctx = None

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *a):
        self.close()

    # --- Core fetches ---

    def get_snapshot(self, codes: List[str]) -> pd.DataFrame:
        if not self.connected or self.ctx is None:
            return pd.DataFrame()
        codes = [normalize_code(c) for c in codes]
        try:
            ret, data = self.ctx.get_market_snapshot(codes)
            if ret == ft.RET_OK and isinstance(data, pd.DataFrame) and not data.empty:
                return data
        except Exception as e:
            if "debug" in sys.argv:
                console.print(f"[dim]snapshot err: {e}[/dim]")
        return pd.DataFrame()

    def get_daily_kline(self, code: str, days: int = 200) -> pd.DataFrame:
        if not self.connected or self.ctx is None:
            return pd.DataFrame()
        code = normalize_code(code)
        try:
            end = datetime.now().strftime("%Y-%m-%d")
            start = (datetime.now() - timedelta(days=days + 30)).strftime("%Y-%m-%d")
            ret, data, _ = self.ctx.request_history_kline(
                code, start=start, end=end, ktype=ft.KLType.K_DAY,
                fields=[ft.KL_FIELD.ALL], max_count=1000
            )
            if ret == ft.RET_OK and isinstance(data, pd.DataFrame) and not data.empty:
                data = data.sort_values("time_key").reset_index(drop=True)
                data["time_key"] = pd.to_datetime(data["time_key"])
                return data.tail(days)
        except Exception as e:
            if "debug" in sys.argv:
                console.print(f"[dim]daily k err {code}: {e}[/dim]")
        return pd.DataFrame()

    def get_3m_kline_recent(self, code: str, num: int = 80) -> pd.DataFrame:
        """Recent 3min bars for VWAP. Uses get_cur_kline (intraday focused)."""
        if not self.connected or self.ctx is None:
            return pd.DataFrame()
        code = normalize_code(code)
        try:
            ret, data = self.ctx.get_cur_kline(code, num=num, ktype=ft.KLType.K_3M)
            if ret == ft.RET_OK and isinstance(data, pd.DataFrame) and not data.empty:
                data = data.sort_values("time_key").reset_index(drop=True)
                data["time_key"] = pd.to_datetime(data["time_key"])
                return data
        except Exception as e:
            if "debug" in sys.argv:
                console.print(f"[dim]3m k err {code}: {e}[/dim]")
        return pd.DataFrame()

    def get_capital_flow_intraday(self, code: str) -> pd.DataFrame:
        if not self.connected or self.ctx is None:
            return pd.DataFrame()
        code = normalize_code(code)
        try:
            ret, data = self.ctx.get_capital_flow(code, period_type=ft.PeriodType.INTRADAY)
            if ret == ft.RET_OK and isinstance(data, pd.DataFrame):
                return data
        except Exception:
            pass
        return pd.DataFrame()

    def get_near_expiries(self, code: str, max_dte: int = GEX_MAX_DTE) -> List[str]:
        """Return list of expiration date strings (YYYY-MM-DD) for 0..max_dte DTE."""
        if not self.connected or self.ctx is None:
            return []
        code = normalize_code(code)
        try:
            ret, data = self.ctx.get_option_expiration_date(code)
            if ret == ft.RET_OK and isinstance(data, pd.DataFrame) and not data.empty:
                df = data.copy()
                df = df[df["option_expiry_date_distance"] >= GEX_MIN_DTE]
                df = df[df["option_expiry_date_distance"] <= max_dte]
                # strike_time is the expiry date str
                exps = df["strike_time"].dropna().astype(str).unique().tolist()
                # sort soonest first
                exps = sorted(exps)[:8]  # cap
                return exps
        except Exception as e:
            if "debug" in sys.argv:
                console.print(f"[dim]expiries err {code}: {e}[/dim]")
        return []

    def get_option_chain_for_expiry(self, code: str, exp_date: str) -> pd.DataFrame:
        if not self.connected or self.ctx is None:
            return pd.DataFrame()
        code = normalize_code(code)
        try:
            ret, data = self.ctx.get_option_chain(
                code, start=exp_date, end=exp_date,
                option_type=ft.OptionType.ALL, option_cond_type=ft.OptionCondType.ALL
            )
            if ret == ft.RET_OK and isinstance(data, pd.DataFrame) and not data.empty:
                return data
        except Exception:
            pass
        return pd.DataFrame()

    def get_option_snapshots(self, option_codes: List[str], batch_size: int = 180) -> pd.DataFrame:
        """Batch snapshot for option greeks + OI. Returns combined df."""
        if not self.connected or self.ctx is None or not option_codes:
            return pd.DataFrame()
        all_dfs = []
        for i in range(0, len(option_codes), batch_size):
            batch = option_codes[i:i+batch_size]
            try:
                ret, df = self.ctx.get_market_snapshot(batch)
                if ret == ft.RET_OK and isinstance(df, pd.DataFrame) and not df.empty:
                    all_dfs.append(df)
            except Exception:
                continue
            time.sleep(0.05)  # be nice
        if all_dfs:
            return pd.concat(all_dfs, ignore_index=True)
        return pd.DataFrame()

    def get_earnings_price_history(self, code: str) -> pd.DataFrame:
        """Moomoo built-in earnings +/- price history (great for post-earn backtest)."""
        if not self.connected or self.ctx is None:
            return pd.DataFrame()
        code = normalize_code(code)
        try:
            ret, data = self.ctx.get_financials_earnings_price_history(code)
            if ret == ft.RET_OK and isinstance(data, pd.DataFrame):
                return data
        except Exception:
            pass
        return pd.DataFrame()

    def get_earnings_price_move(self, code: str) -> pd.DataFrame:
        try:
            if not self.connected or self.ctx is None:
                return pd.DataFrame()
            ret, data = self.ctx.get_financials_earnings_price_move(normalize_code(code))
            if ret == ft.RET_OK and isinstance(data, pd.DataFrame):
                return data
        except Exception:
            pass
        return pd.DataFrame()


# YFinance fallbacks (always available)

def yf_daily_history(ticker: str, period: str = "1y") -> pd.DataFrame:
    try:
        t = yf.Ticker(denormalize(ticker))
        df = t.history(period=period, auto_adjust=True)
        if df.empty:
            return pd.DataFrame()
        df = df.reset_index()
        df = df.rename(columns={"Date": "time_key", "Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"})
        df["time_key"] = pd.to_datetime(df["time_key"])
        df["code"] = ticker
        return df[["code", "time_key", "open", "high", "low", "close", "volume"]].copy()
    except Exception:
        return pd.DataFrame()

def yf_intraday_approx(ticker: str, interval: str = "5m", days: int = 2) -> pd.DataFrame:
    """Fallback intraday. yf 1m/5m limited to last 60d but recent only usually."""
    try:
        t = yf.Ticker(denormalize(ticker))
        df = t.history(period=f"{days}d", interval=interval, auto_adjust=True)
        if df.empty:
            return pd.DataFrame()
        df = df.reset_index().rename(columns={"Datetime": "time_key", "Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"})
        df["time_key"] = pd.to_datetime(df["time_key"])
        return df
    except Exception:
        return pd.DataFrame()

def yf_info(ticker: str) -> Dict:
    try:
        return yf.Ticker(denormalize(ticker)).info or {}
    except Exception:
        return {}

def yf_earnings_dates(ticker: str) -> pd.DataFrame:
    try:
        ed = yf.Ticker(denormalize(ticker)).earnings_dates
        if ed is not None and not ed.empty:
            return ed.reset_index()
    except Exception:
        pass
    return pd.DataFrame()

def yf_option_chain(ticker: str) -> Dict[str, pd.DataFrame]:
    """For demo GEX fallback. Limited greeks (no full live gamma sometimes)."""
    try:
        t = yf.Ticker(denormalize(ticker))
        exps = t.options
        chains = {}
        for e in exps[:4]:  # few nearest
            try:
                oc = t.option_chain(e)
                chains[e] = {"calls": oc.calls, "puts": oc.puts}
            except Exception:
                continue
        return chains
    except Exception:
        return {}


# ========================== ANALYSIS LOGIC ==========================

def compute_ma_signals(daily_df: pd.DataFrame) -> Dict[str, Any]:
    """MA3 / MA9 on close. Returns current state + crossover flag."""
    if daily_df is None or daily_df.empty or "close" not in daily_df:
        return {"ma3": None, "ma9": None, "bullish": False, "cross_up": False, "detail": "no data"}
    df = daily_df.sort_values("time_key").copy()
    df["ma3"] = df["close"].rolling(3).mean()
    df["ma9"] = df["close"].rolling(9).mean()
    if len(df) < 9:
        return {"ma3": None, "ma9": None, "bullish": False, "cross_up": False, "detail": "insufficient history"}

    last = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else last

    ma3 = float(last["ma3"])
    ma9 = float(last["ma9"])
    bullish = ma3 > ma9
    # simple crossover detection in last few bars
    cross_up = False
    for i in range(1, min(MA_CROSS_WINDOW + 1, len(df))):
        a = df.iloc[-i-1]
        b = df.iloc[-i]
        if (a["ma3"] <= a["ma9"]) and (b["ma3"] > b["ma9"]):
            cross_up = True
            break
    detail = f"MA3={ma3:.2f} > MA9={ma9:.2f}" if bullish else f"MA3={ma3:.2f} < MA9={ma9:.2f}"
    if cross_up:
        detail += " | RECENT BULL CROSS"
    return {"ma3": ma3, "ma9": ma9, "bullish": bullish, "cross_up": cross_up, "detail": detail}


def compute_vwap_floor(intraday_df: pd.DataFrame, current_price: Optional[float] = None) -> Dict[str, Any]:
    """Compute VWAP on 3m (or resampled) and measure floor strength."""
    if intraday_df is None or intraday_df.empty:
        return {"vwap": None, "above": False, "hold_bars": 0, "strength": 0.0, "detail": "no intraday data"}
    df = intraday_df.sort_values("time_key").copy().tail(60)
    if "close" not in df or "volume" not in df or len(df) < 5:
        return {"vwap": None, "above": False, "hold_bars": 0, "strength": 0.0, "detail": "insufficient bars"}

    df["typical"] = (df["high"] + df["low"] + df["close"]) / 3.0
    df["cum_vol"] = df["volume"].cumsum()
    df["cum_tpvol"] = (df["typical"] * df["volume"]).cumsum()
    df["vwap"] = df["cum_tpvol"] / df["cum_vol"].replace(0, np.nan)

    vwap = float(df["vwap"].iloc[-1])
    if current_price is None:
        current_price = float(df["close"].iloc[-1])

    above = current_price >= vwap
    # count consecutive recent bars where close >= vwap (floor strength)
    hold = 0
    for i in range(len(df)-1, -1, -1):
        if df["close"].iloc[i] >= df["vwap"].iloc[i]:
            hold += 1
        else:
            break
    hold = min(hold, len(df))
    strength = min(1.0, hold / max(1, VWAP_HOLD_BARS))
    detail = f"Price {current_price:.2f} vs VWAP {vwap:.2f} | hold {hold} bars"
    if above and hold >= VWAP_HOLD_BARS:
        detail += " (STRONG FLOOR)"
    return {"vwap": vwap, "above": above, "hold_bars": hold, "strength": strength, "detail": detail}


def compute_volume_inflow(moomoo_client: Optional[MoomooClient], code: str, snapshot_row: Optional[pd.Series], daily_df: pd.DataFrame) -> Dict[str, Any]:
    """Volume today + inflow signal. Uses share volume + turnover ($ volume) for high-priced names.
    Based on analysis of user's winning trades (e.g. AAPL, NVDA, ADBE, MRVL, TXN etc which often have high $ volume).
    Rank is provided at batch level."""
    vol_today = None
    turnover_today = None
    vol_ratio = None
    inflow = 0.0
    detail = ""

    if snapshot_row is not None and not snapshot_row.empty:
        vol_today = float(snapshot_row.get("volume", 0) or 0)
        turnover_today = float(snapshot_row.get("turnover", 0) or 0)
        # volume_ratio if present (today vs prior period)
        vol_ratio = float(snapshot_row.get("volume_ratio", 1.0) or 1.0)

    if daily_df is not None and not daily_df.empty and "volume" in daily_df and len(daily_df) > 5:
        avg_vol = daily_df["volume"].tail(20).mean()
        if vol_today and avg_vol:
            vol_ratio = vol_ratio or (vol_today / max(1, avg_vol))

    # Capital flow if available (net inflow positive good)
    cap_df = pd.DataFrame()
    if moomoo_client and moomoo_client.connected:
        cap_df = moomoo_client.get_capital_flow_intraday(code)
    if not cap_df.empty and "in_flow" in cap_df:
        try:
            recent = cap_df.tail(5)["in_flow"].astype(float).sum()
            inflow = float(recent)
        except Exception:
            pass

    detail = f"Vol today ~{fmt(vol_today)} | $turnover ~{fmt(turnover_today)} | ratio {fmt(vol_ratio, 1)} | inflow {fmt(inflow)}"
    return {
        "volume_today": vol_today,
        "turnover_today": turnover_today,
        "volume_ratio": vol_ratio or 1.0,
        "inflow": inflow,
        "detail": detail
    }


def find_historic_levels(daily_df: pd.DataFrame, lookback: int = RESISTANCE_LOOKBACK) -> Dict[str, Any]:
    """Simple swing high/low detection for resistance/support context."""
    if daily_df is None or daily_df.empty or len(daily_df) < 10:
        return {"resistances": [], "supports": [], "detail": "no history"}

    df = daily_df.sort_values("time_key").tail(lookback).copy().reset_index(drop=True)
    highs = df["high"].values
    lows = df["low"].values
    n = len(df)

    resistances = []
    supports = []

    # local max: high[i] > high[i-1] and high[i] > high[i+1] (with some buffer for noise)
    for i in range(1, n-1):
        if highs[i] > highs[i-1] * 1.0005 and highs[i] > highs[i+1] * 1.0005:
            resistances.append((df.loc[i, "time_key"], float(highs[i])))
        if lows[i] < lows[i-1] * 0.9995 and lows[i] < lows[i+1] * 0.9995:
            supports.append((df.loc[i, "time_key"], float(lows[i])))

    # also add recent high/low and 20d high
    resistances.append((df["time_key"].iloc[-1], float(df["high"].iloc[-1])))
    supports.append((df["time_key"].iloc[-1], float(df["low"].iloc[-1])))
    if len(df) >= 20:
        resistances.append((df["time_key"].iloc[-20], float(df["high"].tail(20).max())))
        supports.append((df["time_key"].iloc[-20], float(df["low"].tail(20).min())))

    # dedup + sort desc price for resistances
    def dedup_sort(levels, reverse=True):
        seen = set()
        out = []
        for ts, p in levels:
            key = round(p, 2)
            if key not in seen:
                seen.add(key)
                out.append((ts, p))
        out.sort(key=lambda x: x[1], reverse=reverse)
        return out[:6]

    resistances = dedup_sort(resistances)
    supports = dedup_sort(supports, reverse=False)

    current = float(df["close"].iloc[-1])
    nearest_res = min([p for _, p in resistances] + [current * 1.2], key=lambda p: abs(p - current))
    room = (nearest_res - current) / current if nearest_res > current else 0.0

    detail = f"Nearest res ~{fmt(nearest_res)} ({pct(room)} away) | {len(resistances)} swing highs tracked"
    return {
        "resistances": resistances,
        "supports": supports,
        "nearest_resistance": nearest_res,
        "room_to_res": room,
        "current": current,
        "detail": detail
    }


def compute_gex(moomoo_client: Optional[MoomooClient], code: str, underlying_price: float) -> Dict[str, Any]:
    """
    Compute basic GEX for 0-5 DTE using live option gamma + OI from snapshots.
    Formula approx: gex = gamma * OI * 100 * S**2   (signed +calls / -puts)
    Returns net, walls, simple regime.
    """
    if not moomoo_client or not moomoo_client.connected or underlying_price <= 0:
        # Demo fallback using yfinance (very rough: yf gives lastPrice, OI, but greeks limited)
        return _compute_gex_demo_fallback(code, underlying_price)

    exps = moomoo_client.get_near_expiries(code, GEX_MAX_DTE)
    if not exps:
        return {"net_gex": 0.0, "call_wall": None, "put_wall": None, "regime": "unknown", "detail": "no near expiries via API", "dte_count": 0}

    all_rows = []
    for exp in exps:
        chain = moomoo_client.get_option_chain_for_expiry(code, exp)
        if chain.empty:
            continue
        # chain has 'code' (option code), 'strike_price', 'option_type' etc.
        opt_codes = chain["code"].dropna().astype(str).tolist()
        snaps = moomoo_client.get_option_snapshots(opt_codes)
        if snaps.empty:
            continue
        # merge key fields
        try:
            merged = pd.merge(
                chain[["code", "strike_price", "option_type", "stock_owner"]],
                snaps[["code", "option_gamma", "option_open_interest", "option_strike_price", "last_price"]],
                on="code", how="inner"
            )
            merged["expiry"] = exp
            all_rows.append(merged)
        except Exception:
            continue

    if not all_rows:
        return {"net_gex": 0.0, "call_wall": None, "put_wall": None, "regime": "unknown", "detail": "no greeks/OI snapshots", "dte_count": len(exps)}

    gex_df = pd.concat(all_rows, ignore_index=True)
    # clean
    for col in ["option_gamma", "option_open_interest"]:
        gex_df[col] = pd.to_numeric(gex_df[col], errors="coerce").fillna(0.0)
    gex_df = gex_df[gex_df["option_open_interest"] > 0]
    if gex_df.empty:
        return {"net_gex": 0.0, "call_wall": None, "put_wall": None, "regime": "unknown", "detail": "zero OI in near term", "dte_count": len(exps)}

    S = float(underlying_price)
    mult = 100.0
    # option_type usually "CALL" / "PUT" or enum str
    def is_call(row):
        ot = str(row.get("option_type", "")).upper()
        return "CALL" in ot or ot == "C"

    gex_df["is_call"] = gex_df.apply(is_call, axis=1)
    gex_df["strike"] = pd.to_numeric(gex_df.get("strike_price", gex_df.get("option_strike_price", 0)), errors="coerce").fillna(0)
    gex_df["gex_raw"] = gex_df["option_gamma"] * gex_df["option_open_interest"] * mult * (S ** 2)
    gex_df["gex"] = np.where(gex_df["is_call"], gex_df["gex_raw"], -gex_df["gex_raw"])

    net_gex = float(gex_df["gex"].sum())

    # Per strike aggregate (for walls)
    per_strike = gex_df.groupby("strike")["gex"].sum().reset_index()
    per_strike = per_strike.sort_values("strike")

    call_wall = None
    put_wall = None
    if not per_strike.empty:
        # highest positive gex strike ~ call wall (dealer sell pressure on rallies)
        pos = per_strike[per_strike["gex"] > 0]
        if not pos.empty:
            call_wall = float(pos.loc[pos["gex"].idxmax(), "strike"])
        # most negative (largest magnitude negative) ~ put wall
        neg = per_strike[per_strike["gex"] < 0]
        if not neg.empty:
            put_wall = float(neg.loc[neg["gex"].idxmin(), "strike"])

    regime = "stabilizing (+GEX, long gamma dealers)" if net_gex > 0 else "amplifying (-GEX, short gamma dealers)"
    if net_gex > 0 and put_wall and S > put_wall * 0.98:
        regime += " | price near/above put wall -> support zone"
    elif net_gex < 0:
        regime += " | expect bigger moves / respect walls carefully"

    detail = f"netGEX={net_gex:,.0f} | callWall={fmt(call_wall)} putWall={fmt(put_wall)} | {len(exps)} expiries"
    return {
        "net_gex": net_gex,
        "call_wall": call_wall,
        "put_wall": put_wall,
        "regime": regime,
        "detail": detail,
        "dte_count": len(exps),
        "per_strike_sample": per_strike.head(3).to_dict("records") if not per_strike.empty else []
    }


def _compute_gex_demo_fallback(code: str, underlying_price: float) -> Dict[str, Any]:
    """Very rough GEX using yf chain (gamma often missing or 0; use as demo only)."""
    chains = yf_option_chain(code)
    if not chains or underlying_price <= 0:
        return {"net_gex": 0.0, "call_wall": None, "put_wall": None, "regime": "demo-no-data", "detail": "yf demo fallback insufficient greeks", "dte_count": 0}

    rows = []
    S = underlying_price
    mult = 100.0
    for exp, ch in chains.items():
        for side, is_call in [("calls", True), ("puts", False)]:
            df = ch.get(side)
            if df is None or df.empty:
                continue
            for _, r in df.iterrows():
                gamma = float(r.get("gamma", 0.0) or 0.0)
                oi = float(r.get("openInterest", 0) or 0)
                strike = float(r.get("strike", 0) or 0)
                if oi <= 0 or gamma == 0:
                    continue
                g = gamma * oi * mult * (S ** 2) * (1 if is_call else -1)
                rows.append({"strike": strike, "gex": g, "is_call": is_call})

    if not rows:
        return {"net_gex": 0.0, "call_wall": None, "put_wall": None, "regime": "demo-no-oi-gamma", "detail": "yf demo: no usable gamma/OI", "dte_count": len(chains)}

    gex_df = pd.DataFrame(rows)
    net = float(gex_df["gex"].sum())
    per = gex_df.groupby("strike")["gex"].sum().reset_index().sort_values("strike")
    cw = float(per.loc[per["gex"].idxmax(), "strike"]) if (per["gex"] > 0).any() else None
    pw = float(per.loc[per["gex"].idxmin(), "strike"]) if (per["gex"] < 0).any() else None
    regime = "demo-stabilizing" if net > 0 else "demo-amplifying"
    return {
        "net_gex": net,
        "call_wall": cw,
        "put_wall": pw,
        "regime": regime + " (YF DEMO - limited accuracy)",
        "detail": f"demo netGEX ~{net:,.0f}",
        "dte_count": len(chains),
        "per_strike_sample": []
    }


def compute_earnings_backtest(moomoo_client: Optional[MoomooClient], code: str, yf_fallback: bool = True) -> Dict[str, Any]:
    """
    Use Moomoo earnings price history if available (best), else yf earnings_dates + price history.
    Returns avg 5-trading-day post earnings move, recent example, constructive flag.
    """
    moves = []
    last_move = None
    last_date = None

    # Prefer moomoo
    if moomoo_client and moomoo_client.connected:
        hist = moomoo_client.get_earnings_price_history(code)
        if not hist.empty:
            # Expect columns like earnings_time or date, price_before/after, move_5d etc. Inspect lightly.
            # Common fields from doc: various price around earnings day.
            # Try to extract 5d post if present, else compute from price series if attached.
            for col in hist.columns:
                if "move" in col.lower() or "5d" in col.lower() or "post" in col.lower():
                    try:
                        m = pd.to_numeric(hist[col], errors="coerce").dropna()
                        if len(m) >= MIN_EARN_HISTORY:
                            moves = m.tail(6).tolist()
                    except Exception:
                        pass
            # Fallback: if has pre/post close columns
            if not moves and {"close_before", "close_5d_after"} <= set(hist.columns):
                try:
                    hist["move"] = (hist["close_5d_after"] - hist["close_before"]) / hist["close_before"]
                    moves = hist["move"].dropna().tail(6).tolist()
                except Exception:
                    pass

        move_df = moomoo_client.get_earnings_price_move(code)
        if not move_df.empty and len(moves) < 2:
            # heuristic
            for c in move_df.columns:
                if "move" in c.lower() or "pct" in c.lower():
                    m = pd.to_numeric(move_df[c], errors="coerce").dropna()
                    if len(m) > 0:
                        moves = m.tail(6).tolist()

    # yfinance fallback / supplement
    if (not moves or len(moves) < MIN_EARN_HISTORY) and yf_fallback:
        ed = yf_earnings_dates(code)
        if not ed.empty:
            # earnings_dates has 'Earnings Date', 'EPS Estimate', 'Reported EPS', 'Surprise(%)'
            # Get dates, fetch price history around them
            price_hist = yf_daily_history(code, period="2y")
            if not price_hist.empty:
                price_hist = price_hist.set_index("time_key").sort_index()
                dates = pd.to_datetime(ed["Earnings Date"]).dropna().unique()
                for edate in sorted(dates)[-6:]:  # recent
                    try:
                        # find trading day on/after report
                        start = edate.normalize()
                        future = price_hist.loc[start:].head(POST_EARN_DAYS + 2)
                        if len(future) >= POST_EARN_DAYS:
                            pre = price_hist.loc[:start].iloc[-1]["close"] if len(price_hist.loc[:start]) > 0 else np.nan
                            post = future.iloc[POST_EARN_DAYS - 1]["close"]
                            if pd.notna(pre) and pre > 0:
                                moves.append((post - pre) / pre)
                                last_move = moves[-1]
                                last_date = start
                    except Exception:
                        continue

    if not moves:
        return {"avg_5d_post": None, "recent_5d": None, "constructive": False, "count": 0, "detail": "insufficient earnings history"}

    moves = [m for m in moves if pd.notna(m)]
    avg = float(np.mean(moves)) if moves else 0.0
    recent = moves[-1] if moves else None
    constructive = avg > 0.005 or (recent is not None and recent > 0.01)  # mild positive drift bias
    detail = f"Avg 5d post-earn: {pct(avg)} over {len(moves)} events | last: {pct(recent)}"
    if constructive:
        detail += " (CONSTRUCTIVE)"
    return {
        "avg_5d_post": avg,
        "recent_5d": recent,
        "constructive": constructive,
        "count": len(moves),
        "last_date": str(last_date)[:10] if last_date else None,
        "detail": detail
    }


def get_fundamentals_snapshot(snapshot_row: Optional[pd.Series], yf_info_dict: Dict) -> Dict[str, Any]:
    """PE (trailing + forward if avail), market cap etc."""
    pe = None
    pe_ttm = None
    forward_pe = None
    if snapshot_row is not None and not snapshot_row.empty:
        pe = float(snapshot_row.get("pe_ratio", np.nan)) if "pe_ratio" in snapshot_row else None
        pe_ttm = float(snapshot_row.get("pe_ttm_ratio", np.nan)) if "pe_ttm_ratio" in snapshot_row else pe
    forward_pe = yf_info_dict.get("forwardPE") or yf_info_dict.get("forward_pe")
    try:
        forward_pe = float(forward_pe) if forward_pe else None
    except Exception:
        forward_pe = None
    return {
        "pe": pe,
        "pe_ttm": pe_ttm,
        "forward_pe": forward_pe,
        "detail": f"PE_TTM={fmt(pe_ttm)} fwdPE={fmt(forward_pe)}"
    }


def gap_today(snapshot_row: Optional[pd.Series]) -> Dict[str, Any]:
    if snapshot_row is None or snapshot_row.empty:
        return {"gap_pct": 0.0, "is_gap_up": False, "detail": "no snapshot"}
    prev = float(snapshot_row.get("prev_close_price", 0) or 0)
    op = float(snapshot_row.get("open_price", 0) or 0)
    last = float(snapshot_row.get("last_price", 0) or 0)
    if prev <= 0:
        return {"gap_pct": 0.0, "is_gap_up": False, "detail": "no prev close"}
    gap = (op - prev) / prev
    return {
        "gap_pct": gap,
        "is_gap_up": gap >= GAP_UP_MIN_PCT,
        "detail": f"Gap {pct(gap)} (open {op:.2f} vs prev {prev:.2f}) | last {last:.2f}"
    }


# ========================== SCORING & REPORT ==========================

def score_setup(
    gap_info: Dict,
    ma_info: Dict,
    vwap_info: Dict,
    vol_info: Dict,
    gex_info: Dict,
    earn_info: Dict,
    levels_info: Dict,
    is_top_volume: bool = False,
) -> Tuple[float, List[str], List[str]]:
    """Return (score 0-12ish, reasons, warnings)."""
    score = 0.0
    reasons = []
    warnings = []

    # Gap
    g = gap_info.get("gap_pct", 0) or 0
    gap_pts = min(SCORE_WEIGHTS["gap"], SCORE_WEIGHTS["gap"] * (g / 0.02))  # scale to ~2% gap full
    if gap_info.get("is_gap_up"):
        score += gap_pts
        reasons.append(f"Gap up {pct(g)}")
    elif g > 0:
        score += gap_pts * 0.4
        reasons.append(f"Small gap {pct(g)}")
    else:
        warnings.append("No gap or gapped down")

    # MA
    if ma_info.get("bullish"):
        score += SCORE_WEIGHTS["ma"] * (1.1 if ma_info.get("cross_up") else 0.9)
        reasons.append("MA3>MA9 bullish" + (" (cross)" if ma_info.get("cross_up") else ""))
    else:
        warnings.append("MA3 below MA9")

    # VWAP floor
    vstr = vwap_info.get("strength", 0) or 0
    score += SCORE_WEIGHTS["vwap"] * vstr
    if vstr >= 0.8:
        reasons.append("Strong VWAP floor (3m)")
    elif vstr >= 0.4:
        reasons.append("Moderate VWAP support")
    else:
        warnings.append("Weak VWAP floor")

    # Volume (share + $ turnover). High $ volume rewarded extra as it matches user's past winning trades
    # in names like AAPL, NVDA, ADBE, MRVL, TXN, HOOD which often rank high on dollar volume even if share count varies.
    vr = vol_info.get("volume_ratio", 1.0) or 1.0
    turnover_today = vol_info.get("turnover_today", 0) or 0
    vol_pts = min(SCORE_WEIGHTS["volume"], SCORE_WEIGHTS["volume"] * max(0, (vr - 0.8) / 1.5))
    score += vol_pts
    if is_top_volume:
        score += 0.4
        reasons.append("Top traded volume today")
    if turnover_today >= MIN_ABS_TURNOVER_TODAY:
        score += 0.3
        reasons.append(f"High $ turnover ${fmt(turnover_today / 1_000_000, 0)}M")
    if vr >= 1.3:
        reasons.append(f"Volume surge x{fmt(vr,1)}")
    elif vr < 0.7:
        warnings.append("Below avg volume")

    # GEX
    netg = gex_info.get("net_gex", 0) or 0
    gex_pts = 0.0
    if netg > 0:
        gex_pts = SCORE_WEIGHTS["gex"] * 0.9
        reasons.append("Positive near-term GEX (stabilizing)")
        if gex_info.get("put_wall"):
            reasons.append("Put wall support in play")
    else:
        gex_pts = -0.4  # slight penalty for amplifying regime on long swing
        warnings.append("Negative GEX (amplifying moves - higher risk)")
    score += gex_pts

    # Earnings drift
    if earn_info.get("constructive"):
        score += SCORE_WEIGHTS["earnings"]
        reasons.append("Constructive post-earn drift history")
    else:
        if earn_info.get("avg_5d_post", 0) and earn_info["avg_5d_post"] < -0.02:
            warnings.append("Negative historical post-earn drift")
        score += SCORE_WEIGHTS["earnings"] * 0.3

    # Levels / room
    room = levels_info.get("room_to_res", 0) or 0
    if room > 0.03:
        score += SCORE_WEIGHTS["levels"]
        reasons.append(f"Room to resistance {pct(room)}")
    elif room < 0.005:
        warnings.append("Near resistance - limited immediate upside?")

    # clamp
    score = max(0.0, min(12.0, score))
    return round(score, 2), reasons, warnings


def make_candidate_table(candidates: List[Dict]) -> Table:
    table = Table(title="Top Swing Candidates (Gap-Up Long Setups)", show_lines=True)
    table.add_column("Ticker", style="cyan", no_wrap=True)
    table.add_column("Score", style="bold magenta")
    table.add_column("Gap", style="green")
    table.add_column("MA", style="blue")
    table.add_column("VWAP", style="yellow")
    table.add_column("GEX Net", style="magenta")
    table.add_column("PostEarn 5d", style="white")
    table.add_column("Vol", style="dim")
    table.add_column("Why", style="dim")
    for c in candidates:
        table.add_row(
            c["ticker"],
            str(c["score"]),
            pct(c["gap"]),
            "BULL" if c["ma_bull"] else "bear",
            "FLOOR" if c["vwap_strong"] else "weak",
            fmt(c["gex_net"]),
            pct(c["earn_avg"]),
            "TOP" if c["top_vol"] else fmt(c.get("vol_ratio", 1), 1),
            "; ".join(c["reasons"][:2]) if c.get("reasons") else ""
        )
    return table


# ========================== MAIN ACTIONS ==========================

def analyze_ticker(ticker: str, client: Optional[MoomooClient], use_moomoo: bool = True) -> Dict[str, Any]:
    norm = normalize_code(ticker)
    disp = denormalize(norm)
    console.rule(f"[bold]Analyzing {disp} ({norm})[/bold]")

    snap_df = pd.DataFrame()
    daily = pd.DataFrame()
    k3m = pd.DataFrame()
    earn_m = pd.DataFrame()

    if client and use_moomoo and client.connected:
        with console.status("[cyan]Fetching snapshot + daily + 3m + earnings...[/cyan]"):
            snap_df = client.get_snapshot([norm])
            daily = client.get_daily_kline(norm, days=180)
            k3m = client.get_3m_kline_recent(norm, num=60)
            earn_m = client.get_earnings_price_history(norm)

    # Fallbacks
    if daily.empty:
        daily = yf_daily_history(norm, period="1y")
    if k3m.empty:
        k3m = yf_intraday_approx(norm, interval="5m", days=1)  # approx, resample mentally
        # for vwap we can still use it

    snap_row = snap_df.iloc[0] if not snap_df.empty else None
    current_price = float(snap_row["last_price"]) if snap_row is not None and "last_price" in snap_row else (float(daily["close"].iloc[-1]) if not daily.empty else 0.0)

    # Core signals
    gap = gap_today(snap_row)
    ma = compute_ma_signals(daily)
    vwap = compute_vwap_floor(k3m, current_price)
    vol = compute_volume_inflow(client if use_moomoo else None, norm, snap_row, daily)
    levels = find_historic_levels(daily)
    gex = compute_gex(client if use_moomoo else None, norm, current_price)
    earn = compute_earnings_backtest(client if use_moomoo else None, norm)
    fund = get_fundamentals_snapshot(snap_row, yf_info(norm))

    # volume rank n/a for single; user sees absolute
    # use turnover if available for high $ names from past winners
    is_top_vol = (
        (vol.get("turnover_today", 0) or 0) > MIN_ABS_TURNOVER_TODAY or
        (vol.get("volume_today", 0) or 0) > MIN_ABS_VOLUME_TODAY
    )

    score, reasons, warnings = score_setup(gap, ma, vwap, vol, gex, earn, levels, is_top_vol)

    # Pretty print
    rprint(Panel.fit(f"[bold]{disp}[/bold]  Last: {fmt(current_price)}  |  Score: [bold]{score}[/bold] / ~10", title="Summary"))

    # Gap
    color = "green" if gap["is_gap_up"] else "yellow"
    rprint(f"[{color}]GAP: {gap['detail']}[/{color}]")

    # MA
    rprint(f"[blue]MA3/MA9: {ma['detail']}[/blue]")

    # VWAP
    rprint(f"[yellow]VWAP 3m floor: {vwap['detail']}[/yellow]")

    # Volume
    rprint(f"Volume: {vol['detail']} {'[bold green]TOP TRADED[/bold green]' if is_top_vol else ''}")

    # Levels
    rprint(Panel(f"Historic key levels (daily swing):\n"
                 f"Resistances: {[round(p,2) for _,p in levels.get('resistances',[])[:4]]}\n"
                 f"Supports: {[round(p,2) for _,p in levels.get('supports',[])[:3]]}\n"
                 f"{levels['detail']}", title="Resistance / Support Context", border_style="dim"))

    # GEX
    gex_color = "green" if (gex.get("net_gex", 0) or 0) > 0 else "red"
    rprint(Panel(f"[bold {gex_color}]Net GEX (0-5DTE): {fmt(gex.get('net_gex'))}[/]\n"
                 f"Call Wall: {fmt(gex.get('call_wall'))}   Put Wall: {fmt(gex.get('put_wall'))}\n"
                 f"Regime: {gex.get('regime')}\n"
                 f"{gex.get('detail')}", title="Dealer GEX (Gamma Exposure) - 0 to 5 DTE", border_style="magenta"))
    if (gex.get("net_gex", 0) or 0) > 0:
        rprint("[dim]Note (from past winning option trades): Positive GEX + stabilizing regime often good for selling premium (short strangles/iron condors) on these names rather than just long stock.[/dim]")

    # Earnings
    rprint(Panel(f"{earn['detail']}\n{fund['detail']}\nCount: {earn.get('count',0)}", title="Earnings Outlook + Backtest (5d post)", border_style="cyan"))

    # Verdict
    verdict = "STRONG LONG SETUP" if score >= 7.0 and gap["is_gap_up"] and ma["bullish"] and vwap["strength"] > 0.5 else \
              "WATCH / CONDITIONAL" if score >= 4.5 else "WEAK / SKIP"
    rprint(Panel.fit(f"[bold]{verdict}[/bold]  (score {score})\nReasons: {'; '.join(reasons) or 'none'}\nWarnings: {'; '.join(warnings) or 'none'}", title="Verdict"))

    return {
        "ticker": disp,
        "score": score,
        "gap": gap["gap_pct"],
        "ma_bull": ma["bullish"],
        "vwap_strong": vwap["strength"] >= 0.7,
        "gex_net": gex.get("net_gex", 0),
        "earn_avg": earn.get("avg_5d_post"),
        "top_vol": is_top_vol,
        "vol_ratio": vol.get("volume_ratio"),
        "reasons": reasons,
        "warnings": warnings,
        "raw": {"gap": gap, "ma": ma, "vwap": vwap, "gex": gex, "earn": earn, "levels": levels, "fund": fund}
    }


def scan_watchlist(tickers: List[str], client: Optional[MoomooClient], use_moomoo: bool = True,
                   top_n: int = 10, min_score: float = 4.0, require_gap: bool = False,
                   require_ma: bool = True) -> List[Dict]:
    normed = [normalize_code(t) for t in tickers]
    disp_map = {n: denormalize(n) for n in normed}

    console.rule(f"[bold]Scanning {len(normed)} tickers[/bold]")

    # Batch snapshot for speed + volume ranking
    snap_df = pd.DataFrame()
    if client and use_moomoo and client.connected:
        with console.status("[cyan]Batch snapshot for volume ranking...[/cyan]"):
            snap_df = client.get_snapshot(normed)
    if snap_df.empty:
        # fallback: will compute per ticker slow path
        console.print("[yellow]No batch snapshot; falling back to per-ticker (slower).[/yellow]")

    # Compute volumes for ranking -- prefer turnover (dollar volume) because user's winning trades
    # (AAPL, NVDA, ADBE, MRVL, TXN, HOOD etc) are often high-priced names where $ volume better indicates "top traded".
    vol_map = {}
    if not snap_df.empty and "code" in snap_df:
        for _, r in snap_df.iterrows():
            c = r["code"]
            turnover = float(r.get("turnover", 0) or 0)
            vol = float(r.get("volume", 0) or 0)
            # Use turnover if available and >0, else fall back to share volume
            val = turnover if turnover > 0 else vol
            vol_map[c] = val
    # rank
    ranked = sorted(vol_map.items(), key=lambda x: x[1], reverse=True)
    top_vol_codes = set([c for c, _ in ranked[:max(1, int(len(ranked) * TOP_VOLUME_PCT_FOR_FLAG))]])

    results = []
    for code in track(normed, description="Analyzing..."):
        try:
            snap_row = None
            if not snap_df.empty:
                m = snap_df[snap_df["code"] == code]
                snap_row = m.iloc[0] if not m.empty else None

            daily = pd.DataFrame()
            k3m = pd.DataFrame()
            if client and use_moomoo and client.connected:
                daily = client.get_daily_kline(code, days=120)
                k3m = client.get_3m_kline_recent(code, num=40)

            if daily.empty:
                daily = yf_daily_history(code, period="8mo")
            if k3m.empty:
                k3m = yf_intraday_approx(code, interval="5m", days=1)

            current_price = float(snap_row["last_price"]) if snap_row is not None and "last_price" in snap_row else (float(daily["close"].iloc[-1]) if not daily.empty else 100.0)

            gap = gap_today(snap_row)
            ma = compute_ma_signals(daily)
            vwap = compute_vwap_floor(k3m, current_price)
            vol = compute_volume_inflow(client if use_moomoo else None, code, snap_row, daily)
            levels = find_historic_levels(daily)
            gex = compute_gex(client if use_moomoo else None, code, current_price)
            earn = compute_earnings_backtest(client if use_moomoo else None, code)

            is_top = (
                (code in top_vol_codes) or
                (vol.get("turnover_today", 0) or 0) > MIN_ABS_TURNOVER_TODAY or
                (vol.get("volume_today", 0) or 0) > MIN_ABS_VOLUME_TODAY
            )

            sc, reasons, _ = score_setup(gap, ma, vwap, vol, gex, earn, levels, is_top)

            if sc < min_score:
                continue
            if require_gap and not gap.get("is_gap_up"):
                continue
            if require_ma and not ma.get("bullish"):
                continue

            results.append({
                "ticker": disp_map.get(code, denormalize(code)),
                "score": sc,
                "gap": gap.get("gap_pct", 0),
                "ma_bull": ma.get("bullish", False),
                "vwap_strong": vwap.get("strength", 0) >= 0.7,
                "gex_net": gex.get("net_gex", 0),
                "earn_avg": earn.get("avg_5d_post"),
                "top_vol": is_top,
                "vol_ratio": vol.get("volume_ratio", 1.0),
                "reasons": reasons,
            })
        except Exception as e:
            if "debug" in sys.argv:
                console.print(f"[red]err {code}: {e}[/red]")
            continue

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_n]


def list_watchlist():
    console.print(Panel("\n".join(DEFAULT_WATCHLIST), title="Default Watchlist (edit in source)", border_style="dim"))


# ========================== CLI ==========================

def main():
    parser = argparse.ArgumentParser(description="Moomoo OpenD Gap-Up Swing Scanner for short-term longs (1 week horizon)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # common
    def add_common(p):
        p.add_argument("--host", default=DEFAULT_HOST)
        p.add_argument("--port", type=int, default=DEFAULT_PORT)
        p.add_argument("--no-moomoo", action="store_true", help="Force yfinance demo mode (no OpenD needed)")
        p.add_argument("--debug", action="store_true")

    p_analyze = sub.add_parser("analyze", help="Detailed single ticker report")
    p_analyze.add_argument("--ticker", required=True)
    add_common(p_analyze)

    p_scan = sub.add_parser("scan", help="Scan list and rank candidates")
    p_scan.add_argument("--tickers", default="", help="Comma sep tickers (default: built-in watchlist)")
    p_scan.add_argument("--top-n", type=int, default=8)
    p_scan.add_argument("--min-score", type=float, default=4.0)
    p_scan.add_argument("--require-gap", action="store_true")
    p_scan.add_argument("--require-ma", action="store_true", default=True)
    add_common(p_scan)

    p_list = sub.add_parser("list", help="Show default watchlist")
    p_list.set_defaults(func=lambda a: list_watchlist())

    args = parser.parse_args()

    if args.cmd == "list":
        list_watchlist()
        return

    use_moomoo = not args.no_moomoo and MOOMOO_AVAILABLE
    client = None
    if use_moomoo:
        client = MoomooClient(host=args.host, port=args.port)
        if not client.connect():
            console.print("[yellow]Continuing in fallback mode (yfinance only).[/yellow]")
            use_moomoo = False
            client = None

    try:
        if args.cmd == "analyze":
            res = analyze_ticker(args.ticker, client, use_moomoo=use_moomoo)
            # also print raw-ish for power users
            if args.debug:
                console.print(res.get("raw", {}))
        elif args.cmd == "scan":
            tickers = [t.strip() for t in args.tickers.split(",") if t.strip()] if args.tickers else DEFAULT_WATCHLIST
            cands = scan_watchlist(
                tickers, client, use_moomoo=use_moomoo,
                top_n=args.top_n, min_score=args.min_score,
                require_gap=args.require_gap, require_ma=args.require_ma
            )
            if not cands:
                console.print("[yellow]No candidates met filters. Lower --min-score or remove --require-gap.[/yellow]")
            else:
                console.print(make_candidate_table(cands))
                console.print("\n[bold green]Top picks for gap-up swing longs (review full analyze on each).[/bold green]")
                console.print("Run: python3 moomoo_swing.py analyze --ticker TICKER  for details + GEX walls.")
    finally:
        if client:
            client.close()


if __name__ == "__main__":
    main()
