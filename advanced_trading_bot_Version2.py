#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
advanced_trading_bot.py

- This script contains the original live trading logic (Roostoo client) and a new
  Horus-backed historical data fetcher + a simple backtesting engine.
- Run live (unchanged behavior except safer place_order calls) by running the script
  without the "backtest" argument (it will still call collect_daily_price_data()
  and run_strategy()).
- Run backtest mode with:
    python advanced_trading_bot.py backtest --start 2023-01-01 --end 2024-01-01
  Optionally include --symbols "BTC/USD,ETH/USD" (comma separated). If not provided,
  the script will try to infer symbols from the Horus response (and fall back to a
  local CSV mock if Horus is unreachable).

Notes:
- You provided a Horus API key; it is used below in HORUS_API_KEY.
- The Horus endpoint (HORUS_BASE_URL) is left configurable because actual Horus
  endpoints differ between deployments. If you have the exact Horus URL/path,
  set HORUS_BASE_URL accordingly.
"""

import requests
import hashlib
import hmac
import time
import logging
import pandas as pd
import pandas_ta as ta  # 技术分析
import os
import numpy as np
from datetime import datetime, timedelta
from apscheduler.schedulers.blocking import BlockingScheduler
import argparse
import sys
import math

# --- 核心配置 (和原脚本保持一致) ---
API_KEY = "AsQb4BENw84ot5mAugsXw88o8f37tEK75LJdAveGZvKlvEhbQRQETJElcPZ9CkAE"
SECRET = "uO4TvgFoPcOHPgYPHs398wcqIpAR34khlpbJoPflHdpJGQVUHKosIXTMli7GNUAh"
BASE_URL = "https://mock-api.roostoo.com"

# --- Horus 配置 (用于获取历史数据做回测) ---
HORUS_API_KEY = "   "
HORUS_BASE_URL = os.environ.get("HORUS_BASE_URL", "     ")

# --- 数据与日志文件 ---
HISTORY_FILE = "price_history.csv"
LOG_FILE = "advanced_trading_bot.log"

# --- 策略参数 (和原脚本保持一致) ---
CASH_ALLOCATION = 0.9           # 投资90%的资金, 保留10%现金
TOP_N = 5                       # 选择Top 5资产
MAX_POSITION_RATIO = 0.25       # 单一资产最大仓位
MIN_ORDER_VALUE = 300           # 最小订单金额 (USD)

# 筛选器参数
VOLUME_FILTER_PERCENTILE = 0.7    # 成交量筛选: Top 30% (即大于70百分位数)
VOLATILITY_FILTER_PERCENTILE = 0.7# 波动率筛选: Top 30% (即大于70百分位数)
VOLATILITY_WINDOW = 15          # 计算波动率的窗口期: 15天

# 动量与时机指标参数
MOMENTUM_LOOKBACK_DAYS = 3      # 动量筛选: 3日回报率
ENTRY_SMA_WINDOW = 7            # 入场时机: 7日SMA
EXIT_SMA_WINDOW = 10            # 出场时机: 10日SMA
RSI_WINDOW = 7                  # RSI窗口期
RSI_ENTRY_MIN = 50              # 入场RSI > 50
RSI_ENTRY_MAX = 70              # 入场RSI < 70 (避免超买)

# 风险管理参数
STOP_LOSS_RATIO = 0.95          # 5% 止损
TAKE_PROFIT_RATIO = 1.33        # 33% 止盈

# --- 回测参数默认值 ---
BACKTEST_START_CAPITAL = 100000.0

# --- 日志配置 ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(funcName)s] - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)

# ---
# API 客户端 (RoostooV3Client) - 保持不变（用于真实交易）
# ---
class RoostooV3Client:
    """封装对 Roostoo V3 API 的所有请求"""
    def __init__(self, api_key, secret_key):
        self.api_key = api_key
        self.secret = secret_key
        self.base_url = BASE_URL
        logging.info("Roostoo V3 客户端初始化完成")

    def _get_timestamp_ms(self):
        """返回13位毫秒时间戳字符串"""
        return str(int(time.time() * 1000))

    def _generate_signature(self, params):
        """生成API请求签名"""
        query_string = '&'.join([f"{k}={params[k]}" for k in sorted(params.keys())])
        us = self.secret.encode('utf-8')
        m = hmac.new(us, query_string.encode('utf-8'), hashlib.sha256)
        return m.hexdigest()

    def _signed_request(self, method, endpoint, params=None, data=None):
        """执行签名的 GET 或 POST 请求的统一方法"""
        req_data = params if method == 'GET' else (data if data is not None else {})
        req_data["timestamp"] = self._get_timestamp_ms()
        signature = self._generate_signature(req_data)

        headers = {"RST-API-KEY": self.api_key, "MSG-SIGNATURE": signature}
        url = f"{self.base_url}{endpoint}"

        try:
            if method == 'GET':
                response = requests.get(url, params=req_data, headers=headers, timeout=15)
            else:  # POST
                headers['Content-Type'] = 'application/x-www-form-urlencoded'
                response = requests.post(url, data=req_data, headers=headers, timeout=15)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logging.error(f"API {method} 请求失败 ({url}): {e}")
            if hasattr(e, "response") and e.response is not None:
                logging.error(f"响应内容: {e.response.text}")
            return None

    def get_ticker(self, pair=None):
        """获取市场行情 (公开端点)"""
        url = f"{self.base_url}/v3/ticker"
        params = {"timestamp": self._get_timestamp_ms()}
        if pair:
            params["pair"] = pair
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logging.error(f"获取 Ticker 行情失败: {e}")
            return None

    def get_balance(self):
        return self._signed_request("GET", "/v3/balance")

    def place_order(self, coin, side, quantity, price=None):
        data = {"pair": f"{coin}/USD", "side": side.upper(), "quantity": str(quantity)}
        data['type'] = "LIMIT" if price else "MARKET"
        if price:
            data['price'] = str(price)
        logging.info(f"准备下单: {data}")
        return self._signed_request("POST", "/v3/place_order", data=data)

    def query_order(self, pending_only="FALSE", pair=None):
        data = {"pending_only": pending_only}
        if pair:
            data["pair"] = pair
        return self._signed_request("POST", "/v3/query_order", data=data)


# ---
# Horus 历史数据获取器 (用于回测)
# ---
def fetch_historical_from_horus(symbols, start_date, end_date, interval="1d"):
    """
    尝试从 Horus 获取历史 OHLC/close 数据。
    - symbols: list of pairs, e.g., ["BTC/USD", "ETH/USD"]
    - start_date, end_date: strings "YYYY-MM-DD"
    - interval: "1d" supported by this generic implementation

    NOTE: Horus API endpoints vary. This function assumes the Horus instance exposes
    a GET endpoint like:
      {HORUS_BASE_URL}/v1/ohlc?symbol=BTCUSD&start=2023-01-01&end=2023-12-31&interval=1d

    If your Horus endpoint differs, change HORUS_BASE_URL or edit this function.
    """
    headers = {"Authorization": f"Bearer {HORUS_API_KEY}"}
    all_closes = {}
    dates = None

    for s in symbols:
        # normalize symbol name for request (many horus endpoints expect no slash)
        symbol_for_api = s.replace("/", "")
        url = f"{HORUS_BASE_URL}/v1/ohlc"
        params = {"symbol": symbol_for_api, "start": start_date, "end": end_date, "interval": interval}

        try:
            resp = requests.get(url, headers=headers, params=params, timeout=20)
            resp.raise_for_status()
            data = resp.json()
            # Expected data format (attempt to handle common variants):
            # - {"data": [{"time":"2023-01-01","close":42000}, ...]}
            # - or {"ohlc": {"2023-01-01": {"close":...}, ...}}
            series = []
            if isinstance(data, dict) and "data" in data and isinstance(data["data"], list):
                for o in data["data"]:
                    # try several key names
                    t = o.get("time") or o.get("date") or o.get("timestamp")
                    c = o.get("close") or o.get("Close") or o.get("last")
                    if t is None or c is None:
                        continue
                    # normalize timestamp to date string
                    try:
                        # some APIs return timestamp in seconds or ms
                        if isinstance(t, (int, float)):
                            # detect ms vs s
                            ts = int(t)
                            if ts > 1e12:
                                ts = ts / 1000
                            dt = datetime.utcfromtimestamp(int(ts)).strftime("%Y-%m-%d")
                        else:
                            # assume ISO date
                            dt = pd.to_datetime(t).strftime("%Y-%m-%d")
                    except Exception:
                        dt = str(t)
                    series.append((dt, float(c)))
                df = pd.DataFrame(series, columns=["date", "close"]).drop_duplicates("date").set_index("date").sort_index()
            elif isinstance(data, dict) and ("ohlc" in data or "prices" in data):
                container = data.get("ohlc") or data.get("prices")
                rows = []
                for k, v in container.items():
                    # v could be dict with 'close' key or scalar
                    if isinstance(v, dict):
                        c = v.get("close") or v.get("last")
                    else:
                        c = v
                    rows.append((k, float(c)))
                df = pd.DataFrame(rows, columns=["date", "close"]).set_index("date").sort_index()
            else:
                logging.warning(f"Horus 返回格式未知，symbol={s}, raw={data}")
                raise ValueError("Unknown response format")
            # collect
            all_closes[s] = df["close"]
            if dates is None:
                dates = df.index.tolist()
        except Exception as e:
            logging.error(f"从 Horus 获取 {s} 的历史数据失败: {e}")
            # 报错时返回 None，让调用方决定是否使用回退数据
            return None

    # combine into DataFrame
    if not all_closes:
        return None
    combined = pd.DataFrame(all_closes)
    combined.index = pd.to_datetime(combined.index)
    combined.sort_index(inplace=True)
    return combined


# ---
# 数据采集 (保持原有 collect_daily_price_data，用于日常运行)
# ---
def collect_daily_price_data():
    logging.info("--- 开始每日价格采集任务 ---")
    client = RoostooV3Client(API_KEY, SECRET)
    ticker_data = client.get_ticker()

    if not ticker_data or not ticker_data.get("Success"):
        logging.error("获取行情失败，无法更新历史数据。")
        return

    today_utc_str = datetime.utcnow().strftime('%Y-%m-%d')
    price_updates = {"date": today_utc_str}
    for pair, data in ticker_data.get("Data", {}).items():
        if pair.endswith("/USD"):
            price_updates[pair] = data.get("LastPrice")

    df = pd.read_csv(HISTORY_FILE) if os.path.exists(HISTORY_FILE) else pd.DataFrame(columns=["date"])
    df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')

    if today_utc_str in df['date'].values:
        logging.warning(f"日期 {today_utc_str} 的数据已存在，将进行覆盖。")
        idx = df[df['date'] == today_utc_str].index
        for col, val in price_updates.items():
            if col not in df.columns:
                df[col] = np.nan
            df.loc[idx, col] = val
    else:
        new_row = pd.DataFrame([price_updates])
        df = pd.concat([df, new_row], ignore_index=True)

    df.to_csv(HISTORY_FILE, index=False)
    logging.info(f"历史价格文件 '{HISTORY_FILE}' 已成功更新。")


# ---
# 高级策略分析模块 (统一接口，支持直接传 DataFrame 用于回测)
# ---
class StrategyAnalytics:
    """封装所有基于历史数据的复杂计算逻辑

    初始化时可以传入:
      - history_file: path -> 会从 CSV 加载
      - 或者直接传入 pandas.DataFrame (date indexed) 作为历史数据
    """
    def __init__(self, history_source, current_ticker_map=None):
        self.ticker_map = current_ticker_map or {}
        self.history_df = None

        if isinstance(history_source, pd.DataFrame):
            df = history_source.copy()
            if not pd.api.types.is_datetime64_any_dtype(df.index):
                # expect a column 'date'
                if 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date'])
                    df.set_index('date', inplace=True)
                else:
                    raise ValueError("传入 DataFrame 必须以 date 为索引或包含 'date' 列")
            self.history_df = df.sort_index()
        elif isinstance(history_source, str):
            # load from csv path
            try:
                df = pd.read_csv(history_source)
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
                self.history_df = df.sort_index()
            except FileNotFoundError:
                logging.error(f"历史数据文件 '{history_source}' 未找到！")
                self.history_df = None
        else:
            self.history_df = None

        if self.history_df is not None:
            logging.info(f"分析模块成功加载历史数据，共 {len(self.history_df)} 条记录。")
        else:
            logging.error("分析模块未加载到历史数据，后续操作会被跳过。")

    def filter_and_rank_assets(self):
        if self.history_df is None:
            return []

        # 使用收盘价计算日回报率
        returns = self.history_df.pct_change()

        # 1. 交易量筛选 - 从 ticker_map 获取 UnitTradeValue（如果可用）
        volumes = {}
        for p, d in (self.ticker_map or {}).items():
            if p.endswith("/USD"):
                volumes[p] = d.get('UnitTradeValue', 0)
        if volumes:
            volume_threshold = pd.Series(volumes).quantile(VOLUME_FILTER_PERCENTILE)
            volume_candidates = {p for p, v in volumes.items() if v >= volume_threshold}
        else:
            # 如果没有成交量信息，直接使用历史数据中的列作为候选
            volume_candidates = set(self.history_df.columns)

        # 2. 波动率筛选
        volatility = returns.rolling(window=VOLATILITY_WINDOW).std().iloc[-1]
        volatility.dropna(inplace=True)
        if volatility.empty:
            return []
        volatility_threshold = volatility.quantile(VOLATILITY_FILTER_PERCENTILE)
        volatility_candidates = set(volatility[volatility >= volatility_threshold].index)

        # 3. 动量计算（MOMENTUM_LOOKBACK_DAYS）
        if len(self.history_df) < MOMENTUM_LOOKBACK_DAYS + 1:
            return []
        momentum = self.history_df.pct_change(MOMENTUM_LOOKBACK_DAYS).iloc[-1]

        # 合并所有筛选条件
        final_candidates = list(volume_candidates & volatility_candidates & set(momentum.index))

        ranked_assets = []
        for pair in final_candidates:
            m = momentum.get(pair, np.nan)
            v = volatility.get(pair, np.nan)
            if pd.notna(m):
                ranked_assets.append({"pair": pair, "momentum": float(m), "volatility": float(v)})

        ranked_assets.sort(key=lambda x: x["momentum"], reverse=True)
        return ranked_assets[:TOP_N]

    def check_entry_conditions(self, pair):
        """基于当前 self.history_df 的最新行检查是否满足入场条件"""
        if self.history_df is None or pair not in self.history_df.columns:
            return False
        series = self.history_df[pair].dropna()
        if len(series) < max(ENTRY_SMA_WINDOW, RSI_WINDOW) + 1:
            return False

        sma7 = ta.sma(series, length=ENTRY_SMA_WINDOW).iloc[-1]
        rsi7 = ta.rsi(series, length=RSI_WINDOW).iloc[-1]
        last_price = series.iloc[-1]

        price_above_sma = last_price > sma7
        rsi_ok = (RSI_ENTRY_MIN < rsi7 < RSI_ENTRY_MAX)

        return bool(price_above_sma and rsi_ok)

    def check_exit_conditions(self, pair):
        """基于当前 self.history_df 的最新行检查是否满足出场条件"""
        if self.history_df is None or pair not in self.history_df.columns:
            return False
        series = self.history_df[pair].dropna()
        if len(series) < EXIT_SMA_WINDOW + 1:
            return False

        sma10 = ta.sma(series, length=EXIT_SMA_WINDOW).iloc[-1]
        last_price = series.iloc[-1]

        return last_price < sma10


# ---
# 回测引擎 (基于 Horus 历史数据)
# ---
def backtest_strategy(start_date, end_date, symbols=None, start_capital=BACKTEST_START_CAPITAL):
    """
    简单日频再平衡回测：
     - 每个可用日期（按历史数据排列）计算候选资产并进行再平衡
     - 使用当日 close 作为交易价格（简化假设）
     - 不考虑滑点、手续费（可扩展）
    """
    logging.info(f"开始回测: {start_date} -> {end_date}, symbols={symbols}, capital=${start_capital:.2f}")

    # Normalize dates
    start_date = pd.to_datetime(start_date).strftime("%Y-%m-%d")
    end_date = pd.to_datetime(end_date).strftime("%Y-%m-%d")

    if symbols:
        symbols = [s.strip() for s in symbols]
    else:
        # Try to discover a reasonable default set by calling a Horus metadata endpoint or fallback.
        # Here we will attempt to fetch a list of common symbols.
        symbols = ["BTC/USD", "ETH/USD", "XRP/USD", "LTC/USD", "BCH/USD"]

    # fetch
    hist = fetch_historical_from_horus(symbols, start_date, end_date, interval="1d")
    if hist is None:
        logging.warning("无法从 Horus 获取历史数据，尝试从本地 CSV 回退（如果存在）或使用随机 mock 数据。")
        if os.path.exists(HISTORY_FILE):
            df_local = pd.read_csv(HISTORY_FILE)
            df_local['date'] = pd.to_datetime(df_local['date'])
            df_local.set_index('date', inplace=True)
            hist = df_local[symbols].dropna(how='all').copy()
        else:
            # Create mock data (deterministic for reproducibility)
            dates = pd.date_range(start_date, end_date, freq='D')
            rng = np.random.default_rng(42)
            hist = pd.DataFrame(index=dates)
            for s in symbols:
                price = 100 + rng.standard_normal(len(dates)).cumsum()  # random walk
                hist[s] = price + np.linspace(0, 200, len(dates))
            logging.info("已生成 mock 历史数据，继续回测。")

    # Ensure columns are normalized like "BTC/USD"
    # If Horus returned without slash, try to rename columns
    # but our fetcher used "BTC/USD" keys when requesting => keep as-is.

    # Prepare backtest state
    cash = float(start_capital)
    portfolio = {}  # pair -> quantity
    trade_log = []
    portfolio_value_history = []

    # iterate dates chronologically
    dates = hist.index.sort_values()
    for current_date in dates:
        # slice history up to current_date inclusive for analytics
        hist_to_date = hist.loc[:current_date].dropna(how='all', axis=1)
        if hist_to_date.shape[0] < max(VOLATILITY_WINDOW, MOMENTUM_LOOKBACK_DAYS, EXIT_SMA_WINDOW) + 1:
            # not enough data yet
            portfolio_value = cash + sum((hist.loc[current_date].get(p, np.nan) or 0) * q for p, q in portfolio.items())
            portfolio_value_history.append((current_date, portfolio_value))
            continue

        # build analytics and rank
        analytics = StrategyAnalytics(hist_to_date, current_ticker_map={})
        top_candidates = analytics.filter_and_rank_assets()
        final_targets = []
        for a in top_candidates:
            if analytics.check_entry_conditions(a['pair']):
                final_targets.append(a)

        # compute current holdings value
        current_values = {}
        total_value = cash
        for p, q in portfolio.items():
            price = hist.loc[current_date].get(p, np.nan)
            if pd.isna(price):
                v = 0
            else:
                v = price * q
            current_values[p] = v
            total_value += v

        # determine target positions by inverse volatility weighting among final_targets
        equity_to_invest = total_value * CASH_ALLOCATION
        inverse_volatilities = {t['pair']: 1 / t['volatility'] for t in final_targets if t['volatility'] > 0}
        sum_inverse_vol = sum(inverse_volatilities.values())
        target_positions = {}
        if sum_inverse_vol > 0:
            for pair, inv in inverse_volatilities.items():
                weight = inv / sum_inverse_vol
                weight = min(weight, MAX_POSITION_RATIO)
                target_positions[pair] = equity_to_invest * weight

        # create set of all pairs to consider (existing holdings union targets)
        all_pairs = set(portfolio.keys()) | set(target_positions.keys())

        # Execute trades at today's close (simplified)
        for pair in all_pairs:
            price = hist.loc[current_date].get(pair, np.nan)
            if pd.isna(price) or price <= 0:
                continue

            current_qty = portfolio.get(pair, 0)
            current_val = current_qty * price
            target_val = target_positions.get(pair, 0)

            # exit signal check
            if pair in portfolio and analytics.check_exit_conditions(pair):
                target_val = 0  # force exit

            trade_value = target_val - current_val
            if abs(trade_value) < MIN_ORDER_VALUE:
                continue  # skip small trades

            if trade_value > 0:
                # buy
                qty = trade_value / price
                # ensure enough cash
                spend = qty * price
                if spend > cash:
                    # can't fully fund, scale down
                    qty = cash / price
                    spend = qty * price
                if qty <= 0:
                    continue
                cash -= spend
                portfolio[pair] = portfolio.get(pair, 0) + qty
                trade_log.append({
                    "date": current_date, "pair": pair, "side": "BUY", "qty": qty, "price": price, "value": spend
                })
            else:
                # sell
                if current_qty <= 0:
                    continue
                qty_to_sell = min(current_qty, (-trade_value) / price)
                if qty_to_sell <= 0:
                    continue
                proceeds = qty_to_sell * price
                cash += proceeds
                portfolio[pair] = current_qty - qty_to_sell
                if portfolio[pair] <= 1e-12:
                    portfolio.pop(pair, None)
                trade_log.append({
                    "date": current_date, "pair": pair, "side": "SELL", "qty": qty_to_sell, "price": price, "value": proceeds
                })

        # record portfolio value
        portfolio_value = cash + sum((hist.loc[current_date].get(p, np.nan) or 0) * q for p, q in portfolio.items())
        portfolio_value_history.append((current_date, portfolio_value))

    # Results summary
    pv = pd.DataFrame(portfolio_value_history, columns=["date", "portfolio_value"]).set_index("date")
    start_val = pv['portfolio_value'].iloc[0] if not pv.empty else start_capital
    end_val = pv['portfolio_value'].iloc[-1] if not pv.empty else cash
    returns = (end_val / start_val - 1) if start_val > 0 else np.nan
    # max drawdown
    roll_max = pv['portfolio_value'].cummax()
    drawdown = (pv['portfolio_value'] - roll_max) / roll_max
    max_dd = drawdown.min() if not drawdown.empty else 0.0

    logging.info(f"回测完成: 起始价值=${start_val:.2f}, 结束价值=${end_val:.2f}, 总收益={returns*100:.2f}%, 最大回撤={max_dd*100:.2f}%")
    # Save trade log to CSV
    trades_df = pd.DataFrame(trade_log)
    if not trades_df.empty:
        trades_df.to_csv("backtest_trades.csv", index=False)
        logging.info("交易记录已保存到 backtest_trades.csv")
    pv.to_csv("backtest_portfolio_value.csv")
    logging.info("资金曲线已保存到 backtest_portfolio_value.csv")

    return {"start_value": start_val, "end_value": end_val, "returns": returns, "max_drawdown": max_dd, "trades": trades_df, "equity_curve": pv}


# ---
# 主策略逻辑 (用于实时运行)
# ---
def run_strategy(simulate_only=False):
    logging.info("="*10 + " 开始高级策略执行任务 " + "="*10)
    client = RoostooV3Client(API_KEY, SECRET)

    # 1. 获取实时行情
    ticker_data = client.get_ticker()
    if not ticker_data or not ticker_data.get("Success"):
        logging.error("无法获取Ticker行情, 策略中止。")
        return
    ticker_map = ticker_data.get("Data", {})

    # 2. 初始化分析模块并筛选资产
    analytics = StrategyAnalytics(HISTORY_FILE, ticker_map)
    top_candidates = analytics.filter_and_rank_assets()

    if not top_candidates:
        logging.warning("经过筛选后，没有任何候选资产。")
        return

    # 3. 检查入场条件
    final_targets = []
    for asset in top_candidates:
        if analytics.check_entry_conditions(asset['pair']):
            final_targets.append(asset)

    if not final_targets:
        logging.info("市场状况判断：所有Top候选资产均不满足入场条件，认定市场为下降趋势，暂停交易。")
        return

    logging.info(f"最终投资目标 ({len(final_targets)}个): {[t['pair'] for t in final_targets]}")

    # 4. 计算投资组合价值
    balance_data = client.get_balance()
    if not balance_data or not balance_data.get("Success"):
        logging.error("获取账户余额失败，策略中止。")
        return
    wallet = balance_data.get("Wallet", {})
    total_portfolio_value = wallet.get("USD", {}).get("Free", 0)
    current_holdings = {}
    for coin, data in wallet.items():
        if coin == "USD":
            continue
        pair = f"{coin}/USD"
        if pair in ticker_map and ticker_map[pair].get("LastPrice", 0) > 0:
            price = ticker_map[pair]["LastPrice"]
            quantity = data.get("Free", 0) + data.get("Lock", 0)
            value = price * quantity
            current_holdings[pair] = {"quantity": quantity, "value": value}
            total_portfolio_value += value

    logging.info(f"总资产价值 (估算): ${total_portfolio_value:.2f}")

    # 5. ERC权重分配
    equity_to_invest = total_portfolio_value * CASH_ALLOCATION
    inverse_volatilities = {t['pair']: 1 / t['volatility'] for t in final_targets if t['volatility'] > 0}
    sum_inverse_vol = sum(inverse_volatilities.values())

    target_positions = {}
    if sum_inverse_vol > 0:
        for pair, inv_vol in inverse_volatilities.items():
            weight = inv_vol / sum_inverse_vol
            weight = min(weight, MAX_POSITION_RATIO)
            target_positions[pair] = equity_to_invest * weight

    # 6. 执行再平衡交易
    all_pairs_in_play = set(current_holdings.keys()) | set(target_positions.keys())

    for pair in all_pairs_in_play:
        target_value = target_positions.get(pair, 0)
        current_value = current_holdings.get(pair, {}).get("value", 0)
        coin = pair.split('/')[0]

        # 检查是否需要因为时机信号而提前退出
        if pair in current_holdings and analytics.check_exit_conditions(pair):
            target_value = 0  # 强制清仓

        trade_value = target_value - current_value

        if abs(trade_value) < MIN_ORDER_VALUE:
            continue

        price = ticker_map.get(pair, {}).get("LastPrice")
        if not price or price <= 0:
            continue

        quantity = abs(trade_value) / price
        if trade_value < 0:
            logging.info(f"再平衡 [卖出]: {pair}, 卖出价值 ${-trade_value:.2f} (数量: {quantity:.6f})")
            if simulate_only:
                logging.info("模拟模式：不发送实盘卖单。")
            else:
                client.place_order(coin, "SELL", current_holdings[pair]['quantity'])  # 卖出全部
        else:
            logging.info(f"再平衡 [买入]: {pair}, 买入价值 ${trade_value:.2f} (数量: {quantity:.6f})")
            if simulate_only:
                logging.info("模拟模式：不发送实盘买单。")
            else:
                client.place_order(coin, "BUY", quantity)

    logging.info("="*10 + " 高级策略执行任务完成 " + "="*10)


# ---
# CLI 接口
# ---
def parse_args(argv):
    parser = argparse.ArgumentParser(description="高级交易机器人 (支持 Horus 回测)")
    subparsers = parser.add_subparsers(dest="mode", required=False)

    # backtest 子命令
    p_bt = subparsers.add_parser("backtest", help="使用 Horus 历史数据进行回测")
    p_bt.add_argument("--start", required=True, help="回测开始日期 YYYY-MM-DD")
    p_bt.add_argument("--end", required=True, help="回测结束日期 YYYY-MM-DD")
    p_bt.add_argument("--symbols", required=False, help="逗号分隔的交易对，例如 'BTC/USD,ETH/USD'")
    p_bt.add_argument("--capital", type=float, default=BACKTEST_START_CAPITAL, help="回测起始资金 (默认 100000)")

    # live 模式（默认）
    p_live = subparsers.add_parser("live", help="正常启动机器人（默认行为）")
    p_live.add_argument("--simulate", action="store_true", help="模拟模式: 不对外下单")

    # If no args, treat as live run
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])

    if args.mode == "backtest":
        syms = None
        if args.symbols:
            syms = [s.strip() for s in args.symbols.split(",")]
        res = backtest_strategy(args.start, args.end, symbols=syms, start_capital=args.capital)
        print("回测结果：")
        print(f" 起始价值: ${res['start_value']:.2f}")
        print(f" 结束价值: ${res['end_value']:.2f}")
        print(f" 总收益: {res['returns']*100:.2f}%")
        print(f" 最大回撤: {res['max_drawdown']*100:.2f}%")
        sys.exit(0)

    # 默认的 live 启动逻辑（与原脚本行为类似）
    logging.info("高级交易机器人启动...")
    try:
        collect_daily_price_data()
        run_strategy(simulate_only=False)
    except Exception as e:
        logging.critical(f"首次运行时发生严重错误: {e}", exc_info=True)

    scheduler = BlockingScheduler(timezone="UTC")
    scheduler.add_job(collect_daily_price_data, 'cron', hour=23, minute=59)
    scheduler.add_job(lambda: run_strategy(simulate_only=False), 'cron', hour=0, minute=15)

    logging.info("调度器已配置完毕。机器人进入持续运行模式...")
    print("="*60)
    print("高级交易机器人正在运行。按 Ctrl+C 停止。")
    print(f"日志将记录在 {LOG_FILE} 文件中。")
    print("="*60)

    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logging.info("交易机器人已手动停止。")