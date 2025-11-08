#!/usr/bin/env python
# -*- coding: utf-8 -*-

import requests
import hashlib
import hmac
import time
import logging
import pandas as pd
import os
from datetime import datetime
from apscheduler.schedulers.blocking import BlockingScheduler

# --- 核心配置 ---
API_KEY = "MY_API_KEY"  # 请替换为您的真实 API Key
SECRET = "MY_SECRET"    # 请替换为您的真实 Secret
BASE_URL = "https://mock-api.roostoo.com"

# --- 数据与日志文件 ---
HISTORY_FILE = "price_history.csv"
LOG_FILE = "trading_bot.log"

# --- 策略参数 ---
CASH_ALLOCATION = 0.5         # 50% 资金用于持仓
TOP_N = 3                       # 投资组合中的资产数量
MAX_POSITION_RATIO = 0.25       # 单一资产最大仓位
MIN_ORDER_VALUE = 300           # 最小订单金额 (USD)

# 市场趋势与动量参数
MARKET_BENCHMARK_SYMBOL = "BTC/USD"
SMA_WINDOW = 15                 # 15日简单移动平均线
MOMENTUM_LOOKBACK_DAYS = 4      # 4日价格动量
MIN_MOMENTUM_THRESHOLD = 0.03   # 动量阈值 > 3%

# 风险管理参数
STOP_LOSS_RATIO = 0.97          # 3% 止损 (价格变为97%)
TAKE_PROFIT_RATIO = 1.33        # 33% 止盈 (价格变为133%)

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
# API 客户端 (RoostooV3Client)
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
            else: # POST
                headers['Content-Type'] = 'application/x-www-form-urlencoded'
                response = requests.post(url, data=req_data, headers=headers, timeout=15)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logging.error(f"API {method} 请求失败 ({url}): {e}")
            if e.response: logging.error(f"响应内容: {e.response.text}")
            return None

    def get_ticker(self, pair=None):
        """获取市场行情 (公开端点)"""
        url = f"{self.base_url}/v3/ticker"
        params = {"timestamp": self._get_timestamp_ms()}
        if pair: params["pair"] = pair
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
        if price: data['price'] = str(price)
        logging.info(f"准备下单: {data}")
        return self._signed_request("POST", "/v3/place_order", data=data)

    def query_order(self, pending_only="FALSE", pair=None):
        data = {"pending_only": pending_only}
        if pair: data["pair"] = pair
        return self._signed_request("POST", "/v3/query_order", data=data)

# ---
# 数据采集与策略分析
# ---
def collect_daily_price_data():
    """
    [定时任务1] 每日执行一次，采集所有交易对的“收盘价”并存入CSV。
    """
    logging.info("--- 开始每日价格采集任务 ---")
    client = RoostooV3Client(API_KEY, SECRET) # 创建一个临时的客户端实例
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
    
    # 使用 pd.to_datetime 转换以进行正确比较
    df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')

    if today_utc_str in df['date'].values:
        logging.warning(f"日期 {today_utc_str} 的数据已存在，将进行覆盖。")
        idx = df[df['date'] == today_utc_str].index
        for col, val in price_updates.items():
            if col in df.columns:
                df.loc[idx, col] = val
            else: # 新的交易对
                df.loc[idx, col] = val
    else:
        new_row = pd.DataFrame([price_updates])
        df = pd.concat([df, new_row], ignore_index=True)

    df.to_csv(HISTORY_FILE, index=False)
    logging.info(f"历史价格文件 '{HISTORY_FILE}' 已成功更新。")

class StrategyAnalytics:
    """封装所有基于历史数据的计算逻辑"""
    def __init__(self, history_file):
        try:
            self.history_df = pd.read_csv(history_file, index_col='date', parse_dates=True)
            logging.info(f"分析模块成功加载历史数据，共 {len(self.history_df)} 条记录。")
        except FileNotFoundError:
            self.history_df = None
            logging.error(f"历史数据文件 '{history_file}' 未找到！策略分析功能受限。")

    def is_bull_market(self):
        if self.history_df is None or MARKET_BENCHMARK_SYMBOL not in self.history_df.columns:
            logging.warning("无历史数据或基准代币数据，无法判断牛熊市。假定为熊市。")
            return False
        
        if len(self.history_df) < SMA_WINDOW:
            logging.warning(f"历史数据不足 {SMA_WINDOW} 天，无法计算SMA。假定为熊市。")
            return False

        sma = self.history_df[MARKET_BENCHMARK_SYMBOL].rolling(window=SMA_WINDOW).mean()
        last_price = self.history_df[MARKET_BENCHMARK_SYMBOL].iloc[-1]
        last_sma = sma.iloc[-1]

        is_bull = last_price > last_sma
        logging.info(f"牛市判断 ({MARKET_BENCHMARK_SYMBOL}): 最新价={last_price:.2f}, {SMA_WINDOW}日SMA={last_sma:.2f} -> {'牛市' if is_bull else '熊市'}")
        return is_bull

    def get_momentum_signals(self, current_ticker_map):
        if self.history_df is None or len(self.history_df) < MOMENTUM_LOOKBACK_DAYS:
            logging.warning(f"历史数据不足 {MOMENTUM_LOOKBACK_DAYS} 天，无法计算动量。")
            return []
        
        signals = []
        # 使用索引-N来安全地获取N天前的数据
        try:
            lookback_prices = self.history_df.iloc[-MOMENTUM_LOOKBACK_DAYS]
        except IndexError:
            return []

        for pair, data in current_ticker_map.items():
            if not pair.endswith("/USD") or pair not in self.history_df.columns:
                continue

            open_price = lookback_prices.get(pair)
            close_price = data.get("LastPrice")
            if open_price and close_price and open_price > 0:
                momentum = (close_price - open_price) / open_price
                if momentum > MIN_MOMENTUM_THRESHOLD:
                    signals.append({"pair": pair, "momentum": momentum, "price": close_price})
        
        signals.sort(key=lambda x: x["momentum"], reverse=True)
        return signals

# ---
# 主策略逻辑与执行
# ---
def run_strategy():
    """
    [定时任务2] 每日执行一次，完成持仓检查、市场分析和交易决策。
    """
    logging.info("========== 开始每日策略执行任务 ==========")
    client = RoostooV3Client(API_KEY, SECRET)
    analytics = StrategyAnalytics(HISTORY_FILE)

    # 1. 获取实时行情
    ticker_data = client.get_ticker()
    if not ticker_data or not ticker_data.get("Success"):
        logging.error("无法获取 Ticker 行情, 策略中止。")
        return
    ticker_map = ticker_data.get("Data", {})

    # 2. 检查止盈止损 (此部分逻辑较复杂，可以从之前的脚本中完整移入)
    # check_stop_loss_take_profit(client, ticker_map) # 暂时注释，需要时取消注释并实现

    # 3. 判断牛熊市
    if not analytics.is_bull_market():
        logging.info("市场判定为熊市，不执行再平衡。考虑清仓所有非USD资产。")
        # 此处可以添加清仓逻辑，以在熊市中保持空仓
        return
    
    logging.info("市场判定为牛市，继续执行再平衡。")

    # 4. 获取动量信号并选择Top N
    signals = analytics.get_momentum_signals(ticker_map)
    top_targets = signals[:TOP_N]
    target_pairs = {t["pair"] for t in top_targets}

    if not top_targets:
        logging.warning("未找到任何符合动量条件的资产。")
        return
    
    logging.info(f"选定的 Top {TOP_N} 动量目标: {[t['pair'] for t in top_targets]}")

    # 5. 计算投资组合价值和目标仓位
    balance_data = client.get_balance()
    if not balance_data or not balance_data.get("Success"):
        logging.error("无法获取账户余额, 策略中止。")
        return
    
    wallet = balance_data.get("Wallet", balance_data.get("SpotWallet", {}))
    total_usd_balance = wallet.get("USD", {}).get("Free", 0) + wallet.get("USD", {}).get("Lock", 0)
    total_portfolio_value = total_usd_balance
    current_holdings = {}

    for coin, data in wallet.items():
        if coin == "USD": continue
        pair = f"{coin}/USD"
        if pair in ticker_map:
            price = ticker_map[pair].get("LastPrice", 0)
            quantity = data.get("Free", 0) + data.get("Lock", 0)
            if price > 0 and quantity > 0:
                value_usd = quantity * price
                current_holdings[pair] = {"value": value_usd, "quantity": quantity}
                total_portfolio_value += value_usd
    
    logging.info(f"总资产价值 (估算): ${total_portfolio_value:.2f}")

    # 6. 执行再平衡交易
    equity_to_invest = total_portfolio_value * CASH_ALLOCATION
    target_value_per_position = min(
        equity_to_invest / TOP_N if TOP_N > 0 else 0,
        total_portfolio_value * MAX_POSITION_RATIO
    )

    # 卖出逻辑
    for pair, holding in current_holdings.items():
        if pair not in target_pairs:
            coin = pair.split('/')[0]
            logging.info(f"再平衡 [卖出]: {pair} 不在 Top {TOP_N} 列表中，卖出数量 {holding['quantity']:.6f}")
            client.place_order(coin, "SELL", holding['quantity'])

    # 买入逻辑
    for target in top_targets:
        pair, coin, price = target["pair"], target["pair"].split('/')[0], target["price"]
        if price <= 0: continue

        current_value = current_holdings.get(pair, {}).get("value", 0)
        trade_value = target_value_per_position - current_value
        
        if abs(trade_value) < MIN_ORDER_VALUE:
            continue

        if trade_value > 0:
            quantity_to_buy = trade_value / price
            logging.info(f"再平衡 [买入]: {pair}，目标价值 ${target_value_per_position:.2f}，需买入 ${trade_value:.2f} (数量: {quantity_to_buy:.6f})")
            client.place_order(coin, "BUY", quantity_to_buy)

    logging.info("========== 每日策略执行任务完成 ==========")

# ---
# 启动与调度
# ---
if __name__ == "__main__":
    logging.info("交易机器人启动...")
    
    # --- 首次运行 ---
    # 在启动时立即执行一次，确保服务可用
    try:
        collect_daily_price_data()
        run_strategy()
    except Exception as e:
        logging.critical(f"首次运行时发生严重错误: {e}", exc_info=True)

    # --- 设置定时任务 ---
    scheduler = BlockingScheduler(timezone="UTC")
    
    # 任务1: 每日23:59 (UTC) 采集价格数据
    scheduler.add_job(collect_daily_price_data, 'cron', hour=23, minute=59)
    
    # 任务2: 每日00:15 (UTC) 运行策略 (在数据采集后16分钟)
    scheduler.add_job(run_strategy, 'cron', hour=0, minute=15)
    
    logging.info("调度器已配置完毕。机器人进入持续运行模式...")
    print("="*60)
    print("交易机器人正在运行。按 Ctrl+C 停止。")
    print("日志将记录在 trading_bot.log 文件中。")
    print(f"数据采集任务将在每天 23:59 UTC 执行。")
    print(f"策略交易任务将在每天 00:15 UTC 执行。")
    print("="*60)
    
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logging.info("交易机器人已手动停止。")
