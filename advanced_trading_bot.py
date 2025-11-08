#!/usr/bin/env python
# -*- coding: utf-8 -*-

import requests
import hashlib
import hmac
import time
import logging
import pandas as pd
import pandas_ta as ta # 引入专业的技术分析库
import os
import numpy as np
from datetime import datetime
from apscheduler.schedulers.blocking import BlockingScheduler

# --- 核心配置 (根据新策略文档更新) ---
API_KEY = "MY_API_KEY"
SECRET = "MY_SECRET"
BASE_URL = "https://mock-api.roostoo.com"

# --- 数据与日志文件 ---
HISTORY_FILE = "price_history.csv"
LOG_FILE = "advanced_trading_bot.log"

# --- 策略参数 (根据 Wenying-Li 的新文档) ---
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
# API 客户端 (RoostooV3Client) - 保持不变
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
    # 其他API方法 get_balance, place_order, query_order 保持不变...
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
# 数据采集 (保持不变)
# ---
def collect_daily_price_data():
    # ... 此函数与上一版完全相同，用于每日采集数据 ...
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
            if col not in df.columns: df[col] = np.nan
            df.loc[idx, col] = val
    else:
        new_row = pd.DataFrame([price_updates])
        df = pd.concat([df, new_row], ignore_index=True)

    df.to_csv(HISTORY_FILE, index=False)
    logging.info(f"历史价格文件 '{HISTORY_FILE}' 已成功更新。")

# ---
# 高级策略分析模块 (全面重构)
# ---
class StrategyAnalytics:
    """封装所有基于历史数据的复杂计算逻辑"""
    def __init__(self, history_file, current_ticker_map):
        self.ticker_map = current_ticker_map
        try:
            self.history_df = pd.read_csv(history_file)
            self.history_df['date'] = pd.to_datetime(self.history_df['date'])
            self.history_df.set_index('date', inplace=True)
            logging.info(f"分析模块成功加载历史数据，共 {len(self.history_df)} 条记录。")
        except FileNotFoundError:
            self.history_df = None
            logging.error(f"历史数据文件 '{history_file}' 未找到！策略无法执行。")

    def filter_and_rank_assets(self):
        if self.history_df is None: return []

        # 计算日回报率
        returns = self.history_df.pct_change()
        
        # 1. 交易量筛选
        volumes = {p: d.get('UnitTradeValue', 0) for p, d in self.ticker_map.items() if p.endswith("/USD")}
        volume_threshold = pd.Series(volumes).quantile(VOLUME_FILTER_PERCENTILE)
        volume_candidates = {p for p, v in volumes.items() if v >= volume_threshold}
        logging.info(f"交易量筛选: {len(volume_candidates)}/{len(volumes)} 个资产满足条件 (>{volume_threshold:.2f})")

        # 2. 波动率筛选
        volatility = returns.rolling(window=VOLATILITY_WINDOW).std().iloc[-1]
        volatility.dropna(inplace=True)
        volatility_threshold = volatility.quantile(VOLATILITY_FILTER_PERCENTILE)
        volatility_candidates = set(volatility[volatility >= volatility_threshold].index)
        logging.info(f"波动率筛选: {len(volatility_candidates)}/{len(volatility)} 个资产满足条件 (>{volatility_threshold:.6f})")

        # 3. 动量计算
        if len(self.history_df) < MOMENTUM_LOOKBACK_DAYS + 1: return []
        momentum = self.history_df.pct_change(MOMENTUM_LOOKBACK_DAYS).iloc[-1]
        
        # 合并所有筛选条件
        final_candidates = list(volume_candidates & volatility_candidates)
        
        # 为候选者计算排名
        ranked_assets = []
        for pair in final_candidates:
            if pair in momentum.index and pd.notna(momentum[pair]):
                ranked_assets.append({
                    "pair": pair,
                    "momentum": momentum[pair],
                    "volatility": volatility.get(pair, 0)
                })

        # 按动量排名并选出Top N
        ranked_assets.sort(key=lambda x: x["momentum"], reverse=True)
        logging.info(f"筛选和排名完成，选出 Top {len(ranked_assets)} 候选资产。")
        return ranked_assets[:TOP_N]

    def check_entry_conditions(self, pair):
        """检查单个资产是否满足所有入场时机条件"""
        if self.history_df is None or pair not in self.history_df.columns: return False
        
        # 计算指标
        sma7 = ta.sma(self.history_df[pair], length=ENTRY_SMA_WINDOW).iloc[-1]
        rsi7 = ta.rsi(self.history_df[pair], length=RSI_WINDOW).iloc[-1]
        last_price = self.history_df[pair].iloc[-1]
        
        price_above_sma = last_price > sma7
        rsi_ok = RSI_ENTRY_MIN < rsi7 < RSI_ENTRY_MAX
        
        logging.debug(f"入场检查 ({pair}): Price={last_price:.2f}, SMA7={sma7:.2f} (>{'T' if price_above_sma else 'F'}), "
                      f"RSI7={rsi7:.2f} ({RSI_ENTRY_MIN}<RSI<{RSI_ENTRY_MAX} -> {'T' if rsi_ok else 'F'})")
        
        return price_above_sma and rsi_ok

    def check_exit_conditions(self, pair):
        """检查单个资产是否满足出场时机条件"""
        if self.history_df is None or pair not in self.history_df.columns: return False

        sma10 = ta.sma(self.history_df[pair], length=EXIT_SMA_WINDOW).iloc[-1]
        last_price = self.history_df[pair].iloc[-1]

        price_below_sma = last_price < sma10
        if price_below_sma:
            logging.info(f"出场信号 ({pair}): 价格 {last_price:.2f} 已跌破10日均线 {sma10:.2f}。")
            return True
        return False

# ---
# 主策略逻辑 (全面重构)
# ---
def run_strategy():
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

    # 3. 检查入场条件，确定最终目标
    final_targets = []
    for asset in top_candidates:
        if analytics.check_entry_conditions(asset['pair']):
            final_targets.append(asset)
    
    if not final_targets:
        logging.info("市场状况判断：所有Top候选资产均不满足入场条件，认定市场为下降趋势，暂停交易。")
        # 此处可以加入清仓逻辑
        return
        
    logging.info(f"最终投资目标 ({len(final_targets)}个): {[t['pair'] for t in final_targets]}")

    # 4. 计算投资组合价值
    balance_data = client.get_balance()
    if not balance_data or not balance_data.get("Success"): return
    wallet = balance_data.get("Wallet", {})
    total_portfolio_value = wallet.get("USD", {}).get("Free", 0)
    current_holdings = {}
    for coin, data in wallet.items():
        if coin == "USD": continue
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
            # 限制单个仓位不超过25%
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
            target_value = 0 # 强制清仓

        trade_value = target_value - current_value
        
        if abs(trade_value) < MIN_ORDER_VALUE:
            continue

        price = ticker_map.get(pair, {}).get("LastPrice")
        if not price or price <= 0: continue

        quantity = abs(trade_value) / price
        if trade_value < 0:
            logging.info(f"再平衡 [卖出]: {pair}, 卖出价值 ${-trade_value:.2f} (数量: {quantity:.6f})")
            client.place_order(coin, "SELL", current_holdings[pair]['quantity']) # 卖出全部
        else:
            logging.info(f"再平衡 [买入]: {pair}, 买入价值 ${trade_value:.2f} (数量: {quantity:.6f})")
            client.place_order(coin, "BUY", quantity)

    logging.info("="*10 + " 高级策略执行任务完成 " + "="*10)

# ---
# 启动与调度 (保持不变)
# ---
if __name__ == "__main__":
    logging.info("高级交易机器人启动...")
    try:
        collect_daily_price_data()
        run_strategy()
    except Exception as e:
        logging.critical(f"首次运行时发生严重错误: {e}", exc_info=True)

    scheduler = BlockingScheduler(timezone="UTC")
    scheduler.add_job(collect_daily_price_data, 'cron', hour=23, minute=59)
    scheduler.add_job(run_strategy, 'cron', hour=0, minute=15)
    
    logging.info("调度器已配置完毕。机器人进入持续运行模式...")
    print("="*60)
    print("高级交易机器人正在运行。按 Ctrl+C 停止。")
    print(f"日志将记录在 {LOG_FILE} 文件中。")
    print("="*60)
    
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logging.info("交易机器人已手动停止。")