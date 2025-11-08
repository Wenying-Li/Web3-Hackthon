#!/usr/bin/env python
# -*- coding: utf-8 -*-

import requests
import hashlib
import hmac
import time
import logging
from datetime import datetime
from apscheduler.schedulers.blocking import BlockingScheduler

# --- 配置 (來自 python_demo.py 和 momentum.py) ---
# 请替换为您的真实 Key
API_KEY = "MY_API_KEY"
SECRET = "MY_SECRET"
BASE_URL = "https://mock-api.roostoo.com"

# --- 策略参数 (基于PDF和API限制的降级版) ---
CASH_BUFFER_RATIO = 0.1  # 10% 现金缓冲
TOP_N = 5                # 买入 Top 5
TARGET_POSITION_RATIO = (1 - CASH_BUFFER_RATIO) / TOP_N  # 18%
MAX_POSITION_RATIO = 0.25   # 最大仓位
MIN_ORDER_VALUE = 300     # 最小订单金额

MARKET_BULL_SYMBOL = "BTC/USD" #

# 固定的 SL/TP (因无法获取K线，只能使用PDF Step 8 和 momentum.py 的固定值)
STOP_LOSS_RATIO = 0.95  # 5% 止损
TAKE_PROFIT_RATIO = 1.33 # 33% 止盈

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ---
# API 客户端 (基于官方 python_demo.py 和 README.md)
# ---
class RoostooV3Client:
    def __init__(self, api_key, secret_key):
        self.api_key = api_key
        self.secret = secret_key
        self.base_url = BASE_URL
        logging.info("Roostoo V3 客户端初始化")

    def _get_timestamp_ms(self):
        """返回 13 位毫秒时间戳字符串"""
        return str(int(time.time() * 1000))

    def generate_signature(self, params):
        """
        生成 API 请求签名
       
        """
        query_string = '&'.join(["{}={}".format(k, params[k])
                                 for k in sorted(params.keys())])
        us = self.secret.encode('utf-8')
        m = hmac.new(us, query_string.encode('utf-8'), hashlib.sha256)
        return m.hexdigest()

    def _signed_get_request(self, endpoint, params={}):
        """
        执行签名的 GET 请求 (例如 /v3/balance)
       
        """
        params["timestamp"] = self._get_timestamp_ms()
        signature = self.generate_signature(params)
        
        headers = {
            "RST-API-KEY": self.api_key,
            "MSG-SIGNATURE": signature
        }
        
        url = f"{self.base_url}{endpoint}"
        try:
            response = requests.get(url, params=params, headers=headers)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logging.error(f"API GET 请求失败 ({url}): {e}")
            return None

    def _signed_post_request(self, endpoint, data={}):
        """
        执行签名的 POST 请求 (例如 /v3/place_order)
       
        """
        data["timestamp"] = self._get_timestamp_ms()
        signature = self.generate_signature(data)
        
        headers = {
            "RST-API-KEY": self.api_key,
            "MSG-SIGNATURE": signature,
            "Content-Type": "application/x-www-form-urlencoded" #
        }
        
        url = f"{self.base_url}{endpoint}"
        try:
            response = requests.post(url, data=data, headers=headers)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logging.error(f"API POST 请求失败 ({url}): {e}")
            return None

    def get_ticker(self, pair=None):
        """获取市场行情 (Ticker)"""
        params = {"timestamp": self._get_timestamp_ms()}
        if pair:
            params["pair"] = pair
        url = f"{self.base_url}/v3/ticker"
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logging.error(f"API GET (Ticker) 请求失败: {e}")
            return None

    def get_balance(self):
        """获取账户余额"""
        # --- (新增日志) ---
        logging.info("--- 正在调用 /v3/balance API ---")
        response = self._signed_get_request("/v3/balance")
        
        # 打印 API 返回的完整响应，以便调试
        if response:
            # 使用 str(response) 确保字典被完整打印
            logging.info(f"get_balance API 完整响应: {str(response)}")
        else:
            logging.warning("get_balance API 调用失败 (返回 None)")
        # --- (日志结束) ---
        
        return response

    def place_order(self, coin, side, quantity, price=None):
        """
        下单 (市价或限价)
       
        """
        data = {
            "pair": f"{coin}/USD",
            "side": side.upper(),
            "quantity": str(quantity),
        }
        if not price:
            data['type'] = "MARKET"
        else:
            data['type'] = "LIMIT"
            data['price'] = str(price)
            
        logging.info(f"下单: {data}")
        return self._signed_post_request("/v3/place_order", data=data)

    def query_order(self, pair=None, pending_only="FALSE"):
        """查询订单"""
        data = {"pending_only": pending_only}
        if pair:
            data["pair"] = pair
        return self._signed_post_request("/v3/query_order", data=data)

# ---
# 策略逻辑 (基于 API 限制)
# ---

def check_stop_loss_take_profit(client, ticker_map):
    """
    检查并执行止盈止损 (基于 momentum.py 的逻辑)
   
    """
    logging.info("--- 检查止盈止损 ---")
    
    positions = client.query_order(pending_only="FALSE")
    
    if not positions:
        logging.error("API 请求失败，无法检查 SL/TP")
        return

    if not positions.get("Success", False):
        if positions.get("ErrMsg") == "no order matched": #
            logging.info("没有找到已成交的订单，跳过 SL/TP 检查")
        else:
            logging.warning(f"无法获取订单列表以检查 SL/TP: {positions.get('ErrMsg', '未知错误')}")
        return

    filled_buys = [
        o for o in positions.get("OrderMatched", []) 
        if o["Status"] in ("FILLED", "PARTIAL") and o["Side"] == "BUY"
    ]
    
    if not filled_buys:
        logging.info("没有已成交的买单可供检查 SL/TP")
        return

    for order in filled_buys:
        pair = order["Pair"]
        coin = pair.split("/")[0]
        
        if pair not in ticker_map:
            continue
            
        entry_price = order["FilledAverPrice"]
        if entry_price == 0: continue

        current_price = ticker_map[pair].get("LastPrice", 0)
        if current_price == 0: continue
            
        quantity_to_sell = order["FilledQuantity"]
        
        # 止盈
        if current_price >= entry_price * TAKE_PROFIT_RATIO:
            logging.info(f"止盈卖出 {pair}: 数量 {quantity_to_sell} (入场: {entry_price}, 当前: {current_price})")
            client.place_order(coin, "SELL", quantity_to_sell)
            
        # 止损
        elif current_price <= entry_price * STOP_LOSS_RATIO:
            logging.info(f"止损卖出 {pair}: 数量 {quantity_to_sell} (入场: {entry_price}, 当前: {current_price})")
            client.place_order(coin, "SELL", quantity_to_sell)

def run_strategy():
    """执行简化的、基于Ticker的策略逻辑"""
    logging.info("--- (开始) 每日策略执行 (V3 Ticker Only 修正版) ---")
    client = RoostooV3Client(API_KEY, SECRET)

    ticker_data = client.get_ticker()
    if not ticker_data or not ticker_data.get("Success", False):
        logging.error("无法获取 Ticker 行情, 策略中止")
        return
    ticker_map = ticker_data.get("Data", {})
    
    check_stop_loss_take_profit(client, ticker_map)

    if MARKET_BULL_SYMBOL not in ticker_map:
        logging.warning(f"无法找到基准代币 {MARKET_BULL_SYMBOL} 数据, 假定为熊市, 中止")
        return
    
    market_change = ticker_map[MARKET_BULL_SYMBOL].get("Change", 0)
    if market_change <= 0:
        logging.info(f"市场非牛市 ({MARKET_BULL_SYMBOL} 24h Change: {market_change}), 不执行再平衡")
        return
    
    logging.info(f"市场为牛市 ({MARKET_BULL_SYMBOL} 上涨)，继续执行再平衡")

    tokens = []
    for pair, data in ticker_map.items():
        if pair == MARKET_BULL_SYMBOL or not pair.endswith("/USD"):
            continue
        tokens.append({
            "pair": pair,
            "change": data.get("Change", 0), #
            "price": data.get("LastPrice", 0)
        })

    tokens.sort(key=lambda x: x["change"], reverse=True)
    
    top_5_targets = tokens[:TOP_N]
    target_pairs = set(t["pair"] for t in top_5_targets)
    
    if not target_pairs:
        logging.warning("未找到任何符合条件的代币")
        return
        
    logging.info(f"选定的 Top 5 动量目标: {list(target_pairs)}")

    balance_data = client.get_balance()
    if not balance_data or not balance_data.get("Success", False):
        logging.error(f"无法获取账户余额, 策略中止: {balance_data.get('ErrMsg', '')}")
        return

    # --- (修正 #3 - V2) ---
    # 优先检索 "Wallet" (根据 README.md)
    wallet = balance_data.get("Wallet", {})
    if not wallet:
        # 如果 "Wallet" 为空或不存在，则检索 "SpotWallet" (根据 真实日志)
        logging.info("未找到 'Wallet' 键, 尝试 'SpotWallet'...")
        wallet = balance_data.get("SpotWallet", {})
    # --- (修正结束) ---

    total_usd_balance = wallet.get("USD", {}).get("Free", 0) + wallet.get("USD", {}).get("Lock", 0)
    current_holdings = {}
    total_portfolio_value = total_usd_balance
    
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

    target_position_value = total_portfolio_value * TARGET_POSITION_RATIO
    target_position_value = min(target_position_value, total_portfolio_value * MAX_POSITION_RATIO) #

    for pair, holding in current_holdings.items():
        coin = pair.split('/')[0]
        
        if pair not in target_pairs:
            logging.info(f"再平衡退场 [掉出Top 5]: 卖出 {pair} (数量: {holding['quantity']})")
            client.place_order(coin, "SELL", holding['quantity'], price=None)

    for target in top_5_targets:
        pair = target["pair"]
        coin = pair.split('/')[0]
        price = target["price"]
        
        if price == 0: continue

        current_value = current_holdings.get(pair, {}).get("value", 0)
        trade_value = target_position_value - current_value
        
        if abs(trade_value) < MIN_ORDER_VALUE: #
            logging.info(f"跳过 {pair}：交易金额 ${abs(trade_value):.2f} 小于 ${MIN_ORDER_VALUE}")
            continue

        quantity = abs(trade_value) / price
        
        if trade_value > 0:
            logging.info(f"再平衡买入 {pair}: 数量 {quantity:.4f} @ ${price:.2f}")
            client.place_order(coin, "BUY", quantity, price=price) #
        
    logging.info("--- (结束) 每日策略执行 ---")


if __name__ == "__main__":
    logging.info("启动交易机器人 (V3 Ticker Only 官方 API 修正版)...")
    
    run_strategy()
    
    scheduler = BlockingScheduler(timezone="UTC")
    scheduler.add_job(run_strategy, 'cron', hour=0, minute=5)
    logging.info("调度器已设置。等待下一個 00:05 UTC 运行...")
    
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logging.info("机器人已停止")
