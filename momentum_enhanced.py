import requests
import hashlib
import hmac
import time
import logging
import pandas as pd
from apscheduler.schedulers.blocking import BlockingScheduler

# --- 配置 ---
API_KEY = "MY_API_KEY"  # 请替换为您的真实 Key
SECRET = "MY_SECRET"
BASE_URL = "https://mock-api.roostoo.com"

# --- 策略参数 (融合了 ethdubai-hackathon.py 的思想) ---
CASH_ALLOCATION = 0.5  # 50% 资金用于持仓 (另外50%为现金缓冲)
TOP_N = 3                # 买入 Top 3
MAX_POSITION_RATIO = 0.25   # 单一资产最大仓位不超过总资产25%
MIN_ORDER_VALUE = 300     # 最小订单金额

# 牛市判断
MARKET_BENCHMARK_SYMBOL = "BTC/USD"
SMA_WINDOW = 15           # 15日SMA

# 动量计算
MOMENTUM_LOOKBACK_DAYS = 4 # 4日价格动量
MIN_MOMENTUM_THRESHOLD = 0.03 # 动量阈值 > 3%

# 止盈止损
STOP_LOSS_RATIO = 0.97  # 3% 止损
TAKE_PROFIT_RATIO = 1.33 # 33% 止盈

# 数据文件
HISTORY_FILE = "price_history.csv"

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- API 客户端 (与原版 momentum.py 相同，此处省略以节约篇幅) ---
class RoostooV3Client:
    # ... (将原版 momentum.py 中的 RoostooV3Client 完整粘贴到这里) ...
    # 唯一的小改动是在 __init__ 中
    def __init__(self, api_key, secret_key):
        self.api_key = api_key
        self.secret = secret_key
        self.base_url = BASE_URL
        logging.info("Roostoo V3 客户端初始化")

    def _get_timestamp_ms(self):
        return str(int(time.time() * 1000))

    def generate_signature(self, params):
        query_string = '&'.join([f"{k}={params[k]}" for k in sorted(params.keys())])
        us = self.secret.encode('utf-8')
        m = hmac.new(us, query_string.encode('utf-8'), hashlib.sha256)
        return m.hexdigest()

    def _signed_request(self, method, endpoint, params=None, data=None):
        if params is None: params = {}
        if data is None: data = {}
        
        req_data = params if method == 'GET' else data
        req_data["timestamp"] = self._get_timestamp_ms()
        signature = self.generate_signature(req_data)
        
        headers = {"RST-API-KEY": self.api_key, "MSG-SIGNATURE": signature}
        url = f"{self.base_url}{endpoint}"
        
        try:
            if method == 'GET':
                response = requests.get(url, params=req_data, headers=headers)
            else:
                headers['Content-Type'] = 'application/x-www-form-urlencoded'
                response = requests.post(url, data=req_data, headers=headers)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logging.error(f"API {method} 请求失败 ({url}): {e}")
            if e.response:
                logging.error(f"响应内容: {e.response.text}")
            return None
            
    def get_ticker(self, pair=None):
        params = {"timestamp": self._get_timestamp_ms()}
        if pair: params["pair"] = pair
        url = f"{self.base_url}/v3/ticker"
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logging.error(f"API GET (Ticker) 请求失败: {e}")
            return None

    def get_balance(self):
        logging.info("--- 正在调用 /v3/balance API ---")
        return self._signed_request("GET", "/v3/balance")

    def place_order(self, coin, side, quantity, price=None):
        data = {"pair": f"{coin}/USD", "side": side.upper(), "quantity": str(quantity)}
        if price:
            data['type'] = "LIMIT"
            data['price'] = str(price)
        else:
            data['type'] = "MARKET"
        logging.info(f"下单: {data}")
        return self._signed_request("POST", "/v3/place_order", data=data)

    def query_order(self, pending_only="FALSE", pair=None):
        data = {"pending_only": pending_only}
        if pair: data["pair"] = pair
        return self._signed_request("POST", "/v3/query_order", data=data)

# --- 新增：策略分析模块 ---
class StrategyAnalytics:
    def __init__(self, history_file):
        try:
            self.history_df = pd.read_csv(history_file, index_col='date', parse_dates=True)
            logging.info(f"成功加载历史数据，共 {len(self.history_df)} 条记录。")
        except FileNotFoundError:
            self.history_df = None
            logging.error(f"历史数据文件 '{history_file}' 未找到！策略功能将受限。")

    def is_bull_market(self):
        """使用 SMA 判断是否为牛市"""
        if self.history_df is None or MARKET_BENCHMARK_SYMBOL not in self.history_df.columns:
            logging.warning("无历史数据或基准代币数据，无法判断牛熊市。默认为熊市。")
            return False
        
        if len(self.history_df) < SMA_WINDOW:
            logging.warning(f"历史数据不足 {SMA_WINDOW} 天，无法计算SMA。默认为熊市。")
            return False

        # 计算 SMA
        sma = self.history_df[MARKET_BENCHMARK_SYMBOL].rolling(window=SMA_WINDOW).mean()
        
        # 获取最新价格和最新的SMA值
        last_price = self.history_df[MARKET_BENCHMARK_SYMBOL].iloc[-1]
        last_sma = sma.iloc[-1]

        logging.info(f"牛市判断: {MARKET_BENCHMARK_SYMBOL} 最新价: {last_price:.2f}, {SMA_WINDOW}日SMA: {last_sma:.2f}")
        return last_price > last_sma

    def get_momentum_signals(self, ticker_map):
        """计算所有代币的动量信号"""
        if self.history_df is None or len(self.history_df) < MOMENTUM_LOOKBACK_DAYS:
            logging.warning(f"历史数据不足 {MOMENTUM_LOOKBACK_DAYS} 天，无法计算动量。")
            return []
        
        signals = []
        lookback_date = self.history_df.index[-MOMENTUM_LOOKBACK_DAYS]

        for pair, data in ticker_map.items():
            if not pair.endswith("/USD") or pair not in self.history_df.columns:
                continue

            # 获取N天前的价格
            try:
                open_price = self.history_df.loc[lookback_date, pair]
                close_price = data.get("LastPrice", 0)

                if open_price > 0 and close_price > 0:
                    momentum = (close_price - open_price) / open_price
                    if momentum > MIN_MOMENTUM_THRESHOLD:
                        signals.append({
                            "pair": pair,
                            "momentum": momentum,
                            "price": close_price
                        })
            except (KeyError, IndexError):
                continue
        
        # 按动量从高到低排序
        signals.sort(key=lambda x: x["momentum"], reverse=True)
        return signals

# --- 主策略逻辑 (重构) ---
def run_strategy():
    logging.info("--- (开始) 每日策略执行 (增强版) ---")
    client = RoostooV3Client(API_KEY, SECRET)
    analytics = StrategyAnalytics(HISTORY_FILE)

    # 1. 获取实时行情
    ticker_data = client.get_ticker()
    if not ticker_data or not ticker_data.get("Success"):
        logging.error("无法获取 Ticker 行情, 策略中止。")
        return
    ticker_map = ticker_data.get("Data", {})

    # 2. 检查止盈止损 (逻辑不变，可从原版 momentum.py 复制)
    # check_stop_loss_take_profit(client, ticker_map)

    # 3. 判断牛熊市
    if not analytics.is_bull_market():
        logging.info("市场为熊市，不执行再平衡。考虑清仓所有头寸（可选）。")
        # 可在此添加清仓逻辑
        return
    
    logging.info("市场为牛市，继续执行再平衡。")

    # 4. 获取动量信号并选择Top N
    signals = analytics.get_momentum_signals(ticker_map)
    top_targets = signals[:TOP_N]
    target_pairs = {t["pair"] for t in top_targets}

    if not top_targets:
        logging.warning("未找到任何符合条件的动量目标。")
        return
    
    logging.info(f"选定的 Top {TOP_N} 动量目标: {[t['pair'] for t in top_targets]}")

    # 5. 获取账户余额和当前持仓
    balance_data = client.get_balance()
    if not balance_data or not balance_data.get("Success"):
        logging.error("无法获取账户余额, 策略中止。")
        return
    
    wallet = balance_data.get("Wallet", balance_data.get("SpotWallet", {}))
    total_usd_balance = wallet.get("USD", {}).get("Free", 0)
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

    # 6. 计算目标仓位并执行交易
    equity_to_invest = total_portfolio_value * CASH_ALLOCATION
    # 1/N 权重，但要考虑最大仓位限制
    target_value_per_position = min(
        equity_to_invest / TOP_N if TOP_N > 0 else 0,
        total_portfolio_value * MAX_POSITION_RATIO
    )

    # 卖出不再是目标的持仓
    for pair, holding in current_holdings.items():
        if pair not in target_pairs:
            coin = pair.split('/')[0]
            logging.info(f"再平衡退场 [掉出Top N]: 卖出 {pair} (数量: {holding['quantity']})")
            client.place_order(coin, "SELL", holding['quantity'])

    # 买入或调整目标持仓
    for target in top_targets:
        pair = target["pair"]
        coin = pair.split('/')[0]
        price = target["price"]
        if price <= 0: continue

        current_value = current_holdings.get(pair, {}).get("value", 0)
        trade_value = target_value_per_position - current_value
        
        if abs(trade_value) < MIN_ORDER_VALUE:
            logging.info(f"跳过 {pair}：交易金额 ${abs(trade_value):.2f} 小于 ${MIN_ORDER_VALUE}")
            continue

        if trade_value > 0:
            quantity_to_buy = trade_value / price
            logging.info(f"再平衡买入 {pair}: 数量 {quantity_to_buy:.6f}")
            client.place_order(coin, "BUY", quantity_to_buy)

    logging.info("--- (结束) 每日策略执行 ---")

if __name__ == "__main__":
    logging.info("启动交易机器人 (增强版)...")
    run_strategy()  # 启动时立即运行一次

    scheduler = BlockingScheduler(timezone="UTC")
    scheduler.add_job(run_strategy, 'cron', hour=0, minute=15) # 每天00:15 UTC运行
    logging.info("调度器已设置。等待下一个 00:15 UTC 运行...")
    
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logging.info("机器人已停止。")