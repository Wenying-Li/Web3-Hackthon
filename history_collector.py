import requests
import pandas as pd
from datetime import datetime
import time
import os
import logging

# --- 配置 ---
BASE_URL = "https://mock-api.roostoo.com"
HISTORY_FILE = "price_history.csv"  # 存储历史数据的文件
LOG_FILE = "collector.log"

# --- 日志配置 ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)

def get_ticker_all():
    """调用 Ticker API 获取所有交易对的行情"""
    url = f"{BASE_URL}/v3/ticker"
    params = {"timestamp": str(int(time.time() * 1000))}
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logging.error(f"无法获取 Ticker 行情: {e}")
        return None

def update_history():
    """获取当前价格并更新历史数据文件"""
    logging.info("--- 开始每日价格采集 ---")
    
    ticker_data = get_ticker_all()
    if not ticker_data or not ticker_data.get("Success"):
        logging.error("获取行情失败，无法更新历史数据。")
        return

    # 获取当前UTC日期作为时间戳
    today_utc = datetime.utcnow().strftime('%Y-%m-%d')
    
    # 准备新的价格数据
    price_updates = {"date": today_utc}
    for pair, data in ticker_data.get("Data", {}).items():
        if pair.endswith("/USD"):
            price_updates[pair] = data.get("LastPrice")
    
    # 读取旧的历史数据
    if os.path.exists(HISTORY_FILE):
        df = pd.read_csv(HISTORY_FILE)
    else:
        df = pd.DataFrame(columns=["date"])

    # 将新数据添加到DataFrame中
    # 先检查日期是否已存在，避免重复记录
    if today_utc in df['date'].values:
        logging.warning(f"日期 {today_utc} 的数据已存在，将进行覆盖。")
        # 定位并更新行
        df.loc[df['date'] == today_utc, list(price_updates.keys())] = list(price_updates.values())
    else:
        # 添加新行
        new_row = pd.DataFrame([price_updates])
        df = pd.concat([df, new_row], ignore_index=True)

    # 保存更新后的DataFrame到CSV
    df.to_csv(HISTORY_FILE, index=False)
    logging.info(f"历史价格文件 '{HISTORY_FILE}' 已成功更新，当前共 {len(df)} 条记录。")


if __name__ == "__main__":
    update_history()