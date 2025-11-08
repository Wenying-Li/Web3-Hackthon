# Web3-Hackthon
momentum.py实现较为简略，由于roostoo没有K线相关api，所以momentum.py是以24h changes来判断牛市

目前添加了history_collector.py用于粗略记录每日收盘价，手动实现历史数据的记录。trading_bot.py作为运行脚本。(暂时作废了，策略不太符合）

advanced_trading_bot.py应该比较符合我们策略，但是目前存在没有历史数据的问题。不太清楚回测怎么处理，以及刚开始交易的时候如何获取过去几天的数据
