# Web3-Hackthon
详细对比分析 (主要差异)
下面是一个表格，详细列出了两者在关键方面的巨大差异：

特性	ethdubai-hackathon.py (专业级)	momentum.py (简化/实战级)	分析与解读
运行环境与框架	运行在 trade-executor 框架内，这是一个专为量化策略回测和实盘设计的复杂系统。	一个独立的 Python 脚本，使用 apscheduler 进行任务调度，直接通过 requests 库与 API 交互。	ethdubai 版本是策略与执行分离的典范，策略逻辑本身不关心如何下单。momentum.py 则是策略与执行紧密耦合，更像是个人开发者或小型团队的实战脚本。
数据源	历史K线数据 (Candles)。策略使用 TimeBucket.d1 (日K线) 来分析。	实时行情API (Ticker)。策略依赖 /v3/ticker 返回的24小时价格变动 (Change)。	这是最核心的区别。K线数据提供了开盘、收盘、最高、最低价，可以计算复杂的指标（如SMA）。而Ticker的24h变动率是一个非常粗糙的动量信号，极易受短期噪音影响。
市场趋势判断	15日简单移动平均线 (SMA)。WMATIC/USDC 的当前价格必须高于其15日SMA才认为是牛市。	单一资产24小时涨跌幅。BTC/USD 的24小时 Change 必须大于0。	ethdubai 的方法更稳健，平滑了短期波动，能更好地反映中期趋势。momentum.py 的方法非常简单直接，但容易因比特币一天的回调而错失整个市场的机会。
动量信号计算	4日K线动量: (close - open) / open。计算特定时间窗口内的真实价格增长。	24小时价格变动百分比 (Change)。直接使用API返回的现成数据。	ethdubai 的动量信号更具可定制性和统计意义。momentum.py 的信号则完全依赖于交易所API提供的计算结果，缺乏灵活性。
交易执行与交易所	去中心化交易所 (DEX) - SushiSwap on Polygon。通过智能合约进行交易。	中心化交易所 (CEX) - 一个模拟的API (mock-api.roostoo.com)。通过签名的REST API请求下单。	两者面向完全不同的交易场所。DEX交易涉及链上操作、Gas费和滑点，而CEX交易则是通过API在中心化服务器的订单簿上进行。
风险管理 (SL/TP)	与头寸绑定的触发器。在创建头寸时就定义好止盈/止损价格，由框架监控执行。	轮询检查。在每次策略运行时，通过 check_stop_loss_take_profit 函数遍历所有持仓，并与当前市价比较来手动触发。	ethdubai 的方式更高效、及时，因为它可能是事件驱动的。momentum.py 的轮询方式会有延迟，如果价格波动剧烈，可能会在两次检查之间错过最佳止损点。
代码结构与抽象	高度抽象。策略逻辑（decide_trades）和数据准备（create_trading_universe）完全分离。使用了大量辅助类如 PositionManager, AlphaModel。	过程化与实用主义。所有逻辑几乎都在一个文件中，RoostooV3Client 类封装了API调用，主逻辑 run_strategy 线性执行所有步骤。	ethdubai 的代码展示了软件工程的最佳实践，易于测试、扩展和维护。momentum.py 则更注重“快速实现”，把所有东西放在一起，对于小型项目来说足够有效。
资金与仓位管理	更精细。分配总资产的50% (value_allocated_to_positions)，并使用 1/N 加权。	相对简化。保留10%现金 (CASH_BUFFER_RATIO)，剩余资金按 1/N 分配，同时有25%的单资产上限。	两者思想一致，但 momentum.py 的参数更直接。例如，它有一个明确的 MAX_POSITION_RATIO 来防止单个代币仓位过重。
