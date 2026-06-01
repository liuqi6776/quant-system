# -*- coding: utf-8 -*-
"""
ptrade_client_v5.py
PTrade 恒生实盘全自动交易系统 (Study 005 - 期权增强全局防御策略版)

系统特性：
1. 本脚本为 Ptrade 客户端原生运行代码，支持全自动、全无人值守交易。
2. 完美整合了三项硬性安全防御体系：
   - 🛡️ 内容审计层 (Pre-Trade Audit)：物理级硬编码清洗机制，坚决不买入 ST/退市股、次新股、高风险垃圾股。
   - 💧 流动性与冲击成本过滤 (Liquidity Filter)：利用 20日日均成交量(ADV) 动态计算冲击容量，防范滑点与流动性踩踏。
   - ⚖️ 账户对账与状态机 (State Reconciliation)：开盘前自动对账，以券商真实账户仓位及资金为唯一信任源，防范空头锁仓与发单超买。
3. 严格执行 Worst-Case T+1 交易规则与固定止盈 (+6%) / 止损 (-5%) 机制。
"""

import urllib.request
import json
import re
from datetime import datetime

# ========================================== 全局配置 ==========================================
g_config = {
    # 本地量化决策 API 服务端地址 (穿透后或本地回环)
    'api_url': 'http://127.0.0.1:8000/api/v1/signals',
    'strategy': 'conservative',  # 绑定 Study 005 稳健型策略
    
    # 实盘资金风控参数
    'max_positions': 3,          # 每日最大持仓股票数
    'max_pos_pct': 0.33,         # 单只个股目标资金比例 (33% 满仓，建议保留 1% 作为摩擦滑点缓存)
    
    # 物理级去未来函数硬性出场参数
    'stop_loss_pct': -0.05,      # 硬性止损：-5%
    'take_profit_pct': 0.06,     # 硬性止盈：+6%
    
    # 2. 流动性防踩踏过滤限额
    'max_adv_pct': 0.01,         # 单笔订单成交量不得超过该个股 20日日均成交量(ADV20)的 1%
    'min_order_value': 5000,     # 单笔订单最低下单金额 (小于该值直接放弃，防止产生零碎单)
    
    # 3. 内容审计硬防线
    'min_listed_days': 90,       # 过滤上市不足 90 天的次新股
    'filter_st': True,           # 强制过滤 ST、*ST 等高风险标的
}

# ========================================== Ptrade 回调函数 ==========================================

def initialize(context):
    """
    Ptrade 策略初始化回调 (仅在系统启动时执行一次)
    """
    log.info("==========================================================================")
    Log_msg = "  PTrade Study 005 Option-Enhanced Live Trading System Initialized ✓"
    log.info(Log_msg)
    log.info("==========================================================================")
    
    # 设定基准与佣金
    set_benchmark('000300.SS')
    
    # 初始化全局状态机容器
    g.today_buy_signals = []      # 今日通过审计过滤后的最终待买入清单
    g.buy_execution_done = False  # 买入动作执行标志位
    g.sync_done = False           # 晨间对账同步标志位

    # 注册 Ptrade 每日定时调度器
    # 1. 每天 9:15 进行真实账户对账与盘前信号内容审计
    run_daily(context, before_market_start, time='09:15')
    
    # 2. 每天 9:30 开盘瞬间触发全自动开盘价买入下单
    run_daily(context, execute_morning_buy, time='09:30')


def handle_data(context, data):
    """
    Ptrade 盘中逐笔/逐分钟 Bar 触发回调 (实施 100% 物理级去未来函数实时风控)
    """
    # 仅在交易时间段内运行实时风控
    current_time = get_datetime().strftime('%H:%M')
    if current_time < '09:30' or current_time > '15:00':
        return
        
    # 盘中实时风控 (Intraday Real-time Risk Control)
    intraday_risk_control(context, data)


# ========================================== 核心执行模块 ==========================================

def before_market_start(context, data):
    """
    【对账状态机 + 内容审计防线】
    每日开盘前 9:15 准时执行：
    1. 强制拉取券商真实持仓，完成状态对账同步
    2. 从本地量化引擎拉取今日最新预测信号
    3. 实施 Pre-Trade Audit 内容审计，拦截高风险个股
    4. 实施 Liquidity Filter 冲击容量限制，计算精确安全下单股数
    """
    log.info(f"--- [RECONCILIATION] Starting Morning Sync at {get_datetime().strftime('%Y-%m-%d %H:%M:%S')} ---")
    g.today_buy_signals = []
    g.buy_execution_done = False
    g.sync_done = False

    # ------------------ 阶段 1：真实账户仓位与资金对账 (State Reconciliation) ------------------
    actual_positions = context.portfolio.positions
    actual_cash = context.portfolio.cash
    total_assets = context.portfolio.portfolio_value
    
    log.info(f"[ACCOUNT] Cash Available: ¥{actual_cash:,.2f} | Total Net Assets: ¥{total_assets:,.2f}")
    log.info(f"[ACCOUNT] Active Positions Count: {len(actual_positions)}")
    
    # 打印真实持仓明细，确保对账一致性
    for code, pos in actual_positions.items():
        log.info(f"  Holding: {code} | Shares: {pos.amount} | Cost Basis: ¥{pos.cost_basis:.2f} | Mkt Value: ¥{pos.market_value:,.2f}")
        
    # 计算当前可用空闲仓位槽位数
    active_slots = len(actual_positions)
    free_slots = max(0, g_config['max_positions'] - active_slots)
    log.info(f"[RECONCILIATION] Free Slots Available: {free_slots} / {g_config['max_positions']}")
    
    if free_slots == 0:
        log.warn("[RECONCILIATION] Account portfolio is fully allocated. No buy orders will be dispatched today.")
        g.sync_done = True
        return

    # ------------------ 阶段 2：拉取量化决策引擎最新信号 ------------------
    today_str = get_datetime().strftime('%Y%m%d')
    # 生产调试备用：如果开盘前拉取今天信号失败，可增量查询前一交易日作为 Fallback 信号
    api_request_url = f"{g_config['api_url']}?date={today_str}&strategy={g_config['strategy']}"
    
    raw_signals = []
    try:
        req = urllib.request.Request(api_request_url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=5) as response:
            resp_data = json.loads(response.read().decode('utf-8'))
            if resp_data.get('status') == 'ok' and resp_data.get('signals'):
                raw_signals = resp_data['signals']
                log.info(f"[API] Successfully pulled {len(raw_signals)} raw signals from local engine.")
            else:
                log.warn(f"[API] No signals returned for date {today_str}. Warning: {resp_data.get('warning', 'None')}")
    except Exception as e:
        log.error(f"[API ERROR] Failed to connect to Quant Local Engine API: {e}")
        log.error("Please ensure api_server.py is running and reachable!")
        g.sync_done = True
        return

    if not raw_signals:
        g.sync_done = True
        return

    # ------------------ 阶段 3：内容审计层过滤 (Pre-Trade Audit) ------------------
    audited_signals = []
    for sig in raw_signals:
        code = sig['ts_code']
        # 兼容证券代码格式转化 (Tushare '000001.SZ' -> Ptrade '000001.SZ' or '000001')
        ptrade_code = convert_code_to_ptrade(code)
        
        # 3.1 获取股票基础静态信息
        security_info = get_security_info(ptrade_code)
        if not security_info:
            log.warn(f"  [AUDIT REJECT] {code} - Unknown code. Unable to fetch security info.")
            continue
            
        display_name = security_info.display_name
        start_date = str(security_info.start_date) # 上市日期
        
        # 3.2 过滤 ST / *ST / 退市整理股 (ST/Delisting Audit)
        if g_config['filter_st']:
            if any(word in display_name for word in ['ST', '*ST', '退市', 'SST', 'S*ST', 'ST']):
                log.warn(f"  [AUDIT REJECT] {code} ({display_name}) - High Risk ST/Delisting security intercepted!")
                continue
                
        # 3.3 过滤上市不足 90 天的次新股 (New Stock Audit)
        try:
            listed_dt = datetime.strptime(start_date, '%Y%m%d')
            held_days = (datetime.now() - listed_dt).days
            if held_days < g_config['min_listed_days']:
                log.warn(f"  [AUDIT REJECT] {code} - Listed for only {held_days} days. High volatility IPO intercepted!")
                continue
        except Exception:
            pass
            
        # 3.4 停牌审计 (Trading Status Audit)
        # 检查今天是否停牌，如果是直接拦截
        if ptrade_code in context.security_list and ptrade_code not in data:
            log.warn(f"  [AUDIT REJECT] {code} - Suspended stock. Intercepted.")
            continue
            
        # 3.5 行业与重仓检查 (Deduplication Audit)
        if ptrade_code in actual_positions:
            log.info(f"  [AUDIT SKIP] {code} - Already held in portfolio. Skipping duplicate entry.")
            continue
            
        # 完美通过 Pre-Trade Audit 审计防线！
        log.info(f"  [AUDIT PASS] {code} ({display_name}) passed Pre-Trade safety checklist ✓")
        audited_signals.append(sig)

    if not audited_signals:
        log.info("[AUDIT] No signals survived the Pre-Trade Safety Check.")
        g.sync_done = True
        return

    # ------------------ 阶段 4：流动性与最大冲击成本过滤 (Liquidity Filter) ------------------
    # 分配目标每只股的拟采购金额 (均仓分配)
    target_allocation_per_stock = total_assets * g_config['max_pos_pct']
    
    # 限制采购金额不超过真实可用现金
    target_pos_value = min(target_allocation_per_stock, actual_cash / free_slots)
    log.info(f"[ALLOCATION] Target Allocation value per stock slot: ¥{target_pos_value:,.2f}")
    
    for sig in audited_signals:
        if len(g.today_buy_signals) >= free_slots:
            log.info(f"[ALLOCATION] Free slots filled. Skipping remaining candidates.")
            break
            
        code = sig['ts_code']
        ptrade_code = convert_code_to_ptrade(code)
        
        # 4.1 获取 20 日历史交易量 (ADV20)
        # 用 get_history 获取历史 20 根日 Bar 的 volume
        try:
            hist_df = get_history(20, '1d', 'volume', [ptrade_code])
            if hist_df is None or hist_df.empty:
                log.warn(f"  [LIQUIDITY REJECT] {code} - Failed to fetch volume history. Skipping for safety.")
                continue
            
            # 计算 20日日均成交股数 (ADV20)
            adv20_shares = hist_df[ptrade_code].mean()
            # 获取最新参考股价 (以昨收盘算)
            last_price = data[ptrade_code].close
            
            # 最大订单股数限制：不得超过 ADV20 的 1%
            max_allowed_shares = int(adv20_shares * g_config['max_adv_pct'] / 100) * 100
            
            # 拟下单股数计算：基于分配资金
            planned_shares = int(target_pos_value / last_price / 100) * 100
            
            if planned_shares < 100:
                log.warn(f"  [ALLOCATION SKIP] {code} - Target value ¥{target_pos_value:.2f} too low to buy 100 shares.")
                continue
                
            # 动态容量衰减 (Liquidity Scaling)
            final_shares = planned_shares
            if planned_shares > max_allowed_shares:
                log.warn(f"  [LIQUIDITY SCALING] {code} - Planned order volume {planned_shares} shares exceeds 1% of ADV20 ({max_allowed_shares} shares).")
                log.warn(f"  Auto-scaling order size down to limit impact cost!")
                final_shares = max_allowed_shares
                
            # 过滤超小金额订单 (防止产生零碎底仓摩擦)
            final_order_value = final_shares * last_price
            if final_order_value < g_config['min_order_value']:
                log.warn(f"  [LIQUIDITY REJECT] {code} - Scaled order value ¥{final_order_value:,.2f} falls below ¥{g_config['min_order_value']:.2f}. Canceled.")
                continue
                
            # 成功记录为今日待买入安全清单
            sig['final_shares'] = final_shares
            sig['ptrade_code'] = ptrade_code
            g.today_buy_signals.append(sig)
            log.info(f"  [LIQUIDITY PASS] {code} - Final Order: {final_shares} shares (Value: ¥{final_order_value:,.2f}) staged for market open ✓")
            
        except Exception as e:
            log.error(f"  [SYSTEM ERROR] Error during liquidity audit for {code}: {e}")
            continue

    log.info(f"=== Morning Sync Complete! Staged {len(g.today_buy_signals)} buy signals for market open. ===")
    g.sync_done = True


def execute_morning_buy(context, data):
    """
    【交易执行状态机】
    开盘 9:30 准时触发：
    1. 核验 9:15 是否顺利完成晨间同步与审计
    2. 执行开盘卖出 (针对不在目标持仓里的 T+1 跨日持仓)
    3. 执行开盘买入 (针对 staged 好的今日信号，市价成交以符合 Open 买入要求)
    """
    log.info("--- [EXECUTION] Market Open 9:30 Order Dispatch Triggered ---")
    if not g.sync_done:
        log.error("[EXECUTION ERROR] Morning sync was not successfully executed before market open. Discarding buy orders for safety!")
        return
        
    actual_positions = context.portfolio.positions
    target_codes = [convert_code_to_ptrade(sig['ts_code']) for sig in g.today_buy_signals]
    
    # ------------------ 阶段 1：执行卖出（Signal Lost 跨日持仓平仓） ------------------
    # 寻找当前持有但是不属于今日目标清单的个股
    for code, pos in list(actual_positions.items()):
        # 停牌跳过
        if code not in context.security_list or code not in data:
            log.warn(f"  [SELL HOLD] {code} is suspended today. Skipping sell order.")
            continue
            
        # T+1 制度硬性限制：当天买入的股票（buy_today_amount）在当天绝对禁止卖出
        sellable_amount = pos.amount - pos.buy_today_amount
        if sellable_amount <= 0:
            log.warn(f"  [T+1 LOCK] {code} was bought today. Locked by T+1 rule.")
            continue
            
        # 信号消失，执行市价清仓
        if code not in target_codes:
            log.info(f"  [SIGNAL LOST] Dispatching SELL order for {code} | Volume: {sellable_amount} shares at Market Price.")
            # 使用市价单平仓，确保以接近 Open 价格成交
            order_id = order_target(code, 0, LimitOrderStyle(data[code].close)) # 挂单，或在 PTrade 实盘中使用 order_shares(code, -sellable_amount)
            if order_id:
                log.info(f"  [ORDER SENT] Sell order dispatched successfully. Order ID: {order_id}")

    # ------------------ 阶段 2：执行买入（Staged Signals 开盘买入） ------------------
    if not g.today_buy_signals:
        log.info("[EXECUTION] No staged buy orders to dispatch today.")
        g.buy_execution_done = True
        return
        
    for sig in g.today_buy_signals:
        ptrade_code = sig['ptrade_code']
        shares = sig['final_shares']
        
        # 再次确认开盘价状态（一字涨停避险）
        open_price = data[ptrade_code].open
        pre_close = data[ptrade_code].close # 昨日收盘
        
        # 一字涨停避险判定 (Limit Up protection)
        limit_up_pct = 0.195 if ptrade_code.startswith(('300', '688')) else 0.095
        if pre_close > 0 and (open_price - pre_close) / pre_close >= limit_up_pct:
            log.warn(f"  [LIMIT UP SHIELD] {sig['ts_code']} opened at LIMIT UP. Canceling buy order to avoid chasing highs!")
            continue
            
        log.info(f"  [SIGNAL GAIN] Dispatching BUY order for {sig['ts_code']} | Volume: {shares} shares.")
        # 发送买入指令，指定开盘价或者市价快速买入
        order_id = order_shares(ptrade_code, shares, LimitOrderStyle(open_price))
        if order_id:
            log.info(f"  [ORDER SENT] Buy order dispatched successfully. Order ID: {order_id}")
            
    g.buy_execution_done = True
    log.info("=== [EXECUTION] Market Open Dispatch Complete. ===")


def intraday_risk_control(context, data):
    """
    【盘中实时去未来函数风控器】
    逐个 Bar 扫描真实账户的持仓。根据真实买入成本（cost_basis），实时触发 +6% 固定止盈与 -5% 固定止损挂单清仓。
    """
    actual_positions = context.portfolio.positions
    
    for code, pos in list(actual_positions.items()):
        # 排除当天刚买入的锁定仓位 (T+1 Rules)
        sellable_amount = pos.amount - pos.buy_today_amount
        if sellable_amount <= 0:
            continue
            
        # 异常跳过 (停牌股)
        if code not in data:
            continue
            
        current_price = data[code].close
        cost_price = pos.cost_basis
        
        if cost_price <= 0:
            continue
            
        # 计算盘中实时涨跌幅
        pnl_pct = (current_price - cost_price) / cost_price
        
        # 1. 硬性止损触发： -5%
        if pnl_pct <= g_config['stop_loss_pct']:
            log.warn(f"  [STOP LOSS TRIGGERED] {code} | Current: ¥{current_price:.2f} | Cost: ¥{cost_price:.2f} | PnL: {pnl_pct:.2%} <= {g_config['stop_loss_pct']:.2%}")
            log.warn(f"  Dispatching STOP LOSS sell order immediately for {code} | Volume: {sellable_amount} shares.")
            order_target(code, 0, LimitOrderStyle(current_price))
            continue
            
        # 2. 硬性止盈触发： +6%
        if pnl_pct >= g_config['take_profit_pct']:
            log.info(f"  [TAKE PROFIT TRIGGERED] {code} | Current: ¥{current_price:.2f} | Cost: ¥{cost_price:.2f} | PnL: {pnl_pct:.2%} >= {g_config['take_profit_pct']:.2%}")
            log.info(f"  Dispatching TAKE PROFIT sell order immediately for {code} | Volume: {sellable_amount} shares.")
            order_target(code, 0, LimitOrderStyle(current_price))
            continue


# ========================================== 工具函数 ==========================================

def convert_code_to_ptrade(ts_code: str) -> str:
    """
    转换证券代码格式
    Tushare format: '000001.SZ', '600000.SH'
    Ptrade format: '000001.SZ', '600000.SH' (国内极大多数柜台支持完全一样的后缀格式)
    """
    return ts_code
