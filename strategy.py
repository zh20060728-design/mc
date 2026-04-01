import pandas as pd
import numpy as np

# ======================
# 📊 指标函数
# ======================

def EMA(series, period):
    return series.ewm(span=period, adjust=False).mean()

def ATR(df, period=14):
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def ADX(df, period=14):
    high = df['high']
    low = df['low']
    close = df['close']

    plus_dm = high.diff()
    minus_dm = low.diff().abs()

    plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0)
    minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0)

    tr = pd.concat([
        high - low,
        abs(high - close.shift()),
        abs(low - close.shift())
    ], axis=1).max(axis=1)

    atr = tr.rolling(period).mean()

    plus_di = 100 * (pd.Series(plus_dm).rolling(period).mean() / atr)
    minus_di = 100 * (pd.Series(minus_dm).rolling(period).mean() / atr)

    dx = abs(plus_di - minus_di) / (plus_di + minus_di) * 100
    return dx.rolling(period).mean()

# ======================
# 📊 策略核心（15m + 4H）
# ======================

def generate_signals(df, df_4h):

    # === 时间处理 ===
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df_4h['timestamp'] = pd.to_datetime(df_4h['timestamp'], errors='coerce')

    df = df.dropna()
    df_4h = df_4h.dropna()

    # === 转数值 ===
    for col in ['open','high','low','close','volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df_4h[col] = pd.to_numeric(df_4h[col], errors='coerce')

    # === 指标 ===
    df['ema20'] = EMA(df['close'], 20)
    df['atr'] = ATR(df)
    df['adx'] = ADX(df)

    df_4h['ema50'] = EMA(df_4h['close'], 50)

    signals = []

    in_position = False
    entry_price = 0
    stop_loss = 0
    take_profit = 0

    loss_streak = 0

    for i in range(100, len(df)):

        # 🔥 连亏保护
        if loss_streak >= 3:
            continue

        current = df.iloc[i]
        prev = df.iloc[i-1]

        # ======================
        # 🔥 4H趋势过滤（更强）
        # ======================
        current_time = current['timestamp']
        df_4h_match = df_4h[df_4h['timestamp'] <= current_time]

        if len(df_4h_match) < 50:
            continue

        last_4h = df_4h_match.iloc[-1]
        prev_4h = df_4h_match.iloc[-3]

        if last_4h['ema50'] < prev_4h['ema50']:
            continue

        # ======================
        # 🔥 15m策略
        # ======================

        if current['adx'] < 20:
            continue

        # 趋势
        if current['ema20'] < df.iloc[i-10]['ema20']:
            continue

        # ATR过滤（避免震荡）
        if abs(current['close'] - current['ema20']) < current['atr'] * 0.3:
            continue

        # 突破（更宽）
        recent_high = df.iloc[i-30:i]['high'].max()
        breakout = current['high'] > recent_high + current['atr'] * 0.2

        # 区间位置过滤🔥
        recent_low = df.iloc[i-30:i]['low'].min()
        mid = (recent_high + recent_low) / 2
        if current['close'] < mid:
            continue

        # 回踩（更宽）
        pullback = df.iloc[i-5:i]['low'].min() <= current['ema20'] * 1.03

        # 确认K线
        confirm = prev['close'] > prev['open']

        # 强度
        body = abs(current['close'] - current['open'])
        range_ = current['high'] - current['low']
        strong = body > range_ * 0.3

        if current['close'] < current['open']:
            continue

        # ======================
        # 🚀 入场
        # ======================
        if breakout and pullback and confirm and strong:

            if not in_position:

                entry_price = current['close']
                atr = current['atr']

                if np.isnan(atr):
                    continue

                stop_loss = entry_price - 1.2 * atr
                take_profit = entry_price + 3 * (entry_price - stop_loss)

                in_position = True
                print(f"🟢 开仓: {entry_price}")

        # ======================
        # 📉 持仓管理
        # ======================
        if in_position:

            # 保本
            if current['close'] > entry_price + (entry_price - stop_loss):
                stop_loss = entry_price

            if current['low'] <= stop_loss:
                signals.append(("LOSS", stop_loss))
                print("🔴 止损")
                in_position = False
                loss_streak += 1

            elif current['high'] >= take_profit:
                signals.append(("WIN", take_profit))
                print("🟢 止盈")
                in_position = False
                loss_streak = 0

    return signals

# ======================
# 📊 回测
# ======================

def backtest(signals):
    wins = sum(1 for s in signals if s[0] == "WIN")
    losses = sum(1 for s in signals if s[0] == "LOSS")
    total = len(signals)

    win_rate = wins / total * 100 if total > 0 else 0

    print(f"\n总交易次数: {total}")
    print(f"盈利次数: {wins}")
    print(f"亏损次数: {losses}")
    print(f"胜率: {win_rate:.2f}%")

# ======================
# 🚀 主程序
# ======================

if __name__ == "__main__":

    df = pd.read_csv("btc_4h_202501-06.csv")
    df_4h = pd.read_csv("btc_4h_202501-06.csv")

    signals = generate_signals(df, df_4h)
    backtest(signals)
