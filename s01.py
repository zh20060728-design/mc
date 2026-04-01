import numpy as np
import pandas as pd


def ema(series, period):
    return series.ewm(span=period, adjust=False).mean()


def atr(df, period=14):
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift(1)).abs()
    low_close = (df["low"] - df["close"].shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def prepare_ohlcv(df):
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["timestamp", "open", "high", "low", "close", "volume"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def build_4h_from_15m(df_15m):
    temp = df_15m.set_index("timestamp")[["open", "high", "low", "close", "volume"]]
    df_4h = (
        temp.resample("4H")
        .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
        .dropna()
        .reset_index()
    )
    return df_4h


def trendline_value(series, idx_start, idx_end, target_idx):
    x = np.arange(idx_start, idx_end)
    y = series.iloc[idx_start:idx_end].values
    if len(x) < 2:
        return np.nan
    slope, intercept = np.polyfit(x, y, 1)
    return slope * target_idx + intercept


def in_eu_or_us_session(ts):
    # UTC时间下，简化欧盘/美盘过滤
    hour = ts.hour
    eu = 7 <= hour <= 15
    us = 13 <= hour <= 22
    return eu or us


def generate_signals(df_15m, df_4h):
    df_15m = prepare_ohlcv(df_15m)
    df_4h = prepare_ohlcv(df_4h)

    # 4H趋势：EMA50 与 EMA200
    df_4h["ema50"] = ema(df_4h["close"], 50)
    df_4h["ema200"] = ema(df_4h["close"], 200)

    # 15M指标
    df_15m["ema50"] = ema(df_15m["close"], 50)
    df_15m["atr14"] = atr(df_15m, 14)
    df_15m["atr_mean_50"] = df_15m["atr14"].rolling(50).mean()
    df_15m["vol_ma20"] = df_15m["volume"].rolling(20).mean()

    # 挂接最近4H趋势到15M
    trend_cols = df_4h[["timestamp", "ema50", "ema200"]].sort_values("timestamp")
    df = pd.merge_asof(
        df_15m.sort_values("timestamp"),
        trend_cols,
        on="timestamp",
        direction="backward",
        suffixes=("", "_4h"),
    )

    signals = []
    position = None

    for i in range(220, len(df)):
        row = df.iloc[i]

        # 过滤：欧盘/美盘
        if not in_eu_or_us_session(row["timestamp"]):
            continue

        # 过滤：ATR太低不做（当前ATR需大于自身50均值）
        if np.isnan(row["atr14"]) or np.isnan(row["atr_mean_50"]) or row["atr14"] < row["atr_mean_50"]:
            continue

        # 过滤：成交量阈值
        if np.isnan(row["vol_ma20"]) or row["volume"] <= row["vol_ma20"] * 1.2:
            continue

        # 4H方向判断
        trend = "none"
        if row["ema50_4h"] > row["ema200_4h"]:
            trend = "long"
        elif row["ema50_4h"] < row["ema200_4h"]:
            trend = "short"

        if trend == "none":
            continue

        if position is None:
            lookback = 30
            start = i - lookback

            # 第二步：结构线（long用下降趋势线，short用上升趋势线）
            if trend == "long":
                line_prev = trendline_value(df["high"], start, i, i - 1)
                line_now = trendline_value(df["high"], start, i, i)

                breakout = df.iloc[i - 1]["close"] <= line_prev and row["close"] > line_now
                if not breakout:
                    continue

                # 第三步：回踩趋势线 or EMA50 且不破（检查最近5根）
                pulled = False
                for j in range(i - 5, i + 1):
                    lv = trendline_value(df["high"], start, i, j)
                    touched_line = df.iloc[j]["low"] <= lv and df.iloc[j]["close"] >= lv
                    touched_ema = df.iloc[j]["low"] <= df.iloc[j]["ema50"] and df.iloc[j]["close"] >= df.iloc[j]["ema50"]
                    if touched_line or touched_ema:
                        pulled = True
                        break
                if not pulled:
                    continue

                # 第四步：再次突破前高 + 放量
                prior_high = df.iloc[i - 10 : i]["high"].max()
                if row["close"] <= prior_high:
                    continue

                entry = row["close"]
                stop = df.iloc[i - 5 : i + 1]["low"].min()  # 回踩低点
                risk = entry - stop
                if risk <= 0:
                    continue
                tp = entry + 2 * risk  # 2R
                position = {"side": "long", "entry": entry, "stop": stop, "tp": tp, "entry_time": row["timestamp"]}

            elif trend == "short":
                line_prev = trendline_value(df["low"], start, i, i - 1)
                line_now = trendline_value(df["low"], start, i, i)

                breakout = df.iloc[i - 1]["close"] >= line_prev and row["close"] < line_now
                if not breakout:
                    continue

                pulled = False
                for j in range(i - 5, i + 1):
                    lv = trendline_value(df["low"], start, i, j)
                    touched_line = df.iloc[j]["high"] >= lv and df.iloc[j]["close"] <= lv
                    touched_ema = df.iloc[j]["high"] >= df.iloc[j]["ema50"] and df.iloc[j]["close"] <= df.iloc[j]["ema50"]
                    if touched_line or touched_ema:
                        pulled = True
                        break
                if not pulled:
                    continue

                prior_low = df.iloc[i - 10 : i]["low"].min()
                if row["close"] >= prior_low:
                    continue

                entry = row["close"]
                stop = df.iloc[i - 5 : i + 1]["high"].max()
                risk = stop - entry
                if risk <= 0:
                    continue
                tp = entry - 2 * risk
                position = {"side": "short", "entry": entry, "stop": stop, "tp": tp, "entry_time": row["timestamp"]}

        else:
            side = position["side"]
            if side == "long":
                if row["low"] <= position["stop"]:
                    signals.append({**position, "exit": position["stop"], "result": "LOSS", "exit_time": row["timestamp"]})
                    position = None
                elif row["high"] >= position["tp"]:
                    signals.append({**position, "exit": position["tp"], "result": "WIN", "exit_time": row["timestamp"]})
                    position = None
            else:
                if row["high"] >= position["stop"]:
                    signals.append({**position, "exit": position["stop"], "result": "LOSS", "exit_time": row["timestamp"]})
                    position = None
                elif row["low"] <= position["tp"]:
                    signals.append({**position, "exit": position["tp"], "result": "WIN", "exit_time": row["timestamp"]})
                    position = None

    return signals


def backtest(signals):
    wins = sum(1 for s in signals if s["result"] == "WIN")
    losses = sum(1 for s in signals if s["result"] == "LOSS")
    total = len(signals)
    win_rate = (wins / total * 100) if total else 0

    print(f"总交易次数: {total}")
    print(f"盈利次数: {wins}")
    print(f"亏损次数: {losses}")
    print(f"胜率: {win_rate:.2f}%")


if __name__ == "__main__":
    # 默认用15m.csv，4H由15m重采样得到，满足双周期回测
    df_15m = pd.read_csv("15m.csv")
    df_15m = prepare_ohlcv(df_15m)
    df_4h = build_4h_from_15m(df_15m)

    trade_signals = generate_signals(df_15m, df_4h)
    backtest(trade_signals)
