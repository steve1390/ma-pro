import numpy as np
import pandas as pd


def compute_all_indicators(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    모든 기술적 지표를 시점 t 기준 철저히 인과적(미래 데이터 미참조)으로 계산.
    라이브 스크리너와 백테스터가 완전히 동일한 지표 DataFrame을 공유한다.
    """
    out = df.copy()
    box_n = cfg["step2_box"]["box_window_days"]
    s_short = cfg["step3_convergence"]["slope_lookback_short"]
    s_long = cfg["step3_convergence"]["slope_lookback_long"]
    red_n = cfg["step3_convergence"]["gap_reduction_lookback"]
    inv_win = cfg["step1_downtrend"]["lookback_days"]

    # 1. 이동평균선 (20, 40, 50, 200)
    out["MA20"] = out["Close"].rolling(20).mean()
    out["MA40"] = out["Close"].rolling(40).mean()
    out["MA50"] = out["Close"].rolling(50).mean()
    out["MA200"] = out["Close"].rolling(200).mean()

    # 2. 1단계: 장기 하락 이력 지표
    out["Is_Inverse_40_200"] = (out["MA40"] < out["MA200"]).astype(int)
    out["Inv_Days_250"] = out["Is_Inverse_40_200"].rolling(inv_win).sum()
    out["High_1Y"] = out["High"].rolling(inv_win).max()
    out["Low_1Y"] = out["Low"].rolling(inv_win).min()
    out["Drawdown_1Y_Pct"] = ((out["High_1Y"] - out["Close"]) / out["High_1Y"].replace(0, np.nan)) * 100.0
    out["Rebound_From_Low_Pct"] = ((out["Close"] - out["Low_1Y"]) / out["Low_1Y"].replace(0, np.nan)) * 100.0

    # 3. 3단계: 이동평균선 이격도(%) 및 축소율(%)
    out["Gap20_200"] = (out["MA20"] - out["MA200"]).abs() / out["MA200"].replace(0, np.nan) * 100.0
    out["Gap50_200"] = (out["MA50"] - out["MA200"]).abs() / out["MA200"].replace(0, np.nan) * 100.0
    out["Gap20_50"] = (out["MA20"] - out["MA50"]).abs() / out["MA50"].replace(0, np.nan) * 100.0
    out["Gap40_200"] = (out["MA40"] - out["MA200"]).abs() / out["MA200"].replace(0, np.nan) * 100.0

    out["Gap_Composite"] = (out["Gap20_200"] + out["Gap50_200"] + out["Gap20_50"]) / 3.0
    gap_prev = out["Gap_Composite"].shift(red_n)
    out["Gap_Reduction_20d"] = ((gap_prev - out["Gap_Composite"]) / gap_prev.replace(0, np.nan)) * 100.0
    # 최근 20거래일 중 전일 대비 이격도가 축소된 일수 비율 (지속적 수렴 여부 판별)
    out["Gap_Narrow_Days_20d"] = (out["Gap_Composite"].diff() < 0).rolling(red_n).sum()

    # 이동평균선별 기울기(%)
    out["Slope20_5d"] = (out["MA20"] - out["MA20"].shift(s_short)) / out["MA20"].shift(s_short) * 100.0
    out["Slope40_5d"] = (out["MA40"] - out["MA40"].shift(s_short)) / out["MA40"].shift(s_short) * 100.0
    out["Slope50_5d"] = (out["MA50"] - out["MA50"].shift(s_short)) / out["MA50"].shift(s_short) * 100.0
    out["Slope50_10d"] = (out["MA50"] - out["MA50"].shift(10)) / out["MA50"].shift(10) * 100.0
    out["Slope200_20d"] = (out["MA200"] - out["MA200"].shift(s_long)) / out["MA200"].shift(s_long) * 100.0
    out["Slope200_Prev20d"] = out["Slope200_20d"].shift(s_long)

    # 현재가 대비 40일선 이격도(%) 및 5일 단기 수익률
    out["Disparity_40"] = (out["Close"] - out["MA40"]) / out["MA40"].replace(0, np.nan) * 100.0
    out["Ret_1d"] = out["Close"].pct_change() * 100.0
    out["Ret_5d"] = out["Close"].pct_change(5) * 100.0

    # 4. 4단계: 골든크로스 정의 (전일 short <= long AND 당일 short > long)
    out["GC_20_50"] = (out["MA20"].shift(1) <= out["MA50"].shift(1)) & (out["MA20"] > out["MA50"])
    out["GC_20_200"] = (out["MA20"].shift(1) <= out["MA200"].shift(1)) & (out["MA20"] > out["MA200"])
    out["GC_40_200"] = (out["MA40"].shift(1) <= out["MA200"].shift(1)) & (out["MA40"] > out["MA200"])
    out["GC_50_200"] = (out["MA50"].shift(1) <= out["MA200"].shift(1)) & (out["MA50"] > out["MA200"])
    out["GC_Any"] = out["GC_20_50"] | out["GC_20_200"] | out["GC_40_200"] | out["GC_50_200"]

    # 5. 2단계: 철저히 인과적인 박스권 계산 (반드시 t-1일까지 데이터만 사용!)
    prev_high = out["High"].shift(1)
    prev_low = out["Low"].shift(1)
    prev_close = out["Close"].shift(1)

    out["Box_High"] = prev_high.rolling(box_n).max()
    out["Box_Low"] = prev_low.rolling(box_n).min()
    out["Box_Width_Pct"] = ((out["Box_High"] - out["Box_Low"]) / out["Box_Low"].replace(0, np.nan)) * 100.0

    # 저점 하락 중단 여부 (박스권 후반부 저점 >= 전반부 저점 * 0.985)
    half_n = max(box_n // 2, 10)
    out["Box_Low_RecentHalf"] = prev_low.rolling(half_n).min()
    out["Box_Low_PrevHalf"] = prev_low.shift(half_n).rolling(half_n).min()
    out["Stopped_Lower_Low"] = out["Box_Low_RecentHalf"] >= (out["Box_Low_PrevHalf"] * 0.985)

    # 박스권 상단 돌파 여부 (종가 기준) 및 돌파 후 1~3일 지지 확인
    out["Box_Breakout"] = out["Close"] > out["Box_High"]
    # 최근 1~3일 전 돌파 발생 시 그 당시 박스권 상단을 현재 종가가 유지하는지 확인 (룩어헤드 없음)
    hold_1 = out["Box_Breakout"].shift(1).fillna(False) & (out["Close"] >= out["Box_High"].shift(1) * 0.995)
    hold_2 = out["Box_Breakout"].shift(2).fillna(False) & (out["Close"] >= out["Box_High"].shift(2) * 0.995) & (out["Close"].shift(1) >= out["Box_High"].shift(2) * 0.995)
    hold_3 = out["Box_Breakout"].shift(3).fillna(False) & (out["Close"] >= out["Box_High"].shift(3) * 0.995) & (out["Close"].shift(1) >= out["Box_High"].shift(3) * 0.995)
    out["Box_Breakout_Held"] = hold_1 | hold_2 | hold_3

    # 6. 5단계: 거래량 및 거래대금 지표
    out["Vol_MA5"] = out["Volume"].rolling(5).mean()
    out["Vol_MA20"] = out["Volume"].rolling(20).mean()
    out["Vol_MA60"] = out["Volume"].rolling(60).mean()
    out["Vol_Ratio_20"] = out["Volume"] / out["Vol_MA20"].replace(0, np.nan)
    out["Vol_5_vs_20"] = out["Vol_MA5"] / out["Vol_MA20"].replace(0, np.nan)
    out["TradingVal_Eok"] = (out["Close"] * out["Volume"]) / 1e8
    out["Vol_Contracted_Pre"] = out["Vol_MA20"].shift(1) <= (out["Vol_MA60"].shift(1) * 0.95)

    # 최근 20일 상승일 vs 하락일 평균 거래량 비율
    up_vol = np.where(out["Close"] >= out["Close"].shift(1), out["Volume"], np.nan)
    dn_vol = np.where(out["Close"] < out["Close"].shift(1), out["Volume"], np.nan)
    up_vol_ma = pd.Series(up_vol, index=out.index).rolling(20, min_periods=1).mean()
    dn_vol_ma = pd.Series(dn_vol, index=out.index).rolling(20, min_periods=1).mean()
    out["Up_Dn_Vol_Ratio"] = up_vol_ma / dn_vol_ma.replace(0, np.nan)

    # 7. 7단계: 보조지표 (RSI 14, MACD 12/26/9, ATR 14, BB 20/2, ADX 14)
    delta = out["Close"].diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out["RSI14"] = (100.0 - (100.0 / (1.0 + rs))).fillna(50.0)

    ema12 = out["Close"].ewm(span=12, adjust=False).mean()
    ema26 = out["Close"].ewm(span=26, adjust=False).mean()
    out["MACD"] = ema12 - ema26
    out["MACD_Sig"] = out["MACD"].ewm(span=9, adjust=False).mean()
    out["MACD_Hist"] = out["MACD"] - out["MACD_Sig"]

    tr1 = out["High"] - out["Low"]
    tr2 = (out["High"] - out["Close"].shift(1)).abs()
    tr3 = (out["Low"] - out["Close"].shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    out["ATR14"] = tr.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    out["ATR10"] = tr.rolling(10).mean()
    out["ATR40"] = tr.rolling(40).mean()
    out["ATR_Contracted"] = out["ATR10"].shift(1) <= (out["ATR40"].shift(1) * 1.02)

    bb_mid = out["Close"].rolling(20).mean()
    bb_std = out["Close"].rolling(20).std()
    out["BB_Upper"] = bb_mid + 2.0 * bb_std
    out["BB_Lower"] = bb_mid - 2.0 * bb_std
    out["BB_Width"] = ((out["BB_Upper"] - out["BB_Lower"]) / bb_mid.replace(0, np.nan)) * 100.0
    out["BB_Width_MA60"] = out["BB_Width"].rolling(60).mean()

    up_move = out["High"].diff()
    down_move = -out["Low"].diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_di = 100.0 * pd.Series(plus_dm, index=out.index).ewm(alpha=1 / 14, min_periods=14, adjust=False).mean() / out["ATR14"].replace(0, np.nan)
    minus_di = 100.0 * pd.Series(minus_dm, index=out.index).ewm(alpha=1 / 14, min_periods=14, adjust=False).mean() / out["ATR14"].replace(0, np.nan)
    dx = (100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)).fillna(0.0)
    out["ADX14"] = dx.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()

    # 8. 최근 120일 주요 저항선 (목표가 산출용)
    out["Res_120"] = prev_high.rolling(120).max()

    return out
