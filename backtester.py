import numpy as np
import pandas as pd


def build_strategy_masks(df: pd.DataFrame, cfg: dict) -> dict[str, pd.Series]:
    """
    모든 조건은 시점 t에서 확인 가능한 데이터만 사용(룩어헤드 바이어스 원천 차단).
    - 전략 A: 이동평균선 수렴 이후 거래량 급증
    - 전략 B: 이동평균선 수렴 이후 골든크로스 발생
    - 전략 C: 이동평균선 수렴 + 거래량 급증 + 골든크로스 + 박스권 상단 돌파
    """
    u_cfg = cfg["universe"]
    s1 = cfg["step1_downtrend"]
    s2 = cfg["step2_box"]
    s3 = cfg["step3_convergence"]
    s5 = cfg["step5_volume"]
    link_n = s5["temporal_link_days"]

    # 공통 기반: 장기 하락 + 바닥 박스권 + 이동평균선 수렴 + 유동성
    base_downtrend = (
        (df["Inv_Days_250"] >= s1["min_inverse_days"]) &
        (df["Drawdown_1Y_Pct"] >= s1["min_drawdown_from_1y_high_pct"])
    )
    base_box = (
        (df["Box_Width_Pct"] <= s2["allowed_max_box_width_pct"]) &
        (df["Stopped_Lower_Low"]) &
        (df["Close"] >= df["Box_Low"] * 0.99)
    )
    base_conv = (
        (df["Gap20_200"] <= s3["max_gap_20_200_pct"]) &
        (df["Gap50_200"] <= s3["max_gap_50_200_pct"]) &
        (df["Gap20_50"] <= s3["max_gap_20_50_pct"]) &
        (df["Gap_Reduction_20d"] > 0) &
        (df["Slope20_5d"] > 0) &
        (df["Slope50_5d"] > 0) &
        ((df["Slope200_20d"] > df["Slope200_Prev20d"]) | (df["Slope200_20d"] >= s3["ma200_flat_threshold_pct"]))
    )
    base_liq = (df["Vol_MA20"] >= u_cfg["min_avg20_volume"])

    base_all = (base_downtrend & base_box & base_conv & base_liq).fillna(False)

    # 거래량 급증 (당일 주가 상승 + 최소 거래대금 동반)
    vol_surge = (
        (df["Vol_Ratio_20"] >= s5["vol_surge_ratio"]) &
        (df["Ret_1d"] > 0) &
        (df["TradingVal_Eok"] >= u_cfg["min_trading_val_eok"])
    ).fillna(False)

    # 골든크로스 발생
    gc_signal = df["GC_Any"].fillna(False)

    # 거래량 급증 & 골든크로스 인과적 0~3일 연계
    vol_recent_3d = vol_surge.rolling(link_n + 1).max().astype(bool)
    gc_recent_3d = gc_signal.rolling(link_n + 1).max().astype(bool)
    vol_and_gc_linked = (vol_surge & gc_recent_3d) | (gc_signal & vol_recent_3d)

    # 박스권 상단 종가 돌파 + 20/50일선 상회
    box_break = (
        (df["Box_Breakout"] | df["Box_Breakout_Held"]) &
        (df["Close"] > df["MA20"]) &
        (df["Close"] > df["MA50"])
    ).fillna(False)

    return {
        "전략 A (수렴 + 거래량 급증)": (base_all & vol_surge),
        "전략 B (수렴 + 골든크로스)": (base_all & gc_signal),
        "전략 C (수렴 + 거래량 + GC + 박스권돌파)": (base_all & vol_and_gc_linked & box_break),
    }


def run_backtest_suite(indicator_map: dict[str, pd.DataFrame], cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    전체 종목에 대해 전략 A/B/C × 보유기간(10/20/40/60일) 백테스트 수행.
    체결 규칙: t일 신호 발생 -> t+1일 시가(Open) 매수 -> t+H일 종가(Close) 매도.
    수수료, 증권거래세, 슬리피지 반영 및 동일 종목 보유 중 중복 진입 방지.
    """
    bt_cfg = cfg["backtest"]
    horizons = bt_cfg["holding_days"]
    fee = bt_cfg["commission_rate"]
    tax = bt_cfg["sell_tax_rate"]
    slip = bt_cfg["slippage_rate"]
    start_offset = bt_cfg["start_offset_days"]

    trade_logs = []

    for code, df in indicator_map.items():
        if len(df) <= start_offset + max(horizons) + 2:
            continue

        masks = build_strategy_masks(df, cfg)
        opens = df["Open"].values
        closes = df["Close"].values
        lows = df["Low"].values
        dates = df.index

        for strat_name, mask_series in masks.items():
            sig_indices = np.where(mask_series.values)[0]
            sig_indices = sig_indices[sig_indices >= start_offset]
            if len(sig_indices) == 0:
                continue

            for h in horizons:
                last_exit_idx = -1
                for t in sig_indices:
                    entry_idx = t + 1
                    exit_idx = t + h
                    if exit_idx >= len(df):
                        break
                    # 동일 종목 보유 기간 내 중복 진입 차단
                    if entry_idx <= last_exit_idx:
                        continue

                    entry_raw = opens[entry_idx]
                    exit_raw = closes[exit_idx]
                    if entry_raw <= 0 or exit_raw <= 0 or np.isnan(entry_raw) or np.isnan(exit_raw):
                        continue

                    # 비용 반영 (매수 슬리피지·수수료 / 매도 슬리피지·수수료·거래세)
                    buy_cost = entry_raw * (1.0 + slip) * (1.0 + fee)
                    sell_net = exit_raw * (1.0 - slip) * (1.0 - fee - tax)
                    net_ret = (sell_net / buy_cost) - 1.0

                    # 보유 중 최저가 기준 개별 트레이드 최대 낙폭
                    min_low = np.nanmin(lows[entry_idx : exit_idx + 1])
                    intra_mdd = (min_low / entry_raw) - 1.0

                    trade_logs.append({
                        "전략": strat_name,
                        "보유기간(일)": h,
                        "종목코드": code,
                        "신호일": dates[t].strftime("%Y-%m-%d"),
                        "진입일": dates[entry_idx].strftime("%Y-%m-%d"),
                        "청산일": dates[exit_idx].strftime("%Y-%m-%d"),
                        "순수익률(%)": round(net_ret * 100.0, 2),
                        "보유중MDD(%)": round(intra_mdd * 100.0, 2),
                    })
                    last_exit_idx = exit_idx

    trades_df = pd.DataFrame(trade_logs)
    summary_rows = []

    strat_names = [
        "전략 A (수렴 + 거래량 급증)",
        "전략 B (수렴 + 골든크로스)",
        "전략 C (수렴 + 거래량 + GC + 박스권돌파)",
    ]

    for s_name in strat_names:
        for h in horizons:
            if trades_df.empty:
                sub = pd.DataFrame()
            else:
                sub = trades_df[(trades_df["전략"] == s_name) & (trades_df["보유기간(일)"] == h)].copy()

            if sub.empty:
                summary_rows.append({
                    "전략": s_name, "보유기간(일)": h, "거래 횟수": 0,
                    "승률(%)": 0.0, "평균 수익률(%)": 0.0, "평균 손실률(%)": 0.0,
                    "손익비": 0.0, "최대 낙폭 MDD(%)": 0.0, "기대수익(%)": 0.0
                })
                continue

            rets = sub["순수익률(%)"].values
            wins = rets[rets > 0]
            losses = rets[rets <= 0]

            win_rate = len(wins) / len(rets)
            avg_win = float(np.mean(wins)) if len(wins) > 0 else 0.0
            avg_loss = float(np.mean(losses)) if len(losses) > 0 else 0.0
            payoff = round(avg_win / abs(avg_loss), 2) if avg_loss < 0 else (99.0 if avg_win > 0 else 0.0)
            expectancy = round(win_rate * avg_win + (1.0 - win_rate) * avg_loss, 2)

            # 시간순 누적 수익곡선 기반 포트폴리오 MDD (회당 10% 고정 비중 가정)
            sub_sorted = sub.sort_values("청산일")
            equity = (1.0 + (sub_sorted["순수익률(%)"].values / 100.0) * 0.1).cumprod()
            peak = np.maximum.accumulate(equity)
            dd = (equity / peak - 1.0) * 100.0
            port_mdd = round(float(np.min(dd)), 2) if len(dd) > 0 else 0.0

            summary_rows.append({
                "전략": s_name,
                "보유기간(일)": h,
                "거래 횟수": len(rets),
                "승률(%)": round(win_rate * 100.0, 2),
                "평균 수익률(%)": round(avg_win, 2),
                "평균 손실률(%)": round(avg_loss, 2),
                "손익비": payoff,
                "최대 낙폭 MDD(%)": port_mdd,
                "기대수익(%)": expectancy
            })

    return pd.DataFrame(summary_rows), trades_df
