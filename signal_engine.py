import numpy as np
import pandas as pd


def classify_convergence_state(row: pd.Series, cfg: dict) -> tuple[str, str]:
    """
    3단계 C: 이동평균선 수렴 상태 및 세부 위치(20선 선행돌파 / 50선 접근 / 동시 근접) 분류
    """
    c_cfg = cfg["step3_convergence"]
    g20_200 = row["Gap20_200"]
    g50_200 = row["Gap50_200"]
    g20_50 = row["Gap20_50"]
    red20 = row["Gap_Reduction_20d"]
    narrow_days = row["Gap_Narrow_Days_20d"]
    s20 = row["Slope20_5d"]
    s50 = row["Slope50_5d"]
    s200 = row["Slope200_20d"]
    s200_prev = row["Slope200_Prev20d"]

    ma200_flattening = (s200 > s200_prev) or (s200 >= c_cfg["ma200_flat_threshold_pct"])
    short_rising = (s20 > 0) and (s50 > 0)
    within_tight = (
        (g20_200 <= c_cfg["max_gap_20_200_pct"]) and
        (g50_200 <= c_cfg["max_gap_50_200_pct"]) and
        (g20_50 <= c_cfg["max_gap_20_50_pct"])
    )

    # 세부 위치 구분
    ma20_above_200 = row["MA20"] > row["MA200"]
    ma50_above_200 = row["MA50"] > row["MA200"]
    if ma20_above_200 and ma50_above_200:
        pos_detail = "20·50일선 모두 200일선 상향 돌파"
    elif ma20_above_200 and not ma50_above_200:
        pos_detail = "20일선 선행 돌파 + 50일선 200일선 접근"
    elif (not ma20_above_200) and g50_200 <= c_cfg["max_gap_50_200_pct"] and g20_200 <= c_cfg["max_gap_20_200_pct"]:
        pos_detail = "20·50일선 모두 200일선 하단 근접 수렴"
    else:
        pos_detail = "역배열 수렴 진행"

    # 수렴 상태 4단계 분류 (단순 근접 배제: 기울기·추세 필수 반영)
    if (row["GC_Any"] or (ma20_above_200 and row["Close"] > row["MA200"])) and short_rising and ma200_flattening:
        state = "상승 전환"
    elif within_tight and short_rising and ma200_flattening and red20 > 0:
        state = "수렴 완료"
    elif red20 >= 15.0 and narrow_days >= 11 and (s20 > 0 or s50 > 0):
        state = "수렴 진행"
    elif red20 > 0 and (g20_200 <= 7.0 or g50_200 <= 7.0) and s20 > 0:
        state = "수렴 초기"
    else:
        state = "미수렴"

    return state, pos_detail


def evaluate_signal_at_index(df: pd.DataFrame, idx: int, meta: dict, cfg: dict) -> dict | None:
    """
    특정 거래일 인덱스(idx) 시점에서 미래 데이터 없이 1~9단계 전체 분석 및 100점 채점 수행.
    """
    if idx < 250 or idx >= len(df):
        return None

    row = df.iloc[idx]
    prev_row = df.iloc[idx - 1]
    if pd.isna(row["MA200"]) or pd.isna(row["Box_High"]) or pd.isna(row["ATR14"]) or pd.isna(row["Gap_Reduction_20d"]):
        return None

    u_cfg = cfg["universe"]
    s1_cfg = cfg["step1_downtrend"]
    s2_cfg = cfg["step2_box"]
    s3_cfg = cfg["step3_convergence"]
    s4_cfg = cfg["step4_golden_cross"]
    s5_cfg = cfg["step5_volume"]
    s8_cfg = cfg["step8_scoring"]

    price = float(row["Close"])
    vol_ma20 = float(row["Vol_MA20"]) if pd.notna(row["Vol_MA20"]) else 0.0
    if vol_ma20 < u_cfg["min_avg20_volume"]:
        return None

    # =========================================================
    # [1단계] 장기 하락 이력 탐색 (배점 10점)
    # =========================================================
    inv_days = int(row["Inv_Days_250"]) if pd.notna(row["Inv_Days_250"]) else 0
    dd_1y = float(row["Drawdown_1Y_Pct"])
    rebound_pct = float(row["Rebound_From_Low_Pct"])
    s200 = float(row["Slope200_20d"])
    s200_prev = float(row["Slope200_Prev20d"])
    ma200_flat = (s200 > s200_prev) or (s200 >= s3_cfg["ma200_flat_threshold_pct"])

    # 저점 형성 확인 (최근 1년 최저점이 오늘이 아니며 저점 대비 반등 유지)
    win250 = df.iloc[idx - s1_cfg["lookback_days"] + 1 : idx + 1]
    low_pos = int(np.argmin(win250["Low"].values))
    days_since_low = (len(win250) - 1) - low_pos
    bottom_formed = (days_since_low >= 10) and (rebound_pct >= 3.0)

    cond_downtrend = inv_days >= s1_cfg["min_inverse_days"]

    score_1 = 0.0
    if inv_days >= s1_cfg["min_inverse_days"]:
        score_1 += 4.5 if inv_days >= 210 else 3.5
    if dd_1y >= s1_cfg["min_drawdown_from_1y_high_pct"]:
        score_1 += 2.5
    elif dd_1y >= s1_cfg["min_drawdown_from_1y_high_pct"] * 0.7:
        score_1 += 1.5
    if bottom_formed:
        score_1 += 1.5
    if ma200_flat:
        score_1 += 1.5
    score_1 = min(10.0, round(score_1, 1))

    # =========================================================
    # [2단계] 바닥 형성 및 박스권 분석 (배점 15점, 인과적 t-1 기준)
    # =========================================================
    box_n = s2_cfg["box_window_days"]
    box_high = float(row["Box_High"])
    box_low = float(row["Box_Low"])
    box_width = float(row["Box_Width_Pct"])
    stopped_ll = bool(row["Stopped_Lower_Low"])
    vol_contracted = bool(row["Vol_Contracted_Pre"])
    atr_contracted = bool(row["ATR_Contracted"])

    # 박스권 구간 내 하단 지지 횟수 (t-box_n ~ t-1)
    box_slice = df.iloc[idx - box_n : idx]
    support_touches = int(((box_slice["Low"] <= box_low * 1.03) & (box_slice["Close"] >= box_low * 0.99)).sum())
    holding_box_low = price >= box_low * 0.99

    cond_box = (box_width <= s2_cfg["allowed_max_box_width_pct"]) and holding_box_low and (stopped_ll or support_touches >= s2_cfg["min_support_touches"])

    score_2 = 0.0
    if box_width <= s2_cfg["preferred_max_box_width_pct"]:
        score_2 += 4.5
    elif box_width <= s2_cfg["allowed_max_box_width_pct"]:
        score_2 += 2.5
    if support_touches >= s2_cfg["min_support_touches"]:
        score_2 += 3.0
    elif support_touches == 1:
        score_2 += 1.5
    if stopped_ll:
        score_2 += 2.5
    if vol_contracted:
        score_2 += 2.5
    if atr_contracted:
        score_2 += 2.5
    score_2 = min(15.0, round(score_2, 1))

    # =========================================================
    # [3단계] 20·40·50·200일선 수렴 분석 (배점 20점)
    # =========================================================
    conv_state, pos_detail = classify_convergence_state(row, cfg)
    g20_200 = float(row["Gap20_200"])
    g50_200 = float(row["Gap50_200"])
    g20_50 = float(row["Gap20_50"])
    g40_200 = float(row["Gap40_200"])
    red20 = float(row["Gap_Reduction_20d"])
    narrow_days = int(row["Gap_Narrow_Days_20d"])
    s20 = float(row["Slope20_5d"])
    s40 = float(row["Slope40_5d"])
    s50 = float(row["Slope50_5d"])

    cond_convergence = conv_state in ("수렴 완료", "상승 전환", "수렴 진행") and (g20_200 <= s3_cfg["max_gap_20_200_pct"] * 1.5)

    score_3 = 0.0
    if g20_200 <= s3_cfg["max_gap_20_200_pct"]:
        score_3 += 4.0
    elif g20_200 <= s3_cfg["max_gap_20_200_pct"] * 1.5:
        score_3 += 2.0

    if g50_200 <= s3_cfg["max_gap_50_200_pct"]:
        score_3 += 4.0
    elif g50_200 <= s3_cfg["max_gap_50_200_pct"] * 1.5:
        score_3 += 2.0

    if g20_50 <= s3_cfg["max_gap_20_50_pct"]:
        score_3 += 3.0
    elif g20_50 <= s3_cfg["max_gap_20_50_pct"] * 1.5:
        score_3 += 1.5

    # 지속적 이격 축소 평가
    if red20 >= 30.0 and narrow_days >= 12:
        score_3 += 4.0
    elif red20 >= 15.0 and narrow_days >= 10:
        score_3 += 2.5
    elif red20 > 0:
        score_3 += 1.0

    # 기울기 평가 (단순 근접 방지)
    if s20 > 0 and s50 > 0:
        score_3 += 3.0
    elif s20 > 0:
        score_3 += 1.5
    if ma200_flat:
        score_3 += 2.0
    score_3 = min(20.0, round(score_3, 1))

    # =========================================================
    # [4단계] 골든크로스 세분화 탐지 (배점 15점)
    # =========================================================
    gc_win = s4_cfg["recent_gc_window_days"]
    recent_slice = df.iloc[max(0, idx - gc_win + 1) : idx + 1]

    gc_types_found = []
    gc_days_ago_list = []
    for col, label in [
        ("GC_20_50", "20x50"),
        ("GC_20_200", "20x200"),
        ("GC_40_200", "40x200"),
        ("GC_50_200", "50x200"),
    ]:
        hits = np.where(recent_slice[col].values)[0]
        if len(hits) > 0:
            last_hit = hits[-1]
            days_ago = (len(recent_slice) - 1) - last_hit
            hit_date = recent_slice.index[last_hit].strftime("%Y-%m-%d")
            gc_types_found.append(f"{label}({hit_date}, {days_ago}일전)")
            gc_days_ago_list.append(days_ago)

    has_recent_gc = len(gc_types_found) > 0
    min_gc_days_ago = min(gc_days_ago_list) if gc_days_ago_list else 999
    gc_today = bool(row["GC_Any"])

    score_4 = 0.0
    if any("50x200" in x for x in gc_types_found):
        score_4 += 6.0
    if any("40x200" in x for x in gc_types_found):
        score_4 += 5.0
    if any("20x200" in x for x in gc_types_found):
        score_4 += 5.0
    if any("20x50" in x for x in gc_types_found):
        score_4 += 3.5
    if price > row["MA20"] and price > row["MA50"]:
        score_4 += 2.5
    if s20 > 0 and s50 > 0:
        score_4 += 1.5
    score_4 = min(15.0, round(score_4, 1))

    # =========================================================
    # [5단계] 수렴 상태에서 거래량 폭발 탐지 (배점 20점, 인과적 시간차 확인)
    # =========================================================
    vol_ratio = float(row["Vol_Ratio_20"])
    vol_5_20 = float(row["Vol_5_vs_20"])
    tval_eok = float(row["TradingVal_Eok"])
    up_dn_vol = float(row["Up_Dn_Vol_Ratio"]) if pd.notna(row["Up_Dn_Vol_Ratio"]) else 1.0
    price_up_today = price > float(prev_row["Close"])

    vol_surge_today = (
        (vol_ratio >= s5_cfg["vol_surge_ratio"]) and
        price_up_today and
        (tval_eok >= u_cfg["min_trading_val_eok"])
    )

    # 최근 1~3거래일 내 인과적 시간 관계 확인 (미래 참조 없음)
    link_n = s5_cfg["temporal_link_days"]
    prev_link = df.iloc[max(0, idx - link_n) : idx]
    prev_vol_surge = bool(
        (
            (prev_link["Vol_Ratio_20"] >= s5_cfg["vol_surge_ratio"]) &
            (prev_link["Ret_1d"] > 0) &
            (prev_link["TradingVal_Eok"] >= u_cfg["min_trading_val_eok"])
        ).any()
    )
    prev_gc_recent = bool(prev_link["GC_Any"].any())

    if vol_surge_today and gc_today:
        vol_gc_relation = "거래량 급증 + 골든크로스 동시 발생(최우선)"
        rel_bonus = 5.0
    elif gc_today and prev_vol_surge:
        vol_gc_relation = "거래량 급증 선행 후 1~3일 내 골든크로스"
        rel_bonus = 4.0
    elif vol_surge_today and prev_gc_recent:
        vol_gc_relation = "골든크로스 선행 후 1~3일 내 거래량 급증"
        rel_bonus = 4.0
    elif vol_surge_today and bool(row["Box_Breakout"]):
        vol_gc_relation = "수렴 상태 거래량 급증 + 박스권 상단 돌파"
        rel_bonus = 4.0
    else:
        vol_gc_relation = "직접 연계 미확인"
        rel_bonus = 0.0

    # 최근 3일 내 유효 거래량 배수
    max_recent_vol_ratio = max(vol_ratio, float(prev_link["Vol_Ratio_20"].max()) if len(prev_link) else 0.0)
    vol_confirmed = vol_surge_today or (prev_vol_surge and (gc_today or bool(row["Box_Breakout_Held"])))

    score_5 = 0.0
    if vol_ratio >= s5_cfg["strong_vol_surge_ratio"] and price_up_today:
        score_5 += 8.0
    elif vol_ratio >= s5_cfg["vol_surge_ratio"] and price_up_today:
        score_5 += 6.0
    elif prev_vol_surge:
        score_5 += 4.0

    if vol_5_20 >= 1.2:
        score_5 += 3.0
    elif vol_5_20 >= 1.0:
        score_5 += 1.5

    if up_dn_vol >= 1.25:
        score_5 += 2.5
    elif up_dn_vol >= 1.05:
        score_5 += 1.5

    if tval_eok >= u_cfg["min_trading_val_eok"]:
        score_5 += 1.5

    score_5 += rel_bonus
    score_5 = min(20.0, round(score_5, 1))

    # =========================================================
    # [6단계] 박스권 상단 돌파 및 상승 모멘텀 (배점 15점)
    # =========================================================
    box_breakout_today = bool(row["Box_Breakout"])
    box_breakout_held = bool(row["Box_Breakout_Held"])
    box_break_valid = box_breakout_today or box_breakout_held
    above_20_50 = (price > float(row["MA20"])) and (price > float(row["MA50"]))

    rsi_val = float(row["RSI14"])
    macd_val = float(row["MACD"])
    macd_sig = float(row["MACD_Sig"])
    macd_hist = float(row["MACD_Hist"])
    macd_prev_hist = float(prev_row["MACD_Hist"])
    macd_rising = (macd_val > macd_sig) or (macd_hist > macd_prev_hist)
    adx_val = float(row["ADX14"])
    adx_rising = adx_val > float(prev_row["ADX14"])

    score_6 = 0.0
    if box_breakout_today and vol_ratio >= 1.2:
        score_6 += 6.0
    elif box_breakout_held:
        score_6 += 7.0  # 돌파 후 1~3일 지지 확인 가산
    elif box_breakout_today:
        score_6 += 4.5

    if above_20_50:
        score_6 += 3.0
    if rsi_val >= 50.0 and rsi_val <= 72.0:
        score_6 += 2.0
    if macd_rising:
        score_6 += 1.5
    if adx_rising or adx_val >= 20.0:
        score_6 += 1.5
    score_6 = min(15.0, round(score_6, 1))

    # =========================================================
    # [7단계] 보조지표 보완 분석 (배점 5점) & 과열 감점
    # =========================================================
    bb_squeeze = float(row["BB_Width"]) < float(row["BB_Width_MA60"])
    bb_upper_break = price >= float(row["BB_Upper"]) * 0.99

    score_7 = 0.0
    if 50.0 <= rsi_val <= 68.0:
        score_7 += 1.5
    if macd_val > macd_sig:
        score_7 += 1.5
    if bb_squeeze or bb_upper_break:
        score_7 += 1.0
    if adx_val >= 18.0 and adx_rising:
        score_7 += 1.0
    score_7 = min(5.0, round(score_7, 1))

    # 단기 급등·과매수 감점
    disp_40 = float(row["Disparity_40"])
    ret_5d = float(row["Ret_5d"])
    penalty = 0.0
    penalty_notes = []
    if disp_40 > s8_cfg["max_ma40_disparity_pct"]:
        penalty += 12.0
        penalty_notes.append(f"40일선 과도이격(+{disp_40:.1f}%)")
    if ret_5d > 20.0:
        penalty += 8.0
        penalty_notes.append(f"5일 급등(+{ret_5d:.1f}%)")
    if rsi_val >= 76.0:
        penalty += 5.0
        penalty_notes.append(f"RSI 과매수({rsi_val:.1f})")

    total_score = round(max(0.0, min(100.0, score_1 + score_2 + score_3 + score_4 + score_5 + score_6 + score_7 - penalty)), 1)

    # =========================================================
    # [13번] 4대 신호 유형 분류 (D > C > B > A 우선순위)
    # =========================================================
    is_sig_a = cond_downtrend and cond_box and cond_convergence
    is_sig_b = is_sig_a and vol_surge_today
    is_sig_c = is_sig_a and has_recent_gc and (s20 > 0 and s50 > 0) and ma200_flat
    is_sig_d = is_sig_a and vol_confirmed and has_recent_gc and box_break_valid

    if is_sig_d:
        signal_type = "D. 최종 상승 전환 확인"
        sig_priority = 1
    elif is_sig_c:
        signal_type = "C. 골든크로스 신호"
        sig_priority = 2
    elif is_sig_b:
        signal_type = "B. 거래량 급증 신호"
        sig_priority = 3
    elif is_sig_a:
        signal_type = "A. 이동평균선 수렴 신호"
        sig_priority = 4
    else:
        signal_type = "미충족"
        sig_priority = 9

    # =========================================================
    # [8단계] 필수 통과 조건 및 최종 등급 산정
    # =========================================================
    mandatory_ok = cond_downtrend and cond_box and is_sig_a and (tval_eok >= u_cfg["min_trading_val_eok"] * 0.5)
    # 매매 후보 이상 필수 조건: 상승 움직임 + 거래량 증가 + (골든크로스 또는 박스권 돌파)
    candidate_gate = (price_up_today or box_breakout_held) and vol_confirmed and (has_recent_gc or box_break_valid)

    if (not mandatory_ok) or (total_score < s8_cfg["grade_watch"]):
        grade = "제외"
    elif not candidate_gate:
        grade = "관찰 종목"
    else:
        if total_score >= s8_cfg["grade_top"]:
            grade = "최우선 관심 종목"
        elif total_score >= s8_cfg["grade_candidate"]:
            grade = "매매 후보"
        else:
            grade = "관찰 종목"

    # =========================================================
    # [9단계] 매매 적합성 및 위험 관리 (ATR, 박스권, 저항선 기반)
    # =========================================================
    atr14 = float(row["ATR14"])
    recent_support = round(max(box_low, min(float(row["MA20"]), float(row["MA50"]))), 0)
    recent_res = float(row["Res_120"]) if pd.notna(row["Res_120"]) else box_high * 1.15

    stop_loss = min(price * 0.97, max(box_low * 0.985, price - 2.0 * atr14))
    if stop_loss >= price * 0.985:
        stop_loss = price * 0.94
    stop_loss = round(stop_loss, 0)

    box_height = max(box_high - box_low, atr14 * 2.0)
    target1 = round(max(box_high + box_height * 0.9, price + 2.8 * atr14, price * 1.07), 0)
    if recent_res > price * 1.08:
        target1 = round(min(target1, recent_res) if recent_res >= price + 1.8 * atr14 else target1, 0)
    target2 = round(max(box_high + box_height * 1.8, price + 4.8 * atr14, target1 * 1.08), 0)

    risk_pct = round(((price - stop_loss) / price) * 100.0, 2)
    reward1_pct = round(((target1 - price) / price) * 100.0, 2)
    reward2_pct = round(((target2 - price) / price) * 100.0, 2)
    rr_ratio = round(reward1_pct / max(risk_pct, 0.1), 2)

    hold_horizon = "2~4주 (단기 스윙)" if box_breakout_today and vol_ratio >= 2.0 else "4~12주 (추세 스윙)"

    # 신호 포착 근거 및 무효화 조건
    gc_summary = ", ".join(gc_types_found) if gc_types_found else "미발생"
    macd_txt = f"상승({macd_val:.0f}/{macd_sig:.0f})" if macd_val > macd_sig else f"하락({macd_val:.0f}/{macd_sig:.0f})"
    entry_plan = (
        f"진입: {int(price):,}원(돌파 확인) | 손절: {int(stop_loss):,}원(-{risk_pct}%) | "
        f"1차: {int(target1):,}원(+{reward1_pct}%) | 2차: {int(target2):,}원(+{reward2_pct}%) | "
        f"손익비: {rr_ratio:.2f} | 예상보유: {hold_horizon}"
    )
    invalidation = (
        f"종가 기준 박스권 하단({int(box_low):,}원) 이탈 또는 "
        f"50일선({int(row['MA50']):,}원) 하향 이탈 시 무효화"
    )
    if penalty_notes:
        invalidation += f" [주의: {', '.join(penalty_notes)}]"

    reason_text = (
        f"250일 중 {inv_days}일 역배열(1년 고점 대비 -{dd_1y:.1f}%), "
        f"{box_n}일 박스권({int(box_low):,}~{int(box_high):,}원, 폭 {box_width:.1f}%), "
        f"수렴상태 '{conv_state}'({pos_detail}, 20일 축소율 {red20:+.1f}%), "
        f"거래량 {vol_ratio:.2f}배({vol_gc_relation}), 골든크로스: {gc_summary}"
    )

    return {
        "기준일": df.index[idx].strftime("%Y-%m-%d"),
        "종목코드": meta["code"],
        "종목명 및 종목코드": f"{meta['name']} ({meta['code']})",
        "시장": meta["market"],
        "현재가": int(price),
        "최근 1년 고점 대비 하락률(%)": round(-dd_1y, 2),
        "역배열 유지 기간(일)": inv_days,
        "20일 이동평균선": round(float(row["MA20"]), 1),
        "40일 이동평균선": round(float(row["MA40"]), 1),
        "50일 이동평균선": round(float(row["MA50"]), 1),
        "200일 이동평균선": round(float(row["MA200"]), 1),
        "20-200 이격도(%)": round(g20_200, 2),
        "50-200 이격도(%)": round(g50_200, 2),
        "20-50 이격도(%)": round(g20_50, 2),
        "40-200 이격도(%)": round(g40_200, 2),
        "20일간 이격도 축소율(%)": round(red20, 2),
        "수렴 상태": conv_state,
        "이동평균선별 기울기": f"20일:{s20:+.2f}% / 40일:{s40:+.2f}% / 50일:{s50:+.2f}% / 200일:{s200:+.2f}%",
        "골든크로스 발생 여부 및 종류": gc_summary,
        "거래량 증가율(배)": round(vol_ratio, 2),
        "거래대금(억원)": round(tval_eok, 1),
        "박스권 상단 및 하단": f"{int(box_high):,} / {int(box_low):,}",
        "RSI 및 MACD": f"RSI {rsi_val:.1f} / MACD {macd_txt}",
        "40일선 이격도(%)": round(disp_40, 2),
        "예상 손익비": rr_ratio,
        "최종 점수": total_score,
        "최종 등급": grade,
        "신호 유형": signal_type,
        "신호 우선순위": sig_priority,
        "세부점수": f"하락{score_1}/박스{score_2}/수렴{score_3}/GC{score_4}/거래{score_5}/돌파{score_6}/보조{score_7}/감점-{penalty}",
        "신호 포착 근거": reason_text,
        "매매 진입 및 손절 기준": entry_plan,
        "상승 시나리오 무효화 조건": invalidation
    }
