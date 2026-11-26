import io
import copy
import json
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

try:
    import FinanceDataReader as fdr
    FDR_AVAILABLE = True
except ImportError:
    FDR_AVAILABLE = False

# 기존 4개 분석 모듈 연결 (수정 불필요)
from data_loader import (
    load_config, get_filtered_universe, fetch_stock_ohlcv,
    is_excluded_by_meta, validate_ohlcv
)
from indicators import compute_all_indicators
from signal_engine import evaluate_signal_at_index
from backtester import run_backtest_suite, build_strategy_masks

st.set_page_config(
    page_title="이평선 수렴·거래량 돌파 스윙 시스템",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    .block-container { padding-top: 1.2rem; padding-bottom: 2.5rem; }
    .signal-card {
        background: #ffffff;
        border-radius: 14px;
        padding: 16px 20px;
        margin-bottom: 12px;
        border: 1px solid #e5e8eb;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    }
    .b-top { background: #ffe3e3; color: #c92a2a; padding: 4px 10px; border-radius: 16px; font-weight: 800; font-size: 0.82rem; }
    .b-cand { background: #fff3bf; color: #d9480f; padding: 4px 10px; border-radius: 16px; font-weight: 800; font-size: 0.82rem; }
    .b-watch { background: #e7f5ff; color: #1864ab; padding: 4px 10px; border-radius: 16px; font-weight: 800; font-size: 0.82rem; }
    .b-excl { background: #f1f3f5; color: #868e96; padding: 4px 10px; border-radius: 16px; font-weight: 700; font-size: 0.82rem; }
    .b-sig { background: #191f28; color: #ffffff; padding: 4px 10px; border-radius: 16px; font-weight: 700; font-size: 0.8rem; margin-left: 6px; }
    .ew-box {
        background: #ffffff;
        border-radius: 14px;
        padding: 18px 20px;
        border: 2px solid #191f28;
        margin-bottom: 14px;
        box-shadow: 0 3px 12px rgba(0,0,0,0.05);
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# 엘리엇 파동 자동 탐지 & 현재 위치 판정 엔진
# ---------------------------------------------------------
def detect_zigzag_pivots(df: pd.DataFrame, pct_threshold: float = 7.0) -> list[dict]:
    """
    일봉 고가/저가 기준 ZigZag 변곡점(고점 H / 저점 L 교대) 추출.
    pct_threshold(%) 이상 반전 시 직전 고점/저점을 확정한다.
    """
    if df is None or len(df) < 15:
        return []

    pct = pct_threshold / 100.0
    highs = df["High"].values.astype(float)
    lows = df["Low"].values.astype(float)
    dates = df.index

    pivots = []
    direction = 0  # 1: 상승 국면(고점 탐색), -1: 하락 국면(저점 탐색)
    high_idx, high_p = 0, highs[0]
    low_idx, low_p = 0, lows[0]
    ext_idx, ext_p = 0, highs[0]

    for i in range(1, len(df)):
        h, l = highs[i], lows[i]
        if direction == 0:
            if h > high_p:
                high_idx, high_p = i, h
            if l < low_p:
                low_idx, low_p = i, l

            if h >= low_p * (1.0 + pct) and low_idx < i:
                pivots.append({"idx": low_idx, "date": dates[low_idx], "price": low_p, "kind": "L"})
                direction = 1
                ext_idx, ext_p = i, h
            elif l <= high_p * (1.0 - pct) and high_idx < i:
                pivots.append({"idx": high_idx, "date": dates[high_idx], "price": high_p, "kind": "H"})
                direction = -1
                ext_idx, ext_p = i, l
        elif direction == 1:
            if h >= ext_p:
                ext_idx, ext_p = i, h
            elif l <= ext_p * (1.0 - pct):
                pivots.append({"idx": ext_idx, "date": dates[ext_idx], "price": ext_p, "kind": "H"})
                direction = -1
                ext_idx, ext_p = i, l
        else:
            if l <= ext_p:
                ext_idx, ext_p = i, l
            elif h >= ext_p * (1.0 + pct):
                pivots.append({"idx": ext_idx, "date": dates[ext_idx], "price": ext_p, "kind": "L"})
                direction = 1
                ext_idx, ext_p = i, h

    if pivots and ext_idx != pivots[-1]["idx"]:
        pivots.append({
            "idx": ext_idx,
            "date": dates[ext_idx],
            "price": ext_p,
            "kind": "H" if direction == 1 else "L"
        })

    return pivots


def _is_valid_bullish_impulse_5(p0, p1, p2, p3, p4, p5) -> tuple[bool, bool]:
    """상승 충격파(0-1-2-3-4-5) 핵심 규칙 검증 -> (유효 여부, 엄격 4파 미침범 여부)"""
    if not (p0["kind"] == "L" and p1["kind"] == "H" and p2["kind"] == "L" and
            p3["kind"] == "H" and p4["kind"] == "L" and p5["kind"] == "H"):
        return False, False
    w1 = p1["price"] - p0["price"]
    w3 = p3["price"] - p2["price"]
    w5 = p5["price"] - p4["price"]
    if w1 <= 0 or w3 <= 0 or w5 <= 0:
        return False, False
    # 규칙 1: 2파 저점은 0파(시작점) 이탈 불가
    if p2["price"] <= p0["price"]:
        return False, False
    # 규칙 2: 3파 고점은 1파 고점 돌파 & 3파는 1·3·5파 중 최단 불가
    if p3["price"] <= p1["price"]:
        return False, False
    if w3 < w1 and w3 < w5:
        return False, False
    # 규칙 3: 4파 저점은 1파 고점 침범 금지 (변동성 큰 국내장 특성상 3% 허용 판별)
    if p4["price"] < p1["price"] * 0.97 or p4["price"] <= p2["price"]:
        return False, False
    if p5["price"] < p3["price"] * 0.98:
        return False, False
    strict_no_overlap = p4["price"] >= p1["price"]
    return True, strict_no_overlap


def analyze_elliott_wave(df: pd.DataFrame, pct_threshold: float = 7.0) -> dict:
    """
    ZigZag 변곡점 기반 엘리엇 파동(상승 1~5파 / 조정 A~C파) 자동 매핑 및 현재 지점 판정.
    """
    pivots = detect_zigzag_pivots(df, pct_threshold)
    cur_price = float(df["Close"].iloc[-1])
    cur_date = df.index[-1]

    if len(pivots) < 2:
        return {
            "ok": False,
            "message": "선택한 민감도(%)에서 뚜렷한 파동 변곡점이 부족합니다. 상단에서 '파동 민감도(%)'를 낮추거나 분석 기간을 늘려보세요.",
            "pivots": pivots,
        }

    labeled = []
    stage_title = ""
    stage_badge_color = "#191f28"
    stage_desc = ""
    action_tip = ""
    invalidation_price = None
    invalidation_label = ""
    fib_levels = []  # list of (label, price, color)
    rule_notes = []
    matched = False

    n = len(pivots)

    # ---------------------------------------------------------
    # 1) 완료된 5파(0-1-2-3-4-5) 이후 조정파(A / B / C) 진행 중 탐색
    # ---------------------------------------------------------
    for tail_corr in [3, 2, 1, 0]:
        start_i = n - (6 + tail_corr)
        if start_i < 0:
            continue
        seg = pivots[start_i:]
        ok5, strict4 = _is_valid_bullish_impulse_5(*seg[:6])
        if not ok5:
            continue

        p0, p1, p2, p3, p4, p5 = seg[:6]
        base_labels = ["0", "1", "2", "3", "4", "5"]
        labeled = [dict(p, label=lb, group="impulse") for p, lb in zip(seg[:6], base_labels)]
        impulse_span = max(p5["price"] - p0["price"], 1e-6)
        rule_notes = [
            f"2파 저점({int(p2['price']):,}원) > 0파 시작점({int(p0['price']):,}원) 충족",
            f"3파 길이({int(p3['price']-p2['price']):,}원) 최단 금지 규칙 충족",
            "4파 1파 고점 미침범 엄격 충족" if strict4 else "4파 1파 고점 근접(허용 범위 내 쐐기형)"
        ]

        if tail_corr == 0:
            if cur_price < p5["price"] * 0.98:
                stage_title = "📉 5파 완료 후 조정 A파 진행 중"
                stage_badge_color = "#1d4ed8"
                stage_desc = f"상승 5파 고점({int(p5['price']):,}원) 형성 후 첫 번째 하락 조정(A파)이 진행되는 구간입니다."
                action_tip = "추격 매수를 피하고 전체 5파 상승폭의 38.2%~50% 되돌림 지지 여부를 관찰하는 구간입니다."
                invalidation_price = p5["price"]
                invalidation_label = "5파 고점 재돌파 시 연장파 전환"
                fib_levels = [
                    ("전체 5파 23.6% 되돌림", p5["price"] - impulse_span * 0.236, "#64748b"),
                    ("전체 5파 38.2% 지지", p5["price"] - impulse_span * 0.382, "#0284c7"),
                    ("전체 5파 50.0% 지지", p5["price"] - impulse_span * 0.500, "#1d4ed8"),
                ]
            else:
                stage_title = "🏔️ 상승 5파 고점 부근 (추세 말기 경계)"
                stage_badge_color = "#d97706"
                stage_desc = f"상승 5파({int(p5['price']):,}원)가 정점에 도달했거나 연장 진행 중인 구간입니다."
                action_tip = "신규 스윙 진입보다는 보유 물량 분할 익절 및 4파 저점 이탈 경계가 유리한 구간입니다."
                invalidation_price = p4["price"]
                invalidation_label = "4파 저점 이탈 시 5파 종료 확정"
                fib_levels = [
                    ("5파 고점(저항)", p5["price"], "#e11d48"),
                    ("4파 저점(지지)", p4["price"], "#0284c7"),
                ]
            matched = True
            break

        elif tail_corr == 1:
            pA = seg[6]
            labeled.append(dict(pA, label="A", group="correction"))
            a_span = max(p5["price"] - pA["price"], 1e-6)
            stage_title = "⛅ 조정 A파 완료 → B파 반등 진행 중"
            stage_badge_color = "#d97706"
            stage_desc = f"5파 고점({int(p5['price']):,}원)에서 A파({int(pA['price']):,}원) 하락 후 되돌림 반등(B파)이 나타나는 구간입니다."
            action_tip = "B파 반등은 A파 하락폭의 50~61.8% 부근에서 저항을 받고 C파 하락으로 이어질 수 있어 주의가 필요합니다."
            invalidation_price = pA["price"]
            invalidation_label = "A파 저점 이탈 시 C파 하락 본격화"
            fib_levels = [
                ("A파 38.2% 반등 저항", pA["price"] + a_span * 0.382, "#f59f00"),
                ("A파 50.0% 반등 저항", pA["price"] + a_span * 0.500, "#ea580c"),
                ("A파 61.8% 반등 저항", pA["price"] + a_span * 0.618, "#dc2626"),
            ]
            matched = True
            break

        elif tail_corr == 2:
            pA, pB = seg[6], seg[7]
            labeled.append(dict(pA, label="A", group="correction"))
            labeled.append(dict(pB, label="B", group="correction"))
            a_span = max(p5["price"] - pA["price"], 1e-6)
            stage_title = "🌧️ 조정 C파 하락 진행 중 (최종 바닥 탐색)"
            stage_badge_color = "#1d4ed8"
            stage_desc = f"B파 반등 고점({int(pB['price']):,}원) 이후 마지막 조정 파동인 C파가 진행 중입니다."
            action_tip = "C파가 A파 길이의 1.0배 부근에 도달하며 거래량과 이평선 수렴이 나타나는지 바닥 신호를 체크하세요."
            invalidation_price = pB["price"]
            invalidation_label = "B파 고점 돌파 시 조정 종료 신호"
            fib_levels = [
                ("C파 0.618배 목표", max(pB["price"] - a_span * 0.618, 1), "#0284c7"),
                ("C파 1.0배 대칭 바닥", max(pB["price"] - a_span * 1.0, 1), "#1d4ed8"),
            ]
            matched = True
            break

        elif tail_corr == 3:
            pA, pB, pC = seg[6], seg[7], seg[8]
            labeled.append(dict(pA, label="A", group="correction"))
            labeled.append(dict(pB, label="B", group="correction"))
            labeled.append(dict(pC, label="C", group="correction"))
            stage_title = "🌱 ABC 조정 완료 → 새로운 상승 1파 전환 시도"
            stage_badge_color = "#16a34a"
            stage_desc = f"C파 저점({int(pC['price']):,}원)을 바닥으로 다진 뒤 새로운 상승 1파를 시작하려는 초기 전환 구간입니다."
            action_tip = "C파 저점을 손절 기준으로 잡고, B파 고점 돌파 및 거래량 동반 여부를 확인하는 초기 스윙 관찰 구간입니다."
            invalidation_price = pC["price"]
            invalidation_label = "C파 저점 이탈 시 하락 연장"
            fib_levels = [
                ("1차 돌파 기준 (B파 고점)", pB["price"], "#16a34a"),
                ("바닥 지지선 (C파 저점)", pC["price"], "#1d4ed8"),
            ]
            matched = True
            break

    # ---------------------------------------------------------
    # 2) 진행 중인 상승 충격파 (5파 / 4파 / 3파 / 2파) 탐색
    # ---------------------------------------------------------
    if not matched and n >= 5:
        p0, p1, p2, p3, p4 = pivots[-5:]
        if (p0["kind"] == "L" and p1["kind"] == "H" and p2["kind"] == "L" and
                p3["kind"] == "H" and p4["kind"] == "L" and
                p2["price"] > p0["price"] and p3["price"] > p1["price"] and
                p4["price"] >= p1["price"] * 0.97 and p4["price"] < p3["price"] and
                cur_price >= p4["price"]):
            labeled = [dict(p, label=lb, group="impulse") for p, lb in zip([p0, p1, p2, p3, p4], ["0", "1", "2", "3", "4"])]
            w1 = max(p1["price"] - p0["price"], 1e-6)
            w03 = max(p3["price"] - p0["price"], 1e-6)
            broke_3 = cur_price > p3["price"]
            stage_title = "🔥 상승 5파 진행 중 (3파 고점 돌파)" if broke_3 else "📈 상승 5파 진행 중 (3파 고점 돌파 시도)"
            stage_badge_color = "#ea580c"
            stage_desc = f"4파 조정 저점({int(p4['price']):,}원) 지지 후 마지막 상승 파동인 5파가 전개되고 있습니다."
            action_tip = "상승 추세의 후반부이므로 1파 길이 대칭(1.0배) 또는 0~3파의 0.618배 구간 도달 시 분할 익절을 고려하는 구간입니다."
            invalidation_price = p4["price"]
            invalidation_label = "4파 저점 이탈 시 상승 5파 무효화"
            fib_levels = [
                ("3파 고점 (돌파 확인)", p3["price"], "#f59f00"),
                ("5파 1차 목표 (1파 1.0배)", p4["price"] + w1, "#ea580c"),
                ("5파 2차 목표 (0~3파 0.618배)", p4["price"] + w03 * 0.618, "#dc2626"),
            ]
            rule_notes = [
                f"2파({int(p2['price']):,}원) > 0파({int(p0['price']):,}원) 규칙 충족",
                f"3파 고점({int(p3['price']):,}원) > 1파 고점({int(p1['price']):,}원) 충족",
                "4파 1파 고점 미침범 엄격 충족" if p4["price"] >= p1["price"] else "4파 1파 고점 근접 지지"
            ]
            matched = True

    if not matched and n >= 4:
        p0, p1, p2, p3 = pivots[-4:]
        if (p0["kind"] == "L" and p1["kind"] == "H" and p2["kind"] == "L" and p3["kind"] == "H" and
                p2["price"] > p0["price"] and p3["price"] > p1["price"] and
                cur_price <= p3["price"] and cur_price >= p1["price"] * 0.96):
            labeled = [dict(p, label=lb, group="impulse") for p, lb in zip([p0, p1, p2, p3], ["0", "1", "2", "3"])]
            w3 = max(p3["price"] - p2["price"], 1e-6)
            stage_title = "⏸️ 상승 4파 조정 진행 중 (5파 대기)"
            stage_badge_color = "#d97706"
            stage_desc = f"강한 3파 상승({int(p2['price']):,}원 → {int(p3['price']):,}원) 이후 숨고르기(4파 조정)가 진행 중입니다."
            action_tip = "3파 상승폭의 23.6%~38.2% 되돌림 지지와 1파 고점(무효화선) 방어 여부를 확인하면 5파 단기 스윙 타점이 됩니다."
            invalidation_price = p1["price"]
            invalidation_label = "1파 고점 이탈 시 충격파 4파 규칙 위반"
            fib_levels = [
                ("3파 23.6% 지지", p3["price"] - w3 * 0.236, "#16a34a"),
                ("3파 38.2% 핵심 지지", p3["price"] - w3 * 0.382, "#0284c7"),
                ("1파 고점 (절대 방어선)", p1["price"], "#dc2626"),
            ]
            rule_notes = [
                f"2파({int(p2['price']):,}원) > 0파({int(p0['price']):,}원) 규칙 충족",
                f"3파 고점({int(p3['price']):,}원) > 1파 고점({int(p1['price']):,}원) 돌파 확인",
                f"현재가 1파 고점({int(p1['price']):,}원) 상단 유지 중"
            ]
            matched = True

    if not matched and n >= 3:
        p0, p1, p2 = pivots[-3:]
        if (p0["kind"] == "L" and p1["kind"] == "H" and p2["kind"] == "L" and
                p2["price"] > p0["price"] and p2["price"] < p1["price"] and cur_price >= p2["price"]):
            labeled = [dict(p, label=lb, group="impulse") for p, lb in zip([p0, p1, p2], ["0", "1", "2"])]
            w1 = max(p1["price"] - p0["price"], 1e-6)
            ret_ratio = round(((p1["price"] - p2["price"]) / w1) * 100.0, 1)
            broke_1 = cur_price > p1["price"]
            stage_title = "🚀 상승 3파 본격 진행 중 (1파 고점 돌파)" if broke_1 else "🔥 상승 3파 초입 (1파 고점 돌파 시도)"
            stage_badge_color = "#e11d48"
            stage_desc = f"2파 조정({ret_ratio}% 되돌림, {int(p2['price']):,}원)을 마치고 가장 강한 추세가 나오는 3파 구간에 진입했습니다."
            action_tip = "스윙 트레이딩의 핵심 구간입니다. 2파 저점을 손절선으로 두고 1파 대비 1.0배~1.618배 확장 목표가를 추적하세요."
            invalidation_price = p2["price"]
            invalidation_label = "2파 저점 이탈 시 상승 3파 무효화"
            fib_levels = [
                ("1파 고점 (돌파 확인선)", p1["price"], "#f59f00"),
                ("3파 1차 목표 (1.0배)", p2["price"] + w1 * 1.0, "#ea580c"),
                ("3파 핵심 목표 (1.618배)", p2["price"] + w1 * 1.618, "#dc2626"),
            ]
            rule_notes = [
                f"2파 저점({int(p2['price']):,}원)이 0파({int(p0['price']):,}원) 위에서 지지됨 (되돌림 {ret_ratio}%)",
                "1파 고점 상향 돌파로 3파 추세 확인" if broke_1 else f"1파 고점({int(p1['price']):,}원) 돌파 여부 주시 구간"
            ]
            matched = True

    # ---------------------------------------------------------
    # 3) 하락 A-B-C 조정 구조 탐색 (장기 하락 후 바닥 전환 종목 포착)
    # ---------------------------------------------------------
    if not matched and n >= 4:
        p0, pA, pB, pC = pivots[-4:]
        if (p0["kind"] == "H" and pA["kind"] == "L" and pB["kind"] == "H" and pC["kind"] == "L" and
                pA["price"] < p0["price"] and pB["price"] < p0["price"] and pB["price"] > pA["price"] and
                pC["price"] <= pA["price"] * 1.03 and cur_price >= pC["price"]):
            labeled = [dict(p, label=lb, group="correction") for p, lb in zip([p0, pA, pB, pC], ["0", "A", "B", "C"])]
            a_span = max(p0["price"] - pA["price"], 1e-6)
            c_ratio = round(((pB["price"] - pC["price"]) / a_span) * 100.0, 1)
            stage_title = "🌱 하락 ABC 완료 → 상승 1파 전환 초기"
            stage_badge_color = "#16a34a"
            stage_desc = f"고점({int(p0['price']):,}원) 이후 하락 A-B-C(C파={c_ratio}% 비율, 저점 {int(pC['price']):,}원)를 마치고 반등 전환을 시도 중입니다."
            action_tip = "장기 하락 후 바닥 다지기·이평선 수렴 전략과 가장 잘 맞는 자리입니다. B파 고점 돌파와 거래량 급증을 함께 확인하세요."
            invalidation_price = pC["price"]
            invalidation_label = "C파 저점 이탈 시 하락 추세 연장"
            fib_levels = [
                ("1차 저항 (B파 고점)", pB["price"], "#16a34a"),
                ("하락폭 38.2% 회복선", pC["price"] + (p0["price"] - pC["price"]) * 0.382, "#f59f00"),
                ("하락폭 50.0% 회복선", pC["price"] + (p0["price"] - pC["price"]) * 0.500, "#ea580c"),
            ]
            rule_notes = [
                f"고점 대비 하락 3파(A→B→C) 구조 완성 (C/A 비율 {c_ratio}%)",
                f"C파 저점({int(pC['price']):,}원) 방어하며 상승 전환 진행 중"
            ]
            matched = True

    if not matched and n >= 3:
        p0, pA, pB = pivots[-3:]
        if (p0["kind"] == "H" and pA["kind"] == "L" and pB["kind"] == "H" and
                pA["price"] < p0["price"] and pB["price"] < p0["price"] and pB["price"] > pA["price"] and
                cur_price <= pB["price"]):
            labeled = [dict(p, label=lb, group="correction") for p, lb in zip([p0, pA, pB], ["0", "A", "B"])]
            a_span = max(p0["price"] - pA["price"], 1e-6)
            b_ret = round(((pB["price"] - pA["price"]) / a_span) * 100.0, 1)
            stage_title = "🌧️ 하락 C파 진행 중 (조정 마무리 탐색)"
            stage_badge_color = "#1d4ed8"
            stage_desc = f"A파 하락 후 B파({b_ret}% 반등, {int(pB['price']):,}원)에서 저항을 받고 마지막 하락 파동인 C파가 진행 중입니다."
            action_tip = "성급한 저점 매수보다 A파 저점 부근 또는 C파 1.0배 대칭 구간에서 거래량 동반 양봉·박스권 지지를 기다리는 것이 안전합니다."
            invalidation_price = pB["price"]
            invalidation_label = "B파 고점 상향 돌파 시 하락 시나리오 종료"
            fib_levels = [
                ("A파 저점 (지지 테스트)", pA["price"], "#0284c7"),
                ("C파 1.0배 대칭 예상 바닥", max(pB["price"] - a_span * 1.0, 1), "#1d4ed8"),
            ]
            rule_notes = [
                f"고점({int(p0['price']):,}원) → A파({int(pA['price']):,}원) → B파({int(pB['price']):,}원, {b_ret}% 반등)",
                "현재 B파 고점 아래에서 C파 하락 전개 중"
            ]
            matched = True

    if not matched and n >= 2:
        p0, p1 = pivots[-2:]
        if p0["kind"] == "L" and p1["kind"] == "H" and p1["price"] > p0["price"] and p0["price"] < cur_price <= p1["price"]:
            labeled = [dict(p, label=lb, group="impulse") for p, lb in zip([p0, p1], ["0", "1"])]
            w1 = max(p1["price"] - p0["price"], 1e-6)
            cur_ret = round(((p1["price"] - cur_price) / w1) * 100.0, 1)
            stage_title = "🎯 상승 2파 조정 진행 중 (3파 매수 준비)"
            stage_badge_color = "#0284c7"
            stage_desc = f"바닥({int(p0['price']):,}원)에서 첫 상승 1파({int(p1['price']):,}원)가 나온 뒤 현재 {cur_ret}% 되돌림 조정(2파)이 진행 중입니다."
            action_tip = "1파 상승폭의 38.2%~61.8% 구간에서 거래량이 줄며 지지받으면 강력한 3파를 노리는 최적의 스윙 준비 자리입니다."
            invalidation_price = p0["price"]
            invalidation_label = "0파 바닥 이탈 시 상승 1파 무효화"
            fib_levels = [
                ("1파 38.2% 눌림목", p1["price"] - w1 * 0.382, "#16a34a"),
                ("1파 50.0% 눌림목", p1["price"] - w1 * 0.500, "#0284c7"),
                ("1파 61.8% 핵심 지지", p1["price"] - w1 * 0.618, "#1d4ed8"),
            ]
            rule_notes = [
                f"첫 상승 1파(+{round((w1/p0['price'])*100, 1)}%) 확인",
                f"현재 1파 대비 {cur_ret}% 되돌림 진행 중 (0파 이탈 없음)"
            ]
            matched = True

    # ---------------------------------------------------------
    # 4) 표준 5파/3파 외 복합 횡보·삼각수렴 패턴 폴백
    # ---------------------------------------------------------
    if not matched:
        recent_pivs = pivots[-min(len(pivots), 6):]
        labeled = [dict(p, label=f"{p['kind']}{i+1}", group="complex") for i, p in enumerate(recent_pivs)]
        last_p = recent_pivs[-1]
        recent_high = max(p["price"] for p in recent_pivs)
        recent_low = min(p["price"] for p in recent_pivs)
        if cur_price >= last_p["price"]:
            stage_title = "📦 복합 조정 후 단기 반등 파동 진행 중"
            stage_badge_color = "#7c3aed"
            stage_desc = f"최근 변곡점({int(last_p['price']):,}원) 이후 상승 방향 파동이 전개 중이나, 아직 전형적인 5파/ABC 구조가 완성 전인 횡보·복합 구간입니다."
        else:
            stage_title = "📦 복합 횡보 / 삼각수렴 조정 진행 중"
            stage_badge_color = "#64748b"
            stage_desc = f"최근 고점({int(recent_high):,}원)과 저점({int(recent_low):,}원) 사이에서 방향을 모색하는 복합 조정 구간입니다."
        action_tip = "파동 민감도(%)를 1~2% 조절해 대파동/소파동 관점을 함께 확인하고, 박스권 상단 돌파 및 거래량 급증 시점을 기다리세요."
        invalidation_price = recent_low
        invalidation_label = "최근 주요 저점 지지선"
        fib_levels = [
            ("최근 주요 고점 (상단 돌파선)", recent_high, "#e11d48"),
            ("중심 균형가 (50%)", (recent_high + recent_low) / 2.0, "#f59f00"),
            ("최근 주요 저점 (하단 지지선)", recent_low, "#1d4ed8"),
        ]
        rule_notes = [
            "단일 충격파보다는 복합 조정(삼각수렴/박스권) 형태에 가까움",
            "민감도(%) 조정 시 상위/하위 파동 구조 확인 가능"
        ]

    # 파동별 구간 요약 테이블 생성
    wave_rows = []
    for i in range(len(labeled) - 1):
        a, b = labeled[i], labeled[i + 1]
        chg = ((b["price"] - a["price"]) / max(a["price"], 1e-6)) * 100.0
        days = int(b["idx"] - a["idx"])
        wave_rows.append({
            "파동 구간": f"{a['label']} → {b['label']}",
            "시작일": a["date"].strftime("%Y-%m-%d"),
            "종료일": b["date"].strftime("%Y-%m-%d"),
            "시작가": f"{int(a['price']):,}원",
            "종료가": f"{int(b['price']):,}원",
            "등락률(%)": round(chg, 2),
            "소요 거래일": f"{days}일"
        })

    # 마지막 확정 변곡점 → 현재 위치(진행 중 레그) 추가
    last_lbl = labeled[-1]
    cur_chg = ((cur_price - last_lbl["price"]) / max(last_lbl["price"], 1e-6)) * 100.0
    cur_days = int((len(df) - 1) - last_lbl["idx"])
    wave_rows.append({
        "파동 구간": f"{last_lbl['label']} → 📍현재(진행중)",
        "시작일": last_lbl["date"].strftime("%Y-%m-%d"),
        "종료일": cur_date.strftime("%Y-%m-%d"),
        "시작가": f"{int(last_lbl['price']):,}원",
        "종료가": f"{int(cur_price):,}원",
        "등락률(%)": round(cur_chg, 2),
        "소요 거래일": f"{cur_days}일"
    })

    return {
        "ok": True,
        "pivots": pivots,
        "labeled": labeled,
        "cur_date": cur_date,
        "cur_price": cur_price,
        "stage_title": stage_title,
        "stage_badge_color": stage_badge_color,
        "stage_desc": stage_desc,
        "action_tip": action_tip,
        "invalidation_price": invalidation_price,
        "invalidation_label": invalidation_label,
        "fib_levels": fib_levels,
        "rule_notes": rule_notes,
        "wave_table": pd.DataFrame(wave_rows),
    }


# ---------------------------------------------------------
# 데이터 수집 및 종목 검색 마스터 헬퍼
# ---------------------------------------------------------
def fetch_naver_universe_fallback(min_mcap_eok: int = 0, max_per_market: int = 400) -> pd.DataFrame:
    rows = []
    seen = set()
    for api_mkt, kor_mkt in [("KOSPI", "코스피"), ("KOSDAQ", "코스닥")]:
        page = 1
        cnt = 0
        while cnt < max_per_market and page <= 12:
            url = f"https://m.stock.naver.com/api/stocks/marketValue/{api_mkt}?page={page}&pageSize=100"
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            try:
                with urllib.request.urlopen(req, timeout=4) as res:
                    items = json.loads(res.read().decode("utf-8")).get("stocks", [])
                if not items:
                    break
                for it in items:
                    code = str(it.get("itemCode", "")).strip()
                    name = str(it.get("stockName", "")).strip()
                    if not code or code in seen or is_excluded_by_meta(code, name, ""):
                        continue
                    stop_info = it.get("tradeStopType")
                    if isinstance(stop_info, dict) and str(stop_info.get("code", "1")) not in ("1", "", "None"):
                        continue
                    mcap_raw = str(it.get("marketValue", "0")).replace(",", "").strip()
                    mcap_eok = float(mcap_raw) if mcap_raw.replace(".", "", 1).isdigit() else 0.0
                    if min_mcap_eok > 0 and mcap_eok > 0 and mcap_eok < min_mcap_eok:
                        continue
                    seen.add(code)
                    rows.append({
                        "code": code.zfill(6),
                        "name": name,
                        "market": kor_mkt,
                        "mcap_eok": int(mcap_eok),
                        "delisted": False
                    })
                    cnt += 1
                    if cnt >= max_per_market:
                        break
                page += 1
            except Exception:
                break
    return pd.DataFrame(rows)


def fetch_naver_ohlcv_fallback(code: str, min_days: int, count: int = 750) -> pd.DataFrame | None:
    try:
        url = f"https://fchart.stock.naver.com/sise.nhn?symbol={code}&timeframe=day&count={count}&requestType=0"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=5) as res:
            xml_data = res.read().decode("euc-kr", errors="ignore")
        root = ET.fromstring(xml_data)
        rows = []
        for item in root.findall(".//item"):
            parts = item.attrib.get("data", "").split("|")
            if len(parts) >= 6:
                dt, o, h, l, c, v = parts[:6]
                rows.append({
                    "Date": pd.to_datetime(dt, format="%Y%m%d"),
                    "Open": float(o), "High": float(h), "Low": float(l),
                    "Close": float(c), "Volume": float(v)
                })
        if not rows:
            return None
        df = pd.DataFrame(rows).set_index("Date")
        return validate_ohlcv(df, min_days)
    except Exception:
        return None


def safe_get_universe(cfg: dict) -> pd.DataFrame:
    df_u = get_filtered_universe(cfg, include_delisted=False)
    if df_u is None or df_u.empty:
        df_u = fetch_naver_universe_fallback(cfg["universe"]["min_mcap_eok"], max_per_market=400)
    else:
        df_u = df_u.sort_values("mcap_eok", ascending=False).reset_index(drop=True)
    return df_u


@st.cache_data(ttl=3600)
def get_searchable_stock_master() -> pd.DataFrame:
    """종목 검색창용 코스피·코스닥 전체 보통주 목록"""
    try:
        if FDR_AVAILABLE:
            listing = fdr.StockListing("KRX")
            if listing is not None and not listing.empty:
                if "Symbol" in listing.columns and "Code" not in listing.columns:
                    listing = listing.rename(columns={"Symbol": "Code"})
                listing["Market_Clean"] = listing["Market"].astype(str).str.upper().replace({"KOSDAQ GLOBAL": "KOSDAQ"})
                df = listing[listing["Market_Clean"].isin(["KOSPI", "KOSDAQ"])].copy()
                dept_s = df["Dept"].fillna("") if "Dept" in df.columns else pd.Series("", index=df.index)
                keep = [
                    not is_excluded_by_meta(str(c).strip(), str(n).strip(), str(d))
                    for c, n, d in zip(df["Code"], df["Name"], dept_s)
                ]
                df = df[keep].copy()
                marcap = pd.to_numeric(df["Marcap"], errors="coerce").fillna(0) if "Marcap" in df.columns else 0
                out = pd.DataFrame({
                    "code": df["Code"].astype(str).str.zfill(6),
                    "name": df["Name"].astype(str).str.strip(),
                    "market": df["Market_Clean"].map({"KOSPI": "코스피", "KOSDAQ": "코스닥"}),
                    "mcap_eok": (marcap / 1e8).round(0).astype(int) if isinstance(marcap, pd.Series) else 0
                }).sort_values("mcap_eok", ascending=False).drop_duplicates(subset=["code"]).reset_index(drop=True)
                if not out.empty:
                    return out
    except Exception:
        pass
    return fetch_naver_universe_fallback(min_mcap_eok=0, max_per_market=600)


def safe_fetch_ohlcv(code: str, cfg: dict) -> pd.DataFrame | None:
    df = fetch_stock_ohlcv(code, cfg)
    if df is None:
        df = fetch_naver_ohlcv_fallback(code, cfg["universe"]["min_trading_days"], count=750)
    return df


@st.cache_data(ttl=900)
def fetch_ohlcv_for_single_stock(code: str, min_days: int = 120, count: int = 800) -> pd.DataFrame | None:
    """엘리엇 파동 개별 종목 즉시 조회용 (최근 3년 일봉)"""
    if FDR_AVAILABLE:
        try:
            end_dt = datetime.today()
            start_dt = end_dt - timedelta(days=1200)
            raw = fdr.DataReader(code, start_dt.strftime("%Y-%m-%d"), end_dt.strftime("%Y-%m-%d"))
            clean = validate_ohlcv(raw, min_days)
            if clean is not None:
                return clean
        except Exception:
            pass
    return fetch_naver_ohlcv_fallback(code, min_days, count=count)


def fetch_ohlcv_for_backtest(code: str, cfg: dict) -> pd.DataFrame | None:
    """개별 종목 검색 백테스트 전용 (5년치 수집으로 3년 순수 테스트 구간 확보)"""
    min_days = cfg["universe"]["min_trading_days"]
    if FDR_AVAILABLE:
        try:
            end_dt = datetime.today()
            start_dt = end_dt - timedelta(days=1850)
            raw = fdr.DataReader(code, start_dt.strftime("%Y-%m-%d"), end_dt.strftime("%Y-%m-%d"))
            clean = validate_ohlcv(raw, min_days)
            if clean is not None and len(clean) >= 700:
                return clean
        except Exception:
            pass
    return fetch_naver_ohlcv_fallback(code, min_days, count=1250)


def build_excel_bytes(passed_df: pd.DataFrame, bt_summary: pd.DataFrame, bt_trades: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        passed_df.to_excel(writer, sheet_name="최종_선별종목", index=False)
        for prefix, sname in [
            ("D.", "D_최종상승전환"),
            ("C.", "C_골든크로스"),
            ("B.", "B_거래량급증"),
            ("A.", "A_이평선수렴"),
        ]:
            sub = passed_df[passed_df["신호 유형"].str.startswith(prefix)] if not passed_df.empty else pd.DataFrame()
            sub.to_excel(writer, sheet_name=sname, index=False)
        if bt_summary is not None and not bt_summary.empty:
            bt_summary.to_excel(writer, sheet_name="백테스트_요약", index=False)
        if bt_trades is not None and not bt_trades.empty:
            bt_trades.head(5000).to_excel(writer, sheet_name="백테스트_거래내역", index=False)
    return buf.getvalue()


# ---------------------------------------------------------
# 기본 설정 로드 및 세션 상태 초기화
# ---------------------------------------------------------
try:
    BASE_CFG = load_config("config.json")
except Exception:
    st.error("config.json 파일을 찾을 수 없습니다. 깃허브에 config.json을 함께 올려주세요.")
    st.stop()

defaults = {
    "scan_done": False,
    "all_df": pd.DataFrame(),
    "passed_df": pd.DataFrame(),
    "indicator_map": {},
    "updated_at": "",
    "total_target": 0,
    "used_cfg": copy.deepcopy(BASE_CFG),
    "bt_summary": pd.DataFrame(),
    "bt_trades": pd.DataFrame(),
    "search_bt_summary": pd.DataFrame(),
    "search_bt_trades": pd.DataFrame(),
    "search_bt_ind_map": {},
    "search_bt_names": {},
    "search_bt_cfg": copy.deepcopy(BASE_CFG),
    "search_bt_done": False,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ---------------------------------------------------------
# 상단 헤더 및 전략 설정 패널
# ---------------------------------------------------------
st.title("📈 이평선 수렴 · 거래량 돌파 스윙 포착 시스템")
st.caption("코스피·코스닥 3년 일봉 분석 · 이평선 수렴 · 거래량 돌파 · 엘리엇 파동 자동 매핑 · 전략 A/B/C 백테스트")

top_c1, top_c2, top_c3 = st.columns([1.4, 1.4, 1.2])
with top_c1:
    scan_preset = st.selectbox(
        "🔍 전체 시장 스캔 범위 (시가총액 상위 순)",
        [
            "빠른 스캔: 상위 150종목 (약 15~25초)",
            "표준 스캔: 상위 300종목 (약 35~50초)",
            "정밀 스캔: 상위 600종목 (약 1분 30초)",
            "전체 스캔: 조건 통과 전 종목 (약 3~5분)"
        ],
        index=0
    )
with top_c2:
    custom_codes = st.text_input("➕ 스캔에 포함할 종목코드 (선택, 쉼표 구분)", placeholder="예: 005930, 035720, 011170")
with top_c3:
    st.write("")
    run_btn = st.button("🚀 전체 시장 스캔 시작", type="primary", use_container_width=True)

with st.expander("⚙️ 전략·임계값 세부 설정 (전체 스캔 및 백테스트 공통 적용)", expanded=False):
    cfg = copy.deepcopy(BASE_CFG)
    p1, p2, p3, p4 = st.columns(4)
    with p1:
        cfg["universe"]["min_mcap_eok"] = st.number_input("최소 시가총액(억원)", 0, 50000, int(BASE_CFG["universe"]["min_mcap_eok"]), step=100)
        cfg["universe"]["min_trading_val_eok"] = st.number_input("최소 거래대금(억원)", 1, 1000, int(BASE_CFG["universe"]["min_trading_val_eok"]), step=5)
        cfg["universe"]["min_avg20_volume"] = st.number_input("최소 20일 평균거래량", 0, 500000, int(BASE_CFG["universe"]["min_avg20_volume"]), step=10000)
    with p2:
        cfg["step1_downtrend"]["min_inverse_days"] = st.slider("최근 250일 중 역배열 최소 일수", 120, 240, int(BASE_CFG["step1_downtrend"]["min_inverse_days"]), step=10)
        cfg["step1_downtrend"]["min_drawdown_from_1y_high_pct"] = st.slider("1년 고점 대비 최소 하락률(%)", 10.0, 50.0, float(BASE_CFG["step1_downtrend"]["min_drawdown_from_1y_high_pct"]), step=5.0)
    with p3:
        cfg["step2_box"]["box_window_days"] = st.slider("박스권 분석 기간(일)", 20, 60, int(BASE_CFG["step2_box"]["box_window_days"]), step=5)
        cfg["step2_box"]["allowed_max_box_width_pct"] = st.slider("박스권 허용 폭(%)", 10.0, 35.0, float(BASE_CFG["step2_box"]["allowed_max_box_width_pct"]), step=2.5)
        cfg["step5_volume"]["vol_surge_ratio"] = st.slider("거래량 급증 배수(20일 대비)", 1.2, 3.0, float(BASE_CFG["step5_volume"]["vol_surge_ratio"]), step=0.1)
    with p4:
        cfg["step3_convergence"]["max_gap_20_200_pct"] = st.slider("20-200일선 수렴 이격도(%)", 1.5, 6.0, float(BASE_CFG["step3_convergence"]["max_gap_20_200_pct"]), step=0.5)
        cfg["step3_convergence"]["max_gap_50_200_pct"] = st.slider("50-200일선 수렴 이격도(%)", 1.5, 6.0, float(BASE_CFG["step3_convergence"]["max_gap_50_200_pct"]), step=0.5)
        cfg["step3_convergence"]["max_gap_20_50_pct"] = st.slider("20-50일선 수렴 이격도(%)", 1.0, 5.0, float(BASE_CFG["step3_convergence"]["max_gap_20_50_pct"]), step=0.5)

if "150" in scan_preset:
    scan_limit = 150
elif "300" in scan_preset:
    scan_limit = 300
elif "600" in scan_preset:
    scan_limit = 600
else:
    scan_limit = None


# ---------------------------------------------------------
# 전체 시장 스캔 실행
# ---------------------------------------------------------
if run_btn:
    progress_bar = st.progress(0, text="1/3) 코스피·코스닥 종목 목록 조회 및 제외 필터 적용 중...")
    uni_df = safe_get_universe(cfg)

    if uni_df.empty:
        st.error("종목 목록을 불러오지 못했습니다. 잠시 후 다시 시도해 주세요.")
        st.stop()

    target_df = uni_df.head(scan_limit).copy() if scan_limit else uni_df.copy()

    if custom_codes.strip():
        existing_set = set(target_df["code"].astype(str))
        extra_rows = []
        for raw_c in custom_codes.split(","):
            c = raw_c.strip().zfill(6)
            if len(c) == 6 and c.isdigit() and c not in existing_set:
                match_u = uni_df[uni_df["code"] == c]
                if not match_u.empty:
                    extra_rows.append(match_u.iloc[0].to_dict())
                else:
                    extra_rows.append({"code": c, "name": f"추가종목({c})", "market": "직접추가", "mcap_eok": cfg["universe"]["min_mcap_eok"], "delisted": False})
                existing_set.add(c)
        if extra_rows:
            target_df = pd.concat([target_df, pd.DataFrame(extra_rows)], ignore_index=True)

    total_n = len(target_df)
    meta_records = target_df.to_dict("records")
    indicator_map = {}
    eval_rows = []

    def analyze_one(meta):
        df_raw = safe_fetch_ohlcv(meta["code"], cfg)
        if df_raw is None:
            return meta["code"], None, None
        ind_df = compute_all_indicators(df_raw, cfg)
        res = evaluate_signal_at_index(ind_df, len(ind_df) - 1, meta, cfg)
        return meta["code"], ind_df, res

    done_cnt = 0
    with ThreadPoolExecutor(max_workers=10) as ex:
        futs = [ex.submit(analyze_one, m) for m in meta_records]
        for f in as_completed(futs):
            done_cnt += 1
            try:
                code, ind_df, res = f.result()
                if ind_df is not None:
                    indicator_map[code] = ind_df
                if res is not None:
                    eval_rows.append(res)
            except Exception:
                pass
            if done_cnt % 5 == 0 or done_cnt == total_n:
                pct = int((done_cnt / total_n) * 100)
                progress_bar.progress(pct, text=f"2/3) 3년 일봉 수집 및 9단계 지표 계산 중... ({done_cnt} / {total_n}종목)")

    progress_bar.progress(100, text="3/3) 분석 완료!")

    all_df = pd.DataFrame(eval_rows)
    if not all_df.empty:
        all_df = all_df.sort_values(
            by=["신호 우선순위", "최종 점수", "거래량 증가율(배)"],
            ascending=[True, False, False]
        ).reset_index(drop=True)
        passed_df = all_df[all_df["최종 등급"] != "제외"].reset_index(drop=True)
    else:
        passed_df = pd.DataFrame()

    st.session_state.scan_done = True
    st.session_state.all_df = all_df
    st.session_state.passed_df = passed_df
    st.session_state.indicator_map = indicator_map
    st.session_state.updated_at = datetime.now().strftime("%Y-%m-%d %H:%M")
    st.session_state.total_target = total_n
    st.session_state.bt_summary = pd.DataFrame()
    st.session_state.bt_trades = pd.DataFrame()
    st.session_state.used_cfg = copy.deepcopy(cfg)
    progress_bar.empty()


all_df = st.session_state.all_df
passed_df = st.session_state.passed_df
indicator_map = st.session_state.indicator_map
used_cfg = st.session_state.used_cfg

if st.session_state.scan_done and not all_df.empty:
    cnt_d = int(passed_df["신호 유형"].str.startswith("D.").sum()) if not passed_df.empty else 0
    cnt_c = int(passed_df["신호 유형"].str.startswith("C.").sum()) if not passed_df.empty else 0
    cnt_b = int(passed_df["신호 유형"].str.startswith("B.").sum()) if not passed_df.empty else 0
    cnt_a = int(passed_df["신호 유형"].str.startswith("A.").sum()) if not passed_df.empty else 0

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("🔥 D. 최종 상승전환", f"{cnt_d}종목", "최우선 시그널")
    m2.metric("✨ C. 골든크로스", f"{cnt_c}종목", "수렴 후 돌파")
    m3.metric("⚡ B. 거래량 급증", f"{cnt_b}종목", "수렴 + 1.5배+")
    m4.metric("🧲 A. 이평선 수렴", f"{cnt_a}종목", "바닥 응축")
    m5.metric("📊 분석 완료", f"{len(all_df)} / {st.session_state.total_target}", st.session_state.updated_at)

st.divider()

# ---------------------------------------------------------
# 6대 대시보드 탭 (⑤ 엘리엇 파동 분석 / ⑥ 백테스트는 스캔 없이도 즉시 사용 가능)
# ---------------------------------------------------------
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "① 앤트위키 시그널 지도",
    "② 결과표 & 엑셀 저장",
    "③ 종목별 근거 & 매매전략",
    "④ 20·40·50·200일 차트",
    "⑤ 엘리엇 파동 분석",
    "⑥ 전략 A/B/C 백테스트"
])

SCAN_NOTICE = "👆 상단의 **[🚀 전체 시장 스캔 시작]** 버튼을 누르시면 시장 전체 분석 결과가 표시됩니다. (특정 종목을 바로 분석하려면 **⑤ 엘리엇 파동 분석** 또는 **⑥ 전략 A/B/C 백테스트** 탭을 바로 눌러보세요!)"

# [탭 1] 앤트위키 트리맵
with tab1:
    if not st.session_state.scan_done or all_df.empty:
        st.info(SCAN_NOTICE)
    else:
        t1_mode = st.radio(
            "지도 표시 대상",
            ["✅ 조건 통과 종목 (D/C/B/A 시그널)", "📋 전체 분석 종목 중 점수 상위 40개"],
            horizontal=True
        )
        map_df = passed_df.copy() if "통과" in t1_mode else all_df.sort_values("최종 점수", ascending=False).head(40).copy()

        if map_df.empty:
            st.info("현재 엄격 조건을 모두 통과한 종목이 0개입니다. 위에서 '전체 분석 종목 중 점수 상위 40개'를 선택하거나 상단 설정을 조금 완화해 보세요.")
        else:
            map_df["tile_size"] = map_df["거래대금(억원)"].clip(lower=10)
            fig_tree = px.treemap(
                map_df,
                path=[px.Constant("이평선 수렴 포착 지도"), "신호 유형", "최종 등급", "종목명 및 종목코드"],
                values="tile_size",
                color="최종 점수",
                color_continuous_scale=[[0.0, "#bfdbfe"], [0.5, "#fef08a"], [0.8, "#fca5a5"], [1.0, "#dc2626"]],
                range_color=[45, 100],
                custom_data=["최종 점수", "현재가", "20-200 이격도(%)", "거래량 증가율(배)", "최종 등급", "수렴 상태"]
            )
            fig_tree.update_traces(
                texttemplate="<b>%{label}</b><br>%{customdata[0]}점 · %{customdata[5]}<br>%{customdata[1]:,}원 (거래 %{customdata[3]}배)",
                textposition="middle center",
                hovertemplate=(
                    "<b>%{label}</b><br>등급: %{customdata[4]} (%{customdata[0]}점)<br>"
                    "수렴 상태: %{customdata[5]} (20-200 이격 %{customdata[2]}%)<br>"
                    "현재가: %{customdata[1]:,}원 · 거래량: %{customdata[3]}배<extra></extra>"
                )
            )
            fig_tree.update_layout(margin=dict(t=20, l=10, r=10, b=10), height=560)
            st.plotly_chart(fig_tree, use_container_width=True)

# [탭 2] 23개 항목 전체 결과표 및 CSV / Excel 다운로드
with tab2:
    if not st.session_state.scan_done or all_df.empty:
        st.info(SCAN_NOTICE)
    else:
        f_c1, f_c2, f_c3 = st.columns([2, 1, 1])
        with f_c1:
            include_excl = st.checkbox("65점 미만 / 필수조건 미달 종목도 함께 보기 (근접 후보 점검용)", value=False)
        view_df = all_df if include_excl else passed_df

        cols_order = [
            "종목명 및 종목코드", "시장", "현재가", "최근 1년 고점 대비 하락률(%)", "역배열 유지 기간(일)",
            "20일 이동평균선", "40일 이동평균선", "50일 이동평균선", "200일 이동평균선",
            "20-200 이격도(%)", "50-200 이격도(%)", "20-50 이격도(%)", "20일간 이격도 축소율(%)",
            "수렴 상태", "이동평균선별 기울기", "골든크로스 발생 여부 및 종류",
            "거래량 증가율(배)", "거래대금(억원)", "박스권 상단 및 하단", "RSI 및 MACD",
            "예상 손익비", "최종 점수", "최종 등급", "신호 유형",
            "신호 포착 근거", "매매 진입 및 손절 기준", "상승 시나리오 무효화 조건"
        ]
        show_cols = [c for c in cols_order if c in view_df.columns]

        today_tag = datetime.now().strftime("%Y%m%d")
        with f_c2:
            if not view_df.empty:
                st.download_button(
                    "📥 CSV 파일 저장",
                    data=view_df[show_cols].to_csv(index=False).encode("utf-8-sig"),
                    file_name=f"ma_signals_{today_tag}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
        with f_c3:
            if not view_df.empty:
                xlsx_bytes = build_excel_bytes(view_df[show_cols], st.session_state.bt_summary, st.session_state.bt_trades)
                st.download_button(
                    "📊 엑셀(.xlsx) 저장",
                    data=xlsx_bytes,
                    file_name=f"ma_report_{today_tag}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )

        if view_df.empty:
            st.info("통과한 종목이 없습니다. 위 체크박스를 켜면 점수 상위 근접 종목들을 확인할 수 있습니다.")
        else:
            st.dataframe(view_df[show_cols], use_container_width=True, height=540, hide_index=True)

# [탭 3] 종목별 포착 근거 & 매매 전략 카드
with tab3:
    if not st.session_state.scan_done or all_df.empty:
        st.info(SCAN_NOTICE)
    else:
        c_scope = st.radio(
            "카드 표시 필터",
            ["✅ 선별 종목 (D/C/B/A 통과)", "🔎 전체 점수 상위 15종목 (제외 포함)"],
            horizontal=True
        )
        card_df = passed_df if "선별" in c_scope else all_df.sort_values("최종 점수", ascending=False).head(15)

        if card_df.empty:
            st.info("표시할 선별 종목이 없습니다. '전체 점수 상위 15종목'을 선택해 보세요.")
        else:
            for _, r in card_df.iterrows():
                g = r["최종 등급"]
                b_cls = "b-top" if g == "최우선 관심 종목" else "b-cand" if g == "매매 후보" else "b-watch" if g == "관찰 종목" else "b-excl"
                rr_color = "#2b8a3e" if float(r["예상 손익비"]) >= 1.5 else "#d9480f"
                st.markdown(f"""
                <div class="signal-card">
                    <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:8px;">
                        <div>
                            <span class="{b_cls}">{g} ({r['최종 점수']}점)</span>
                            <span class="b-sig">{r['신호 유형']}</span>
                            <b style="font-size:1.15rem; margin-left:8px;">{r['종목명 및 종목코드']}</b>
                            <span style="color:#868e96; font-weight:600;">[{r['시장']}]</span>
                        </div>
                        <div style="text-align:right;">
                            <b style="font-size:1.15rem;">{int(r['현재가']):,}원</b>
                            <span style="color:{rr_color}; font-weight:800; margin-left:8px;">손익비 {r['예상 손익비']:.2f}</span>
                        </div>
                    </div>
                    <div style="background:#f8f9fa; padding:9px 12px; border-radius:10px; margin-top:10px; font-size:0.84rem;">
                        <b>[세부 채점]</b> {r['세부점수']} &nbsp;|&nbsp; <b>[수렴 상태]</b> {r['수렴 상태']} (20일 축소율 {r['20일간 이격도 축소율(%)']:+.1f}%)
                    </div>
                    <div style="font-size:0.86rem; color:#212529; margin-top:10px; line-height:1.6;">
                        • <b>이평선 간격:</b> 20-200 <b>{r['20-200 이격도(%)']}%</b> · 50-200 <b>{r['50-200 이격도(%)']}%</b> · 20-50 <b>{r['20-50 이격도(%)']}%</b> · 40일선 이격 {r['40일선 이격도(%)']:+.1f}%<br>
                        • <b>포착 근거:</b> {r['신호 포착 근거']}<br>
                        • <b>매매 기준:</b> <span style="color:#1964d8; font-weight:700;">{r['매매 진입 및 손절 기준']}</span><br>
                        • <b>무효화 조건:</b> <span style="color:#c92a2a;">{r['상승 시나리오 무효화 조건']}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

# [탭 4] 20·40·50·200일 이동평균선 & 박스권 정밀 차트 검수
with tab4:
    if not st.session_state.scan_done or all_df.empty:
        st.info(SCAN_NOTICE)
    else:
        sort_for_chart = all_df.sort_values(["신호 우선순위", "최종 점수"], ascending=[True, False]).reset_index(drop=True)
        chart_opts = [
            f"[{r['신호 유형']} · {r['최종 점수']}점] {r['종목명 및 종목코드']} ({r['최종 등급']})"
            for _, r in sort_for_chart.iterrows()
        ]
        sel_chart_opt = st.selectbox("🔎 정밀 차트를 검수할 종목을 선택하세요", chart_opts)
        sel_row = sort_for_chart.iloc[chart_opts.index(sel_chart_opt)]
        sel_code = str(sel_row["종목코드"])

        if sel_code in indicator_map:
            cdf = indicator_map[sel_code].tail(260).copy().reset_index()
            date_col = cdf.columns[0]
            cdf = cdf.rename(columns={date_col: "Date"})

            fig = make_subplots(
                rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.04,
                row_heights=[0.60, 0.20, 0.20],
                subplot_titles=(
                    f"{sel_row['종목명 및 종목코드']} | 20·40·50·200일선 수렴 및 인과적 박스권",
                    "거래량 & 20일 평균 거래량",
                    "RSI(14)"
                )
            )

            fig.add_trace(go.Candlestick(
                x=cdf["Date"], open=cdf["Open"], high=cdf["High"], low=cdf["Low"], close=cdf["Close"],
                name="일봉", increasing_line_color="#e12343", decreasing_line_color="#1964d8"
            ), row=1, col=1)

            fig.add_trace(go.Scatter(x=cdf["Date"], y=cdf["MA20"], name="20일선", line=dict(color="#f59f00", width=1.8)), row=1, col=1)
            fig.add_trace(go.Scatter(x=cdf["Date"], y=cdf["MA40"], name="40일선", line=dict(color="#e11d48", width=2.2)), row=1, col=1)
            fig.add_trace(go.Scatter(x=cdf["Date"], y=cdf["MA50"], name="50일선", line=dict(color="#16a34a", width=2.0)), row=1, col=1)
            fig.add_trace(go.Scatter(x=cdf["Date"], y=cdf["MA200"], name="200일선", line=dict(color="#1d4ed8", width=2.6, dash="dot")), row=1, col=1)

            latest_box_h = float(cdf["Box_High"].iloc[-1])
            latest_box_l = float(cdf["Box_Low"].iloc[-1])
            fig.add_hline(y=latest_box_h, line_dash="dash", line_color="#d97706", annotation_text=f"박스상단 {int(latest_box_h):,}", row=1, col=1)
            fig.add_hline(y=latest_box_l, line_dash="dash", line_color="#0284c7", annotation_text=f"박스하단 {int(latest_box_l):,}", row=1, col=1)

            gc_pts = cdf[cdf["GC_Any"] == True]
            if not gc_pts.empty:
                fig.add_trace(go.Scatter(
                    x=gc_pts["Date"], y=gc_pts["Close"],
                    mode="markers", name="골든크로스 발생일",
                    marker=dict(symbol="star", size=14, color="#f59f00", line=dict(width=1.2, color="#191f28"))
                ), row=1, col=1)

            v_cols = ["#fca5a5" if c >= o else "#93c5fd" for c, o in zip(cdf["Close"], cdf["Open"])]
            fig.add_trace(go.Bar(x=cdf["Date"], y=cdf["Volume"], name="거래량", marker_color=v_cols), row=2, col=1)
            fig.add_trace(go.Scatter(x=cdf["Date"], y=cdf["Vol_MA20"], name="20일 평균거래량", line=dict(color="#191f28", width=1.6)), row=2, col=1)

            fig.add_trace(go.Scatter(x=cdf["Date"], y=cdf["RSI14"], name="RSI(14)", line=dict(color="#7c3aed", width=2)), row=3, col=1)
            fig.add_hline(y=50, line_dash="dash", line_color="#868e96", row=3, col=1)
            fig.add_hline(y=70, line_dash="dot", line_color="#e12343", row=3, col=1)

            fig.update_layout(height=760, xaxis_rangeslider_visible=False, hovermode="x unified", margin=dict(t=40, l=20, r=20, b=20))
            st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------
# [탭 5] 종목별 엘리엇 파동 분석 (차트 대입 & 현재 지점 진단)
# ---------------------------------------------------------
with tab5:
    st.subheader("🌊 종목별 엘리엇 파동 자동 대입 & 현재 지점 분석")
    st.caption("일봉 변곡점(ZigZag)에 엘리엇 파동(상승 1·2·3·4·5파 및 조정 A·B·C파) 규칙과 피보나치 비율을 대입해 현재 주가 위치를 표시합니다.")

    # 종목 선택 방식 (스캔 결과에서 선택 OR 전체 종목 직접 검색)
    ew_source_opts = ["🔎 원하는 종목 직접 검색", "📋 방금 스캔한 종목에서 선택"] if (st.session_state.scan_done and not all_df.empty) else ["🔎 원하는 종목 직접 검색"]
    default_src_idx = 1 if (st.session_state.scan_done and not all_df.empty) else 0
    ew_source = st.radio("분석할 종목 선택 방식", ew_source_opts, index=default_src_idx, horizontal=True)

    ew_c1, ew_c2, ew_c3 = st.columns([2.2, 1.1, 1.3])
    ew_code = None
    ew_name = ""

    with ew_c1:
        if "스캔한 종목" in ew_source and not all_df.empty:
            ew_scan_df = all_df.sort_values(["신호 우선순위", "최종 점수"], ascending=[True, False]).reset_index(drop=True)
            ew_scan_opts = [f"[{r['신호 유형']} · {r['최종 점수']}점] {r['종목명 및 종목코드']}" for _, r in ew_scan_df.iterrows()]
            picked = st.selectbox("스캔 종목 선택", ew_scan_opts, key="ew_scan_pick")
            picked_row = ew_scan_df.iloc[ew_scan_opts.index(picked)]
            ew_code = str(picked_row["종목코드"])
            ew_name = str(picked_row["종목명 및 종목코드"])
        else:
            master_ew = get_searchable_stock_master()
            if not master_ew.empty:
                ew_labels = [f"{r['name']} ({r['code']}) [{r['market']}]" for _, r in master_ew.iterrows()]
                ew_pick = st.selectbox(
                    "종목명 또는 종목코드 검색 (타이핑해서 찾으세요)",
                    options=ew_labels,
                    index=0,
                    key="ew_search_pick"
                )
                pick_idx = ew_labels.index(ew_pick)
                ew_code = str(master_ew.iloc[pick_idx]["code"])
                ew_name = f"{master_ew.iloc[pick_idx]['name']} ({ew_code})"
            else:
                manual_ew = st.text_input("종목코드 6자리 입력", value="005930", key="ew_manual_code")
                ew_code = manual_ew.strip().zfill(6)
                ew_name = f"종목 ({ew_code})"

    with ew_c2:
        ew_period_label = st.selectbox(
            "분석 기간",
            ["최근 1년 (단기 소파동)", "최근 2년 (중기 스윙)", "최근 3년 (장기 대파동)"],
            index=1,
            key="ew_period"
        )
        ew_bars = 250 if "1년" in ew_period_label else 500 if "2년" in ew_period_label else 750

    with ew_c3:
        ew_pct = st.slider(
            "파동 민감도 (변동폭 %)",
            min_value=3.0, max_value=15.0, value=7.0, step=0.5,
            help="숫자가 작을수록 세밀한 단기 파동, 클수록 굵직한 대파동으로 분석합니다.",
            key="ew_pct"
        )

    show_ma_overlay = st.checkbox("차트에 40일선·200일선 함께 겹쳐 보기", value=True, key="ew_ma_toggle")

    if ew_code:
        # 이미 스캔된 데이터가 있으면 재사용하고, 없으면 즉시 3년 일봉 조회
        ew_df_full = None
        if ew_code in indicator_map and len(indicator_map[ew_code]) >= ew_bars:
            ew_df_full = indicator_map[ew_code]
        else:
            with st.spinner(f"{ew_name} 일봉 데이터를 불러와 엘리엇 파동을 계산 중입니다..."):
                raw_single = fetch_ohlcv_for_single_stock(ew_code, min_days=120, count=800)
                if raw_single is not None:
                    ew_df_full = compute_all_indicators(raw_single, cfg)

        if ew_df_full is None or ew_df_full.empty:
            st.error(f"{ew_name}의 일봉 데이터를 불러오지 못했습니다. 종목코드를 확인하거나 잠시 후 다시 시도해 주세요.")
        else:
            ew_df = ew_df_full.tail(ew_bars).copy()
            ew_res = analyze_elliott_wave(ew_df, pct_threshold=ew_pct)

            if not ew_res["ok"]:
                st.warning(ew_res["message"])
            else:
                # 1) 현재 파동 위치 요약 카드
                inv_txt = f"{int(ew_res['invalidation_price']):,}원 ({ew_res['invalidation_label']})" if ew_res["invalidation_price"] else "-"
                rules_html = " &nbsp;|&nbsp; ".join([f"✔ {note}" for note in ew_res["rule_notes"]])

                st.markdown(f"""
                <div class="ew-box" style="border-color:{ew_res['stage_badge_color']};">
                    <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:8px;">
                        <div>
                            <span style="background:{ew_res['stage_badge_color']}; color:#fff; padding:6px 14px; border-radius:20px; font-weight:800; font-size:0.95rem;">
                                현재 위치: {ew_res['stage_title']}
                            </span>
                            <b style="font-size:1.15rem; margin-left:10px;">{ew_name}</b>
                        </div>
                        <div style="font-size:1.15rem; font-weight:800;">
                            현재가: {int(ew_res['cur_price']):,}원
                            <span style="font-size:0.85rem; color:#6b7684; font-weight:500;">({ew_res['cur_date'].strftime('%Y-%m-%d')} 기준)</span>
                        </div>
                    </div>
                    <div style="margin-top:12px; font-size:0.93rem; color:#212529; line-height:1.6;">
                        • <b>파동 진단:</b> {ew_res['stage_desc']}<br>
                        • <b>스윙 전략 포인트:</b> <span style="color:#1964d8; font-weight:700;">{ew_res['action_tip']}</span><br>
                        • <b>파동 무효화(손절) 기준:</b> <span style="color:#c92a2a; font-weight:700;">{inv_txt}</span>
                    </div>
                    <div style="background:#f8f9fa; padding:8px 12px; border-radius:8px; margin-top:10px; font-size:0.82rem; color:#495057;">
                        {rules_html}
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # 2) 엘리엇 파동 오버레이 Plotly 차트
                fig_ew = go.Figure()

                # 일봉 캔들 (파동선이 잘 보이도록 깔끔한 톤)
                fig_ew.add_trace(go.Candlestick(
                    x=ew_df.index,
                    open=ew_df["Open"], high=ew_df["High"], low=ew_df["Low"], close=ew_df["Close"],
                    name="일봉",
                    increasing_line_color="rgba(225, 35, 67, 0.55)",
                    decreasing_line_color="rgba(25, 100, 216, 0.55)",
                    increasing_fillcolor="rgba(225, 35, 67, 0.30)",
                    decreasing_fillcolor="rgba(25, 100, 216, 0.30)"
                ))

                # 선택 시 40일선·200일선 오버레이
                if show_ma_overlay and "MA40" in ew_df.columns and "MA200" in ew_df.columns:
                    fig_ew.add_trace(go.Scatter(
                        x=ew_df.index, y=ew_df["MA40"], name="40일선",
                        line=dict(color="rgba(245, 159, 0, 0.7)", width=1.6)
                    ))
                    fig_ew.add_trace(go.Scatter(
                        x=ew_df.index, y=ew_df["MA200"], name="200일선",
                        line=dict(color="rgba(71, 85, 105, 0.75)", width=1.8, dash="dot")
                    ))

                # 전체 배경 ZigZag (연한 회색 실선)
                all_pivs = ew_res["pivots"]
                fig_ew.add_trace(go.Scatter(
                    x=[p["date"] for p in all_pivs],
                    y=[p["price"] for p in all_pivs],
                    mode="lines",
                    name="전체 변곡선(ZigZag)",
                    line=dict(color="rgba(148, 163, 184, 0.45)", width=1.5, dash="dot")
                ))

                # 핵심 엘리엇 파동선 (충격파 빨강 / 조정파 파랑 / 복합 보라)
                labeled = ew_res["labeled"]
                imp_pts = [p for p in labeled if p["group"] == "impulse"]
                corr_pts = [p for p in labeled if p["group"] == "correction"]
                comp_pts = [p for p in labeled if p["group"] == "complex"]

                if len(imp_pts) >= 2:
                    fig_ew.add_trace(go.Scatter(
                        x=[p["date"] for p in imp_pts],
                        y=[p["price"] for p in imp_pts],
                        mode="lines+markers",
                        name="엘리엇 충격파 (1~5파)",
                        line=dict(color="#e11d48", width=3.5),
                        marker=dict(size=9, color="#e11d48", line=dict(width=1.5, color="#ffffff"))
                    ))

                if corr_pts:
                    # 5파 완료 후 이어지는 조정파라면 5파 꼭짓점부터 A로 연결
                    corr_line_pts = ([imp_pts[-1]] + corr_pts) if imp_pts else corr_pts
                    if len(corr_line_pts) >= 2:
                        fig_ew.add_trace(go.Scatter(
                            x=[p["date"] for p in corr_line_pts],
                            y=[p["price"] for p in corr_line_pts],
                            mode="lines+markers",
                            name="엘리엇 조정파 (A~C파)",
                            line=dict(color="#1d4ed8", width=3.5),
                            marker=dict(size=9, color="#1d4ed8", line=dict(width=1.5, color="#ffffff"))
                        ))

                if len(comp_pts) >= 2:
                    fig_ew.add_trace(go.Scatter(
                        x=[p["date"] for p in comp_pts],
                        y=[p["price"] for p in comp_pts],
                        mode="lines+markers",
                        name="복합/횡보 파동",
                        line=dict(color="#7c3aed", width=3.0),
                        marker=dict(size=8, color="#7c3aed", line=dict(width=1.5, color="#ffffff"))
                    ))

                # 마지막 확정 변곡점 -> 현재 주가(진행 중 파동 점선 + 별표 마커)
                last_p = labeled[-1]
                fig_ew.add_trace(go.Scatter(
                    x=[last_p["date"], ew_res["cur_date"]],
                    y=[last_p["price"], ew_res["cur_price"]],
                    mode="lines+markers",
                    name="📍 현재 진행 중인 파동",
                    line=dict(color=ew_res["stage_badge_color"], width=3.2, dash="dash"),
                    marker=dict(
                        size=[8, 16],
                        symbol=["circle", "star"],
                        color=[ew_res["stage_badge_color"], "#f59f00"],
                        line=dict(width=1.5, color="#191f28")
                    )
                ))

                # 파동 번호 라벨(0, 1, 2, 3, 4, 5, A, B, C) 배지 표시
                for p in labeled:
                    y_shift = 20 if p["kind"] == "H" else -22
                    lbl_bg = "#e11d48" if p["group"] == "impulse" else "#1d4ed8" if p["group"] == "correction" else "#7c3aed"
                    fig_ew.add_annotation(
                        x=p["date"],
                        y=p["price"],
                        text=f"<b>{p['label']}</b><br><span style='font-size:10px;'>{int(p['price']):,}</span>",
                        showarrow=False,
                        yshift=y_shift,
                        bgcolor=lbl_bg,
                        font=dict(color="#ffffff", size=12),
                        borderpad=4,
                        opacity=0.92
                    )

                # 현재 위치 강조 콜아웃
                fig_ew.add_annotation(
                    x=ew_res["cur_date"],
                    y=ew_res["cur_price"],
                    text=f"<b>📍 현재 지점</b><br>{int(ew_res['cur_price']):,}원",
                    showarrow=True,
                    arrowhead=2,
                    ax=35,
                    ay=-35,
                    bgcolor="#191f28",
                    font=dict(color="#ffffff", size=11),
                    borderpad=5
                )

                # 피보나치 핵심 지지/목표가 수평선
                for fib_lbl, fib_price, fib_col in ew_res["fib_levels"]:
                    fig_ew.add_hline(
                        y=fib_price,
                        line_dash="dot",
                        line_color=fib_col,
                        line_width=1.3,
                        annotation_text=f"{fib_lbl}: {int(fib_price):,}원",
                        annotation_position="top right",
                        annotation_font=dict(size=11, color=fib_col)
                    )

                # 파동 무효화(손절) 기준선
                if ew_res["invalidation_price"]:
                    fig_ew.add_hline(
                        y=ew_res["invalidation_price"],
                        line_dash="dash",
                        line_color="#dc2626",
                        line_width=1.8,
                        annotation_text=f"⚠️ 무효화선: {int(ew_res['invalidation_price']):,}원",
                        annotation_position="bottom right",
                        annotation_font=dict(size=11, color="#dc2626")
                    )

                fig_ew.update_layout(
                    title=f"{ew_name} 엘리엇 파동 차트 ({ew_period_label} · 민감도 {ew_pct}%)",
                    height=600,
                    xaxis_rangeslider_visible=False,
                    hovermode="x unified",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    margin=dict(t=65, l=20, r=30, b=20)
                )
                st.plotly_chart(fig_ew, use_container_width=True)

                # 3) 파동별 구간 상세 표
                st.markdown("#### 📊 파동 구간별 가격 · 변동률 · 소요 기간")
                st.dataframe(ew_res["wave_table"], use_container_width=True, hide_index=True)
                st.caption("※ 엘리엇 파동 카운팅은 알고리즘(ZigZag 변곡점 + 핵심 3대 규칙)으로 자동 산출된 참고용 해석입니다. 민감도(%)에 따라 상위/하위 파동 관점이 달라질 수 있으므로 이동평균선 수렴 및 거래량 시그널과 함께 종합 판단하세요.")

# ---------------------------------------------------------
# [탭 6] 전략 A · B · C 과거 3년 백테스트 (종목 검색 모드 + 전체 스캔 모드)
# ---------------------------------------------------------
with tab6:
    st.subheader("전략 A · B · C 과거 3년 백테스트 성과 비교 (10 · 20 · 40 · 60일 보유)")
    st.caption("체결 기준: t일 장 마감 신호 확정 ➔ t+1일 시가(Open) 진입 ➔ 수수료(0.015%)·거래세(0.18%)·슬리피지(0.15%) 차감 반영 (룩어헤드 바이어스 차단)")

    bt_mode = st.radio(
        "🧪 백테스트 대상 선택",
        ["🔎 원하는 종목 직접 검색해서 백테스트 (추천)", "🌐 방금 스캔한 전체 종목으로 백테스트"],
        horizontal=True
    )

    if "직접 검색" in bt_mode:
        master_df = get_searchable_stock_master()
        label_to_meta = {}
        options_list = []

        if not master_df.empty:
            for _, row in master_df.iterrows():
                lbl = f"{row['name']} ({row['code']}) [{row['market']}]"
                options_list.append(lbl)
                label_to_meta[lbl] = {"code": str(row["code"]), "name": str(row["name"])}

        s_c1, s_c2 = st.columns([2.2, 1.3])
        with s_c1:
            selected_labels = st.multiselect(
                "종목명 또는 종목코드 검색 (여러 종목 동시 선택 가능)",
                options=options_list,
                placeholder="예: 삼성전자, 카카오, 005930 등을 입력해 선택하세요"
            )
        with s_c2:
            manual_bt_codes = st.text_input(
                "목록에 없는 종목코드 직접 입력 (쉼표 구분)",
                placeholder="예: 005930, 035720"
            )

        if st.button("🧪 선택한 종목으로 전략 A/B/C 백테스트 실행", type="primary"):
            target_codes = {}
            for lbl in selected_labels:
                m = label_to_meta[lbl]
                target_codes[m["code"]] = m["name"]

            if manual_bt_codes.strip():
                code_lookup = master_df.set_index("code")["name"].to_dict() if not master_df.empty else {}
                for raw_c in manual_bt_codes.split(","):
                    c = raw_c.strip().zfill(6)
                    if len(c) == 6 and c.isdigit():
                        target_codes[c] = code_lookup.get(c, f"종목({c})")

            if not target_codes:
                st.warning("백테스트할 종목을 1개 이상 검색해 선택하거나 종목코드를 입력해 주세요.")
            else:
                search_ind_map = {}
                failed_names = []
                with st.spinner(f"선택한 {len(target_codes)}개 종목의 과거 5년 일봉(3년 백테스트 구간 확보)을 불러와 계산 중입니다..."):
                    for c, nm in target_codes.items():
                        df_raw = fetch_ohlcv_for_backtest(c, cfg)
                        if df_raw is not None:
                            search_ind_map[c] = compute_all_indicators(df_raw, cfg)
                        else:
                            failed_names.append(f"{nm}({c})")

                    if search_ind_map:
                        bt_sum, bt_tr = run_backtest_suite(search_ind_map, cfg)
                        if not bt_tr.empty:
                            bt_tr.insert(3, "종목명", bt_tr["종목코드"].map(lambda x: target_codes.get(str(x), str(x))))
                        st.session_state.search_bt_summary = bt_sum
                        st.session_state.search_bt_trades = bt_tr
                        st.session_state.search_bt_ind_map = search_ind_map
                        st.session_state.search_bt_names = target_codes
                        st.session_state.search_bt_cfg = copy.deepcopy(cfg)
                        st.session_state.search_bt_done = True
                    else:
                        st.session_state.search_bt_done = False

                if failed_names:
                    st.warning(f"일봉 데이터가 부족(상장 2년 미만 또는 거래정지)해 제외된 종목: {', '.join(failed_names)}")

        if st.session_state.search_bt_done:
            s_sum = st.session_state.search_bt_summary
            s_tr = st.session_state.search_bt_trades
            s_map = st.session_state.search_bt_ind_map
            s_names = st.session_state.search_bt_names
            s_cfg = st.session_state.search_bt_cfg

            tested_str = ", ".join([f"{nm}({cd})" for cd, nm in s_names.items() if cd in s_map])
            st.success(f"백테스트 완료 대상 ({len(s_map)}종목): **{tested_str}**")
            st.dataframe(s_sum, use_container_width=True, hide_index=True)

            total_trades_cnt = int(s_sum["거래 횟수"].sum()) if not s_sum.empty else 0
            if total_trades_cnt == 0:
                st.info(
                    "선택한 종목에서 현재 엄격한 조건(1년 중 180일+ 역배열 · 이격도 3% 이내 · 거래량 1.5배+ 등)을 모두 만족한 과거 신호가 0건입니다. "
                    "상단의 **[⚙️ 전략·임계값 세부 설정]**을 열어 역배열 일수나 수렴 이격도 기준을 조금 완화한 뒤 다시 실행해 보세요."
                )
            else:
                b_c1, b_c2 = st.columns(2)
                with b_c1:
                    fig_win = px.bar(
                        s_sum, x="보유기간(일)", y="승률(%)", color="전략", barmode="group",
                        title="보유기간별 전략 승률(%) 비교", text="승률(%)"
                    )
                    fig_win.update_layout(height=360, legend=dict(orientation="h", yanchor="bottom", y=-0.38))
                    st.plotly_chart(fig_win, use_container_width=True)
                with b_c2:
                    fig_exp = px.bar(
                        s_sum, x="보유기간(일)", y="기대수익(%)", color="전략", barmode="group",
                        title="보유기간별 1회 매매 기대수익(%) 비교", text="기대수익(%)"
                    )
                    fig_exp.add_hline(y=0, line_dash="dash", line_color="#495057")
                    fig_exp.update_layout(height=360, legend=dict(orientation="h", yanchor="bottom", y=-0.38))
                    st.plotly_chart(fig_exp, use_container_width=True)

            st.markdown("#### 📈 검색 종목 과거 신호 발생 시점 차트 확인")
            chart_codes = list(s_map.keys())
            pick_code = st.selectbox(
                "차트로 신호 위치를 확인할 종목 선택",
                options=chart_codes,
                format_func=lambda c: f"{s_names.get(c, c)} ({c})"
            )
            pdf = s_map[pick_code]
            masks = build_strategy_masks(pdf, s_cfg)
            start_off = s_cfg["backtest"]["start_offset_days"]
            plot_slice = pdf.iloc[max(0, start_off - 40):].copy()

            fig_bt = go.Figure()
            fig_bt.add_trace(go.Scatter(x=plot_slice.index, y=plot_slice["Close"], name="종가", line=dict(color="#495057", width=1.6)))
            fig_bt.add_trace(go.Scatter(x=plot_slice.index, y=plot_slice["MA20"], name="20일선", line=dict(color="#f59f00", width=1.4)))
            fig_bt.add_trace(go.Scatter(x=plot_slice.index, y=plot_slice["MA50"], name="50일선", line=dict(color="#16a34a", width=1.6)))
            fig_bt.add_trace(go.Scatter(x=plot_slice.index, y=plot_slice["MA200"], name="200일선", line=dict(color="#1d4ed8", width=2.2, dash="dot")))

            marker_styles = {
                "전략 A (수렴 + 거래량 급증)": ("triangle-up", "#0284c7", 12),
                "전략 B (수렴 + 골든크로스)": ("diamond", "#d97706", 12),
                "전략 C (수렴 + 거래량 + GC + 박스권돌파)": ("star", "#e11d48", 16),
            }
            for s_name, m_series in masks.items():
                valid_mask = m_series.iloc[start_off:]
                hit_dates = valid_mask[valid_mask].index
                if len(hit_dates) > 0:
                    sym, col, sz = marker_styles[s_name]
                    fig_bt.add_trace(go.Scatter(
                        x=hit_dates,
                        y=pdf.loc[hit_dates, "Close"],
                        mode="markers",
                        name=s_name,
                        marker=dict(symbol=sym, size=sz, color=col, line=dict(width=1, color="#ffffff"))
                    ))

            fig_bt.update_layout(
                title=f"{s_names.get(pick_code, pick_code)} ({pick_code}) 백테스트 구간 신호 발생 위치",
                height=480, hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(t=60, l=20, r=20, b=20)
            )
            st.plotly_chart(fig_bt, use_container_width=True)

            if not s_tr.empty:
                st.markdown("#### 📋 개별 매매 거래 내역")
                st.dataframe(s_tr.sort_values("신호일", ascending=False), use_container_width=True, hide_index=True, height=320)
                st.download_button(
                    "📥 검색 종목 백테스트 거래내역 CSV 다운로드",
                    data=s_tr.to_csv(index=False).encode("utf-8-sig"),
                    file_name=f"search_backtest_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )

    else:
        if not st.session_state.scan_done or not indicator_map:
            st.info("먼저 상단의 **[🚀 전체 시장 스캔 시작]** 버튼을 눌러 종목 데이터를 불러오거나, 위에서 **[🔎 원하는 종목 직접 검색해서 백테스트]**를 선택해 주세요.")
        else:
            if st.button("🧪 방금 스캔한 전체 종목으로 전략 A/B/C 백테스트 실행", type="primary"):
                with st.spinner("전략 A, B, C에 대해 10·20·40·60일 보유 성과를 계산 중입니다..."):
                    bt_sum, bt_tr = run_backtest_suite(indicator_map, used_cfg)
                    if not bt_tr.empty and not all_df.empty:
                        code_nm = all_df.set_index("종목코드")["종목명 및 종목코드"].to_dict()
                        bt_tr.insert(3, "종목명", bt_tr["종목코드"].map(lambda x: code_nm.get(str(x), str(x))))
                    st.session_state.bt_summary = bt_sum
                    st.session_state.bt_trades = bt_tr

            bt_sum = st.session_state.bt_summary
            bt_tr = st.session_state.bt_trades

            if bt_sum.empty:
                st.info("위 버튼을 누르시면 이미 스캔된 전체 종목에 대해 전략 A, B, C의 성과를 즉시 비교해 드립니다.")
            else:
                st.dataframe(bt_sum, use_container_width=True, hide_index=True)
                b_c1, b_c2 = st.columns(2)
                with b_c1:
                    fig_win = px.bar(
                        bt_sum, x="보유기간(일)", y="승률(%)", color="전략", barmode="group",
                        title="보유기간별 전략 승률(%) 비교", text="승률(%)"
                    )
                    fig_win.update_layout(height=380, legend=dict(orientation="h", yanchor="bottom", y=-0.38))
                    st.plotly_chart(fig_win, use_container_width=True)
                with b_c2:
                    fig_exp = px.bar(
                        bt_sum, x="보유기간(일)", y="기대수익(%)", color="전략", barmode="group",
                        title="보유기간별 1회 매매 기대수익(%) 비교", text="기대수익(%)"
                    )
                    fig_exp.add_hline(y=0, line_dash="dash", line_color="#495057")
                    fig_exp.update_layout(height=380, legend=dict(orientation="h", yanchor="bottom", y=-0.38))
                    st.plotly_chart(fig_exp, use_container_width=True)

                if not bt_tr.empty:
                    st.download_button(
                        "📥 전체 백테스트 거래내역 CSV 다운로드",
                        data=bt_tr.to_csv(index=False).encode("utf-8-sig"),
                        file_name=f"backtest_trades_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv"
                    )
