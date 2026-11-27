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

try:
    import yfinance as yf
    YF_AVAILABLE = True
except ImportError:
    YF_AVAILABLE = False

# 기존 4개 분석 모듈 연결 (수정 불필요)
from data_loader import (
    load_config, get_filtered_universe, fetch_stock_ohlcv,
    is_excluded_by_meta, validate_ohlcv
)
from indicators import compute_all_indicators
from signal_engine import evaluate_signal_at_index
from backtester import run_backtest_suite, build_strategy_masks

st.set_page_config(
    page_title="한·미 통합 이평선 수렴 · 엘리엇 파동 스윙 시스템",
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
    .idx-card {
        background: #ffffff;
        border-radius: 12px;
        padding: 12px 14px;
        border: 1px solid #e5e8eb;
        box-shadow: 0 2px 6px rgba(0,0,0,0.03);
        margin-bottom: 8px;
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
# 0. 미국 주요 지수 & S&P500·나스닥100 핵심 유니버스 정의
# ---------------------------------------------------------
US_INDICES = [
    {"code": "^GSPC", "name": "S&P 500 지수", "market": "미국지수", "unit": "PT"},
    {"code": "^NDX", "name": "나스닥 100 지수", "market": "미국지수", "unit": "PT"},
    {"code": "^IXIC", "name": "나스닥 종합 지수", "market": "미국지수", "unit": "PT"},
    {"code": "^DJI", "name": "다우존스 산업지수", "market": "미국지수", "unit": "PT"},
    {"code": "^RUT", "name": "러셀 2000 (중소형)", "market": "미국지수", "unit": "PT"},
    {"code": "^SOX", "name": "필라델피아 반도체", "market": "미국지수", "unit": "PT"},
    {"code": "^VIX", "name": "VIX 공포지수", "market": "미국지수", "unit": "PT"},
]

US_INDEX_ETFS = [
    {"code": "SPY", "name": "S&P 500 ETF (SPY)", "market": "지수ETF", "unit": "USD"},
    {"code": "QQQ", "name": "나스닥 100 ETF (QQQ)", "market": "지수ETF", "unit": "USD"},
    {"code": "DIA", "name": "다우존스 ETF (DIA)", "market": "지수ETF", "unit": "USD"},
    {"code": "IWM", "name": "러셀 2000 ETF (IWM)", "market": "지수ETF", "unit": "USD"},
    {"code": "SOXX", "name": "미국 반도체 ETF (SOXX)", "market": "지수ETF", "unit": "USD"},
]

# S&P 500 & 나스닥 100 대표 종목 (105개 주요주)
US_CORE_STOCKS = [
    # 빅테크 · AI · 반도체
    ("NVDA", "엔비디아", "나스닥100"), ("AAPL", "애플", "나스닥100"), ("MSFT", "마이크로소프트", "나스닥100"),
    ("AMZN", "아마존", "나스닥100"), ("GOOGL", "알파벳(구글)", "나스닥100"), ("META", "메타", "나스닥100"),
    ("TSLA", "테슬라", "나스닥100"), ("AVGO", "브로드컴", "나스닥100"), ("TSM", "TSMC", "NYSE"),
    ("AMD", "AMD", "나스닥100"), ("INTC", "인텔", "나스닥100"), ("QCOM", "퀄컴", "나스닥100"),
    ("MU", "마이크론", "나스닥100"), ("ASML", "ASML", "나스닥100"), ("AMAT", "어플라이드머티리얼즈", "나스닥100"),
    ("LRCX", "램리서치", "나스닥100"), ("KLAC", "KLA", "나스닥100"), ("TXN", "텍사스인스트루먼트", "나스닥100"),
    ("ADI", "아날로그디바이스", "나스닥100"), ("MRVL", "마벨테크놀로지", "나스닥100"), ("ARM", "ARM홀딩스", "나스닥100"),
    ("ON", "온세미컨덕터", "나스닥100"), ("NXPI", "NXP반도체", "나스닥100"), ("SNPS", "시높시스", "나스닥100"),
    ("CDNS", "케이던스", "나스닥100"),
    # 소프트웨어 · 클라우드 · 보안 · 통신
    ("ORCL", "오라클", "S&P500"), ("CRM", "세일즈포스", "S&P500"), ("ADBE", "어도비", "나스닥100"),
    ("NFLX", "넷플릭스", "나스닥100"), ("PLTR", "팔란티어", "나스닥100"), ("NOW", "서비스나우", "S&P500"),
    ("INTU", "인튜이트", "나스닥100"), ("CRWD", "크라우드스트라이크", "나스닥100"), ("PANW", "팔로알토네트웍스", "나스닥100"),
    ("FTNT", "포티넷", "나스닥100"), ("ZS", "지스케일러", "나스닥100"), ("DDOG", "데이터독", "나스닥100"),
    ("SNOW", "스노우플레이크", "NYSE"), ("SHOP", "쇼피파이", "나스닥100"), ("APP", "앱러빈", "나스닥100"),
    ("UBER", "우버", "S&P500"), ("ABNB", "에어비앤비", "나스닥100"), ("DASH", "도어대시", "나스닥100"),
    ("CSCO", "시스코", "나스닥100"), ("IBM", "IBM", "S&P500"), ("TMUS", "티모바일", "나스닥100"),
    ("VZ", "버라이즌", "S&P500"), ("T", "AT&T", "S&P500"), ("DIS", "월트디즈니", "S&P500"),
    # 바이오 · 헬스케어
    ("LLY", "일라이릴리", "S&P500"), ("NVO", "노보노디스크", "NYSE"), ("UNH", "유나이티드헬스", "S&P500"),
    ("JNJ", "존슨앤드존슨", "S&P500"), ("ABBV", "애브비", "S&P500"), ("MRK", "머크", "S&P500"),
    ("PFE", "화이자", "S&P500"), ("TMO", "써모피셔", "S&P500"), ("ABT", "애보트", "S&P500"),
    ("ISRG", "인튜이티브서지컬", "나스닥100"), ("AMGN", "암젠", "나스닥100"), ("GILD", "길리어드", "나스닥100"),
    ("VRTX", "버텍스", "나스닥100"), ("REGN", "리제네론", "나스닥100"), ("BMY", "브리스톨마이어스", "S&P500"),
    ("MRNA", "모더나", "S&P500"), ("DXCM", "덱스콤", "나스닥100"),
    # 금융 · 핀테크 · 크립토
    ("JPM", "JP모건", "S&P500"), ("BAC", "뱅크오브아메리카", "S&P500"), ("WFC", "웰스파고", "S&P500"),
    ("GS", "골드만삭스", "S&P500"), ("MS", "모건스탠리", "S&P500"), ("BLK", "블랙록", "S&P500"),
    ("V", "비자", "S&P500"), ("MA", "마스터카드", "S&P500"), ("AXP", "아메리칸익스프레스", "S&P500"),
    ("PYPL", "페이팔", "나스닥100"), ("COIN", "코인베이스", "S&P500"), ("HOOD", "로빈후드", "나스닥"),
    ("SOFI", "소파이", "나스닥"), ("SCHW", "찰스슈왑", "S&P500"),
    # 소비재 · 리테일
    ("WMT", "월마트", "S&P500"), ("COST", "코스트코", "나스닥100"), ("HD", "홈디포", "S&P500"),
    ("LOW", "로우스", "S&P500"), ("TGT", "타겟", "S&P500"), ("PG", "P&G", "S&P500"),
    ("KO", "코카콜라", "S&P500"), ("PEP", "펩시코", "나스닥100"), ("MCD", "맥도날드", "S&P500"),
    ("SBUX", "스타벅스", "나스닥100"), ("NKE", "나이키", "S&P500"), ("LULU", "룰루레몬", "나스닥100"),
    ("BKNG", "부킹홀딩스", "나스닥100"),
    # 산업 · 우주 · 방산 · 에너지 · 원전 · 전기차
    ("GE", "GE에어로스페이스", "S&P500"), ("GEV", "GE버노바", "S&P500"), ("CAT", "캐터필러", "S&P500"),
    ("BA", "보잉", "S&P500"), ("LMT", "록히드마틴", "S&P500"), ("RTX", "RTX", "S&P500"),
    ("HON", "허니웰", "나스닥100"), ("DE", "디어", "S&P500"), ("XOM", "엑슨모빌", "S&P500"),
    ("CVX", "셰브론", "S&P500"), ("NEE", "넥스트에라에너지", "S&P500"), ("CEG", "컨스텔레이션에너지", "나스닥100"),
    ("VST", "비스트라에너지", "S&P500"), ("ENPH", "엔페이즈에너지", "S&P500"), ("ALB", "앨버말", "S&P500"),
    ("F", "포드", "S&P500"), ("GM", "제너럴모터스", "S&P500"), ("RIVN", "리비안", "나스닥"),
    ("RKLB", "로켓랩", "나스닥"), ("IONQ", "아이온큐", "NYSE"),
]


def fmt_price(val: float, unit: str = "KRW") -> str:
    if pd.isna(val):
        return "-"
    if unit == "USD":
        return f"${val:,.2f}"
    if unit == "PT":
        return f"{val:,.2f} pt"
    return f"{int(round(val)):,}원"


def infer_unit(code: str, is_us_mode: bool = False) -> str:
    c = str(code).strip().upper()
    if c.startswith("^"):
        return "PT"
    if is_us_mode or (not c.isdigit()):
        return "USD"
    return "KRW"


# ---------------------------------------------------------
# 1. 엘리엇 파동 자동 탐지 & 현재 위치 판정 엔진 (KRW / USD / PT 공용)
# ---------------------------------------------------------
def detect_zigzag_pivots(df: pd.DataFrame, pct_threshold: float = 7.0) -> list[dict]:
    if df is None or len(df) < 15:
        return []

    pct = pct_threshold / 100.0
    highs = df["High"].values.astype(float)
    lows = df["Low"].values.astype(float)
    dates = df.index

    pivots = []
    direction = 0
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
    if not (p0["kind"] == "L" and p1["kind"] == "H" and p2["kind"] == "L" and
            p3["kind"] == "H" and p4["kind"] == "L" and p5["kind"] == "H"):
        return False, False
    w1 = p1["price"] - p0["price"]
    w3 = p3["price"] - p2["price"]
    w5 = p5["price"] - p4["price"]
    if w1 <= 0 or w3 <= 0 or w5 <= 0:
        return False, False
    if p2["price"] <= p0["price"]:
        return False, False
    if p3["price"] <= p1["price"]:
        return False, False
    if w3 < w1 and w3 < w5:
        return False, False
    if p4["price"] < p1["price"] * 0.97 or p4["price"] <= p2["price"]:
        return False, False
    if p5["price"] < p3["price"] * 0.98:
        return False, False
    return True, (p4["price"] >= p1["price"])


def analyze_elliott_wave(df: pd.DataFrame, pct_threshold: float = 7.0, unit: str = "KRW") -> dict:
    pivots = detect_zigzag_pivots(df, pct_threshold)
    cur_price = float(df["Close"].iloc[-1])
    cur_date = df.index[-1]

    if len(pivots) < 2:
        return {
            "ok": False,
            "message": "선택한 민감도(%)에서 뚜렷한 파동 변곡점이 부족합니다. 지수는 변동폭이 작으므로 '파동 민감도(%)'를 3~5%로 낮춰보세요.",
            "pivots": pivots,
        }

    labeled = []
    stage_title = ""
    stage_badge_color = "#191f28"
    stage_desc = ""
    action_tip = ""
    invalidation_price = None
    invalidation_label = ""
    fib_levels = []
    rule_notes = []
    matched = False
    n = len(pivots)

    # 1) 완료된 5파 이후 조정파(A / B / C) 탐색
    for tail_corr in [3, 2, 1, 0]:
        start_i = n - (6 + tail_corr)
        if start_i < 0:
            continue
        seg = pivots[start_i:]
        ok5, strict4 = _is_valid_bullish_impulse_5(*seg[:6])
        if not ok5:
            continue

        p0, p1, p2, p3, p4, p5 = seg[:6]
        labeled = [dict(p, label=lb, group="impulse") for p, lb in zip(seg[:6], ["0", "1", "2", "3", "4", "5"])]
        impulse_span = max(p5["price"] - p0["price"], 1e-6)
        rule_notes = [
            f"2파 저점({fmt_price(p2['price'], unit)}) > 0파({fmt_price(p0['price'], unit)}) 충족",
            f"3파 길이({fmt_price(p3['price']-p2['price'], unit)}) 최단 금지 규칙 충족",
            "4파 1파 고점 미침범 엄격 충족" if strict4 else "4파 1파 고점 근접(허용 범위 내)"
        ]

        if tail_corr == 0:
            if cur_price < p5["price"] * 0.98:
                stage_title = "📉 5파 완료 후 조정 A파 진행 중"
                stage_badge_color = "#1d4ed8"
                stage_desc = f"상승 5파 고점({fmt_price(p5['price'], unit)}) 형성 후 첫 하락 조정(A파)이 진행 중입니다."
                action_tip = "추격 매수를 피하고 전체 5파 상승폭의 38.2%~50% 되돌림 지지 여부를 관찰하세요."
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
                stage_desc = f"상승 5파({fmt_price(p5['price'], unit)})가 정점에 도달했거나 연장 진행 중입니다."
                action_tip = "신규 진입보다 분할 익절 및 4파 저점 방어 여부 경계가 유리한 구간입니다."
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
            stage_desc = f"5파 고점({fmt_price(p5['price'], unit)})에서 A파({fmt_price(pA['price'], unit)}) 하락 후 반등(B파) 구간입니다."
            action_tip = "B파 반등은 A파 하락폭의 50~61.8% 부근에서 저항을 받고 C파로 이어질 수 있어 주의가 필요합니다."
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
            stage_desc = f"B파 반등 고점({fmt_price(pB['price'], unit)}) 이후 마지막 조정 파동인 C파가 진행 중입니다."
            action_tip = "C파가 A파 길이의 1.0배 부근에
