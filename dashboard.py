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
            p3["kind"] == "H" and p4["kind
