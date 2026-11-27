import io
import copy
import json
import urllib.request
import urllib.parse
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
    page_title="한·미 이평선 수렴·거래량 돌파 스윙 시스템",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    .block-container { padding-top: 1.1rem; padding-bottom: 2.5rem; }
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
    .macro-card {
        background: #ffffff;
        border-radius: 12px;
        padding: 12px 14px;
        border: 1px solid #e5e8eb;
        box-shadow: 0 2px 6px rgba(0,0,0,0.03);
        margin-bottom: 8px;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# 0. 주요 글로벌 지수·환율 및 미국 대표 종목 기본 마스터
# ---------------------------------------------------------
MACRO_SYMBOLS = [
    {"symbol": "^GSPC", "name": "S&P 500", "unit": "INDEX", "sub": "미국 대형주 500"},
    {"symbol": "^NDX", "name": "나스닥 100", "unit": "INDEX", "sub": "미국 기술주 100"},
    {"symbol": "KRW=X", "name": "원/달러 환율", "unit": "FX", "sub": "USD/KRW"},
    {"symbol": "^KS11", "name": "코스피", "unit": "INDEX", "sub": "KOSPI 종합"},
    {"symbol": "^VIX", "name": "VIX 공포지수", "unit": "VIX", "sub": "미국 변동성 지수"},
]

US_CORE_STOCKS = [
    # 빅테크 · AI · 반도체
    ("NVDA", "엔비디아"), ("AAPL", "애플"), ("MSFT", "마이크로소프트"), ("AMZN", "아마존"),
    ("GOOGL", "알파벳(구글)"), ("META", "메타"), ("TSLA", "테슬라"), ("AVGO", "브로드컴"),
    ("TSM", "TSMC"), ("AMD", "AMD"), ("INTC", "인텔"), ("QCOM", "퀄컴"),
    ("MU", "마이크론"), ("ASML", "ASML"), ("TXN", "텍사스인스트루먼트"), ("AMAT", "어플라이드머티리얼즈"),
    ("LRCX", "램리서치"), ("KLAC", "KLA"), ("ADI", "아날로그디바이스"), ("ARM", "ARM홀딩스"),
    ("MRVL", "마벨테크놀로지"), ("ON", "온세미컨덕터"), ("SMCI", "슈퍼마이크로"), ("DELL", "델테크놀로지스"),
    # 소프트웨어 · 클라우드 · 사이버보안
    ("ORCL", "오라클"), ("CRM", "세일즈포스"), ("ADBE", "어도비"), ("NFLX", "넷플릭스"),
    ("PLTR", "팔란티어"), ("NOW", "서비스나우"), ("INTU", "인튜이트"), ("PANW", "팔로알토네트웍스"),
    ("CRWD", "크라우드스트라이크"), ("SNPS", "시놉시스"), ("CDNS", "케이던스"), ("FTNT", "포티넷"),
    ("SNOW", "스노우플레이크"), ("DDOG", "데이터독"), ("ZS", "지스케일러"), ("NET", "클라우드플레어"),
    ("SHOP", "쇼피파이"), ("APP", "앱러빈"), ("IBM", "IBM"), ("CSCO", "시스코"),
    # 전기차 · 우주 · 방산 · 산업 · 전력/원전
    ("UBER", "우버"), ("RIVN", "리비안"), ("LCID", "루시드"), ("F", "포드"), ("GM", "제너럴모터스"),
    ("BA", "보잉"), ("CAT", "캐터필러"), ("DE", "디어"), ("GE", "GE에어로스페이스"),
    ("GEV", "GE버노바"), ("LMT", "록히드마틴"), ("RTX", "RTX"), ("NOC", "노스롭그루먼"),
    ("HON", "허니웰"), ("UPS", "UPS"), ("FDX", "페덱스"), ("UNP", "유니온퍼시픽"),
    ("CEG", "컨스텔레이션에너지"), ("VST", "비스트라에너지"), ("NEE", "넥스트에라에너지"),
    ("ENPH", "엔페이즈에너지"), ("FSLR", "퍼스트솔라"), ("RKLB", "로켓랩"), ("IONQ", "아이온큐"),
    # 바이오 · 헬스케어
    ("LLY", "일라이릴리"), ("NVO", "노보노디스크"), ("UNH", "유나이티드헬스"), ("JNJ", "존슨앤드존슨"),
    ("ABBV", "애브비"), ("MRK", "머크"), ("PFE", "화이자"), ("TMO", "써모피셔"),
    ("ABT", "애보트"), ("DHR", "다나허"), ("AMGN", "암젠"), ("GILD", "길리어드"),
    ("ISRG", "인튜이티브서지컬"), ("VRTX", "버텍스"), ("REGN", "리제네론"), ("BMY", "브리스톨마이어스"),
    ("MRNA", "모더나"), ("CVS", "CVS헬스"),
    # 금융 · 핀테크 · 크립토
    ("BRK-B", "버크셔해서웨이B"), ("JPM", "JP모건"), ("V", "비자"), ("MA", "마스터카드"),
    ("BAC", "뱅크오브아메리카"), ("WFC", "웰스파고"), ("GS", "골드만삭스"), ("MS", "모건스탠리"),
    ("AXP", "아메리칸익스프레스"), ("BLK", "블랙록"), ("SCHW", "찰스슈왑"), ("PYPL", "페이팔"),
    ("COIN", "코인베이스"), ("HOOD", "로빈후드"), ("SOFI", "소파이"), ("MSTR", "스트래티지"),
    # 소비재 · 리테일 · 에너지 · 통신
    ("WMT", "월마트"), ("COST", "코스트코"), ("HD", "홈디포"), ("LOW", "로우스"),
    ("TGT", "타겟"), ("PG", "프록터앤갬블"), ("KO", "코카콜라"), ("PEP", "펩시코"),
    ("MCD", "맥도날드"), ("SBUX", "스타벅스"), ("NKE", "나이키"), ("LULU", "룰루레몬"),
    ("DIS", "디즈니"), ("BKNG", "부킹홀딩스"), ("ABNB", "에어비앤비"), ("SPOT", "스포티파이"),
    ("XOM", "엑슨모빌"), ("CVX", "셰브론"), ("COP", "코노코필립스"), ("SLB", "슐럼버거"),
    ("ALB", "앨버말"), ("FCX", "프리포트맥모란"), ("T", "AT&T"), ("VZ", "버라이즌"), ("TMUS", "T모바일"),
    # 대표 지수 ETF
    ("SPY", "S&P500 ETF"), ("QQQ", "나스닥100 ETF"), ("SOXX", "필라델피아반도체 ETF"), ("DIA", "다우존스 ETF"),
]


def fmt_price(val: float, currency_mode: str = "KRW") -> str:
    if currency_mode == "USD":
        return f"${val:,.2f}"
    if currency_mode == "FX":
        return f"{val:,.2f}원"
    if currency_mode in ("INDEX", "VIX"):
        return f"{val:,.2f}"
    return f"{int(round(val)):,}원"


# ---------------------------------------------------------
# 1. Yahoo Finance 직결 수집기 (지수·환율·미국 주식용, 별도 패키지 불필요)
# ---------------------------------------------------------
def fetch_yahoo_chart_raw(symbol: str, range_str: str = "3y") -> pd.DataFrame | None:
    """Yahoo Finance v8 JSON 차트 API로 지수·환율·미국주식 일봉 수집"""
    try:
        encoded = urllib.parse.quote(symbol)
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{encoded}?range={range_str}&interval=1d"
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
            }
        )
        with urllib.request.urlopen(req, timeout=6) as res:
            payload = json.loads(res.read().decode("utf-8"))
        results = payload.get("chart", {}).get("result")
        if not results:
            return None
        item = results[0]
        timestamps = item.get("timestamp", [])
        quotes = item.get("indicators", {}).get("quote", [])
        if not timestamps or not quotes:
            return None
        q = quotes[0]
        dates = pd.to_datetime(timestamps, unit="s", utc=True).tz_convert("America/New_York").tz_localize(None).normalize()
        df = pd.DataFrame({
            "Open": q.get("open", []),
            "High": q.get("high", []),
            "Low": q.get("low", []),
            "Close": q.get("close", []),
            "Volume": q.get("volume", []),
        }, index=dates)
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df["Volume"] = df["Volume"].fillna(0.0)
        df = df.dropna(subset=["Open", "High", "Low", "Close"])
        df = df[(df["Close"] > 0) & (df["High"] >= df["Low"])]
        df = df[~df.index.duplicated(keep="last")].sort_index()
        return df if not df.empty else None
    except Exception:
        return None


def fetch_us_ohlcv(symbol: str, min_days: int = 460, range_str: str = "3y") -> pd.DataFrame | None:
    """미국 개별 종목 일봉 수집 (Yahoo v8 -> yfinance -> FDR 순차 안전 폴백)"""
    df = fetch_yahoo_chart_raw(symbol, range_str=range_str)
    if df is not None:
        cleaned = validate_ohlcv(df, min_days)
        if cleaned is not None:
            return cleaned

    if YF_AVAILABLE:
        try:
            raw = yf.download(symbol, period=range_str, auto_adjust=False, progress=False)
            if raw is not None and not raw.empty:
                if isinstance(raw.columns, pd.MultiIndex):
                    raw.columns = raw.columns.get_level_values(0)
                cleaned = validate_ohlcv(raw, min_days)
                if cleaned is not None:
                    return cleaned
        except Exception:
            pass

    if FDR_AVAILABLE:
        try:
            days = 1850 if range_str == "5y" else 1150
            start_s = (datetime.today() - timedelta(days=days)).strftime("%Y-%m-%d")
            raw = fdr.DataReader(symbol, start_s)
            cleaned = validate_ohlcv(raw, min_days)
            if cleaned is not None:
                return cleaned
        except Exception:
            pass

    return None


@st.cache_data(ttl=300)
def load_macro_dashboard_data() -> dict:
    """S&P 500, 나스닥 100, 원/달러 환율, 코스피, VIX 최근 3년 데이터 일괄 수집"""
    macro_map = {}

    def _fetch_one(info):
        sym = info["symbol"]
        df = fetch_yahoo_chart_raw(sym, range_str="3y")
        if df is None and FDR_AVAILABLE:
            try:
                fdr_sym = "USD/KRW" if sym == "KRW=X" else ("KS11" if sym == "^KS11" else sym)
                start_s = (datetime.today() - timedelta(days=1150)).strftime("%Y-%m-%d")
                raw = fdr.DataReader(fdr_sym, start_s)
                if raw is not None and not raw.empty:
                    if "Volume" not in raw.columns:
                        raw["Volume"] = 0.0
                    df = raw[["Open", "High", "Low", "Close", "Volume"]].dropna(subset=["Close"]).sort_index()
            except Exception:
                df = None
        return sym, df

    with ThreadPoolExecutor(max_workers=5) as ex:
        futs = [ex.submit(_fetch_one, item) for item in MACRO_SYMBOLS]
        for f in as_completed(futs):
            sym, df = f.result()
            if df is not None and len(df) >= 2:
                macro_map[sym] = df

    return macro_map


@st.cache_data(ttl=3600)
def get_us_stock_master() -> pd.DataFrame:
    """미국 주식 마스터 (내장 핵심 120+종목 + S&P500 전체 자동 병합)"""
    core_dict = {tk: nm for tk, nm in US_CORE_STOCKS}
    rows = [{"code": tk, "name": nm, "market": "미국(US)", "mcap_eok": 50000, "delisted": False} for tk, nm in US_CORE_STOCKS]
    seen = set(core_dict.keys())

    if FDR_AVAILABLE:
        try:
            sp500 = fdr.StockListing("S&P500")
            if sp500 is not None and not sp500.empty and "Symbol" in sp500.columns:
                for _, r in sp500.iterrows():
                    sym = str(r["Symbol"]).strip().replace(".", "-")
                    nm = str(r.get("Name", sym)).strip()
                    if sym and sym not in seen:
                        seen.add(sym)
                        rows.append({"code": sym, "name": nm, "market": "미국(US)", "mcap_eok": 30000, "delisted": False})
        except Exception:
            pass

    return pd.DataFrame(rows)


# ---------------------------------------------------------
# 2. 미국 주식 평가 결과 USD($) + KRW 환산 포맷팅 보정기
# ---------------------------------------------------------
def format_us_eval_result(res: dict, ind_df: pd.DataFrame, usd_krw: float) -> dict:
    """signal_engine 결과를 미국 주식 달러($) 소수점 2자리 및 원화 환산 기준으로 정교하게 포맷"""
    row = ind_df.iloc[-1]
    price = float(row["Close"])
    box_high = float(row["Box_High"])
    box_low = float(row["Box_Low"])
    ma50 = float(row["MA50"])
    atr14 = float(row["ATR14"])
    recent_res = float(row["Res_120"]) if pd.notna(row["Res_120"]) else box_high * 1.15

    # 달러($) 소수점 2자리 정밀 손절/목표가 재계산
    stop_loss = min(price * 0.97, max(box_low * 0.985, price - 2.0 * atr14))
    if stop_loss >= price * 0.985:
        stop_loss = price * 0.94
    stop_loss = round(stop_loss, 2)

    box_height = max(box_high - box_low, atr14 * 2.0)
    target1 = round(max(box_high + box_height * 0.9, price + 2.8 * atr14, price * 1.07), 2)
    if recent_res > price * 1.08:
        target1 = round(min(target1, recent_res) if recent_res >= price + 1.8 * atr14 else target1, 2)
    target2 = round(max(box_high + box_height * 1.8, price + 4.8 * atr14, target1 * 1.08), 2)

    risk_pct = round(((price - stop_loss) / price) * 100.0, 2)
    reward1_pct = round(((target1 - price) / price) * 100.0, 2)
    reward2_pct = round(((target2 - price) / price) * 100.0, 2)
    rr_ratio = round(reward1_pct / max(risk_pct, 0.1), 2)

    out = dict(res)
    out["현재가"] = round(price, 2)
    out["현재가 표시"] = f"${price:,.2f} (약 {int(round(price * usd_krw)):,}원)"
    out["박스권 상단 및 하단"] = f"${box_high:,.2f} / ${box_low:,.2f}"
    out["예상 손익비"] = rr_ratio
    out["매매 진입 및 손절 기준"] = (
        f"진입: ${price:,.2f} (약 {int(round(price * usd_krw)):,}원) | "
        f"손절: ${stop_loss:,.2f}(-{risk_pct}%) | "
        f"1차: ${target1:,.2f}(+{reward1_pct}%) | 2차: ${target2:,.2f}(+{reward2_pct}%) | "
        f"손익비: {rr_ratio:.2f}"
    )
    out["상승 시나리오 무효화 조건"] = (
        f"종가 기준 박스권 하단(${box_low:,.2f}) 이탈 또는 "
        f"50일선(${ma50:,.2f}) 하향 이탈 시 무효화"
    )
    out["신호 포착 근거"] = (
        f"250일 중 {out['역배열 유지 기간(일)']}일 역배열(1년 고점 대비 {out['최근 1년 고점 대비 하락률(%)']:.1f}%), "
        f"박스권(${box_low:,.2f}~${box_high:,.2f}), "
        f"수렴상태 '{out['수렴 상태']}'(20일 축소율 {out['20일간 이격도 축소율(%)']:+.1f}%), "
        f"거래량 {out['거래량 증가율(배)']:.2f}배, 골든크로스: {out['골든크로스 발생 여부 및 종류']}"
    )
    return out


# ---------------------------------------------------------
# 3. 엘리엇 파동 자동 탐지 & 현재 위치 판정 엔진 (KRW / USD / INDEX 통합)
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
            "idx": ext_idx, "date": dates[ext_idx], "price": ext_p,
            "kind": "H" if direction == 1 else "L"
        })
    return pivots


def _is_valid_bullish_impulse_5(p0, p1, p2, p3, p4, p5) -> tuple[bool, bool]:
    if not (p0["kind"] == "L" and p1["kind"] == "H" and p2["kind"] == "L" and
            p3["kind"] == "H" and p4["kind"] == "L" and p5["kind"] == "H"):
        return False, False
    w1, w3, w5 = p1["price"] - p0["price"], p3["price"] - p2["price"], p5["price"] - p4["price"]
    if w1 <= 0 or w3 <= 0 or w5 <= 0:
        return False, False
    if p2["price"] <= p0["price"] or p3["price"] <= p1["price"]:
        return False, False
    if w3 < w1 and w3 < w5:
        return False, False
    if p4["price"] < p1["price"] * 0.97 or p4["price"] <= p2["price"]:
        return False, False
    if p5["price"] < p3["price"] * 0.98:
        return False, False
    return True, (p4["price"] >= p1["price"])


def analyze_elliott_wave(df: pd.DataFrame, pct_threshold: float = 7.0, currency_mode: str = "KRW") -> dict:
    pivots = detect_zigzag_pivots(df, pct_threshold)
    cur_price = float(df["Close"].iloc[-1])
    cur_date = df.index[-1]
    fp = lambda v: fmt_price(v, currency_mode)

    if len(pivots) < 2:
        return {
            "ok": False,
            "message": "선택한 민감도(%)에서 뚜렷한 파동 변곡점이 부족합니다. '파동 민감도(%)'를 낮추거나 분석 기간을 늘려보세요.",
            "pivots": pivots,
        }

    labeled = []
    stage_title, stage_badge_color = "", "#191f28"
    stage_desc, action_tip = "", ""
    invalidation_price, invalidation_label = None, ""
    fib_levels, rule_notes = [], []
    matched = False
    n = len(pivots)

    # 1) 5파 완료 후 조정파(A/B/C)
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
            f"2파 저점({fp(p2['price'])}) > 0파 시작점({fp(p0['price'])}) 충족",
            "3파 최단 금지 규칙 충족",
            "4파 1파 고점 미침범 엄격 충족" if strict4 else "4파 1파 고점 근접(허용 범위 내)"
        ]

        if tail_corr == 0:
            if cur_price < p5["price"] * 0.98:
                stage_title = "📉 5파 완료 후 조정 A파 진행 중"
                stage_badge_color = "#1d4ed8"
                stage_desc = f"상승 5파 고점({fp(p5['price'])}) 형성 후 첫 번째 하락 조정(A파)이 진행 중입니다."
                action_tip = "추격 매수를 피하고 전체 5파 상승폭의 38.2%~50% 지지 여부를 관찰하세요."
                invalidation_price, invalidation_label = p5["price"], "5파 고점 재돌파 시 연장파 전환"
                fib_levels = [
                    ("5파 23.6% 되돌림", p5["price"] - impulse_span * 0.236, "#64748b"),
                    ("5파 38.2% 지지", p5["price"] - impulse_span * 0.382, "#0284c7"),
                    ("5파 50.0% 지지", p5["price"] - impulse_span * 0.500, "#1d4ed8"),
                ]
            else:
                stage_title = "🏔️ 상승 5파 고점 부근 (추세 말기 경계)"
                stage_badge_color = "#d97706"
                stage_desc = f"상승 5파({fp(p5['price'])})가 정점에 도달했거나 연장 진행 중입니다."
                action_tip = "신규 스윙 진입보다 분할 익절 및 4파 저점 이탈 경계가 유리합니다."
                invalidation_price, invalidation_label = p4["price"], "4파 저점 이탈 시 5파 종료 확정"
                fib_levels = [("5파 고점(저항)", p5["price"], "#e11d48"), ("4파 저점(지지)", p4["price"], "#0284c7")]
            matched = True
            break
        elif tail_corr == 1:
            pA = seg[6]
            labeled.append(dict(pA, label="A", group="correction"))
            a_span = max(p5["price"] - pA["price"], 1e-6)
            stage_title, stage_badge_color = "⛅ 조정 A파 완료 → B파 반등 진행 중", "#d97706"
            stage_desc = f"5파 고점({fp(p5['price'])})에서 A파({fp(pA['price'])}) 하락 후 되돌림 반등(B파) 중입니다."
            action_tip = "B파 반등은 A파 하락폭의 50~61.8%에서 저항을 받고 C파로 이어질 수 있어 주의하세요."
            invalidation_price, invalidation_label = pA["price"], "A파 저점 이탈 시 C파 본격화"
            fib_levels = [
                ("A파 38.2% 저항", pA["price"] + a_span * 0.382, "#f59f00"),
                ("A파 50.0% 저항", pA["price"] + a_span * 0.500, "#ea580c"),
                ("A파 61.8% 저항", pA["price"] + a_span * 0.618, "#dc2626"),
            ]
            matched = True
            break
        elif tail_corr == 2:
            pA, pB = seg[6], seg[7]
            labeled.extend([dict(pA, label="A", group="correction"), dict(pB, label="B", group="correction")])
            a_span = max(p5["price"] - pA["price"], 1e-6)
            stage_title, stage_badge_color = "🌧️ 조정 C파 하락 진행 중 (최종 바닥 탐색)", "#1d4ed8"
            stage_desc = f"B파 반등 고점({fp(pB['price'])}) 이후 마지막 조정 파동인 C파가 진행 중입니다."
            action_tip = "C파가 A파 길이의 1.0배 부근에서 거래량과 이평선 수렴이 나오는지 확인하세요."
            invalidation_price, invalidation_label = pB["price"], "B파 고점 돌파 시 조정 종료"
            fib_levels = [
                ("C파 0.618배 목표", max(pB["price"] - a_span * 0.618, 0.01), "#0284c7"),
                ("C파 1.0배 대칭 바닥", max(pB["price"] - a_span * 1.0, 0.01), "#1d4ed8"),
            ]
            matched = True
            break
        elif tail_corr == 3:
            pA, pB, pC = seg[6], seg[7], seg[8]
            labeled.extend([dict(pA, label="A", group="correction"), dict(pB, label="B", group="correction"), dict(pC, label="C", group="correction")])
            stage_title, stage_badge_color = "🌱 ABC 조정 완료 → 새로운 상승 1파 전환 시도", "#16a34a"
            stage_desc = f"C파 저점({fp(pC['price'])}) 지지 후 새로운 상승 1파를 시작하려는 초기 전환 구간입니다."
            action_tip = "C파 저점을 손절선으로 두고 B파 고점 돌파 및 거래량 동반 여부를 관찰하세요."
            invalidation_price, invalidation_label = pC["price"], "C파 저점 이탈 시 하락 연장"
            fib_levels = [("1차 돌파 기준 (B파 고점)", pB["price"], "#16a34a"), ("바닥 지지 (C파 저점)", pC["price"], "#1d4ed8")]
            matched = True
            break

    # 2) 진행 중인 상승 충격파 (5파 / 4파 / 3파)
    if not matched and n >= 5:
        p0, p1, p2, p3, p4 = pivots[-5:]
        if (p0["kind"] == "L" and p1["kind"] == "H" and p2["kind"] == "L" and p3["kind"] == "H" and p4["kind"] == "L" and
                p2["price"] > p0["price"] and p3["price"] > p1["price"] and
                p4["price"] >= p1["price"] * 0.97 and p4["price"] < p3["price"] and cur_price >= p4["price"]):
            labeled = [dict(p, label=lb, group="impulse") for p, lb in zip([p0, p1, p2, p3, p4], ["0", "1", "2", "3", "4"])]
            w1, w03 = max(p1["price"] - p0["price"], 1e-6), max(p3["price"] - p0["price"], 1e-6)
            stage_title = "🔥 상승 5파 진행 중 (3파 고점 돌파)" if cur_price > p3["price"] else "📈 상승 5파 진행 중 (3파 고점 돌파 시도)"
            stage_badge_color = "#ea580c"
            stage_desc = f"4파 저점({fp(p4['price'])}) 지지 후 마지막 상승 파동인 5파가 전개 중입니다."
            action_tip = "추세 후반부이므로 1파 대칭(1.0배) 또는 0~3파의 0.618배 구간에서 분할 익절을 고려하세요."
            invalidation_price, invalidation_label = p4["price"], "4파 저점 이탈 시 5파 무효화"
            fib_levels = [
                ("3파 고점 (돌파 기준)", p3["price"], "#f59f00"),
                ("5파 1차 목표 (1파 1.0배)", p4["price"] + w1, "#ea580c"),
                ("5파 2차 목표 (0~3파 0.618배)", p4["price"] + w03 * 0.618, "#dc2626"),
            ]
            rule_notes = [f"2파({fp(p2['price'])}) > 0파({fp(p0['price'])}) 충족", f"3파({fp(p3['price'])}) > 1파({fp(p1['price'])}) 충족"]
            matched = True

    if not matched and n >= 4:
        p0, p1, p2, p3 = pivots[-4:]
        if (p0["kind"] == "L" and p1["kind"] == "H" and p2["kind"] == "L" and p3["kind"] == "H" and
                p2["price"] > p0["price"] and p3["price"] > p1["price"] and
                p1["price"] * 0.96 <= cur_price <= p3["price"]):
            labeled = [dict(p, label=lb, group="impulse") for p, lb in zip([p0, p1, p2, p3], ["0", "1", "2", "3"])]
            w3 = max(p3["price"] - p2["price"], 1e-6)
            stage_title, stage_badge_color = "⏸️ 상승 4파 조정 진행 중 (5파 대기)", "#d97706"
            stage_desc = f"강한 3파 상승({fp(p2['price'])} → {fp(p3['price'])}) 이후 숨고르기(4파 조정)가 진행 중입니다."
            action_tip = "3파 상승폭의 23.6%~38.2% 지지와 1파 고점(방어선) 유지 여부를 확인하세요."
            invalidation_price, invalidation_label = p1["price"], "1파 고점 이탈 시 4파 규칙 위반"
            fib_levels = [
                ("3파 23.6% 지지", p3["price"] - w3 * 0.236, "#16a34a"),
                ("3파 38.2% 핵심 지지", p3["price"] - w3 * 0.382, "#0284c7"),
                ("1파 고점 (방어선)", p1["price"], "#dc2626"),
            ]
            rule_notes = [f"2파({fp(p2['price'])}) > 0파({fp(p0['price'])}) 충족", f"현재가 1파 고점({fp(p1['price'])}) 상단 유지"]
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
            stage_desc = f"2파 조정({ret_ratio}% 되돌림, {fp(p2['price'])}) 이후 가장 강한 추세가 나오는 3파 구간입니다."
            action_tip = "스윙 트레이딩 핵심 구간입니다. 2파 저점을 손절선으로 두고 1파 대비 1.0~1.618배 확장 목표를 추적하세요."
            invalidation_price, invalidation_label = p2["price"], "2파 저점 이탈 시 상승 3파 무효화"
            fib_levels = [
                ("1파 고점 (돌파선)", p1["price"], "#f59f00"),
                ("3파 1차 목표 (1.0배)", p2["price"] + w1 * 1.0, "#ea580c"),
                ("3파 핵심 목표 (1.618배)", p2["price"] + w1 * 1.618, "#dc2626"),
            ]
            rule_notes = [f"2파 저점({fp(p2['price'])})이 0파({fp(p0['price'])}) 상단 지지 (되돌림 {ret_ratio}%)"]
            matched = True

    # 3) 하락 A-B-C 구조
    if not matched and n >= 4:
        p0, pA, pB, pC = pivots[-4:]
        if (p0["kind"] == "H" and pA["kind"] == "L" and pB["kind"] == "H" and pC["kind"] == "L" and
                pA["price"] < p0["price"] and pB["price"] < p0["price"] and pB["price"] > pA["price"] and
                pC["price"] <= pA["price"] * 1.03 and cur_price >= pC["price"]):
            labeled = [dict(p, label=lb, group="correction") for p, lb in zip([p0, pA, pB, pC], ["0", "A", "B", "C"])]
            a_span = max(p0["price"] - pA["price"], 1e-6)
            c_ratio = round(((pB["price"] - pC["price"]) / a_span) * 100.0, 1)
            stage_title, stage_badge_color = "🌱 하락 ABC 완료 → 상승 1파 전환 초기", "#16a34a"
            stage_desc = f"고점({fp(p0['price'])}) 이후 하락 A-B-C(저점 {fp(pC['price'])})를 마치고 반등 전환을 시도 중입니다."
            action_tip = "장기 하락 후 바닥 다지기·이평선 수렴 전략과 가장 잘 맞는 구간입니다. B파 고점 돌파와 거래량을 체크하세요."
            invalidation_price, invalidation_label = pC["price"], "C파 저점 이탈 시 하락 연장"
            fib_levels = [
                ("1차 저항 (B파 고점)", pB["price"], "#16a34a"),
                ("하락폭 38.2% 회복선", pC["price"] + (p0["price"] - pC["price"]) * 0.382, "#f59f00"),
                ("하락폭 50.0% 회복선", pC["price"] + (p0["price"] - pC["price"]) * 0.500, "#ea580c"),
            ]
            rule_notes = [f"고점 대비 하락 3파(A→B→C) 완성 (C/A 비율 {c_ratio}%)"]
            matched = True

    if not matched and n >= 3:
        p0, pA, pB = pivots[-3:]
        if (p0["kind"] == "H" and pA["kind"] == "L" and pB["kind"] == "H" and
                pA["price"] < p0["price"] and pB["price"] < p0["price"] and pB["price"] > pA["price"] and cur_price <= pB["price"]):
            labeled = [dict(p, label=lb, group="correction") for p, lb in zip([p0, pA, pB], ["0", "A", "B"])]
            a_span = max(p0["price"] - pA["price"], 1e-6)
            stage_title, stage_badge_color = "🌧️ 하락 C파 진행 중 (조정 마무리 탐색)", "#1d4ed8"
            stage_desc = f"B파 반등({fp(pB['price'])}) 저항 이후 마지막 하락 파동인 C파가 진행 중입니다."
            action_tip = "A파 저점 부근 또는 C파 1.0배 대칭 구간에서 박스권 지지와 거래량 증가를 기다리세요."
            invalidation_price, invalidation_label = pB["price"], "B파 고점 돌파 시 하락 시나리오 종료"
            fib_levels = [
                ("A파 저점 (지지 테스트)", pA["price"], "#0284c7"),
                ("C파 1.0배 대칭 바닥", max(pB["price"] - a_span * 1.0, 0.01), "#1d4ed8"),
            ]
            rule_notes = [f"고점({fp(p0['price'])}) → A파({fp(pA['price'])}) → B파({fp(pB['price'])}) 확인"]
            matched = True

    if not matched and n >= 2:
        p0, p1 = pivots[-2:]
        if p0["kind"] == "L" and p1["kind"] == "H" and p1["price"] > p0["price"] and p0["price"] < cur_price <= p1["price"]:
            labeled = [dict(p, label=lb, group="impulse") for p, lb in zip([p0, p1], ["0", "1"])]
            w1 = max(p1["price"] - p0["price"], 1e-6)
            cur_ret = round(((p1["price"] - cur_price) / w1) * 100.0, 1)
            stage_title, stage_badge_color = "🎯 상승 2파 조정 진행 중 (3파 매수 준비)", "#0284c7"
            stage_desc = f"바닥({fp(p0['price'])})에서 첫 상승 1파({fp(p1['price'])}) 이후 {cur_ret}% 되돌림 조정(2파) 중입니다."
            action_tip = "1파의 38.2%~61.8% 구간에서 거래량 감소 후 지지가 나오면 3파를 노리는 최적의 스윙 준비 구간입니다."
            invalidation_price, invalidation_label = p0["price"], "0파 바닥 이탈 시 1파 무효화"
            fib_levels = [
                ("1파 38.2% 눌림목", p1["price"] - w1 * 0.382, "#16a34a"),
                ("1파 50.0% 눌림목", p1["price"] - w1 * 0.500, "#0284c7"),
                ("1파 61.8% 핵심 지지", p1["price"] - w1 * 0.618, "#1d4ed8"),
            ]
            rule_notes = [f"현재 1파 대비 {cur_ret}% 되돌림 진행 중 (0파 미이탈)"]
            matched = True

    # 4) 복합 횡보·삼각수렴 폴백
    if not matched:
        recent_pivs = pivots[-min(len(pivots), 6):]
        labeled = [dict(p, label=f"{p['kind']}{i+1}", group="complex") for i, p in enumerate(recent_pivs)]
        recent_high = max(p["price"] for p in recent_pivs)
        recent_low = min(p["price"] for p in recent_pivs)
        if cur_price >= recent_pivs[-1]["price"]:
            stage_title, stage_badge_color = "📦 복합 조정 후 단기 반등 파동 진행 중", "#7c3aed"
            stage_desc = f"최근 변곡점({fp(recent_pivs[-1]['price'])}) 이후 반등 중이며 박스권/복합 조정 구조입니다."
        else:
            stage_title, stage_badge_color = "📦 복합 횡보 / 삼각수렴 조정 진행 중", "#64748b"
            stage_desc = f"고점({fp(recent_high)})과 저점({fp(recent_low)}) 사이에서 방향을 모색 중입니다."
        action_tip = "파동 민감도(%)를 1~2% 조절해 대파동/소파동을 비교하고 박스권 상단 돌파를 확인하세요."
        invalidation_price, invalidation_label = recent_low, "최근 주요 저점 지지선"
        fib_levels = [
            ("최근 고점 (돌파선)", recent_high, "#e11d48"),
            ("중심가 (50%)", (recent_high + recent_low) / 2.0, "#f59f00"),
            ("최근 저점 (지지선)", recent_low, "#1d4ed8"),
        ]
        rule_notes = ["복합 조정(박스권/삼각수렴) 구간 · 민감도 조정 시 상위/하위 파동 확인 가능"]

    wave_rows = []
    for i in range(len(labeled) - 1):
        a, b = labeled[i], labeled[i + 1]
        chg = ((b["price"] - a["price"]) / max(a["price"], 1e-6)) * 100.0
        wave_rows.append({
            "파동 구간": f"{a['label']} → {b['label']}",
            "시작일": a["date"].strftime("%Y-%m-%d"),
            "종료일": b["date"].strftime("%Y-%m-%d"),
            "시작가": fp(a["price"]),
            "종료가": fp(b["price"]),
            "등락률(%)": round(chg, 2),
            "소요 거래일": f"{int(b['idx'] - a['idx'])}일"
        })

    last_lbl = labeled[-1]
    cur_chg = ((cur_price - last_lbl["price"]) / max(last_lbl["price"], 1e-6)) * 100.0
    wave_rows.append({
        "파동 구간": f"{last_lbl['label']} → 📍현재(진행중)",
        "시작일": last_lbl["date"].strftime("%Y-%m-%d"),
        "종료일": cur_date.strftime("%Y-%m-%d"),
        "시작가": fp(last_lbl["price"]),
        "종료가": fp(cur_price),
        "등락률(%)": round(cur_chg, 2),
        "소요 거래일": f"{int((len(df) - 1) - last_lbl['idx'])}일"
    })

    return {
        "ok": True,
        "currency_mode": currency_mode,
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


def render_elliott_wave_view(df_window: pd.DataFrame, title_name: str, pct: float, currency_mode: str, show_ma: bool = True):
    """엘리엇 파동 카드 + 오버레이 차트 + 구간 표 공통 렌더러"""
    ew_res = analyze_elliott_wave(df_window, pct_threshold=pct, currency_mode=currency_mode)
    if not ew_res["ok"]:
        st.warning(ew_res["message"])
        return

    fp = lambda v: fmt_price(v, currency_mode)
    inv_txt = f"{fp(ew_res['invalidation_price'])} ({ew_res['invalidation_label']})" if ew_res["invalidation_price"] else "-"
    rules_html = " &nbsp;|&nbsp; ".join([f"✔ {note}" for note in ew_res["rule_notes"]])

    st.markdown(f"""
    <div class="ew-box" style="border-color:{ew_res['stage_badge_color']};">
        <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:8px;">
            <div>
                <span style="background:{ew_res['stage_badge_color']}; color:#fff; padding:6px 14px; border-radius:20px; font-weight:800; font-size:0.95rem;">
                    현재 위치: {ew_res['stage_title']}
                </span>
                <b style="font-size:1.15rem; margin-left:10px;">{title_name}</b>
            </div>
            <div style="font-size:1.15rem; font-weight:800;">
                현재 값: {fp(ew_res['cur_price'])}
                <span style="font-size:0.85rem; color:#6b7684; font-weight:500;">({ew_res['cur_date'].strftime('%Y-%m-%d')})</span>
            </div>
        </div>
        <div style="margin-top:12px; font-size:0.92rem; color:#212529; line-height:1.6;">
            • <b>파동 진단:</b> {ew_res['stage_desc']}<br>
            • <b>전략 포인트:</b> <span style="color:#1964d8; font-weight:700;">{ew_res['action_tip']}</span><br>
            • <b>무효화(손절) 기준:</b> <span style="color:#c92a2a; font-weight:700;">{inv_txt}</span>
        </div>
        <div style="background:#f8f9fa; padding:8px 12px; border-radius:8px; margin-top:10px; font-size:0.82rem; color:#495057;">
            {rules_html}
        </div>
    </div>
    """, unsafe_allow_html=True)

    fig_ew = go.Figure()
    fig_ew.add_trace(go.Candlestick(
        x=df_window.index,
        open=df_window["Open"], high=df_window["High"], low=df_window["Low"], close=df_window["Close"],
        name="일봉",
        increasing_line_color="rgba(225, 35, 67, 0.55)", decreasing_line_color="rgba(25, 100, 216, 0.55)",
        increasing_fillcolor="rgba(225, 35, 67, 0.28)", decreasing_fillcolor="rgba(25, 100, 216, 0.28)"
    ))

    if show_ma and "MA40" in df_window.columns and "MA200" in df_window.columns:
        fig_ew.add_trace(go.Scatter(x=df_window.index, y=df_window["MA40"], name="40일선", line=dict(color="rgba(245,159,0,0.7)", width=1.6)))
        fig_ew.add_trace(go.Scatter(x=df_window.index, y=df_window["MA200"], name="200일선", line=dict(color="rgba(71,85,105,0.75)", width=1.8, dash="dot")))

    all_pivs = ew_res["pivots"]
    fig_ew.add_trace(go.Scatter(
        x=[p["date"] for p in all_pivs], y=[p["price"] for p in all_pivs],
        mode="lines", name="전체 변곡선(ZigZag)", line=dict(color="rgba(148,163,184,0.45)", width=1.5, dash="dot")
    ))

    labeled = ew_res["labeled"]
    imp_pts = [p for p in labeled if p["group"] == "impulse"]
    corr_pts = [p for p in labeled if p["group"] == "correction"]
    comp_pts = [p for p in labeled if p["group"] == "complex"]

    if len(imp_pts) >= 2:
        fig_ew.add_trace(go.Scatter(
            x=[p["date"] for p in imp_pts], y=[p["price"] for p in imp_pts],
            mode="lines+markers", name="충격파 (1~5파)",
            line=dict(color="#e11d48", width=3.5), marker=dict(size=9, color="#e11d48", line=dict(width=1.5, color="#fff"))
        ))
    if corr_pts:
        corr_line = ([imp_pts[-1]] + corr_pts) if imp_pts else corr_pts
        if len(corr_line) >= 2:
            fig_ew.add_trace(go.Scatter(
                x=[p["date"] for p in corr_line], y=[p["price"] for p in corr_line],
                mode="lines+markers", name="조정파 (A~C파)",
                line=dict(color="#1d4ed8", width=3.5), marker=dict(size=9, color="#1d4ed8", line=dict(width=1.5, color="#fff"))
            ))
    if len(comp_pts) >= 2:
        fig_ew.add_trace(go.Scatter(
            x=[p["date"] for p in comp_pts], y=[p["price"] for p in comp_pts],
            mode="lines+markers", name="복합/횡보 파동",
            line=dict(color="#7c3aed", width=3.0), marker=dict(size=8, color="#7c3aed", line=dict(width=1.5, color="#fff"))
        ))

    last_p = labeled[-1]
    fig_ew.add_trace(go.Scatter(
        x=[last_p["date"], ew_res["cur_date"]], y=[last_p["price"], ew_res["cur_price"]],
        mode="lines+markers", name="📍 현재 진행 파동",
        line=dict(color=ew_res["stage_badge_color"], width=3.2, dash="dash"),
        marker=dict(size=[8, 16], symbol=["circle", "star"], color=[ew_res["stage_badge_color"], "#f59f00"], line=dict(width=1.5, color="#191f28"))
    ))

    for p in labeled:
        y_shift = 20 if p["kind"] == "H" else -22
        lbl_bg = "#e11d48" if p["group"] == "impulse" else "#1d4ed8" if p["group"] == "correction" else "#7c3aed"
        fig_ew.add_annotation(
            x=p["date"], y=p["price"],
            text=f"<b>{p['label']}</b><br><span style='font-size:10px;'>{fp(p['price'])}</span>",
            showarrow=False, yshift=y_shift, bgcolor=lbl_bg, font=dict(color="#fff", size=12), borderpad=4, opacity=0.92
        )

    fig_ew.add_annotation(
        x=ew_res["cur_date"], y=ew_res["cur_price"],
        text=f"<b>📍 현재 지점</b><br>{fp(ew_res['cur_price'])}",
        showarrow=True, arrowhead=2, ax=35, ay=-35, bgcolor="#191f28", font=dict(color="#fff", size=11), borderpad=5
    )

    for fib_lbl, fib_price, fib_col in ew_res["fib_levels"]:
        fig_ew.add_hline(
            y=fib_price, line_dash="dot", line_color=fib_col, line_width=1.3,
            annotation_text=f"{fib_lbl}: {fp(fib_price)}", annotation_position="top right",
            annotation_font=dict(size=11, color=fib_col)
        )

    if ew_res["invalidation_price"]:
        fig_ew.add_hline(
            y=ew_res["invalidation_price"], line_dash="dash", line_color="#dc2626", line_width=1.8,
            annotation_text=f"⚠️ 무효화선: {fp(ew_res['invalidation_price'])}", annotation_position="bottom right",
            annotation_font=dict(size=11, color="#dc2626")
        )

    fig_ew.update_layout(
        title=f"{title_name} 엘리엇 파동 차트 (민감도 {pct}%)",
        height=580, xaxis_rangeslider_visible=False, hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=65, l=20, r=30, b=20)
    )
    st.plotly_chart(fig_ew, use_container_width=True)

    st.markdown("#### 📊 파동 구간별 가격 · 변동률 · 소요 기간")
    st.dataframe(ew_res["wave_table"], use_container_width=True, hide_index=True)


# ---------------------------------------------------------
# 4. 한국 주식 수집 및 마스터 헬퍼
# ---------------------------------------------------------
def fetch_naver_universe_fallback(min_mcap_eok: int = 0, max_per_market: int = 400) -> pd.DataFrame:
    rows, seen = [], set()
    for api_mkt, kor_mkt in [("KOSPI", "코스피"), ("KOSDAQ", "코스닥")]:
        page, cnt = 1, 0
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
                    rows.append({"code": code.zfill(6), "name": name, "market": kor_mkt, "mcap_eok": int(mcap_eok), "delisted": False})
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
                rows.append({"Date": pd.to_datetime(dt, format="%Y%m%d"), "Open": float(o), "High": float(h), "Low": float(l), "Close": float(c), "Volume": float(v)})
        if not rows:
            return None
        return validate_ohlcv(pd.DataFrame(rows).set_index("Date"), min_days)
    except Exception:
        return None


def safe_get_kr_universe(cfg: dict) -> pd.DataFrame:
    df_u = get_filtered_universe(cfg, include_delisted=False)
    if df_u is None or df_u.empty:
        return fetch_naver_universe_fallback(cfg["universe"]["min_mcap_eok"], max_per_market=400)
    return df_u.sort_values("mcap_eok", ascending=False).reset_index(drop=True)


@st.cache_data(ttl=3600)
def get_kr_stock_master() -> pd.DataFrame:
    try:
        if FDR_AVAILABLE:
            listing = fdr.StockListing("KRX")
            if listing is not None and not listing.empty:
                if "Symbol" in listing.columns and "Code" not in listing.columns:
                    listing = listing.rename(columns={"Symbol": "Code"})
                listing["Market_Clean"] = listing["Market"].astype(str).str.upper().replace({"KOSDAQ GLOBAL": "KOSDAQ"})
                df = listing[listing["Market_Clean"].isin(["KOSPI", "KOSDAQ"])].copy()
                dept_s = df["Dept"].fillna("") if "Dept" in df.columns else pd.Series("", index=df.index)
                keep = [not is_excluded_by_meta(str(c).strip(), str(n).strip(), str(d)) for c, n, d in zip(df["Code"], df["Name"], dept_s)]
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


def safe_fetch_kr_ohlcv(code: str, cfg: dict) -> pd.DataFrame | None:
    df = fetch_stock_ohlcv(code, cfg)
    if df is None:
        df = fetch_naver_ohlcv_fallback(code, cfg["universe"]["min_trading_days"], count=750)
    return df


@st.cache_data(ttl=900)
def fetch_single_kr_ohlcv(code: str, min_days: int = 120, count: int = 800) -> pd.DataFrame | None:
    if FDR_AVAILABLE:
        try:
            start_s = (datetime.today() - timedelta(days=1200)).strftime("%Y-%m-%d")
            clean = validate_ohlcv(fdr.DataReader(code, start_s), min_days)
            if clean is not None:
                return clean
        except Exception:
            pass
    return fetch_naver_ohlcv_fallback(code, min_days, count=count)


def fetch_kr_ohlcv_for_bt(code: str, cfg: dict) -> pd.DataFrame | None:
    min_days = cfg["universe"]["min_trading_days"]
    if FDR_AVAILABLE:
        try:
            start_s = (datetime.today() - timedelta(days=1850)).strftime("%Y-%m-%d")
            clean = validate_ohlcv(fdr.DataReader(code, start_s), min_days)
            if clean is not None and len(clean) >= 700:
                return clean
        except Exception:
            pass
    return fetch_naver_ohlcv_fallback(code, min_days, count=1250)


def build_excel_bytes(passed_df: pd.DataFrame, bt_summary: pd.DataFrame, bt_trades: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        passed_df.to_excel(writer, sheet_name="최종_선별종목", index=False)
        for prefix, sname in [("D.", "D_최종상승전환"), ("C.", "C_골든크로스"), ("B.", "B_거래량급증"), ("A.", "A_이평선수렴")]:
            sub = passed_df[passed_df["신호 유형"].str.startswith(prefix)] if not passed_df.empty else pd.DataFrame()
            sub.to_excel(writer, sheet_name=sname, index=False)
        if bt_summary is not None and not bt_summary.empty:
            bt_summary.to_excel(writer, sheet_name="백테스트_요약", index=False)
        if bt_trades is not None and not bt_trades.empty:
            bt_trades.head(5000).to_excel(writer, sheet_name="백테스트_거래내역", index=False)
    return buf.getvalue()


# ---------------------------------------------------------
# 5. 기본 설정 로드 및 세션 상태 초기화
# ---------------------------------------------------------
try:
    BASE_CFG = load_config("config.json")
except Exception:
    st.error("config.json 파일을 찾을 수 없습니다. 깃허브에 config.json을 함께 올려주세요.")
    st.stop()

defaults = {
    "scan_done": False,
    "scan_market": "KR",
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
    "search_bt_market": "KR",
    "search_bt_done": False,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ---------------------------------------------------------
# 6. 최상단: 글로벌 지수(S&P500 · 나스닥100 · 코스피 · VIX) & 원/달러 환율 현황판
# ---------------------------------------------------------
hdr1, hdr2 = st.columns([4.2, 1.0])
with hdr1:
    st.title("📈 한·미 이평선 수렴 · 거래량 돌파 스윙 시스템")
    st.caption("S&P 500 · 나스닥 100 · 원/달러 환율 모니터링 + 한국·미국 주식 스캔 · 엘리엇 파동 · 전략 A/B/C 백테스트")
with hdr2:
    if st.button("🔄 지수·환율 갱신", use_container_width=True):
        load_macro_dashboard_data.clear()
        st.rerun()

macro_map = load_macro_dashboard_data()
# 실시간 원/달러 환율 추출 (미국 주식 원화 환산 및 거래대금 계산용)
if "KRW=X" in macro_map and not macro_map["KRW=X"].empty:
    LIVE_USD_KRW = float(macro_map["KRW=X"]["Close"].iloc[-1])
else:
    LIVE_USD_KRW = 1380.0

macro_cols = st.columns(5)
for idx, info in enumerate(MACRO_SYMBOLS):
    sym = info["symbol"]
    with macro_cols[idx]:
        if sym in macro_map and len(macro_map[sym]) >= 2:
            mdf = macro_map[sym]
            cur_v = float(mdf["Close"].iloc[-1])
            prev_v = float(mdf["Close"].iloc[-2])
            chg_v = ((cur_v - prev_v) / prev_v) * 100.0 if prev_v > 0 else 0.0
            ma200_v = float(mdf["Close"].rolling(200).mean().iloc[-1]) if len(mdf) >= 200 else cur_v
            trend_note = "200일선 위 ▲" if cur_v >= ma200_v else "200일선 아래 ▼"
            val_str = f"{cur_v:,.1f}원" if info["unit"] == "FX" else f"{cur_v:,.2f}"
            st.metric(
                label=f"{info['name']}",
                value=val_str,
                delta=f"{chg_v:+.2f}% ({trend_note})",
                delta_color="inverse" if info["unit"] in ("FX", "VIX") else "normal"
            )
        else:
            st.metric(label=info["name"], value="조회 대기", delta="-")

st.divider()

# ---------------------------------------------------------
# 7. 스캔 시장 선택 (한국 vs 미국) & 실행 컨트롤러
# ---------------------------------------------------------
m_sel_c1, m_sel_c2, m_sel_c3, m_sel_c4 = st.columns([1.2, 1.4, 1.3, 1.1])
with m_sel_c1:
    market_choice = st.radio(
        "🌐 스캔할 주식 시장",
        ["🇰🇷 한국 주식 (KRX)", "🇺🇸 미국 주식 (US)"],
        horizontal=True
    )
is_us_mode = "미국" in market_choice

with m_sel_c2:
    if is_us_mode:
        scan_preset = st.selectbox(
            "🔍 미국 스캔 범위",
            [
                "미국 핵심 120종목 (나스닥100·주요주, 약 15~25초)",
                "미국 상위 250종목 (S&P500 포함, 약 35~50초)",
                "S&P 500 전체 (약 500종목, 약 1분 30초)"
            ],
            index=0
        )
    else:
        scan_preset = st.selectbox(
            "🔍 한국 스캔 범위 (시총 상위 순)",
            [
                "빠른 스캔: 상위 150종목 (약 15~25초)",
                "표준 스캔: 상위 300종목 (약 35~50초)",
                "정밀 스캔: 상위 600종목 (약 1분 30초)",
                "전체 스캔: 조건 통과 전 종목 (약 3~5분)"
            ],
            index=0
        )

with m_sel_c3:
    placeholder_txt = "예: NVDA, TSLA, IONQ" if is_us_mode else "예: 005930, 035720"
    custom_codes = st.text_input("➕ 추가할 종목코드/티커 (쉼표 구분)", placeholder=placeholder_txt)

with m_sel_c4:
    st.write("")
    run_btn = st.button("🚀 시장 스캔 시작", type="primary", use_container_width=True)

with st.expander("⚙️ 전략·임계값 세부 설정 (한국·미국 스캔 및 백테스트 공통 적용)", expanded=False):
    cfg = copy.deepcopy(BASE_CFG)
    p1, p2, p3, p4 = st.columns(4)
    with p1:
        cfg["universe"]["min_mcap_eok"] = st.number_input("최소 시가총액(억원)", 0, 50000, int(BASE_CFG["universe"]["min_mcap_eok"]), step=100)
        cfg["universe"]["min_trading_val_eok"] = st.number_input("최소 거래대금(억원 환산)", 1, 1000, int(BASE_CFG["universe"]["min_trading_val_eok"]), step=5)
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

# 스캔 수량 파싱
if is_us_mode:
    scan_limit = 120 if "120" in scan_preset else (250 if "250" in scan_preset else None)
else:
    scan_limit = 150 if "150" in scan_preset else (300 if "300" in scan_preset else (600 if "600" in scan_preset else None))


# ---------------------------------------------------------
# 8. 시장 스캔 실행 로직 (한국 & 미국 통합)
# ---------------------------------------------------------
if run_btn:
    mkt_code = "US" if is_us_mode else "KR"
    mkt_label = "미국(S&P500·나스닥100)" if is_us_mode else "한국(코스피·코스닥)"
    progress_bar = st.progress(0, text=f"1/3) {mkt_label} 종목 목록 조회 중...")

    uni_df = get_us_stock_master() if is_us_mode else safe_get_kr_universe(cfg)
    if uni_df.empty:
        st.error(f"{mkt_label} 종목 목록을 불러오지 못했습니다. 잠시 후 다시 시도해 주세요.")
        st.stop()

    target_df = uni_df.head(scan_limit).copy() if scan_limit else uni_df.copy()

    # 사용자 추가 종목 병합
    if custom_codes.strip():
        existing_set = set(target_df["code"].astype(str))
        extra_rows = []
        for raw_c in custom_codes.split(","):
            c = raw_c.strip().upper() if is_us_mode else raw_c.strip().zfill(6)
            valid_c = (len(c) >= 1) if is_us_mode else (len(c) == 6 and c.isdigit())
            if valid_c and c not in existing_set:
                match_u = uni_df[uni_df["code"] == c]
                if not match_u.empty:
                    extra_rows.append(match_u.iloc[0].to_dict())
                else:
                    extra_rows.append({
                        "code": c, "name": c if is_us_mode else f"추가종목({c})",
                        "market": "미국(US)" if is_us_mode else "직접추가",
                        "mcap_eok": cfg["universe"]["min_mcap_eok"], "delisted": False
                    })
                existing_set.add(c)
        if extra_rows:
            target_df = pd.concat([target_df, pd.DataFrame(extra_rows)], ignore_index=True)

    total_n = len(target_df)
    meta_records = target_df.to_dict("records")
    indicator_map = {}
    eval_rows = []

    def analyze_one(meta):
        if is_us_mode:
            df_raw = fetch_us_ohlcv(meta["code"], min_days=cfg["universe"]["min_trading_days"], range_str="3y")
        else:
            df_raw = safe_fetch_kr_ohlcv(meta["code"], cfg)
        if df_raw is None:
            return meta["code"], None, None

        ind_df = compute_all_indicators(df_raw, cfg)
        if is_us_mode:
            # 미국 달러 거래대금을 실시간 환율로 원화 억원 단위로 정확히 환산
            ind_df["TradingVal_Eok"] = ind_df["TradingVal_Eok"] * LIVE_USD_KRW

        res = evaluate_signal_at_index(ind_df, len(ind_df) - 1, meta, cfg)
        if res is not None:
            if is_us_mode:
                res = format_us_eval_result(res, ind_df, LIVE_USD_KRW)
            else:
                res["현재가 표시"] = f"{int(res['현재가']):,}원"
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
                progress_bar.progress(pct, text=f"2/3) {mkt_label} 3년 일봉 수집 및 9단계 분석 중... ({done_cnt} / {total_n}종목)")

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
    st.session_state.scan_market = mkt_code
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
scanned_is_us = st.session_state.scan_market == "US"

if st.session_state.scan_done and not all_df.empty:
    mkt_badge = "🇺🇸 미국 증시" if scanned_is_us else "🇰🇷 한국 증시"
    cnt_d = int(passed_df["신호 유형"].str.startswith("D.").sum()) if not passed_df.empty else 0
    cnt_c = int(passed_df["신호 유형"].str.startswith("C.").sum()) if not passed_df.empty else 0
    cnt_b = int(passed_df["신호 유형"].str.startswith("B.").sum()) if not passed_df.empty else 0
    cnt_a = int(passed_df["신호 유형"].str.startswith("A.").sum()) if not passed_df.empty else 0

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("🔥 D. 최종 상승전환", f"{cnt_d}종목", mkt_badge)
    m2.metric("✨ C. 골든크로스", f"{cnt_c}종목", "수렴 후 돌파")
    m3.metric("⚡ B. 거래량 급증", f"{cnt_b}종목", "수렴 + 1.5배+")
    m4.metric("🧲 A. 이평선 수렴", f"{cnt_a}종목", "바닥 응축")
    m5.metric("📊 분석 완료", f"{len(all_df)} / {st.session_state.total_target}", st.session_state.updated_at)

# ---------------------------------------------------------
# 9. 7대 대시보드 탭
# ---------------------------------------------------------
tab0, tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🌐 지수·환율 차트",
    "① 시그널 지도",
    "② 결과표 & 엑셀",
    "③ 근거 & 매매전략",
    "④ 20·40·50·200일 차트",
    "⑤ 엘리엇 파동 분석",
    "⑥ 전략 A/B/C 백테스트"
])

SCAN_NOTICE = "👆 상단에서 시장(한국/미국)을 고르고 **[🚀 시장 스캔 시작]** 버튼을 누르시면 결과가 표시됩니다. (**🌐 지수·환율 차트**, **⑤ 엘리엇 파동**, **⑥ 백테스트** 탭은 스캔 없이 바로 사용 가능합니다!)"

# =========================================================
# [탭 0] S&P 500 · 나스닥 100 · 원/달러 환율 · 코스피 상세 차트
# =========================================================
with tab0:
    st.subheader("🌐 미국 주요 지수(S&P 500 · 나스닥 100) · 코스피 · 원/달러 환율 상세 분석")
    st.caption(f"현재 적용 기준 환율: 1달러 = {LIVE_USD_KRW:,.2f}원 (실시간 야후 파이낸스 연동)")

    mc1, mc2 = st.columns([2, 1])
    with mc1:
        macro_opt_labels = [f"{m['name']} ({m['symbol']}) - {m['sub']}" for m in MACRO_SYMBOLS]
        sel_macro_lbl = st.selectbox("🔎 상세 차트를 확인할 지수 / 환율 선택", macro_opt_labels, index=0)
        sel_macro = MACRO_SYMBOLS[macro_opt_labels.index(sel_macro_lbl)]
    with mc2:
        macro_period = st.radio("조회 기간", ["6개월", "1년", "3년"], index=1, horizontal=True)
        macro_bars = 125 if macro_period == "6개월" else (250 if macro_period == "1년" else 750)

    sym = sel_macro["symbol"]
    if sym not in macro_map or macro_map[sym].empty:
        st.warning("해당 지수/환율 데이터를 불러오지 못했습니다. 상단 [🔄 지수·환율 갱신] 버튼을 눌러주세요.")
    else:
        full_m = macro_map[sym].copy()
        full_m["MA20"] = full_m["Close"].rolling(20).mean()
        full_m["MA40"] = full_m["Close"].rolling(40).mean()
        full_m["MA50"] = full_m["Close"].rolling(50).mean()
        full_m["MA200"] = full_m["Close"].rolling(200).mean()

        delta = full_m["Close"].diff()
        ag = delta.clip(lower=0.0).ewm(alpha=1/14, min_periods=14, adjust=False).mean()
        al = (-delta.clip(upper=0.0)).ewm(alpha=1/14, min_periods=14, adjust=False).mean()
        full_m["RSI14"] = (100.0 - (100.0 / (1.0 + ag / al.replace(0, np.nan)))).fillna(50.0)

        view_m = full_m.tail(macro_bars)
        latest_m = full_m.iloc[-1]
        y1_slice = full_m.tail(250)
        hi_1y, lo_1y = float(y1_slice["High"].max()), float(y1_slice["Low"].min())
        cur_val = float(latest_m["Close"])
        dd_hi = ((cur_val - hi_1y) / hi_1y) * 100.0 if hi_1y > 0 else 0.0
        ma200_gap = ((cur_val - float(latest_m["MA200"])) / float(latest_m["MA200"])) * 100.0 if pd.notna(latest_m["MA200"]) else 0.0

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("현재 값", fmt_price(cur_val, sel_macro["unit"]))
        k2.metric("52주 최고 대비", f"{dd_hi:+.2f}%", f"최고 {fmt_price(hi_1y, sel_macro['unit'])}")
        k3.metric("200일선 이격도", f"{ma200_gap:+.2f}%", f"200일선 {fmt_price(float(latest_m['MA200']), sel_macro['unit'])}")
        k4.metric("현재 RSI(14)", f"{float(latest_m['RSI14']):.1f}", "과열(70+) / 침체(30-)")

        fig_m = make_subplots(
            rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05,
            row_heights=[0.74, 0.26],
            subplot_titles=(f"{sel_macro['name']} ({sym}) 일봉 · 20/40/50/200일선", "RSI(14)")
        )
        fig_m.add_trace(go.Candlestick(
            x=view_m.index, open=view_m["Open"], high=view_m["High"], low=view_m["Low"], close=view_m["Close"],
            name=sel_macro["name"], increasing_line_color="#e12343", decreasing_line_color="#1964d8"
        ), row=1, col=1)
        fig_m.add_trace(go.Scatter(x=view_m.index, y=view_m["MA20"], name="20일선", line=dict(color="#f59f00", width=1.6)), row=1, col=1)
        fig_m.add_trace(go.Scatter(x=view_m.index, y=view_m["MA40"], name="40일선", line=dict(color="#e11d48", width=2.0)), row=1, col=1)
        fig_m.add_trace(go.Scatter(x=view_m.index, y=view_m["MA50"], name="50일선", line=dict(color="#16a34a", width=1.8)), row=1, col=1)
        fig_m.add_trace(go.Scatter(x=view_m.index, y=view_m["MA200"], name="200일선", line=dict(color="#1d4ed8", width=2.4, dash="dot")), row=1, col=1)

        fig_m.add_trace(go.Scatter(x=view_m.index, y=view_m["RSI14"], name="RSI(14)", line=dict(color="#7c3aed", width=2)), row=2, col=1)
        fig_m.add_hline(y=50, line_dash="dash", line_color="#868e96", row=2, col=1)
        fig_m.add_hline(y=70, line_dash="dot", line_color="#e12343", row=2, col=1)
        fig_m.add_hline(y=30, line_dash="dot", line_color="#1964d8", row=2, col=1)

        fig_m.update_layout(height=580, xaxis_rangeslider_visible=False, hovermode="x unified", margin=dict(t=40, l=20, r=20, b=20))
        st.plotly_chart(fig_m, use_container_width=True)

        # S&P500 vs 나스닥100 vs 코스피 최근 1년 상대 수익률 비교 차트
        st.markdown("#### 📊 S&P 500 · 나스닥 100 · 코스피 · 원/달러 환율 최근 1년 추이 비교 (1년 전 = 100 기준)")
        fig_cmp = go.Figure()
        cmp_colors = {"^GSPC": "#e11d48", "^NDX": "#7c3aed", "^KS11": "#1964d8", "KRW=X": "#16a34a"}
        for item in MACRO_SYMBOLS[:4]:
            s_code = item["symbol"]
            if s_code in macro_map and len(macro_map[s_code]) >= 30:
                sub_s = macro_map[s_code]["Close"].tail(250)
                norm_s = (sub_s / float(sub_s.iloc[0])) * 100.0
                fig_cmp.add_trace(go.Scatter(
                    x=norm_s.index, y=norm_s.values, mode="lines",
                    name=f"{item['name']} ({norm_s.iloc[-1] - 100:+.1f}%)",
                    line=dict(color=cmp_colors.get(s_code, "#333"), width=2.4)
                ))
        fig_cmp.add_hline(y=100, line_dash="dash", line_color="#94a3b8")
        fig_cmp.update_layout(height=340, hovermode="x unified", yaxis_title="상대 지수 (1년 전=100)", margin=dict(t=20, l=20, r=20, b=20))
        st.plotly_chart(fig_cmp, use_container_width=True)

# =========================================================
# [탭 1] 앤트위키 시그널 지도
# =========================================================
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
            st.info("현재 엄격 조건을 모두 통과한 종목이 0개입니다. '전체 분석 종목 중 점수 상위 40개'를 선택하거나 상단 설정을 조금 완화해 보세요.")
        else:
            map_df["tile_size"] = map_df["거래대금(억원)"].clip(lower=10)
            fig_tree = px.treemap(
                map_df,
                path=[px.Constant("이평선 수렴 포착 지도"), "신호 유형", "최종 등급", "종목명 및 종목코드"],
                values="tile_size",
                color="최종 점수",
                color_continuous_scale=[[0.0, "#bfdbfe"], [0.5, "#fef08a"], [0.8, "#fca5a5"], [1.0, "#dc2626"]],
                range_color=[45, 100],
                custom_data=["최종 점수", "현재가 표시", "20-200 이격도(%)", "거래량 증가율(배)", "최종 등급", "수렴 상태"]
            )
            fig_tree.update_traces(
                texttemplate="<b>%{label}</b><br>%{customdata[0]}점 · %{customdata[5]}<br>%{customdata[1]} (거래 %{customdata[3]}배)",
                textposition="middle center",
                hovertemplate=(
                    "<b>%{label}</b><br>등급: %{customdata[4]} (%{customdata[0]}점)<br>"
                    "수렴 상태: %{customdata[5]} (20-200 이격 %{customdata[2]}%)<br>"
                    "현재가: %{customdata[1]} · 거래량: %{customdata[3]}배<extra></extra>"
                )
            )
            fig_tree.update_layout(margin=dict(t=20, l=10, r=10, b=10), height=560)
            st.plotly_chart(fig_tree, use_container_width=True)

# =========================================================
# [탭 2] 전체 결과표 & 엑셀 저장
# =========================================================
with tab2:
    if not st.session_state.scan_done or all_df.empty:
        st.info(SCAN_NOTICE)
    else:
        f_c1, f_c2, f_c3 = st.columns([2, 1, 1])
        with f_c1:
            include_excl = st.checkbox("65점 미만 / 필수조건 미달 종목도 함께 보기 (근접 후보 점검용)", value=False)
        view_df = all_df if include_excl else passed_df

        cols_order = [
            "종목명 및 종목코드", "시장", "현재가 표시", "최근 1년 고점 대비 하락률(%)", "역배열 유지 기간(일)",
            "20일 이동평균선", "40일 이동평균선", "50일 이동평균선", "200일 이동평균선",
            "20-200 이격도(%)", "50-200 이격도(%)", "20-50 이격도(%)", "20일간 이격도 축소율(%)",
            "수렴 상태", "이동평균선별 기울기", "골든크로스 발생 여부 및 종류",
            "거래량 증가율(배)", "거래대금(억원)", "박스권 상단 및 하단", "RSI 및 MACD",
            "예상 손익비", "최종 점수", "최종 등급", "신호 유형",
            "신호 포착 근거", "매매 진입 및 손절 기준", "상승 시나리오 무효화 조건"
        ]
        show_cols = [c for c in cols_order if c in view_df.columns]

        today_tag = datetime.now().strftime("%Y%m%d")
        prefix_file = "us" if scanned_is_us else "kr"
        with f_c2:
            if not view_df.empty:
                st.download_button(
                    "📥 CSV 파일 저장",
                    data=view_df[show_cols].to_csv(index=False).encode("utf-8-sig"),
                    file_name=f"{prefix_file}_ma_signals_{today_tag}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
        with f_c3:
            if not view_df.empty:
                xlsx_bytes = build_excel_bytes(view_df[show_cols], st.session_state.bt_summary, st.session_state.bt_trades)
                st.download_button(
                    "📊 엑셀(.xlsx) 저장",
                    data=xlsx_bytes,
                    file_name=f"{prefix_file}_ma_report_{today_tag}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )

        if view_df.empty:
            st.info("통과한 종목이 없습니다. 위 체크박스를 켜면 점수 상위 근접 종목들을 확인할 수 있습니다.")
        else:
            st.dataframe(view_df[show_cols], use_container_width=True, height=540, hide_index=True)

# =========================================================
# [탭 3] 종목별 포착 근거 & 매매 전략 카드
# =========================================================
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
                            <b style="font-size:1.1rem;">{r['현재가 표시']}</b>
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

# =========================================================
# [탭 4] 20·40·50·200일 이동평균선 & 박스권 정밀 차트
# =========================================================
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
        c_mode = "USD" if scanned_is_us else "KRW"

        if sel_code in indicator_map:
            cdf = indicator_map[sel_code].tail(260).copy().reset_index()
            cdf = cdf.rename(columns={cdf.columns[0]: "Date"})

            fig = make_subplots(
                rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.04,
                row_heights=[0.60, 0.20, 0.20],
                subplot_titles=(f"{sel_row['종목명 및 종목코드']} | 20·40·50·200일선 수렴 및 박스권", "거래량 & 20일 평균 거래량", "RSI(14)")
            )
            fig.add_trace(go.Candlestick(
                x=cdf["Date"], open=cdf["Open"], high=cdf["High"], low=cdf["Low"], close=cdf["Close"],
                name="일봉", increasing_line_color="#e12343", decreasing_line_color="#1964d8"
            ), row=1, col=1)
            fig.add_trace(go.Scatter(x=cdf["Date"], y=cdf["MA20"], name="20일선", line=dict(color="#f59f00", width=1.8)), row=1, col=1)
            fig.add_trace(go.Scatter(x=cdf["Date"], y=cdf["MA40"], name="40일선", line=dict(color="#e11d48", width=2.2)), row=1, col=1)
            fig.add_trace(go.Scatter(x=cdf["Date"], y=cdf["MA50"], name="50일선", line=dict(color="#16a34a", width=2.0)), row=1, col=1)
            fig.add_trace(go.Scatter(x=cdf["Date"], y=cdf["MA200"], name="200일선", line=dict(color="#1d4ed8", width=2.6, dash="dot")), row=1, col=1)

            bh, bl = float(cdf["Box_High"].iloc[-1]), float(cdf["Box_Low"].iloc[-1])
            fig.add_hline(y=bh, line_dash="dash", line_color="#d97706", annotation_text=f"박스상단 {fmt_price(bh, c_mode)}", row=1, col=1)
            fig.add_hline(y=bl, line_dash="dash", line_color="#0284c7", annotation_text=f"박스하단 {fmt_price(bl, c_mode)}", row=1, col=1)

            gc_pts = cdf[cdf["GC_Any"] == True]
            if not gc_pts.empty:
                fig.add_trace(go.Scatter(
                    x=gc_pts["Date"], y=gc_pts["Close"], mode="markers", name="골든크로스 발생일",
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

# =========================================================
# [탭 5] 엘리엇 파동 분석 (한국 주식 · 미국 주식 · 지수/환율 통합)
# =========================================================
with tab5:
    st.subheader("🌊 종목 · 지수별 엘리엇 파동 자동 대입 & 현재 지점 분석")
    st.caption("한국 주식, 미국 주식, 그리고 S&P 500 · 나스닥 100 · 원/달러 환율 차트에도 엘리엇 파동을 바로 대입해 현재 위치를 확인할 수 있습니다.")

    ew_src_opts = ["🇰🇷 한국 주식 검색", "🇺🇸 미국 주식 검색", "🌐 지수·환율 (S&P500·나스닥100·환율)"]
    if st.session_state.scan_done and not all_df.empty:
        ew_src_opts.insert(0, "📋 방금 스캔한 종목에서 선택")

    ew_source = st.radio("분석 대상 분류", ew_src_opts, horizontal=True, key="ew_src_radio")

    ew_c1, ew_c2, ew_c3 = st.columns([2.2, 1.1, 1.3])
    ew_code, ew_name, ew_curr_mode = None, "", "KRW"
    ew_custom_ticker = ""

    with ew_c1:
        if "스캔한 종목" in ew_source:
            ew_scan_df = all_df.sort_values(["신호 우선순위", "최종 점수"], ascending=[True, False]).reset_index(drop=True)
            ew_scan_opts = [f"[{r['신호 유형']} · {r['최종 점수']}점] {r['종목명 및 종목코드']}" for _, r in ew_scan_df.iterrows()]
            picked = st.selectbox("스캔 종목 선택", ew_scan_opts, key="ew_scan_pick")
            picked_row = ew_scan_df.iloc[ew_scan_opts.index(picked)]
            ew_code = str(picked_row["종목코드"])
            ew_name = str(picked_row["종목명 및 종목코드"])
            ew_curr_mode = "USD" if scanned_is_us else "KRW"

        elif "한국 주식" in ew_source:
            kr_master = get_kr_stock_master()
            if not kr_master.empty:
                kr_lbls = [f"{r['name']} ({r['code']}) [{r['market']}]" for _, r in kr_master.iterrows()]
                ew_pick = st.selectbox("한국 종목 검색 (종목명 또는 코드)", kr_lbls, index=0, key="ew_kr_pick")
                p_row = kr_master.iloc[kr_lbls.index(ew_pick)]
                ew_code, ew_name, ew_curr_mode = str(p_row["code"]), f"{p_row['name']} ({p_row['code']})", "KRW"

        elif "미국 주식" in ew_source:
            us_master = get_us_stock_master()
            us_lbls = [f"{r['name']} ({r['code']})" for _, r in us_master.iterrows()]
            ew_pick = st.selectbox("미국 종목 검색 (종목명 또는 티커)", us_lbls, index=0, key="ew_us_pick")
            p_row = us_master.iloc[us_lbls.index(ew_pick)]
            ew_code, ew_name, ew_curr_mode = str(p_row["code"]), f"{p_row['name']} ({p_row['code']})", "USD"
            ew_custom_ticker = st.text_input("목록에 없는 미국 티커 직접 입력 (예: OKLO, SOUN)", key="ew_us_manual").strip().upper()
            if ew_custom_ticker:
                ew_code, ew_name = ew_custom_ticker, f"{ew_custom_ticker} (미국)"

        else:
            m_lbls = [f"{m['name']} ({m['symbol']})" for m in MACRO_SYMBOLS]
            m_pick = st.selectbox("지수 / 환율 선택", m_lbls, index=0, key="ew_macro_pick")
            m_info = MACRO_SYMBOLS[m_lbls.index(m_pick)]
            ew_code, ew_name, ew_curr_mode = m_info["symbol"], f"{m_info['name']} ({m_info['symbol']})", m_info["unit"]

    with ew_c2:
        ew_period_label = st.selectbox("분석 기간", ["최근 1년 (소파동)", "최근 2년 (중기 스윙)", "최근 3년 (대파동)"], index=1, key="ew_period")
        ew_bars = 250 if "1년" in ew_period_label else (500 if "2년" in ew_period_label else 750)

    with ew_c3:
        default_pct = 4.0 if ew_curr_mode in ("INDEX", "FX") else 7.0
        ew_pct = st.slider(
            "파동 민감도 (변동폭 %)",
            min_value=2.0, max_value=15.0, value=default_pct, step=0.5,
            help="지수·환율은 3~5%, 개별주는 6~10%가 파동이 깔끔하게 잡힙니다.",
            key="ew_pct_slider"
        )

    show_ma_overlay = st.checkbox("차트에 40일선·200일선 함께 겹쳐 보기", value=True, key="ew_ma_toggle")

    if ew_code:
        ew_df_full = None
        if ew_curr_mode in ("INDEX", "FX", "VIX"):
            raw_m = macro_map.get(ew_code)
            if raw_m is not None and not raw_m.empty:
                ew_df_full = raw_m.copy()
                ew_df_full["MA40"] = ew_df_full["Close"].rolling(40).mean()
                ew_df_full["MA200"] = ew_df_full["Close"].rolling(200).mean()
        elif ew_code in indicator_map and len(indicator_map[ew_code]) >= ew_bars:
            ew_df_full = indicator_map[ew_code]
        else:
            with st.spinner(f"{ew_name} 일봉 데이터를 불러와 엘리엇 파동을 계산 중입니다..."):
                if ew_curr_mode == "USD":
                    raw_s = fetch_us_ohlcv(ew_code, min_days=120, range_str="3y")
                else:
                    raw_s = fetch_single_kr_ohlcv(ew_code, min_days=120, count=800)
                if raw_s is not None:
                    ew_df_full = compute_all_indicators(raw_s, cfg)

        if ew_df_full is None or ew_df_full.empty:
            st.error(f"{ew_name}의 데이터를 불러오지 못했습니다. 티커/코드를 확인해 주세요.")
        else:
            render_elliott_wave_view(ew_df_full.tail(ew_bars).copy(), ew_name, ew_pct, ew_curr_mode, show_ma_overlay)
            st.caption("※ 엘리엇 파동 카운팅은 ZigZag 변곡점과 핵심 규칙 기반 자동 해석입니다. 민감도(%)를 조절해 대파동과 소파동을 함께 비교해 보세요.")

# =========================================================
# [탭 6] 전략 A · B · C 백테스트 (한국 주식 · 미국 주식 검색 통합)
# =========================================================
with tab6:
    st.subheader("전략 A · B · C 과거 3년 백테스트 성과 비교 (10 · 20 · 40 · 60일 보유)")
    st.caption("체결 기준: t일 장 마감 신호 확정 ➔ t+1일 시가(Open) 진입 ➔ 수수료·거래세·슬리피지 차감 (룩어헤드 바이어스 차단)")

    bt_mode = st.radio(
        "🧪 백테스트 대상 선택",
        ["🇰🇷 한국 종목 직접 검색", "🇺🇸 미국 종목 직접 검색", "🌐 방금 스캔한 전체 종목"],
        horizontal=True
    )

    if "직접 검색" in bt_mode:
        bt_is_us = "미국" in bt_mode
        master_bt = get_us_stock_master() if bt_is_us else get_kr_stock_master()
        lbl_to_meta = {}
        opt_list = []

        if not master_bt.empty:
            for _, row in master_bt.iterrows():
                lbl = f"{row['name']} ({row['code']})" if bt_is_us else f"{row['name']} ({row['code']}) [{row['market']}]"
                opt_list.append(lbl)
                lbl_to_meta[lbl] = {"code": str(row["code"]), "name": str(row["name"])}

        b_s1, b_s2 = st.columns([2.2, 1.3])
        with b_s1:
            ph = "예: 엔비디아, NVDA, TSLA 등을 검색해 선택하세요" if bt_is_us else "예: 삼성전자, 카카오, 005930 등을 검색해 선택하세요"
            selected_bt_lbls = st.multiselect("종목 검색 (여러 종목 동시 선택 가능)", options=opt_list, placeholder=ph)
        with b_s2:
            ph_m = "예: NVDA, IONQ" if bt_is_us else "예: 005930, 035720"
            manual_bt_codes = st.text_input("직접 입력 (쉼표 구분)", placeholder=ph_m, key="bt_manual_inp")

        if st.button("🧪 선택한 종목으로 전략 A/B/C 백테스트 실행", type="primary"):
            target_codes = {}
            for lbl in selected_bt_lbls:
                m = lbl_to_meta[lbl]
                target_codes[m["code"]] = m["name"]

            if manual_bt_codes.strip():
                c_lookup = master_bt.set_index("code")["name"].to_dict() if not master_bt.empty else {}
                for raw_c in manual_bt_codes.split(","):
                    c = raw_c.strip().upper() if bt_is_us else raw_c.strip().zfill(6)
                    if (bt_is_us and len(c) >= 1) or ((not bt_is_us) and len(c) == 6 and c.isdigit()):
                        target_codes[c] = c_lookup.get(c, c)

            if not target_codes:
                st.warning("백테스트할 종목을 1개 이상 선택하거나 입력해 주세요.")
            else:
                bt_cfg = copy.deepcopy(cfg)
                if bt_is_us:
                    # 미국 주식 거래 비용 반영 (증권거래세 면제, 해외주식 수수료 0.1% 가정)
                    bt_cfg["backtest"]["commission_rate"] = 0.001
                    bt_cfg["backtest"]["sell_tax_rate"] = 0.00003

                search_ind_map = {}
                failed_names = []
                with st.spinner(f"선택한 {len(target_codes)}개 종목의 5년 일봉(순수 3년 백테스트 확보)을 불러와 계산 중입니다..."):
                    for c, nm in target_codes.items():
                        if bt_is_us:
                            df_raw = fetch_us_ohlcv(c, min_days=bt_cfg["universe"]["min_trading_days"], range_str="5y")
                        else:
                            df_raw = fetch_kr_ohlcv_for_bt(c, bt_cfg)

                        if df_raw is not None:
                            ind_df = compute_all_indicators(df_raw, bt_cfg)
                            if bt_is_us:
                                ind_df["TradingVal_Eok"] = ind_df["TradingVal_Eok"] * LIVE_USD_KRW
                            search_ind_map[c] = ind_df
                        else:
                            failed_names.append(f"{nm}({c})")

                    if search_ind_map:
                        bt_sum, bt_tr = run_backtest_suite(search_ind_map, bt_cfg)
                        if not bt_tr.empty:
                            bt_tr.insert(3, "종목명", bt_tr["종목코드"].map(lambda x: target_codes.get(str(x), str(x))))
                        st.session_state.search_bt_summary = bt_sum
                        st.session_state.search_bt_trades = bt_tr
                        st.session_state.search_bt_ind_map = search_ind_map
                        st.session_state.search_bt_names = target_codes
                        st.session_state.search_bt_cfg = bt_cfg
                        st.session_state.search_bt_market = "US" if bt_is_us else "KR"
                        st.session_state.search_bt_done = True
                    else:
                        st.session_state.search_bt_done = False

                if failed_names:
                    st.warning(f"일봉 데이터가 부족(상장 2년 미만 등)해 제외된 종목: {', '.join(failed_names)}")

        if st.session_state.search_bt_done:
            s_sum = st.session_state.search_bt_summary
            s_tr = st.session_state.search_bt_trades
            s_map = st.session_state.search_bt_ind_map
            s_names = st.session_state.search_bt_names
            s_cfg = st.session_state.search_bt_cfg

            tested_str = ", ".join([f"{nm}({cd})" for cd, nm in s_names.items() if cd in s_map])
            st.success(f"백테스트 완료 ({len(s_map)}종목): **{tested_str}**")
            st.dataframe(s_sum, use_container_width=True, hide_index=True)

            if int(s_sum["거래 횟수"].sum()) == 0:
                st.info("선택한 종목에서 현재 엄격 조건에 일치한 과거 신호가 0건입니다. 상단 [⚙️ 전략·임계값 세부 설정]에서 역배열 일수나 이격도 기준을 조금 넓힌 뒤 다시 실행해 보세요.")
            else:
                b_c1, b_c2 = st.columns(2)
                with b_c1:
                    fig_win = px.bar(s_sum, x="보유기간(일)", y="승률(%)", color="전략", barmode="group", title="보유기간별 전략 승률(%)", text="승률(%)")
                    fig_win.update_layout(height=360, legend=dict(orientation="h", yanchor="bottom", y=-0.38))
                    st.plotly_chart(fig_win, use_container_width=True)
                with b_c2:
                    fig_exp = px.bar(s_sum, x="보유기간(일)", y="기대수익(%)", color="전략", barmode="group", title="보유기간별 기대수익(%)", text="기대수익(%)")
                    fig_exp.add_hline(y=0, line_dash="dash", line_color="#495057")
                    fig_exp.update_layout(height=360, legend=dict(orientation="h", yanchor="bottom", y=-0.38))
                    st.plotly_chart(fig_exp, use_container_width=True)

            st.markdown("#### 📈 검색 종목 과거 신호 발생 시점 차트 확인")
            chart_codes = list(s_map.keys())
            pick_code = st.selectbox("신호 위치 확인할 종목 선택", options=chart_codes, format_func=lambda c: f"{s_names.get(c, c)} ({c})")
            pdf = s_map[pick_code]
            masks = build_strategy_masks(pdf, s_cfg)
            start_off = s_cfg["backtest"]["start_offset_days"]
            plot_slice = pdf.iloc[max(0, start_off - 40):]

            fig_bt = go.Figure()
            fig_bt.add_trace(go.Scatter(x=plot_slice.index, y=plot_slice["Close"], name="종가", line=dict(color="#495057", width=1.6)))
            fig_bt.add_trace(go.Scatter(x=plot_slice.index, y=plot_slice["MA20"], name="20일선", line=dict(color="#f59f00", width=1.4)))
            fig_bt.add_trace(go.Scatter(x=plot_slice.index, y=plot_slice["MA50"], name="50일선", line=dict(color="#16a34a", width=1.6)))
            fig_bt.add_trace(go.Scatter(x=plot_slice.index, y=plot_slice["MA200"], name="200일선", line=dict(color="#1d4ed8", width=2.2, dash="dot")))

            styles = {
                "전략 A (수렴 + 거래량 급증)": ("triangle-up", "#0284c7", 12),
                "전략 B (수렴 + 골든크로스)": ("diamond", "#d97706", 12),
                "전략 C (수렴 + 거래량 + GC + 박스권돌파)": ("star", "#e11d48", 16),
            }
            for s_name, m_series in masks.items():
                valid_mask = m_series.iloc[start_off:]
                hit_dates = valid_mask[valid_mask].index
                if len(hit_dates) > 0:
                    sym, col, sz = styles[s_name]
                    fig_bt.add_trace(go.Scatter(
                        x=hit_dates, y=pdf.loc[hit_dates, "Close"], mode="markers", name=s_name,
                        marker=dict(symbol=sym, size=sz, color=col, line=dict(width=1, color="#fff"))
                    ))
            fig_bt.update_layout(title=f"{s_names.get(pick_code, pick_code)} ({pick_code}) 신호 발생 위치", height=480, hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.02))
            st.plotly_chart(fig_bt, use_container_width=True)

            if not s_tr.empty:
                st.dataframe(s_tr.sort_values("신호일", ascending=False), use_container_width=True, hide_index=True, height=320)
                st.download_button(
                    "📥 검색 종목 백테스트 거래내역 CSV 다운로드",
                    data=s_tr.to_csv(index=False).encode("utf-8-sig"),
                    file_name=f"search_backtest_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )

    else:
        if not st.session_state.scan_done or not indicator_map:
            st.info("먼저 상단에서 **[🚀 시장 스캔 시작]**을 누르거나, 위에서 한국/미국 종목 직접 검색을 선택해 주세요.")
        else:
            if st.button("🧪 방금 스캔한 전체 종목으로 전략 A/B/C 백테스트 실행", type="primary"):
                bt_all_cfg = copy.deepcopy(used_cfg)
                if scanned_is_us:
                    bt_all_cfg["backtest"]["commission_rate"] = 0.001
                    bt_all_cfg["backtest"]["sell_tax_rate"] = 0.00003
                with st.spinner("전략 A, B, C에 대해 10·20·40·60일 보유 성과를 계산 중입니다..."):
                    bt_sum, bt_tr = run_backtest_suite(indicator_map, bt_all_cfg)
                    if not bt_tr.empty and not all_df.empty:
                        code_nm = all_df.set_index("종목코드")["종목명 및 종목코드"].to_dict()
                        bt_tr.insert(3, "종목명", bt_tr["종목코드"].map(lambda x: code_nm.get(str(x), str(x))))
                    st.session_state.bt_summary = bt_sum
                    st.session_state.bt_trades = bt_tr

            bt_sum, bt_tr = st.session_state.bt_summary, st.session_state.bt_trades
            if bt_sum.empty:
                st.info("위 버튼을 누르시면 방금 스캔된 전체 종목에 대해 전략 A, B, C 성과를 즉시 비교합니다.")
            else:
                st.dataframe(bt_sum, use_container_width=True, hide_index=True)
                b_c1, b_c2 = st.columns(2)
                with b_c1:
                    fig_win = px.bar(bt_sum, x="보유기간(일)", y="승률(%)", color="전략", barmode="group", title="보유기간별 전략 승률(%)", text="승률(%)")
                    fig_win.update_layout(height=360, legend=dict(orientation="h", yanchor="bottom", y=-0.38))
                    st.plotly_chart(fig_win, use_container_width=True)
                with b_c2:
                    fig_exp = px.bar(bt_sum, x="보유기간(일)", y="기대수익(%)", color="전략", barmode="group", title="보유기간별 기대수익(%)", text="기대수익(%)")
                    fig_exp.add_hline(y=0, line_dash="dash", line_color="#495057")
                    fig_exp.update_layout(height=360, legend=dict(orientation="h", yanchor="bottom", y=-0.38))
                    st.plotly_chart(fig_exp, use_container_width=True)

                if not bt_tr.empty:
                    st.download_button(
                        "📥 전체 백테스트 거래내역 CSV 다운로드",
                        data=bt_tr.to_csv(index=False).encode("utf-8-sig"),
                        file_name=f"backtest_trades_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv"
                    )
