import os
import json
import logging
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import numpy as np
import FinanceDataReader as fdr
from tqdm import tqdm

logger = logging.getLogger(__name__)

EXCLUDE_KEYWORDS = [
    "스팩", "SPAC", "ETF", "ETN", "ELW", "리츠", "인버스", "레버리지", "선물",
    "KODEX", "TIGER", "KBSTAR", "RISE", "ACE", "SOL", "HANARO", "ARIRANG",
    "PLUS", "KOSEF", "KIWOOM", "TIMEFOLIO", "WOORI", "TREX", "SMART", "FOCUS",
    "인프라", "맥쿼리", "펀드", "채권", "국고채", "회사채", "단기채"
]

RISK_DEPT_KEYWORDS = [
    "관리종목", "투자주의환기", "정리매매", "거래정지", "상장폐지", "투자위험"
]


def load_config(path: str = "config.json") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def is_excluded_by_meta(code: str, name: str, dept: str = "") -> bool:
    """ETF/ETN/SPAC/우선주/관리·정지 종목 제외 필터"""
    if not code or len(code) != 6 or not code.isdigit():
        return True
    # 보통주만 허용 (우선주 및 종류주식은 종목코드 끝자리가 0이 아님)
    if code[-1] != "0":
        return True

    upper_name = str(name).upper()
    if any(kw in upper_name for kw in EXCLUDE_KEYWORDS):
        return True
    if str(name).endswith(("우", "우B", "우C", "1우", "2우", "3우B", "(전환)")):
        return True

    dept_str = str(dept)
    if any(rk in dept_str for rk in RISK_DEPT_KEYWORDS):
        return True

    return False


def get_filtered_universe(cfg: dict, include_delisted: bool = False) -> pd.DataFrame:
    """
    코스피·코스닥 전체 상장 종목 로드 및 1차 제외 조건 적용.
    실제 데이터가 조회되지 않으면 빈 DataFrame 반환(임의 데이터 생성 금지).
    """
    u_cfg = cfg["universe"]
    allowed_markets = set(u_cfg["markets"])
    min_mcap_won = u_cfg["min_mcap_eok"] * 1e8

    try:
        listing = fdr.StockListing("KRX")
    except Exception as e:
        logger.error(f"KRX 종목 리스트 조회 실패: {e}")
        return pd.DataFrame()

    if listing is None or listing.empty:
        return pd.DataFrame()

    # 버전별 컬럼명 유연 대응
    col_map = {}
    if "Symbol" in listing.columns and "Code" not in listing.columns:
        col_map["Symbol"] = "Code"
    listing = listing.rename(columns=col_map)

    if "Market" not in listing.columns or "Code" not in listing.columns:
        return pd.DataFrame()

    # KOSPI / KOSDAQ 필터
    listing["Market_Clean"] = listing["Market"].astype(str).str.upper().replace({"KOSDAQ GLOBAL": "KOSDAQ"})
    df = listing[listing["Market_Clean"].isin(allowed_markets)].copy()

    # 이름·코드·소속부 제외 필터
    dept_series = df["Dept"].fillna("") if "Dept" in df.columns else pd.Series("", index=df.index)
    mask_keep = [
        not is_excluded_by_meta(str(c).strip(), str(n).strip(), str(d))
        for c, n, d in zip(df["Code"], df["Name"], dept_series)
    ]
    df = df[mask_keep].copy()

    # 시가총액 기준 필터 (Marcap 컬럼이 존재하는 경우)
    if "Marcap" in df.columns and min_mcap_won > 0:
        df["Marcap"] = pd.to_numeric(df["Marcap"], errors="coerce").fillna(0)
        df = df[df["Marcap"] >= min_mcap_won].copy()
    else:
        df["Marcap"] = 0.0

    result = pd.DataFrame({
        "code": df["Code"].astype(str).str.zfill(6),
        "name": df["Name"].astype(str).str.strip(),
        "market": df["Market_Clean"].map({"KOSPI": "코스피", "KOSDAQ": "코스닥"}),
        "mcap_eok": (df["Marcap"] / 1e8).round(0).astype(int),
        "delisted": False
    }).drop_duplicates(subset=["code"]).reset_index(drop=True)

    # 백테스트 생존편향 완화를 위한 상장폐지 종목 병합 옵션
    if include_delisted:
        try:
            delist = fdr.StockListing("KRX-DELISTING")
            if delist is not None and not delist.empty and "Symbol" in delist.columns:
                delist["DelistingDate"] = pd.to_datetime(delist.get("DelistingDate"), errors="coerce")
                cutoff = datetime.today() - timedelta(days=u_cfg["lookback_calendar_days"])
                recent_delist = delist[
                    (delist["DelistingDate"] >= cutoff) &
                    (delist["Market"].isin(["KOSPI", "KOSDAQ"]))
                ].copy()
                d_keep = [
                    not is_excluded_by_meta(str(c).strip(), str(n).strip(), "")
                    for c, n in zip(recent_delist["Symbol"], recent_delist["Name"])
                ]
                recent_delist = recent_delist[d_keep]
                if not recent_delist.empty:
                    delist_df = pd.DataFrame({
                        "code": recent_delist["Symbol"].astype(str).str.zfill(6),
                        "name": recent_delist["Name"].astype(str).str.strip() + "(상폐)",
                        "market": recent_delist["Market"].map({"KOSPI": "코스피", "KOSDAQ": "코스닥"}),
                        "mcap_eok": 0,
                        "delisted": True
                    })
                    result = pd.concat([result, delist_df], ignore_index=True).drop_duplicates(subset=["code"])
        except Exception as e:
            logger.warning(f"상장폐지 종목 리스트 병합 생략: {e}")

    return result


def validate_ohlcv(df: pd.DataFrame, min_days: int) -> pd.DataFrame | None:
    """결측값·가격 오류·거래정지 엄격 검증 (임의 데이터 보간 금지)"""
    if df is None or df.empty:
        return None

    required = ["Open", "High", "Low", "Close", "Volume"]
    if not all(col in df.columns for col in required):
        return None

    clean = df[required].copy()
    clean.index = pd.to_datetime(clean.index)
    clean = clean[~clean.index.duplicated(keep="last")].sort_index()

    # 수치형 변환 및 결측/이상치 제거
    for col in required:
        clean[col] = pd.to_numeric(clean[col], errors="coerce")

    if clean[required].isna().any().any():
        clean = clean.dropna(subset=required)

    # 가격 오류(0 이하, 고가 < 저가) 제거
    valid_mask = (
        (clean["Open"] > 0) &
        (clean["High"] > 0) &
        (clean["Low"] > 0) &
        (clean["Close"] > 0) &
        (clean["High"] >= clean["Low"]) &
        (clean["Volume"] >= 0)
    )
    clean = clean[valid_mask]

    if len(clean) < min_days:
        return None

    # 최근 5거래일 내 거래정지(거래량 0) 제외
    if (clean["Volume"].tail(5) == 0).any():
        return None

    return clean


def fetch_stock_ohlcv(code: str, cfg: dict, as_of_date: str | None = None) -> pd.DataFrame | None:
    """단일 종목 3년 일봉 수집 + 일일 로컬 캐시 활용"""
    u_cfg = cfg["universe"]
    cache_dir = cfg["paths"]["cache_dir"]
    os.makedirs(cache_dir, exist_ok=True)

    end_dt = datetime.strptime(as_of_date, "%Y-%m-%d") if as_of_date else datetime.today()
    start_dt = end_dt - timedelta(days=u_cfg["lookback_calendar_days"])
    today_tag = end_dt.strftime("%Y%m%d")
    cache_path = os.path.join(cache_dir, f"{code}_{today_tag}.csv")

    if u_cfg.get("use_cache", True) and os.path.exists(cache_path):
        try:
            cached = pd.read_csv(cache_path, index_col=0, parse_dates=True)
            return validate_ohlcv(cached, u_cfg["min_trading_days"])
        except Exception:
            pass

    try:
        raw = fdr.DataReader(code, start_dt.strftime("%Y-%m-%d"), end_dt.strftime("%Y-%m-%d"))
        clean = validate_ohlcv(raw, u_cfg["min_trading_days"])
        if clean is not None and u_cfg.get("use_cache", True):
            clean.to_csv(cache_path)
        return clean
    except Exception:
        return None


def fetch_universe_ohlcv(universe_df: pd.DataFrame, cfg: dict, max_stocks: int | None = None) -> dict[str, pd.DataFrame]:
    """전체 유니버스 일봉 병렬 수집"""
    target_df = universe_df.head(max_stocks) if max_stocks else universe_df
    data_map: dict[str, pd.DataFrame] = {}
    workers = cfg["universe"].get("max_workers", 8)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_map = {
            executor.submit(fetch_stock_ohlcv, row["code"], cfg): row["code"]
            for _, row in target_df.iterrows()
        }
        for future in tqdm(as_completed(future_map), total=len(future_map), desc="일봉 데이터 수집"):
            code = future_map[future]
            try:
                df = future.result()
                if df is not None:
                    data_map[code] = df
            except Exception:
                continue

    return data_map
