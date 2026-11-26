import io
import copy
import json
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 이전 단계에서 만든 모듈 4종 연결
from data_loader import load_config, get_filtered_universe, fetch_stock_ohlcv, is_excluded_by_meta, validate_ohlcv
from indicators import compute_all_indicators
from signal_engine import evaluate_signal_at_index
from backtester import run_backtest_suite

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
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# 클라우드 환경 안정성 보장용 안전 수집 래퍼 (KRX 차단 시 네이버 자동 전환)
# ---------------------------------------------------------
def fetch_naver_universe_fallback(min_mcap_eok: int, max_per_market: int = 300) -> pd.DataFrame:
    rows = []
    seen = set()
    for api_mkt, kor_mkt in [("KOSPI", "코스피"), ("KOSDAQ", "코스닥")]:
        page = 1
        cnt = 0
        while cnt < max_per_market and page <= 10:
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
                    if mcap_eok > 0 and mcap_eok < min_mcap_eok:
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


def fetch_naver_ohlcv_fallback(code: str, min_days: int) -> pd.DataFrame | None:
    try:
        url = f"https://fchart.stock.naver.com/sise.nhn?symbol={code}&timeframe=day&count=750&requestType=0"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=4) as res:
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


def safe_fetch_ohlcv(code: str, cfg: dict) -> pd.DataFrame | None:
    df = fetch_stock_ohlcv(code, cfg)
    if df is None:
        df = fetch_naver_ohlcv_fallback(code, cfg["universe"]["min_trading_days"])
    return df


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

if "scan_done" not in st.session_state:
    st.session_state.scan_done = False
    st.session_state.all_df = pd.DataFrame()
    st.session_state.passed_df = pd.DataFrame()
    st.session_state.indicator_map = {}
    st.session_state.updated_at = ""
    st.session_state.total_target = 0
    st.session_state.bt_summary = pd.DataFrame()
    st.session_state.bt_trades = pd.DataFrame()
    st.session_state.used_cfg = copy.deepcopy(BASE_CFG)


# ---------------------------------------------------------
# 상단 헤더 및 간단 조작 패널
# ---------------------------------------------------------
st.title("📈 이평선 수렴 · 거래량 돌파 스윙 포착 시스템")
st.caption("코스피·코스닥 3년 일봉 분석 · 장기 하락 → 바닥 박스권 → 20·40·50·200일선 수렴 → 거래량 급증 & 골든크로스 → 백테스트")

top_c1, top_c2, top_c3 = st.columns([1.4, 1.4, 1.2])
with top_c1:
    scan_preset = st.selectbox(
        "🔍 스캔 범위 선택 (시가총액 상위 순)",
        [
            "빠른 스캔: 상위 150종목 (약 15~25초)",
            "표준 스캔: 상위 300종목 (약 35~50초)",
            "정밀 스캔: 상위 600종목 (약 1분 30초)",
            "전체 스캔: 조건 통과 전 종목 (약 3~5분)"
        ],
        index=0
    )
with top_c2:
    custom_codes = st.text_input("➕ 꼭 포함할 종목코드 (선택, 쉼표 구분)", placeholder="예: 005930, 035720, 011170")
with top_c3:
    st.write("")
    run_btn = st.button("🚀 지금 스캔 시작하기", type="primary", use_container_width=True)

# 세부 조건 설정 패널 (기본값은 config.json 그대로 적용)
with st.expander("⚙️ 전략·임계값 세부 설정 (기본값 그대로 실행해도 됩니다)", expanded=False):
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

# 스캔 수량 파싱
if "150" in scan_preset:
    scan_limit = 150
elif "300" in scan_preset:
    scan_limit = 300
elif "600" in scan_preset:
    scan_limit = 600
else:
    scan_limit = None


# ---------------------------------------------------------
# 스캔 실행 로직 (실시간 진행률 바 표시)
# ---------------------------------------------------------
if run_btn:
    progress_bar = st.progress(0, text="1/3) 코스피·코스닥 종목 목록 조회 및 제외 필터 적용 중...")
    uni_df = safe_get_universe(cfg)

    if uni_df.empty:
        st.error("종목 목록을 불러오지 못했습니다. 잠시 후 다시 시도해 주세요.")
        st.stop()

    if scan_limit:
        target_df = uni_df.head(scan_limit).copy()
    else:
        target_df = uni_df.copy()

    # 사용자 추가 종목 병합
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

    progress_bar.progress(100, text="3/3) 분석 완료! 결과를 정리했습니다.")

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
    st.session_state.used_cfg = cfg
    progress_bar.empty()


# ---------------------------------------------------------
# 초기 대기 안내 화면
# ---------------------------------------------------------
if not st.session_state.scan_done:
    st.info("👆 상단의 **[🚀 지금 스캔 시작하기]** 버튼을 누르시면 코스피·코스닥 3년 일봉을 분석해 **이동평균선 수렴 · 거래량 급증 · 골든크로스 · 박스권 돌파** 종목을 자동으로 발굴합니다.")
    st.stop()

all_df = st.session_state.all_df
passed_df = st.session_state.passed_df
indicator_map = st.session_state.indicator_map
used_cfg = st.session_state.used_cfg

if all_df.empty:
    st.warning("유효한 일봉 데이터를 확보한 종목이 없습니다. 최소 거래량·시가총액 조건을 확인하거나 다시 스캔해 주세요.")
    st.stop()

# 상단 핵심 지표 요약
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
# 6대 대시보드 탭
# ---------------------------------------------------------
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "① 앤트위키 시그널 지도",
    "② 결과표 & 엑셀 저장",
    "③ 종목별 근거 & 매매전략",
    "④ 20·40·50·200일 차트",
    "⑤ 4분면 수렴 궤적",
    "⑥ 전략 A/B/C 백테스트"
])

# [탭 1] 앤트위키 트리맵
with tab1:
    t1_mode = st.radio(
        "지도 표시 대상",
        ["✅ 조건 통과 종목 (D/C/B/A 시그널)", "📋 전체 분석 종목 중 점수 상위 40개"],
        horizontal=True
    )
    map_df = passed_df.copy() if "통과" in t1_mode else all_df.sort_values("최종 점수", ascending=False).head(40).copy()

    if map_df.empty:
        st.info("현재 엄격 조건을 모두 통과한 종목이 0개입니다. 위에서 '전체 분석 종목 중 점수 상위 40개'를 선택하거나, 상단 설정에서 수렴 이격도/역배열 일수를 조금 완화해 보세요.")
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
                "RSI(14) & MACD 히스토그램"
            )
        )

        # 일봉 캔들
        fig.add_trace(go.Candlestick(
            x=cdf["Date"], open=cdf["Open"], high=cdf["High"], low=cdf["Low"], close=cdf["Close"],
            name="일봉", increasing_line_color="#e12343", decreasing_line_color="#1964d8"
        ), row=1, col=1)

        # 4개 핵심 이동평균선
        fig.add_trace(go.Scatter(x=cdf["Date"], y=cdf["MA20"], name="20일선", line=dict(color="#f59f00", width=1.8)), row=1, col=1)
        fig.add_trace(go.Scatter(x=cdf["Date"], y=cdf["MA40"], name="40일선", line=dict(color="#e11d48", width=2.2)), row=1, col=1)
        fig.add_trace(go.Scatter(x=cdf["Date"], y=cdf["MA50"], name="50일선", line=dict(color="#16a34a", width=2.0)), row=1, col=1)
        fig.add_trace(go.Scatter(x=cdf["Date"], y=cdf["MA200"], name="200일선", line=dict(color="#1d4ed8", width=2.6, dash="dot")), row=1, col=1)

        # 최근 박스권 상단/하단 수평선
        latest_box_h = float(cdf["Box_High"].iloc[-1])
        latest_box_l = float(cdf["Box_Low"].iloc[-1])
        fig.add_hline(y=latest_box_h, line_dash="dash", line_color="#d97706", annotation_text=f"박스상단 {int(latest_box_h):,}", row=1, col=1)
        fig.add_hline(y=latest_box_l, line_dash="dash", line_color="#0284c7", annotation_text=f"박스하단 {int(latest_box_l):,}", row=1, col=1)

        # 골든크로스 지점 별표 표시
        gc_pts = cdf[cdf["GC_Any"] == True]
        if not gc_pts.empty:
            fig.add_trace(go.Scatter(
                x=gc_pts["Date"], y=gc_pts["Close"],
                mode="markers", name="골든크로스 발생일",
                marker=dict(symbol="star", size=14, color="#f59f00", line=dict(width=1.2, color="#191f28"))
            ), row=1, col=1)

        # 거래량
        v_cols = ["#fca5a5" if c >= o else "#93c5fd" for c, o in zip(cdf["Close"], cdf["Open"])]
        fig.add_trace(go.Bar(x=cdf["Date"], y=cdf["Volume"], name="거래량", marker_color=v_cols), row=2, col=1)
        fig.add_trace(go.Scatter(x=cdf["Date"], y=cdf["Vol_MA20"], name="20일 평균거래량", line=dict(color="#191f28", width=1.6)), row=2, col=1)

        # RSI & MACD Hist
        fig.add_trace(go.Scatter(x=cdf["Date"], y=cdf["RSI14"], name="RSI(14)", line=dict(color="#7c3aed", width=2)), row=3, col=1)
        fig.add_hline(y=50, line_dash="dash", line_color="#868e96", row=3, col=1)
        fig.add_hline(y=70, line_dash="dot", line_color="#e12343", row=3, col=1)

        fig.update_layout(height=760, xaxis_rangeslider_visible=False, hovermode="x unified", margin=dict(t=40, l=20, r=20, b=20))
        st.plotly_chart(fig, use_container_width=True)

# [탭 5] 앤트위키 4분면 이평선 수렴·전환 궤적
with tab5:
    st.subheader("이평선 수렴 & 상승 전환 4분면 궤적 (최근 20영업일)")
    st.write("세로 기준선(100)은 **20일선과 200일선이 만나는 지점**이며, 가로 기준선(100) 위쪽은 **20일선이 상승 기울기로 전환해 200일선을 향해 가속**하는 구간입니다.")

    q_df = passed_df.head(14) if not passed_df.empty else all_df.sort_values("최종 점수", ascending=False).head(14)

    fig_q = go.Figure()
    quads = [
        (100, 115, 100, 115, "rgba(254,202,202,0.35)", "1사분면: 골든크로스 돌파 & 상승가속", 107.5, 113.5),
        (85, 100, 100, 115, "rgba(254,240,138,0.35)", "2사분면: 역배열 수렴 & 상승 전환 임박", 92.5, 113.5),
        (85, 100, 85, 100, "rgba(191,219,254,0.35)", "3사분면: 장기 역배열 침체 구간", 92.5, 86.5),
        (100, 115, 85, 100, "rgba(233,213,255,0.35)", "4사분면: 정배열 중 단기 숨고르기", 107.5, 86.5),
    ]
    for x0, x1, y0, y1, c, lbl, lx, ly in quads:
        fig_q.add_shape(type="rect", x0=x0, x1=x1, y0=y0, y1=y1, fillcolor=c, line=dict(width=0), layer="below")
        fig_q.add_annotation(x=lx, y=ly, text=f"<b>{lbl}</b>", showarrow=False)

    fig_q.add_hline(y=100, line_dash="dash", line_color="#475569")
    fig_q.add_vline(x=100, line_dash="dash", line_color="#475569")

    for _, r in q_df.iterrows():
        code = str(r["종목코드"])
        if code not in indicator_map:
            continue
        idf = indicator_map[code]
        if len(idf) < 25:
            continue
        pts = [len(idf) - 1 - k * 4 for k in range(4, -1, -1)]
        xs = [float(np.clip(100.0 + ((idf.iloc[i]["MA20"] - idf.iloc[i]["MA200"]) / idf.iloc[i]["MA200"]) * 100.0, 85.5, 114.5)) for i in pts]
        ys = [float(np.clip(100.0 + idf.iloc[i]["Slope20_5d"] * 2.5, 85.5, 114.5)) for i in pts]
        short_name = r["종목명 및 종목코드"].split(" (")[0]

        fig_q.add_trace(go.Scatter(
            x=xs, y=ys, mode="lines+markers+text",
            name=f"{short_name} ({r['최종 점수']}점)",
            text=["", "", "", "", short_name],
            textposition="top center",
            line=dict(width=3, shape="spline"),
            marker=dict(size=[5, 6, 7, 8, 13])
        ))

    fig_q.update_layout(
        xaxis=dict(title="20일선 vs 200일선 상대 위치 지수 (100 = 교차 / 100+ = 상향 돌파)", range=[85, 115]),
        yaxis=dict(title="20일선 상승 전환 모멘텀 (100+ = 상승 기울기 가속)", range=[85, 115]),
        height=560, margin=dict(t=20, l=20, r=20, b=20)
    )
    st.plotly_chart(fig_q, use_container_width=True)

# [탭 6] 전략 A / B / C 독립 백테스트 비교 대시보드
with tab6:
    st.subheader("전략 A · B · C 과거 3년 백테스트 성과 비교 (10 · 20 · 40 · 60일 보유)")
    st.caption("체결 기준: t일 장 마감 신호 확정 ➔ t+1일 시가(Open) 진입 ➔ 수수료(0.015%)·거래세(0.18%)·슬리피지(0.15%) 차감 반영 (룩어헤드 바이어스 차단)")

    if st.button("🧪 방금 스캔한 종목으로 전략 A/B/C 백테스트 실행", type="primary"):
        with st.spinner("전략 A, B, C에 대해 10·20·40·60일 보유 성과를 계산 중입니다..."):
            bt_sum, bt_tr = run_backtest_suite(indicator_map, used_cfg)
            st.session_state.bt_summary = bt_sum
            st.session_state.bt_trades = bt_tr

    bt_sum = st.session_state.bt_summary
    bt_tr = st.session_state.bt_trades

    if bt_sum.empty:
        st.info("위 **[🧪 백테스트 실행]** 버튼을 누르시면 이미 수집된 3년 일봉 데이터를 활용해 전략 A, B, C의 승률·기대수익·손익비·MDD를 즉시 비교해 드립니다.")
    else:
        st.dataframe(bt_sum, use_container_width=True, hide_index=True)

        b_c1, b_c2 = st.columns(2)
        with b_c1:
            fig_win = px.bar(
                bt_sum, x="보유기간(일)", y="승률(%)", color="전략", barmode="group",
                title="보유기간별 전략 승률(%) 비교", text="승률(%)"
            )
            fig_win.update_layout(height=380, legend=dict(orientation="h", yanchor="bottom", y=-0.35))
            st.plotly_chart(fig_win, use_container_width=True)
        with b_c2:
            fig_exp = px.bar(
                bt_sum, x="보유기간(일)", y="기대수익(%)", color="전략", barmode="group",
                title="보유기간별 1회 매매 기대수익(%) 비교", text="기대수익(%)"
            )
            fig_exp.add_hline(y=0, line_dash="dash", line_color="#495057")
            fig_exp.update_layout(height=380, legend=dict(orientation="h", yanchor="bottom", y=-0.35))
            st.plotly_chart(fig_exp, use_container_width=True)

        if not bt_tr.empty:
            st.download_button(
                "📥 백테스트 개별 거래내역 CSV 다운로드",
                data=bt_tr.to_csv(index=False).encode("utf-8-sig"),
                file_name=f"backtest_trades_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
