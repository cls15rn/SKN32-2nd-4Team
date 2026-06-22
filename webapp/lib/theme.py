"""
webapp/lib/theme.py

와이어프레임(이미지 5~12)의 시각 언어를 그대로 따른다:
- 크림색 사이드바, 흰 본문, 딥마룬(brick) 데이터 강조, 카드형 레이아웃.
Streamlit 기본 위젯과 싸우지 않고, 카드/막대는 직접 HTML+CSS로 그려 픽셀을 통제한다.
"""
from __future__ import annotations

import streamlit as st

# --- 팔레트 (단일 출처) ---------------------------------------------------
INK = "#2b2b2b"
MUTED = "#8d887b"
FAINT = "#a8a294"
MAROON = "#7c2b2b"
MAROON_DEEP = "#6a2424"
TRACK = "#ece6da"       # 막대 배경 트랙
CARD_BG = "#ffffff"
CARD_BORDER = "#e8e4d8"
CREAM = "#f1eee4"       # 사이드바
ACTIVE = "#d8e1f4"      # 활성 nav
ACTIVE_INK = "#3a57b0"
GOOD = "#3f7d52"

CSS = f"""
<style>
@import url('https://cdn.jsdelivr.net/gh/orioncactus/pretendard@v1.3.9/dist/web/static/pretendard.css');

html, body, [class*="css"], .stApp {{
    font-family: 'Pretendard', 'Noto Sans KR', -apple-system, BlinkMacSystemFont, sans-serif;
    color: {INK};
}}
.stApp {{ background: #ffffff; }}

/* 상단 기본 헤더/메뉴 정리 */
header[data-testid="stHeader"] {{ background: transparent; }}
#MainMenu, footer {{ visibility: hidden; }}

/* 본문 폭/여백 */
.block-container {{ padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1180px; }}

/* ---------------- 사이드바 ---------------- */
[data-testid="stSidebar"] {{ background: {CREAM}; border-right: 1px solid {CARD_BORDER}; }}
[data-testid="stSidebar"] .block-container {{ padding-top: 1.4rem; }}

.sb-brand {{ font-size: 1.35rem; font-weight: 800; letter-spacing:-.02em; color:{INK};
            padding: .1rem .2rem .2rem; }}
.sb-foot {{ margin-top: 1.4rem; padding-top: 1rem; border-top:1px solid {CARD_BORDER};
            color:{FAINT}; font-size:.74rem; line-height:1.7; }}
.sb-foot b {{ color:{MUTED}; font-weight:600; }}

/* st.navigation 링크 모양 다듬기 */
[data-testid="stSidebar"] a[data-testid="stPageLink-NavLink"] {{
    border-radius: 9px; padding: .42rem .6rem; margin: 1px 0; font-size:.92rem;
}}
[data-testid="stSidebar"] a[data-testid="stPageLink-NavLink"]:hover {{ background: #e9e5d8; }}

/* ---------------- 페이지 헤더 ---------------- */
.page-title {{ font-size: 2rem; font-weight: 800; letter-spacing:-.02em; margin:0 0 .15rem; }}
.page-sub {{ color:{MUTED}; font-size:.96rem; margin:0 0 1.3rem; }}

/* ---------------- 카드 ---------------- */
.card {{ background:{CARD_BG}; border:1px solid {CARD_BORDER}; border-radius:16px;
         padding:1.15rem 1.3rem; margin-bottom:1.05rem; }}
.card h3 {{ font-size:1.12rem; font-weight:700; margin:.1rem 0 .15rem; }}
.card .csub {{ color:{MUTED}; font-size:.84rem; margin:0 0 1rem; line-height:1.5; }}

/* KPI */
.kpi-wrap {{ display:flex; gap:.8rem; flex-wrap:wrap; margin-bottom:1.05rem; }}
.kpi {{ flex:1 1 0; min-width:150px; background:{CARD_BG}; border:1px solid {CARD_BORDER};
        border-radius:14px; padding:.95rem 1.05rem; }}
.kpi .k-label {{ color:{MUTED}; font-size:.82rem; margin-bottom:.35rem; }}
.kpi .k-val {{ font-size:1.7rem; font-weight:800; letter-spacing:-.01em; }}
.kpi .k-val small {{ font-size:.95rem; font-weight:700; color:{MAROON}; margin-left:.25rem; }}
.kpi.accent .k-val {{ color:{MAROON}; }}

/* 가로 막대 (이탈률·예상손실·집중도) */
.hbar {{ margin: .55rem 0 .8rem; }}
.hbar-top {{ display:flex; align-items:baseline; gap:.5rem; margin-bottom:.32rem; }}
.hbar-label {{ font-weight:700; font-size:.95rem; }}
.hbar-meta {{ color:{MUTED}; font-size:.82rem; }}
.hbar-right {{ margin-left:auto; font-weight:800; color:{MAROON}; font-size:1.0rem; }}
.hbar-right .conc {{ color:{MUTED}; font-weight:600; font-size:.8rem; margin-left:.35rem; }}
.hbar-track {{ background:{TRACK}; border-radius:8px; height:13px; width:100%; overflow:hidden; }}
.hbar-fill {{ background:{MAROON}; height:100%; border-radius:8px; }}

/* 작은 칩 (구간 내 흔한 신호 등) */
.chip {{ display:inline-block; background:#f1efe8; color:{MUTED}; border-radius:999px;
         padding:.2rem .6rem; font-size:.78rem; margin:.15rem .25rem .15rem 0; }}
.tag {{ display:inline-block; background:#f0ede4; color:{MUTED}; border-radius:7px;
        padding:.16rem .5rem; font-size:.78rem; margin-right:.35rem; }}
.tag.sig {{ color:{MAROON_DEEP}; background:#f3e6e2; font-weight:600; }}

/* 우선대응 고객 리스트 행 */
.cust {{ border-top:1px solid #efece2; padding:.85rem .2rem; }}
.cust:first-child {{ border-top:none; }}
.cust-head {{ display:flex; align-items:center; gap:.55rem; }}
.cust-rank {{ width:24px; height:24px; border-radius:50%; background:#f0ede4; color:{MUTED};
             font-size:.8rem; font-weight:700; display:inline-flex; align-items:center; justify-content:center; }}
.cust-id {{ font-weight:800; font-size:1.02rem; letter-spacing:.01em; }}
.cust-loss {{ margin-left:auto; text-align:right; }}
.cust-loss .big {{ color:{MAROON}; font-weight:800; font-size:1.15rem; }}
.cust-loss .calc {{ color:{MUTED}; font-size:.78rem; }}
.cust-cause {{ color:{MUTED}; font-size:.86rem; margin-top:.4rem; }}
.cust-cause b {{ color:{INK}; font-weight:600; }}

/* 검증 통과 배지 */
.badge-ok {{ display:inline-block; background:{ACTIVE}; color:{ACTIVE_INK}; font-weight:700;
            font-size:.78rem; border-radius:999px; padding:.2rem .65rem; }}
.stat-grid {{ display:flex; gap:2.2rem; flex-wrap:wrap; margin:.4rem 0 .2rem; }}
.stat .s-label {{ color:{MUTED}; font-size:.84rem; }}
.stat .s-val {{ font-size:1.55rem; font-weight:800; }}
.stat .s-sub {{ color:{FAINT}; font-size:.78rem; }}

.note {{ color:{FAINT}; font-size:.8rem; line-height:1.6; margin-top:.4rem; }}
.callout {{ background:#faf8f2; border:1px solid {CARD_BORDER}; border-radius:14px;
            padding:1rem 1.15rem; color:{INK}; font-size:.92rem; line-height:1.65; }}
.callout b {{ color:{MAROON}; }}
hr.soft {{ border:none; border-top:1px solid #efece2; margin:.9rem 0; }}

/* 이탈/유지 분할 바 (홈) */
.splitbar {{ display:flex; height:30px; border-radius:9px; overflow:hidden; margin:.15rem 0 .2rem; }}
.splitbar .seg {{ display:flex; align-items:center; justify-content:center; font-size:.82rem;
                 font-weight:700; white-space:nowrap; }}
.splitbar .seg.churn {{ background:{MAROON}; color:#fff; }}
.splitbar .seg.keep {{ background:{TRACK}; color:{INK}; }}

/* 데이터 업로드 플레이스홀더 (추후 개발) */
.uploadbox {{ border:1.5px dashed #cfc9bb; border-radius:14px; padding:1.6rem 1.2rem;
             text-align:center; background:#fcfbf7; }}
.uploadbox .u-title {{ font-weight:700; color:{MUTED}; font-size:.98rem; margin-bottom:.3rem; }}
.uploadbox .u-sub {{ font-size:.82rem; color:{FAINT}; line-height:1.6; }}
.uploadbox .u-pill {{ display:inline-block; margin-top:.7rem; font-size:.74rem; color:{MUTED};
                     background:#f0ede4; border-radius:999px; padding:.2rem .75rem; }}

/* 분석B 속성 그룹 패널 — 속성별로 묶어 시각적으로 구분 */
.attr-group {{ border:1px solid {CARD_BORDER}; border-radius:12px; background:#fcfbf7;
              padding:.75rem .95rem .4rem; margin-bottom:.7rem; }}
.attr-group .ag-title {{ font-weight:700; font-size:.95rem; color:{INK}; margin:0 0 .55rem;
                        display:flex; align-items:center; gap:.45rem; }}
.attr-group .ag-title::before {{ content:""; display:inline-block; width:3px; height:13px;
                                background:{MAROON}; border-radius:2px; }}

/* 섹션 구분 라벨 (홈 흐름: 현황→핵심→행동→근거) */
.eyebrow {{ font-size:.74rem; letter-spacing:.05em; color:{FAINT}; font-weight:700;
           margin:.25rem 0 .55rem; }}

/* 진입 카드 (홈 → 각 페이지 관문, 카드 전체가 링크) */
.entry-grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(180px,1fr));
              gap:.8rem; margin-bottom:1.05rem; }}
.entry-link {{ text-decoration:none; color:inherit; display:block; }}
.entry {{ position:relative; background:{CARD_BG}; border:1px solid {CARD_BORDER};
         border-radius:14px; padding:.9rem 1rem .95rem; min-height:106px;
         transition:border-color .12s ease, background .12s ease; }}
.entry.muted {{ background:#fcfbf7; }}
.entry-link:hover .entry {{ border-color:{MAROON}; background:#fcfaf5; }}
.entry .e-ico {{ font-size:1.25rem; line-height:1; }}
.entry .e-t {{ font-weight:800; font-size:1.02rem; margin:.4rem 0 .18rem; }}
.entry.muted .e-t {{ color:{MUTED}; }}
.entry .e-d {{ color:{MUTED}; font-size:.82rem; line-height:1.5; }}
.entry .e-go {{ position:absolute; top:.7rem; right:.85rem; color:{FAINT}; font-size:.95rem; }}
.entry-link:hover .e-go {{ color:{MAROON}; }}

/* 아키텍처 선언 + 이원화 레이어(거시/미시) 상태 */
.arch {{ background:#fbf9f3; border:1px solid {CARD_BORDER}; border-radius:16px;
        padding:1.1rem 1.25rem; margin-bottom:1.15rem; }}
.arch-decl {{ font-size:1.0rem; font-weight:700; line-height:1.6; color:{INK};
             margin-bottom:.95rem; }}
.layer-grid {{ display:grid; grid-template-columns:1fr auto 1fr; gap:.7rem; align-items:stretch; }}
.layer {{ background:{CARD_BG}; border:1px solid {CARD_BORDER}; border-radius:12px;
         padding:.8rem .95rem; }}
.layer.macro {{ border-left:3px solid {MAROON}; }}
.layer.micro {{ border-left:3px solid #0f6e56; }}
.layer .l-top {{ display:flex; align-items:baseline; gap:.5rem; margin-bottom:.35rem; flex-wrap:wrap; }}
.layer .l-name {{ font-weight:800; font-size:.98rem; }}
.layer .l-tag {{ font-size:.72rem; color:{MUTED}; background:#f0ede4; border-radius:6px; padding:.1rem .42rem; }}
.layer .l-meta {{ color:{MUTED}; font-size:.82rem; line-height:1.5; }}
.layer .l-meta b {{ color:{INK}; font-weight:700; }}
.layer .l-stat {{ display:inline-block; margin-top:.45rem; font-size:.8rem; font-weight:700; }}
.layer .l-stat.ok {{ color:#1d7d56; }}
.layer .l-stat.warn {{ color:#9a6a1a; }}
.layer-plus {{ display:flex; align-items:center; justify-content:center; color:{MUTED};
              font-size:1.3rem; font-weight:800; }}

/* 레이어 패널 내 간단 표 (분석 B) */
table.mini {{ width:100%; border-collapse:collapse; font-size:.84rem; margin-top:.35rem; }}
table.mini th {{ text-align:left; color:{MUTED}; font-weight:700; padding:.38rem .5rem;
                border-bottom:1px solid {CARD_BORDER}; }}
table.mini td {{ padding:.38rem .5rem; border-bottom:1px solid #efe9dd; }}
table.mini td:nth-child(3), table.mini th:nth-child(3) {{ color:{INK}; font-variant-numeric:tabular-nums; }}
</style>
"""


def inject():
    st.markdown(CSS, unsafe_allow_html=True)


def html(s: str):
    st.markdown(s, unsafe_allow_html=True)


# --- 렌더 헬퍼 (HTML 문자열 반환) ----------------------------------------
def page_header(title: str, sub: str = "") -> str:
    sub_html = f'<div class="page-sub">{sub}</div>' if sub else ""
    return f'<div class="page-title">{title}</div>{sub_html}'


def kpi(label: str, value: str, accent: bool = False, suffix: str = "") -> str:
    cls = "kpi accent" if accent else "kpi"
    suf = f"<small>{suffix}</small>" if suffix else ""
    return f'<div class="{cls}"><div class="k-label">{label}</div><div class="k-val">{value}{suf}</div></div>'


def kpi_row(cards: list[str]) -> str:
    return f'<div class="kpi-wrap">{"".join(cards)}</div>'


def hbar(label: str, pct: float, right: str, meta: str = "", conc: str = "",
         maxpct: float = 100.0) -> str:
    width = max(0.0, min(100.0, pct / maxpct * 100.0))
    meta_html = f'<span class="hbar-meta">{meta}</span>' if meta else ""
    conc_html = f'<span class="conc">{conc}</span>' if conc else ""
    return (
        '<div class="hbar"><div class="hbar-top">'
        f'<span class="hbar-label">{label}</span>{meta_html}'
        f'<span class="hbar-right">{right}{conc_html}</span></div>'
        f'<div class="hbar-track"><div class="hbar-fill" style="width:{width:.1f}%"></div></div></div>'
    )


def card(inner: str) -> str:
    return f'<div class="card">{inner}</div>'


def card_title(title: str, sub: str = "") -> str:
    sub_html = f'<div class="csub">{sub}</div>' if sub else ""
    return f"<h3>{title}</h3>{sub_html}"
