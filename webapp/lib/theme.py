"""
webapp/lib/theme.py

와이어프레임(이미지 5~12)의 시각 언어를 그대로 따른다:
- 크림색 사이드바, 흰 본문, 딥마룬(brick) 데이터 강조, 카드형 레이아웃.
Streamlit 기본 위젯과 싸우지 않고, 카드/막대는 직접 HTML+CSS로 그려 픽셀을 통제한다.
"""
from __future__ import annotations

import streamlit as st

# --- 팔레트 (단일 출처) ---------------------------------------------------
INK = "#1f2740"          # 본문 텍스트 (네이비)
MUTED = "#667085"        # 보조 텍스트
FAINT = "#98a2b3"        # 흐린 텍스트
MAROON = "#3b5bdb"       # 주 강조색(블루) — 이름은 유지, 값만 블루로 전 컴포넌트 전파
MAROON_DEEP = "#2f49b8"
CORAL = "#f0655f"        # 이탈(위험) 강조
TRACK = "#eaeef6"        # 막대 배경 트랙
CARD_BG = "#ffffff"
CARD_BORDER = "#e6e9f2"
CREAM = "#1e2538"        # 사이드바 (다크 네이비)
SIDEBAR_TX = "#c7cfe2"   # 사이드바 본문 텍스트(라이트)
SIDEBAR_MUT = "#8b95ad"  # 사이드바 보조 텍스트
ACTIVE = "#2b4cc7"       # 활성 nav 배경(블루)
ACTIVE_INK = "#ffffff"   # 활성 nav 텍스트
GOOD = "#2e9e6b"

CSS = f"""
<style>
@import url('https://cdn.jsdelivr.net/gh/orioncactus/pretendard@v1.3.9/dist/web/static/pretendard.css');

html, body, [class*="css"], .stApp {{
    font-family: 'Pretendard', 'Noto Sans KR', -apple-system, BlinkMacSystemFont, sans-serif;
    color: {INK};
}}
.stApp {{ background: #f5f7fb; }}

/* 상단 기본 헤더/메뉴 정리 */
header[data-testid="stHeader"] {{ background: transparent; }}
#MainMenu, footer {{ visibility: hidden; }}

/* 본문 폭/여백 */
.block-container {{ padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1180px; }}

/* ---------------- 사이드바 (다크 네이비) ---------------- */
[data-testid="stSidebar"] {{ background: {CREAM}; border-right: 1px solid #2a3350; }}
[data-testid="stSidebar"] .block-container {{ padding-top: 1.4rem; }}
[data-testid="stSidebar"] * {{ color: {SIDEBAR_TX}; }}

.sb-brand {{ font-size: 1.2rem; font-weight: 800; letter-spacing:-.02em; color:#ffffff;
            padding: .1rem .2rem .15rem; }}
.sb-brand .sb-sub {{ display:block; font-size:.72rem; font-weight:500; letter-spacing:0;
                    color:{SIDEBAR_MUT}; margin-top:.1rem; }}
.sb-model {{ background:rgba(255,255,255,.05); border:1px solid rgba(255,255,255,.10);
            border-radius:12px; padding:.8rem .9rem; margin-top:1rem; }}
.sb-model .m-h {{ font-size:.8rem; font-weight:800; color:#ffffff; margin-bottom:.5rem; }}
.sb-model .m-row {{ font-size:.78rem; color:{SIDEBAR_MUT}; line-height:1.7; }}
.sb-model .m-row b {{ color:#dbe2f2; font-weight:600; }}
.sb-foot {{ margin-top: 1.2rem; padding-top: 1rem; border-top:1px solid rgba(255,255,255,.12);
            color:{SIDEBAR_MUT}; font-size:.74rem; line-height:1.7; }}
.sb-foot b {{ color:#cdd6ea; font-weight:600; }}

/* st.navigation 링크 — 다크 사이드바용 */
[data-testid="stSidebar"] a[data-testid="stPageLink-NavLink"] {{
    border-radius: 9px; padding: .5rem .65rem; margin: 1px 0; font-size:.92rem;
}}
[data-testid="stSidebar"] a[data-testid="stPageLink-NavLink"] p {{ color: {SIDEBAR_TX}; }}
[data-testid="stSidebar"] a[data-testid="stPageLink-NavLink"]:hover {{ background: rgba(255,255,255,.07); }}
[data-testid="stSidebar"] a[data-testid="stPageLink-NavLink"][aria-current="page"] {{ background: {ACTIVE}; }}
[data-testid="stSidebar"] a[data-testid="stPageLink-NavLink"][aria-current="page"] p {{ color: #ffffff; }}

/* ---------------- 페이지 헤더 ---------------- */
.page-title {{ font-size: 2rem; font-weight: 800; letter-spacing:-.02em; margin:0 0 .15rem; }}
.page-sub {{ color:{MUTED}; font-size:.96rem; margin:0 0 1.3rem; }}

/* ---------------- 카드 ---------------- */
.card {{ background:{CARD_BG}; border:1px solid {CARD_BORDER}; border-radius:16px;
         padding:1.15rem 1.3rem; margin-bottom:1.05rem;
         box-shadow:0 1px 3px rgba(20,30,60,.05), 0 1px 2px rgba(20,30,60,.03); }}
.card h3 {{ font-size:1.12rem; font-weight:700; margin:.1rem 0 .15rem; }}
.card .csub {{ color:{MUTED}; font-size:.84rem; margin:0 0 1rem; line-height:1.5; }}

/* KPI */
.kpi-wrap {{ display:flex; gap:.8rem; flex-wrap:wrap; margin-bottom:1.05rem; }}
.kpi {{ position:relative; flex:1 1 0; min-width:150px; background:{CARD_BG};
        border:1px solid {CARD_BORDER}; border-radius:14px; padding:.95rem 1.05rem;
        box-shadow:0 1px 3px rgba(20,30,60,.05), 0 1px 2px rgba(20,30,60,.03); }}
.kpi .k-label {{ color:{MUTED}; font-size:.82rem; margin-bottom:.35rem; }}
.kpi .k-val {{ font-size:1.7rem; font-weight:800; letter-spacing:-.01em; }}
.kpi .k-val small {{ font-size:.95rem; font-weight:700; color:{MAROON}; margin-left:.25rem; }}
.kpi.accent .k-val {{ color:{MAROON}; }}
.kpi .k-ico {{ position:absolute; top:.85rem; right:1rem; font-size:1.25rem; opacity:.9; }}

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
.chip {{ display:inline-block; background:#eef1f8; color:{MUTED}; border-radius:999px;
         padding:.2rem .6rem; font-size:.78rem; margin:.15rem .25rem .15rem 0; }}
.tag {{ display:inline-block; background:#eef1f8; color:{MUTED}; border-radius:7px;
        padding:.16rem .5rem; font-size:.78rem; margin-right:.35rem; }}
.tag.sig {{ color:{MAROON_DEEP}; background:#e7eeff; font-weight:600; }}

/* 우선대응 고객 리스트 행 */
.cust {{ border-top:1px solid #eef0f6; padding:.85rem .2rem; }}
.cust:first-child {{ border-top:none; }}
.cust-head {{ display:flex; align-items:center; gap:.55rem; }}
.cust-rank {{ width:24px; height:24px; border-radius:50%; background:#eef1f8; color:{MUTED};
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
.callout {{ background:#f3f6fc; border:1px solid #dde6f7; border-radius:14px;
            padding:1rem 1.15rem; color:{INK}; font-size:.92rem; line-height:1.65; }}
.callout b {{ color:{MAROON}; }}
hr.soft {{ border:none; border-top:1px solid #eef0f6; margin:.9rem 0; }}

/* 이탈/유지 분할 바 (홈) */
.splitbar {{ display:flex; height:30px; border-radius:9px; overflow:hidden; margin:.15rem 0 .2rem; }}
.splitbar .seg {{ display:flex; align-items:center; justify-content:center; font-size:.82rem;
                 font-weight:700; white-space:nowrap; }}
.splitbar .seg.churn {{ background:{CORAL}; color:#fff; }}
.splitbar .seg.keep {{ background:{TRACK}; color:{INK}; }}

/* 인사이트 / 활용 안내 박스 */
.info-box {{ background:#eef3ff; border:1px solid #d6e2ff; border-radius:14px; padding:.95rem 1.1rem;
            margin-bottom:1.05rem; }}
.info-box .ib-h {{ font-weight:800; color:{MAROON}; margin-bottom:.45rem; font-size:.95rem; }}
.good-box {{ background:#eaf7f0; border:1px solid #cdeadb; border-radius:14px; padding:.95rem 1.1rem;
            margin-bottom:1.05rem; }}
.good-box .ib-h {{ font-weight:800; color:{GOOD}; margin-bottom:.45rem; font-size:.95rem; }}
.info-box .ib-body, .good-box .ib-body {{ font-size:.84rem; line-height:1.7; color:#51607a; }}

/* 차트 카드 제목 */
.ch-t {{ font-weight:700; font-size:.98rem; margin:.1rem 0 .45rem; }}
.ch-t .ch-sub {{ font-weight:400; font-size:.78rem; color:{FAINT}; }}

/* 데이터 업로드 플레이스홀더 (추후 개발) */
.uploadbox {{ border:1.5px dashed #c2cbe0; border-radius:14px; padding:1.6rem 1.2rem;
             text-align:center; background:#f7f9fd; }}
.uploadbox .u-title {{ font-weight:700; color:{MUTED}; font-size:.98rem; margin-bottom:.3rem; }}
.uploadbox .u-sub {{ font-size:.82rem; color:{FAINT}; line-height:1.6; }}
.uploadbox .u-pill {{ display:inline-block; margin-top:.7rem; font-size:.74rem; color:{MUTED};
                     background:#eef1f8; border-radius:999px; padding:.2rem .75rem; }}

/* 분석B 속성 그룹 패널 — 속성별로 묶어 시각적으로 구분 */
.attr-group {{ border:1px solid {CARD_BORDER}; border-radius:12px; background:#f7f9fd;
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
.entry.muted {{ background:#f7f9fd; }}
.entry-link:hover .entry {{ border-color:{MAROON}; background:#f0f4fd; }}
.entry .e-ico {{ font-size:1.25rem; line-height:1; }}
.entry .e-t {{ font-weight:800; font-size:1.02rem; margin:.4rem 0 .18rem; }}
.entry.muted .e-t {{ color:{MUTED}; }}
.entry .e-d {{ color:{MUTED}; font-size:.82rem; line-height:1.5; }}
.entry .e-go {{ position:absolute; top:.7rem; right:.85rem; color:{FAINT}; font-size:.95rem; }}
.entry-link:hover .e-go {{ color:{MAROON}; }}

/* 아키텍처 선언 + 이원화 레이어(거시/미시) 상태 */
.arch {{ background:#f3f6fc; border:1px solid {CARD_BORDER}; border-radius:16px;
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
.layer .l-tag {{ font-size:.72rem; color:{MUTED}; background:#eef1f8; border-radius:6px; padding:.1rem .42rem; }}
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
table.mini td {{ padding:.38rem .5rem; border-bottom:1px solid #eef0f6; }}
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


def kpi(label: str, value: str, accent: bool = False, suffix: str = "",
        icon: str = "") -> str:
    cls = "kpi accent" if accent else "kpi"
    suf = f"<small>{suffix}</small>" if suffix else ""
    ico = f'<span class="k-ico">{icon}</span>' if icon else ""
    return (f'<div class="{cls}">{ico}<div class="k-label">{label}</div>'
            f'<div class="k-val">{value}{suf}</div></div>')


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
