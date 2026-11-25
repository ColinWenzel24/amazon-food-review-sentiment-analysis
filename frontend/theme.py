"""Amazon-inspired look and feel: dark navy navigation, yellow pill buttons, orange accents,
white cards with light gray borders, teal links."""

import streamlit as st

# Palette (Amazon storefront-inspired)
NAVY = "#131921"        # top navigation bar
NAVY_LIGHT = "#232F3E"  # secondary nav bar
ORANGE = "#FF9900"      # accent
STAR = "#FFA41C"        # rating stars / histogram bars
YELLOW = "#FFD814"      # primary button
YELLOW_BORDER = "#FCD200"
TEAL = "#007185"        # links
INK = "#0F1111"         # body text
MUTED = "#565959"       # secondary text
BORDER = "#D5D9D9"
TRACK = "#F0F2F2"

SENTIMENT_STYLE = {
    "Positive": {"color": "#067D62", "bg": "#E6F4F1", "icon": "▲", "emoji": "😊"},
    "Neutral": {"color": "#565959", "bg": "#F0F2F2", "icon": "●", "emoji": "😐"},
    "Negative": {"color": "#B12704", "bg": "#FCEDEA", "icon": "▼", "emoji": "😞"},
}

_CSS = f"""
<style>
html, body, [class*="css"], .stMarkdown, .stTextArea textarea, .stTextInput input {{
    font-family: "Amazon Ember", Arial, "Helvetica Neue", sans-serif;
}}
/* Hide Streamlit chrome so our nav bar sits at the top */
header[data-testid="stHeader"], [data-testid="stToolbar"], [data-testid="stDecoration"] {{ display: none; }}
.block-container {{ padding-top: 0 !important; max-width: 1240px; }}

/* Sticky footer: the page is at least one screen tall and the footer is pushed to the bottom.
   Streamlit's default 160px bottom padding is removed so nothing shows below the footer. */
[data-testid="stMainBlockContainer"] {{
    padding-bottom: 0 !important; min-height: 100vh; display: flex; flex-direction: column;
}}
[data-testid="stMainBlockContainer"] > [data-testid="stVerticalBlock"] {{ flex: 1 0 auto; }}
[data-testid="stElementContainer"]:has(.footer) {{ margin-top: auto; }}
[data-testid="stElementContainer"]:has(.footer) [data-testid="stMarkdownContainer"] {{ margin-bottom: 0; }}

/* ---------- Top navigation bar ---------- */
.bleed {{ width: 100vw; margin-left: calc(50% - 50vw); padding-left: max(24px, calc(50vw - 600px));
          padding-right: max(24px, calc(50vw - 600px)); box-sizing: border-box; }}
.nav {{
    background: {NAVY}; color: #fff; padding-top: 12px; padding-bottom: 12px; margin-top: -1rem;
    display: flex; align-items: center; justify-content: space-between; gap: 16px;
}}
.nav .brand {{ font-size: 1.55rem; font-weight: 700; letter-spacing: -0.5px; line-height: 1; }}
.nav .brand .accent {{ color: {ORANGE}; }}
.nav .brand .underline {{
    display: block; height: 3px; width: 70%; margin-top: 4px; border-radius: 0 0 50% 50%;
    background: {ORANGE};
}}
.nav .tagline {{ color: #CCCCCC; font-size: 0.8rem; }}
.nav .pill {{
    border: 1px solid #FFFFFF33; border-radius: 4px; padding: 6px 10px; font-size: 0.78rem; color: #fff;
}}
.nav .pill b {{ display: block; font-size: 0.86rem; }}
.subnav {{
    background: {NAVY_LIGHT}; color: #fff; margin-bottom: 18px; padding-top: 7px; padding-bottom: 7px;
    font-size: 0.85rem; display: flex; gap: 22px; flex-wrap: wrap;
}}
.subnav span {{ opacity: 0.92; }}
.subnav .dot {{ color: {ORANGE}; }}

/* ---------- Tabs styled like Amazon category links ---------- */
.stTabs [data-baseweb="tab-list"] {{
    gap: 4px; border-bottom: 1px solid {BORDER};
}}
.stTabs [data-baseweb="tab"] {{
    padding: 8px 14px; font-weight: 600; color: {INK}; border-radius: 4px 4px 0 0;
}}
.stTabs [data-baseweb="tab"]:hover {{ color: {TEAL}; background: #F7FAFA; }}
.stTabs [aria-selected="true"] {{ color: #C45500 !important; }}
.stTabs [data-baseweb="tab-highlight"] {{ background-color: {ORANGE}; height: 3px; }}

/* ---------- Buttons ---------- */
button[kind="primary"], [data-testid="stBaseButton-primary"] {{
    background: {YELLOW} !important; color: {INK} !important; border: 1px solid {YELLOW_BORDER} !important;
    border-radius: 100px !important; font-weight: 500; box-shadow: 0 2px 5px rgba(213,217,217,.5);
}}
button[kind="primary"]:hover, [data-testid="stBaseButton-primary"]:hover {{
    background: #F7CA00 !important; border-color: #F2C200 !important;
}}
button[kind="secondary"], [data-testid="stBaseButton-secondary"] {{
    border-radius: 8px !important; border: 1px solid {BORDER} !important; background: #fff !important;
    color: {INK} !important; box-shadow: 0 2px 5px rgba(213,217,217,.5);
}}
button[kind="secondary"]:hover, [data-testid="stBaseButton-secondary"]:hover {{
    background: #F7FAFA !important;
}}
[data-testid="stDownloadButton"] button {{ border-radius: 100px !important; }}

/* ---------- Inputs ---------- */
.stTextArea textarea, .stTextInput input {{ border-radius: 8px !important; background: #fff !important; }}
.stTextArea textarea:focus, .stTextInput input:focus {{
    box-shadow: 0 0 0 3px #C8F3FA, 0 1px 2px rgba(15,17,17,.15) inset !important; border-color: {TEAL} !important;
}}

/* ---------- Cards ---------- */
.card {{
    background: #fff; border: 1px solid {BORDER}; border-radius: 8px; padding: 18px 20px; height: 100%;
}}
.card h4 {{ margin: 0 0 2px; font-size: 1.05rem; color: {INK}; }}
.card .meta {{ color: {MUTED}; font-size: 0.8rem; margin-bottom: 12px; }}
.section-title {{ font-size: 1.3rem; font-weight: 700; color: {INK}; margin: 6px 0 2px; }}
.section-sub {{ color: {MUTED}; font-size: 0.9rem; margin-bottom: 14px; }}
.kicker {{ color: #C45500; font-size: 0.78rem; font-weight: 700; text-transform: uppercase; letter-spacing: .4px; }}

/* Sentiment badge */
.badge {{
    display: inline-flex; align-items: center; gap: 6px; padding: 4px 12px; border-radius: 100px;
    font-weight: 700; font-size: 1.05rem;
}}
.conf {{ color: {MUTED}; font-size: 0.85rem; margin-left: 8px; }}
.stars {{ color: {STAR}; letter-spacing: 1px; }}

/* Probability rows styled like the Amazon customer-review histogram */
.hist-row {{ display: grid; grid-template-columns: 78px 1fr 52px; align-items: center; gap: 10px; margin: 7px 0; }}
.hist-row .lbl {{ color: {TEAL}; font-size: 0.88rem; }}
.hist-row.top .lbl {{ font-weight: 700; color: {INK}; }}
.hist-row .track {{
    height: 20px; background: {TRACK}; border: 1px solid #E3E6E6; border-radius: 4px; overflow: hidden;
    box-shadow: inset 0 1px 2px rgba(15,17,17,.12);
}}
.hist-row .fill {{ height: 100%; background: {STAR}; border-right: 1px solid #DE7921; }}
.hist-row .pct {{ text-align: right; font-size: 0.88rem; color: {INK}; }}

/* Word-contribution chips */
.chips {{ display: flex; flex-wrap: wrap; gap: 6px; margin-top: 4px; }}
.chip {{ border-radius: 100px; padding: 2px 10px; font-size: 0.82rem; border: 1px solid; }}
.chip.for {{ background: #E6F4F1; border-color: #9ED3C7; color: #045C48; }}
.chip.against {{ background: #FCEDEA; border-color: #F3B8AA; color: #8C1F03; }}

/* Agreement banner (Amazon alert box style) */
.alert {{ border-radius: 8px; padding: 12px 16px; border: 1px solid; margin: 6px 0 16px;
          display: flex; gap: 12px; align-items: flex-start; }}
.alert .icon {{ font-size: 1.2rem; line-height: 1.2; }}
.alert.ok {{ border-color: #067D62; background: #F2FAF8; box-shadow: 0 0 0 4px #E6F4F1 inset; }}
.alert.warn {{ border-color: #C45500; background: #FFF8EE; box-shadow: 0 0 0 4px #FEF3E2 inset; }}
.alert.info {{ border-color: {TEAL}; background: #F5FAFB; box-shadow: 0 0 0 4px #E7F4F5 inset; }}

/* Stat tiles */
.stat {{ background: #fff; border: 1px solid {BORDER}; border-radius: 8px; padding: 14px 16px; }}
.stat .v {{ font-size: 1.6rem; font-weight: 700; color: {INK}; line-height: 1.2; }}
.stat .k {{ color: {MUTED}; font-size: 0.82rem; }}

/* Tables */
[data-testid="stDataFrame"] {{ border: 1px solid {BORDER}; border-radius: 8px; }}

.footer {{
    background: {NAVY_LIGHT}; color: #DDD; margin-top: 40px; padding-top: 18px; padding-bottom: 18px;
    font-size: 0.8rem;
    text-align: center;
}}
.footer a {{ color: #fff; }}
a {{ color: {TEAL}; }}
a:hover {{ color: #C7511F; }}
</style>
"""


def apply_theme() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)
