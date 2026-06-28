CUSTOM_CSS = """
<style>
/* ── Google Font ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* ── Base ── */
*, html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
}
.stApp {
    background: #0E1117;
}

/* ── Hide default Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 1.5rem 2rem 3rem; max-width: 1200px; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #0D1117;
    border-right: 1px solid #21262D;
}
[data-testid="stSidebar"] * { color: #C9D1D9 !important; }
[data-testid="stSidebarNav"] { display: none; }

/* ── Sidebar logo title ── */
.sb-brand {
    font-size: 18px;
    font-weight: 700;
    color: #fff !important;
    letter-spacing: -0.3px;
    padding: 6px 0 18px 0;
    display: flex;
    align-items: center;
    gap: 8px;
}
.sb-divider {
    border: none;
    border-top: 1px solid #21262D;
    margin: 10px 0;
}
.sb-nav-item {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 9px 12px;
    border-radius: 8px;
    margin-bottom: 3px;
    cursor: pointer;
    font-size: 14px;
    color: #8B949E;
    text-decoration: none;
    transition: background 0.2s, color 0.2s;
}
.sb-nav-item:hover, .sb-nav-item.active {
    background: #161B22;
    color: #fff !important;
}
.sb-status-dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    background: #10B981;
    display: inline-block;
    margin-right: 6px;
    box-shadow: 0 0 6px #10B981;
}
.sb-status-label { font-size: 13px; color: #8B949E; margin: 4px 0; }
.sb-version { font-size: 11px; color: #484F58; margin-top: 8px; }

/* ── Cards ── */
.card {
    background: #161B22;
    border: 1px solid #21262D;
    border-radius: 14px;
    padding: 24px;
    margin-bottom: 16px;
    transition: border-color 0.25s, transform 0.2s, box-shadow 0.2s;
}
.card:hover {
    border-color: #3B82F6;
    transform: translateY(-2px);
    box-shadow: 0 8px 30px rgba(59,130,246,0.12);
}
.card-glass {
    background: rgba(22,27,34,0.7);
    backdrop-filter: blur(16px);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 16px;
    padding: 24px;
}

/* ── Hero ── */
.hero-section {
    background: linear-gradient(135deg, #0D1117 0%, #161B22 50%, #0D1117 100%);
    border: 1px solid #21262D;
    border-radius: 20px;
    padding: 56px 48px;
    text-align: center;
    position: relative;
    overflow: hidden;
    margin-bottom: 32px;
}
.hero-section::before {
    content: '';
    position: absolute;
    top: -60px; left: 50%;
    transform: translateX(-50%);
    width: 500px; height: 300px;
    background: radial-gradient(ellipse, rgba(59,130,246,0.18) 0%, transparent 70%);
    pointer-events: none;
}
.hero-title {
    font-size: 42px;
    font-weight: 700;
    color: #fff;
    letter-spacing: -1px;
    margin: 0 0 14px;
    line-height: 1.1;
}
.hero-title span {
    background: linear-gradient(135deg, #3B82F6, #8B5CF6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.hero-subtitle {
    font-size: 17px;
    color: #8B949E;
    margin: 0 0 32px;
    line-height: 1.6;
}

/* ── Metric cards ── */
.metric-card {
    background: #161B22;
    border: 1px solid #21262D;
    border-radius: 14px;
    padding: 20px 22px;
    text-align: center;
    transition: all 0.25s;
}
.metric-card:hover {
    border-color: #3B82F6;
    box-shadow: 0 4px 20px rgba(59,130,246,0.1);
}
.metric-icon { font-size: 28px; margin-bottom: 10px; }
.metric-value {
    font-size: 28px;
    font-weight: 700;
    color: #fff;
    letter-spacing: -0.5px;
}
.metric-label { font-size: 13px; color: #8B949E; margin-top: 4px; }

/* ── Feature cards ── */
.feature-card {
    background: #161B22;
    border: 1px solid #21262D;
    border-radius: 14px;
    padding: 28px 24px;
    height: 100%;
    transition: all 0.3s;
    position: relative;
    overflow: hidden;
}
.feature-card::after {
    content: '';
    position: absolute;
    bottom: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, #3B82F6, #8B5CF6);
    transform: scaleX(0);
    transition: transform 0.3s;
}
.feature-card:hover::after { transform: scaleX(1); }
.feature-card:hover {
    border-color: #30363D;
    transform: translateY(-3px);
    box-shadow: 0 12px 40px rgba(0,0,0,0.4);
}
.feature-icon { font-size: 34px; margin-bottom: 14px; }
.feature-title {
    font-size: 16px;
    font-weight: 600;
    color: #E6EDF3;
    margin-bottom: 8px;
}
.feature-desc { font-size: 13.5px; color: #8B949E; line-height: 1.6; }

/* ── Section headings ── */
.section-title {
    font-size: 22px;
    font-weight: 700;
    color: #ffffff !important;
    margin: 8px 0 20px;
    letter-spacing: -0.4px;
    text-shadow: 0 1px 2px rgba(0, 0, 0, 0.35);
}
.section-subtitle {
    font-size: 14px;
    color: #d1d5db !important;
    margin-bottom: 24px;
}

/* ── Form inputs ── */
.stTextInput input, .stNumberInput input,
.stSelectbox select, .stSlider {
    background: #0D1117 !important;
    border: 1px solid #21262D !important;
    border-radius: 9px !important;
    color: #E6EDF3 !important;
}
.stTextInput input:focus, .stNumberInput input:focus {
    border-color: #3B82F6 !important;
    box-shadow: 0 0 0 3px rgba(59,130,246,0.15) !important;
}
label { color: #C9D1D9 !important; font-size: 13.5px !important; }

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #3B82F6, #2563EB) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.65rem 1.6rem !important;
    font-size: 15px !important;
    font-weight: 600 !important;
    letter-spacing: 0.2px !important;
    transition: all 0.2s !important;
    box-shadow: 0 4px 15px rgba(59,130,246,0.3) !important;
    width: 100%;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(59,130,246,0.45) !important;
    background: linear-gradient(135deg, #2563EB, #1D4ED8) !important;
}
.btn-secondary > button {
    background: #161B22 !important;
    border: 1px solid #30363D !important;
    color: #C9D1D9 !important;
    box-shadow: none !important;
}
.btn-secondary > button:hover {
    border-color: #3B82F6 !important;
    color: #fff !important;
    box-shadow: none !important;
}
.btn-success > button {
    background: linear-gradient(135deg, #10B981, #059669) !important;
    box-shadow: 0 4px 15px rgba(16,185,129,0.3) !important;
}

/* ── Result cards ── */
.result-anemic {
    background: linear-gradient(135deg, rgba(239,68,68,0.12), rgba(239,68,68,0.04));
    border: 1px solid rgba(239,68,68,0.35);
    border-radius: 16px;
    padding: 28px;
}
.result-normal {
    background: linear-gradient(135deg, rgba(16,185,129,0.12), rgba(16,185,129,0.04));
    border: 1px solid rgba(16,185,129,0.35);
    border-radius: 16px;
    padding: 28px;
}
.result-title {
    font-size: 26px;
    font-weight: 700;
    letter-spacing: -0.5px;
    margin-bottom: 6px;
}
.result-confidence {
    font-size: 14px;
    color: #8B949E;
    margin-bottom: 20px;
}

/* ── Badge ── */
.badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 0.3px;
}
.badge-danger  { background: rgba(239,68,68,0.18);  color: #EF4444; }
.badge-success { background: rgba(16,185,129,0.18); color: #10B981; }
.badge-warning { background: rgba(245,158,11,0.18); color: #F59E0B; }
.badge-info    { background: rgba(59,130,246,0.18); color: #3B82F6; }

/* ── Progress bar ── */
.prob-bar-wrap { margin: 10px 0; }
.prob-bar-label {
    display: flex;
    justify-content: space-between;
    font-size: 13px;
    color: #8B949E;
    margin-bottom: 5px;
}
.prob-bar-bg {
    background: #21262D;
    border-radius: 8px;
    height: 10px;
    overflow: hidden;
}
.prob-bar-fill {
    height: 100%;
    border-radius: 8px;
    transition: width 0.6s ease;
}

/* ── Upload zone ── */
.upload-zone {
    border: 2px dashed #30363D;
    border-radius: 16px;
    padding: 40px 24px;
    text-align: center;
    transition: border-color 0.2s, background 0.2s;
}
.upload-zone:hover {
    border-color: #3B82F6;
    background: rgba(59,130,246,0.04);
}

/* ── Tooltips / info boxes ── */
.info-box {
    background: rgba(59,130,246,0.08);
    border-left: 3px solid #3B82F6;
    border-radius: 0 8px 8px 0;
    padding: 12px 16px;
    font-size: 13px;
    color: #8B949E;
    margin: 12px 0;
}

/* ── Tech stack pills ── */
.tech-pill {
    display: inline-block;
    background: #161B22;
    border: 1px solid #21262D;
    border-radius: 20px;
    padding: 6px 14px;
    font-size: 13px;
    color: #8B949E;
    margin: 4px 3px;
    transition: all 0.2s;
}
.tech-pill:hover {
    border-color: #3B82F6;
    color: #3B82F6;
}

/* ── Footer ── */
.footer {
    background: #0D1117;
    border-top: 1px solid #21262D;
    padding: 28px 0 16px;
    text-align: center;
    margin-top: 60px;
}
.footer-title { font-size: 15px; font-weight: 600; color: #E6EDF3; margin-bottom: 10px; }
.footer-sub   { font-size: 12px; color: #484F58; margin-top: 12px; }

/* ── Divider ── */
.gradient-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, #3B82F6, transparent);
    border: none;
    margin: 28px 0;
}

/* ── Selectbox / radio ── */
div[data-baseweb="select"] > div {
    background: #0D1117 !important;
    border-color: #21262D !important;
    border-radius: 9px !important;
}

/* ── Spinner ── */
.stSpinner > div { border-top-color: #3B82F6 !important; }

/* ── Image preview ── */
.img-preview {
    border-radius: 14px;
    border: 1px solid #21262D;
    overflow: hidden;
}
</style>
"""


def inject_css():
    import streamlit as st
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
