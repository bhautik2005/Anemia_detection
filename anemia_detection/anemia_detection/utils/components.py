import streamlit as st
import streamlit.components.v1 as components


# ── Sidebar ────────────────────────────────────────────────
def render_sidebar(active_page="Home"):
    with st.sidebar:
        st.markdown("""
        <div class="sb-brand">
            🩸 AI Anemia Detection
        </div>
        """, unsafe_allow_html=True)

        pages = [
            ("🏠", "Home", "app.py"),
            ("🩺", "Clinical Prediction", "pages/1_Clinical_Prediction.py"),
            ("👁️", "Eye Image Prediction", "pages/2_Eye_Image_Prediction.py"),
            ("ℹ️", "About", "pages/3_About.py"),
        ]

        for icon, label, page_path in pages:
            if st.button(
                f"{icon} &nbsp; {label}",
                key=f"nav_{label}",
                use_container_width=True,
                type="secondary",
            ):
                st.switch_page(page_path)

        st.markdown('<hr class="sb-divider"/>', unsafe_allow_html=True)

        st.markdown("**Model Status**", unsafe_allow_html=False)
        st.markdown("""
        <div class="sb-status-label">
            <span class="sb-status-dot"></span> Clinical Model &nbsp;✅
        </div>
        <div class="sb-status-label">
            <span class="sb-status-dot"></span> CNN Eye Model &nbsp;✅
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<hr class="sb-divider"/>', unsafe_allow_html=True)
        st.markdown('<div class="sb-version">v1.0 · Made with Streamlit</div>',
                    unsafe_allow_html=True)


# ── Hero Section ────────────────────────────────────────────
def render_hero(title: str, highlight: str, subtitle: str, emoji: str = "🩸"):
    st.markdown(f"""
    <div class="hero-section">
        <div style="font-size:52px;margin-bottom:14px">{emoji}</div>
        <div class="hero-title">{title} <span>{highlight}</span></div>
        <p class="hero-subtitle">{subtitle}</p>
    </div>
    """, unsafe_allow_html=True)


# ── Metric card row ─────────────────────────────────────────
def render_metrics(metrics: list):
    """
    metrics: list of (icon, value, label) tuples
    """
    cols = st.columns(len(metrics))
    for col, (icon, value, label) in zip(cols, metrics):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-icon">{icon}</div>
                <div class="metric-value">{value}</div>
                <div class="metric-label">{label}</div>
            </div>
            """, unsafe_allow_html=True)


# ── Feature card grid ───────────────────────────────────────
def render_feature_cards(features: list):
    """
    features: list of (icon, title, desc) tuples
    """
    n = len(features)
    cols = st.columns(n if n <= 3 else 3)
    for i, (icon, title, desc) in enumerate(features):
        with cols[i % 3]:
            st.markdown(f"""
            <div class="feature-card">
                <div class="feature-icon">{icon}</div>
                <div class="feature-title">{title}</div>
                <p class="feature-desc">{desc}</p>
            </div>
            """, unsafe_allow_html=True)


# ── Section heading ─────────────────────────────────────────
def section_heading(title: str, subtitle: str = ""):
    st.markdown(
        f'<div class="section-title" style="color:#ffffff !important;">{title}</div>',
        unsafe_allow_html=True,
    )
    if subtitle:
        st.markdown(
            f'<div class="section-subtitle" style="color:#d1d5db !important;">{subtitle}</div>',
            unsafe_allow_html=True,
        )


# ── Result card ─────────────────────────────────────────────
def render_result_card(prediction: str, confidence: float,
                       prob_anemic: float, prob_normal: float,
                       recommendation: str):
    is_anemic  = prediction.lower() in ("anemic", "anemia detected")
    card_cls   = "result-anemic" if is_anemic else "result-normal"
    icon       = "🔴" if is_anemic else "🟢"
    color      = "#EF4444" if is_anemic else "#10B981"
    badge_cls  = "badge-danger" if is_anemic else "badge-success"
    label      = "Anemia Detected" if is_anemic else "No Anemia"
    conf_pct   = f"{confidence * 100:.1f}%"

    bar_anemic = f"{prob_anemic * 100:.1f}"
    bar_normal = f"{prob_normal * 100:.1f}"

    html = f"""
    <div class="{card_cls}">
        <div style="display:flex;align-items:center;gap:14px;margin-bottom:10px">
            <span style="font-size:40px">{icon}</span>
            <div>
                <div class="result-title" 
                style="color:{color}">{label}</div>
                <div class="result-confidence" style="color:#C9D1D9">Confidence: <b>{conf_pct}</b></div>
            </div>
            <div style="margin-left:auto">
                <span class="badge {badge_cls}"style="color:#C9D1D9">{label}</span>
            </div>
        </div>

        <div class="prob-bar-wrap">
            <div class="prob-bar-label">
                <span style="color:#C9D1D9">🔴 Anemic</span><span style="color:#C9D1D9">{bar_anemic}%</span>
            </div>
            <div class="prob-bar-bg">
                <div class="prob-bar-fill"
                     style="width:{bar_anemic}%;background:{'#EF4444' if is_anemic else '#374151'}">
                </div>
            </div>
        </div>
        <div class="prob-bar-wrap">
            <div class="prob-bar-label">
                <span style="color:#C9D1D9">🟢 Non-Anemic</span><span style="color:#C9D1D9">{bar_normal}%</span>
            </div>
            <div class="prob-bar-bg">
                <div class="prob-bar-fill"
                     style="width:{bar_normal}%;background:{'#10B981' if not is_anemic else '#374151'}">
                </div>
            </div>
        </div>

        <hr style="border-color:#21262D;margin:18px 0"/>
        <div style="font-size:13.5px;color:#8B949E">
            <b style="color:#C9D1D9">📋 Recommendation:</b><br/>
            {recommendation}
        </div>
    </div>
    """
    components.html(html, height=260)


# ── Info box ────────────────────────────────────────────────
def info_box(text: str):
    st.markdown(f'<div class="info-box">ℹ️ &nbsp;{text}</div>',
                unsafe_allow_html=True)


# ── Gradient divider ────────────────────────────────────────
def gradient_divider():
    st.markdown('<hr class="gradient-divider"/>', unsafe_allow_html=True)


# ── Footer ──────────────────────────────────────────────────
def render_footer():
    st.markdown("""
    <div class="footer">
        <div class="footer-title">🩸 AI Anemia Detection Platform</div>
        <div style="margin:8px 0">
            <span class="tech-pill">🐍 Python</span>
            <span class="tech-pill">⚡ Streamlit</span>
            <span class="tech-pill">🧠 TensorFlow</span>
            <span class="tech-pill">📊 Scikit-Learn</span>
            <span class="tech-pill">🔬 Deep Learning</span>
            <span class="tech-pill">👁️ OpenCV</span>
        </div>
        <div class="footer-sub">© 2024 · Built for AnemiaFusionNet Project · v1.0</div>
    </div>
    """, unsafe_allow_html=True)


# ── Toast-style notification ─────────────────────────────────
def toast_success(msg: str):
    st.markdown(f"""
    <div style="background:rgba(16,185,129,0.1);border:1px solid rgba(16,185,129,0.3);
         border-radius:10px;padding:12px 18px;font-size:14px;color:#10B981;margin:10px 0">
        ✅ &nbsp; {msg}
    </div>
    """, unsafe_allow_html=True)


def toast_error(msg: str):
    st.markdown(f"""
    <div style="background:rgba(239,68,68,0.1);border:1px solid rgba(239,68,68,0.3);
         border-radius:10px;padding:12px 18px;font-size:14px;color:#EF4444;margin:10px 0">
        ❌ &nbsp; {msg}
    </div>
    """, unsafe_allow_html=True)


def toast_warning(msg: str):
    st.markdown(f"""
    <div style="background:rgba(245,158,11,0.1);border:1px solid rgba(245,158,11,0.3);
         border-radius:10px;padding:12px 18px;font-size:14px;color:#F59E0B;margin:10px 0">
        ⚠️ &nbsp; {msg}
    </div>
    """, unsafe_allow_html=True)


# ── Input form card wrapper ──────────────────────────────────
def card_start(title: str = "", icon: str = ""):
    head = f"<b>{icon} {title}</b>" if title else ""
    st.markdown(f"""
    <div class="card">
        {"<div style='font-size:15px;color:#C9D1D9;margin-bottom:18px'>"+head+"</div>" if head else ""}
    """, unsafe_allow_html=True)


def card_end():
    st.markdown("</div>", unsafe_allow_html=True)
