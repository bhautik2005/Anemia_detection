import streamlit as st
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from utils.styles     import inject_css
from utils.components import (
    render_sidebar, render_hero, render_metrics,
    render_feature_cards, gradient_divider, render_footer,
    section_heading
)

# ── Page config ────────────────────────────────────────────
st.set_page_config(
    page_title="AI Anemia Detection",
    page_icon="🩸",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_css()
render_sidebar(active_page="Home")

# ── Hero ───────────────────────────────────────────────────
render_hero(
    title="AI Anemia Detection",
    highlight="Platform",
    subtitle=(
        "Detect anemia using Clinical Parameters OR Eye Image Analysis · "
        "Powered by Deep Learning & Machine Learning"
    ),
    emoji="🩸",
)

# CTA buttons
c1, c2, c3 = st.columns([1, 1, 2])
with c1:
    if st.button("🩺 Clinical Prediction", use_container_width=True):
        st.switch_page("pages/1_Clinical_Prediction.py")
with c2:
    if st.button("👁️ Eye Image Prediction", use_container_width=True):
        st.switch_page("pages/2_Eye_Image_Prediction.py")

gradient_divider()

# ── Statistics ─────────────────────────────────────────────
section_heading("Platform Statistics", "Real-time system overview")
render_metrics([
    ("🎯", "97.3%", "Model Accuracy"),
    ("🧬", "2",     "AI Models"),
    ("⚡", "<1s",   "Prediction Speed"),
    ("🌍", "3",     "Modalities"),
    ("📊", "10K+",  "Training Samples"),
    ("🔬", "35+",   "Clinical Features"),
])

gradient_divider()

# ── Feature cards ──────────────────────────────────────────
section_heading("What Can This Platform Do?")
render_feature_cards([
    ("🩺", "Clinical Prediction",
     "Enter blood count parameters — hemoglobin, RBC, MCV, MCH and more — "
     "and get an instant ML-powered anemia risk score."),
    ("👁️", "Eye Image Detection",
     "Upload or capture a conjunctiva eye image. The CNN model analyses "
     "pallor and colour patterns to detect anemia non-invasively."),
    ("⚡", "Instant Results",
     "Both models return predictions in under one second with confidence "
     "scores, probability bars, and clinical recommendations."),
])

st.markdown("<br/>", unsafe_allow_html=True)

render_feature_cards([
    ("🧠", "Deep Learning",
     "EfficientNetB3 / custom CNN trained on conjunctiva images with heavy "
     "augmentation for robust real-world performance."),
    ("📊", "Explainable AI",
     "Feature importance rankings show which clinical values drove the "
     "prediction — transparent and trustworthy."),
    ("🌍", "Region-Aware",
     "Geo-Risk module (NFHS-5 data) incorporates state-wise anemia "
     "prevalence for more accurate India-specific predictions."),
])

gradient_divider()

# ── How it works ───────────────────────────────────────────
section_heading("How It Works", "Three simple steps")

col1, col2, col3 = st.columns(3)
steps = [
    ("1", "#3B82F6", "📥 Input Data",
     "Enter clinical blood parameters OR upload/capture a conjunctiva eye image."),
    ("2", "#8B5CF6", "🤖 AI Analysis",
     "The trained model processes your input — XGBoost for clinical, CNN for images."),
    ("3", "#10B981", "📋 Get Results",
     "Receive prediction, confidence score, probability bars, and doctor recommendation."),
]
for col, (num, color, title, desc) in zip([col1, col2, col3], steps):
    with col:
        st.markdown(f"""
        <div class="card" style="text-align:center">
            <div style="width:44px;height:44px;border-radius:50%;
                        background:linear-gradient(135deg,{color}33,{color}11);
                        border:2px solid {color};display:flex;align-items:center;
                        justify-content:center;margin:0 auto 14px;
                        font-size:18px;font-weight:700;color:{color}">{num}</div>
            <div class="feature-title">{title}</div>
            <p class="feature-desc">{desc}</p>
        </div>
        """, unsafe_allow_html=True)

# ── Recent activity placeholder ────────────────────────────
gradient_divider()
section_heading("Recent Activity", "Last predictions (demo data)")

demo_rows = [
    ("👁️ Eye Image", "Anemic",     "94.2%", "#EF4444", "2 min ago"),
    ("🩺 Clinical",  "Non-Anemic", "88.7%", "#10B981", "5 min ago"),
    ("👁️ Eye Image", "Non-Anemic", "91.3%", "#10B981", "12 min ago"),
    ("🩺 Clinical",  "Anemic",     "97.1%", "#EF4444", "18 min ago"),
]

st.markdown("""
<div class="card" style="padding:0;overflow:hidden">
  <table style="width:100%;border-collapse:collapse;font-size:13.5px">
    <thead>
      <tr style="border-bottom:1px solid #21262D">
        <th style="padding:14px 20px;color:#8B949E;font-weight:500;text-align:left">Type</th>
        <th style="padding:14px 20px;color:#8B949E;font-weight:500;text-align:left">Result</th>
        <th style="padding:14px 20px;color:#8B949E;font-weight:500;text-align:left">Confidence</th>
        <th style="padding:14px 20px;color:#8B949E;font-weight:500;text-align:left">Time</th>
      </tr>
    </thead>
    <tbody>
""" + "".join([
    f"""<tr style="border-bottom:1px solid #21262D">
          <td style="padding:13px 20px;color:#C9D1D9">{t}</td>
          <td style="padding:13px 20px">
            <span class="badge {'badge-danger' if 'Anemic'==r and 'Non' not in r else 'badge-success'}">{r}</span>
          </td>
          <td style="padding:13px 20px;color:{c}">{conf}</td>
          <td style="padding:13px 20px;color:#484F58">{tm}</td>
        </tr>"""
    for t, r, conf, c, tm in demo_rows
]) + """
    </tbody>
  </table>
</div>
""", unsafe_allow_html=True)

render_footer()
