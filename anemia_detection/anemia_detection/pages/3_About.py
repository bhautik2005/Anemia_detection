import streamlit as st
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.styles     import inject_css
from utils.components import (render_sidebar, render_hero, section_heading,
                               gradient_divider, render_footer, render_metrics)

st.set_page_config(page_title="About · AI Anemia Detection",
                   page_icon="ℹ️", layout="wide",
                   initial_sidebar_state="expanded")
inject_css()
render_sidebar(active_page="About")

render_hero(
    title="About",
    highlight="AnemiaFusionNet",
    subtitle="A Multimodal AI Framework for Region-Aware Anemia Detection",
    emoji="🔬",
)

gradient_divider()

# ── Overview ────────────────────────────────────────────────
section_heading("Project Overview")
st.markdown("""
<div class="card">
  <p style="font-size:14.5px;color:#8B949E;line-height:1.8">
  <b style="color:#E6EDF3">AnemiaFusionNet</b> is a multimodal deep-learning system that detects
  anemia by fusing information from three independent data sources:
  conjunctiva eye images, clinical blood-count parameters, and
  geographical risk data (NFHS-5).  Instead of relying on a single modality,
  the framework extracts feature vectors independently and combines them
  through a <b style="color:#3B82F6">Transformer-based fusion network</b>
  to produce a final Anemic / Non-Anemic classification.
  </p>
  <p style="font-size:14.5px;color:#8B949E;line-height:1.8;margin-top:12px">
  This web application exposes <b style="color:#E6EDF3">Module 1</b> (Eye CNN) and
  <b style="color:#E6EDF3">Module 2</b> (Clinical XGBoost) as standalone prediction
  tools, ready for real-world screening use.
  </p>
</div>
""", unsafe_allow_html=True)

gradient_divider()

# ── Objectives ──────────────────────────────────────────────
section_heading("Objectives")
objs = [
    ("🎯", "Non-invasive detection",
     "Detect anemia from a smartphone conjunctiva photo — no blood test required."),
    ("📊", "Multimodal fusion",
     "Combine eye images, clinical parameters, and geographic risk for higher accuracy."),
    ("🌍", "Region-aware",
     "Incorporate NFHS-5 state-wise anemia prevalence as a geo-risk weight."),
    ("⚡", "Real-time screening",
     "Sub-second inference suitable for rural health worker deployments."),
    ("🔬", "Explainability",
     "Feature importance scores and probability outputs for clinical trust."),
    ("🏥", "Clinical integration",
     "Designed to augment — not replace — physician diagnosis."),
]
c1, c2, c3 = st.columns(3)
for i, (icon, title, desc) in enumerate(objs):
    with [c1, c2, c3][i % 3]:
        st.markdown(f"""
        <div class="feature-card" style="margin-bottom:14px">
          <div class="feature-icon">{icon}</div>
          <div class="feature-title">{title}</div>
          <p class="feature-desc">{desc}</p>
        </div>
        """, unsafe_allow_html=True)

gradient_divider()

# ── Models ──────────────────────────────────────────────────
section_heading("AI Models")
m1, m2 = st.columns(2)

with m1:
    st.markdown("""
    <div class="card" style="border-color:#3B82F633">
      <div style="font-size:15px;font-weight:600;color:#E6EDF3;margin-bottom:14px">
        👁️ CNN Eye Model &nbsp;<span class="badge badge-success">Module 1</span>
      </div>
      <div style="font-size:13.5px;color:#8B949E;line-height:1.85">
        <b style="color:#C9D1D9">Architecture:</b> Custom 4-block CNN + AdaptiveAvgPool<br/>
        <b style="color:#C9D1D9">Input:</b> Conjunctiva eye image → resized to 224×224 RGB<br/>
        <b style="color:#C9D1D9">Output:</b> 2-class softmax · [Non-Anemic, Anemic]<br/>
        <b style="color:#C9D1D9">Feature vector:</b> 128-dim (for fusion module)<br/>
        <b style="color:#C9D1D9">Augmentation:</b> Flip, rotate, colour jitter, crop<br/>
        <b style="color:#C9D1D9">Training:</b> Weighted sampler + cosine LR schedule<br/>
        <b style="color:#C9D1D9">File:</b> <code style="color:#3B82F6">best_anemia_cnn.pth</code>
      </div>
    </div>
    """, unsafe_allow_html=True)

with m2:
    st.markdown("""
    <div class="card" style="border-color:#10B98133">
      <div style="font-size:15px;font-weight:600;color:#E6EDF3;margin-bottom:14px">
        🩺 Clinical ML Model &nbsp;<span class="badge badge-info">Module 2</span>
      </div>
      <div style="font-size:13.5px;color:#8B949E;line-height:1.85">
        <b style="color:#C9D1D9">Algorithm:</b> XGBoost (best of 4 compared)<br/>
        <b style="color:#C9D1D9">Input:</b> 35 clinical + morphology + colour features<br/>
        <b style="color:#C9D1D9">Output:</b> Anomaly probability · Anemic / Non-Anemic<br/>
        <b style="color:#C9D1D9">Key features:</b> Hgb, anemia_risk_score, pallor_index<br/>
        <b style="color:#C9D1D9">Evaluation:</b> 5-fold CV · F1 + AUC-ROC<br/>
        <b style="color:#C9D1D9">Class balance:</b> Weighted loss + stratified split<br/>
        <b style="color:#C9D1D9">File:</b> <code style="color:#10B981">clinical_model.pkl</code>
      </div>
    </div>
    """, unsafe_allow_html=True)

gradient_divider()

# ── Workflow ────────────────────────────────────────────────
section_heading("Project Workflow")
steps = [
    ("1", "Dataset Collection",        "Image + Clinical + Geo data assembled"),
    ("2", "Preprocessing",             "Cleaning, normalisation, label encoding"),
    ("3", "Geo-Risk Weights",           "NFHS-5 state-wise prevalence → risk scores"),
    ("4", "Image Feature Extraction",  "CNN trained on conjunctiva images"),
    ("5", "Clinical Feature Extraction","XGBoost trained on 35 features"),
    ("6", "Geo Feature Extraction",    "State embedding + prevalence weight"),
    ("7", "Transformer Fusion",        "Concatenate → Transformer encoder → classify"),
    ("8", "Model Training & Validation","Best F1 checkpoint saved"),
    ("9", "Performance Evaluation",    "Accuracy, F1, AUC-ROC, confusion matrix"),
    ("10","Deployment",                "Streamlit web app (this platform)"),
]
for num, title, desc in steps:
    color = "#3B82F6" if int(num) % 2 == 0 else "#8B5CF6"
    st.markdown(f"""
    <div style="display:flex;gap:16px;align-items:flex-start;
                padding:12px 0;border-bottom:1px solid #21262D">
      <div style="min-width:34px;height:34px;border-radius:50%;
                  background:{color}22;border:1.5px solid {color};
                  display:flex;align-items:center;justify-content:center;
                  font-size:13px;font-weight:700;color:{color}">{num}</div>
      <div>
        <div style="font-size:14px;font-weight:600;color:#E6EDF3">{title}</div>
        <div style="font-size:13px;color:#8B949E;margin-top:2px">{desc}</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

gradient_divider()

# ── Tech Stack ──────────────────────────────────────────────
section_heading("Technology Stack")
tech = [
    ("🐍", "Python 3.10",      "Core language"),
    ("⚡", "Streamlit",        "Web frontend"),
    ("🔥", "PyTorch",          "CNN training & inference"),
    ("📊", "XGBoost",          "Clinical ML model"),
    ("🧮", "Scikit-Learn",     "Preprocessing & metrics"),
    ("🖼️", "Pillow / OpenCV",  "Image handling"),
    ("🐼", "Pandas / NumPy",   "Data manipulation"),
    ("📈", "Matplotlib",       "Visualisations"),
    ("☁️", "Google Colab",     "Model training (free GPU)"),
    ("🗂️", "GitHub",           "Version control"),
]
cols = st.columns(5)
for i, (icon, name, role) in enumerate(tech):
    with cols[i % 5]:
        st.markdown(f"""
        <div class="metric-card" style="margin-bottom:10px">
          <div class="metric-icon">{icon}</div>
          <div style="font-size:13px;font-weight:600;color:#E6EDF3">{name}</div>
          <div class="metric-label">{role}</div>
        </div>
        """, unsafe_allow_html=True)

gradient_divider()

# ── Deliverables ────────────────────────────────────────────
section_heading("Deliverables")
deliverables = [
    "✅ Dataset Preparation & Preprocessing pipeline",
    "✅ Eye CNN Model (best_anemia_cnn.pth)",
    "✅ Clinical XGBoost Model (clinical_model.pkl)",
    "✅ Geo-Risk Weight Assignment Module (NFHS-5)",
    "🔄 Transformer Fusion Architecture (in progress)",
    "✅ Performance Evaluation Report (F1, AUC, CM)",
    "✅ Source Code & GitHub Repository",
    "✅ Streamlit Web Application (this platform)",
    "🔄 Final PPT Presentation",
    "🔄 Project Demonstration Video",
]
dl_col1, dl_col2 = st.columns(2)
for i, item in enumerate(deliverables):
    with (dl_col1 if i % 2 == 0 else dl_col2):
        st.markdown(f"""
        <div style="padding:9px 0;border-bottom:1px solid #21262D;
                    font-size:13.5px;color:#8B949E">{item}</div>
        """, unsafe_allow_html=True)

gradient_divider()

# ── Future ──────────────────────────────────────────────────
section_heading("Future Enhancements")
future = [
    ("🔀", "Full Fusion",      "Combine all 3 models in Transformer fusion network"),
    ("📱", "Mobile App",       "React Native app for rural health workers"),
    ("🗺️", "District Maps",    "Live NFHS-5 geo-risk heatmap dashboard"),
    ("🔗", "EMR Integration",  "Connect to hospital Electronic Medical Records"),
    ("🌐", "Multi-language",   "Hindi, Gujarati, Tamil UI for wider reach"),
    ("☁️", "Cloud Deploy",     "AWS / GCP serverless deployment at scale"),
]
fc1, fc2, fc3 = st.columns(3)
for i, (icon, title, desc) in enumerate(future):
    with [fc1, fc2, fc3][i % 3]:
        st.markdown(f"""
        <div class="feature-card" style="margin-bottom:10px">
          <div class="feature-icon">{icon}</div>
          <div class="feature-title">{title}</div>
          <p class="feature-desc">{desc}</p>
        </div>
        """, unsafe_allow_html=True)

render_footer()
